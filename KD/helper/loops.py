# 导入 Python2 兼容模块，确保 print 是函数形式，division 是真除法
from __future__ import print_function, division

# sys 模块：用 sys.stdout.flush() 强制刷新打印缓冲区，确保终端立即显示输出
import sys
# time 模块：用 time.time() 获取当前时间戳（秒），计算耗时
import time
# PyTorch 核心库：张量操作、自动求导、GPU 计算等
import torch
# PyTorch 函数式接口：包含 softmax、one_hot、log、relu 等常用函数
# F.xxx 和 nn.xxx 的区别：F.xxx 是纯函数（无可学习参数），nn.xxx 是模块（可能有参数）
import torch.nn.functional as F
import ot
from ot.lp import emd
import numpy as np
# 从同目录下的 util.py 导入两个工具
from .util import AverageMeter, accuracy
# AverageMeter：一个小工具类，功能是跟踪一系列数值并计算它们的平均值
#   用法：meter = AverageMeter()
#         meter.update(值, 权重)  → 内部累加
#         meter.avg  → 返回加权平均值
#         meter.val  → 返回最近一次更新的值
# accuracy：计算 top-k 准确率的函数
#   输入模型预测和真实标签，返回 top1 和 top5 的准确率

# 从同目录下的 feature_visualization.py 导入特征可视化器
from .feature_visualization import FeatureVisualizer
# FeatureVisualizer：把模型中间层的特征图保存为图片，用于调试和分析
# 在正常训练中不影响结果，只是一个辅助工具


# ==============================================================================
# 函数1: train_vanilla — 标准训练（不含蒸馏，专门用于训练 teacher 模型）
# ==============================================================================
def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    # 参数说明：
    #   epoch：当前是第几个 epoch（用于打印信息）
    #   train_loader：训练数据加载器，每次迭代返回一个 batch 的 (图片, 标签)
    #   model：要训练的模型（teacher）
    #   criterion：损失函数（CrossEntropyLoss）
    #   optimizer：优化器（SGD）
    #   opt：所有超参数配置（包含 print_freq 等）
    """vanilla training"""

    # 把模型设为训练模式
    # 训练模式下：Dropout 会随机丢弃神经元，BatchNorm 用当前 batch 的均值和方差
    model.train()

    # 创建 5 个 AverageMeter 实例，分别跟踪 5 种指标的平均值
    batch_time = AverageMeter()  # 每个 batch 从开始到结束的总耗时
    data_time = AverageMeter()   # 数据从硬盘加载到内存/GPU 的耗时
    losses = AverageMeter()      # 损失值的平均
    top1 = AverageMeter()        # top1 准确率的平均
    top5 = AverageMeter()        # top5 准确率的平均

    # 记录当前时间，作为第一个 batch 数据加载耗时的起点
    end = time.time()

    # 遍历训练集的所有 batch
    # enumerate 返回 (索引idx, 数据)
    # train_loader 每次返回一个 tuple: (input, target)
    #   input: 一批图片，形状 (batch_size, 3, 32, 32)，比如 (64, 3, 32, 32)
    #   target: 这批图片的真实标签，形状 (batch_size,)，比如 (64,)，每个值是 0-99 的整数
    for idx, (input, target) in enumerate(train_loader):

        # 计算数据加载耗时：当前时间 - 上一个 batch 结束的时间
        # 这个时间主要花在：从硬盘读图片 → 解码 → 数据增强 → 转成 tensor
        data_time.update(time.time() - end)

        # 确保输入是 float32 类型（有些情况下可能是 float64/double，会导致计算变慢）
        input = input.float()

        # 如果有 GPU，把数据搬到 GPU 上（GPU 计算比 CPU 快几十到几百倍）
        if torch.cuda.is_available():
            input = input.cuda()    # 图片搬到 GPU，形状不变还是 (64, 3, 32, 32)
            target = target.cuda()  # 标签搬到 GPU，形状不变还是 (64,)

        # ===================forward（前向传播）=====================
        # 把图片喂进模型，得到预测结果
        # output 形状: (batch_size, num_classes)，比如 (64, 100)
        # output[i][j] 表示第 i 张图片属于第 j 类的"分数"（logit，还没经过 softmax）
        output = model(input)

        # 计算交叉熵损失
        # criterion = nn.CrossEntropyLoss()，内部先做 softmax 再算负对数似然
        # 输入：output (64, 100) 和 target (64,)
        # 输出：一个标量（单个数字），表示这个 batch 的平均损失
        loss = criterion(output, target)

        # 计算 top1 和 top5 准确率
        # acc1: 模型第一预测就是正确答案的比例
        # acc5: 正确答案在模型前5个预测中的比例
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # 更新各指标的运行平均值
        # loss.item()：把 tensor 标量转成 Python float 数字
        # input.size(0)：这个 batch 有多少张图（用于加权平均，因为最后一个 batch 可能不满）
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))    # acc1[0] 取出 tensor 中的值
        top5.update(acc5[0], input.size(0))

        # ===================backward（反向传播）=====================
        # 第一步：清空上一个 batch 累积的梯度
        # PyTorch 默认会累加梯度，如果不清空，这次的梯度会加到上次的上面
        optimizer.zero_grad()

        # 第二步：反向传播，计算 loss 对模型每个参数的梯度
        # PyTorch 自动沿着计算图从 loss 往回走，算出每个参数的 ∂loss/∂参数
        loss.backward()

        # 第三步：用梯度更新参数
        # 对于 SGD with momentum：
        #   速度 = momentum * 旧速度 + 梯度
        #   参数 = 参数 - 学习率 * 速度
        optimizer.step()

        # ===================meters（更新计时）=====================
        # 记录这整个 batch（数据加载 + 前向 + 反向 + 更新）的总耗时
        batch_time.update(time.time() - end)
        # 重置时间起点，作为下一个 batch 数据加载耗时的起点
        end = time.time()

        # tensorboard logger
        # 这里是预留的 TensorBoard 记录位置，pass 表示什么都不做
        pass

        # 每隔 print_freq 个 batch（默认100）打印一次训练状态
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            # 格式说明：
            #   [{0}][{1}/{2}] → 比如 [1][100/782]，表示第1个epoch的第100个batch，共782个batch
            #   .val 是当前这个 batch 的值
            #   .avg 是从第一个 batch 到现在的所有 batch 的平均值
            #   括号外面是当前值，括号里面是平均值
            #   比如 Loss 2.3456 (2.5678) 表示当前batch的loss是2.3456，平均loss是2.5678

            # 强制刷新输出缓冲区
            # 有些环境（如重定向到文件）会缓冲输出，flush 确保立即显示
            sys.stdout.flush()

    # 一个 epoch 的所有 batch 都跑完了，打印这个 epoch 的整体平均准确率
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # 返回这个 epoch 的平均 top1 准确率和平均 loss
    # 外层代码用这两个值记录到 TensorBoard
    return top1.avg, losses.avg
# ==============================================================================
def elot_emd(a, b, M, nb_dummies=1, log=False, **kwargs):
    # equivalent OT problem
    b_extended = np.append(b, [(np.sum(a)) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b)) / nb_dummies] * nb_dummies)
    M_extended = np.zeros((len(a_extended), len(b_extended)))
    M_extended[:len(a), :len(b)] = M

    # call emd solver
    gamma, log_ot = emd(a_extended, b_extended, M_extended, log=True,
                        **kwargs)

    if log_ot['warning'] is not None:
        raise ValueError("Error in the EMD resolution: try to increase the"
                         " number of dummy points")
    log_ot['partial_w_dist'] = np.sum(M * gamma[:len(a), :len(b)])

    if log:
        return gamma[:len(a), :len(b)], log_ot
    else:
        return gamma[:len(a), :len(b)]


def elot_entropic(a, b, M, reg, nb_dummies=1, numItermax=1000,
                  stopThr=1e-100, verbose=False, log=False, **kwargs):
    # equivalent OT problem
    b_extended = np.append(b, [(np.sum(a)) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b)) / nb_dummies] * nb_dummies)
    M_extended = np.zeros((len(a_extended), len(b_extended)))
    M_extended[:len(a), :len(b)] = M

    # call sinkhorn solver
    gamma, log_ot = ot.sinkhorn(a_extended, b_extended, M_extended, reg, numItermax=numItermax,
                                stopThr=stopThr, verbose=verbose, log=True, **kwargs)

    # if log_ot['warning'] is not None:
    #     raise ValueError("Error in the EMD resolution: try to increase the"
    #                      " number of dummy points")
    log_ot['partial_w_dist'] = np.sum(M * gamma[:len(a), :len(b)])

    if log:
        return gamma[:len(a), :len(b)], log_ot
    else:
        return gamma[:len(a), :len(b)]
# 函数2: train_distill — 标准蒸馏训练（支持十几种蒸馏方法的统一框架）
# ==============================================================================
def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    # 参数说明：
    #   epoch：当前第几个 epoch
    #   train_loader：训练数据加载器
    #   module_list：所有模块的列表 [student, (可能的适配模块...), teacher]
    #   criterion_list：三个损失函数 [分类损失, KL蒸馏损失, 其他蒸馏损失]
    #   optimizer：优化器，只管理 student（和适配模块）的参数
    #   opt：超参数配置
    """One epoch distillation"""

    # 先把 module_list 中的所有模块设为训练模式
    for module in module_list:
        module.train()

    # 然后把 teacher（module_list 的最后一个元素）设回评估模式
    # teacher 永远不训练，只提供稳定的知识
    # eval 模式下 Dropout 关闭、BatchNorm 用全局统计量，输出确定且稳定
    module_list[-1].eval()

    # 两种两阶段方法的特殊处理：某些适配模块在蒸馏阶段不训练
    if opt.distill == 'abound':
        module_list[1].eval()   # AB 方法：connector 已在预训练阶段训练好，这里冻结
    elif opt.distill == 'factor':
        module_list[2].eval()   # Factor Transfer：paraphraser 已在预训练阶段训练好

    # 从 criterion_list 中按索引取出三个损失函数
    criterion_cls = criterion_list[0]  # 分类损失：CrossEntropyLoss，student预测 vs 真实标签
    criterion_div = criterion_list[1]  # KL散度损失：DistillKL，student预测 vs teacher预测
    criterion_kd = criterion_list[2]   # 其他蒸馏损失：根据蒸馏方法不同而不同

    # 从 module_list 中取出 student 和 teacher
    model_s = module_list[0]    # student 是第一个加入的
    model_t = module_list[-1]   # teacher 是最后一个加入的

    # 创建 5 个 AverageMeter 跟踪各项指标
    batch_time = AverageMeter()  # 每个 batch 总耗时
    data_time = AverageMeter()   # 数据加载耗时
    losses = AverageMeter()      # 总损失
    top1 = AverageMeter()        # top1 准确率
    top5 = AverageMeter()        # top5 准确率

    # 记录起始时间
    end = time.time()

    # 遍历训练集的所有 batch
    for idx, data in enumerate(train_loader):
        # 这里用 data 接收而不是直接解包，因为不同蒸馏方法的 dataloader 返回不同数量的值

        # CRD 方法的 dataloader 额外返回对比学习需要的采样索引
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
            # input: 图片 (B, 3, 32, 32)
            # target: 标签 (B,)
            # index: 每张图片在整个数据集中的全局索引 (B,)
            # contrast_idx: 为每张图片采样的负样本索引 (B, nce_k)
        else:
            input, target, index = data
            # 其他方法只需要图片、标签和索引

        # 记录数据加载耗时
        data_time.update(time.time() - end)

        # 转为 float32 类型
        input = input.float()

        # 把数据搬到 GPU
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()  # CRD 的对比索引也要搬到 GPU

        # ===================forward（前向传播）=====================

        # preact 控制是否返回激活函数之前的特征
        # 大多数方法用激活后的特征（preact=False）
        preact = False
        if opt.distill in ['abound']:
            preact = True
            # AB (Activation Boundary) 方法需要 ReLU 之前的特征
            # 因为它要判断特征值是正还是负（激活边界）

        # Student 前向传播
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        # is_feat=True：让模型不仅返回最终预测，还返回所有中间层的特征
        # feat_s：一个列表，包含 student 每一层的特征图
        #   比如 [f0(B,16,32,32), f1(B,16,32,32), ..., f_last(B,64)]
        #   最后一个元素是全连接层之前的特征向量
        # logit_s：student 的最终分类预测，形状 (B, num_classes)

        # Teacher 前向传播
        with torch.no_grad():
            # torch.no_grad()：告诉 PyTorch 不要记录这些操作的计算图
            # 因为 teacher 不需要反向传播，这样可以节省大量 GPU 内存
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]
            # detach()：彻底把每个特征从计算图中分离出来
            # 双重保险：即使有意外的梯度流动也不会影响 teacher

        # ===== 计算分类损失（所有蒸馏方法都有这一项）=====
        loss_cls = criterion_cls(logit_s, target)
        # CrossEntropyLoss(student的预测logit, 真实标签)
        # 让 student 学会正确分类

        # ===== 计算 KL 散度蒸馏损失（所有蒸馏方法都有这一项）=====
        loss_div = criterion_div(logit_s, logit_t)
        # DistillKL(student的logit, teacher的logit)
        # 内部流程：
        #   1. 用温度 T 软化两者的 logit：p_s = softmax(logit_s/T), p_t = softmax(logit_t/T)
        #   2. 计算 KL 散度：KL(p_t || p_s)
        #   3. 乘以 T² 补偿梯度缩放
        # 让 student 的输出概率分布尽量接近 teacher 的

        # ===== 根据不同蒸馏方法计算第三项损失 =====
        # 这是各种蒸馏方法的核心区别所在

        if opt.distill == 'kd':
            loss_kd = 0
            # 标准 KD（Hinton 2015）：只用 KL 散度，不需要额外的特征损失
            # 所有知识都通过 logit 层面传递

        elif opt.distill == 'afd':
            loss_kd = criterion_kd(feat_s, feat_t)
            # AFD（Attention-based Feature Distillation）：
            # 输入 student 和 teacher 的所有层特征
            # 内部用注意力机制自动学习哪些层应该对齐

        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            # module_list[1] 是 ConvReg（卷积回归模块/regressor）
            # feat_s[opt.hint_layer] 是 student 第 hint_layer 层的特征
            # 通过 regressor 把 student 特征映射到和 teacher 相同的维度
            # 比如 student 特征是 (B,64,16,16)，regressor 映射为 (B,256,8,8)
            f_t = feat_t[opt.hint_layer]
            # teacher 对应层的特征，形状已经是 (B,256,8,8)
            loss_kd = criterion_kd(f_s, f_t)
            # HintLoss = MSELoss：计算映射后的 student 特征和 teacher 特征的均方误差

        elif opt.distill == 'crd':
            f_s = feat_s[-1]   # student 的最后一层特征（全连接层前的特征向量）
            f_t = feat_t[-1]   # teacher 的最后一层特征
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            # CRDLoss：对比表示蒸馏
            # 把 student 和 teacher 的特征投影到同一个低维空间
            # 同一张图片的 student 和 teacher 特征是正样本对（应该接近）
            # 不同图片的特征是负样本对（应该远离）
            # index：当前样本在数据集中的位置（用于查找 memory bank）
            # contrast_idx：采样的负样本索引
            g_s = [feat_s[-2]]  # 倒数第二层特征（这两行可能是为组合损失准备的，但CRD本身没用到）
            g_t = [feat_t[-2]]

        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]  # student 的中间层特征（去掉第0层和最后一层）
            g_t = feat_t[1:-1]  # teacher 的中间层特征
            loss_group = criterion_kd(g_s, g_t)
            # Attention Transfer：
            # 对每一层计算注意力图：对特征图在通道维度求 p 范数的平方 → (B, H, W)
            # 然后让 student 的注意力图接近 teacher 的
            # loss_group 是一个列表，每层一个 loss
            loss_kd = sum(loss_group)
            # 把所有层的损失加起来

        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]  # student 中间层
            g_t = feat_t[1:-1]  # teacher 中间层
            loss_group = criterion_kd(g_s, g_t)
            # NST（Neuron Selectivity Transfer）：
            # 用多项式核函数匹配 student 和 teacher 的特征分布
            loss_kd = sum(loss_group)

        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]   # 只用倒数第二层（avgpool 之前的特征图）
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            # SP（Similarity Preserving）：
            # 计算 batch 内样本间的相似性矩阵 G = F·F^T
            # 让 student 的 G_s 接近 teacher 的 G_t
            loss_kd = sum(loss_group)

        elif opt.distill == 'ickd':
            g_s = [feat_s[-2]]   # 倒数第二层
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            # ICKD（Inter-Channel Correlation KD）：
            # 计算通道间的相关性矩阵（Gram Matrix）：G = F·F^T（在通道维度）
            # 让 student 的通道相关性接近 teacher 的
            loss_kd = sum(loss_group)

        elif opt.distill == 'rkd':
            f_s = feat_s[-1]   # 最后一层特征向量
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
            # RKD（Relational Knowledge Distillation）：
            # 计算 batch 内样本间的两种关系：
            #   距离关系：||f_i - f_j||（样本对之间的欧氏距离）
            #   角度关系：cos(f_i-f_k, f_j-f_k)（三元组之间的角度）
            # 让 student 保持和 teacher 相同的样本间关系

        elif opt.distill == 'pkt':
            f_s = feat_s[-1]   # 最后一层特征向量
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
            # PKT（Probabilistic Knowledge Transfer）：
            # 计算样本间的余弦相似度，转化为概率分布
            # 用 KL 散度匹配 student 和 teacher 的相似度分布

        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]  # 中间层
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            # KDSVD：对特征做 SVD 分解，匹配主要的奇异向量
            loss_kd = sum(loss_group)

        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])  # student 最后层特征经过线性嵌入
            f_t = module_list[2](feat_t[-1])  # teacher 最后层特征经过线性嵌入
            # module_list[1] 和 [2] 分别是 student 和 teacher 的 Embed 模块
            loss_kd = criterion_kd(f_s, f_t)
            # CC（Correlation Congruence）：匹配相邻样本特征差的相关性

        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]  # student 中间层
            g_t = feat_t[1:-1]  # teacher 中间层
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            # VID（Variational Information Distillation）：
            # 和其他方法不同，criterion_kd 这里是一个列表，每层有一个独立的 VIDLoss
            # 每个 VIDLoss 内部有一个 regressor 网络来预测 teacher 特征
            # 用变分推断最大化 student 和 teacher 特征之间的互信息
            loss_kd = sum(loss_group)

        elif opt.distill == 'abound':
            loss_kd = 0
            # AB（Activation Boundary）：
            # 主要的蒸馏工作在预训练阶段（init_epochs）已经完成
            # 这里不再加额外损失，只用 cls + div

        elif opt.distill == 'fsp':
            loss_kd = 0
            # FSP（Flow of Solution Procedure）：
            # 同上，主要工作在预训练阶段完成

        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            # module_list[1] 是 Translator：把 student 倒数第二层特征压缩成 factor
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            # module_list[2] 是 Paraphraser：从 teacher 倒数第二层特征提取 factor
            # is_factor=True：只返回编码后的 factor，不返回重建结果
            loss_kd = criterion_kd(factor_s, factor_t)
            # FactorTransfer：让 student 的 factor 接近 teacher 的 factor

        else:
            # 不支持的蒸馏方法，抛出异常
            raise NotImplementedError(opt.distill)

        # ===== 计算总损失 =====
        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        # total_loss = γ × 分类损失 + α × KL蒸馏损失 + β × 其他蒸馏损失
        # 三个权重通过命令行传入，不同蒸馏方法推荐不同的权重组合
        # 比如标准 KD：gamma=1, alpha=1, beta=0（不用第三项）
        # 比如 FitNet：gamma=1, alpha=1, beta=100（第三项权重很大）

        # 计算 student 的 top1 和 top5 准确率
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        # 更新各指标的运行平均值
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward（反向传播）=====================
        optimizer.zero_grad()  # 清空上一步的梯度
        loss.backward()        # 反向传播：计算 total_loss 对所有可训练参数的梯度
                               # 梯度会流过 student 和适配模块，但不会流过 teacher
                               # 因为 teacher 的特征已经 detach() 了
        optimizer.step()       # 用梯度更新 student（和适配模块）的参数

        # ===================meters（更新计时）=====================
        batch_time.update(time.time() - end)  # 记录这个 batch 的总耗时
        end = time.time()                     # 重置计时起点

        # 每隔 print_freq 个 batch 打印一次训练状态
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    # 打印这个 epoch 的整体平均准确率
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # 返回平均 top1 准确率和平均 loss
    return top1.avg, losses.avg


# ==============================================================================
# 函数3: sinkhorn — Sinkhorn 算法（闭集蒸馏用）
# ==============================================================================
def sinkhorn(image, label):
    """
    用 Sinkhorn 算法计算最优传输矩阵。
    闭集场景：teacher 和 student 都只看 20 个类。

    核心思想：把"student/teacher 的 logit 对应到标签"这件事
    建模为最优传输问题，找到一个传输矩阵 P，
    让 P 表示"第 i 个样本的预测应该传输多少给第 j 个样本的标签"。

    参数:
        image: 模型的 logit 输出，形状 (B, num_classes)
               可以是 student 的 logit (B, 21) 也可以是 teacher 的 logit (B, 100)
        label: 真实标签，形状 (B,)

    返回:
        传输矩阵 P，形状 (B, B)
    """

    # 只保留前 20 个类的 logit，因为闭集场景只有 20 个类
    # image 从 (B, num_classes) 变成 (B, 20)
    image = image[:, :20]

    # 把标签转成 one-hot 编码
    # label=3 → [0,0,0,1,0,0,...,0]
    # f_l 形状: (B, 20)
    f_l = F.one_hot(label, num_classes=20).float().cuda()

    # 计算代价矩阵的核矩阵 K
    # image @ f_l.T：logit 矩阵 (B,20) 乘以 one-hot 矩阵的转置 (20,B) → 结果 (B,B)
    # 结果矩阵 [i,j] = 第 i 个样本的 logit 向量 · 第 j 个样本的 one-hot 标签
    #                 = 第 i 个样本在第 j 个样本的真实类别上的预测分数
    # 除以 4 是温度缩放：温度越高，K 矩阵越平滑
    # exp(-x/4)：预测分数越高（越匹配），K 值越大
    K = torch.exp(-image @ f_l.T / 4)

    # 初始化 Sinkhorn 算法的变量
    v = torch.ones(K.shape[0], device='cuda')  # 列缩放向量，初始全为 1，形状 (B,)
    a = torch.ones(K.shape[0], device='cuda')  # 行边际约束（期望的行和），均匀分布
    b = torch.ones(K.shape[0], device='cuda')  # 列边际约束（期望的列和），均匀分布

    # Sinkhorn 迭代：交替做行归一化和列归一化
    # 经过足够多次迭代后，传输矩阵的行和与列和分别接近 a 和 b
    for i in range(5):   # 迭代 5 次
        # 行归一化：u_i = a_i / (K 的第 i 行与 v 的内积)
        # K @ v：矩阵 K (B,B) 乘以向量 v (B,) → 结果 (B,)，即 K 的每一行与 v 的内积
        # 1e-6：防止除以零
        u = torch.div(a, K @ v + 1e-6)

        # 列归一化：v_j = b_j / (K 的第 j 列与 u 的内积)
        # K.T @ u：K 的转置 (B,B) 乘以 u (B,) → 结果 (B,)
        v = torch.div(b, K.T @ u + 1e-6)

    # 构造最终的传输矩阵 P = diag(u) · K · diag(v)
    # torch.diag(u)：把向量 u 变成对角矩阵 (B,B)
    # 矩阵乘法：(B,B) · (B,B) · (B,B) → (B,B)
    # P[i,j] 表示样本 i 应该"传输"多少概率质量给标签 j
    return torch.diag(u) @ K @ torch.diag(v)


# ==============================================================================
# 函数4: train_distill_close — 闭集蒸馏训练
# ==============================================================================
def train_distill_close(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """
    闭集蒸馏：teacher 和 student 看到的都是同样的 20 个类。
    和 train_distill 的唯一区别：
    loss_div 不再用标准的 KL(softmax(s) || softmax(t))，
    而是先用 Sinkhorn 算法把 logit 转成传输矩阵，再做 KL 散度。
    """

    # 所有模块设为训练模式
    for module in module_list:
        module.train()
    # teacher 设回评估模式（不训练，只提供知识）
    module_list[-1].eval()

    # 两阶段方法的特殊处理
    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    # 取出三个损失函数
    criterion_cls = criterion_list[0]  # 分类损失 CrossEntropyLoss
    criterion_div = criterion_list[1]  # KL 散度损失 DistillKL
    criterion_kd = criterion_list[2]   # 其他蒸馏损失（闭集模式下不使用）

    # 取出 student 和 teacher
    model_s = module_list[0]
    model_t = module_list[-1]

    # 创建指标跟踪器
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    # 遍历所有 batch
    for idx, data in enumerate(train_loader):
        # 根据蒸馏方法解包数据
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        # student 前向传播
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        # teacher 前向传播（不计算梯度）
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # ===== 分类损失 =====
        loss_cls = criterion_cls(logit_s, target)
        # student 预测 vs 真实标签的交叉熵

        # ===== 闭集蒸馏核心：用 Sinkhorn 替代标准 softmax =====
        p_s = sinkhorn(logit_s, target)
        # 把 student 的 logit 通过 Sinkhorn 算法转成传输矩阵 (B, B)
        # 这比简单的 softmax 多了"样本间最优匹配"的结构信息

        p_t = sinkhorn(logit_t, target)
        # teacher 的 logit 也转成传输矩阵

        loss_div = criterion_div(p_s, p_t)
        # 用 KL 散度比较 student 和 teacher 的传输矩阵
        # 标准 KD 比较的是两个 (B, C) 的概率矩阵
        # 闭集蒸馏比较的是两个 (B, B) 的传输矩阵
        # 后者包含了更丰富的结构信息（样本间的关系）

        # ===== 第三项损失 =====
        if opt.distill == 'kd':
            loss_kd = 0  # 闭集蒸馏不加额外损失
        else:
            raise NotImplementedError(opt.distill)

        # 总损失 = γ × 分类损失 + α × 蒸馏损失 + β × 其他损失
        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        # 计算准确率
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # 定期打印训练状态
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    # 打印 epoch 整体准确率
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


# ==============================================================================
# 函数5: pot — Partial Optimal Transport 算法（开放集蒸馏用）
# ==============================================================================
def pot(image, label, m, num_iter, num_cls):
    """
    Partial Optimal Transport (POT) 算法。

    和 Sinkhorn 的关键区别：Sinkhorn 要求所有质量都必须传输（全质量守恒），
    POT 允许只传输一部分质量（部分质量守恒）。

    这对开放集场景非常重要：
    - teacher 认识 100 个类，student 只认识 21 个类
    - 一个 batch 里有些样本属于 student 认识的类，有些不认识
    - POT 只传输"认识的类"对应的质量，忽略"不认识的类"

    参数:
        image: 模型的 logit，形状 (B, num_classes)
        label: 标签，形状 (B,)，已知类是 0-19，未知类是 20
        m: 需要传输的总质量（= 已知类的样本数量）
        num_iter: 迭代次数（student 迭代3次，teacher 迭代8次）
        num_cls: 类别数（student=21, teacher=100）

    返回:
        传输矩阵 P，形状 (B, B)
    """

    # 把标签转成 one-hot 编码
    # student 场景：label 值域 0-20，num_cls=21 → f_l 形状 (B, 21)
    # teacher 场景：label 值域 0-20（但 teacher 有100类），num_cls=100 → f_l 形状 (B, 100)
    f_l = F.one_hot(label, num_classes=num_cls).float().cuda()

    # 计算初始传输矩阵
    # image @ f_l.T：(B, num_cls) × (num_cls, B) → (B, B)
    # exp(-x/4)：越匹配的样本-标签对，传输量越大
    P = torch.exp(-image @ f_l.T / 4)

    # POT 迭代
    for i in range(num_iter):
        # 计算每一行的总和
        # P.sum(dim=1)：对每一行求和，形状 (B,)
        # 第 i 个值 = 第 i 个样本向所有标签传输的总量
        row_sum = P.sum(dim=1).cuda()

        # 关键步骤：行归一化时允许"不满传输"
        # torch.maximum(row_sum, 1.0)：
        #   如果某行总和 > 1，scale = row_sum（需要缩小）
        #   如果某行总和 ≤ 1，scale = 1（不缩放，允许这行的传输量小于1）
        # 这就是"partial"的含义：
        #   Sinkhorn 强制每行总和 = 1（所有质量必须传走）
        #   POT 只要求每行总和 ≤ 1（允许有些质量传不走）
        # 对于未知类的样本，它们的 logit 和已知类的标签不匹配
        # P 值天然很小，行总和本来就 < 1，不会被缩放
        # 效果：未知类样本的传输量很少，等于被"忽略"了
        scale = torch.maximum(row_sum, torch.tensor(1.0, device='cuda'))

        # 行归一化：每行除以 scale
        # unsqueeze(1) 把 (B,) 变成 (B,1)，用于广播除法
        P = P / scale.unsqueeze(1)

        # 全局质量缩放：确保传输的总质量 = m（已知类的样本数）
        # P.sum()：整个矩阵所有元素的总和
        # 乘以 m/P.sum() 让总传输量精确等于 m
        P = P * m / P.sum()

    return P


# ==============================================================================
# 函数6: train_distill_open — 开放集蒸馏训练（论文的核心方法）
# ==============================================================================
def train_distill_open(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """
    开放集蒸馏训练。
    场景：teacher 认识 100 个类，student 只认识 21 个类（20个已知类 + 1个"其他"类）。
    核心：用 Partial OT 只传输 teacher 和 student 共同认识的类别的知识。
    """

    # 所有模块设为训练模式
    for module in module_list:
        module.train()
    # teacher 设回评估模式
    module_list[-1].eval()

    # 两阶段方法特殊处理
    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    # 取出三个损失函数
    criterion_cls = criterion_list[0]  # 分类损失
    criterion_div = criterion_list[1]  # KL蒸馏损失（开放集中实际不用这个，直接算交叉熵）
    criterion_kd = criterion_list[2]   # 其他蒸馏损失

    # 取出 student 和 teacher
    model_s = module_list[0]
    model_t = module_list[-1]

    # 创建指标跟踪器
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    # 遍历所有 batch
    for idx, data in enumerate(train_loader):
        # 解包数据
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        # student 前向传播
        # logit_s 形状: (B, 21)，21个类的预测分数
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        # teacher 前向传播（不计算梯度）
        # logit_t 形状: (B, 100)，100个类的预测分数
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # ===== 开放集标签处理：把未知类统一映射为第 20 类 =====
        target[target >= 20] = 20
        # 原始标签范围 0-99（CIFAR-100 的 100 个类）
        # 处理后：0-19 保持不变（student 认识的已知类）
        #         20-99 全部变成 20（student 不认识的，统一为"其他"类）
        # 这就是为什么 student 输出 21 个类：类0~类19 + 类20("其他")

        # ===== 分类损失 =====
        loss_cls = criterion_cls(logit_s, target)
        # CrossEntropyLoss(student的21维预测, 处理后的标签0-20)
        # student 需要学会两件事：
        #   1. 正确分类已知类（0-19）
        #   2. 把未知类识别为"其他"（20）

        # ===== 开放集蒸馏核心：Partial OT =====

        # 找出已知类的样本
        mask = (target >= 0) & (target <= 19)
        # mask 是一个布尔张量，形状 (B,)
        # 已知类样本对应 True，未知类（标签=20）对应 False

        # 计算需要传输的总质量 = 已知类的样本数
        m = mask.sum().item()
        # .sum()：数 True 的个数
        # .item()：转成 Python int
        # 比如 batch=64，其中 40 个已知类 → m=40

        # 对 student 的 logit 做 Partial OT
        p_s = pot(logit_s, target, m, num_iter=3, num_cls=21)
        # logit_s: (B, 21)
        # target: (B,)，值域 0-20
        # m: 已知类样本数
        # num_iter=3：迭代 3 次（student 类别空间小，收敛快）
        # num_cls=21：student 有 21 个类
        # 返回：传输矩阵 p_s，形状 (B, B)

        # 对 teacher 的 logit 做 Partial OT
        p_t = pot(logit_t, target, m, num_iter=8, num_cls=100)
        # logit_t: (B, 100)
        # num_iter=8：迭代 8 次（teacher 类别空间大，需要更多迭代）
        # num_cls=100：teacher 有 100 个类
        # 注意：这里 label 的 one-hot 是按 100 类编码的
        #   标签 0-19 会在前 20 维有 1
        #   标签 20（"其他"）会在第 20 维有 1
        #   但 teacher 的 logit 有 100 维，所以只有前 21 维和标签有匹配
        # 返回：传输矩阵 p_t，形状 (B, B)

        # ===== 蒸馏损失：用 teacher 的传输计划引导 student =====
        loss_div = -torch.sum(p_t @ torch.log(p_s))
        # p_t @ torch.log(p_s)：
        #   p_t: (B, B)，teacher 的传输矩阵
        #   torch.log(p_s): (B, B)，student 传输矩阵的逐元素对数
        #   矩阵乘法 (B,B) × (B,B) → (B,B)
        # torch.sum(...)：对结果矩阵所有元素求和 → 标量
        # 取负号：这本质上是交叉熵 -Σ P_t · log(P_s)
        #   当 P_s 接近 P_t 时，loss_div 最小
        # 为什么不用标准的 KL 散度 criterion_div？
        #   因为 p_s 和 p_t 来自不同的类别空间（21 vs 100），
        #   但经过 POT 后都变成了 (B,B) 的传输矩阵，可以直接比较
        # POT 的妙处：
        #   对于已知类样本，p_t 和 p_s 的传输量都大 → loss 主要来自它们
        #   对于未知类样本，p_t 和 p_s 的传输量都接近 0 → 对 loss 贡献很小
        #   效果：自动忽略了未知类，只在已知类上做蒸馏

        # ===== 第三项损失 =====
        if opt.distill == 'kd':
            loss_kd = 0  # 开放集蒸馏目前只实现了 kd 模式
        else:
            raise NotImplementedError(opt.distill)

        # 总损失
        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        # 计算准确率
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新 student 参数

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # 定期打印
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


# ==============================================================================
# 函数7: validate — 标准验证函数（用于评估 teacher 或闭集 student）
# ==============================================================================
def validate(val_loader, model, criterion, opt):
    """
    在测试集上评估模型。
    不做反向传播，不更新参数，只计算准确率和 loss。
    """

    # 创建指标跟踪器（验证时不需要 data_time）
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 设为评估模式：
    #   Dropout 关闭 → 所有神经元参与计算
    #   BatchNorm 用训练时积累的全局统计量 → 输出确定性
    model.eval()

    # 创建特征可视化器（调试用，不影响评估结果）
    vis_feator = FeatureVisualizer()

    # torch.no_grad()：不记录计算图，不计算梯度
    # 好处1：节省 GPU 内存（不需要存中间结果用于反向传播）
    # 好处2：加快计算速度（不需要构建计算图）
    with torch.no_grad():
        end = time.time()

        # 遍历测试集的每个 batch
        for idx, data in enumerate(val_loader):
            # input: 测试图片 (B, 3, 32, 32)
            # target: 真实标签 (B,)
            # input, target = data
            input, target, index = data
            input = input.float()  # 转为 float32
            if torch.cuda.is_available():
                input = input.cuda()    # 搬到 GPU
                target = target.cuda()

            # 前向传播
            feats, output = model(input, is_feat=True, preact=False)
            # feats: 各层特征（这里主要是给 FeatureVisualizer 用的）
            # output: 最终预测 logit，形状 (B, num_classes)

            # 计算损失（仅用于记录和打印，不用于反向传播）
            loss = criterion(output, target)

            # 计算 top1 和 top5 准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # 更新平均值
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # 记录耗时
            batch_time.update(time.time() - end)
            end = time.time()

            # 定期打印评估进度
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        # 打印整个测试集的最终结果
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    # 返回三个值：平均 top1 准确率、平均 top5 准确率、平均 loss
    return top1.avg, top5.avg, losses.avg


# ==============================================================================
# 函数8: validate_open — 开放集验证函数
# ==============================================================================
def validate_open(val_loader, model, criterion, opt):
    """
    开放集验证。
    和 validate 的唯一区别：target[target >= 20] = 20
    把未知类标签统一映射为 20，和训练时保持一致。
    如果不做这个映射，标签范围是 0-99，但 student 只输出 21 类，
    CrossEntropyLoss 会因为标签超出范围而报错。
    """

    # 创建指标跟踪器
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 设为评估模式
    model.eval()

    # 创建特征可视化器
    vis_feator = FeatureVisualizer()

    # 不计算梯度
    with torch.no_grad():
        end = time.time()

        # 遍历测试集的每个 batch
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # 前向传播
            # output 形状: (B, 21)，student 只有 21 个输出
            feats, output = model(input, is_feat=True, preact=False)

            # 开放集标签映射：和 train_distill_open 中完全一样的处理
            target[target >= 20] = 20
            # 0-19: 已知类，保持不变
            # 20-99: 未知类，映射为 20

            # 计算损失
            # output (B, 21) vs target (B,) 值域 0-20 → 合法
            loss = criterion(output, target)

            # 计算准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # 更新平均值
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # 记录耗时
            batch_time.update(time.time() - end)
            end = time.time()

            # 定期打印
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        # 打印最终结果
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    # 返回平均 top1、top5 和 loss
    return top1.avg, top5.avg, losses.avg