# 导入 print_function：使 Python 2 中的 print 行为与 Python 3 一致
# 导入 division：使 Python 2 中的除法行为与 Python 3 一致（即整数相除返回浮点数）
from __future__ import print_function, division

import time          # 用于计时，统计每个 batch 和 epoch 的耗时
import sys           # 用于调用 sys.stdout.flush()，强制刷新标准输出缓冲区
import torch         # PyTorch 核心库，提供张量运算、自动求导等功能
import torch.optim as optim                    # 优化器模块，用于定义 SGD 等优化算法
import torch.backends.cudnn as cudnn           # cuDNN 后端配置，用于加速 GPU 上的卷积运算
from .util import AverageMeter                 # 从同目录下的 util 模块导入 AverageMeter，用于计算并记录指标的均值


def init(model_s, model_t, init_modules, criterion, train_loader, logger, opt):
    """
    预训练初始化函数，用于在正式知识蒸馏训练之前，
    对特定蒸馏方法（abound / factor / fsp）所需的辅助模块进行预热训练。

    参数说明：
        model_s      : 学生模型（Student Model）
        model_t      : 教师模型（Teacher Model），参数已固定，仅用于提取特征
        init_modules : 需要预训练的辅助模块列表（如连接器、因子分解器等）
        criterion    : 损失函数
        train_loader : 训练数据的 DataLoader
        logger       : 日志记录器，用于写入 TensorBoard 或其他日志系统
        opt          : 超参数配置对象（包含学习率、动量、权重衰减、epoch 数等）
    """

    # 将教师模型设置为评估模式（关闭 Dropout、BatchNorm 使用全局统计量）
    # 教师模型仅作为特征提供者，不参与梯度更新
    model_t.eval()

    # 将学生模型也设置为评估模式
    # 在预训练阶段，学生模型的主干参数不更新，仅用于提取中间特征
    model_s.eval()

    # 将需要预训练的辅助模块设置为训练模式
    # 只有 init_modules 中的参数会被优化器更新
    init_modules.train()

    # 判断当前环境是否有可用的 GPU
    if torch.cuda.is_available():
        # 将学生模型移动到 GPU 显存
        model_s.cuda()
        # 将教师模型移动到 GPU 显存
        model_t.cuda()
        # 将辅助模块移动到 GPU 显存
        init_modules.cuda()
        # 开启 cuDNN 自动寻找最优卷积算法的功能，可加速固定输入尺寸的训练
        cudnn.benchmark = True

    # 针对特定的小型学生模型架构 + factor 蒸馏方法，使用更小的学习率 0.01
    # 这些模型参数量少，使用较大学习率容易不稳定
    if opt.model_s in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                       'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and \
            opt.distill == 'factor':
        lr = 0.01   # 对上述组合使用固定的小学习率
    else:
        lr = opt.learning_rate  # 其余情况使用配置文件中指定的学习率

    # 定义 SGD 优化器，只优化 init_modules 中的参数
    optimizer = optim.SGD(
        init_modules.parameters(),   # 只传入辅助模块的参数，学生/教师模型参数不参与优化
        lr=lr,                       # 学习率，由上方逻辑确定
        momentum=opt.momentum,       # 动量系数，用于加速收敛并抑制震荡，通常取 0.9
        weight_decay=opt.weight_decay  # L2 正则化系数，防止过拟合
    )

    # 创建 AverageMeter 对象，用于追踪并计算每个 epoch 内 batch 时间的均值
    batch_time = AverageMeter()

    # 创建 AverageMeter 对象，用于追踪并计算数据加载时间的均值
    data_time = AverageMeter()

    # 创建 AverageMeter 对象，用于追踪并计算每个 epoch 内训练损失的均值
    losses = AverageMeter()

    # 外层循环：遍历所有预训练 epoch，从第 1 轮开始到 opt.init_epochs 结束（含）
    for epoch in range(1, opt.init_epochs + 1):

        # 重置 batch_time 计数器，清除上一个 epoch 的统计数据
        batch_time.reset()

        # 重置 data_time 计数器
        data_time.reset()

        # 重置 losses 计数器
        losses.reset()

        # 记录当前时刻，作为计时起点（用于计算 data_time 和 batch_time）
        end = time.time()

        # 内层循环：遍历训练集中的每个 mini-batch
        # idx 为当前 batch 的索引，data 为 DataLoader 返回的数据
        for idx, data in enumerate(train_loader):

            # CRD（对比表示蒸馏）方法需要额外的对比样本索引 contrast_idx
            if opt.distill in ['crd']:
                # 解包：输入图像、标签、样本索引、对比负样本索引
                input, target, index, contrast_idx = data
            else:
                # 其他蒸馏方法只需要输入图像、标签、样本索引
                input, target, index = data

            # 更新数据加载时间：从上一个 end 时刻到现在，即数据加载耗时
            data_time.update(time.time() - end)

            # 将输入张量转为 float32 类型（部分 DataLoader 可能返回 float64 或其他类型）
            input = input.float()

            # 如果 GPU 可用，将数据转移到 GPU
            if torch.cuda.is_available():
                input = input.cuda()       # 输入图像移到 GPU
                target = target.cuda()     # 标签移到 GPU
                index = index.cuda()       # 样本索引移到 GPU
                if opt.distill in ['crd']:
                    # 对比蒸馏还需要将对比样本索引移到 GPU
                    contrast_idx = contrast_idx.cuda()

            # ============= 前向传播 ==============

            # 判断是否需要提取预激活特征（abound 方法需要在激活函数之前的特征图）
            preact = (opt.distill == 'abound')  # 若使用 abound 方法则为 True，否则 False

            # 学生模型前向传播，提取中间特征列表 feat_s 和最终输出（用 _ 忽略）
            # is_feat=True 表示返回所有中间层特征，preact 控制是否返回预激活特征
            feat_s, _ = model_s(input, is_feat=True, preact=preact)

            # 教师模型前向传播，使用 torch.no_grad() 禁用梯度计算
            # 教师模型仅作参考，不需要反向传播，禁用梯度可节省显存和加快速度
            with torch.no_grad():
                # 教师模型提取中间特征列表 feat_t 和最终输出（忽略）
                feat_t, _ = model_t(input, is_feat=True, preact=preact)
                # 对每个教师特征图调用 .detach()，确保它们完全脱离计算图
                # 即使在 no_grad 块外意外使用也不会产生梯度
                feat_t = [f.detach() for f in feat_t]

            # ---- 根据不同蒸馏方法计算对应损失 ----

            if opt.distill == 'abound':
                # ABound（激活边界）蒸馏：
                # 使用辅助模块 init_modules[0] 对学生的中间层特征（去掉首尾层）做变换
                # feat_s[1:-1] 表示去掉第一层和最后一层的中间特征
                g_s = init_modules[0](feat_s[1:-1])
                # 教师对应的中间层特征，作为监督目标
                g_t = feat_t[1:-1]
                # 计算各层之间的损失，criterion 返回一个损失列表
                loss_group = criterion(g_s, g_t)
                # 将各层损失求和，得到总损失
                loss = sum(loss_group)

            elif opt.distill == 'factor':
                # Factor Transfer（因子迁移）蒸馏：
                # 取教师模型倒数第二层的特征图（通常是最后一个卷积层的输出）
                f_t = feat_t[-2]
                # 通过辅助模块（因子分解器）对教师特征进行编码和重建
                # init_modules[0] 返回 (编码结果, 重建结果)，这里只使用重建结果
                _, f_t_rec = init_modules[0](f_t)
                # 计算重建特征与原始教师特征之间的重建损失（如 MSE）
                loss = criterion(f_t_rec, f_t)

            elif opt.distill == 'fsp':
                # FSP（流解方案）蒸馏：
                # 利用 Gram 矩阵（特征图之间的内积）对齐学生和教师的特征流
                # feat_s[:-1] 和 feat_t[:-1] 表示去掉最后一层的所有中间特征
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                # 将各层 FSP 损失求和
                loss = sum(loss_group)

            else:
                # 若 opt.distill 不属于以上三种方法，则抛出未实现异常
                raise NotImplemented('Not supported in init training: {}'.format(opt.distill))

            # 更新损失均值计数器：loss.item() 将张量损失转为 Python 标量，input.size(0) 为 batch size
            losses.update(loss.item(), input.size(0))

            # ===================反向传播=====================

            # 清空优化器中所有参数的梯度缓存（防止梯度累积）
            optimizer.zero_grad()

            # 对损失进行反向传播，计算各参数的梯度
            loss.backward()

            # 根据计算出的梯度，更新 init_modules 中的参数
            optimizer.step()

            # 更新 batch_time：从上一个 end 时刻到现在，即整个 batch 的处理耗时
            batch_time.update(time.time() - end)

            # 重置计时起点，为下一个 batch 计时做准备
            end = time.time()

        # ---- 当前 epoch 结束 ----

        # 将当前 epoch 的平均训练损失记录到日志（如 TensorBoard）
        # 'init_train_loss' 为日志标签，losses.avg 为本 epoch 的平均损失，epoch 为横轴步数
        logger.log_value('init_train_loss', losses.avg, epoch)

        # 打印当前 epoch 的训练统计信息到控制台
        # [{0}/{1}] 显示当前 epoch 和总 epoch 数
        # batch_time.val 为最后一个 batch 的耗时，batch_time.avg 为本 epoch 平均耗时
        # losses.val 为最后一个 batch 的损失，losses.avg 为本 epoch 平均损失
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
               epoch, opt.init_epochs, batch_time=batch_time, losses=losses))

        # 强制刷新标准输出缓冲区，确保日志在后台运行时也能实时输出
        # 在某些 HPC 集群或重定向输出的场景下非常重要
        sys.stdout.flush()