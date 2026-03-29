# ============================================================
# train_elot.py - ELOT 蒸馏专用训练循环
#
# 放置位置: KD/helper/train_elot.py
#
# 与 helper/loops.py 中的 train_distill 风格一致, 但额外处理:
#   1. 从模型中提取分类头权重, 用于构建动态代价矩阵
#   2. 追踪全局迭代步数, 用于 ELOT 的 warmup 策略
#   3. 组合分类损失 + ELOT 传输损失
#
# 总损失: L = γ · L_cls + β · L_elot
#   - L_cls: 学生预测与真实标签的交叉熵 (硬标签损失)
#   - L_elot: 基于动态代价矩阵的弹性最优传输损失
# ============================================================

from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy
from distiller_zoo.ELOT import get_classifier_weight


def train_distill_elot(epoch, train_loader, module_list, criterion_list,
                       optimizer, opt, global_iter_counter):
    """
    一个 epoch 的 ELOT 蒸馏训练

    参数:
        epoch: int, 当前 epoch 数 (用于打印信息)
        train_loader: DataLoader, 训练数据加载器
            返回 (image, label, index) 三元组 (因为 is_instance=True)
        module_list: nn.ModuleList, 包含 [student, elot_loss, teacher]
            - module_list[0]: 学生模型 (参数会被优化)
            - module_list[1]: ELOTLoss 模块 (包含可训练的映射层)
            - module_list[-1]: 教师模型 (参数固定, 只提供知识)
        criterion_list: nn.ModuleList, 损失函数列表
            - criterion_list[0]: 分类损失 (nn.CrossEntropyLoss)
        optimizer: torch.optim.Optimizer, 管理学生模型 + ELOTLoss 映射层的参数
        opt: argparse.Namespace, 超参数配置, 需包含:
            - opt.gamma: 分类损失的权重
            - opt.beta: ELOT 损失的权重
            - opt.print_freq: 打印间隔
        global_iter_counter: list, 长度为1的列表 [当前全局迭代步数]
            使用列表而非 int, 是为了在函数间传递可变引用

    返回:
        top1.avg: float, 本 epoch 平均 top1 准确率
        losses.avg: float, 本 epoch 平均总损失
    """

    # ---- 设置模块的训练/评估模式 ----
    for module in module_list:
        module.train()
    # 教师模型始终为评估模式:
    #   Dropout 关闭, BatchNorm 用全局统计量 → 输出稳定
    module_list[-1].eval()

    # 取出各组件
    model_s = module_list[0]       # 学生模型
    elot_loss_fn = module_list[1]  # ELOTLoss (含可训练映射层)
    model_t = module_list[-1]      # 教师模型 (参数冻结)

    # 分类损失
    criterion_cls = criterion_list[0]  # nn.CrossEntropyLoss

    # 创建指标追踪器 (与 loops.py 中的 train_distill 保持一致)
    batch_time = AverageMeter()   # 每个 batch 的总耗时
    data_time = AverageMeter()    # 数据加载耗时
    losses = AverageMeter()       # 总损失
    losses_cls = AverageMeter()   # 分类损失 (单独追踪, 方便调试)
    losses_elot = AverageMeter()  # ELOT 损失 (单独追踪)
    top1 = AverageMeter()         # top1 准确率
    top5 = AverageMeter()         # top5 准确率

    end = time.time()

    for idx, data in enumerate(train_loader):
        # 解包数据: (图片, 标签, 样本索引)
        # 因为使用了 CIFAR100Instance (is_instance=True)
        input, target, index = data

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()

        # ==================== 前向传播 ====================

        # 学生前向传播: 获取中间层特征 + 分类输出
        feat_s, logit_s = model_s(input, is_feat=True, preact=False)
        # feat_s: 各层特征的列表
        #   feat_s[-1] 是 FC 前的特征向量, 形状 (B, s_dim)
        # logit_s: 分类 logit, 形状 (B, num_classes)

        # 教师前向传播: 不计算梯度 (教师参数不更新)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=False)
            feat_t = [f.detach() for f in feat_t]

        # ==================== 计算分类损失 ====================
        # 让学生学会正确分类 (与真实硬标签对齐)
        loss_cls = criterion_cls(logit_s, target)

        # ==================== 计算 ELOT 传输损失 ====================
        # 提取分类头权重
        w_s = get_classifier_weight(model_s)          # (num_classes, s_dim)
        w_t = get_classifier_weight(model_t).detach()  # (num_classes, t_dim), 不传梯度

        # 获取 FC 层之前的特征向量 (列表最后一个元素)
        f_s = feat_s[-1]           # (B, s_dim), 学生特征, 保留梯度
        f_t = feat_t[-1].detach()  # (B, t_dim), 教师特征, 已 detach

        # 计算 ELOT 损失
        current_iter = global_iter_counter[0]
        loss_elot = elot_loss_fn(f_s, f_t, w_s, w_t, target, current_iter)

        # ==================== 组合总损失 ====================
        # L = γ · L_cls + β · L_elot
        loss = opt.gamma * loss_cls + opt.beta * loss_elot

        # ==================== 计算准确率 ====================
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        # 更新各指标
        losses.update(loss.item(), input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses_elot.update(loss_elot.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ==================== 反向传播 ====================
        optimizer.zero_grad()   # 清空上一步梯度
        loss.backward()         # 反向传播: 梯度流过学生模型和 ELOTLoss 映射层
                                # 不会流过教师 (已 detach)
        optimizer.step()        # 用梯度更新学生和映射层的参数

        # ==================== 更新计时和迭代计数 ====================
        batch_time.update(time.time() - end)
        end = time.time()

        # 更新全局迭代步数 (用于 ELOT warmup 判断)
        global_iter_counter[0] += 1

        # ==================== 定期打印 ====================
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'L_cls {lcls.val:.4f} L_elot {lelot.val:.4f}\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                loss=losses, lcls=losses_cls, lelot=losses_elot,
                top1=top1, top5=top5))
            sys.stdout.flush()

    # 一个 epoch 结束, 打印整体平均准确率
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg