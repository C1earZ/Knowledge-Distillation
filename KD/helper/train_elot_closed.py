# ============================================================
# train_elot_closed.py - 闭集 ELOT 蒸馏训练循环
#
# 放置位置: KD/helper/train_elot_closed.py
#
# 包含两个函数:
#   1. pretrain_projections() — FitNet 风格的映射层预训练 (可选)
#   2. train_distill_elot_closed() — 闭集 ELOT 蒸馏的主训练循环
#
# 与 train_elot.py 的区别:
#   - 教师权重需要筛选 (100 类 → 20 类)
#   - 支持映射层预训练 (FitNet 风格)
#   - 标签已经重映射到 [0, 19]
# ============================================================

from __future__ import print_function, division

import sys
import time
import torch
import torch.optim as optim

from .util import AverageMeter, accuracy
from distiller_zoo.ELOT_closed import get_classifier_weight, ProjectionPretrainLoss


# ============================================================
# 函数 1: pretrain_projections — FitNet 风格映射层预训练
#
# 借鉴 FitNet 的思路:
#   FitNet 在正式蒸馏前, 先单独训练一个 ConvReg 模块,
#   让 ConvReg(student_feat) ≈ teacher_feat (MSE 损失)
#   这样 ConvReg 学会了如何对齐两个空间
#
# 这里的做法:
#   冻结学生主干网络, 只训练 ELOTClosedLoss 中的两个映射层
#   损失 = MSE(proj_t(w_t), proj_s(w_s))   (权重空间对齐)
#         + MSE(proj_t(feat_t), proj_s(feat_s))  (特征空间对齐)
#
# 预训练完成后, 映射层已经学会如何将教师和学生的空间对齐
# 后续的端到端蒸馏可以从更好的起点开始
#
# 参考: helper/pretrain.py 中的 init() 函数
# ============================================================

def pretrain_projections(model_s, model_t, elot_loss_fn, train_loader,
                         opt, num_epochs=10, lr=0.01):
    """
    FitNet 风格的映射层预训练

    参数:
        model_s:      学生模型 (主干冻结, 只用来提取特征)
        model_t:      教师模型 (始终冻结, 提供参考)
        elot_loss_fn: ELOTClosedLoss 实例 (其中的 proj_t, proj_s 是要训练的)
        train_loader: 训练数据加载器
        opt:          超参数配置
        num_epochs:   预训练的 epoch 数 (默认 10, 类似 FitNet 的 init_epochs)
        lr:           预训练学习率 (通常比主训练小)

    效果:
        训练完成后, elot_loss_fn 中的 proj_t 和 proj_s 已经预热,
        能够合理地将教师和学生的权重/特征映射到共享空间
    """
    print("\n" + "="*60)
    print("开始 FitNet 风格映射层预训练 ({} epochs)".format(num_epochs))
    print("="*60)

    # 教师和学生都设为评估模式 (主干不训练)
    model_t.eval()
    model_s.eval()

    # 只训练映射层 (proj_t 和 proj_s)
    elot_loss_fn.train()

    # 创建只优化映射层参数的优化器
    # 类似 FitNet pretrain 中的做法: 只优化适配模块
    proj_optimizer = optim.SGD(
        [
            {'params': elot_loss_fn.proj_t.parameters()},
            {'params': elot_loss_fn.proj_s.parameters()},
        ],
        lr=lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )

    # 预训练损失函数
    pretrain_criterion = ProjectionPretrainLoss(
        weight_align_coeff=1.0,
        feat_align_coeff=1.0
    )

    # 获取类别索引 (用于筛选教师权重)
    class_indices = elot_loss_fn.class_indices  # (20,) LongTensor

    # AverageMeter 跟踪损失
    losses = AverageMeter()
    batch_time = AverageMeter()

    for epoch in range(1, num_epochs + 1):
        losses.reset()
        batch_time.reset()
        end = time.time()

        for idx, data in enumerate(train_loader):
            input, target, index = data

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # 前向传播 (不计算梯度, 因为主干冻结)
            with torch.no_grad():
                feat_s, _ = model_s(input, is_feat=True, preact=False)
                feat_t, _ = model_t(input, is_feat=True, preact=False)

                # 提取 FC 前的特征向量
                f_s = feat_s[-1].detach()  # (B, s_dim)
                f_t = feat_t[-1].detach()  # (B, t_dim)

                # 提取分类头权重
                w_s = get_classifier_weight(model_s).detach()  # (20, s_dim)
                w_t_full = get_classifier_weight(model_t).detach()  # (100, t_dim)

                # 筛选教师权重 (100 类 → 20 类)
                w_t = w_t_full[class_indices]  # (20, t_dim)

            # 映射到共享空间 (这里需要梯度, 因为要训练映射层)
            w_t_proj = elot_loss_fn.proj_t(w_t)    # (20, proj_dim)
            w_s_proj = elot_loss_fn.proj_s(w_s)    # (20, proj_dim)
            feat_t_proj = elot_loss_fn.proj_t(f_t)  # (B, proj_dim)
            feat_s_proj = elot_loss_fn.proj_s(f_s)  # (B, proj_dim)

            # 计算预训练损失
            loss = pretrain_criterion(w_t_proj, w_s_proj,
                                      feat_t_proj, feat_s_proj)

            # 反向传播: 只更新映射层
            proj_optimizer.zero_grad()
            loss.backward()
            proj_optimizer.step()

            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        # 每个 epoch 打印一次
        print('预训练 Epoch [{}/{}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {losses.avg:.4f}'.format(
              epoch, num_epochs,
              batch_time=batch_time, losses=losses))
        sys.stdout.flush()

    print("映射层预训练完成!\n")


# ============================================================
# 函数 2: train_distill_elot_closed — 闭集 ELOT 蒸馏主训练循环
# ============================================================

def train_distill_elot_closed(epoch, train_loader, module_list,
                               criterion_list, optimizer, opt,
                               global_iter_counter):
    """
    一个 epoch 的闭集 ELOT 蒸馏训练

    与 train_elot.py 中的 train_distill_elot 几乎相同,
    关键区别: 教师权重在传给 ELOT 损失前不做筛选
    (筛选逻辑已经在 ELOTClosedLoss.forward() 内部完成)

    参数:
        epoch: int, 当前 epoch 数
        train_loader: DataLoader, 子集训练数据
            返回 (image, label, index), label 值域 [0, 19]
        module_list: nn.ModuleList
            [0] = 学生模型 (20 类输出)
            [1] = ELOTClosedLoss (含映射层)
            [-1] = 教师模型 (100 类输出, 参数冻结)
        criterion_list: nn.ModuleList
            [0] = CrossEntropyLoss (分类损失)
        optimizer: SGD, 管理学生模型 + ELOT 映射层的参数
        opt: 超参数, 需包含 gamma, beta, print_freq
        global_iter_counter: list [int], 全局迭代步数

    返回:
        top1.avg: 本 epoch 平均 top1 准确率
        losses.avg: 本 epoch 平均总损失
    """

    # ---- 设置训练/评估模式 ----
    for module in module_list:
        module.train()
    module_list[-1].eval()  # 教师始终为评估模式

    # 取出各组件
    model_s = module_list[0]       # 学生模型
    elot_loss_fn = module_list[1]  # ELOTClosedLoss
    model_t = module_list[-1]      # 教师模型

    criterion_cls = criterion_list[0]  # CrossEntropyLoss

    # 指标追踪器
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_elot = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for idx, data in enumerate(train_loader):
        input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()

        # ==================== 前向传播 ====================

        # 学生前向: 获取特征 + 分类输出
        feat_s, logit_s = model_s(input, is_feat=True, preact=False)
        # logit_s: (B, 20), 学生只有 20 类输出
        # feat_s[-1]: (B, s_dim), FC 前的特征向量

        # 教师前向: 不计算梯度
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=False)
            feat_t = [f.detach() for f in feat_t]
        # logit_t: (B, 100), 教师有 100 类输出
        # feat_t[-1]: (B, t_dim)

        # ==================== 分类损失 ====================
        # target 值域 [0, 19], logit_s 有 20 个输出 → 合法
        loss_cls = criterion_cls(logit_s, target)

        # ==================== ELOT 传输损失 ====================
        # 提取分类头权重
        w_s = get_classifier_weight(model_s)          # (20, s_dim)
        w_t_full = get_classifier_weight(model_t).detach()  # (100, t_dim)

        f_s = feat_s[-1]           # (B, s_dim), 保留梯度
        f_t = feat_t[-1].detach()  # (B, t_dim)

        current_iter = global_iter_counter[0]

        # ELOTClosedLoss 内部会:
        #   1. 用 class_indices 从 w_t_full 中筛选 20 类
        #   2. 映射到共享空间
        #   3. 构建 20×20 代价矩阵
        #   4. 求解 OT 并返回损失
        loss_elot = elot_loss_fn(f_s, f_t, w_s, w_t_full,
                                  target, current_iter)

        # ==================== 总损失 ====================
        # L = γ · L_cls + β · L_elot
        loss = opt.gamma * loss_cls + opt.beta * loss_elot

        # ==================== 准确率 ====================
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        # 更新指标
        losses.update(loss.item(), input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses_elot.update(loss_elot.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ==================== 反向传播 ====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ==================== 更新计时和迭代计数 ====================
        batch_time.update(time.time() - end)
        end = time.time()
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

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg