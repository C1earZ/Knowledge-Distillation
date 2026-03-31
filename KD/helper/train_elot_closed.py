# ============================================================
# train_elot_closed_v2.py - 闭集 ELOT 蒸馏训练循环 (含特征对齐)
#
# 放置位置: KD/helper/train_elot_closed.py (替换原文件)
#
# 总损失公式:
#   L_total = α · CE(y | σ(f_s(x); τ))           ← 硬标签交叉熵
#           + β · D_feature(T_t(F^t) || T_s(F^s)) ← FitNet 特征对齐
#           + γ · ⟨P*, C⟩                         ← ELOT 最优传输
#
# 对应代码中:
#   L_total = opt.gamma * loss_cls
#           + opt.beta_feat * loss_feat
#           + opt.beta * loss_elot
#
# 与原 train_elot_closed.py 的区别:
#   新增 loss_feat 项: FitNet 风格的中间层特征对齐损失
#   训练循环中额外提取指定层的特征, 传给 FeatureAlignLoss
# ============================================================

from __future__ import print_function, division

import sys
import time
import torch
import torch.optim as optim

from .util import AverageMeter, accuracy
from distiller_zoo.ELOT_closed import get_classifier_weight, ProjectionPretrainLoss


# ============================================================
# 函数 1: pretrain_projections — FitNet 风格映射层预训练 (不变)
# ============================================================

def pretrain_projections(model_s, model_t, elot_loss_fn, train_loader,
                         opt, num_epochs=10, lr=0.01):
    """
    FitNet 风格的映射层预训练 (与原版完全相同, 不涉及特征对齐)
    冻结学生主干, 只训练 ELOT 的 proj_t 和 proj_s 映射层
    """
    print("\n" + "="*60)
    print("开始 FitNet 风格映射层预训练 ({} epochs)".format(num_epochs))
    print("="*60)

    model_t.eval()
    model_s.eval()
    elot_loss_fn.train()

    proj_optimizer = optim.SGD(
        [
            {'params': elot_loss_fn.proj_t.parameters()},
            {'params': elot_loss_fn.proj_s.parameters()},
        ],
        lr=lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )

    pretrain_criterion = ProjectionPretrainLoss(
        weight_align_coeff=1.0,
        feat_align_coeff=1.0
    )

    class_indices = elot_loss_fn.class_indices

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

            with torch.no_grad():
                feat_s, _ = model_s(input, is_feat=True, preact=False)
                feat_t, _ = model_t(input, is_feat=True, preact=False)
                f_s = feat_s[-1].detach()
                f_t = feat_t[-1].detach()
                w_s = get_classifier_weight(model_s).detach()
                w_t_full = get_classifier_weight(model_t).detach()
                w_t = w_t_full[class_indices]

            w_t_proj = elot_loss_fn.proj_t(w_t)
            w_s_proj = elot_loss_fn.proj_s(w_s)
            feat_t_proj = elot_loss_fn.proj_t(f_t)
            feat_s_proj = elot_loss_fn.proj_s(f_s)

            loss = pretrain_criterion(w_t_proj, w_s_proj,
                                      feat_t_proj, feat_s_proj)

            proj_optimizer.zero_grad()
            loss.backward()
            proj_optimizer.step()

            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        print('预训练 Epoch [{}/{}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {losses.avg:.4f}'.format(
              epoch, num_epochs,
              batch_time=batch_time, losses=losses))
        sys.stdout.flush()

    print("映射层预训练完成!\n")


# ============================================================
# 函数 2: train_distill_elot_closed — 闭集蒸馏主训练循环 (含特征对齐)
# ============================================================

def train_distill_elot_closed(epoch, train_loader, module_list,
                               criterion_list, optimizer, opt,
                               global_iter_counter,
                               s_align_indices=None, t_align_indices=None):
    """
    一个 epoch 的闭集 ELOT 蒸馏训练 (含 FitNet 特征对齐)

    参数:
        epoch: int, 当前 epoch 数
        train_loader: DataLoader, 子集训练数据
            返回 (image, label, index), label 值域 [0, 19]
        module_list: nn.ModuleList
            [0] = 学生模型 (20 类输出)
            [1] = ELOTClosedLoss (含映射层)
            [2] = FeatureAlignLoss (特征对齐模块, 如果有的话)
                  如果没有特征对齐, 这个位置是教师模型
            [-1] = 教师模型 (100 类输出, 参数冻结)
        criterion_list: nn.ModuleList
            [0] = CrossEntropyLoss (分类损失)
        optimizer: SGD, 管理学生 + ELOT映射层 + 特征对齐变换层 的参数
        opt: 超参数, 需包含:
            - gamma: 分类损失权重
            - beta: ELOT 损失权重
            - beta_feat: 特征对齐损失权重 (新增)
            - print_freq: 打印间隔
        global_iter_counter: list [int], 全局迭代步数
        s_align_indices: list of int, 学生 feat 列表中要对齐的层索引
            如 [4, 6] 表示 feat_s[4] 和 feat_s[6]
            如果为 None, 不做特征对齐
        t_align_indices: list of int, 教师 feat 列表中对应的层索引
            如 [7, 13] 表示 feat_t[7] 和 feat_t[13]

    返回:
        top1.avg: 本 epoch 平均 top1 准确率
        losses.avg: 本 epoch 平均总损失
    """

    # 是否启用特征对齐
    use_feat_align = (s_align_indices is not None and t_align_indices is not None)

    # ---- 设置训练/评估模式 ----
    for module in module_list:
        module.train()
    module_list[-1].eval()  # 教师始终评估模式

    # ---- 取出各组件 ----
    model_s = module_list[0]       # 学生模型
    elot_loss_fn = module_list[1]  # ELOTClosedLoss
    if use_feat_align:
        # module_list 结构: [student, elot_loss, feat_align, teacher]
        feat_align_fn = module_list[2]  # FeatureAlignLoss
        model_t = module_list[-1]       # 教师模型
    else:
        # module_list 结构: [student, elot_loss, teacher]
        feat_align_fn = None
        model_t = module_list[-1]

    criterion_cls = criterion_list[0]  # CrossEntropyLoss

    # ---- 指标追踪器 ----
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()    # 交叉熵损失
    losses_feat = AverageMeter()   # 特征对齐损失
    losses_elot = AverageMeter()   # ELOT 损失
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

        # 学生前向: 获取所有中间层特征 + 分类输出
        feat_s, logit_s = model_s(input, is_feat=True, preact=False)
        # feat_s: 列表, 包含所有层的特征
        # logit_s: (B, 20)

        # 教师前向: 不计算梯度
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=False)
            feat_t = [f.detach() for f in feat_t]
        # feat_t: 列表, 包含所有层的特征
        # logit_t: (B, 100)

        # ==================== 第一项: 分类损失 ====================
        # α · CE(y | σ(f_s(x); τ))
        loss_cls = criterion_cls(logit_s, target)

        # ==================== 第二项: 特征对齐损失 ====================
        # β_feat · D_feature(T_t(F^t) || T_s(F^s))
        if use_feat_align:
            # 从 feat 列表中取出要对齐的层
            g_s = [feat_s[i] for i in s_align_indices]
            # 如 s_align_indices=[4,6] → g_s = [feat_s[4], feat_s[6]]
            # → [(B,32,16,16), (B,64,8,8)]

            g_t = [feat_t[i] for i in t_align_indices]
            # 如 t_align_indices=[7,13] → g_t = [feat_t[7], feat_t[13]]
            # → [(B,512,16,16), (B,1024,8,8)]

            loss_feat = feat_align_fn(g_s, g_t)
        else:
            loss_feat = torch.tensor(0.0, device=input.device)

        # ==================== 第三项: ELOT 传输损失 ====================
        # γ · ⟨P*, C⟩
        w_s = get_classifier_weight(model_s)
        w_t_full = get_classifier_weight(model_t).detach()
        f_s = feat_s[-1]           # (B, s_dim)
        f_t = feat_t[-1].detach()  # (B, t_dim)
        current_iter = global_iter_counter[0]

        loss_elot = elot_loss_fn(f_s, f_t, w_s, w_t_full,
                                  target, current_iter)

        # ==================== 总损失 ====================
        # L_total = α · L_cls + β_feat · L_feat + γ · L_elot
        # 对应 opt 中的参数名:
        #   opt.gamma   → α (分类损失权重, 公式里的 α)
        #   opt.beta_feat → β (特征对齐权重, 公式里的 β)
        #   opt.beta    → γ (ELOT 损失权重, 公式里的 γ)
        beta_feat = getattr(opt, 'beta_feat', 0.0)
        loss = opt.gamma * loss_cls + beta_feat * loss_feat + opt.beta * loss_elot

        # ==================== 准确率 ====================
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        # ---- 更新指标 ----
        losses.update(loss.item(), input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses_feat.update(loss_feat.item() if torch.is_tensor(loss_feat) else loss_feat,
                           input.size(0))
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
                  'L_cls {lcls.val:.4f} L_feat {lfeat.val:.4f} '
                  'L_elot {lelot.val:.4f}\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                loss=losses, lcls=losses_cls,
                lfeat=losses_feat, lelot=losses_elot,
                top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg