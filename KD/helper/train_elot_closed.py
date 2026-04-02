# ============================================================
# train_elot_closed.py - 闭集 ELOT 蒸馏训练循环
#
# 放置位置: KD/helper/train_elot_closed.py
#
# 包含三个函数:
#   1. pretrain_conv_reg      — FitNet ConvReg 预训练 (阶段 1)
#   2. pretrain_projections   — ELOT 映射层预训练 (可选)
#   3. train_distill_elot_closed — 正式蒸馏训练 (阶段 2)
# ============================================================

from __future__ import print_function, division

import sys
import time
import torch
import torch.optim as optim

from .util import AverageMeter, accuracy
from distiller_zoo.ELOT_closed import get_classifier_weight, ProjectionPretrainLoss
from distiller_zoo.FitNet import HintLoss


# ============================================================
# 函数 1: ConvReg 预训练 (FitNet 原论文两阶段)
# ============================================================

def pretrain_conv_reg(model_s, model_t, conv_reg, train_loader, opt,
                      logger, num_epochs=30, lr=None):
    """
    FitNet 原论文的 ConvReg 预训练

    与 helper/pretrain.py 中 init 函数的逻辑完全一致:
      - 学生和教师都设为 eval 模式 (主干参数冻结)
      - 只优化 ConvReg 的参数
      - 用 HintLoss (MSE) 让 ConvReg(feat_s[hint]) ≈ feat_t[hint]
    """
    print("\n" + "=" * 60)
    print("FitNet ConvReg 预训练 ({} epochs, hint_layer={})".format(
        num_epochs, opt.hint_layer))
    print("=" * 60)

    if lr is None:
        lr = opt.learning_rate

    model_t.eval()
    model_s.eval()
    conv_reg.train()

    reg_optimizer = optim.SGD(
        conv_reg.parameters(),
        lr=lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )

    hint_criterion = HintLoss()
    if torch.cuda.is_available():
        hint_criterion = hint_criterion.cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(1, num_epochs + 1):
        batch_time.reset()
        losses.reset()
        end = time.time()

        for idx, data in enumerate(train_loader):
            input, target, index = data
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()

            with torch.no_grad():
                feat_s, _ = model_s(input, is_feat=True, preact=False)
                feat_t, _ = model_t(input, is_feat=True, preact=False)
                feat_s = [f.detach() for f in feat_s]
                feat_t = [f.detach() for f in feat_t]

            f_s = conv_reg(feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss = hint_criterion(f_s, f_t)

            reg_optimizer.zero_grad()
            loss.backward()
            reg_optimizer.step()

            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        # 记录到 TensorBoard
        logger.log_value('conv_reg_pretrain_loss', losses.avg, epoch)

        print('ConvReg 预训练 Epoch [{}/{}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {losses.avg:.4f}'.format(
              epoch, num_epochs,
              batch_time=batch_time, losses=losses))
        sys.stdout.flush()

    print("ConvReg 预训练完成!\n")


# ============================================================
# 函数 2: ELOT 映射层预训练 (可选)
# ============================================================

def pretrain_projections(model_s, model_t, elot_loss_fn, train_loader,
                         opt, logger, num_epochs=10, lr=0.01):
    """
    ELOT 映射层预训练
    冻结学生主干, 只训练 ELOT 的 proj_t 和 proj_s 映射层
    """
    print("\n" + "=" * 60)
    print("ELOT 映射层预训练 ({} epochs)".format(num_epochs))
    print("=" * 60)

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

        # 记录到 TensorBoard
        logger.log_value('elot_proj_pretrain_loss', losses.avg, epoch)

        print('ELOT 映射层预训练 Epoch [{}/{}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {losses.avg:.4f}'.format(
              epoch, num_epochs,
              batch_time=batch_time, losses=losses))
        sys.stdout.flush()

    print("ELOT 映射层预训练完成!\n")


# ============================================================
# 函数 3: 闭集 ELOT 蒸馏主训练循环 (阶段 2)
# ============================================================

def train_distill_elot_closed(epoch, train_loader, module_list,
                               criterion_list, optimizer, opt,
                               global_iter_counter):
    """
    一个 epoch 的闭集 ELOT 蒸馏训练

    参数:
        module_list: [student, elot_loss, conv_reg(可选), teacher]
        criterion_list: [CrossEntropyLoss, HintLoss(可选)]
    """

    beta_feat = getattr(opt, 'beta_feat', 0.0)
    use_hint = beta_feat > 0

    # ---- 设置训练/评估模式 ----
    for module in module_list:
        module.train()
    module_list[-1].eval()

    # ---- 取出各组件 ----
    model_s = module_list[0]
    elot_loss_fn = module_list[1]
    if use_hint:
        conv_reg = module_list[2]
        hint_loss_fn = criterion_list[1]
    model_t = module_list[-1]

    criterion_cls = criterion_list[0]

    # ---- 指标追踪器 ----
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_feat = AverageMeter()
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
        feat_s, logit_s = model_s(input, is_feat=True, preact=False)

        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=False)
            feat_t = [f.detach() for f in feat_t]

        # ==================== 第一项: 分类损失 ====================
        loss_cls = criterion_cls(logit_s, target)

        # ==================== 第二项: FitNet 特征对齐 ====================
        if use_hint:
            f_s = conv_reg(feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_feat = hint_loss_fn(f_s, f_t)
        else:
            loss_feat = torch.tensor(0.0, device=input.device)

        # ==================== 第三项: ELOT 传输损失 ====================
        w_s = get_classifier_weight(model_s)
        w_t_full = get_classifier_weight(model_t).detach()
        f_s_vec = feat_s[-1]
        f_t_vec = feat_t[-1].detach()
        current_iter = global_iter_counter[0]

        loss_elot = elot_loss_fn(f_s_vec, f_t_vec, w_s, w_t_full,
                                  target, current_iter)

        # ==================== 总损失 ====================
        loss = opt.gamma * loss_cls + beta_feat * loss_feat + opt.beta * loss_elot

        # ==================== 准确率 ====================
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        feat_val = loss_feat.item() if torch.is_tensor(loss_feat) else loss_feat
        losses_feat.update(feat_val, input.size(0))
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