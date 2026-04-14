#!/usr/bin/env python
# ============================================================
# tune_elot_closed.py - ELOT 闭集蒸馏调参脚本
#
# 功能:
#   1. 一次性跑 5 个子集, 输出每个子集的 best_acc + 5 子集平均
#   2. 使用 Optuna 自动搜索超参数
#   3. 支持诊断模式: 只跑几个 epoch 看 C_weight/C_mmd 数量级
#
# 使用方法:
#   # 诊断模式: 跑 3 个 epoch, 看 C_weight 和 C_mmd 的数量级
#   python tune_elot_closed.py \
#       --path_t ./save/models/ResNet50_cifar100_lr_0.05_decay_0.0005_trial_1/ResNet50_best.pth \
#       --mode diagnose --epochs 3
#
#   # 调参模式: Optuna 自动搜索, 每组参数跑 30 epoch
#   python tune_elot_closed.py \
#       --path_t ./save/models/ResNet50_cifar100_lr_0.05_decay_0.0005_trial_1/ResNet50_best.pth \
#       --mode tune --epochs 30 --n_trials 20
#
#   # 最终训练: 用找到的最优参数跑完整 240 epoch
#   python tune_elot_closed.py \
#       --path_t ./save/models/ResNet50_cifar100_lr_0.05_decay_0.0005_trial_1/ResNet50_best.pth \
#       --mode train --epochs 240 \
#       --beta 5.0 --lambda_mmd 0.01 --beta_feat 100.0
# ============================================================

from __future__ import print_function

import os
import argparse
import socket
import time
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import tensorboard_logger as tb_logger

from models import model_dict
from models.util import ConvReg
from dataset.cifar100_subset import get_cifar100_closed_subset_dataloaders
from dataset.cifar100_subset import CLASSES_PER_SUBSET
from helper.util import adjust_learning_rate
from helper.loops import validate

from distiller_zoo.ELOT_closed import ELOTClosedLoss
from distiller_zoo.FitNet import HintLoss
from helper.train_elot_closed import (
    train_distill_elot_closed,
    pretrain_conv_reg,
)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_teacher_name(model_path):
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    model_name = get_teacher_name(model_path)
    model = model_dict[model_name](num_classes=n_cls)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    return model


def train_one_subset(subset_id, opt):
    """
    训练一个子集, 返回该子集的 best test accuracy

    参数:
        subset_id: int, 0-4
        opt: 包含所有超参数的 namespace

    返回:
        best_acc: float, 该子集的最佳测试准确率
    """
    print("\n" + "=" * 60)
    print("  子集 {} (类 {}~{})".format(
        subset_id, subset_id * 20, subset_id * 20 + 19))
    print("=" * 60)

    set_random_seed(opt.seed, deterministic=True)

    # ==================== 数据 ====================
    train_loader, val_loader, n_data, class_indices = \
        get_cifar100_closed_subset_dataloaders(
            subset_id=subset_id,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers
        )
    num_classes_student = CLASSES_PER_SUBSET  # 20
    num_classes_teacher = 100

    # ==================== 模型 ====================
    model_t = load_teacher(opt.path_t, num_classes_teacher)
    model_s = model_dict[opt.model_s](num_classes=num_classes_student)

    # ==================== 探测维度 ====================
    data_dummy = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data_dummy, is_feat=True)
    feat_s, _ = model_s(data_dummy, is_feat=True)
    t_dim = feat_t[-1].shape[1]
    s_dim = feat_s[-1].shape[1]

    # ==================== ConvReg ====================
    use_hint = opt.beta_feat > 0
    conv_reg = None
    if use_hint:
        s_shape = feat_s[opt.hint_layer].shape
        t_shape = feat_t[opt.hint_layer].shape
        conv_reg = ConvReg(s_shape, t_shape)

    # ==================== ELOT ====================
    elot_criterion = ELOTClosedLoss(
        t_dim=t_dim, s_dim=s_dim, proj_dim=opt.proj_dim,
        num_classes_student=num_classes_student,
        class_indices=class_indices,
        lambda_mmd=opt.lambda_mmd, epsilon=opt.ot_epsilon,
        sigma=opt.mmd_sigma, warmup_iters=opt.warmup_iters,
        nb_dummies=opt.nb_dummies,
    )

    # ==================== 组装 ====================
    module_list = nn.ModuleList([model_s, elot_criterion])
    trainable_list = nn.ModuleList([model_s, elot_criterion])
    if use_hint:
        module_list.append(conv_reg)
        trainable_list.append(conv_reg)
    module_list.append(model_t)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_list = nn.ModuleList([criterion_cls])
    if use_hint:
        criterion_list.append(HintLoss())

    optimizer = optim.SGD(
        trainable_list.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()

    # ==================== ConvReg 预训练 ====================
    if use_hint and opt.init_epochs > 0:
        # 简化版预训练: 不需要 logger
        model_t.eval()
        model_s.eval()
        conv_reg.train()
        reg_optimizer = optim.SGD(
            conv_reg.parameters(), lr=opt.learning_rate,
            momentum=opt.momentum, weight_decay=opt.weight_decay
        )
        hint_criterion = HintLoss().cuda() if torch.cuda.is_available() else HintLoss()

        for ep in range(1, opt.init_epochs + 1):
            for idx, data in enumerate(train_loader):
                input, target, index = data
                input = input.float()
                if torch.cuda.is_available():
                    input = input.cuda()
                with torch.no_grad():
                    feat_s_pre, _ = model_s(input, is_feat=True, preact=False)
                    feat_t_pre, _ = model_t(input, is_feat=True, preact=False)
                    feat_s_pre = [f.detach() for f in feat_s_pre]
                    feat_t_pre = [f.detach() for f in feat_t_pre]
                f_s = conv_reg(feat_s_pre[opt.hint_layer])
                f_t = feat_t_pre[opt.hint_layer]
                loss = hint_criterion(f_s, f_t)
                reg_optimizer.zero_grad()
                loss.backward()
                reg_optimizer.step()
            if ep % 10 == 0 or ep == opt.init_epochs:
                print("  ConvReg 预训练 epoch {}/{}, loss={:.4f}".format(
                    ep, opt.init_epochs, loss.item()))

    # ==================== 正式训练 ====================
    global_iter_counter = [0]
    best_acc = 0

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)

        train_acc, train_loss = train_distill_elot_closed(
            epoch, train_loader, module_list, criterion_list,
            optimizer, opt, global_iter_counter
        )

        test_acc, test_acc_top5, test_loss = validate(
            val_loader, model_s, criterion_cls, opt
        )

        if test_acc > best_acc:
            best_acc = test_acc

        # 只在关键 epoch 打印测试结果 (减少输出)
        if epoch % 10 == 0 or epoch == opt.epochs:
            print("  子集{} Epoch {}/{}: test_acc={:.2f}%, best={:.2f}%".format(
                subset_id, epoch, opt.epochs, test_acc, best_acc))

    print("\n  >>> 子集 {} 最终 best_acc: {:.2f}%".format(subset_id, best_acc))
    return best_acc


def run_all_subsets(opt):
    """
    依次跑 5 个子集, 汇总结果
    """
    print("\n" + "#" * 60)
    print("  ELOT 闭集蒸馏 - 全部 5 个子集")
    print("  参数: beta={}, beta_feat={}, lambda_mmd={}, "
          "proj_dim={}, warmup_iters={}, ot_epsilon={}".format(
        opt.beta, opt.beta_feat, opt.lambda_mmd,
        opt.proj_dim, opt.warmup_iters, opt.ot_epsilon))
    print("#" * 60)

    results = {}
    for subset_id in range(5):
        acc = train_one_subset(subset_id, opt)
        results[subset_id] = acc

    # ==================== 汇总 ====================
    print("\n" + "=" * 60)
    print("  最终结果汇总")
    print("=" * 60)
    for sid in range(5):
        print("  子集 {} (类 {:2d}~{:2d}): {:.2f}%".format(
            sid, sid * 20, sid * 20 + 19, results[sid]))
    avg_acc = np.mean([v.item() for v in results.values()])
    print("  " + "-" * 40)
    print("  5 子集平均: {:.2f}%".format(avg_acc))
    print("=" * 60)

    return avg_acc, results


def parse_args():
    parser = argparse.ArgumentParser('ELOT 闭集蒸馏调参')

    # 模式
    parser.add_argument('--mode', type=str, default='diagnose',
                        choices=['diagnose', 'tune', 'train'],
                        help='diagnose=看数量级, tune=Optuna调参, train=最终训练')

    # 基础
    parser.add_argument('--path_t', type=str, required=True)
    parser.add_argument('--model_s', type=str, default='resnet14')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--init_epochs', type=int, default=10,
                        help='ConvReg 预训练轮数 (调参时可以短一些)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

    # 优化器
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # 损失权重
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--beta_feat', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=5.0)

    # FitNet
    parser.add_argument('--hint_layer', type=int, default=4)

    # ELOT
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--lambda_mmd', type=float, default=0.1)
    parser.add_argument('--ot_epsilon', type=float, default=0.1)
    parser.add_argument('--mmd_sigma', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--nb_dummies', type=int, default=1)

    # Optuna
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Optuna 搜索次数')
    parser.add_argument('--study_name', type=str, default='elot_closed',
                        help='Optuna study 名称')

    opt = parser.parse_args()
    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(',')]
    return opt


def main():
    opt = parse_args()

    if opt.mode == 'diagnose':
        print("\n>>> 诊断模式: 跑全部 5 个子集, 每个 {} epochs, 观察 C_weight/C_mmd 数量级".format(
            opt.epochs))
        print(">>> 请确保 ELOT_closed.py 中已加入诊断打印代码\n")
        for sid in range(5):
            train_one_subset(sid, opt)

    elif opt.mode == 'tune':
        # ==================== Optuna 调参模式 ====================
        try:
            import optuna
        except ImportError:
            print("请先安装 Optuna: python -m pip install optuna")
            sys.exit(1)

        def objective(trial):
            """Optuna 目标函数: 返回 5 子集平均准确率 (越大越好)"""
            # 搜索超参数范围
            opt.beta = trial.suggest_float('beta', 1.0, 100.0, log=True)
            opt.lambda_mmd = trial.suggest_float('lambda_mmd', 0.5, 5.0, log=True)
            opt.proj_dim = trial.suggest_categorical('proj_dim', [64, 128, 256])
            opt.warmup_iters = trial.suggest_categorical('warmup_iters', [100, 500, 1000, 2000])
            opt.ot_epsilon = trial.suggest_float('ot_epsilon', 0.01, 1.0, log=True)

            print("\n>>> Trial {}: beta={:.3f}, lambda_mmd={:.4f}, "
                  "proj_dim={}, warmup_iters={}, "
                  "ot_epsilon={:.3f}".format(
                trial.number, opt.beta, opt.lambda_mmd,
                opt.proj_dim, opt.warmup_iters,
                opt.ot_epsilon))

            accs = []
            for subset_id in range(5):
                acc = train_one_subset(subset_id, opt)
                accs.append(acc.item())

                # 每跑完一个子集，向 Optuna 报告当前平均准确率
                current_avg = np.mean(accs)
                trial.report(current_avg, subset_id)

                # 如果 Optuna 判断这组参数没前途，提前终止
                if trial.should_prune():
                    print("  >>> Trial {} 被剪枝 (跑完子集{}, 当前平均={:.2f}%)".format(
                        trial.number, subset_id, current_avg))
                    raise optuna.TrialPruned()

            avg_acc = np.mean(accs)
            for sid in range(5):
                trial.set_user_attr('subset_{}_acc'.format(sid), accs[sid])

            return avg_acc
        # 创建 Optuna study
        study = optuna.create_study(
            study_name=opt.study_name,
            direction='maximize',  # 最大化准确率
            storage='sqlite:///optuna_elot_closed.db',  # 结果存数据库, 可中断恢复
            load_if_exists=True,  # 如果之前跑过, 继续
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,  # 前 5 次不剪枝，先积累基准
                n_warmup_steps=1,  # 至少跑完 2 个子集才开始剪枝
            ),
        )

        study.optimize(objective, n_trials=opt.n_trials)

        # 打印最佳结果
        print("\n" + "=" * 60)
        print("  Optuna 调参完成!")
        print("=" * 60)
        print("  最佳 trial: #{}".format(study.best_trial.number))
        print("  最佳平均准确率: {:.2f}%".format(study.best_value))
        print("  最佳参数:")
        for key, value in study.best_params.items():
            print("    {} = {}".format(key, value))
        print("=" * 60)

        # 打印最佳 trial 每个子集的结果
        print("\n  最佳 trial 各子集结果:")
        for sid in range(5):
            key = 'subset_{}_acc'.format(sid)
            if key in study.best_trial.user_attrs:
                print("    子集 {}: {:.2f}%".format(
                    sid, study.best_trial.user_attrs[key]))

    elif opt.mode == 'train':
        # ==================== 最终训练模式 ====================
        print("\n>>> 最终训练模式: 用指定参数跑 5 个子集, {} epochs".format(opt.epochs))
        avg_acc, results = run_all_subsets(opt)


if __name__ == '__main__':
    main()