# ============================================================
# train_student_elot_closed.py - 闭集 ELOT 蒸馏 (含特征对齐)
#
# 放置位置: KD/train_student_elot_closed.py (替换原文件)
#
# 总损失:
#   L = α · CE + β · D_feature(T_t(F^t)||T_s(F^s)) + γ · ⟨P*, C⟩
#
# 使用方法:
#   python train_student_elot_closed.py \
#       --path_t ./save/models/ResNet50_cifar100_.../ResNet50_best.pth \
#       --model_s resnet14 \
#       --subset_id 0 \
#       --gamma 1.0 \
#       --beta_feat 100.0 \
#       --beta 0.5 \
#       --feat_dim 128
# ============================================================

from __future__ import print_function

import os
import argparse
import socket
import time
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import tensorboard_logger as tb_logger

from models import model_dict
from dataset.cifar100_subset import get_cifar100_closed_subset_dataloaders
from dataset.cifar100_subset import CLASSES_PER_SUBSET
from helper.util import adjust_learning_rate
from helper.loops import validate

from distiller_zoo.ELOT_closed import ELOTClosedLoss
from distiller_zoo.FeatureAlign import FeatureAlignLoss, get_default_align_pairs
from helper.train_elot_closed import (
    train_distill_elot_closed,
    pretrain_projections,
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
    print('==> 加载教师模型...')
    model_name = get_teacher_name(model_path)
    model = model_dict[model_name](num_classes=n_cls)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    print('==> 教师模型加载完成: {}'.format(model_name))
    return model


def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('闭集 ELOT + 特征对齐 蒸馏训练')

    # ==================== 训练基础参数 ====================
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=240)

    # ==================== 优化器参数 ====================
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # ==================== 模型与数据集 ====================
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100'])
    parser.add_argument('--model_s', type=str, default='resnet14',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32',
                                 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4',
                                 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'ResNet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, required=True)

    # ==================== 闭集子集参数 ====================
    parser.add_argument('--subset_id', type=int, default=0, choices=[0, 1, 2, 3, 4])

    # ==================== 损失权重 ====================
    # L_total = gamma * L_cls + beta_feat * L_feat + beta * L_elot
    parser.add_argument('-r', '--gamma', type=float, default=1.0,
                        help='分类损失权重 (公式中的 α)')
    parser.add_argument('--beta_feat', type=float, default=100.0,
                        help='特征对齐损失权重 (公式中的 β)')
    parser.add_argument('-b', '--beta', type=float, default=0.5,
                        help='ELOT 损失权重 (公式中的 γ)')

    # ==================== 特征对齐参数 ====================
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='特征对齐共享空间的通道数')
    parser.add_argument('--s_align_layers', type=str, default=None,
                        help='学生 feat 列表中要对齐的层索引, 逗号分隔.'
                             '如 "4,6" 表示 stage2末尾和stage3末尾.'
                             '不指定则自动推断')
    parser.add_argument('--t_align_layers', type=str, default=None,
                        help='教师 feat 列表中对应的层索引, 逗号分隔.'
                             '如 "7,13". 不指定则自动推断')

    # ==================== ELOT 参数 ====================
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--lambda_mmd', type=float, default=0.1)
    parser.add_argument('--ot_epsilon', type=float, default=0.1)
    parser.add_argument('--mmd_sigma', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--nb_dummies', type=int, default=1)

    # ==================== 映射层预训练 (可选) ====================
    parser.add_argument('--pretrain_proj', action='store_true', default=False)
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--pretrain_lr', type=float, default=0.01)

    # ==================== 其他 ====================
    parser.add_argument('--trial', type=str, default='1')
    parser.add_argument('--seed', type=int, default=0)

    opt = parser.parse_args()

    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    opt.model_t = get_teacher_name(opt.path_t)
    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(',')]

    opt.model_name = (
        'S:{}_T:{}_elot_closed_sub{}_r:{}_bfeat:{}_b:{}_fdim:{}_trial:{}'.format(
            opt.model_s, opt.model_t, opt.subset_id,
            opt.gamma, opt.beta_feat, opt.beta,
            opt.feat_dim, opt.trial
        )
    )

    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def main():
    best_acc = 0
    opt = parse_option()

    set_random_seed(opt.seed, deterministic=True)
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # ==================== 加载数据 ====================
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data, class_indices = \
            get_cifar100_closed_subset_dataloaders(
                subset_id=opt.subset_id,
                batch_size=opt.batch_size,
                num_workers=opt.num_workers
            )
        num_classes_student = CLASSES_PER_SUBSET  # 20
        num_classes_teacher = 100
    else:
        raise NotImplementedError(opt.dataset)

    # ==================== 创建模型 ====================
    model_t = load_teacher(opt.path_t, num_classes_teacher)
    model_s = model_dict[opt.model_s](num_classes=num_classes_student)

    # ==================== 探测特征维度 ====================
    data_dummy = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()

    feat_t, _ = model_t(data_dummy, is_feat=True)
    feat_s, _ = model_s(data_dummy, is_feat=True)

    t_dim = feat_t[-1].shape[1]  # ResNet50: 2048
    s_dim = feat_s[-1].shape[1]  # ResNet14: 64

    # ==================== 确定特征对齐层 ====================
    # 方式1: 用户手动指定
    # 方式2: 自动推断 (根据模型名和空间分辨率匹配)
    if opt.s_align_layers is not None and opt.t_align_layers is not None:
        # 手动指定: 如 --s_align_layers "4,6" --t_align_layers "7,13"
        s_align_indices = [int(x) for x in opt.s_align_layers.split(',')]
        t_align_indices = [int(x) for x in opt.t_align_layers.split(',')]
        print("\n手动指定对齐层:")
        for si, ti in zip(s_align_indices, t_align_indices):
            print("  feat_s[{}] ↔ feat_t[{}]".format(si, ti))
    else:
        # 自动推断: 按空间分辨率匹配, 取各 stage 末尾
        s_align_indices, t_align_indices = get_default_align_pairs(
            opt.model_s, opt.model_t
        )

    # 获取对齐层的 shape (用探测的 feat 数据)
    s_align_shapes = [feat_s[i].shape for i in s_align_indices]
    t_align_shapes = [feat_t[i].shape for i in t_align_indices]

    print("\n" + "="*60)
    print("实验配置:")
    print("  教师: {} (100类, 特征维度 {})".format(opt.model_t, t_dim))
    print("  学生: {} (20类, 特征维度 {})".format(opt.model_s, s_dim))
    print("  子集: {} → 类别 {}~{}".format(
        opt.subset_id, class_indices[0], class_indices[-1]))
    print("  损失权重: α={}, β_feat={}, γ_elot={}".format(
        opt.gamma, opt.beta_feat, opt.beta))
    print("  特征对齐层:")
    for i, (ss, ts) in enumerate(zip(s_align_shapes, t_align_shapes)):
        print("    第{}对: 学生 {} → {} ↔ 教师 {} → {}".format(
            i, ss, opt.feat_dim, ts, opt.feat_dim))
    print("="*60 + "\n")

    # ==================== 创建 ELOT 损失 ====================
    elot_criterion = ELOTClosedLoss(
        t_dim=t_dim, s_dim=s_dim, proj_dim=opt.proj_dim,
        num_classes_student=num_classes_student,
        class_indices=class_indices,
        lambda_mmd=opt.lambda_mmd, epsilon=opt.ot_epsilon,
        sigma=opt.mmd_sigma, warmup_iters=opt.warmup_iters,
        nb_dummies=opt.nb_dummies,
    )

    # ==================== 创建特征对齐模块 ====================
    feat_align_criterion = FeatureAlignLoss(
        s_shapes=s_align_shapes,   # 学生对齐层的 shape 列表
        t_shapes=t_align_shapes,   # 教师对齐层的 shape 列表
        feat_dim=opt.feat_dim      # 共享空间通道数
    )

    # ==================== 组装模块列表 ====================
    # module_list: [student, elot_loss, feat_align, teacher]
    #   [0] = student
    #   [1] = ELOT 损失 (含映射层)
    #   [2] = 特征对齐 (含变换层)
    #   [-1] = teacher
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    module_list.append(elot_criterion)
    module_list.append(feat_align_criterion)  # 新增: 特征对齐模块
    module_list.append(model_t)

    # trainable_list: 学生 + ELOT映射层 + 特征对齐变换层
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    trainable_list.append(elot_criterion)
    trainable_list.append(feat_align_criterion)  # 新增: 对齐变换层也要训练

    # ==================== 损失函数 ====================
    criterion_cls = nn.CrossEntropyLoss()
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)

    # ==================== 优化器 ====================
    optimizer = optim.SGD(
        trainable_list.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )

    # ==================== GPU ====================
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # ==================== 映射层预训练 (可选) ====================
    if opt.pretrain_proj:
        pretrain_projections(
            model_s=model_s, model_t=model_t,
            elot_loss_fn=elot_criterion,
            train_loader=train_loader, opt=opt,
            num_epochs=opt.pretrain_epochs, lr=opt.pretrain_lr
        )

    # ==================== 全局迭代计数器 ====================
    global_iter_counter = [0]

    # ==================== 主训练循环 ====================
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> 训练中... Epoch {}/{}".format(epoch, opt.epochs))

        time1 = time.time()

        # 传入对齐层索引
        train_acc, train_loss = train_distill_elot_closed(
            epoch, train_loader, module_list, criterion_list,
            optimizer, opt, global_iter_counter,
            s_align_indices=s_align_indices,   # 新增参数
            t_align_indices=t_align_indices    # 新增参数
        )

        time2 = time.time()
        print('Epoch {}, 耗时 {:.2f}s'.format(epoch, time2 - time1))

        # ==================== 测试集评估 ====================
        test_acc, test_acc_top5, test_loss = validate(
            val_loader, model_s, criterion_cls, opt
        )

        # ==================== 记录日志 ====================
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        # ==================== 保存最佳模型 ====================
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'elot_loss': elot_criterion.state_dict(),
                'feat_align': feat_align_criterion.state_dict(),
                'best_acc': best_acc,
                'subset_id': opt.subset_id,
                'class_indices': class_indices,
            }
            save_file = os.path.join(
                opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('保存最佳模型! Acc: {:.2f}%'.format(best_acc))
            torch.save(state, save_file)

        if epoch % opt.save_freq == 0:
            print('==> 保存 checkpoint...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'elot_loss': elot_criterion.state_dict(),
                'feat_align': feat_align_criterion.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{}.pth'.format(epoch))
            torch.save(state, save_file)

    # ==================== 训练完成 ====================
    print("\n训练完成! 最佳准确率: {:.2f}%".format(best_acc))

    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'elot_loss': elot_criterion.state_dict(),
        'feat_align': feat_align_criterion.state_dict(),
    }
    save_file = os.path.join(
        opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()