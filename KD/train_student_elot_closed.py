# ============================================================
# train_student_elot_closed.py - 闭集 ELOT 蒸馏主训练脚本
#
# 放置位置: KD/train_student_elot_closed.py
#
# 使用方法:
#   python train_student_elot_closed.py \
#       --path_t ./save/models/resnet110_cifar100_.../resnet110_best.pth \
#       --model_s resnet20 \
#       --subset_id 0 \
#       --gamma 1.0 --beta 0.5 \
#       --pretrain_proj          (可选: 启用 FitNet 风格预训练)
#       --pretrain_epochs 10     (可选: 预训练 epoch 数)
#
# 运行全部 5 个子集的实验 (bash 脚本示例):
#   for sid in 0 1 2 3 4; do
#       python train_student_elot_closed.py \
#           --path_t ./save/models/.../resnet110_best.pth \
#           --model_s resnet20 \
#           --subset_id $sid \
#           --trial 1
#   done
#
# 总损失: L = γ · L_cls + β · L_elot
#   L_cls:  学生预测 vs 真实标签的交叉熵 (硬标签)
#   L_elot: 基于动态代价矩阵的弹性最优传输损失 (闭集 20 类版本)
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

# ==================== 从原有框架导入 ====================
from models import model_dict                                  # 模型字典
from dataset.cifar100_subset import get_cifar100_closed_subset_dataloaders  # 子集数据加载
from dataset.cifar100_subset import CLASSES_PER_SUBSET         # 每个子集的类数 (20)
from helper.util import adjust_learning_rate                    # 学习率调度
from helper.loops import validate                               # 测试集评估

# ==================== 导入闭集 ELOT 模块 ====================
from distiller_zoo.ELOT_closed import ELOTClosedLoss            # 闭集 ELOT 损失
from helper.train_elot_closed import (
    train_distill_elot_closed,   # 闭集 ELOT 训练循环
    pretrain_projections,         # FitNet 风格映射层预训练
)


# ============================================================
# 工具函数
# ============================================================

def set_random_seed(seed, deterministic=False):
    """固定所有随机种子, 保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_teacher_name(model_path):
    """从 checkpoint 路径中解析教师模型架构名"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    """加载预训练的教师模型 (100 类)"""
    print('==> 加载教师模型...')
    model_name = get_teacher_name(model_path)
    model = model_dict[model_name](num_classes=n_cls)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    print('==> 教师模型加载完成: {}'.format(model_name))
    return model


# ============================================================
# 参数解析
# ============================================================

def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('闭集 ELOT 知识蒸馏训练')

    # ==================== 训练基础参数 ====================
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=240,
                        help='总训练轮数')

    # ==================== 优化器参数 ====================
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210',
                        help='学习率衰减的 epoch 节点')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # ==================== 模型与数据集 ====================
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100'])
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32',
                                 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4',
                                 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'ResNet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'],
                        help='学生模型架构')
    parser.add_argument('--path_t', type=str, required=True,
                        help='教师模型 checkpoint 路径')

    # ==================== 闭集子集参数 ====================
    parser.add_argument('--subset_id', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='子集编号 0-4, 每个子集包含 20 个类')

    # ==================== 损失权重 ====================
    parser.add_argument('-r', '--gamma', type=float, default=1.0,
                        help='分类损失权重 γ')
    parser.add_argument('-b', '--beta', type=float, default=0.5,
                        help='ELOT 损失权重 β')

    # ==================== ELOT 专有参数 ====================
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='映射层输出维度')
    parser.add_argument('--lambda_mmd', type=float, default=0.1,
                        help='MMD 项权重 λ')
    parser.add_argument('--ot_epsilon', type=float, default=0.1,
                        help='OT 熵正则化系数')
    parser.add_argument('--mmd_sigma', type=float, default=1.0,
                        help='RBF 核带宽 σ')
    parser.add_argument('--warmup_iters', type=int, default=500,
                        help='标准 OT warmup 迭代数')
    parser.add_argument('--nb_dummies', type=int, default=1,
                        help='ELOT 虚拟点数量')

    # ==================== FitNet 风格预训练 (可选) ====================
    parser.add_argument('--pretrain_proj', action='store_true', default=False,
                        help='是否启用 FitNet 风格映射层预训练')
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                        help='映射层预训练的 epoch 数')
    parser.add_argument('--pretrain_lr', type=float, default=0.01,
                        help='映射层预训练的学习率')

    # ==================== 其他 ====================
    parser.add_argument('--trial', type=str, default='1',
                        help='实验编号')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')

    opt = parser.parse_args()

    # ---- 特殊模型学习率 ----
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # ---- 解析教师模型名 ----
    opt.model_t = get_teacher_name(opt.path_t)

    # ---- 解析学习率衰减节点 ----
    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(',')]

    # ---- 实验名称 (包含子集编号, 方便区分 5 个子集的实验) ----
    opt.model_name = (
        'S:{}_T:{}_elot_closed_sub{}_r:{}_b:{}_proj:{}_lmmd:{}_eps:{}'
        '_wup:{}_pretrain:{}_trial:{}'.format(
            opt.model_s, opt.model_t, opt.subset_id,
            opt.gamma, opt.beta, opt.proj_dim,
            opt.lambda_mmd, opt.ot_epsilon,
            opt.warmup_iters,
            'yes' if opt.pretrain_proj else 'no',
            opt.trial
        )
    )

    # ---- 存储路径 ----
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


# ============================================================
# 主函数
# ============================================================

def main():
    best_acc = 0
    opt = parse_option()

    # ==================== 固定随机种子 ====================
    set_random_seed(opt.seed, deterministic=True)

    # ==================== TensorBoard ====================
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # ==================== 加载闭集子集数据 ====================
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data, class_indices = \
            get_cifar100_closed_subset_dataloaders(
                subset_id=opt.subset_id,
                batch_size=opt.batch_size,
                num_workers=opt.num_workers
            )
        # class_indices: 当前子集对应教师的哪 20 个原始类
        # 如 subset_id=0 → [0, 1, ..., 19]
        num_classes_student = CLASSES_PER_SUBSET  # 20
        num_classes_teacher = 100
    else:
        raise NotImplementedError(opt.dataset)

    # ==================== 创建模型 ====================
    # 教师: 100 类输出 (用完整 CIFAR-100 训练过)
    model_t = load_teacher(opt.path_t, num_classes_teacher)

    # 学生: 20 类输出 (只学当前子集的 20 个类)
    model_s = model_dict[opt.model_s](num_classes=num_classes_student)

    # ==================== 探测特征维度 ====================
    data_dummy = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()

    feat_t, _ = model_t(data_dummy, is_feat=True)
    feat_s, _ = model_s(data_dummy, is_feat=True)

    t_dim = feat_t[-1].shape[1]  # 教师特征维度 (如 resnet110 → 64)
    s_dim = feat_s[-1].shape[1]  # 学生特征维度 (如 resnet20 → 64)

    print("\n" + "="*60)
    print("实验配置:")
    print("  教师模型: {} (100 类, 特征维度 {})".format(opt.model_t, t_dim))
    print("  学生模型: {} (20 类, 特征维度 {})".format(opt.model_s, s_dim))
    print("  子集编号: {} → 原始类别 {} ~ {}".format(
        opt.subset_id, class_indices[0], class_indices[-1]))
    print("  映射维度: {}".format(opt.proj_dim))
    print("  映射层预训练: {}".format('是' if opt.pretrain_proj else '否'))
    print("="*60 + "\n")

    # ==================== 创建闭集 ELOT 损失 ====================
    elot_criterion = ELOTClosedLoss(
        t_dim=t_dim,
        s_dim=s_dim,
        proj_dim=opt.proj_dim,
        num_classes_student=num_classes_student,
        class_indices=class_indices,       # 关键: 传入子集的原始类别列表
        lambda_mmd=opt.lambda_mmd,
        epsilon=opt.ot_epsilon,
        sigma=opt.mmd_sigma,
        warmup_iters=opt.warmup_iters,
        nb_dummies=opt.nb_dummies,
    )

    # ==================== 组装模块列表 ====================
    # module_list: [student, elot_loss, teacher]
    module_list = nn.ModuleList([])
    module_list.append(model_s)         # [0] 学生
    module_list.append(elot_criterion)  # [1] ELOT 损失 (含映射层)
    module_list.append(model_t)         # [2] 教师

    # trainable_list: 只有学生和映射层参与优化
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    trainable_list.append(elot_criterion)

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

    # ==================== GPU 设置 ====================
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # ==================== 验证教师准确率 ====================
    # 注意: 教师是 100 类, 验证集是 20 类 (标签 0-19)
    # 但教师输出 100 维, CrossEntropyLoss 取的是前 20 类的 logit
    # 所以这里的准确率不完全反映教师在这 20 类上的真实水平
    # (因为教师输出的 100 类中, 可能其他类的 logit 更高)
    # 这只是一个粗略的健全性检查
    print("注意: 教师模型有 100 类输出, 而验证集只有 20 类")
    print("以下教师准确率仅供参考 (教师可能把一些图片分到子集外的类)")

    # 为了正确评估, 我们需要一个简单的 wrapper
    # 但为了简洁, 这里跳过教师在子集上的精确评估
    # 你可以自行添加: 把教师 logit 的子集列筛选出来再算准确率

    # ==================== FitNet 风格映射层预训练 (可选) ====================
    if opt.pretrain_proj:
        pretrain_projections(
            model_s=model_s,
            model_t=model_t,
            elot_loss_fn=elot_criterion,
            train_loader=train_loader,
            opt=opt,
            num_epochs=opt.pretrain_epochs,
            lr=opt.pretrain_lr
        )

    # ==================== 全局迭代计数器 ====================
    global_iter_counter = [0]

    # ==================== 主训练循环 ====================
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> 训练中... Epoch {}/{}".format(epoch, opt.epochs))

        time1 = time.time()

        train_acc, train_loss = train_distill_elot_closed(
            epoch, train_loader, module_list, criterion_list,
            optimizer, opt, global_iter_counter
        )

        time2 = time.time()
        print('Epoch {}, 耗时 {:.2f}s, 迭代步数: {}'.format(
            epoch, time2 - time1, global_iter_counter[0]))

        # ==================== 测试集评估 ====================
        # validate 内部会调用 model(input, is_feat=True)
        # 学生输出 20 类, 标签 0-19 → 合法
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
                'best_acc': best_acc,
                'subset_id': opt.subset_id,
                'class_indices': class_indices,
            }
            save_file = os.path.join(
                opt.save_folder, '{}_best.pth'.format(opt.model_s)
            )
            print('保存最佳模型! Acc: {:.2f}%'.format(best_acc))
            torch.save(state, save_file)

        # ==================== 定期保存 ====================
        if epoch % opt.save_freq == 0:
            print('==> 保存 checkpoint...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'elot_loss': elot_criterion.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{}.pth'.format(epoch)
            )
            torch.save(state, save_file)

    # ==================== 训练完成 ====================
    print("\n" + "="*60)
    print("训练完成!")
    print("  子集 {}: 类别 {} ~ {}".format(
        opt.subset_id, class_indices[0], class_indices[-1]))
    print("  最佳准确率: {:.2f}%".format(best_acc))
    print("="*60)

    # 保存最终模型
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'elot_loss': elot_criterion.state_dict(),
        'subset_id': opt.subset_id,
        'class_indices': class_indices,
    }
    save_file = os.path.join(
        opt.save_folder, '{}_last.pth'.format(opt.model_s)
    )
    torch.save(state, save_file)


if __name__ == '__main__':
    main()