
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
from models import model_dict                          # 模型字典
from dataset.cifar100 import get_cifar100_dataloaders   # 数据加载
from helper.util import adjust_learning_rate            # 学习率调度
from helper.loops import validate                       # 测试集评估

# ==================== 导入 ELOT 相关新模块 ====================
from distiller_zoo.ELOT import ELOTLoss                 # ELOT 损失
from helper.train_elot import train_distill_elot         # ELOT 训练循环


# ============================================================
# 工具函数 (从 train_student.py 复用)
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
    """从 checkpoint 路径中解析教师模型的架构名
    路径格式: .../resnet110_cifar100_.../resnet110_best.pth
    → 取倒数第二段目录名, 用 '_' 分割取第一个词
    """
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]    # 如 'resnet110'
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]
        # 如 'wrn_40_2'


def load_teacher(model_path, n_cls):
    """加载预训练的教师模型"""
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
    """解析命令行参数"""
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('ELOT 知识蒸馏训练')

    # ==================== 训练基础参数 ====================
    parser.add_argument('--print_freq', type=int, default=100,
                        help='打印训练信息的间隔 (每多少个 batch 打印一次)')
    parser.add_argument('--save_freq', type=int, default=40,
                        help='保存 checkpoint 的间隔 (每多少个 epoch 保存一次)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练批次大小')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载的工作进程数')
    parser.add_argument('--epochs', type=int, default=240,
                        help='总训练轮数')

    # ==================== 优化器参数 ====================
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='初始学习率')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210',
                        help='学习率衰减的 epoch 节点, 逗号分隔')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='每次衰减的倍率 (衰减为原来的 lr_decay_rate 倍)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='L2 正则化系数')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD 动量')

    # ==================== 模型与数据集 ====================
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100'],
                        help='数据集 (目前只支持 cifar100)')
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32',
                                 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4',
                                 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'ResNet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'],
                        help='学生模型架构')
    parser.add_argument('--path_t', type=str, required=True,
                        help='教师模型 checkpoint 路径 (必须指定)')

    # ==================== 损失权重 ====================
    parser.add_argument('-r', '--gamma', type=float, default=1.0,
                        help='分类损失 (CrossEntropy) 的权重 γ')
    parser.add_argument('-b', '--beta', type=float, default=0.5,
                        help='ELOT 传输损失的权重 β')

    # ==================== ELOT 专有参数 ====================
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='映射层的输出维度 (教师和学生权重/特征映射到此维度)')
    parser.add_argument('--lambda_mmd', type=float, default=0.1,
                        help='公式2中 MMD 项的权重 λ')
    parser.add_argument('--ot_epsilon', type=float, default=0.1,
                        help='OT 的熵正则化系数 (0=精确EMD, >0=Sinkhorn)')
    parser.add_argument('--mmd_sigma', type=float, default=1.0,
                        help='RBF 核的带宽参数 σ')
    parser.add_argument('--warmup_iters', type=int, default=500,
                        help='前多少次迭代使用标准 OT 作为 warmup '
                             '(参考代码中 id_iter <= 3 的逻辑)')
    parser.add_argument('--nb_dummies', type=int, default=1,
                        help='ELOT 虚拟点数量')

    # ==================== 其他 ====================
    parser.add_argument('--trial', type=str, default='1',
                        help='实验编号, 用于区分同一配置的多次重复实验')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')

    opt = parser.parse_args()

    # ---- 特殊模型的学习率调整 (与 train_student.py 一致) ----
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # ---- 解析教师模型名 ----
    opt.model_t = get_teacher_name(opt.path_t)

    # ---- 解析学习率衰减节点 ----
    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(',')]

    # ---- 生成实验名称 (包含关键超参数, 方便对比实验) ----
    opt.model_name = (
        'S:{}_T:{}_elot_r:{}_b:{}_proj:{}_lmmd:{}_eps:{}_wup:{}_trial:{}'.format(
            opt.model_s, opt.model_t, opt.gamma, opt.beta,
            opt.proj_dim, opt.lambda_mmd, opt.ot_epsilon,
            opt.warmup_iters, opt.trial
        )
    )

    # ---- 设置存储路径 ----
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
    logger = None
    if tb_logger is not None:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # ==================== 加载数据 ====================
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            is_instance=True  # 返回 (图片, 标签, 索引) 三元组
        )
        num_classes = 100
    else:
        raise NotImplementedError(opt.dataset)

    # ==================== 创建模型 ====================
    model_t = load_teacher(opt.path_t, num_classes)
    model_s = model_dict[opt.model_s](num_classes=num_classes)

    # ==================== 探测特征维度 ====================
    # 用随机假数据做一次前向传播, 获取各层特征的形状
    data_dummy = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()

    feat_t, _ = model_t(data_dummy, is_feat=True)
    feat_s, _ = model_s(data_dummy, is_feat=True)

    # 获取 FC 层之前的特征维度 (即列表最后一个元素的第二维)
    t_dim = feat_t[-1].shape[1]  # 教师特征维度
    s_dim = feat_s[-1].shape[1]  # 学生特征维度

    print('教师特征维度: {}, 学生特征维度: {}, 映射维度: {}'.format(
        t_dim, s_dim, opt.proj_dim))
    print('类别数: {}'.format(num_classes))

    # ==================== 创建 ELOT 损失模块 ====================
    elot_criterion = ELOTLoss(
        t_dim=t_dim,               # 教师特征维度
        s_dim=s_dim,               # 学生特征维度
        proj_dim=opt.proj_dim,     # 映射后统一维度
        num_classes=num_classes,   # 类别总数
        lambda_mmd=opt.lambda_mmd, # MMD 权重 λ
        epsilon=opt.ot_epsilon,    # OT 正则化系数
        sigma=opt.mmd_sigma,       # RBF 核带宽
        warmup_iters=opt.warmup_iters,  # warmup 迭代数
        nb_dummies=opt.nb_dummies,      # ELOT 虚拟点数
    )

    # ==================== 组装模块列表 ====================
    # module_list 结构: [student, elot_loss, teacher]
    # 与 train_student.py 中的组织方式一致:
    #   module_list[0] = student, module_list[-1] = teacher
    module_list = nn.ModuleList([])
    module_list.append(model_s)         # [0] 学生模型
    module_list.append(elot_criterion)  # [1] ELOT 损失 (含可训练映射层)
    module_list.append(model_t)         # [2] 教师模型 (参数固定)

    # trainable_list: 只有学生模型和 ELOT 映射层参与优化
    # 教师模型不在其中, 所以不会被更新
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    trainable_list.append(elot_criterion)  # 映射层也需要训练

    # ==================== 损失函数列表 ====================
    criterion_cls = nn.CrossEntropyLoss()
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # [0] 分类损失

    # ==================== 创建优化器 ====================
    # 优化学生模型 + ELOT 映射层的所有参数
    optimizer = optim.SGD(
        trainable_list.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )

    # ==================== GPU 设置 ====================
    if torch.cuda.is_available():
        module_list.cuda()       # 把 student、elot_loss、teacher 都搬到 GPU
        criterion_list.cuda()    # 把损失函数搬到 GPU
        cudnn.benchmark = True

    # ==================== 验证教师准确率 (健全性检查) ====================
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('教师模型准确率: {:.2f}%'.format(teacher_acc))

    # ==================== 全局迭代计数器 ====================
    # 使用列表包裹 int, 以便在 train_distill_elot 函数内部修改
    global_iter_counter = [0]

    # ==================== 主训练循环 ====================
    for epoch in range(1, opt.epochs + 1):
        # 调整学习率 (到了 150/180/210 epoch 时衰减)
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> 训练中... Epoch {}/{}".format(epoch, opt.epochs))

        time1 = time.time()

        # 执行一个 epoch 的 ELOT 蒸馏训练
        train_acc, train_loss = train_distill_elot(
            epoch, train_loader, module_list, criterion_list,
            optimizer, opt, global_iter_counter
        )

        time2 = time.time()
        print('Epoch {}, 耗时 {:.2f}s, 全局迭代步数: {}'.format(
            epoch, time2 - time1, global_iter_counter[0]))

        # ==================== 测试集评估 ====================
        # 复用原框架的 validate 函数
        test_acc, test_acc_top5, test_loss = validate(
            val_loader, model_s, criterion_cls, opt
        )

        # ==================== 记录日志 ====================
        if logger is not None:
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
                'elot_loss': elot_criterion.state_dict(),  # 保存映射层权重
                'best_acc': best_acc,
            }
            save_file = os.path.join(
                opt.save_folder, '{}_best.pth'.format(opt.model_s)
            )
            print('保存最佳模型! Acc: {:.2f}%'.format(best_acc))
            torch.save(state, save_file)

        # ==================== 定期保存 checkpoint ====================
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
    print('\n训练完成! 最佳准确率: {:.2f}%'.format(best_acc))

    # 保存最终模型
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'elot_loss': elot_criterion.state_dict(),
    }
    save_file = os.path.join(
        opt.save_folder, '{}_last.pth'.format(opt.model_s)
    )
    torch.save(state, save_file)


if __name__ == '__main__':
    main()