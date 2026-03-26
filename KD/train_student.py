
from __future__ import print_function

import os
import argparse
import socket
import time
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy.distutils.core import numpy_cmdclass

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate
# 根据当前 epoch 调整学习率（到指定 epoch 时衰减）

from distiller_zoo import DistillKL, HintLoss


from helper.loops import train_distill as train, validate, train_distill
# train_distill: 标准蒸馏训练循环，导入后重命名为 train
# validate: 标准验证函数


from helper.pretrain import init
# init: 两阶段方法（FitNet、FSP 等）的预训练函数
# 用于先训练适配模块（regressor），再做整体蒸馏

import numpy as np
import random


def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes


def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')
    # ==================== 训练基础参数 ====================
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    # 每训练 100 个 batch 打印一次训练信息（loss、准确率等）
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    # 每 500 个 batch 记录一次 TensorBoard 日志
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    # 每 40 个 epoch 保存一次模型 checkpoint
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    # 每批训练 64 张图片
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    # 数据加载用 8 个子进程并行读取，加速数据预处理
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    # 总共训练 240 个 epoch
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')
    # 两阶段方法（FitNet、FSP 等）的预训练阶段用 30 个 epoch
    # 先训练适配模块（regressor），再做整体蒸馏
    # ==================== 优化器参数 ====================
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    # 初始学习率 0.05
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    # 在第 150、180、210 个 epoch 时衰减学习率
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    # 每次衰减为原来的 0.1 倍
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    # L2 正则化系数，防止过拟合
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # SGD 动量系数
    # ==================== 数据集 ====================
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    # 数据集，目前只支持 CIFAR-100
    # ==================== 模型 ====================
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    # student 模型架构，默认是 resnet8（小模型）
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    # teacher 模型 checkpoint 文件的路径
    # ==================== 蒸馏方法 ====================
    parser.add_argument('--distill', type=str, default='kd', choices=['afd', 'ickd', 'kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst', 'close'])
    # 选择蒸馏方法，默认是 kd（标准知识蒸馏）
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    # 实验编号，用于区分同一配置的多次重复实验
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # 随机种子，保证实验可复现
    # ==================== 损失函数权重 ====================
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    # 分类损失（CrossEntropy）的权重
    # total_loss = gamma * L_cls + alpha * L_div + beta * L_kd
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    # KL 散度蒸馏损失的权重
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')
    # 其他蒸馏损失（特征匹配等）的权重
    # ==================== KL 蒸馏参数 ====================
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    # 温度参数 T=4，让 softmax 输出更平滑
    # T 越大，概率分布越平滑，暗知识（dark knowledge）越多
    # ==================== FitNet 参数 ====================
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])
    # FitNet 方法中选择哪一层做特征匹配
    opt = parser.parse_args()  # 解析所有命令行参数，存入 opt 对象
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.011
    # ==================== 根据机器设置存储路径 ====================
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'
    # ==================== 学习率衰减 ====================
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    # ==================== 从 teacher 路径中解析 teacher 模型名 ====================
    opt.model_t = get_teacher_name(opt.path_t)
    # ==================== 生成实验名称 ====================
    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    # ==================== 创建 TensorBoard 日志文件夹 ====================
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    # ==================== 创建模型保存文件夹 ====================
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def get_teacher_name(model_path):
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]    # 普通模型直接返回第一段，如 'resnet110'
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]
        # WRN 模型拼接三段：'wrn' + '_' + '40' + '_' + '2' = 'wrn_40_2'

def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model   # 返回加载好权重的 teacher 模型


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    best_acc = 0

    opt = parse_option()

    # ==================== 固定随机种子 ====================
    set_random_seed(opt.seed, True)
    # ==================== 创建 TensorBoard 日志记录器 ====================
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # ==================== 加载数据集 ====================
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,    # 负样本数量
                                                                               mode=opt.mode)  # 正样本选择方式
        else:
            # 其他蒸馏方法用标准 dataloader
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            # is_instance=True: 每个 batch 额外返回样本索引 (image, label, index)
    else:
        raise NotImplementedError(opt.dataset)

    # ==================== 加载 teacher 和创建 student ====================
    num_classes = 100;
    model_t = load_teacher(opt.path_t, num_classes)
    # 加载预训练的 teacher 模型，numclasses 个类（CIFAR-100 全部类别）
    model_s = model_dict[opt.model_s](num_classes)
    # ==================== 探测各层特征的 shape ====================
    data = torch.randn(2, 3, 32, 32)
    # 生成随机假数据：2 张 3 通道 32×32 的图片
    # 只是为了探测 shape，不是真实图片

    model_t.eval()   # teacher 设为评估模式（关闭 Dropout 和 BN 的随机行为）
    model_s.eval()   # student 也暂时设为评估模式

    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)
    # ==================== 创建模块列表 ====================
    module_list = nn.ModuleList([])
    module_list.append(model_s)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # ==================== 创建损失函数 ====================
    criterion_cls = nn.CrossEntropyLoss()
    # 分类损失：student 的预测 vs 真实标签

    # criterion_div = DistillKL(opt.kd_T)

    #criterion_fea = CCLoss(opt)
    # 被注释掉的特征损失，作者实验过但最终没用

    # ==================== 根据蒸馏方法创建第三项损失 ====================
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'elot':
        criterion_kd = DistillKL(opt.kd_T)
    else:
        raise NotImplementedError(opt.distill)

    # ==================== 把三个损失函数打包 ====================
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # [0] 分类损失
    criterion_list.append(criterion_kd)     # [1] elot损失

    # ==================== 创建优化器 ====================
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # ==================== 把 teacher 加入 module_list ====================
    module_list.append(model_t)

    # ==================== GPU 设置 ====================
    if torch.cuda.is_available():
        module_list.cuda()       # 把 student 和 teacher 都搬到 GPU
        criterion_list.cuda()    # 把所有损失函数搬到 GPU

    # ==================== 验证 teacher 准确率（健全性检查）====================
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)

    print('teacher accuracy: ', teacher_acc)


    # ==================== 主训练循环 ====================
    for epoch in range(1, opt.epochs + 1):   # 从第 1 到第 240 个 epoch

        adjust_learning_rate(epoch, opt, optimizer)
        # 到了 150/180/210 epoch 时学习率乘以 0.1
        print("==> training...")

        time1 = time.time()  # 记录开始时间
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        # 执行一个 epoch 的开放集蒸馏训练
        # 内部遍历所有 batch，做前向传播、算 loss、反向传播、更新参数
        # 使用 Partial OT 处理 teacher(100类) 和 student(21类) 的类别不匹配
        # 返回平均训练准确率和平均 loss
        time2 = time.time()  # 记录结束时间
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # 打印耗时，如 'epoch 1, total time 45.32'

        # ==================== 记录训练指标 ====================
        logger.log_value('train_acc', train_acc, epoch)    # 训练准确率
        logger.log_value('train_loss', train_loss, epoch)  # 训练损失

        # ==================== 在测试集上评估 student ====================
        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)


        # ==================== 记录测试指标 ====================
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # ==================== 保存最佳模型 ====================
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # ==================== 定期保存 checkpoint ====================
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))

            torch.save(state, save_file)

    # ==================== 训练结束后的收尾工作 ====================

    print('best accuracy:', best_acc)


    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))

    torch.save(state, save_file)


if __name__ == '__main__':
    main()