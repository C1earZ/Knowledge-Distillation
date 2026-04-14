# from __future__ import print_function 是为了兼容 Python2 的 print 语法
# 在 Python3 中可以忽略，但保留不影响运行
from __future__ import print_function

import os          # 操作系统接口：文件路径拼接、创建文件夹、判断文件是否存在等
import argparse    # 命令行参数解析器：让你可以在终端通过 --xxx 传入参数
import socket      # 网络模块：这里只用到 gethostname() 获取机器名
import time        # 时间模块：用来计算每个 epoch 的训练耗时

import tensorboard_logger as tb_logger  # TensorBoard 日志记录器：把训练过程的指标保存成可视化图表
import torch                            # PyTorch 核心库：张量操作、自动求导等
import torch.optim as optim             # 优化器模块：SGD、Adam 等梯度下降算法
import torch.nn as nn                   # 神经网络模块：各种层（卷积、全连接、BN等）和损失函数
import torch.backends.cudnn as cudnn    # cuDNN 后端：NVIDIA 的深度学习加速库

from models import model_dict  # 从 models/__init__.py 导入模型字典
                                # model_dict = {'resnet110': resnet110, 'wrn_40_2': wrn_40_2, ...}
                                # 通过名字就能找到对应的模型构造函数

from dataset.cifar100 import get_cifar100_dataloaders  # 导入 CIFAR-100 数据集加载函数
                                                        # 返回 train_loader 和 val_loader

from helper.util import adjust_learning_rate, accuracy, AverageMeter
# adjust_learning_rate: 根据当前 epoch 调整学习率（到了指定 epoch 就衰减）
# accuracy: 计算 top1 和 top5 准确率
# AverageMeter: 工具类，用于跟踪计算一系列数值的平均值

from helper.loops import train_vanilla as train, validate
# train_vanilla: 标准的训练循环函数（不含蒸馏），导入后重命名为 train
# validate: 验证/测试函数，在测试集上评估模型表现


def parse_option():
    """
    解析命令行参数，配置所有训练超参数和路径。
    返回一个 opt 对象，后续代码通过 opt.xxx 访问各参数。
    """

    hostname = socket.gethostname()  # 获取当前计算机名，用于判断在哪台机器上运行
                                     # 比如返回 "visiongpu03" 或 "my-laptop"

    parser = argparse.ArgumentParser('argument for training')  # 创建参数解析器

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

    # ==================== 优化器参数 ====================
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    # 初始学习率 0.05

    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    # 在第 150、180、210 个 epoch 时衰减学习率
    # 用字符串格式是因为命令行只能传字符串，后面会解析成整数列表

    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    # 每次衰减为原来的 0.1 倍
    # epoch 1-149: lr=0.05, epoch 150-179: lr=0.005, epoch 180-209: lr=0.0005, epoch 210-240: lr=0.00005

    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    # L2 正则化系数，防止模型参数过大导致过拟合

    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # SGD 动量系数，让参数更新有惯性，减少震荡

    # ==================== 模型和数据集 ====================
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'MobileNetV2', 'ResNet50',])
    # 选择模型架构，默认是 resnet110
    # choices 限定了只能从这些模型中选，传入其他名字会报错

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    # 数据集，目前只支持 CIFAR-100

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    # 实验编号，用于区分同一配置的多次重复实验
    # -t 是简写，--trial 是全称，两种方式都可以传

    opt = parser.parse_args()  # 解析命令行参数，把所有参数值存入 opt 对象
                                # 没有传的参数使用 default 的值

    # ==================== 特殊模型的学习率调整 ====================
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01
    # 轻量级模型参数少，每个参数对输出的影响更大
    # 用默认的 0.05 学习率太大会导致训练不稳定，所以降到 0.01

    # ==================== 根据机器设置存储路径 ====================
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'          # 实验室 GPU 服务器的路径
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'               # 其他机器（包括你自己的电脑）的默认路径
        opt.tb_path = './save/tensorboard'

    # ==================== 解析学习率衰减节点 ====================
    iterations = opt.lr_decay_epochs.split(',')  # '150,180,210' → ['150', '180', '210']（字符串列表）
    opt.lr_decay_epochs = list([])               # 清空，准备存放整数列表
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))      # 把每个字符串转成整数，追加到列表
    # 最终 opt.lr_decay_epochs = [150, 180, 210]

    # ==================== 生成实验名称 ====================
    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)
    # 拼接成类似 'resnet110_cifar100_lr_0.05_decay_0.0005_trial_0' 的字符串
    # 用于命名保存模型和日志的文件夹，一眼就能看出实验配置

    # ==================== 创建 TensorBoard 日志文件夹 ====================
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # 拼接路径：'./save/tensorboard/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0'
    if not os.path.isdir(opt.tb_folder):   # 如果文件夹不存在
        os.makedirs(opt.tb_folder)         # 递归创建（连父目录一起创建）

    # ==================== 创建模型 checkpoint 保存文件夹 ====================
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    # 拼接路径：'./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0'
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt  # 返回配置好的参数对象


def main():
    best_acc = 0  # 用于跟踪训练过程中的最高测试准确率，初始为 0

    opt = parse_option()  # 解析所有命令行参数，拿到配置

    # ==================== 加载数据集 ====================
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        # train_loader: 训练集加载器，每次返回一个 batch 的 (图片, 标签)
        # val_loader: 测试集加载器，用于评估模型表现
        n_cls = 100  # CIFAR-100 有 100 个类别
    else:
        raise NotImplementedError(opt.dataset)  # 不支持的数据集，抛出异常

    # ==================== 创建模型 ====================
    model = model_dict[opt.model](num_classes=n_cls)
    # model_dict['resnet110'] 取出 resnet110 这个构造函数
    # 然后调用 resnet110(num_classes=100) 创建一个 110 层的 ResNet
    # 此时参数是随机初始化的

    # ==================== 创建优化器 ====================
    optimizer = optim.SGD(model.parameters(),        # 优化模型的所有参数
                          lr=opt.learning_rate,       # 学习率 0.05
                          momentum=opt.momentum,      # 动量 0.9
                          weight_decay=opt.weight_decay)  # L2 正则化 5e-4

    # ==================== 创建损失函数 ====================
    criterion = nn.CrossEntropyLoss()
    # 交叉熵损失函数，用于分类任务
    # 输入是模型的预测 logit 和真实标签，输出一个标量 loss

    # ==================== GPU 设置 ====================
    if torch.cuda.is_available():          # 如果有 GPU 可用
        model = model.cuda()               # 把模型搬到 GPU 上
        criterion = criterion.cuda()       # 把损失函数也搬到 GPU 上
        cudnn.benchmark = True             # 让 cuDNN 自动选择最快的卷积算法
                                           # 第一次运行会慢一点（在尝试不同算法）
                                           # 之后每次都用最快的那个

    # ==================== 创建 TensorBoard 日志记录器 ====================
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # logdir: 日志保存到哪个文件夹
    # flush_secs=2: 每 2 秒把缓存中的日志数据写入硬盘

    # ==================== 训练循环 ====================
    for epoch in range(1, opt.epochs + 1):  # 从第 1 个 epoch 循环到第 240 个 epoch

        adjust_learning_rate(epoch, opt, optimizer)
        # 根据当前 epoch 调整学习率
        # 到了第 150/180/210 个 epoch 时，学习率乘以 0.1

        print("==> training...")  # 打印提示信息

        time1 = time.time()  # 记录训练开始的时间戳（秒）
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        # 执行一个 epoch 的训练：
        # 内部会遍历 train_loader 的所有 batch
        # 每个 batch 做：前向传播 → 算 loss → 反向传播 → 更新参数
        # 返回这个 epoch 的平均训练准确率和平均 loss
        time2 = time.time()  # 记录训练结束的时间戳
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # 打印这个 epoch 花了多少秒，比如 'epoch 1, total time 45.32'

        # ==================== 记录训练指标到 TensorBoard ====================
        logger.log_value('train_acc', train_acc, epoch)    # 记录训练准确率
        logger.log_value('train_loss', train_loss, epoch)  # 记录训练损失

        # ==================== 在测试集上评估 ====================
        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
        # 在测试集上评估模型，不更新参数
        # 返回 top1 准确率、top5 准确率、测试 loss

        # ==================== 记录测试指标到 TensorBoard ====================
        logger.log_value('test_acc', test_acc, epoch)            # 记录测试 top1 准确率
        logger.log_value('test_acc_top5', test_acc_top5, epoch)  # 记录测试 top5 准确率
        logger.log_value('test_loss', test_loss, epoch)          # 记录测试损失

        # ==================== 保存最佳模型 ====================
        if test_acc > best_acc:    # 如果当前 epoch 的测试准确率超过了历史最佳
            best_acc = test_acc    # 更新历史最佳准确率
            state = {
                'epoch': epoch,                       # 当前是第几个 epoch
                'model': model.state_dict(),          # 模型所有层的参数（权重和偏置）
                'best_acc': best_acc,                 # 最佳准确率
                'optimizer': optimizer.state_dict(),  # 优化器状态（动量缓存等）
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            # 保存路径，比如 './save/models/.../resnet110_best.pth'
            print('saving the best model!')
            torch.save(state, save_file)  # 保存到硬盘，覆盖之前的 best 模型

        # ==================== 定期保存 checkpoint ====================
        if epoch % opt.save_freq == 0:    # 每 40 个 epoch（epoch=40, 80, 120, ...）
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,                 # 注意：这里存的是当前准确率，不是 best
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            # 每次存一个新文件，比如 ckpt_epoch_40.pth、ckpt_epoch_80.pth
            # 不会覆盖之前的，用于训练中断后从最近的 checkpoint 恢复
            torch.save(state, save_file)

    # ==================== 训练结束后的收尾工作 ====================
    print('best accuracy:', best_acc)
    # 打印整个训练过程中的最高准确率
    # 注释说明：论文中报告的结果用的是最后一个 epoch 的模型，不是 best 的

    # 保存最后一个 epoch 的模型
    state = {
        'opt': opt,                           # 保存所有超参数配置（方便以后查看用了什么设置）
        'model': model.state_dict(),          # 最后一个 epoch 的模型参数
        'optimizer': optimizer.state_dict(),  # 最后一个 epoch 的优化器状态
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    # 比如 './save/models/.../resnet110_last.pth'
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
    # 只有直接运行 python train_teacher.py 时才执行 main()
    # 如果被其他文件 import，不会自动执行训练