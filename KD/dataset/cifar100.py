# ============================================================
# cifar100.py - CIFAR-100 数据集加载模块
# 提供三种数据加载方式：
#   1. 标准加载（get_cifar100_dataloaders）
#   2. 子集加载，只取前20类（get_cifar100_subset_dataloaders）
#   3. 对比学习采样加载（get_cifar100_dataloaders_sample）
# ============================================================

from __future__ import print_function  # 兼容Python2的print语法，在Python3中无实际作用

import os  # 用于文件路径操作，比如判断目录是否存在、创建目录
import socket  # 用于获取当前机器的主机名，根据不同服务器选择不同数据路径
import numpy as np  # 数值计算库，用于数组操作（如筛选样本索引）
from torch.utils.data import DataLoader  # PyTorch数据加载器，负责批量读取、打乱、多线程加载数据
from torchvision import datasets, transforms  # datasets:内置数据集(CIFAR100等); transforms:图像预处理操作
from PIL import Image  # Python图像处理库，用于将numpy数组转换为PIL Image格式

"""
均值和标准差（用于图像标准化），这里注释掉了但保留作参考
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),  # CIFAR-100数据集 RGB三通道的均值
}
std = {
    'cifar100': (0.2675, 0.2565, 0.2761),  # CIFAR-100数据集 RGB三通道的标准差
}
"""


def get_data_folder():
    """
    根据当前运行的服务器主机名，返回数据集的存储路径
    不同服务器上数据集放的位置不一样，用这个函数自动判断
    """
    hostname = socket.gethostname()  # 获取当前机器的主机名，例如 "visiongpu01" 或 "yonglong-home"

    # 根据主机名前缀判断是哪台服务器，返回对应的数据路径
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'  # visiongpu服务器上的数据路径
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'  # yonglong个人电脑上的数据路径
    else:
        data_folder = './data/'  # 其他机器（包括本地开发）默认放在当前目录的data文件夹下

    if not os.path.isdir(data_folder):  # 判断该路径是否已经存在
        os.makedirs(data_folder)  # 如果不存在则自动创建（包括所有中间目录）

    return data_folder  # 返回数据存储路径字符串


# ============================================================
# 第一个类：CIFAR100Instance
# 继承自 torchvision 的 CIFAR100，重写 __getitem__ 方法
# 区别：在返回 (图片, 标签) 的基础上，额外返回样本的 index（索引）
# 用途：训练时需要知道每个样本在数据集中的位置（CRD等对比学习方法需要）
# ============================================================
class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    在标准CIFAR100基础上，__getitem__额外返回样本索引index
    """

    def __getitem__(self, index):
        # index: DataLoader传入的样本索引，表示取第index个样本

        if self.train:
            img, target = self.data[index], self.targets[index]  # 训练集：取第index张图和对应标签
        else:
            img, target = self.data[index], self.targets[index]  # 测试集：同上（这里两个分支代码相同，实际可合并）

        # 将 numpy 数组格式的图片转换为 PIL Image 格式
        # 原因：torchvision的transforms期望输入是PIL Image
        # self.data[index] 是 shape为(32,32,3) 的 numpy uint8 数组
        img = Image.fromarray(img)

        if self.transform is not None:  # 如果定义了图像变换（如RandomCrop、Normalize等）
            img = self.transform(img)  # 对图片应用变换，变换后img变为 Tensor

        if self.target_transform is not None:  # 如果定义了标签变换（通常为None）
            target = self.target_transform(target)  # 对标签应用变换

        return img, target, index  # 返回三元组：(图片Tensor, 类别标签, 样本在数据集中的索引)


# ============================================================
# 第一个函数：get_cifar100_dataloaders
# 标准的CIFAR-100数据加载，返回训练集和测试集的DataLoader
# is_instance=True 时额外返回训练集大小 n_data（CRD方法需要用到）
# ============================================================
def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    cifar 100 标准数据加载器
    batch_size: 每批次加载多少张图片
    num_workers: 用多少个子进程并行加载数据，越大加载越快但占内存
    is_instance: 是否使用CIFAR100Instance（额外返回样本索引）
    """
    data_folder = get_data_folder()  # 获取数据集存储路径

    # 训练集的图像预处理流水线（transforms.Compose将多个操作串联起来）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪：先在图片四周各填充4个像素，再随机裁剪回32×32
        # 作用：数据增强，让模型对物体位置不敏感
        transforms.RandomHorizontalFlip(),  # 随机水平翻转（以50%概率左右镜像）
        # 作用：数据增强，让模型对方向不敏感
        transforms.ToTensor(),  # 将PIL Image (H,W,C) uint8 转为 Tensor (C,H,W) float32
        # 同时将像素值从[0,255]缩放到[0.0,1.0]
        transforms.Normalize(  # 标准化：(pixel - mean) / std
            (0.5071, 0.4867, 0.4408),  # CIFAR-100 RGB三通道的均值
            (0.2675, 0.2565, 0.2761)  # CIFAR-100 RGB三通道的标准差
        ),  # 标准化后数据分布接近均值0方差1，有利于训练稳定
    ])

    # 测试集的图像预处理（比训练集少了随机增强操作，保证测试结果可复现）
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor并缩放到[0,1]
        transforms.Normalize(  # 同样做标准化，使用与训练集相同的均值和标准差
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
    ])

    if is_instance:
        # 使用自定义的CIFAR100Instance类，__getitem__会额外返回样本索引
        train_set = CIFAR100Instance(
            root=data_folder,  # 数据集存储/下载的根目录
            download=True,  # 如果本地没有则自动下载
            train=True,  # 加载训练集（True）还是测试集（False）
            transform=train_transform  # 应用训练集预处理
        )
        n_data = len(train_set)  # 记录训练集总样本数（CIFAR-100训练集共50000张）
    else:
        # 使用标准的torchvision CIFAR100类
        train_set = datasets.CIFAR100(
            root=data_folder,
            download=True,
            train=True,
            transform=train_transform
        )

    # 创建训练集的DataLoader（负责批量、随机、多线程地提供数据）
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,  # 每次迭代返回batch_size张图片
        shuffle=True,  # 每个epoch开始时打乱数据顺序，防止模型记住顺序
        num_workers=num_workers  # 使用多少个子进程预加载数据（提高GPU利用率）
    )

    # 创建测试集Dataset（使用test_transform，无数据增强）
    test_set = datasets.CIFAR100(
        root=data_folder,
        download=True,
        train=False,  # False表示加载测试集（CIFAR-100测试集共10000张）
        transform=test_transform
    )

    # 创建测试集的DataLoader
    test_loader = DataLoader(
        test_set,
        batch_size=int(batch_size / 2),  # 测试集batch_size用一半，因为测试时不需要反向传播，显存压力小
        shuffle=False,  # 测试集不需要打乱，保持固定顺序便于复现结果
        num_workers=int(num_workers / 2)  # 测试集用一半的worker数量
    )

    if is_instance:
        return train_loader, test_loader, n_data  # 额外返回训练集大小（CRD方法建立memory bank需要）
    else:
        return train_loader, test_loader  # 只返回两个DataLoader
