# ============================================================
# cifar100_subset.py - CIFAR-100 子集数据加载模块 (闭集蒸馏专用)
#
# 放置位置: KD/dataset/cifar100_subset.py
#
# 功能: 将 CIFAR-100 的 100 个类均匀分成 5 个子集, 每个子集 20 类
#   子集 0: 类 [0,  1,  ..., 19]
#   子集 1: 类 [20, 21, ..., 39]
#   子集 2: 类 [40, 41, ..., 59]
#   子集 3: 类 [60, 61, ..., 79]
#   子集 4: 类 [80, 81, ..., 99]
#
# 学生模型只看某一个子集, 标签会被重新映射到 [0, 19]
# 教师模型仍然用完整的 100 类数据集训练 (无需修改)
#
# 使用方法:
#   from dataset.cifar100_subset import get_cifar100_closed_subset_dataloaders
#   train_loader, val_loader, n_data, class_indices = \
#       get_cifar100_closed_subset_dataloaders(subset_id=0, batch_size=64)
# ============================================================

from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def get_data_folder():
    """
    根据当前运行的服务器主机名, 返回数据集的存储路径
    与 cifar100.py 中的 get_data_folder() 完全一致
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


# ============================================================
# 定义 5 个子集的类别划分
# 每个子集包含 20 个连续的类 (按 CIFAR-100 原始类别编号)
# ============================================================
SUBSET_CLASSES = {
    0: list(range(0, 20)),    # 子集0: 类 0-19
    1: list(range(20, 40)),   # 子集1: 类 20-39
    2: list(range(40, 60)),   # 子集2: 类 40-59
    3: list(range(60, 80)),   # 子集3: 类 60-79
    4: list(range(80, 100)),  # 子集4: 类 80-99
}

NUM_SUBSETS = 5        # 总共 5 个子集
CLASSES_PER_SUBSET = 20  # 每个子集 20 个类


class CIFAR100SubsetInstance(datasets.CIFAR100):
    """
    CIFAR-100 子集数据集, 在标准 CIFAR100 基础上:
      1. 只保留指定子集中的类别
      2. 将原始标签重映射到 [0, 19]
      3. __getitem__ 额外返回样本索引 index (蒸馏方法需要)

    参数:
        subset_id: int, 子集编号 0-4
        其余参数与 torchvision.datasets.CIFAR100 一致
    """

    def __init__(self, root, subset_id=0, train=True,
                 transform=None, target_transform=None, download=False):
        # 先调用父类构造函数, 加载完整的 CIFAR-100 数据
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        # 获取当前子集包含的原始类别列表
        assert subset_id in SUBSET_CLASSES, \
            "subset_id 必须在 0-4 之间, 收到: {}".format(subset_id)
        self.subset_id = subset_id
        self.class_indices = SUBSET_CLASSES[subset_id]  # 如 [0, 1, ..., 19]

        # 构建 "原始标签 → 新标签" 的映射字典
        # 例如子集1: {20: 0, 21: 1, ..., 39: 19}
        self.label_map = {orig: new for new, orig in enumerate(self.class_indices)}

        # 筛选: 只保留属于当前子集的样本
        # self.data: numpy 数组, shape=(50000, 32, 32, 3) 或 (10000, ...)
        # self.targets: 列表, 长度=50000 或 10000
        keep_mask = [t in self.label_map for t in self.targets]
        keep_indices = [i for i, keep in enumerate(keep_mask) if keep]

        # 更新 data 和 targets: 只保留子集中的样本
        self.data = self.data[keep_indices]
        self.targets = [self.label_map[self.targets[i]] for i in keep_indices]
        # 现在 self.targets 中的值都在 [0, 19] 范围内

    def __getitem__(self, index):
        """
        返回: (图片 Tensor, 重映射后的标签 0-19, 样本在子集中的索引)
        """
        img, target = self.data[index], self.targets[index]

        # numpy 数组 → PIL Image (torchvision transforms 期望 PIL 输入)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_closed_subset_dataloaders(subset_id=0, batch_size=128,
                                            num_workers=8):
    """
    获取指定子集的训练集和测试集 DataLoader

    参数:
        subset_id: int, 子集编号 0-4, 决定使用哪 20 个类
        batch_size: int, 每批次加载的图片数量
        num_workers: int, 数据加载并行工作进程数

    返回:
        train_loader: 训练集 DataLoader (每个 batch 返回 (img, label, index))
        val_loader:   测试集 DataLoader (每个 batch 返回 (img, label, index))
        n_data:       训练集总样本数 (每个子集约 10000 张)
        class_indices: list, 该子集包含的原始类别编号列表
                       如 subset_id=0 → [0, 1, ..., 19]
                       这个列表会传给 ELOT 损失, 用于筛选教师的分类头权重
    """
    data_folder = get_data_folder()

    # 训练集图像预处理 (含数据增强)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # 随机裁剪: 增强位置鲁棒性
        transforms.RandomHorizontalFlip(),         # 随机水平翻转
        transforms.ToTensor(),                     # PIL → Tensor, [0,255] → [0,1]
        transforms.Normalize(                      # 标准化: (pixel - mean) / std
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
    ])

    # 测试集图像预处理 (无数据增强, 保证评估结果可复现)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
    ])

    # 创建训练集 Dataset
    train_set = CIFAR100SubsetInstance(
        root=data_folder,
        subset_id=subset_id,
        download=True,
        train=True,
        transform=train_transform
    )
    n_data = len(train_set)  # 训练集样本数 (约 10000)

    # 创建训练集 DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,           # 每个 epoch 打乱顺序
        num_workers=num_workers
    )

    # 创建测试集 Dataset
    test_set = CIFAR100SubsetInstance(
        root=data_folder,
        subset_id=subset_id,
        download=True,
        train=False,
        transform=test_transform
    )

    # 创建测试集 DataLoader
    val_loader = DataLoader(
        test_set,
        batch_size=int(batch_size / 2),
        shuffle=False,
        num_workers=int(num_workers / 2)
    )

    # 返回该子集的原始类别编号列表 (传给 ELOT 损失用)
    class_indices = SUBSET_CLASSES[subset_id]

    print("="*60)
    print("闭集子集 {} 数据加载完成:".format(subset_id))
    print("  原始类别: {} ~ {}".format(class_indices[0], class_indices[-1]))
    print("  训练集样本数: {}".format(n_data))
    print("  测试集样本数: {}".format(len(test_set)))
    print("  标签范围: 0 ~ {} (已重映射)".format(CLASSES_PER_SUBSET - 1))
    print("="*60)

    return train_loader, val_loader, n_data, class_indices


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    # 测试子集 0 的数据加载
    train_loader, val_loader, n_data, class_indices = \
        get_cifar100_closed_subset_dataloaders(subset_id=0, batch_size=64)

    print("\n--- 验证第一个 batch ---")
    for img, target, index in train_loader:
        print("图片 shape:", img.shape)          # 应该是 (64, 3, 32, 32)
        print("标签范围:", target.min().item(), "~", target.max().item())  # 应该是 0~19
        print("索引范围:", index.min().item(), "~", index.max().item())
        break

    # 测试所有 5 个子集
    print("\n--- 验证所有 5 个子集 ---")
    for sid in range(5):
        _, _, n, cls = get_cifar100_closed_subset_dataloaders(subset_id=sid, batch_size=64)
        print("子集 {}: {} 样本, 类别 {}".format(sid, n, cls[:3]))