# ============================================================
# resnet.py - 专为 CIFAR 数据集设计的 ResNet 网络结构
# 包含两种基本模块：BasicBlock 和 Bottleneck
# 以及主网络 ResNet，支持多种深度配置
# 参考来源：
#   https://github.com/facebook/fb.resnet.torch
#   https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# ============================================================

from __future__ import absolute_import  # 兼容Python2的绝对导入语法，Python3中无实际作用

# ResNet论文：Deep Residual Learning for Image Recognition
# 核心思想：在普通卷积网络中加入"跳跃连接"（shortcut），让梯度可以直接跳过某些层传回去
# 解决了深层网络训练时梯度消失的问题

import torch.nn as nn          # PyTorch神经网络模块，提供Conv2d/BatchNorm2d/Linear等层
import torch.nn.functional as F  # PyTorch函数式接口，提供relu等无参数的操作
import math                    # Python数学库（本文件中实际未使用，可能是历史遗留）

# __all__ 控制 from resnet import * 时导出哪些名字
# 这里只导出 'resnet' 字符串，实际意义不大，主要是一种代码规范声明
__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    """
    创建一个 3×3 卷积层（带padding，不带bias）
    in_planes:  输入特征图的通道数
    out_planes: 输出特征图的通道数
    stride:     卷积步长，默认1（步长=2时特征图尺寸减半）
    """
    return nn.Conv2d(
        in_planes,      # 输入通道数
        out_planes,     # 输出通道数
        kernel_size=3,  # 3×3的卷积核
        stride=stride,  # 步长（控制输出特征图大小）
        padding=1,      # 四周填充1个像素，保证stride=1时输出尺寸不变（32→32）
        bias=False      # 不用bias，因为后面紧跟BatchNorm，BN自带偏置效果
    )


# ============================================================
# BasicBlock：ResNet的基本残差块（用于较浅的网络：ResNet20/32/56等）
# 结构：Conv→BN→ReLU→Conv→BN→(+残差)→ReLU
# expansion=1 表示输出通道数和输入通道数相同
# ============================================================
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道相对于输入通道的倍数，BasicBlock不扩张所以是1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        """
        inplanes:   输入特征图的通道数
        planes:     该Block内部（和输出）的通道数
        stride:     第一个卷积的步长（=2时该Block会让特征图尺寸减半）
        downsample: 下采样层（当输入输出维度不一致时，对残差做变换使其匹配）
        is_last:    是否是该stage的最后一个Block（用于控制是否返回preact特征）
        """
        super(BasicBlock, self).__init__()  # 调用父类nn.Module的初始化

        self.is_last = is_last  # 记录是否是最后一个block，用于特征提取

        # 第一个卷积层：可能改变通道数和特征图尺寸（当stride=2时）
        self.conv1 = conv3x3(inplanes, planes, stride)

        # BatchNorm层：对conv1的输出做批归一化
        # 作用：稳定训练，加速收敛，有一定正则化效果
        self.bn1 = nn.BatchNorm2d(planes)

        # ReLU激活函数：inplace=False表示不在原地修改，创建新的tensor
        # inplace=False是因为需要保留原值用于后续梯度计算
        self.relu = nn.ReLU(inplace=False)

        # 第二个卷积层：通道数不变，stride=1，特征图尺寸不变
        self.conv2 = conv3x3(planes, planes)

        # 第二个BatchNorm层
        self.bn2 = nn.BatchNorm2d(planes)

        # 下采样层：当输入输出的通道数或尺寸不一致时，需要对残差做变换
        # 比如：输入是[64通道,32×32]，输出是[128通道,16×16]，残差必须也变成[128,16×16]才能相加
        self.downsample = downsample

        self.stride = stride  # 保存stride，备用

    def forward(self, x):
        """前向传播：实现残差连接"""

        # 这个block可能收到两种形式的输入：
        # 1. 直接的tensor（第一个block）
        # 2. (tensor, features列表) 的元组（后续block，用于收集中间特征）
        if isinstance(x, tuple):
            x, features = x   # 解包：x是特征图，features是之前收集的特征列表
        else:
            features = []     # 第一个block：初始化空的特征列表

        x = self.relu(x)      # 先对输入做ReLU（这是pre-activation ResNet的设计）
        residual = x          # 保存输入作为残差（shortcut路径）

        # 主路径（main path）：
        out = self.conv1(x)   # 第一个3×3卷积
        out = self.bn1(out)   # BatchNorm
        out = self.relu(out)  # ReLU激活

        out = self.conv2(out) # 第二个3×3卷积
        out = self.bn2(out)   # BatchNorm（注意这里还没有ReLU）

        # 如果需要下采样（输入输出维度不一致），对残差做变换
        if self.downsample is not None:
            residual = self.downsample(x)  # 用1×1卷积调整残差的通道数和尺寸

        out += residual  # 残差连接：主路径输出 + shortcut路径
                         # 这是ResNet的核心！让梯度可以直接通过残差连接传回去

        preact = out     # 保存ReLU前的输出（preactivation特征，某些方法需要用到）

        out = F.relu(out)  # 最终的ReLUxq激活

        # 返回：当前输出 + 更新后的特征列表（把当前block的输出加入列表）
        return out, features + [out]


# ============================================================
# Bottleneck：瓶颈残差块（用于较深的网络：ResNet50/101/152等）
# 结构：1×1Conv→BN→ReLU→3×3Conv→BN→ReLU→1×1Conv→BN→(+残差)→ReLU
# expansion=4 表示输出通道数是中间通道数的4倍
# 优点：用1×1卷积先降维再升维，减少3×3卷积的计算量
# ============================================================
class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数 = planes × 4，比如planes=64时输出256通道

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        """
        inplanes: 输入通道数
        planes:   中间层通道数（输出通道数 = planes × 4）
        """
        super(Bottleneck, self).__init__()  # 调用父类初始化

        self.is_last = is_last  # 是否是最后一个block

        # 第一个1×1卷积：降维，把inplanes通道压缩到planes通道
        # kernel_size=1：只做通道维度的线性变换，不改变空间尺寸
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # BN层

        # 第二个3×3卷积：在低维度上做空间特征提取，stride可能使尺寸减半
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3,
            stride=stride,  # 控制是否下采样
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)  # BN层

        # 第三个1×1卷积：升维，把planes通道扩张到planes×4通道
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)  # BN层

        self.relu = nn.ReLU(inplace=False)  # ReLU激活（inplace=False同上）
        self.downsample = downsample        # 下采样层（维度不匹配时用）
        self.stride = stride                # 保存步长

    def forward(self, x):
        """Bottleneck的前向传播"""
        residual = x  # 保存输入作为残差

        # 1×1卷积 → BN → ReLU（降维）
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3×3卷积 → BN → ReLU（特征提取）
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1×1卷积 → BN（升维，注意这里还没有ReLU）
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果输入输出维度不一致，对残差做变换
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # 残差连接
        preact = out     # 保存激活前的特征

        out = F.relu(out)  # 最终ReLU

        # Bottleneck 根据 is_last 决定返回格式
        if self.is_last:
            return out, preact  # 最后一个block返回激活后和激活前的特征
        else:
            return out          # 其他block只返回激活后的特征


# ============================================================
# ResNet 主网络类
# 由一个初始卷积层 + 三个stage（每个stage包含多个BasicBlock/Bottleneck）+ 全连接层组成
# ============================================================
class ResNet(nn.Module):

    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10):
        """
        depth:       网络总深度，如20、56、110
        num_filters: 每个stage的通道数列表，如[16, 16, 32, 64]
                     [初始卷积通道, stage1通道, stage2通道, stage3通道]
        block_name:  使用哪种块，'basicblock' 或 'bottleneck'
        num_classes: 分类数量，CIFAR-100是100，这里是21（20已知类+1未知类）
        """
        super(ResNet, self).__init__()  # 调用父类nn.Module初始化

        # 根据 block_name 确定每个stage有多少个block
        if block_name.lower() == 'basicblock':
            # BasicBlock的深度公式：depth = 6n + 2
            # 比如depth=20：n=(20-2)//6=3，每个stage有3个block
            assert (depth - 2) % 6 == 0, \
                'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6  # 每个stage包含的block数量
            block = BasicBlock    # 使用BasicBlock类
        elif block_name.lower() == 'bottleneck':
            # Bottleneck的深度公式：depth = 9n + 2
            assert (depth - 2) % 9 == 0, \
                'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9  # 每个stage包含的block数量
            block = Bottleneck    # 使用Bottleneck类
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = num_filters[0]  # 当前的输入通道数，初始=16，随着网络加深会更新

        # 第一个卷积层：3通道RGB图像 → num_filters[0]通道特征图
        # kernel_size=3, padding=1 保证32×32的输入输出尺寸不变
        self.conv1 = nn.Conv2d(
            3,               # 输入通道（RGB三通道）
            num_filters[0],  # 输出通道（16）
            kernel_size=3,
            padding=1,
            bias=False       # 后面有BN，不需要bias
        )
        self.bn1 = nn.BatchNorm2d(num_filters[0])  # 第一个BN层
        self.relu = nn.ReLU(inplace=False)          # ReLU激活

        # 三个stage，每个stage包含n个BasicBlock
        # stage1：特征图32×32，通道数=num_filters[1]=16，stride=1（尺寸不变）
        self.layer1 = self._make_layer(block, num_filters[1], n)

        # stage2：特征图16×16，通道数=num_filters[2]=32，stride=2（尺寸减半）
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)

        # stage3：特征图8×8，通道数=num_filters[3]=64，stride=2（尺寸减半）
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)

        # 全局平均池化：把8×8的特征图压缩成1×1
        # 输入[batch,64,8,8] → 输出[batch,64,1,1]
        self.avgpool = nn.AvgPool2d(8)

            # 全连接分类层：把64维特征映射到num_classes个类别分数
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

            # 权重初始化：好的初始化让训练更稳定
        for m in self.modules():  # 遍历网络中所有的子模块
            if isinstance(m, nn.Conv2d):
                # Kaiming初始化：专为ReLU激活函数设计的初始化方法
                # 防止信号在深层网络中消失或爆炸
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # BN的weight（gamma）初始化为1：初始时不缩放
                nn.init.constant_(m.weight, 1)
                # BN的bias（beta）初始化为0：初始时不偏移
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建一个stage，包含多个残差块
        block:   使用BasicBlock还是Bottleneck
        planes:  该stage的通道数
        blocks:  该stage包含多少个block（即n）
        stride:  第一个block的步长（=2时该stage对特征图下采样）
        """
        downsample = None  # 默认不需要下采样

        # 如果stride!=1（需要下采样）或通道数发生变化，需要创建下采样层
        # 下采样层用1×1卷积+BN对残差做变换，使其维度与主路径输出一致
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,           # 输入通道数
                    planes * block.expansion, # 输出通道数（与主路径输出一致）
                    kernel_size=1,            # 1×1卷积，只做通道变换
                    stride=stride,            # 与主路径stride一致，同步下采样
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),  # BN层
            )

        layers = list([])  # 用列表存储该stage的所有block

        # 第一个block：可能需要下采样（stride=2）和通道变换（downsample）
        layers.append(
            block(
                self.inplanes,          # 输入通道数
                planes,                 # 输出通道数
                stride,                 # 步长
                downsample,             # 下采样层（可能为None）
                is_last=(blocks == 1)   # 如果只有1个block，它就是最后一个
            )
        )

        # 更新当前输入通道数（后续block的输入通道数 = 第一个block的输出通道数）
        self.inplanes = planes * block.expansion

        # 后续的block：stride=1，通道数不变，不需要downsample
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    is_last=(i == blocks - 1)  # 最后一个block标记为is_last=True
                )
            )

        # nn.Sequential把列表中的层串联起来，forward时依次执行
        return nn.Sequential(*layers)  # *layers 把列表展开为位置参数

    def get_feat_modules(self):
        """
        返回用于特征提取的模块列表（某些蒸馏方法需要对特定层注册hook）
        """
        feat_m = nn.ModuleList([])  # ModuleList：可以存储多个nn.Module的容器
        feat_m.append(self.conv1)   # 第一个卷积层
        feat_m.append(self.bn1)     # 第一个BN层
        feat_m.append(self.relu)    # ReLU
        feat_m.append(self.layer1)  # stage1
        feat_m.append(self.layer2)  # stage2
        feat_m.append(self.layer3)  # stage3
        return feat_m

    def get_bn_before_relu(self):
        """
        返回每个stage最后一个block中ReLU之前的BN层
        用于某些需要访问激活前特征的蒸馏方法（如abound）
        """
        if isinstance(self.layer1[0], Bottleneck):
            # Bottleneck结构：取最后一个block的第三个BN（bn3）
            bn1 = self.layer1[-1].bn3  # stage1最后一个block的bn3
            bn2 = self.layer2[-1].bn3  # stage2最后一个block的bn3
            bn3 = self.layer3[-1].bn3  # stage3最后一个block的bn3
        elif isinstance(self.layer1[0], BasicBlock):
            # BasicBlock结构：取最后一个block的第二个BN（bn2）
            bn1 = self.layer1[-1].bn2  # stage1最后一个block的bn2
            bn2 = self.layer2[-1].bn2  # stage2最后一个block的bn2
            bn3 = self.layer3[-1].bn2  # stage3最后一个block的bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]  # 返回三个BN层的列表

    def forward(self, x, is_feat=False, preact=False):
        """
        前向传播
        x:       输入图片 Tensor，shape=[batch, 3, 32, 32]
        is_feat: 是否同时返回中间层特征（蒸馏时需要）
        preact:  是否返回激活前的特征（某些蒸馏方法需要）
        """
        x = self.conv1(x)   # 第一个卷积：[batch,3,32,32] → [batch,16,32,32]
        x = self.bn1(x)     # BatchNorm
        # x = self.relu(x)  # 注释掉了：这里不做ReLU，留给BasicBlock内部做（pre-activation设计）
        f0 = x              # 保存第0层特征（conv1+bn1的输出，未激活）

        x, f1 = self.layer1(x)  # stage1：[batch,16,32,32] → [batch,16,32,32]
                                  # f1是stage1中每个block输出的特征列表
        f1_act = [self.relu(f) for f in f1]  # 对f1中每个特征做ReLU激活
                                              # 列表推导式：对列表中每个元素应用relu

        x, f2 = self.layer2(x)  # stage2：[batch,16,32,32] → [batch,32,16,16]（尺寸减半）
        f2_act = [self.relu(f) for f in f2]  # 对f2做ReLU

        x, f3 = self.layer3(x)  # stage3：[batch,32,16,16] → [batch,64,8,8]（尺寸再减半）
        f3_act = [self.relu(f) for f in f3]  # 对f3做ReLU

        x = self.avgpool(self.relu(x))  # 先ReLU再全局平均池化：[batch,64,8,8] → [batch,64,1,1]
        x = x.view(x.size(0), -1)       # 展平：[batch,64,1,1] → [batch,64]
                                         # x.size(0)=batch_size，-1表示自动计算剩余维度

        f4 = x   # 保存展平后的特征向量（全连接层之前的特征）

        x = self.fc(x)  # 全连接分类层：[batch,64] → [batch,num_classes]
                         # 输出每个类别的分数（logits）

        if is_feat:
            # 蒸馏模式：返回所有中间层特征 + 最终分类输出
            # [self.relu(f0)] 是第0层特征（加上relu）
            # f1_act + f2_act + f3_act 是三个stage的所有block的特征
            # [f4] 是最终的全局特征向量
            return [self.relu(f0)] + f1_act + f2_act + f3_act + [f4], x
        else:
            # 普通模式：只返回分类输出
            return x


# ============================================================
# 各种深度的 ResNet 工厂函数
# 通过调用这些函数创建对应深度的网络实例
# **kwargs 允许传入任意额外参数（如 num_classes）
# ============================================================

def resnet8(**kwargs):
    """ResNet-8：3个stage，每个stage 1个block，最浅的版本"""
    return ResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)
    # [16,16,32,64]：初始16通道，stage1=16，stage2=32，stage3=64

def resnet14(**kwargs):
    """ResNet-14：3个stage，每个stage 2个block"""
    return ResNet(14, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet20(**kwargs):
    """ResNet-20：3个stage，每个stage 3个block，CIFAR常用baseline"""
    return ResNet(20, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet32(**kwargs):
    """ResNet-32：3个stage，每个stage 5个block"""
    return ResNet(32, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet44(**kwargs):
    """ResNet-44：3个stage，每个stage 7个block"""
    return ResNet(44, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet56(**kwargs):
    """ResNet-56：3个stage，每个stage 9个block"""
    return ResNet(56, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet110(**kwargs):
    """ResNet-110：3个stage，每个stage 18个block，较深的版本"""
    return ResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet8x4(**kwargs):
    """ResNet-8 宽版本：通道数扩大4倍 [32,64,128,256]，参数量更多"""
    return ResNet(8, [32, 64, 128, 256], 'basicblock', **kwargs)

def resnet32x4(**kwargs):
    """ResNet-32 宽版本：通道数扩大4倍 [32,64,128,256]，常用作教师模型"""
    return ResNet(32, [32, 64, 128, 256], 'basicblock', **kwargs)


# ============================================================
# 测试代码：只有直接运行这个文件时才会执行（不是被import时）
# 用于验证网络结构是否正确
# ============================================================
if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)  # 创建假数据：2张3通道32×32的随机图片
    net = resnet20(num_classes=100) # 创建resnet20，100个类别
    feats, logit = net(x, is_feat=True, preact=True)  # 前向传播，获取所有特征

    import pdb
    pdb.set_trace()  # 断点调试：程序暂停，可以在命令行查看变量（调试用）

    for f in feats:
        print(f.shape, f.min().item())  # 打印每层特征的shape和最小值，检查是否正常
    # print(logit.shape)  # 注释掉了，可以取消注释查看输出shape

    # 验证BN层是否正确
    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')    # 正确：是BatchNorm2d层
        else:
            print('warning') # 警告：不是期望的BatchNorm2d层