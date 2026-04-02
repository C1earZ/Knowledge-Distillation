# ============================================================
# models/util.py - 知识蒸馏辅助模块集合
#
# 包含各种蒸馏方法所需的适配模块:
#   - Paraphraser / Translator: Factor Transfer 方法
#   - Connector / ConnectorV2: AB / Overhaul 方法
#   - ConvReg: FitNet 方法的核心回归模块
#   - Regress: 简单线性回归
#   - Embed / LinearEmbed / MLPEmbed: 特征嵌入模块 (CRD/CC 等)
#   - Normalize: L2 归一化层
#   - Flatten: 展平层
#   - PoolEmbed: 池化+嵌入
# ============================================================

from __future__ import print_function
# 兼容 Python2 的 print 语法, Python3 中无实际作用

import torch.nn as nn
# PyTorch 神经网络模块, 提供 Conv2d, BatchNorm2d, Linear,
# Sequential, ModuleList 等所有层和容器

import math
# Python 标准数学库, 这里用 math.sqrt() 做 Kaiming 初始化的计算

import numpy as np
# NumPy 数值计算库, 这里用 np.arange() 生成层索引数组


# ============================================================
# LAYER 字典: 每个模型架构对应的"可蒸馏中间层"索引列表
#
# 用途: 某些蒸馏方法 (如 KDSVD, AT) 需要知道模型有哪些中间层
#       可以用来做特征匹配, 这个字典提供了预定义的层索引
#
# 计算规则:
#   resnet 系列: depth=D, 每个 stage n=(D-2)//2 个 block
#     层索引从 1 到 3n (3个stage, 每个n个block)
#   wrn 系列: depth=D, 每个 stage n=(D-4)//2 个 block
#     层索引从 1 到 3n
# ============================================================
LAYER = {
    'resnet20': np.arange(1, (20 - 2) // 2 + 1),
    # resnet20: n=(20-2)//2=9, 生成 [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 共 9 个中间层 (3个stage × 3个block)

    'resnet56': np.arange(1, (56 - 2) // 2 + 1),
    # resnet56: n=(56-2)//2=27, 生成 [1, 2, ..., 27]
    # 共 27 个中间层 (3个stage × 9个block)

    'resnet110': np.arange(2, (110 - 2) // 2 + 1, 2),
    # resnet110: (110-2)//2=54, 但 step=2, 所以只取偶数索引
    # 生成 [2, 4, 6, ..., 54], 共 27 个 (每隔一个block取一个)
    # 原因: resnet110 太深, 取全部 54 层计算量太大, 隔一个取一个

    'wrn40x2': np.arange(1, (40 - 4) // 2 + 1),
    # wrn_40_2: n=(40-4)//2=18, 生成 [1, 2, ..., 18]

    'wrn28x2': np.arange(1, (28 - 4) // 2 + 1),
    # wrn_28_2: n=(28-4)//2=12, 生成 [1, 2, ..., 12]

    'wrn16x2': np.arange(1, (16 - 4) // 2 + 1),
    # wrn_16_2: n=(16-4)//2=6, 生成 [1, 2, 3, 4, 5, 6]

    'resnet34': np.arange(1, (34 - 2) // 2 + 1),
    # resnet34: n=(34-2)//2=16, 生成 [1, 2, ..., 16]

    'resnet18': np.arange(1, (18 - 2) // 2 + 1),
    # resnet18: n=(18-2)//2=8, 生成 [1, 2, ..., 8]

    'resnet34im': np.arange(1, (34 - 2) // 2 + 1),
    # resnet34 ImageNet 版本, 层数相同

    'resnet18im': np.arange(1, (18 - 2) // 2 + 1),
    # resnet18 ImageNet 版本, 层数相同

    'resnet32x4': np.arange(1, (32 - 2) // 2 + 1),
    # resnet32x4: n=(32-2)//2=15, 生成 [1, 2, ..., 15]
    # x4 表示通道数扩大4倍, 但 block 数量和 resnet32 一样
}


# ============================================================
# Paraphraser — Factor Transfer 方法的教师端模块
#
# 论文: "Paraphrasing Complex Network: Network Compression
#        via Factor Transfer" (NeurIPS 2018)
#
# 结构: 编码器(encoder) + 解码器(decoder)
#   编码器: 把教师特征压缩成低维 "factor"
#   解码器: 从 factor 重建教师特征 (用于预训练)
#
# 预训练时: 用重建损失训练 Paraphraser
#   loss = MSE(decoder(encoder(feat_t)), feat_t)
# 蒸馏时: 只用编码器提取 factor, 让学生的 factor 接近教师的
# ============================================================
class Paraphraser(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer"""

    def __init__(self, t_shape, k=0.5, use_bn=False):
        """
        参数:
            t_shape: tuple, 教师特征的形状 (N, C, H, W)
                N = batch_size (不使用)
                C = 通道数 (用于确定编码器输入/输出维度)
                H, W = 空间尺寸 (编码器不改变空间尺寸)
            k: float, 压缩比例, 默认 0.5
                factor 的通道数 = C × k
                比如教师特征 256 通道 → factor 128 通道
            use_bn: bool, 是否在卷积后加 BatchNorm
                默认 False, 即不加 BN
        """
        super(Paraphraser, self).__init__()
        # 调用 nn.Module 的初始化

        in_channel = t_shape[1]
        # 输入通道数 = 教师特征的通道数 C
        # 比如 t_shape = (64, 256, 8, 8) → in_channel = 256

        out_channel = int(t_shape[1] * k)
        # 输出通道数 = 输入通道数 × 压缩比
        # 比如 256 × 0.5 = 128

        # ---- 编码器: 3 层 3×3 卷积, 通道数从 C → C → C*k → C*k ----
        self.encoder = nn.Sequential(
            # nn.Sequential: 把多个层串联, forward 时依次执行

            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            # 第一层: 3×3 卷积, 输入 C 通道, 输出 C 通道
            # stride=1, padding=1 → 空间尺寸不变
            # 作用: 在原始通道空间做特征变换

            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            # 条件 BN: 如果 use_bn=True 就加 BN, 否则加空的 Sequential (什么都不做)
            # 这是一种常见的条件模块写法

            nn.LeakyReLU(0.1, inplace=True),
            # LeakyReLU 激活: 负值不完全置零, 而是乘以 0.1
            # 比 ReLU 好处: 避免"神经元死亡" (输出全零就再也学不动了)
            # inplace=True: 直接修改输入 tensor, 节省内存

            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            # 第二层: 3×3 卷积, 通道数从 C 降到 C*k (256→128)
            # 这是实际做压缩的关键层

            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            # 条件 BN

            nn.LeakyReLU(0.1, inplace=True),
            # LeakyReLU 激活

            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            # 第三层: 3×3 卷积, 通道数保持 C*k
            # 在压缩后的空间做进一步特征提炼

            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            # 条件 BN

            nn.LeakyReLU(0.1, inplace=True),
            # LeakyReLU 激活
        )

        # ---- 解码器: 3 层 3×3 转置卷积, 通道数从 C*k → C*k → C → C ----
        # 转置卷积 (ConvTranspose2d) 可以理解为卷积的"逆操作"
        # 这里 kernel=3, stride=1, padding=1, 所以空间尺寸不变
        # 只做通道维度的"升维还原"
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            # 第一层转置卷积: C*k → C*k, 空间不变

            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            # 第二层转置卷积: C*k → C, 通道数恢复 (128→256)
            # 这是实际做"解压缩"的关键层

            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            # 第三层转置卷积: C → C, 在原始维度做进一步调整

            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s, is_factor=False):
        """
        前向传播

        参数:
            f_s: 输入特征, 形状 (B, C, H, W)
                虽然参数名是 f_s, 但预训练时传入的是教师特征 f_t
            is_factor: bool
                True  → 只返回编码后的 factor (蒸馏阶段用)
                False → 返回 (factor, 重建结果) (预训练阶段用)
        """
        factor = self.encoder(f_s)
        # 编码: (B, C, H, W) → (B, C*k, H, W)
        # 比如 (64, 256, 8, 8) → (64, 128, 8, 8)

        if is_factor:
            return factor
            # 蒸馏阶段: 只需要 factor, 不需要重建

        rec = self.decoder(factor)
        # 解码/重建: (B, C*k, H, W) → (B, C, H, W)
        # 比如 (64, 128, 8, 8) → (64, 256, 8, 8)

        return factor, rec
        # 预训练阶段: 返回 factor 和重建结果
        # 预训练损失 = MSE(rec, f_s), 让重建尽量接近原始特征


# ============================================================
# Translator — Factor Transfer 方法的学生端模块
#
# 作用: 把学生特征编码成 factor, 维度与 Paraphraser 的 factor 一致
# 结构: 只有编码器 (不需要解码器, 因为学生不需要重建)
#
# 蒸馏时:
#   teacher_factor = Paraphraser.encoder(feat_t)
#   student_factor = Translator.encoder(feat_s)
#   loss = MSE(student_factor, teacher_factor)
# ============================================================
class Translator(nn.Module):

    def __init__(self, s_shape, t_shape, k=0.5, use_bn=True):
        """
        参数:
            s_shape: tuple, 学生特征的形状 (N, C_s, H, W)
                C_s = 学生的通道数 (决定编码器输入维度)
            t_shape: tuple, 教师特征的形状 (N, C_t, H, W)
                C_t × k = factor 的通道数 (决定编码器输出维度)
                这样学生的 factor 和教师的 factor 维度一致
            k: float, 压缩比 (与 Paraphraser 的 k 必须相同)
            use_bn: bool, 是否使用 BatchNorm, 默认 True
                注意: 和 Paraphraser 不同, Translator 默认开 BN
        """
        super(Translator, self).__init__()

        in_channel = s_shape[1]
        # 输入通道数 = 学生特征的通道数 C_s
        # 比如学生是 resnet20, 倒数第二层特征 (64, 8, 8) → in_channel = 64

        out_channel = int(t_shape[1] * k)
        # 输出通道数 = 教师通道数 × 压缩比
        # 与 Paraphraser 的 out_channel 一致
        # 比如教师 256 × 0.5 = 128

        # 编码器: 3 层 3×3 卷积
        # 通道变化: C_s → C_s → C_t*k → C_t*k
        self.encoder = nn.Sequential(

            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            # 第一层: 在学生通道空间做特征变换, 通道数不变

            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            # 条件 BN (Translator 默认开启)

            nn.LeakyReLU(0.1, inplace=True),
            # LeakyReLU, 斜率 0.1

            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            # 第二层: 学生通道 → 教师 factor 通道 (64→128)
            # 这是跨维度映射的关键层

            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            # 第三层: 在 factor 空间做进一步调整

            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s):
        """
        前向传播: 把学生特征编码成 factor

        输入: f_s, 形状 (B, C_s, H, W), 学生中间层特征
        输出: factor, 形状 (B, C_t*k, H, W), 与教师 factor 同维度
        """
        return self.encoder(f_s)


# ============================================================
# Connector — AB (Activation Boundary) 蒸馏方法的适配模块
#
# 论文: "Knowledge Transfer via Distillation of Activation
#        Boundaries Formed by Hidden Neurons" (AAAI 2019)
#
# 作用: 当学生和教师的中间层特征维度不同时,
#       用 ConvReg 做维度适配; 维度相同时跳过 (恒等映射)
#
# 它管理多个层的适配, 每个层一个独立的 ConvReg 或 Identity
# ============================================================
class Connector(nn.Module):
    """Connect for Knowledge Transfer via Distillation of
    Activation Boundaries Formed by Hidden Neurons"""

    def __init__(self, s_shapes, t_shapes):
        """
        参数:
            s_shapes: list of tuple, 学生各层特征的形状列表
                如 [(2,16,32,32), (2,32,16,16), (2,64,8,8)]
            t_shapes: list of tuple, 教师对应各层特征的形状列表
                如 [(2,32,32,32), (2,64,16,16), (2,128,8,8)]
                长度必须与 s_shapes 一致
        """
        super(Connector, self).__init__()

        self.s_shapes = s_shapes
        # 保存学生各层的 shape, 备用

        self.t_shapes = t_shapes
        # 保存教师各层的 shape, 备用

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))
        # nn.ModuleList: 像 Python list 一样存储多个 nn.Module
        # 但会被 PyTorch 正确注册为子模块 (参数会被优化器发现)
        # 调用 _make_conenctors 静态方法创建每一层的适配器

    @staticmethod
    def _make_conenctors(s_shapes, t_shapes):
        """
        为每一对 (学生层, 教师层) 创建适配器

        逻辑:
            如果通道数和空间尺寸都相同 → 不需要适配, 用空 Sequential
            否则 → 用 ConvReg 做维度变换
        """
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        # 检查: 学生和教师的对齐层数必须相同

        connectors = []
        # 用普通 list 收集, 最后由调用者包装成 ModuleList

        for s, t in zip(s_shapes, t_shapes):
            # 遍历每一对, s 和 t 都是 (N, C, H, W)

            if s[1] == t[1] and s[2] == t[2]:
                # s[1] = 学生通道数, t[1] = 教师通道数
                # s[2] = 学生空间高度, t[2] = 教师空间高度
                # 如果通道数和空间尺寸都一样, 不需要任何变换
                connectors.append(nn.Sequential())
                # 空的 Sequential: forward(x) 直接返回 x, 等价于恒等映射

            else:
                # 维度不匹配, 需要用 ConvReg 做适配
                connectors.append(ConvReg(s, t, use_relu=False))
                # use_relu=False: AB 方法需要的是激活前的特征
                # (判断激活边界需要看正负值, 不能过 ReLU)

        return connectors

    def forward(self, g_s):
        """
        前向传播: 对学生的每一层特征做适配

        参数:
            g_s: list of Tensor, 学生各层的特征图列表

        返回:
            out: list of Tensor, 适配后的特征图列表
                 通道数和空间尺寸与教师对应层一致
        """
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))
            # 第 i 层学生特征 → 第 i 个 connector → 适配后的特征

        return out


# ============================================================
# ConnectorV2 — Feature Distillation Overhaul 的适配模块
#
# 论文: "A Comprehensive Overhaul of Feature Distillation"
#       (ICCV 2019)
#
# 与 Connector 的区别:
#   Connector: 用 ConvReg (3×3 卷积, 可能改变空间尺寸)
#   ConnectorV2: 只用 1×1 卷积 + BN (只改通道数, 不改空间尺寸)
#   ConnectorV2 更轻量, 且带 Kaiming 初始化
# ============================================================
class ConnectorV2(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation (ICCV 2019)"""

    def __init__(self, s_shapes, t_shapes):
        """
        参数:
            s_shapes: list of tuple, 学生各层特征的形状
            t_shapes: list of tuple, 教师各层特征的形状
        """
        super(ConnectorV2, self).__init__()

        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))
        # 调用实例方法 (不是 staticmethod) 创建适配器列表

    def _make_conenctors(self, s_shapes, t_shapes):
        """为每一层创建 1×1 卷积适配器"""

        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        # 检查长度一致

        t_channels = [t[1] for t in t_shapes]
        # 提取教师每层的通道数, 如 [32, 64, 128]

        s_channels = [s[1] for s in s_shapes]
        # 提取学生每层的通道数, 如 [16, 32, 64]

        connectors = nn.ModuleList([
            self._build_feature_connector(t, s)
            for t, s in zip(t_channels, s_channels)
        ])
        # 列表推导: 为每一对 (教师通道数, 学生通道数) 创建适配器
        # 注意参数顺序: (t_channel, s_channel)

        return connectors

    @staticmethod
    def _build_feature_connector(t_channel, s_channel):
        """
        构建单层的适配器: 1×1 卷积 + BN

        参数:
            t_channel: int, 教师的通道数 (输出维度)
            s_channel: int, 学生的通道数 (输入维度)

        返回:
            nn.Sequential, 包含 [Conv2d(1×1), BatchNorm2d]
        """
        C = [
            nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1,
                      padding=0, bias=False),
            # 1×1 卷积: 只做通道维度的线性变换
            # 把学生的 s_channel 通道映射到教师的 t_channel 通道
            # kernel_size=1: 不看邻域, 每个空间位置独立变换
            # bias=False: 后面有 BN, 不需要 bias

            nn.BatchNorm2d(t_channel)
            # BN: 对变换后的特征做归一化
        ]

        # ---- 手动 Kaiming 初始化 ----
        # 虽然 PyTorch 的 Conv2d 默认也是 Kaiming 初始化,
        # 但这里显式地做一遍, 确保初始化方式符合预期
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # n = 1 × 1 × t_channel = t_channel (对于 1×1 卷积)
                # Kaiming 公式中的 fan_out

                m.weight.data.normal_(0, math.sqrt(2. / n))
                # 用正态分布初始化权重
                # 标准差 = sqrt(2/fan_out), 这是 Kaiming He 初始化
                # 适合 ReLU 激活函数, 保持信号方差稳定

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # BN 的 gamma 初始化为 1 (初始不缩放)

                m.bias.data.zero_()
                # BN 的 beta 初始化为 0 (初始不偏移)

        return nn.Sequential(*C)
        # 把列表展开为 Sequential 的位置参数
        # forward 时: x → Conv2d(1×1) → BN → 输出


    def forward(self, g_s):
        """
        前向传播: 对学生的每一层特征做 1×1 卷积适配

        参数:
            g_s: list of Tensor, 学生各层特征

        返回:
            out: list of Tensor, 适配后的特征 (通道数与教师匹配)
        """
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))
        return out


# ============================================================
# ConvReg — FitNet 方法的卷积回归模块 (核心模块)
#
# 论文: "FitNets: Hints for Thin Deep Nets" (ICLR 2015)
#
# 作用: 把学生的中间层特征映射到教师中间层特征的维度
#       既处理通道数差异, 也处理空间尺寸差异
#
# 结构: Conv2d (或 ConvTranspose2d) + BN + ReLU
#
# 维度适配策略 (根据空间尺寸关系自动选择):
#   s_H == 2 × t_H → 学生比教师大一倍
#     用 stride=2 的 3×3 卷积下采样
#   s_H × 2 == t_H → 学生比教师小一倍
#     用 stride=2 的 4×4 转置卷积上采样
#   s_H >= t_H → 学生大于等于教师 (任意比例)
#     用自适应大小的卷积核 (kernel = 1+s_H-t_H)
#   s_H < t_H 且不是2倍关系 → 不支持, 抛异常
# ============================================================
class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        """
        参数:
            s_shape: tuple, 学生特征的形状 (N, C_s, H_s, W_s)
            t_shape: tuple, 教师特征的形状 (N, C_t, H_t, W_t)
            use_relu: bool, 是否在输出时加 ReLU 激活
                True  → Conv + BN + ReLU (FitNet 默认)
                False → Conv + BN (AB 方法需要激活前的值)
        """
        super(ConvReg, self).__init__()

        self.use_relu = use_relu
        # 保存是否使用 ReLU 的标志

        s_N, s_C, s_H, s_W = s_shape
        # 解包学生特征的形状
        # s_N: batch_size (不使用)
        # s_C: 学生通道数, 如 32
        # s_H, s_W: 学生空间尺寸, 如 16×16

        t_N, t_C, t_H, t_W = t_shape
        # 解包教师特征的形状
        # t_C: 教师通道数, 如 512
        # t_H, t_W: 教师空间尺寸, 如 16×16

        # ---- 根据空间尺寸关系选择卷积类型 ----
        if s_H == 2 * t_H:
            # 情况1: 学生空间尺寸是教师的 2 倍
            # 例如: 学生 32×32, 教师 16×16
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
            # 3×3 卷积, stride=2 做 2 倍下采样
            # 输出尺寸: (H_s + 2×padding - kernel) // stride + 1
            #         = (32 + 2 - 3) // 2 + 1 = 16 ✓

        elif s_H * 2 == t_H:
            # 情况2: 教师空间尺寸是学生的 2 倍
            # 例如: 学生 8×8, 教师 16×16
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2,
                                           padding=1)
            # 转置卷积 (反卷积), stride=2 做 2 倍上采样
            # 输出尺寸: (H_s - 1) × stride - 2×padding + kernel
            #         = (8 - 1) × 2 - 2 + 4 = 16 ✓

        elif s_H >= t_H:
            # 情况3: 学生空间 >= 教师空间 (任意比例, 不一定是2倍)
            # 例如: 学生 16×16, 教师 16×16 (相等)
            # 或者: 学生 10×10, 教师 8×8 (略大)
            self.conv = nn.Conv2d(s_C, t_C,
                                  kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
            # 自适应卷积核大小, 无 padding
            # 当 s_H == t_H 时: kernel = (1, 1), 即 1×1 卷积 (只变通道)
            # 当 s_H > t_H 时: kernel > 1, 通过"恰好覆盖多余像素"来缩小
            # 输出尺寸 = s_H - kernel + 1 = s_H - (1+s_H-t_H) + 1 = t_H ✓

        else:
            # 情况4: 学生比教师小, 且不是 2 倍关系 → 不支持
            raise NotImplemented('student size {}, teacher size {}'.format(
                s_H, t_H))

        self.bn = nn.BatchNorm2d(t_C)
        # BN 层: 对卷积输出做归一化
        # 通道数 = t_C (教师的通道数, 即卷积的输出通道数)

        self.relu = nn.ReLU(inplace=True)
        # ReLU 激活: inplace=True 直接修改输入, 节省内存

    def forward(self, x):
        """
        前向传播

        输入: x, 形状 (B, s_C, s_H, s_W), 学生某层的特征
        输出: 形状 (B, t_C, t_H, t_W), 映射后的特征
              通道数和空间尺寸与教师对应层一致
        """
        x = self.conv(x)
        # 卷积变换: 通道数 s_C → t_C, 空间尺寸 s_H → t_H

        if self.use_relu:
            return self.relu(self.bn(x))
            # Conv → BN → ReLU (FitNet 默认路径)
        else:
            return self.bn(x)
            # Conv → BN (AB 方法路径, 保留正负值信息)


# ============================================================
# Regress — 简单线性回归
#
# 作用: 把学生的特征向量 (展平后的1D向量) 线性映射到教师维度
# 比 ConvReg 更简单: 只用全连接层, 不处理空间结构
# 用于特征向量 (而非特征图) 的维度对齐
# ============================================================
class Regress(nn.Module):
    """Simple Linear Regression for hints"""

    def __init__(self, dim_in=1024, dim_out=1024):
        """
        参数:
            dim_in: int, 输入维度 (学生特征向量的长度)
            dim_out: int, 输出维度 (教师特征向量的长度)
        """
        super(Regress, self).__init__()

        self.linear = nn.Linear(dim_in, dim_out)
        # 全连接层: 输入 dim_in 维, 输出 dim_out 维
        # 参数量 = dim_in × dim_out + dim_out (权重 + 偏置)

        self.relu = nn.ReLU(inplace=True)
        # ReLU 激活

    def forward(self, x):
        """
        前向传播

        输入: x, 可能是 4D (B, C, H, W) 或 2D (B, D)
        输出: (B, dim_out), 映射后的特征向量
        """
        x = x.view(x.shape[0], -1)
        # 展平: (B, C, H, W) → (B, C×H×W)
        # 如果已经是 2D 则不变
        # x.shape[0] = batch_size, -1 = 自动计算剩余维度

        x = self.linear(x)
        # 线性变换: (B, dim_in) → (B, dim_out)

        x = self.relu(x)
        # ReLU 激活

        return x


# ============================================================
# Embed — 特征嵌入模块 (用于 CRD 等对比学习蒸馏方法)
#
# 作用: 把高维特征映射到低维嵌入空间, 然后做 L2 归一化
# 归一化后的向量在超球面上, 适合用余弦相似度计算距离
#
# CRD 方法需要: 学生和教师各一个 Embed,
#   映射到同一个低维空间, 然后做对比学习
# ============================================================
class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        """
        参数:
            dim_in: int, 输入特征维度 (模型倒数第二层特征的维度)
            dim_out: int, 嵌入空间维度 (通常 128)
        """
        super(Embed, self).__init__()

        self.linear = nn.Linear(dim_in, dim_out)
        # 线性投影: dim_in → dim_out (如 2048 → 128)

        self.l2norm = Normalize(2)
        # L2 归一化: 让输出向量的 L2 范数 = 1
        # 归一化后向量在 128 维超球面上

    def forward(self, x):
        """
        前向传播

        输入: x, 形状 (B, dim_in) 或 (B, C, H, W)
        输出: (B, dim_out), L2 归一化后的嵌入向量
        """
        x = x.view(x.shape[0], -1)
        # 展平: 如果是 4D 特征图, 先拉成 1D 向量

        x = self.linear(x)
        # 线性投影: (B, dim_in) → (B, dim_out)

        x = self.l2norm(x)
        # L2 归一化: 每个向量除以自身的 L2 范数
        # 输出的每个向量 ||x||_2 = 1

        return x


# ============================================================
# LinearEmbed — 纯线性嵌入 (不带归一化)
#
# 与 Embed 的区别: 没有 L2 归一化
# 用于需要保留向量幅度信息的场景
# ============================================================
class LinearEmbed(nn.Module):
    """Linear Embedding"""

    def __init__(self, dim_in=1024, dim_out=128):
        """
        参数:
            dim_in: int, 输入维度
            dim_out: int, 输出维度
        """
        super(LinearEmbed, self).__init__()

        self.linear = nn.Linear(dim_in, dim_out)
        # 单层线性变换, 无激活, 无归一化

    def forward(self, x):
        """
        输入: x, (B, dim_in) 或 (B, C, H, W)
        输出: (B, dim_out)
        """
        x = x.view(x.shape[0], -1)
        # 展平

        x = self.linear(x)
        # 线性变换

        return x


# ============================================================
# MLPEmbed — 非线性嵌入 (两层 MLP + L2 归一化)
#
# 与 Embed 的区别: 多了一个隐层, 能学到非线性变换
# 结构: Linear(dim_in → 2×dim_out) → ReLU → Linear(2×dim_out → dim_out) → L2Norm
# 隐层维度 = 2×dim_out, 比输出维度大, 类似 bottleneck 的反向设计
# ============================================================
class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""

    def __init__(self, dim_in=1024, dim_out=128):
        """
        参数:
            dim_in: int, 输入维度
            dim_out: int, 最终嵌入维度
        """
        super(MLPEmbed, self).__init__()

        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        # 第一层: dim_in → 2×dim_out
        # 隐层比输出宽, 给网络更多表达空间

        self.relu = nn.ReLU(inplace=True)
        # ReLU 激活: 引入非线性

        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        # 第二层: 2×dim_out → dim_out

        self.l2norm = Normalize(2)
        # L2 归一化

    def forward(self, x):
        """
        输入: x, (B, dim_in) 或 (B, C, H, W)
        输出: (B, dim_out), L2 归一化后的嵌入向量
        """
        x = x.view(x.shape[0], -1)
        # 展平

        x = self.relu(self.linear1(x))
        # 第一层 + ReLU: (B, dim_in) → (B, 2×dim_out)

        x = self.l2norm(self.linear2(x))
        # 第二层 + L2 归一化: (B, 2×dim_out) → (B, dim_out), ||x||=1

        return x


# ============================================================
# Normalize — L2 (或 Lp) 归一化层
#
# 作用: 让每个样本的特征向量的 Lp 范数 = 1
# 公式: out = x / ||x||_p
#
# p=2 时就是标准的 L2 归一化:
#   ||x||_2 = sqrt(x_1² + x_2² + ... + x_d²)
#   out_i = x_i / ||x||_2
#
# 归一化后向量在单位超球面上, 向量间的点积 = 余弦相似度
# ============================================================
class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        """
        参数:
            power: int, 范数的阶数 p
                p=2 → L2 范数 (欧几里得范数)
                p=1 → L1 范数 (曼哈顿范数)
        """
        super(Normalize, self).__init__()
        self.power = power
        # 保存范数阶数

    def forward(self, x):
        """
        前向传播

        输入: x, 形状 (B, D), 每行是一个 D 维特征向量
        输出: (B, D), 每行的 Lp 范数 = 1
        """
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        # 计算 Lp 范数:
        # 1. x.pow(self.power): 每个元素取 p 次方
        #    x = [a, b, c] → [a², b², c²] (p=2)
        # 2. .sum(1, keepdim=True): 沿 dim=1 (特征维度) 求和, 保持维度
        #    [a²+b²+c²], shape=(B, 1)
        # 3. .pow(1./self.power): 取 1/p 次方
        #    (a²+b²+c²)^(1/2) = ||x||_2
        # 最终 norm shape = (B, 1)

        out = x.div(norm)
        # 每个样本的特征向量除以自己的范数
        # 广播: (B, D) / (B, 1) → (B, D)
        # 结果: 每行向量的 Lp 范数 = 1

        return out


# ============================================================
# Flatten — 展平层
#
# 作用: 把 4D 特征图展平成 2D 矩阵
# (B, C, H, W) → (B, C×H×W)
#
# 在 PyTorch 新版本中可以用 nn.Flatten() 代替
# 这里自定义是为了兼容旧版本
# ============================================================
class Flatten(nn.Module):
    """flatten module"""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        """
        输入: feat, 形状 (B, C, H, W)
        输出: 形状 (B, C×H×W)
        """
        return feat.view(feat.size(0), -1)
        # feat.size(0) = batch_size
        # -1 = 自动计算: C × H × W


# ============================================================
# PoolEmbed — 池化 + 嵌入模块
#
# 作用: 先对特征图做自适应池化 (统一空间尺寸),
#       再展平并做线性投影 + L2 归一化
#
# 用于某些需要把不同层 (不同空间尺寸) 的特征
# 统一到同一维度嵌入空间的方法
#
# 注意: pool_size 和 nChannels 是为特定模型 (resnet 系列, CIFAR)
#       硬编码的, 换模型可能需要修改
# ============================================================
class PoolEmbed(nn.Module):
    """pool and embed"""

    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        """
        参数:
            layer: int, 对应模型的第几层 (0-4)
                不同层的通道数和空间尺寸不同,
                这里根据 layer 值硬编码了对应的参数
            dim_out: int, 嵌入空间维度
            pool_type: str, 'avg' 或 'max'
                选择平均池化还是最大池化
        """
        super().__init__()

        # ---- 根据层号确定通道数和池化目标尺寸 ----
        # 这些值对应的是 resnet 系列 (CIFAR 输入 32×32) 的特征维度
        if layer == 0:
            pool_size = 8
            # 池化目标: 8×8
            nChannels = 16
            # 第 0 层通道数: 16 (conv1 输出)
        elif layer == 1:
            pool_size = 8
            nChannels = 16
            # 第 1 层: 16 通道, 池化到 8×8
        elif layer == 2:
            pool_size = 6
            nChannels = 32
            # 第 2 层: 32 通道 (stage2), 池化到 6×6
        elif layer == 3:
            pool_size = 4
            nChannels = 64
            # 第 3 层: 64 通道 (stage3), 池化到 4×4
        elif layer == 4:
            pool_size = 1
            nChannels = 64
            # 第 4 层: 64 通道 (avgpool 后), 池化到 1×1
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        # ---- 构建嵌入流水线 ----
        self.embed = nn.Sequential()
        # 空的 Sequential, 逐步添加子模块

        if layer <= 3:
            # 前 4 层 (0-3) 是特征图, 需要先池化到固定空间尺寸
            if pool_type == 'max':
                self.embed.add_module(
                    'MaxPool',
                    nn.AdaptiveMaxPool2d((pool_size, pool_size))
                )
                # 自适应最大池化: 无论输入多大, 输出固定为 pool_size × pool_size
                # 'MaxPool' 是该子模块在 Sequential 中的名字

            elif pool_type == 'avg':
                self.embed.add_module(
                    'AvgPool',
                    nn.AdaptiveAvgPool2d((pool_size, pool_size))
                )
                # 自适应平均池化: 输出固定为 pool_size × pool_size
        # layer=4 时不池化 (已经是 1D 向量了)

        self.embed.add_module('Flatten', Flatten())
        # 展平: (B, C, pool_H, pool_W) → (B, C × pool_H × pool_W)

        self.embed.add_module(
            'Linear',
            nn.Linear(nChannels * pool_size * pool_size, dim_out)
        )
        # 线性投影: (B, nChannels × pool_size²) → (B, dim_out)
        # 比如 layer=0: 16 × 8 × 8 = 1024 → 128

        self.embed.add_module('Normalize', Normalize(2))
        # L2 归一化: 输出向量的 L2 范数 = 1

    def forward(self, x):
        """
        前向传播

        输入: x, 形状 (B, C, H, W) 的特征图 (layer 0-3)
              或 (B, D) 的特征向量 (layer 4)
        输出: (B, dim_out), L2 归一化后的嵌入向量
        """
        return self.embed(x)
        # 依次执行: [池化] → 展平 → 线性投影 → L2 归一化


# ============================================================
# 测试代码: 验证 ConnectorV2 的功能
# ============================================================
if __name__ == '__main__':
    import torch

    # ---- 构造假数据: 模拟学生和教师各 3 层的特征 ----
    g_s = [
        torch.randn(2, 16, 16, 16),
        # 学生第 1 层: batch=2, 16 通道, 16×16 空间
        torch.randn(2, 32, 8, 8),
        # 学生第 2 层: 32 通道, 8×8 空间
        torch.randn(2, 64, 4, 4),
        # 学生第 3 层: 64 通道, 4×4 空间
    ]
    g_t = [
        torch.randn(2, 32, 16, 16),
        # 教师第 1 层: 32 通道, 16×16
        torch.randn(2, 64, 8, 8),
        # 教师第 2 层: 64 通道, 8×8
        torch.randn(2, 128, 4, 4),
        # 教师第 3 层: 128 通道, 4×4
    ]

    # 获取各层形状
    s_shapes = [s.shape for s in g_s]
    # [(2,16,16,16), (2,32,8,8), (2,64,4,4)]
    t_shapes = [t.shape for t in g_t]
    # [(2,32,16,16), (2,64,8,8), (2,128,4,4)]

    # 创建 ConnectorV2
    net = ConnectorV2(s_shapes, t_shapes)
    # 内部为每一层创建 1×1 卷积:
    #   第 1 层: Conv2d(16, 32, 1×1) + BN(32)
    #   第 2 层: Conv2d(32, 64, 1×1) + BN(64)
    #   第 3 层: Conv2d(64, 128, 1×1) + BN(128)

    # 前向传播: 把学生特征映射到教师维度
    out = net(g_s)

    # 验证输出形状
    for f in out:
        print(f.shape)
    # 期望输出:
    #   torch.Size([2, 32, 16, 16])   ← 16→32 通道, 空间不变
    #   torch.Size([2, 64, 8, 8])     ← 32→64 通道, 空间不变
    #   torch.Size([2, 128, 4, 4])    ← 64→128 通道, 空间不变