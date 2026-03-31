# ============================================================
# FeatureAlign.py - FitNet 风格中间层特征对齐模块
#
# 放置位置: KD/distiller_zoo/FeatureAlign.py
#
# 对应公式中的第二项:
#   β · D_feature(T_t(F^t) || T_s(F^s))
#
# 其中:
#   F^t, F^s: 教师和学生的中间层特征图 (4D tensor)
#   T_t, T_s: 各自的变换函数 (1×1 卷积, 对齐通道数到共享维度)
#   D_feature: 距离度量 (MSE)
#
# 与标准 FitNet 的区别:
#   标准 FitNet: 只变换学生侧 → T_s(F^s) ≈ F^t
#   本模块:     双侧变换 → T_t(F^t) ≈ T_s(F^s)
#   原因: ResNet50(2048维) 和 ResNet14(64维) 差距太大,
#         双侧变换到共享空间比单侧膨胀更合理
#
# 支持多层对齐:
#   可以同时对齐多对特征层 (如 16×16 和 8×8)
#   每对特征层有独立的变换模块
#
# 使用方法:
#   # 创建时指定要对齐的层对
#   align_module = FeatureAlignLoss(
#       s_shapes=[(B,32,16,16), (B,64,8,8)],   # 学生特征 shape
#       t_shapes=[(B,512,16,16), (B,1024,8,8)], # 教师特征 shape
#       feat_dim=128                             # 共享空间通道数
#   )
#   # 训练时传入对应层的特征
#   loss = align_module(
#       [feat_s[4], feat_s[6]],   # 学生: stage2末尾, stage3末尾
#       [feat_t[7], feat_t[13]]   # 教师: stage2末尾, stage3末尾
#   )
# ============================================================

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureTransform(nn.Module):
    """
    单层特征变换模块: 用 1×1 卷积将特征图通道数变换到共享维度

    为什么用 1×1 卷积而不是 nn.Linear?
      因为中间层特征是 4D tensor (B, C, H, W), 不是 1D 向量
      1×1 卷积等价于对每个空间位置做线性变换, 保留空间结构

    为什么加 BatchNorm?
      - 稳定训练, 加速收敛
      - FitNet 的 ConvReg 模块也用了 BN
      - 在 models/util.py 的 ConvReg 中可以看到同样的设计
    """

    def __init__(self, in_channels, out_channels):
        """
        参数:
            in_channels: int, 输入特征图的通道数
                (如教师 stage2: 512, 学生 stage2: 32)
            out_channels: int, 变换后的通道数 (共享维度 feat_dim)
                (如 128)
        """
        super(FeatureTransform, self).__init__()

        # 1×1 卷积: 只做通道维度的线性变换, 不改变空间尺寸
        # (B, in_channels, H, W) → (B, out_channels, H, W)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        # BatchNorm: 对变换后的特征做归一化
        self.bn = nn.BatchNorm2d(out_channels)

        # 权重初始化: Kaiming 初始化, 适合 ReLU 之前的卷积层
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        """
        输入: (B, in_channels, H, W)
        输出: (B, out_channels, H, W)
        """
        return self.bn(self.conv(x))


class FeatureAlignLoss(nn.Module):
    """
    FitNet 风格多层特征对齐损失

    支持同时对齐多对教师-学生特征层, 每对有独立的变换模块。
    双侧变换: 教师和学生各自通过 1×1 卷积映射到共享通道维度,
    然后计算 MSE 距离。

    对于 ResNet50 (教师) → ResNet14 (学生), 推荐对齐:
      学生 stage2 末尾 (32, 16, 16) ↔ 教师 stage2 末尾 (512, 16, 16)
      学生 stage3 末尾 (64, 8, 8)   ↔ 教师 stage3 末尾 (1024, 8, 8)

    feat 列表中的具体索引:

    ResNet14 (resnet.py, depth=14, n=2 blocks/stage):
      feat_s[0] = relu(f0)         (16, 32, 32)  conv1 输出
      feat_s[1] = stage1 block1    (16, 32, 32)
      feat_s[2] = stage1 block2    (16, 32, 32)  ← stage1 末尾
      feat_s[3] = stage2 block1    (32, 16, 16)
      feat_s[4] = stage2 block2    (32, 16, 16)  ← stage2 末尾 ★
      feat_s[5] = stage3 block1    (64, 8, 8)
      feat_s[6] = stage3 block2    (64, 8, 8)    ← stage3 末尾 ★
      feat_s[7] = avgpool 后向量   (64,)

    ResNet50 (resnetv2.py, Bottleneck [3,4,6,3]):
      feat_t[0]  = relu(f0)        (64, 32, 32)   conv1 输出
      feat_t[1-3]   = stage1       (256, 32, 32)   3 个 Bottleneck
      feat_t[4-7]   = stage2       (512, 16, 16)   4 个 Bottleneck
      feat_t[8-13]  = stage3       (1024, 8, 8)    6 个 Bottleneck
      feat_t[14-16] = stage4       (2048, 4, 4)    3 个 Bottleneck
      feat_t[17]    = avgpool 后   (2048,)

    推荐对齐方案 (按空间分辨率匹配, 取每个 stage 最后一个 block):
      feat_s[4] (32, 16, 16) ↔ feat_t[7]  (512, 16, 16)   16×16 分辨率
      feat_s[6] (64, 8, 8)   ↔ feat_t[13] (1024, 8, 8)    8×8 分辨率
    """

    def __init__(self, s_shapes, t_shapes, feat_dim=128):
        """
        参数:
            s_shapes: list of tuple, 要对齐的学生特征层的 shape
                如 [(2, 32, 16, 16), (2, 64, 8, 8)]
                第一维是 batch (探测时用的假数据 batch=2), 用不到
                第二维是通道数, 用于构建变换模块
            t_shapes: list of tuple, 对应的教师特征层的 shape
                如 [(2, 512, 16, 16), (2, 1024, 8, 8)]
                必须和 s_shapes 等长, 一一对应
            feat_dim: int, 共享空间的通道数
                教师和学生特征都映射到这个维度后再算 MSE
                建议: 64 或 128
        """
        super(FeatureAlignLoss, self).__init__()

        assert len(s_shapes) == len(t_shapes), \
            "学生和教师的对齐层数必须相同, 收到 {} vs {}".format(
                len(s_shapes), len(t_shapes))

        self.num_pairs = len(s_shapes)  # 对齐多少对特征层
        self.feat_dim = feat_dim

        # 为每对特征层创建独立的变换模块
        # T_s: 学生特征 → 共享空间
        self.transforms_s = nn.ModuleList([
            FeatureTransform(s_shape[1], feat_dim)  # s_shape[1] = 通道数
            for s_shape in s_shapes
        ])

        # T_t: 教师特征 → 共享空间
        self.transforms_t = nn.ModuleList([
            FeatureTransform(t_shape[1], feat_dim)  # t_shape[1] = 通道数
            for t_shape in t_shapes
        ])

        # MSE 损失
        self.mse = nn.MSELoss()

        # 打印对齐配置
        print("\n特征对齐模块配置:")
        for i, (s, t) in enumerate(zip(s_shapes, t_shapes)):
            print("  第{}对: 学生 {} → {} ↔ 教师 {} → {}".format(
                i, s[1], feat_dim, t[1], feat_dim))
            # 检查空间分辨率是否匹配
            if len(s) == 4 and len(t) == 4:
                if s[2] != t[2] or s[3] != t[3]:
                    print("    警告: 空间分辨率不匹配 ({}×{} vs {}×{}), "
                          "将自动用 adaptive_avg_pool2d 对齐".format(
                              s[2], s[3], t[2], t[3]))

    def forward(self, g_s, g_t):
        """
        计算多层特征对齐损失

        参数:
            g_s: list of Tensor, 学生的特征图列表
                每个 Tensor shape = (B, C_s, H, W)
                列表长度 = self.num_pairs
            g_t: list of Tensor, 教师的特征图列表 (已 detach)
                每个 Tensor shape = (B, C_t, H, W)
                列表长度 = self.num_pairs

        返回:
            loss: 标量, 所有对齐层的 MSE 损失之和
        """
        assert len(g_s) == self.num_pairs, \
            "学生特征数量 ({}) 和对齐层数 ({}) 不匹配".format(
                len(g_s), self.num_pairs)
        assert len(g_t) == self.num_pairs, \
            "教师特征数量 ({}) 和对齐层数 ({}) 不匹配".format(
                len(g_t), self.num_pairs)

        total_loss = 0

        for i in range(self.num_pairs):
            f_s = g_s[i]  # 学生第 i 对特征, (B, C_s, H_s, W_s)
            f_t = g_t[i]  # 教师第 i 对特征, (B, C_t, H_t, W_t)

            # 如果空间分辨率不同, 用 adaptive_avg_pool2d 对齐到较小的那个
            s_H, s_W = f_s.shape[2], f_s.shape[3]
            t_H, t_W = f_t.shape[2], f_t.shape[3]
            if s_H != t_H or s_W != t_W:
                target_H = min(s_H, t_H)
                target_W = min(s_W, t_W)
                if s_H != target_H:
                    f_s = F.adaptive_avg_pool2d(f_s, (target_H, target_W))
                if t_H != target_H:
                    f_t = F.adaptive_avg_pool2d(f_t, (target_H, target_W))

            # 双侧变换到共享空间
            # T_s(F^s): (B, C_s, H, W) → (B, feat_dim, H, W)
            f_s_proj = self.transforms_s[i](f_s)

            # T_t(F^t): (B, C_t, H, W) → (B, feat_dim, H, W)
            # 教师侧 detach, 不传梯度给教师主干
            f_t_proj = self.transforms_t[i](f_t.detach())

            # D_feature = MSE(T_s(F^s), T_t(F^t))
            loss_i = self.mse(f_s_proj, f_t_proj)

            total_loss += loss_i

        return total_loss


# ============================================================
# 工具函数: 根据模型架构自动确定对齐层索引
# ============================================================

def get_align_layer_indices(model_name, role='student'):
    """
    根据模型名自动返回推荐的对齐层索引 (取每个 stage 的末尾)

    参数:
        model_name: str, 模型架构名
        role: str, 'student' 或 'teacher'

    返回:
        dict, 键为空间分辨率字符串, 值为 feat 列表中的索引
        如 {'16x16': 4, '8x8': 6}

    使用方法:
        s_indices = get_align_layer_indices('resnet14', 'student')
        t_indices = get_align_layer_indices('ResNet50', 'teacher')
        # 然后取交集的分辨率, 确定要对齐哪些层
    """

    # ============ resnet.py 系列 (BasicBlock, 3 stages) ============
    # depth → n = (depth-2)//6 blocks per stage
    # feat = [relu(f0)] + f1_act(n items) + f2_act(n items) + f3_act(n items) + [f4]
    # stage1 末尾 = index n      (32×32)
    # stage2 末尾 = index 2n     (16×16)
    # stage3 末尾 = index 3n     (8×8)
    # f4 (向量)   = index 3n+1

    resnet_configs = {
        'resnet8':  1,   # n=1
        'resnet14': 2,   # n=2
        'resnet20': 3,   # n=3
        'resnet32': 5,   # n=5
        'resnet44': 7,   # n=7
        'resnet56': 9,   # n=9
        'resnet110': 18, # n=18
    }

    # resnet_x4 系列: 通道数不同但 block 数量相同
    resnet_x4_configs = {
        'resnet8x4': 1,
        'resnet32x4': 5,
    }

    if model_name in resnet_configs:
        n = resnet_configs[model_name]
        return {
            '32x32': n,       # stage1 末尾
            '16x16': 2 * n,   # stage2 末尾
            '8x8': 3 * n,     # stage3 末尾
        }

    if model_name in resnet_x4_configs:
        n = resnet_x4_configs[model_name]
        return {
            '32x32': n,
            '16x16': 2 * n,
            '8x8': 3 * n,
        }

    # ============ resnetv2.py (Bottleneck, 4 stages) ============
    # ResNet50: blocks = [3, 4, 6, 3]
    # feat = [relu(f0)] + f1_act(3) + f2_act(4) + f3_act(6) + f4_act(3) + [f5]
    # stage1 末尾 = index 3      (32×32, 256ch)
    # stage2 末尾 = index 3+4=7  (16×16, 512ch)
    # stage3 末尾 = index 7+6=13 (8×8, 1024ch)
    # stage4 末尾 = index 13+3=16 (4×4, 2048ch)

    if model_name == 'ResNet50':
        blocks = [3, 4, 6, 3]
        cumsum = [sum(blocks[:i+1]) for i in range(len(blocks))]
        return {
            '32x32': cumsum[0],    # 3
            '16x16': cumsum[1],    # 7
            '8x8': cumsum[2],      # 13
            '4x4': cumsum[3],      # 16
        }

    # ============ WRN 系列 (3 stages) ============
    # wrn_D_W: depth=D, n=(D-4)//6 blocks per stage
    # feat = f1_act(n) + f2_act(n) + f3_act(n) + [relu(f4), f5]
    # 注意: WRN 的 feat 列表不以 f0 开头!
    # stage1 末尾 = index n-1     (32×32)
    # stage2 末尾 = index 2n-1    (16×16)
    # stage3 末尾 = index 3n-1    (8×8)

    wrn_configs = {
        'wrn_16_1': 2,  'wrn_16_2': 2,
        'wrn_40_1': 6,  'wrn_40_2': 6,
    }

    if model_name in wrn_configs:
        n = wrn_configs[model_name]
        return {
            '32x32': n - 1,
            '16x16': 2 * n - 1,
            '8x8': 3 * n - 1,
        }

    # 未知模型: 返回空, 让用户手动指定
    print("警告: 未知模型 '{}', 请手动指定对齐层索引".format(model_name))
    return {}


def get_default_align_pairs(model_s_name, model_t_name):
    """
    根据教师和学生的模型名, 自动确定推荐的对齐层对

    逻辑: 找到两边都有的空间分辨率, 取每个分辨率的 stage 末尾

    参数:
        model_s_name: str, 学生模型名 (如 'resnet14')
        model_t_name: str, 教师模型名 (如 'ResNet50')

    返回:
        s_layer_indices: list of int, 学生 feat 列表的索引
        t_layer_indices: list of int, 教师 feat 列表的索引

    示例:
        >>> get_default_align_pairs('resnet14', 'ResNet50')
        ([4, 6], [7, 13])
        # 意思是: feat_s[4] ↔ feat_t[7] (16×16)
        #         feat_s[6] ↔ feat_t[13] (8×8)
    """
    s_layers = get_align_layer_indices(model_s_name, 'student')
    t_layers = get_align_layer_indices(model_t_name, 'teacher')

    # 找交集: 两边都有的空间分辨率
    common_resolutions = sorted(
        set(s_layers.keys()) & set(t_layers.keys()),
        key=lambda r: -int(r.split('x')[0])  # 按分辨率从大到小排序
    )

    # 排除 32×32 (太浅, 特征不够有区分性)
    # 也排除 4×4 (学生可能没有这个分辨率)
    preferred = ['16x16', '8x8']
    selected = [r for r in preferred if r in common_resolutions]

    if not selected:
        # 退而求其次: 用所有共同分辨率
        selected = common_resolutions

    s_indices = [s_layers[r] for r in selected]
    t_indices = [t_layers[r] for r in selected]

    print("\n自动对齐层配置 ({} → {}):".format(model_t_name, model_s_name))
    for r, si, ti in zip(selected, s_indices, t_indices):
        print("  分辨率 {}: feat_s[{}] ↔ feat_t[{}]".format(r, si, ti))

    return s_indices, t_indices