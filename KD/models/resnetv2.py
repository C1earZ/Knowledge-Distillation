'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
# 模块说明：PyTorch 实现的 ResNet，参考何恺明 2015 年论文《深度残差学习》

import torch                          # 导入 PyTorch 核心库
import torch.nn as nn                 # 导入神经网络模块（含 Conv2d、BN、Linear 等）
import torch.nn.functional as F       # 导入函数式 API（含 relu 等无参数操作）


# ─────────────────────────────────────────────
# BasicBlock：用于 ResNet-18 / ResNet-34
# 结构：Conv3x3 → BN → ReLU → Conv3x3 → BN → (+shortcut)
# ─────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1   # 输出通道相对于 planes 的扩张倍数（BasicBlock 不扩张，故为 1）

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        # in_planes：输入通道数
        # planes：中间/输出通道数
        # stride：第一个卷积的步幅（stride=2 时做下采样）
        # is_last：是否是该 stage 的最后一个 block（用于控制是否输出 preact 特征）
        super(BasicBlock, self).__init__()   # 调用父类 nn.Module 初始化

        self.is_last = is_last               # 保存 is_last 标志

        # 第一个 3×3 卷积，stride 控制是否下采样，padding=1 保持空间尺寸（当 stride=1 时）
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)    # conv1 之后的 BatchNorm

        # 第二个 3×3 卷积，stride 固定为 1，不再做下采样
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)    # conv2 之后的 BatchNorm

        self.relu = nn.ReLU(inplace=False)   # ReLU 激活函数（inplace=False 保留原始张量，便于特征提取）

        # shortcut（捷径连接）：默认为恒等映射（空 Sequential = 什么都不做）
        self.shortcut = nn.Sequential()

        # 当步幅不为 1（发生下采样）或通道数不匹配时，需要用 1×1 卷积对齐维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                # 1×1 卷积：调整通道数和空间尺寸，使 shortcut 与主路输出一致
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)  # shortcut 分支的 BN
            )

    def forward(self, x):
        # 兼容层间传递特征列表的设计：
        # 若 x 是 tuple，则解包为 (张量, 已收集的特征列表)；否则初始化空列表
        if isinstance(x, tuple):
            x, features = x
        else:
            features = []

        x = self.relu(x)   # 对输入先做 ReLU（Pre-activation 风格的入口激活）

        # 主路前向：conv1 → bn1 → relu → conv2 → bn2
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积块（含激活）
        out = self.bn2(self.conv2(out))         # 第二个卷积块（暂不激活，等加完 shortcut）

        out += self.shortcut(x)   # 残差相加：主路输出 + shortcut（捷径）输出

        # 返回当前输出，并把本 block 的输出 out 追加到特征列表（供知识蒸馏等用途）
        return out, features + [out]


# ─────────────────────────────────────────────
# Bottleneck：用于 ResNet-50 / 101 / 152
# 结构：Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+shortcut)
# 三层设计：先降维 → 3×3 卷积 → 再升维，节省参数
# ─────────────────────────────────────────────
class Bottleneck(nn.Module):
    expansion = 4   # 输出通道是 planes 的 4 倍（bottleneck 特有的扩张系数）

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last

        # 第一个 1×1 卷积：降维（in_planes → planes），不改变空间尺寸
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第二个 3×3 卷积：在低维空间提取空间特征，stride 控制下采样
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 第三个 1×1 卷积：升维（planes → expansion*planes = 4*planes）
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.relu = nn.ReLU(inplace=False)   # ReLU 激活

        # shortcut 默认恒等映射
        self.shortcut = nn.Sequential()

        # 维度不匹配时，1×1 卷积对齐 shortcut
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # 同 BasicBlock，兼容 tuple 输入以传递特征列表
        if isinstance(x, tuple):
            x, features = x
        else:
            features = []

        x = self.relu(x)   # 输入先激活

        # 主路：1×1 降维 → 3×3 特征提取 → 1×1 升维
        out = self.relu(self.bn1(self.conv1(x)))   # conv1 + bn + relu
        out = self.relu(self.bn2(self.conv2(out))) # conv2 + bn + relu
        out = self.bn3(self.conv3(out))             # conv3 + bn（不激活，等残差相加后再激活）

        out += self.shortcut(x)   # 残差相加

        return out, features + [out]   # 返回输出及更新后的特征列表


# ─────────────────────────────────────────────
# ResNet 主网络
# ─────────────────────────────────────────────
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False):
        # block：使用的 block 类型（BasicBlock 或 Bottleneck）
        # num_blocks：各 stage 包含的 block 数量，如 [3,4,6,3]
        # num_classes：分类数
        # zero_init_residual：是否将每个残差分支最后一个 BN 的 γ 初始化为 0
        super(ResNet, self).__init__()

        self.in_planes = 64   # 记录当前输入通道数，随各 stage 累积更新

        # stem 层：3×3 卷积，输入 3 通道（RGB），输出 64 通道，不下采样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)   # stem 的 BN

        # 四个 stage，通道数逐渐翻倍，从 stage2 开始空间尺寸减半（stride=2）
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)  # stage1：不下采样
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # stage2：下采样 ÷2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # stage3：下采样 ÷2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # stage4：下采样 ÷2

        # 全局平均池化：将任意空间尺寸压缩为 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接分类头：512*expansion → num_classes
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.relu = nn.ReLU(inplace=False)   # 共用的 ReLU（inplace=False）

        # ── 权重初始化 ──
        for m in self.modules():   # 遍历所有子模块
            if isinstance(m, nn.Conv2d):
                # 卷积层：Kaiming 正态初始化，适配 ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # BN/GN：γ=1（不缩放），β=0（不偏移）
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 可选：将每个残差分支最后一个 BN 的 γ 初始化为 0
        # 使残差分支初始输出为 0，整个 block 退化为恒等映射，有利于训练初期稳定
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)   # Bottleneck 最后一个 BN
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)   # BasicBlock 最后一个 BN

    def get_feat_modules(self):
        """返回用于提取中间特征的模块列表（供知识蒸馏框架调用）"""
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)    # stem 卷积
        feat_m.append(self.bn1)      # stem BN
        feat_m.append(self.layer1)   # stage1
        feat_m.append(self.layer2)   # stage2
        feat_m.append(self.layer3)   # stage3
        feat_m.append(self.layer4)   # stage4
        return feat_m

    def get_bn_before_relu(self):
        """返回每个 stage 最后一个 block 中、ReLU 之前的 BN 层（供 AT 等蒸馏方法使用）"""
        if isinstance(self.layer1[0], Bottleneck):
            # Bottleneck：最后一个 BN 是 bn3
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            # BasicBlock：最后一个 BN 是 bn2
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')   # 未知 block 类型报错
        return [bn1, bn2, bn3, bn4]   # 返回四个 stage 各自的目标 BN

    def _make_layer(self, block, planes, num_blocks, stride):
        """构建一个 stage（由多个 block 串联而成）"""
        # 第一个 block 使用指定 stride（可能下采样），其余 block stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            # 最后一个 block 标记 is_last=True
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            # 更新 in_planes 为当前输出通道数（含 expansion 倍数）
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)   # 将所有 block 串联为 Sequential

    def forward(self, x, is_feat=False, preact=False):
        # is_feat=True 时同时返回中间特征；preact 参数当前未启用（已注释）

        # stem 前向：conv → bn（注意：此处暂不做 ReLU，留到 layer1 入口处激活）
        x = self.bn1(self.conv1(x))
        f0 = x   # 保存 stem 输出（激活前）作为 f0 特征

        # 四个 stage 前向，每个 layer 返回 (最终输出, 各 block 输出列表)
        x, f1 = self.layer1(x)
        f1_act = [self.relu(f) for f in f1]   # 对 stage1 每个 block 输出做 ReLU

        x, f2 = self.layer2(x)
        f2_act = [self.relu(f) for f in f2]   # 对 stage2 每个 block 输出做 ReLU

        x, f3 = self.layer3(x)
        f3_act = [self.relu(f) for f in f3]   # 对 stage3 每个 block 输出做 ReLU

        x, f4 = self.layer4(x)
        f4_act = [self.relu(f) for f in f4]   # 对 stage4 每个 block 输出做 ReLU

        # 全局平均池化（先对 x 做最终 ReLU）：(B, C, H, W) → (B, C, 1, 1)
        out = self.avgpool(self.relu(x))
        out = out.view(out.size(0), -1)   # 展平：(B, C, 1, 1) → (B, C)
        f5 = out                           # 保存池化后的全局特征向量 f5

        out = self.linear(out)   # 全连接层：(B, C) → (B, num_classes)

        if is_feat:
            # 返回所有中间特征列表 + 分类 logits
            # 特征顺序：f0(stem) + f1各block + f2各block + f3各block + f4各block + f5(池化后)
            return [self.relu(f0)] + f1_act + f2_act + f3_act + f4_act + [f5], out
        else:
            return out   # 普通推理只返回 logits


# ── 各版本 ResNet 的工厂函数 ──

def ResNet18(**kwargs):
    # 2+2+2+2 = 8 个 BasicBlock → 18 层
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    # 3+4+6+3 = 16 个 BasicBlock → 34 层
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def ResNet50(**kwargs):
    # 3+4+6+3 = 16 个 Bottleneck（每个 3 层）→ 50 层
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def ResNet101(**kwargs):
    # 3+4+23+3 = 33 个 Bottleneck → 101 层
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def ResNet152(**kwargs):
    # 3+8+36+3 = 50 个 Bottleneck → 152 层
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


# ── 测试入口 ──
if __name__ == '__main__':
    net = ResNet50(num_classes=1000)          # 构建 ResNet50，1000 类（ImageNet）
    from ptflops import get_model_complexity_info  # 导入 FLOPs 计算工具

    x = torch.randn(2, 3, 32, 32)            # 构造随机输入：batch=2, 3通道, 32×32

    # 计算模型在 224×224 输入下的 FLOPs 和参数量
    flops, params = get_model_complexity_info(net, (3, 224, 224))

    import pdb
    pdb.set_trace()   # 断点调试：可在此处检查 flops/params

    # 前向传播，获取中间特征和分类结果
    feats, logit = net(x, is_feat=True, preact=True)

    # 打印每个特征图的形状和最小值（检查有无数值异常）
    for f in feats:
        print(f.shape, f.min().item())

    print(logit.shape)   # 打印输出 logits 形状，预期 (2, 1000)

    # 验证 get_bn_before_relu 返回的是否都是 BN 层
    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')     # 类型正确
        else:
            print('warning')  # 类型异常，提示检查

# ### 整体架构一览
# 输入 (B,3,H,W)
#    ↓ conv1 + bn1          ← stem
#    ↓ layer1 (stage1)      ← 不下采样，64ch
#    ↓ layer2 (stage2)      ← ÷2，128ch
#    ↓ layer3 (stage3)      ← ÷2，256ch
#    ↓ layer4 (stage4)      ← ÷2，512ch
#    ↓ avgpool → flatten    ← 全局池化
#    ↓ linear               ← 分类输出