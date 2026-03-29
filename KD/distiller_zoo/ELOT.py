
# ELOT 求解器适配代码仓库 (ELOT_25/DeepDA/office/elot.py)
# 依赖: pip install POT (Python Optimal Transport)
# ============================================================

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
from ot.lp import emd



# ============================================================
# 第一部分: ELOT 求解器
# 直接从导师仓库 ELOT_25/DeepDA/office/elot.py 适配而来
# 核心思想: 通过添加虚拟点 (dummy points), 将弹性 OT 问题
#           转化为标准 OT 问题求解
# ============================================================

def elot_emd(a, b, M, nb_dummies=1, log=False, **kwargs):
    """
    ELOT 的精确求解版本 (基于线性规划 EMD)

    通过在源和目标分布上各添加 nb_dummies 个虚拟点,
    将弹性 OT 问题转化为等价的标准 OT 问题。
    虚拟点与真实点之间的传输代价设为 0,
    从而允许部分质量 "传输" 到虚拟点上 (即不被传输)。

    参数:
        a: (K,) numpy 数组, 源分布的质量向量 (教师侧)
        b: (K,) numpy 数组, 目标分布的质量向量 (学生侧)
        M: (K, K) numpy 数组, 代价矩阵 C_kl
        nb_dummies: int, 虚拟点数量, 越多越灵活但计算量越大
        log: bool, 是否返回日志信息

    返回:
        gamma: (K, K) numpy 数组, 最优传输计划 (截取真实点部分)
    """
    # 扩展边际分布: 在源和目标各添加虚拟点
    # 虚拟点的质量设为对方总质量的均分
    b_extended = np.append(b, [(np.sum(a)) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b)) / nb_dummies] * nb_dummies)

    # 扩展代价矩阵: 虚拟点之间及与真实点之间的传输代价为 0
    M_extended = np.zeros((len(a_extended), len(b_extended)))
    M_extended[:len(a), :len(b)] = M  # 只有真实点之间有代价

    # 调用标准 EMD 求解器
    gamma, log_ot = emd(a_extended, b_extended, M_extended, log=True, **kwargs)

    if log_ot['warning'] is not None:
        raise ValueError("EMD 求解出错: 尝试增加 nb_dummies 的值")

    log_ot['partial_w_dist'] = np.sum(M * gamma[:len(a), :len(b)])

    if log:
        return gamma[:len(a), :len(b)], log_ot
    else:
        return gamma[:len(a), :len(b)]


def elot_entropic(a, b, M, reg, nb_dummies=1, numItermax=1000,
                  stopThr=1e-100, verbose=False, log=False, **kwargs):
    """
    ELOT 的熵正则化版本 (基于 Sinkhorn 算法)

    与 elot_emd 原理相同, 但使用 Sinkhorn 算法求解熵正则化版本,
    计算速度更快, 且传输计划更平滑 (不会是稀疏的)。

    参数:
        a, b: 源和目标分布的质量向量
        M: 代价矩阵
        reg: float, 熵正则化系数 (越大传输计划越平滑, 但可能偏离精确解)
        nb_dummies: 虚拟点数量
        numItermax: Sinkhorn 最大迭代次数
        stopThr: 收敛阈值

    返回:
        gamma: 最优传输计划 (截取真实点部分)
    """
    b_extended = np.append(b, [(np.sum(a)) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b)) / nb_dummies] * nb_dummies)
    M_extended = np.zeros((len(a_extended), len(b_extended)))
    M_extended[:len(a), :len(b)] = M

    gamma, log_ot = ot.sinkhorn(
        a_extended, b_extended, M_extended, reg,
        numItermax=numItermax, stopThr=stopThr,
        verbose=verbose, log=True, **kwargs
    )

    log_ot['partial_w_dist'] = np.sum(M * gamma[:len(a), :len(b)])

    if log:
        return gamma[:len(a), :len(b)], log_ot
    else:
        return gamma[:len(a), :len(b)]


# ============================================================
# 第二部分: MMD (最大均值差异) 计算
# 用 RBF (高斯) 核衡量两组特征样本分布之间的距离
# ============================================================

def compute_rbf_mmd2(X, Y, sigma=1.0):
    n = X.size(0)
    m = Y.size(0)
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=X.device)
    gamma = 1.0 / (2.0 * sigma ** 2)
    xx_dist = torch.cdist(X, X) ** 2
    kxx = torch.exp(-gamma * xx_dist)
    yy_dist = torch.cdist(Y, Y) ** 2
    kyy = torch.exp(-gamma * yy_dist)
    xy_dist = torch.cdist(X, Y) ** 2
    kxy = torch.exp(-gamma * xy_dist)
    mmd2 = kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
    return mmd2


# ============================================================
# 映射层 (Projection Layer)
# 将教师/学生不同维度的权重和特征映射到统一维度空间
# ============================================================

class ProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        # Xavier 初始化: 让映射初始时既不放大也不缩小信号
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


# ============================================================
# 工具函数 — 从模型中提取分类头权重
# 兼容 KD 框架中所有支持的模型架构
# ============================================================

def get_classifier_weight(model):
    """
    从不同架构的模型中提取分类头 (最后一层全连接层) 的权重矩阵

    支持的模型架构 (对应 KD/models/ 目录):
        - resnet.py 系列 (resnet8/20/32/56/110/8x4/32x4): model.fc.weight
        - resnetv2.py (ResNet50): model.linear.weight
        - wrn.py (wrn_16_1/2, wrn_40_1/2): model.fc.weight
        - mobilenetv2.py (MobileNetV2): model.classifier[0].weight
        - vgg.py: model.fc.weight (假设结构一致)
    参数:
        model: nn.Module, 模型实例 (可能被 DataParallel 包裹)

    返回:
        weight: (num_classes, feat_dim) 的权重 tensor
    """
    # 如果模型被 DataParallel 包裹, 先取出内部模型
    if isinstance(model, nn.DataParallel):
        model = model.module

    # 按优先级依次尝试不同的属性名
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        # resnet.py, wrn.py, vgg.py 等
        return model.fc.weight
    elif hasattr(model, 'linear') and isinstance(model.linear, nn.Linear):
        # resnetv2.py (ResNet50)
        return model.linear.weight
    elif hasattr(model, 'classifier'):
        # mobilenetv2.py: classifier 是 nn.Sequential([nn.Linear(...)])
        if isinstance(model.classifier, nn.Sequential):
            for m in model.classifier:
                if isinstance(m, nn.Linear):
                    return m.weight
        elif isinstance(model.classifier, nn.Linear):
            return model.classifier.weight

    raise ValueError(
        "无法自动提取分类头权重, 请检查模型结构。"
        "支持的属性名: model.fc, model.linear, model.classifier"
    )


# ============================================================
# ELOTLoss — ELOT 蒸馏损失 (核心模块)
# ============================================================

class ELOTLoss(nn.Module):
    """
    基于弹性最优传输 (ELOT) 的知识蒸馏损失

    整体流程:
        1. 通过可训练映射层将教师和学生的分类头权重映射到统一空间
        2. 计算权重向量的欧氏距离矩阵 ||w_k^T - w_l^S||²
        3. 按当前 batch 的标签, 计算类条件特征分布的 MMD² 矩阵
        4. 组合得到动态代价矩阵 C_kl = ||·||² - λ·MMD²
        5. 使用 ELOT 求解器求最优传输计划 π
        6. 计算传输损失 L = Σ_{k,l} π_{kl} · C_{kl}

    Warmup 策略 (来自 ELOT 训练代码 train.py 中的逻辑):
        导师代码中: if id_iter <= 3: 用标准 OT; else: 用 ELOT
        本模块中: 前 warmup_iters 次迭代使用标准 OT (质量完全守恒),
        之后切换到 ELOT (允许部分质量不传输)。
        理由: 训练初期特征不稳定, 标准 OT 提供更强的约束;
        训练后期 ELOT 可以自动忽略噪声类别匹配。
    """

    def __init__(self, t_dim, s_dim, proj_dim, num_classes,
                 lambda_mmd=0.1, epsilon=0.1, sigma=1.0,
                 warmup_iters=500, nb_dummies=1):
        """
        参数:
            t_dim: int, 教师模型的特征维度
                (即分类头权重矩阵的列数, 如 resnet110 → 64)
            s_dim: int, 学生模型的特征维度
                (如 resnet20 → 64, resnet8x4 → 256)
            proj_dim: int, 映射后的统一维度
                (建议取 min(t_dim, s_dim) 或 128)
            num_classes: int, 类别总数 (如 CIFAR-100 → 100)
            lambda_mmd: float, 公式中的 λ, 控制 MMD 奖励项的强度
            epsilon: float, OT 的熵正则化系数
                (0 = 精确 EMD, >0 = Sinkhorn 近似)
            sigma: float, RBF 核的带宽参数 σ, 影响 MMD 对距离的敏感度
            warmup_iters: int, 前多少次迭代使用标准 OT 作为 warmup
                (参考导师代码 train.py 中 id_iter <= 3 的逻辑)
            nb_dummies: int, ELOT 虚拟点数量
        """
        super(ELOTLoss, self).__init__()

        # 可训练映射层: 教师和学生各一个, 映射到同一 proj_dim 维空间
        self.proj_t = ProjectionLayer(t_dim, proj_dim)
        self.proj_s = ProjectionLayer(s_dim, proj_dim)

        # 保存超参数
        self.lambda_mmd = lambda_mmd
        self.epsilon = epsilon
        self.sigma = sigma
        self.warmup_iters = warmup_iters
        self.nb_dummies = nb_dummies
        self.num_classes = num_classes
        self.proj_dim = proj_dim

    def forward(self, feat_s, feat_t, w_s, w_t, target, current_iter):
        """
        计算 ELOT 蒸馏损失

        参数:
            feat_s: (B, s_dim) 学生的特征向量 (FC 层之前, 保留梯度)
            feat_t: (B, t_dim) 教师的特征向量 (已 detach, 不参与梯度)
            w_s: (num_classes, s_dim) 学生分类头权重 (保留梯度)
            w_t: (num_classes, t_dim) 教师分类头权重 (已 detach)
            target: (B,) 真实标签, 值域 [0, num_classes-1]
            current_iter: int, 当前全局迭代步数 (用于判断是否仍在 warmup)

        返回:
            loss: 标量 tensor, ELOT 传输损失
        """
        K = self.num_classes
        device = feat_s.device

        # ========== 步骤 1: 映射分类头权重到统一空间 ==========
        # w_t 已 detach: 不通过梯度更新教师
        w_t_proj = self.proj_t(w_t.detach())   # (K, proj_dim)
        w_s_proj = self.proj_s(w_s)            # (K, proj_dim), 保留梯度

        # ========== 步骤 2: 计算权重欧氏距离矩阵 ==========
        # C_weight[k,l] = ||w_k^T_proj - w_l^S_proj||²
        # 衡量教师第k类和学生第l类在决策空间中的语义接近程度
        C_weight = torch.cdist(w_t_proj, w_s_proj) ** 2  # (K, K)

        # ========== 步骤 3: 计算类条件 MMD 矩阵 ==========
        # 将特征也映射到统一空间 (复用同一组映射层)
        with torch.no_grad():
            feat_t_proj = self.proj_t(feat_t)  # (B, proj_dim)
        feat_s_proj = self.proj_s(feat_s)      # (B, proj_dim)

        C_mmd = torch.zeros(K, K, device=device)

        # 只对当前 batch 中实际出现的类别计算 MMD, 避免无效计算
        unique_labels = torch.unique(target)

        for k_label in unique_labels:
            # 教师中属于第 k 类的特征
            mask_k = (target == k_label)
            feat_t_k = feat_t_proj[mask_k]

            for l_label in unique_labels:
                # 学生中属于第 l 类的特征
                mask_l = (target == l_label)
                feat_s_l = feat_s_proj[mask_l]

                # 计算 MMD² (detach 以避免 MMD 梯度干扰映射层)
                # MMD 仅用于构建代价矩阵, 梯度通过 C_weight 和传输损失回传
                C_mmd[k_label, l_label] = compute_rbf_mmd2(
                    feat_t_k.detach(), feat_s_l.detach(), self.sigma
                )

        # ========== 步骤 4: 构建动态代价矩阵 (公式2) ==========
        # C_kl = ||w_k^T - w_l^S||² - λ · MMD²(P_T^(k), P_S^(l))
        # MMD² 的负值作为奖励: 分布越相似 → MMD² 越小 → 代价越低 → 鼓励匹配
        C = C_weight - self.lambda_mmd * C_mmd

        # ========== 步骤 5: 求解最优传输 ==========
        # 均匀分布作为边际约束 (假设每个类别同等重要)
        a = ot.unif(K)  # (K,) 源分布 (教师侧)
        b = ot.unif(K)  # (K,) 目标分布 (学生侧)

        # 代价矩阵转到 CPU/numpy (POT 求解器在 CPU 上运行)
        C_cpu = C.detach().cpu().numpy()

        # Warmup 策略:
        # 参考导师代码 train.py 第~280行: if id_iter <= 3: 用标准 OT
        # 这里推广为 warmup_iters 参数控制
        if current_iter <= self.warmup_iters:
            # 标准 OT: 所有质量必须完全传输 (强约束, 稳定训练初期)
            if self.epsilon == 0:
                pi = ot.emd(a, b, C_cpu)
            else:
                pi = ot.sinkhorn(a, b, C_cpu, reg=self.epsilon)
        else:
            # ELOT: 弹性传输, 允许部分质量通过虚拟点 "丢弃"
            if self.epsilon == 0:
                pi = elot_emd(a, b, C_cpu, nb_dummies=self.nb_dummies)
            else:
                pi = elot_entropic(a, b, C_cpu, reg=self.epsilon,
                                   nb_dummies=self.nb_dummies)

        # ========== 步骤 6: 计算传输损失 ==========
        # L_OT = Σ_{k,l} π_{kl} · C_{kl}
        pi_tensor = torch.from_numpy(pi).float().to(device)
        loss = torch.sum(pi_tensor * C)

        return loss