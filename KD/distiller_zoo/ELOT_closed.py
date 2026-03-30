# ============================================================
# ELOT_closed.py - 闭集 ELOT 蒸馏损失
#
# 放置位置: KD/distiller_zoo/ELOT_closed.py
#
# 与原始 ELOT.py 的关键区别:
#   1. 教师有 100 类, 学生只有 20 类
#   2. 在计算代价矩阵前, 从教师的 100 类分类头中
#      筛选出对应的 20 类权重向量
#   3. 代价矩阵从 100×100 缩小为 20×20
#   4. OT 在 20 维分布上求解, 计算更快
#
# 映射层方案:
#   采用"双映射到共享空间"的方案 (与原始 ELOT 一致):
#     教师权重 (t_dim,) → proj_t → (proj_dim,)
#     学生权重 (s_dim,) → proj_s → (proj_dim,)
#   这样做的优点:
#     - 不要求教师和学生特征维度相同
#     - 共享空间维度可自由选择
#     - 两个映射层各自学习最佳映射方向
#
#   另一个备选方案是 FitNet 风格的"单映射":
#     学生权重 (s_dim,) → proj → (t_dim,)
#   但这要求映射到教师的高维空间, 灵活性较差
#
# 映射层训练:
#   提供两种模式 (通过 pretrain_proj 参数控制):
#     模式1 (默认): 端到端训练 — 映射层和学生模型一起训练
#     模式2 (可选): FitNet 风格预训练 — 先冻结学生主干,
#                   只训练映射层对齐教师和学生的特征空间,
#                   然后再做端到端蒸馏
#   预训练代码在 helper/train_elot_closed.py 中提供
# ============================================================

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
from ot.lp import emd


# ============================================================
# ELOT 求解器 (与 ELOT.py 中完全一致, 这里复制过来避免循环依赖)
# ============================================================

def elot_emd(a, b, M, nb_dummies=1, log=False, **kwargs):
    """
    ELOT 的精确求解版本 (基于线性规划 EMD)
    通过添加虚拟点将弹性 OT 转化为标准 OT
    """
    b_extended = np.append(b, [(np.sum(a)) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b)) / nb_dummies] * nb_dummies)
    M_extended = np.zeros((len(a_extended), len(b_extended)))
    M_extended[:len(a), :len(b)] = M

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
# MMD 计算 (与 ELOT.py 一致)
# ============================================================

def compute_rbf_mmd2(X, Y, sigma=1.0):
    """
    用 RBF 核计算两组样本之间的 MMD² (最大均值差异的平方)
    X: (n, d) 第一组样本
    Y: (m, d) 第二组样本
    sigma: RBF 核的带宽参数
    """
    n = X.size(0)
    m = Y.size(0)
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=X.device)
    gamma = 1.0 / (2.0 * sigma ** 2)
    kxx = torch.exp(-gamma * torch.cdist(X, X) ** 2)
    kyy = torch.exp(-gamma * torch.cdist(Y, Y) ** 2)
    kxy = torch.exp(-gamma * torch.cdist(X, Y) ** 2)
    mmd2 = kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
    return mmd2


# ============================================================
# 映射层 (Projection Layer)
# ============================================================

class ProjectionLayer(nn.Module):
    """
    线性映射层, 将特征/权重从一个维度映射到共享空间
    与 FitNet 的 ConvReg 类似, 但这里操作的是 1D 向量而非特征图
    """
    def __init__(self, in_dim, out_dim):
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


# ============================================================
# 工具函数: 从模型中提取分类头权重
# ============================================================

def get_classifier_weight(model):
    """
    从不同架构的模型中提取分类头权重矩阵
    返回: (num_classes, feat_dim) 的权重 tensor
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        return model.fc.weight
    elif hasattr(model, 'linear') and isinstance(model.linear, nn.Linear):
        return model.linear.weight
    elif hasattr(model, 'classifier'):
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
# ELOTClosedLoss — 闭集 ELOT 蒸馏损失 (核心模块)
# ============================================================

class ELOTClosedLoss(nn.Module):
    """
    闭集 ELOT 蒸馏损失

    与原始 ELOTLoss 的区别:
      - 教师有 K_t=100 类, 学生有 K_s=20 类
      - class_indices 指定学生对应教师的哪 20 个类
      - 计算代价矩阵前, 从教师 100 类中筛选出 20 类权重
      - OT 在 20×20 的代价矩阵上求解
      - MMD 只在 20 个类上计算 (标签已重映射到 0-19)

    整体流程:
      1. 筛选: w_t_full (100, t_dim) → w_t (20, t_dim)
      2. 映射: w_t → proj_t → (20, proj_dim)
              w_s → proj_s → (20, proj_dim)
      3. 权重距离: C_weight[k,l] = ||w_k^T - w_l^S||²  (20×20)
      4. MMD: C_mmd[k,l] = MMD²(P_T^(k), P_S^(l))      (20×20)
      5. 代价矩阵: C = C_weight - λ · C_mmd              (20×20)
      6. OT 求解: 在 20 维分布上求传输计划 π
      7. 传输损失: L = Σ π_{kl} · C_{kl}
    """

    def __init__(self, t_dim, s_dim, proj_dim, num_classes_student,
                 class_indices, lambda_mmd=0.1, epsilon=0.1, sigma=1.0,
                 warmup_iters=500, nb_dummies=1):
        """
        参数:
            t_dim: int, 教师模型的特征维度
                (分类头权重矩阵的列数, 如 resnet110 → 64)
            s_dim: int, 学生模型的特征维度
                (如 resnet20 → 64, resnet8x4 → 256)
            proj_dim: int, 映射后的共享空间维度
                (建议取 min(t_dim, s_dim) 或 128)
            num_classes_student: int, 学生的类别数 (=20)
            class_indices: list, 学生对应教师的原始类别编号
                如 [0, 1, ..., 19] 或 [40, 41, ..., 59]
                长度必须等于 num_classes_student
            lambda_mmd: float, MMD 奖励项的权重 λ
            epsilon: float, OT 的熵正则化系数
            sigma: float, RBF 核带宽 σ
            warmup_iters: int, 前多少次迭代用标准 OT (warmup)
            nb_dummies: int, ELOT 虚拟点数量
        """
        super(ELOTClosedLoss, self).__init__()

        # 检查参数合法性
        assert len(class_indices) == num_classes_student, \
            "class_indices 长度 ({}) 必须等于 num_classes_student ({})".format(
                len(class_indices), num_classes_student)

        # 两个映射层: 教师和学生各一个, 映射到同一 proj_dim 维空间
        # 这与 FitNet 的 ConvReg 思路类似: 通过可训练层对齐不同维度的特征
        # 区别在于 FitNet 用单个映射 (学生→教师), 我们用双映射 (各自→共享空间)
        self.proj_t = ProjectionLayer(t_dim, proj_dim)
        self.proj_s = ProjectionLayer(s_dim, proj_dim)

        # 保存超参数
        self.lambda_mmd = lambda_mmd
        self.epsilon = epsilon
        self.sigma = sigma
        self.warmup_iters = warmup_iters
        self.nb_dummies = nb_dummies
        self.num_classes_student = num_classes_student
        self.proj_dim = proj_dim

        # 保存类别索引 (注册为 buffer, 这样 .cuda() 时会自动搬到 GPU)
        # class_indices: 学生的第 k 类 对应 教师的第 class_indices[k] 类
        self.register_buffer(
            'class_indices',
            torch.LongTensor(class_indices)
        )

    def forward(self, feat_s, feat_t, w_s, w_t_full, target, current_iter):
        """
        计算闭集 ELOT 蒸馏损失

        参数:
            feat_s: (B, s_dim) 学生的特征向量 (FC 层之前, 保留梯度)
            feat_t: (B, t_dim) 教师的特征向量 (已 detach)
            w_s: (20, s_dim) 学生分类头权重 (保留梯度)
            w_t_full: (100, t_dim) 教师分类头的完整权重 (已 detach)
            target: (B,) 重映射后的标签, 值域 [0, 19]
            current_iter: int, 当前全局迭代步数 (用于 warmup 判断)

        返回:
            loss: 标量 tensor, 闭集 ELOT 传输损失
        """
        K = self.num_classes_student  # 20
        device = feat_s.device

        # ========== 步骤 1: 从教师 100 类中筛选出对应的 20 类权重 ==========
        # w_t_full: (100, t_dim), 教师的完整分类头权重
        # self.class_indices: (20,), 如 [0, 1, ..., 19]
        # w_t: (20, t_dim), 只保留对应 20 类的权重行
        w_t = w_t_full[self.class_indices].detach()  # 确保不传梯度给教师

        # ========== 步骤 2: 映射到共享空间 ==========
        w_t_proj = self.proj_t(w_t)    # (20, proj_dim), 教师映射后的权重
        w_s_proj = self.proj_s(w_s)    # (20, proj_dim), 学生映射后的权重, 保留梯度

        # ========== 步骤 3: 计算权重欧氏距离矩阵 ==========
        # C_weight[k,l] = ||w_k^T_proj - w_l^S_proj||²
        # 教师第 k 类和学生第 l 类在映射空间中的距离
        C_weight = torch.cdist(w_t_proj, w_s_proj) ** 2  # (20, 20)

        # ========== 步骤 4: 计算类条件 MMD 矩阵 ==========
        # 将特征也映射到共享空间
        with torch.no_grad():
            feat_t_proj = self.proj_t(feat_t)  # (B, proj_dim)
        feat_s_proj = self.proj_s(feat_s)      # (B, proj_dim)

        C_mmd = torch.zeros(K, K, device=device)

        # 只对当前 batch 中实际出现的类别计算 MMD
        # target 的值域是 [0, 19] (已重映射), 所以可以直接用作索引
        unique_labels = torch.unique(target)

        for k_label in unique_labels:
            mask_k = (target == k_label)
            feat_t_k = feat_t_proj[mask_k]

            for l_label in unique_labels:
                mask_l = (target == l_label)
                feat_s_l = feat_s_proj[mask_l]

                # MMD 只用于构建代价矩阵, detach 防止干扰映射层梯度
                C_mmd[k_label, l_label] = compute_rbf_mmd2(
                    feat_t_k.detach(), feat_s_l.detach(), self.sigma
                )

        # ========== 步骤 5: 构建动态代价矩阵 ==========
        # C = C_weight - λ · C_mmd
        C = C_weight - self.lambda_mmd * C_mmd

        # ========== 步骤 6: 求解最优传输 ==========
        # 均匀分布作为边际约束 (20 维)
        a = ot.unif(K)  # (20,)
        b = ot.unif(K)  # (20,)

        C_cpu = C.detach().cpu().numpy()

        # Warmup 策略: 前 warmup_iters 次用标准 OT, 之后用 ELOT
        if current_iter <= self.warmup_iters:
            # 标准 OT: 所有质量必须完全传输
            if self.epsilon == 0:
                pi = ot.emd(a, b, C_cpu)
            else:
                pi = ot.sinkhorn(a, b, C_cpu, reg=self.epsilon)
        else:
            # ELOT: 弹性传输, 允许部分质量不传输
            if self.epsilon == 0:
                pi = elot_emd(a, b, C_cpu, nb_dummies=self.nb_dummies)
            else:
                pi = elot_entropic(a, b, C_cpu, reg=self.epsilon,
                                   nb_dummies=self.nb_dummies)

        # ========== 步骤 7: 计算传输损失 ==========
        pi_tensor = torch.from_numpy(pi).float().to(device)
        loss = torch.sum(pi_tensor * C)

        return loss


# ============================================================
# ProjectionPretrainLoss — 映射层预训练损失 (FitNet 风格)
#
# 可选使用: 在正式蒸馏前, 先训练映射层使得
#   proj_t(w_t) ≈ proj_s(w_s) (权重对齐)
#   proj_t(feat_t) ≈ proj_s(feat_s) (特征对齐)
#
# 这与 FitNet 的 ConvReg 预训练思路一致:
#   FitNet: 先训练 regressor 让 reg(feat_s) ≈ feat_t
#   这里:   先训练 proj_t, proj_s 让映射后的空间对齐
#
# 用法: 在 train_elot_closed.py 的 pretrain_projections() 中调用
# ============================================================

class ProjectionPretrainLoss(nn.Module):
    """
    映射层预训练损失

    两部分:
      1. 权重对齐损失: MSE(proj_t(w_t[k]), proj_s(w_s[k])) 对应类别 k
         让教师第 k 类的映射后权重 ≈ 学生第 k 类的映射后权重
      2. 特征对齐损失: MSE(proj_t(feat_t), proj_s(feat_s)) 逐样本
         让同一张图片的教师和学生特征在映射空间中接近

    注意: 预训练时只优化映射层参数, 学生主干冻结
    """

    def __init__(self, weight_align_coeff=1.0, feat_align_coeff=1.0):
        """
        参数:
            weight_align_coeff: 权重对齐损失的权重
            feat_align_coeff:   特征对齐损失的权重
        """
        super(ProjectionPretrainLoss, self).__init__()
        self.weight_align_coeff = weight_align_coeff
        self.feat_align_coeff = feat_align_coeff
        self.mse = nn.MSELoss()

    def forward(self, w_t_proj, w_s_proj, feat_t_proj, feat_s_proj):
        """
        参数:
            w_t_proj: (K, proj_dim) 映射后的教师权重
            w_s_proj: (K, proj_dim) 映射后的学生权重
            feat_t_proj: (B, proj_dim) 映射后的教师特征
            feat_s_proj: (B, proj_dim) 映射后的学生特征

        返回:
            loss: 标量, 预训练总损失
        """
        # 权重对齐: 对应类别的映射后权重应该接近
        loss_weight = self.mse(w_s_proj, w_t_proj.detach())

        # 特征对齐: 同一张图片的映射后特征应该接近
        loss_feat = self.mse(feat_s_proj, feat_t_proj.detach())

        loss = self.weight_align_coeff * loss_weight + \
               self.feat_align_coeff * loss_feat

        return loss