#!/usr/bin/env python3
"""
Region 3 三模块阈值学习

三个模块独立学习阈值:
1. OOD 马氏距离阈值 (状态分布外检测)
2. 决策因子贡献度阈值 (逻辑一致性检测)
3. 激活链路汉明距离阈值 (神经元激活模式检测)

输入:
- /mnt/data/aloha_act_500_optimized/ (497 条 normal)
- /mnt/data/aloha_fault_trajectories/ (400 条 fault, 8 种×50)

输出:
- region3_thresholds.json (三个模块的最优阈值)
"""

import numpy as np, glob, json, pickle
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("="*70)
print("Region 3 三模块阈值学习")
print("="*70)

NORMAL_DIR = Path("/mnt/data/aloha_act_500_optimized")
FAULT_DIR = Path("/mnt/data/aloha_fault_trajectories")
OUTPUT_DIR = Path("/mnt/data/region3_training_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== 1. 数据加载 ==============
print("\n[1/5] 加载轨迹数据...")

def load_trajectories(directory):
    """加载所有轨迹"""
    files = glob.glob(str(directory / "*.npz"))
    trajectories = []
    for f in files:
        try:
            data = np.load(f)
            traj = {
                "states": data["states"],
                "actions": data["actions"],
                "gradients": data.get("gradients"),
                "hook_a": data.get("hook_a"),
                "hook_b": data.get("hook_b"),
                "fault_type": str(data.get("fault_type", "normal")),
            }
            trajectories.append(traj)
        except Exception as e:
            pass
    return trajectories

normal_trajs = load_trajectories(NORMAL_DIR)
fault_trajs = load_trajectories(FAULT_DIR)

print(f"Normal: {len(normal_trajs)} 条")
print(f"Fault: {len(fault_trajs)} 条")

# ============== 2. 特征提取 ==============
print("\n[2/5] 提取各模块特征...")

# ----- 模块 1: OOD 马氏距离 -----
print("\n  [2.1] OOD 马氏距离特征...")

# 从 normal 数据学习状态分布
normal_states = np.vstack([t["states"] for t in normal_trajs])
print(f"  Normal 状态样本：{len(normal_states)}")

# 使用鲁棒协方差估计
robust_cov = MinCovDet(random_state=42).fit(normal_states)
robust_mean = robust_cov.location_
robust_cov_matrix = robust_cov.covariance_

# 计算所有样本的马氏距离
def mahalanobis_score(states, mean, cov_inv):
    diff = states - mean
    left = np.dot(diff, cov_inv)
    mahal = np.sqrt(np.sum(left * diff, axis=1))
    return mahal

cov_inv = np.linalg.inv(robust_cov_matrix)

normal_mahal_scores = []
for t in normal_trajs:
    scores = mahalanobis_score(t["states"], robust_mean, cov_inv)
    normal_mahal_scores.append(np.max(scores))  # 取轨迹最大

fault_mahal_scores = []
for t in fault_trajs:
    scores = mahalanobis_score(t["states"], robust_mean, cov_inv)
    fault_mahal_scores.append(np.max(scores))

print(f"  Normal 马氏距离：mean={np.mean(normal_mahal_scores):.2f}, std={np.std(normal_mahal_scores):.2f}")
print(f"  Fault 马氏距离：mean={np.mean(fault_mahal_scores):.2f}, std={np.std(fault_mahal_scores):.2f}")

# ----- 模块 2: 决策因子贡献度 -----
print("\n  [2.2] 决策因子贡献度特征...")

def compute_logic_score(gradients, states):
    """
    计算逻辑一致性分数
    S_logic = Σ(φ_i for i in legal_features) / Σ(φ_j for all j)
    """
    if gradients is None or len(gradients) == 0:
        return 0.5
    
    # 梯度绝对值作为贡献度近似
    grad_norms = np.linalg.norm(gradients, axis=1)
    
    # 假设前 7 维是合法特征 (base_x, base_y, base_v, base_w, ee_x, ee_y, ee_z)
    # 后 7 维可能包含噪声
    if gradients.ndim == 2 and gradients.shape[1] >= 14:
        legal_grads = np.abs(gradients[:, :7])
        all_grads = np.abs(gradients)
        
        logic_scores = []
        for i in range(len(gradients)):
            legal_sum = np.sum(legal_grads[i])
            all_sum = np.sum(all_grads[i]) + 1e-8
            logic_scores.append(legal_sum / all_sum)
        
        return np.mean(logic_scores)
    else:
        return 0.5

normal_logic_scores = []
for t in normal_trajs:
    score = compute_logic_score(t.get("gradients"), t["states"])
    normal_logic_scores.append(score)

fault_logic_scores = []
for t in fault_trajs:
    score = compute_logic_score(t.get("gradients"), t["states"])
    fault_logic_scores.append(score)

print(f"  Normal 逻辑分数：mean={np.mean(normal_logic_scores):.3f}, std={np.std(normal_logic_scores):.3f}")
print(f"  Fault 逻辑分数：mean={np.mean(fault_logic_scores):.3f}, std={np.std(fault_logic_scores):.3f}")

# ----- 模块 3: 激活链路汉明距离 -----
print("\n  [2.3] 激活链路汉明距离特征...")

def extract_activation_mask(hook_a, hook_b):
    """从 hook 激活提取二值掩码"""
    if hook_a is None or hook_b is None:
        return None
    
    # 简化：取平均激活的二值化
    mask_a = (np.mean(hook_a, axis=(0, 1)) > 0).astype(float)
    mask_b = (np.mean(hook_b, axis=(0, 1)) > 0).astype(float)
    return np.concatenate([mask_a, mask_b])

# 从 normal 数据学习参考掩码库
print("  学习参考掩码库...")
normal_masks = []
for t in normal_trajs:
    mask = extract_activation_mask(t.get("hook_a"), t.get("hook_b"))
    if mask is not None:
        normal_masks.append(mask)

if len(normal_masks) > 0:
    normal_masks = np.array(normal_masks)
    # K-Means 聚类得到 K 个参考掩码
    from sklearn.cluster import KMeans
    K = min(10, len(normal_masks) // 50)  # 自适应 K
    if K >= 2:
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(normal_masks)
        reference_masks = kmeans.cluster_centers_
        print(f"  参考掩码库：{K} 个聚类中心")
    else:
        reference_masks = np.array([np.mean(normal_masks, axis=0)])
        print(f"  参考掩码库：1 个平均掩码")
else:
    reference_masks = None
    print("  ⚠️ 无 hook 数据，跳过掩码库学习")

# 计算汉明距离
def hamming_distance_to_library(mask, library):
    """计算掩码与库中所有掩码的最小汉明距离"""
    if library is None or mask is None:
        return 0.5
    
    distances = []
    for ref_mask in library:
        # 汉明距离 = 不同位的比例
        dist = np.mean(mask != (ref_mask > 0.5))
        distances.append(dist)
    
    return min(distances) if distances else 0.5

normal_link_scores = []
for t in normal_trajs:
    mask = extract_activation_mask(t.get("hook_a"), t.get("hook_b"))
    score = hamming_distance_to_library(mask, reference_masks)
    normal_link_scores.append(score)

fault_link_scores = []
for t in fault_trajs:
    mask = extract_activation_mask(t.get("hook_a"), t.get("hook_b"))
    score = hamming_distance_to_library(mask, reference_masks)
    fault_link_scores.append(score)

print(f"  Normal 链路距离：mean={np.mean(normal_link_scores):.3f}, std={np.std(normal_link_scores):.3f}")
print(f"  Fault 链路距离：mean={np.mean(fault_link_scores):.3f}, std={np.std(fault_link_scores):.3f}")

# ============== 3. ROC 曲线与阈值学习 ==============
print("\n[3/5] 学习最优阈值...")

def learn_threshold(normal_scores, fault_scores, module_name):
    """使用 Youden 指数学习最优阈值"""
    all_scores = normal_scores + fault_scores
    all_labels = [0]*len(normal_scores) + [1]*len(fault_scores)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Youden 指数
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    # 精准率 - 召回率
    prec, rec, prec_thresh = precision_recall_curve(all_labels, all_scores)
    
    print(f"\n  {module_name}:")
    print(f"    最优阈值：{optimal_threshold:.6f}")
    print(f"    检测率 (TPR): {optimal_tpr*100:.1f}%")
    print(f"    虚警率 (FPR): {optimal_fpr*100:.1f}%")
    print(f"    AUC-ROC: {roc_auc:.4f}")
    
    return {
        "threshold": float(optimal_threshold),
        "tpr": float(optimal_tpr),
        "fpr": float(optimal_fpr),
        "auc": float(roc_auc),
        "fpr_curve": fpr.tolist(),
        "tpr_curve": tpr.tolist(),
        "thresholds_curve": thresholds.tolist()
    }

# 学习三个模块的阈值
thresholds = {}

thresholds["ood_mahalanobis"] = learn_threshold(
    normal_mahal_scores, fault_mahal_scores,
    "模块 1: OOD 马氏距离"
)

thresholds["logic_consistency"] = learn_threshold(
    [-s for s in normal_logic_scores],  # 逻辑分数越低越异常
    [-s for s in fault_logic_scores],
    "模块 2: 决策因子贡献度"
)

if reference_masks is not None and len(normal_link_scores) > 0:
    thresholds["activation_link"] = learn_threshold(
        normal_link_scores, fault_link_scores,
        "模块 3: 激活链路汉明距离"
    )
else:
    thresholds["activation_link"] = {"threshold": 0.5, "tpr": 0.5, "fpr": 0.5, "auc": 0.5, "note": "no_hook_data"}

# ============== 4. 保存结果 ==============
print("\n[4/5] 保存结果...")

# 保存阈值配置
output_config = {
    "modules": {
        "ood_mahalanobis": {
            "weight": 0.35,
            "threshold": thresholds["ood_mahalanobis"]["threshold"],
            "tpr": thresholds["ood_mahalanobis"]["tpr"],
            "fpr": thresholds["ood_mahalanobis"]["fpr"],
            "auc": thresholds["ood_mahalanobis"]["auc"],
            "description": "马氏距离检测状态分布外"
        },
        "logic_consistency": {
            "weight": 0.35,
            "threshold": thresholds["logic_consistency"]["threshold"],
            "tpr": thresholds["logic_consistency"]["tpr"],
            "fpr": thresholds["logic_consistency"]["fpr"],
            "auc": thresholds["logic_consistency"]["auc"],
            "description": "梯度敏感性检测逻辑一致性"
        },
        "activation_link": {
            "weight": 0.30,
            "threshold": thresholds["activation_link"]["threshold"],
            "tpr": thresholds["activation_link"].get("tpr", 0.5),
            "fpr": thresholds["activation_link"].get("fpr", 0.5),
            "auc": thresholds["activation_link"].get("auc", 0.5),
            "description": "汉明距离检测激活模式异常"
        }
    },
    "fusion": {
        "method": "weighted_sum",
        "weights": [0.35, 0.35, 0.30],
        "risk_threshold": 0.5
    },
    "training_data": {
        "n_normal": len(normal_trajs),
        "n_fault": len(fault_trajs)
    }
}

with open(OUTPUT_DIR / "region3_thresholds.json", "w") as f:
    json.dump(output_config, f, indent=2)

# 保存参考统计 (用于在线检测)
ref_stats = {
    "robust_mean": robust_mean.tolist(),
    "robust_cov_inv": cov_inv.tolist(),
    "reference_masks": reference_masks.tolist() if reference_masks is not None else None,
    "K": K if reference_masks is not None else 0
}

with open(OUTPUT_DIR / "region3_reference_stats.pkl", "wb") as f:
    pickle.dump(ref_stats, f)

print(f"阈值配置：{OUTPUT_DIR}/region3_thresholds.json")
print(f"参考统计：{OUTPUT_DIR}/region3_reference_stats.pkl")

# ============== 5. 可视化 ==============
print("\n[5/5] 生成图表...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

module_names = ["ood_mahalanobis", "logic_consistency", "activation_link"]
titles = ["OOD 马氏距离", "决策因子贡献度", "激活链路汉明距离"]

for ax, module, title in zip(axes, module_names, titles):
    th = thresholds[module]
    if "note" in th:
        ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14)
        ax.set_title(f"{title}\n(无 hook 数据)")
        continue
    
    ax.plot(th["fpr_curve"], th["tpr_curve"], color='darkorange', lw=2, 
            label=f'AUC = {th["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 标记最优阈值点
    ax.scatter([th["fpr"]], [th["tpr"]], color='red', s=100, zorder=5,
               label=f'阈值={th["threshold"]:.4f}\nTPR={th["tpr"]:.1%}, FPR={th["fpr"]:.1%}')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{title}\n(AUC={th["auc"]:.3f})', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "region3_roc_curves.png", dpi=200, bbox_inches='tight')

print(f"ROC 曲线：{OUTPUT_DIR}/region3_roc_curves.png")

# ============== 6. 总结 ==============
print("\n" + "="*70)
print("✅ Region 3 三模块阈值学习完成!")
print("="*70)

print(f"\n各模块性能:")
for module, data in output_config["modules"].items():
    print(f"  {module}:")
    print(f"    阈值：{data['threshold']:.6f}")
    print(f"    TPR: {data['tpr']:.1f}%, FPR: {data['fpr']:.1f}%, AUC: {data['auc']:.4f}")

print(f"\n融合权重：OOD={output_config['fusion']['weights'][0]}, Logic={output_config['fusion']['weights'][1]}, Link={output_config['fusion']['weights'][2]}")
print(f"\n输出文件:")
print(f"  - {OUTPUT_DIR}/region3_thresholds.json")
print(f"  - {OUTPUT_DIR}/region3_reference_stats.pkl")
print(f"  - {OUTPUT_DIR}/region3_roc_curves.png")
print("="*70)
