#!/usr/bin/env python3
"""Region 3 三模块阈值学习 (简化快速版)"""

import numpy as np, glob, json, pickle
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.covariance import MinCovDet
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("="*70)
print("Region 3 三模块阈值学习 (简化版)")
print("="*70)

NORMAL_DIR = Path("/mnt/data/aloha_act_500_optimized")
FAULT_DIR = Path("/mnt/data/aloha_fault_trajectories")
OUTPUT_DIR = Path("/mnt/data/region3_training_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== 1. 采样加载数据 ==============
print("\n[1/4] 加载轨迹数据 (采样)...")

def load_trajectories_sample(directory, max_samples=100):
    """采样加载轨迹"""
    files = glob.glob(str(directory / "*.npz"))
    np.random.seed(42)
    sampled_files = np.random.choice(files, min(max_samples, len(files)), replace=False)
    
    trajectories = []
    for f in sampled_files:
        try:
            data = np.load(f)
            trajectories.append({
                "states": data["states"],
                "actions": data["actions"],
                "gradients": data.get("gradients"),
                "hook_a": data.get("hook_a"),
                "hook_b": data.get("hook_b"),
                "fault_type": str(data.get("fault_type", "normal")),
            })
        except:
            pass
    
    return trajectories

normal_trajs = load_trajectories_sample(NORMAL_DIR, 100)
fault_trajs = load_trajectories_sample(FAULT_DIR, 100)

print(f"Normal: {len(normal_trajs)} 条 (采样)")
print(f"Fault: {len(fault_trajs)} 条 (采样)")

# ============== 2. 模块 1: OOD 马氏距离 ==============
print("\n[2/4] 模块 1: OOD 马氏距离...")

normal_states = np.vstack([t["states"] for t in normal_trajs])
print(f"  Normal 状态样本：{len(normal_states)}")

robust_cov = MinCovDet(random_state=42).fit(normal_states)
robust_mean = robust_cov.location_
cov_inv = np.linalg.inv(robust_cov.covariance_)

def mahal_score(states, mean, cov_inv):
    diff = states - mean
    return np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))

normal_mahal = [np.max(mahal_score(t["states"], robust_mean, cov_inv)) for t in normal_trajs]
fault_mahal = [np.max(mahal_score(t["states"], robust_mean, cov_inv)) for t in fault_trajs]

print(f"  Normal: mean={np.mean(normal_mahal):.2f}")
print(f"  Fault: mean={np.mean(fault_mahal):.2f}")

# ============== 3. 模块 2: 决策因子贡献度 ==============
print("\n[3/4] 模块 2: 决策因子贡献度...")

def logic_score(traj):
    grads = traj.get("gradients")
    if grads is None or len(grads) == 0:
        return 0.5
    if grads.ndim == 2 and grads.shape[1] >= 14:
        legal = np.abs(grads[:, :7])
        all_g = np.abs(grads)
        return np.mean(np.sum(legal, axis=1) / (np.sum(all_g, axis=1) + 1e-8))
    return 0.5

normal_logic = [logic_score(t) for t in normal_trajs]
fault_logic = [logic_score(t) for t in fault_trajs]

print(f"  Normal: mean={np.mean(normal_logic):.3f}")
print(f"  Fault: mean={np.mean(fault_logic):.3f}")

# ============== 4. 模块 3: 激活链路 ==============
print("\n[4/4] 模块 3: 激活链路汉明距离...")

def extract_mask(hook_a, hook_b):
    if hook_a is None or hook_b is None:
        return None
    mask_a = (np.mean(hook_a, axis=(0, 1)) > 0).astype(float)
    mask_b = (np.mean(hook_b, axis=(0, 1)) > 0).astype(float)
    return np.concatenate([mask_a, mask_b])

# 采样计算掩码 (避免内存爆炸)
normal_masks = []
for t in normal_trajs[:50]:  # 只取 50 条
    mask = extract_mask(t.get("hook_a"), t.get("hook_b"))
    if mask is not None:
        normal_masks.append(mask)

if len(normal_masks) >= 10:
    K = 5
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(np.array(normal_masks))
    ref_masks = kmeans.cluster_centers_
    print(f"  参考掩码库：{K} 个聚类")
else:
    ref_masks = None
    print("  无足够 hook 数据")

def hamming_dist(mask, library):
    if library is None or mask is None:
        return 0.5
    return min([np.mean(mask != (ref > 0.5)) for ref in library])

normal_link = [hamming_dist(extract_mask(t.get("hook_a"), t.get("hook_b")), ref_masks) for t in normal_trajs]
fault_link = [hamming_dist(extract_mask(t.get("hook_a"), t.get("hook_b")), ref_masks) for t in fault_trajs]

print(f"  Normal: mean={np.mean(normal_link):.3f}")
print(f"  Fault: mean={np.mean(fault_link):.3f}")

# ============== 5. 阈值学习 ==============
print("\n" + "="*50)
print("最优阈值 (Youden 指数)")
print("="*50)

def learn(normal_scores, fault_scores, name, invert=False):
    all_scores = normal_scores + fault_scores
    all_labels = [0]*len(normal_scores) + [1]*len(fault_scores)
    
    if invert:
        all_scores = [-s for s in all_scores]
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    youden = tpr - fpr
    opt_idx = np.argmax(youden)
    opt_th = thresholds[opt_idx]
    opt_tpr = tpr[opt_idx]
    opt_fpr = fpr[opt_idx]
    
    print(f"\n{name}:")
    print(f"  阈值：{opt_th:.6f}")
    print(f"  TPR: {opt_tpr*100:.1f}%, FPR: {opt_fpr*100:.1f}%, AUC: {roc_auc:.4f}")
    
    return {"threshold": float(opt_th), "tpr": float(opt_tpr), "fpr": float(opt_fpr), "auc": float(roc_auc)}

th_ood = learn(normal_mahal, fault_mahal, "OOD 马氏距离")
th_logic = learn(normal_logic, fault_logic, "决策因子", invert=True)
th_link = learn(normal_link, fault_link, "激活链路") if ref_masks else {"threshold": 0.5, "tpr": 0.5, "fpr": 0.5, "auc": 0.5}

# ============== 6. 保存结果 ==============
print("\n保存结果...")

config = {
    "modules": {
        "ood_mahalanobis": {"weight": 0.35, **th_ood, "desc": "马氏距离检测状态分布外"},
        "logic_consistency": {"weight": 0.35, **th_logic, "desc": "梯度敏感性检测逻辑一致性"},
        "activation_link": {"weight": 0.30, **th_link, "desc": "汉明距离检测激活模式异常"}
    },
    "fusion": {"method": "weighted_sum", "weights": [0.35, 0.35, 0.30], "risk_threshold": 0.5},
    "reference_stats": {
        "robust_mean": robust_mean.tolist(),
        "cov_inv": cov_inv.tolist(),
        "ref_masks": ref_masks.tolist() if ref_masks is not None else None
    },
    "training_data": {"n_normal": len(normal_trajs), "n_fault": len(fault_trajs), "note": "sampled"}
}

with open(OUTPUT_DIR / "region3_thresholds.json", "w") as f:
    json.dump(config, f, indent=2)

with open(OUTPUT_DIR / "region3_reference_stats.pkl", "wb") as f:
    pickle.dump(config["reference_stats"], f)

# ROC 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
modules = [("ood_mahalanobis", "OOD 马氏距离", th_ood), 
           ("logic_consistency", "决策因子贡献度", th_logic),
           ("activation_link", "激活链路", th_link)]

for ax, (key, title, th) in zip(axes, modules):
    if th["auc"] == 0.5:
        ax.text(0.5, 0.5, "无数据", ha='center', va='center')
        ax.set_title(title)
        continue
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.scatter([th["fpr"]], [th["tpr"]], color='red', s=80, zorder=5)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'{title}\n(AUC={th["auc"]:.3f}, th={th["threshold"]:.4f})')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "region3_roc.png", dpi=150, bbox_inches='tight')

print(f"\n输出:")
print(f"  {OUTPUT_DIR}/region3_thresholds.json")
print(f"  {OUTPUT_DIR}/region3_reference_stats.pkl")
print(f"  {OUTPUT_DIR}/region3_roc.png")

print("\n" + "="*70)
print("✅ Region 3 三模块阈值学习完成!")
print("="*70)
