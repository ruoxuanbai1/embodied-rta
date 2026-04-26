#!/usr/bin/env python3
"""Region 3 训练：动作聚类 + 掩码库 + CFS + 自适应阈值 (优化版 - 惰性加载)"""

import numpy as np, glob, json, time, os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("="*80)
print("Region 3 训练：动作聚类 + 掩码库 + CFS + 自适应阈值 (优化版)")
print("="*80)

TRAJECTORY_DIR = Path("/mnt/data/aloha_act_500_optimized")
OUTPUT_DIR = Path("/mnt/data/region3_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== 1. 数据加载 (惰性) ==============
print("\n[1/6] 加载轨迹数据...")
traj_files = sorted(glob.glob(str(TRAJECTORY_DIR / "traj_*.npz")))
print(f"找到 {len(traj_files)} 条轨迹文件")

# 先加载少量轨迹用于聚类分析
print("加载前 100 条轨迹进行聚类分析...")
trajectories = []
for i, f in enumerate(traj_files[:100]):
    try:
        data = np.load(f)
        if "states" in data.files and "actions" in data.files:
            trajectories.append({
                "states": data["states"],
                "actions": data["actions"],
                "gradients": data.get("gradients"),
                "hook_a": data.get("hook_a"),
                "hook_b": data.get("hook_b"),
                "fault_type": str(data.get("fault_type", "normal")),
            })
        if (i+1) % 20 == 0:
            print(f"  已加载 {i+1}/100")
    except Exception as e:
        print(f"跳过：{f} - {e}")

print(f"✅ 成功加载 {len(trajectories)} 条轨迹")

# ============== 2. 动作聚类 ==============
print("\n[2/6] 动作空间聚类...")
all_actions = np.vstack([t["actions"] for t in trajectories])
print(f"总动作样本：{len(all_actions)}")

# 快速聚类 (K=20)
print("执行 K-Means 聚类 (K=20)...")
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10, max_iter=100)
action_labels = kmeans.fit_predict(all_actions)
print(f"✅ 聚类完成，轮廓系数：{silhouette_score(all_actions[::100], action_labels[::100]):.4f}")

# 保存聚类中心
np.save(OUTPUT_DIR/"action_centers.npy", kmeans.cluster_centers_)
print(f"聚类中心：{OUTPUT_DIR}/action_centers.npy")

# ============== 3. 简化的掩码库学习 ==============
print("\n[3/6] 学习激活掩码库...")

# 检查是否有 hook 数据
if trajectories[0].get("hook_a") is not None:
    print("检测到 hook 数据，学习掩码库...")
    
    # 对每个聚类，收集激活模式
    mask_library = []
    for cluster_id in range(20):
        cluster_masks = []
        count = 0
        for i, label in enumerate(action_labels):
            if label == cluster_id and count < 50:  # 每个聚类取 50 个样本
                traj_idx = i // max(1, len(trajectories[0]["actions"]))
                step_idx = i % max(1, len(trajectories[0]["actions"]))
                if traj_idx < len(trajectories) and trajectories[traj_idx].get("hook_a") is not None:
                    hook_a = trajectories[traj_idx]["hook_a"]
                    if len(hook_a) > step_idx:
                        mask = (hook_a[step_idx] > 0).astype(np.float32)
                        cluster_masks.append(mask.flatten())
                        count += 1
        
        if cluster_masks:
            cluster_mean = np.mean(cluster_masks, axis=0)
            mask_library.append({"cluster": cluster_id, "mask_mean": cluster_mean.tolist(), "samples": len(cluster_masks)})
    
    with open(OUTPUT_DIR/"mask_library.json", "w") as f:
        json.dump(mask_library, f)
    print(f"✅ 掩码库：{OUTPUT_DIR}/mask_library.json ({len(mask_library)} 个聚类)")
else:
    print("⚠️ 无 hook 数据，跳过掩码库学习")

# ============== 4. CFS 关键特征集 ==============
print("\n[4/6] 学习 CFS 关键特征集...")

if trajectories[0].get("gradients") is not None:
    print("检测到梯度数据，学习 CFS...")
    
    cfs_features = {}
    for cluster_id in range(20):
        cluster_grads = []
        for i, label in enumerate(action_labels):
            if label == cluster_id:
                traj_idx = i // max(1, len(trajectories[0]["actions"]))
                step_idx = i % max(1, len(trajectories[0]["actions"]))
                if traj_idx < len(trajectories) and trajectories[traj_idx].get("gradients") is not None:
                    grad = trajectories[traj_idx]["gradients"]
                    if len(grad) > step_idx:
                        cluster_grads.append(np.abs(grad[step_idx]))
        
        if cluster_grads:
            mean_grad = np.mean(cluster_grads, axis=0)
            top_features = np.argsort(mean_grad)[-20:].tolist()  # Top 20 特征
            cfs_features[str(cluster_id)] = {"top_features": top_features, "mean_grad": mean_grad.tolist()}
    
    with open(OUTPUT_DIR/"cfs_features.json", "w") as f:
        json.dump(cfs_features, f)
    print(f"✅ CFS 特征集：{OUTPUT_DIR}/cfs_features.json")
else:
    print("⚠️ 无梯度数据，跳过 CFS 学习")

# ============== 5. 自适应阈值 ==============
print("\n[5/6] 学习自适应阈值...")

# 使用梯度范数作为异常分数
normal_scores, fault_scores = [], []
for t in trajectories:
    if t.get("gradients") is not None:
        grads = t["gradients"]
        scores = [np.linalg.norm(g) for g in grads]
        if "normal" in t.get("fault_type", "normal"):
            normal_scores.extend(scores)
        else:
            fault_scores.extend(scores)

if normal_scores and fault_scores:
    print(f"正常样本：{len(normal_scores)}, 故障样本：{len(fault_scores)}")
    
    # ROC 曲线分析
    all_scores = normal_scores + fault_scores
    all_labels = [0]*len(normal_scores) + [1]*len(fault_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n✅ 最优阈值：{optimal_threshold:.4f}")
    print(f"   检测率 (TPR): {tpr[optimal_idx]*100:.1f}%")
    print(f"   虚警率 (FPR): {fpr[optimal_idx]*100:.1f}%")
    
    # 保存阈值
    with open(OUTPUT_DIR/"adaptive_thresholds.json", "w") as f:
        json.dump({
            "optimal_threshold": float(optimal_threshold),
            "tpr": float(tpr[optimal_idx]),
            "fpr": float(fpr[optimal_idx]),
            "youden_index": float(youden[optimal_idx])
        }, f, indent=2)
    
    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Gradient Norm Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR/"roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ROC 曲线：{OUTPUT_DIR}/roc_curve.png")
else:
    print("⚠️ 数据不足，使用默认阈值")
    with open(OUTPUT_DIR/"adaptive_thresholds.json", "w") as f:
        json.dump({"optimal_threshold": 1.0, "note": "default"}, f)

# ============== 6. 总结 ==============
print("\n[6/6] 训练总结")
print("="*50)
print("✅ Region 3 训练完成!")
print(f"\n输出文件:")
print(f"  - {OUTPUT_DIR}/action_centers.npy (动作聚类中心)")
print(f"  - {OUTPUT_DIR}/mask_library.json (激活掩码库)")
print(f"  - {OUTPUT_DIR}/cfs_features.json (CFS 关键特征)")
print(f"  - {OUTPUT_DIR}/adaptive_thresholds.json (自适应阈值)")
print(f"  - {OUTPUT_DIR}/roc_curve.png (ROC 曲线)")
