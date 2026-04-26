#!/usr/bin/env python3
"""
Region 3 阈值学习 - 使用正常 + 故障数据

输入:
- /mnt/data/aloha_act_500_optimized/ (497 条 normal)
- /mnt/data/aloha_fault_trajectories/ (400 条 fault, 8 种×50)

输出:
- adaptive_thresholds.json (最优阈值 + ROC 曲线)
"""

import numpy as np, glob, json
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("="*70)
print("Region 3 阈值学习 (正常 + 故障数据)")
print("="*70)

NORMAL_DIR = Path("/mnt/data/aloha_act_500_optimized")
FAULT_DIR = Path("/mnt/data/aloha_fault_trajectories")
OUTPUT_DIR = Path("/mnt/data/region3_training_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== 1. 数据加载 ==============
print("\n[1/4] 加载轨迹数据...")

def load_scores(directory, label):
    """从轨迹中提取异常分数（梯度范数）"""
    scores = []
    files = glob.glob(str(directory / "*.npz"))
    for i, f in enumerate(files):
        try:
            data = np.load(f)
            if "gradients" in data.files:
                grads = data["gradients"]
                # 计算每条轨迹的平均梯度范数
                traj_score = np.mean([np.linalg.norm(g) for g in grads])
                scores.append(traj_score)
        except Exception as e:
            pass
        if (i+1) % 100 == 0:
            print(f"  {label}: {i+1}/{len(files)}")
    print(f"  ✅ {label}: {len(scores)} 条轨迹")
    return scores

normal_scores = load_scores(NORMAL_DIR, "Normal")
fault_scores = load_scores(FAULT_DIR, "Fault")

print(f"\n总计：{len(normal_scores)} normal + {len(fault_scores)} fault = {len(normal_scores)+len(fault_scores)} 条")

# ============== 2. ROC 曲线分析 ==============
print("\n[2/4] ROC 曲线分析...")

# 准备数据
all_scores = normal_scores + fault_scores
all_labels = [0]*len(normal_scores) + [1]*len(fault_scores)

# 计算 ROC
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

# Youden 指数最优阈值
youden = tpr - fpr
optimal_idx = np.argmax(youden)
optimal_threshold = thresholds[optimal_idx]
optimal_tpr = tpr[optimal_idx]
optimal_fpr = fpr[optimal_idx]

print(f"\n{'='*50}")
print("最优阈值 (Youden 指数最大化)")
print(f"{'='*50}")
print(f"阈值：{optimal_threshold:.6f}")
print(f"检测率 (TPR): {optimal_tpr*100:.1f}%")
print(f"虚警率 (FPR): {optimal_fpr*100:.1f}%")
print(f"Youden 指数：{youden[optimal_idx]:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# 检查是否达到目标 (TPR>90%, FPR<5%)
if optimal_tpr >= 0.90 and optimal_fpr <= 0.05:
    print(f"\n✅ 达到目标！(TPR≥90%, FPR≤5%)")
elif optimal_tpr >= 0.85:
    print(f"\n⚠️ 接近目标，检测率良好")
else:
    print(f"\n⚠️ 检测率偏低，可能需要更多故障数据")

# ============== 3. 保存阈值 ==============
print("\n[3/4] 保存结果...")

thresholds_data = {
    "optimal_threshold": float(optimal_threshold),
    "tpr": float(optimal_tpr),
    "fpr": float(optimal_fpr),
    "youden_index": float(youden[optimal_idx]),
    "auc_roc": float(roc_auc),
    "n_normal": len(normal_scores),
    "n_fault": len(fault_scores),
    "fault_types": ["F1_lighting", "F2_occlusion", "F3_adversarial", "F4_payload", 
                    "F5_friction", "F6_dynamic", "F7_sensor", "F8_compound"]
}

with open(OUTPUT_DIR / "adaptive_thresholds.json", "w") as f:
    json.dump(thresholds_data, f, indent=2)

print(f"阈值：{OUTPUT_DIR}/adaptive_thresholds.json")

# ============== 4. 可视化 ==============
print("\n[4/4] 生成图表...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC 曲线
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
axes[0].scatter([optimal_fpr], [optimal_tpr], color='red', s=100, zorder=5, 
                label=f'最优阈值\nTPR={optimal_tpr:.1%}, FPR={optimal_fpr:.1%}')
axes[0].axhline(y=0.90, color='green', linestyle=':', alpha=0.5, label='目标 TPR≥90%')
axes[0].axvline(x=0.05, color='green', linestyle=':', alpha=0.5, label='目标 FPR≤5%')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('Region 3 ROC Curve - Gradient Norm Detection', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1.05])

# 分数分布
axes[1].hist(normal_scores, bins=50, alpha=0.6, label=f'Normal (n={len(normal_scores)})', color='green')
axes[1].hist(fault_scores, bins=50, alpha=0.6, label=f'Fault (n={len(fault_scores)})', color='red')
axes[1].axvline(x=optimal_threshold, color='blue', linestyle='--', lw=2, 
                label=f'最优阈值 = {optimal_threshold:.4f}')
axes[1].set_xlabel('异常分数 (梯度范数)', fontsize=12)
axes[1].set_ylabel('样本数', fontsize=12)
axes[1].set_title('正常 vs 故障 分数分布', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "region3_threshold_analysis.png", dpi=200, bbox_inches='tight')
print(f"图表：{OUTPUT_DIR}/region3_threshold_analysis.png")

# ============== 5. 总结 ==============
print("\n" + "="*70)
print("✅ Region 3 阈值学习完成!")
print("="*70)
print(f"\n关键结果:")
print(f"  最优阈值：{optimal_threshold:.6f}")
print(f"  检测率：{optimal_tpr*100:.1f}% (目标≥90%)")
print(f"  虚警率：{optimal_fpr*100:.1f}% (目标≤5%)")
print(f"  AUC-ROC: {roc_auc:.4f}")
print(f"\n输出文件:")
print(f"  - {OUTPUT_DIR}/adaptive_thresholds.json")
print(f"  - {OUTPUT_DIR}/region3_threshold_analysis.png")
print("="*70)
