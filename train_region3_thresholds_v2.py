#!/usr/bin/env python3
"""
Region 3 阈值学习 v2 - 使用多种特征

特征:
1. 状态标准差 (正常轨迹更平滑)
2. 动作标准差
3. 状态变化率
4. 动作变化率
"""

import numpy as np, glob, json
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("="*70)
print("Region 3 阈值学习 v2 (多特征)")
print("="*70)

NORMAL_DIR = Path("/mnt/data/aloha_act_500_optimized")
FAULT_DIR = Path("/mnt/data/aloha_fault_trajectories")
OUTPUT_DIR = Path("/mnt/data/region3_training_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== 1. 特征提取 ==============
print("\n[1/4] 提取特征...")

def extract_features(filepath):
    """从单条轨迹提取特征"""
    data = np.load(filepath)
    states = data["states"]
    actions = data["actions"]
    
    # 特征 1-2: 状态/动作标准差
    state_std = np.std(states)
    action_std = np.std(actions)
    
    # 特征 3-4: 状态/动作变化率 (差分)
    state_diff = np.diff(states, axis=0)
    action_diff = np.diff(actions, axis=0)
    state_diff_std = np.std(state_diff)
    action_diff_std = np.std(action_diff)
    
    # 特征 5-6: 状态/动作最大值
    state_max = np.max(np.abs(states))
    action_max = np.max(np.abs(actions))
    
    # 特征 7-8: 状态/动作均值
    state_mean = np.mean(states)
    action_mean = np.mean(actions)
    
    # 特征 9-10: 故障后变化 (50 步后)
    if len(states) > 100:
        state_before = states[:50]
        state_after = states[50:]
        state_shift = np.abs(np.mean(state_after) - np.mean(state_before))
        action_shift = np.abs(np.mean(actions[50:]) - np.mean(actions[:50]))
    else:
        state_shift = 0
        action_shift = 0
    
    return [state_std, action_std, state_diff_std, action_diff_std, 
            state_max, action_max, state_mean, action_mean, state_shift, action_shift]

def load_features(directory, label):
    files = glob.glob(str(directory / "*.npz"))
    features = []
    for i, f in enumerate(files):
        try:
            feat = extract_features(f)
            features.append(feat)
        except Exception as e:
            pass
        if (i+1) % 100 == 0:
            print(f"  {label}: {i+1}/{len(files)}")
    print(f"  ✅ {label}: {len(features)} 条")
    return np.array(features)

X_normal = load_features(NORMAL_DIR, "Normal")
X_fault = load_features(FAULT_DIR, "Fault")

y_normal = np.zeros(len(X_normal))
y_fault = np.ones(len(X_fault))

X = np.vstack([X_normal, X_fault])
y = np.concatenate([y_normal, y_fault])

print(f"\n特征矩阵：{X.shape}")
print(f"标签：{len(y_normal)} normal + {len(y_fault)} fault")

# ============== 2. 随机森林分类 ==============
print("\n[2/4] 训练随机森林...")

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X, y)

# 交叉验证
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"交叉验证准确率：{cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# 特征重要性
feature_names = ["state_std", "action_std", "state_diff_std", "action_diff_std",
                 "state_max", "action_max", "state_mean", "action_mean", 
                 "state_shift", "action_shift"]
importances = rf.feature_importances_

print("\n特征重要性:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name:15s}: {imp:.4f}")

# ============== 3. ROC 曲线 ==============
print("\n[3/4] ROC 曲线分析...")

# 使用随机森林的预测概率作为分数
scores = rf.predict_proba(X)[:, 1]

fpr, tpr, thresholds = roc_curve(y, scores)
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
print(f"阈值：{optimal_threshold:.6f} (RF 概率)")
print(f"检测率 (TPR): {optimal_tpr*100:.1f}%")
print(f"虚警率 (FPR): {optimal_fpr*100:.1f}%")
print(f"AUC-ROC: {roc_auc:.4f}")

if optimal_tpr >= 0.90 and optimal_fpr <= 0.05:
    print(f"\n✅ 达到目标！(TPR≥90%, FPR≤5%)")
elif optimal_tpr >= 0.80 and optimal_fpr <= 0.10:
    print(f"\n✅ 良好性能 (TPR≥80%, FPR≤10%)")
elif roc_auc >= 0.7:
    print(f"\n⚠️ 中等区分度，需要更多数据")
else:
    print(f"\n⚠️ 区分度不足")

# ============== 4. 保存结果 ==============
print("\n[4/4] 保存结果...")

thresholds_data = {
    "method": "RandomForest",
    "optimal_threshold": float(optimal_threshold),
    "tpr": float(optimal_tpr),
    "fpr": float(optimal_fpr),
    "auc_roc": float(roc_auc),
    "cv_accuracy": float(cv_scores.mean()),
    "feature_importances": dict(zip(feature_names, [float(x) for x in importances])),
    "n_normal": len(X_normal),
    "n_fault": len(X_fault)
}

with open(OUTPUT_DIR / "adaptive_thresholds_v2.json", "w") as f:
    json.dump(thresholds_data, f, indent=2)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].scatter([optimal_fpr], [optimal_tpr], color='red', s=100, zorder=5)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title(f'ROC Curve (AUC={roc_auc:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 特征重要性
axes[1].barh(feature_names, importances)
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importances')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "region3_threshold_v2.png", dpi=200, bbox_inches='tight')

print(f"阈值：{OUTPUT_DIR}/adaptive_thresholds_v2.json")
print(f"图表：{OUTPUT_DIR}/region3_threshold_v2.png")

print("\n" + "="*70)
print("✅ Region 3 阈值学习 v2 完成!")
print("="*70)
