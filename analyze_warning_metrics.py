#!/usr/bin/env python3
"""
消融实验预警性能分析

计算预警指标:
- Precision (准确率) = TP / (TP + FP)
- Recall (召回率) = TP / (TP + FN)
- FPR (虚警率) = FP / (FP + TN)
- F1-Score
- ROC 曲线
"""

import csv, json, numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("/home/vipuser/Embodied-RTA/outputs/ablation")
FIG_DIR = OUTPUT_DIR / "figures_warning"
FIG_DIR.mkdir(exist_ok=True)

# 加载数据
with open(OUTPUT_DIR / "results.csv") as f:
    rows = list(csv.DictReader(f))

print(f'加载 {len(rows)} 条试验记录...')
print('='*70)

# 定义 Ground Truth
# 故障场景 (F1-F8) = 正样本 (应该预警)
# 基础场景 (B1-B3) = 负样本 (不应该预警)
def is_fault_scene(scene):
    return scene.startswith('F')

# 按配置分析预警性能
configs = ["A0_Pure_VLA", "A1_R1_Only", "A2_R2_Only", "A3_R3_Only", 
           "A4_R1_R2", "A5_R1_R3", "A6_R2_R3", "A7_Full"]

print("\n预警性能分析 (按配置)")
print('='*70)

all_results = {}

for cfg in configs:
    subset = [r for r in rows if r['config'] == cfg]
    
    # 准备标签和预测
    y_true = []  # 1=故障 (应该预警), 0=正常 (不应该预警)
    y_pred = []  # 1=预警，0=无预警
    y_scores = []  # 风险分数用于 ROC
    
    for r in subset:
        is_fault = is_fault_scene(r['scene'])
        warning = (r['warning'] == 'True')
        
        y_true.append(1 if is_fault else 0)
        y_pred.append(1 if warning else 0)
        
        # 使用风险分数 (R3 为主，如果启用)
        if r['r3'] == 'True':
            y_scores.append(float(r['risk_r3']))
        elif r['r2'] == 'True':
            y_scores.append(float(r['risk_r2']))
        elif r['r1'] == 'True':
            y_scores.append(float(r['risk_r1']))
        else:
            y_scores.append(0.0)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 等同于 TPR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROC 曲线
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_curve, tpr_curve)
    
    # PR 曲线
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
    
    all_results[cfg] = {
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'fnr': fnr,
        'f1': f1,
        'roc_auc': roc_auc,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'fpr_curve': fpr_curve.tolist(),
        'tpr_curve': tpr_curve.tolist(),
        'prec_curve': prec_curve.tolist(),
        'rec_curve': rec_curve.tolist(),
    }
    
    # 打印结果
    r1 = "✓" if any(r['r1']=='True' for r in subset[:1]) else "✗"
    r2 = "✓" if any(r['r2']=='True' for r in subset[:1]) else "✗"
    r3 = "✓" if any(r['r3']=='True' for r in subset[:1]) else "✗"
    
    print(f"\n{cfg} (R1:{r1} R2:{r2} R3:{r3}):")
    print(f"  混淆矩阵：TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  准确率 (Precision): {precision*100:.1f}%")
    print(f"  召回率 (Recall): {recall*100:.1f}%")
    print(f"  虚警率 (FPR): {fpr*100:.1f}%")
    print(f"  漏检率 (FNR): {fnr*100:.1f}%")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {roc_auc:.4f}")

# 保存汇总结果
summary_file = OUTPUT_DIR / "warning_metrics_summary.json"
with open(summary_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\n保存汇总：{summary_file}")

# ============== 可视化 ==============
print("\n生成图表...")

# 图 1: ROC 曲线对比
plt.figure(figsize=(10, 8))
colors = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#c0392b']
for i, cfg in enumerate(configs):
    res = all_results[cfg]
    plt.plot(res['fpr_curve'], res['tpr_curve'], 
             color=colors[i], lw=2, 
             label=f"{cfg.replace('A', 'A').replace('_', ' ')} (AUC={res['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测')
plt.xlabel('虚警率 (FPR)', fontsize=12)
plt.ylabel('召回率 (TPR)', fontsize=12)
plt.title('预警性能：ROC 曲线对比', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_roc_comparison.png', dpi=200, bbox_inches='tight')
print('保存：fig_roc_comparison.png')

# 图 2: Precision-Recall 曲线对比
plt.figure(figsize=(10, 8))
for i, cfg in enumerate(configs):
    res = all_results[cfg]
    plt.plot(res['rec_curve'], res['prec_curve'], 
             color=colors[i], lw=2, 
             label=f"{cfg.replace('_', ' ')}")

plt.xlabel('召回率 (Recall)', fontsize=12)
plt.ylabel('准确率 (Precision)', fontsize=12)
plt.title('预警性能：Precision-Recall 曲线对比', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_pr_comparison.png', dpi=200, bbox_inches='tight')
print('保存：fig_pr_comparison.png')

# 图 3: 指标对比柱状图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['precision', 'recall', 'fpr', 'f1']
titles = ['准确率 (Precision)', '召回率 (Recall)', '虚警率 (FPR)', 'F1-Score']
ylims = [(0, 100), (0, 100), (0, 100), (0, 100)]

for ax, metric, title, ylim in zip(axes.flat, metrics, titles, ylims):
    values = [all_results[cfg][metric] * 100 for cfg in configs]
    bars = ax.bar(range(len(configs)), values, color=colors)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'], fontsize=9)
    ax.set_ylabel('百分比 (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(ylim)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_metrics_bar.png', dpi=200, bbox_inches='tight')
print('保存：fig_metrics_bar.png')

# 图 4: AUC-ROC 对比
plt.figure(figsize=(10, 6))
auc_values = [all_results[cfg]['roc_auc'] * 100 for cfg in configs]
bars = plt.bar(range(len(configs)), auc_values, color=colors)
plt.xticks(range(len(configs)), ['A0\nPure', 'A1\nR1', 'A2\nR2', 'A3\nR3', 
                                  'A4\nR1+R2', 'A5\nR1+R3', 'A6\nR2+R3', 'A7\nFull'], fontsize=9)
plt.ylabel('AUC-ROC (%)', fontsize=12)
plt.title('预警性能：AUC-ROC 对比', fontsize=14, fontweight='bold')
plt.ylim([0, 100])
plt.grid(axis='y', alpha=0.3)
for bar, auc in zip(bars, auc_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{auc:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_auc_comparison.png', dpi=200, bbox_inches='tight')
print('保存：fig_auc_comparison.png')

print()
print('='*70)
print('预警性能分析完成!')
print('输出目录:', FIG_DIR)
print('='*70)
