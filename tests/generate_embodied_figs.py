#!/usr/bin/env python3
"""
具身智能三层 RTA - 论文级图表生成
生成 8 张核心结果图
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
with open('/home/admin/.openclaw/workspace/Embodied-RTA/outputs/raw_data/all_trials.json', 'r') as f:
    trials = json.load(f)

df = pd.DataFrame(trials)
summary = pd.read_csv('/home/admin/.openclaw/workspace/Embodied-RTA/outputs/csv/methods_comparison_summary.csv')

output_dir = '/home/admin/.openclaw/workspace/Embodied-RTA/outputs/figures/final'
import os
os.makedirs(output_dir, exist_ok=True)

print(f"加载 {len(trials)} 次试验数据")
print(f"方法数：{df['method'].nunique()}")
print(f"场景数：{df['scenario'].nunique()}")

# ============ Fig 1: 成功率对比 (热力图) ============
print("生成 Fig 1: 成功率热力图...")
fig, ax = plt.subplots(figsize=(12, 8))

pivot_success = summary.pivot(index='scenario', columns='method', values='success_rate')
im = ax.imshow(pivot_success.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# 标注数值
for i in range(len(pivot_success.index)):
    for j in range(len(pivot_success.columns)):
        val = pivot_success.values[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

ax.set_xticks(range(len(pivot_success.columns)))
ax.set_yticks(range(len(pivot_success.index)))
ax.set_xticklabels(pivot_success.columns, rotation=45, ha='right')
ax.set_yticklabels(pivot_success.index)
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Scenario', fontsize=12)
ax.set_title('Embodied AI RTA: Success Rate by Method and Scenario (8×4×30=960 trials)', fontsize=14)

plt.colorbar(im, ax=ax, label='Success Rate')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig1_success_rate_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 2: 各方法平均成功率 (柱状图) ============
print("生成 Fig 2: 方法平均成功率...")
fig, ax = plt.subplots(figsize=(12, 6))

method_avg = summary.groupby('method')['success_rate'].mean().sort_values(ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(method_avg)))

bars = ax.barh(method_avg.index, method_avg.values * 100, color=colors)
ax.set_xlabel('Average Success Rate (%)', fontsize=12)
ax.set_title('Overall Performance Comparison (Across 4 Scenarios)', fontsize=14)
ax.set_xlim(0, 100)

# 标注数值
for bar, val in zip(bars, method_avg.values):
    ax.text(val * 100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%', va='center')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig2_method_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 3: 各场景成功率 (分组柱状图) ============
print("生成 Fig 3: 分场景对比...")
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(summary['scenario'].unique()))
width = 0.1
methods = summary['method'].unique()

for i, method in enumerate(methods):
    method_data = summary[summary['method'] == method]
    ax.bar(x + i * width, method_data['success_rate'].values * 100, width, label=method)

ax.set_xlabel('Scenario', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('Success Rate by Scenario and Method', fontsize=14)
ax.set_xticks(x + width * 3)
ax.set_xticklabels(summary['scenario'].unique())
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig3_success_by_scenario.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 4: 干预次数对比 ============
print("生成 Fig 4: 干预次数分析...")
fig, ax = plt.subplots(figsize=(12, 6))

# 过滤有干预的数据
intervention_data = summary[summary['avg_interventions'].notna() & (summary['avg_interventions'] > 0)]
if len(intervention_data) > 0:
    pivot_int = intervention_data.pivot(index='method', columns='scenario', values='avg_interventions')
    im = ax.imshow(pivot_int.values, cmap='YlOrRd', aspect='auto')
    
    for i in range(len(pivot_int.index)):
        for j in range(len(pivot_int.columns)):
            val = pivot_int.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8)
    
    ax.set_xticks(range(len(pivot_int.columns)))
    ax.set_yticks(range(len(pivot_int.index)))
    ax.set_xticklabels(pivot_int.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_int.index)
    ax.set_title('Average Interventions by Method and Scenario', fontsize=14)
    plt.colorbar(im, ax=ax, label='Avg Interventions')
    plt.tight_layout()

plt.savefig(f'{output_dir}/fig4_interventions_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 5: 计算时间对比 ============
print("生成 Fig 5: 计算效率...")
fig, ax = plt.subplots(figsize=(12, 6))

computation_data = summary.groupby('method')['computation_time'].mean().sort_values(ascending=True)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(computation_data)))

bars = ax.barh(computation_data.index, computation_data.values, color=colors)
ax.set_xlabel('Computation Time (ms)', fontsize=12)
ax.set_title('Computational Efficiency Comparison', fontsize=14)

for bar, val in zip(bars, computation_data.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}ms', va='center')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig5_computation_time.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 6: 成功率 - 效率权衡 (散点图) ============
print("生成 Fig 6: 成功率 - 效率权衡...")
fig, ax = plt.subplots(figsize=(10, 8))

method_stats = summary.groupby('method').agg({
    'success_rate': 'mean',
    'computation_time': 'mean'
}).reset_index()

for i, row in method_stats.iterrows():
    ax.scatter(row['computation_time'], row['success_rate'] * 100, s=200, alpha=0.7)
    ax.annotate(row['method'], (row['computation_time'], row['success_rate'] * 100), 
                fontsize=9, ha='center', va='bottom')

ax.set_xlabel('Computation Time (ms)', fontsize=12)
ax.set_ylabel('Average Success Rate (%)', fontsize=12)
ax.set_title('Trade-off: Success Rate vs Computational Cost', fontsize=14)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig6_tradeoff_scatter.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 7: 预警提前时间 ============
print("生成 Fig 7: 预警提前时间...")
fig, ax = plt.subplots(figsize=(12, 6))

leadtime_data = summary[summary['avg_lead_time'].notna() & (summary['avg_lead_time'] > 0)]
if len(leadtime_data) > 0:
    pivot_leadtime = leadtime_data.pivot(index='method', columns='scenario', values='avg_lead_time')
    im = ax.imshow(pivot_leadtime.values, cmap='YlGn', aspect='auto')
    
    for i in range(len(pivot_leadtime.index)):
        for j in range(len(pivot_leadtime.columns)):
            val = pivot_leadtime.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}s', ha='center', va='center', fontsize=8)
    
    ax.set_xticks(range(len(pivot_leadtime.columns)))
    ax.set_yticks(range(len(pivot_leadtime.index)))
    ax.set_xticklabels(pivot_leadtime.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_leadtime.index)
    ax.set_title('Warning Lead Time by Method and Scenario (seconds)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Lead Time (s)')
    plt.tight_layout()

plt.savefig(f'{output_dir}/fig7_lead_time_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 8: 综合雷达图 ============
print("生成 Fig 8: 综合性能雷达图...")
fig = plt.figure(figsize=(12, 10))

# 选择 Top 4 方法
top_methods = ['Ours_Full', 'CBF_Visual', 'R1_R2', 'Pure_VLA']
metrics = ['success_rate', 'avg_interventions', 'computation_time']

# 归一化
summary_norm = summary.copy()
for metric in metrics:
    if summary_norm[metric].max() > summary_norm[metric].min():
        summary_norm[f'{metric}_norm'] = (summary_norm[metric] - summary_norm[metric].min()) / (summary_norm[metric].max() - summary_norm[metric].min())
    else:
        summary_norm[f'{metric}_norm'] = 0.5

# 成功率越高越好，其他越低越好
summary_norm['score'] = (
    summary_norm['success_rate_norm'] * 0.5 + 
    (1 - summary_norm['avg_interventions_norm'].fillna(0)) * 0.25 +
    (1 - summary_norm['computation_time_norm']) * 0.25
)

method_scores = summary_norm.groupby('method')['score'].mean().sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(method_scores)))

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(method_scores.index, method_scores.values * 100, color=colors)
ax.set_xlabel('Composite Score (Success×0.5 + LowIntervention×0.25 + Efficiency×0.25)', fontsize=12)
ax.set_title('Overall Performance Ranking (Higher is Better)', fontsize=14)
ax.set_xlim(0, 100)

for bar, val in zip(bars, method_scores.values):
    ax.text(val * 100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig8_composite_ranking.png', dpi=200, bbox_inches='tight')
plt.close()

print("\n✅ 所有图表生成完成!")
print(f"输出目录：{output_dir}")

# 生成统计摘要
print("\n" + "="*70)
print("统计摘要")
print("="*70)
print(f"总试验次数：{len(trials)}")
print(f"方法数：{df['method'].nunique()}")
print(f"场景数：{df['scenario'].nunique()}")
print(f"每配置重复次数：30")
print(f"\n最佳方法：{method_scores.idxmax()} (综合得分：{method_scores.max()*100:.1f})")
print(f"最高成功率场景：{summary.groupby('scenario')['success_rate'].mean().idxmax()}")
print(f"最具挑战场景：{summary.groupby('scenario')['success_rate'].mean().idxmin()}")
