#!/usr/bin/env python3
"""
具身智能三层 RTA - 扩展版论文级图表生成
8 场景 × 13 方法 × 30 次 = 3120 次试验
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('/home/admin/.openclaw/workspace/Embodied-RTA/outputs/csv/all_trials_extended.csv')
summary = pd.read_csv('/home/admin/.openclaw/workspace/Embodied-RTA/outputs/csv/methods_comparison_extended.csv')

# 转换成功率
summary['success_rate'] = summary['success']

output_dir = '/home/admin/.openclaw/workspace/Embodied-RTA/outputs/figures/final'
import os
os.makedirs(output_dir, exist_ok=True)

print(f"加载 {len(df)} 次试验数据")
print(f"方法数：{df['method'].nunique()}")
print(f"场景数：{df['scenario'].nunique()}")

# 场景名称映射
SCENARIO_NAMES = {
    's1_lighting_drop': 'S1: Lighting Drop',
    's2_camera_occlusion': 'S2: Camera Occlusion',
    's3_adversarial_patch': 'S3: Adversarial Patch',
    's4_payload_shift': 'S4: Payload Shift',
    's5_joint_friction': 'S5: Joint Friction',
    's6_dynamic_crowd': 'S6: Dynamic Crowd',
    's7_narrow_corridor': 'S7: Narrow Corridor',
    's8_compound_hell': 'S8: Compound Hell',
}

# 方法名称映射
METHOD_NAMES = {
    'Pure_VLA': 'Pure VLA',
    'R1_Only': 'R1 Only',
    'R2_Only': 'R2 Only',
    'R3_Only': 'R3 Only',
    'R1_R2': 'R1+R2',
    'R1_R3': 'R1+R3',
    'R2_R3': 'R2+R3',
    'Ours_Full': 'Ours (Full)',
    'DeepReach': 'DeepReach',
    'Recovery_RL': 'Recovery RL',
    'PETS': 'PETS',
    'CBF_QP': 'CBF-QP',
    'Shielded_RL': 'Shielded RL',
}

# ============ Fig 1: 成功率热力图 (8×13) ============
print("生成 Fig 1: 成功率热力图...")
fig, ax = plt.subplots(figsize=(16, 10))

pivot_success = summary.pivot(index='scenario', columns='method', values='success_rate')
pivot_success.index = [SCENARIO_NAMES.get(x, x) for x in pivot_success.index]
pivot_success.columns = [METHOD_NAMES.get(x, x) for x in pivot_success.columns]

im = ax.imshow(pivot_success.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# 标注数值
for i in range(len(pivot_success.index)):
    for j in range(len(pivot_success.columns)):
        val = pivot_success.values[i, j]
        color = 'white' if val > 0.5 else 'black'
        fontsize = 7 if val > 0.8 else 6
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=fontsize)

ax.set_xticks(range(len(pivot_success.columns)))
ax.set_yticks(range(len(pivot_success.index)))
ax.set_xticklabels(pivot_success.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(pivot_success.index, fontsize=9)
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Scenario', fontsize=12)
ax.set_title('Embodied AI RTA: Success Rate Heatmap (13 Methods × 8 Scenarios × 30 Runs = 3120 Trials)', fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax, label='Success Rate')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig1_success_rate_heatmap_extended.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 2: 方法平均成功率排名 ============
print("生成 Fig 2: 方法平均成功率...")
fig, ax = plt.subplots(figsize=(14, 8))

method_avg = summary.groupby('method')['success_rate'].mean().sort_values(ascending=True)
method_avg.index = [METHOD_NAMES.get(x, x) for x in method_avg.index]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(method_avg)))

bars = ax.barh(method_avg.index, method_avg.values * 100, color=colors)
ax.set_xlabel('Average Success Rate (%)', fontsize=12)
ax.set_title('Overall Performance Comparison (Across 8 Scenarios)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)

# 标注数值
for bar, val in zip(bars, method_avg.values):
    ax.text(val * 100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig2_method_comparison_extended.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 3: 分场景成功率 (分组柱状图 - 简化版 Top5) ============
print("生成 Fig 3: 分场景对比...")

# 为每个场景生成单独的图
for scenario in df['scenario'].unique():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenario_data = summary[summary['scenario'] == scenario]
    scenario_data = scenario_data.sort_values('success_rate', ascending=True)
    scenario_data['method_display'] = scenario_data['method'].map(METHOD_NAMES)
    
    colors = plt.cm.RdYlGn(scenario_data['success_rate'].values)
    bars = ax.barh(scenario_data['method_display'], scenario_data['success_rate'].values * 100, color=colors)
    
    ax.set_xlabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'{SCENARIO_NAMES.get(scenario, scenario)}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    for bar, val in zip(bars, scenario_data['success_rate'].values):
        ax.text(val * 100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_success_{scenario}.png', dpi=200, bbox_inches='tight')
    plt.close()

# ============ Fig 4: 干预次数热力图 ============
print("生成 Fig 4: 干预次数分析...")
fig, ax = plt.subplots(figsize=(14, 9))

pivot_int = summary.pivot(index='scenario', columns='method', values='interventions')
pivot_int.index = [SCENARIO_NAMES.get(x, x) for x in pivot_int.index]
pivot_int.columns = [METHOD_NAMES.get(x, x) for x in pivot_int.columns]

# 只对有干预的方法着色
pivot_int_filled = pivot_int.fillna(0)
im = ax.imshow(pivot_int_filled.values, cmap='YlOrRd', aspect='auto')

# 标注数值
for i in range(len(pivot_int.index)):
    for j in range(len(pivot_int.columns)):
        val = pivot_int_filled.values[i, j]
        if val > 0:
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7, color='black')

ax.set_xticks(range(len(pivot_int.columns)))
ax.set_yticks(range(len(pivot_int.index)))
ax.set_xticklabels(pivot_int.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(pivot_int.index, fontsize=9)
ax.set_title('Average Interventions by Method and Scenario', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Avg Interventions')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig4_interventions_heatmap_extended.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 5: 计算效率对比 ============
print("生成 Fig 5: 计算效率...")
fig, ax = plt.subplots(figsize=(14, 8))

comp_data = summary.groupby('method')['computation_time_ms'].mean().sort_values(ascending=True)
comp_data.index = [METHOD_NAMES.get(x, x) for x in comp_data.index]
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(comp_data)))

bars = ax.barh(comp_data.index, comp_data.values, color=colors)
ax.set_xlabel('Computation Time (ms)', fontsize=12)
ax.set_title('Computational Efficiency Comparison', fontsize=14, fontweight='bold')

for bar, val in zip(bars, comp_data.values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}ms', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig5_computation_time_extended.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 6: 成功率 - 效率权衡散点图 ============
print("生成 Fig 6: 成功率 - 效率权衡...")
fig, ax = plt.subplots(figsize=(12, 10))

method_stats = summary.groupby('method').agg({
    'success_rate': 'mean',
    'computation_time_ms': 'mean'
}).reset_index()

method_stats['method_display'] = method_stats['method'].map(METHOD_NAMES)

# 按类别着色
categories = {
    'Pure_VLA': 'Baseline',
    'R1_Only': 'Ablation',
    'R2_Only': 'Ablation',
    'R3_Only': 'Ablation',
    'R1_R2': 'Ablation',
    'R1_R3': 'Ablation',
    'R2_R3': 'Ablation',
    'Ours_Full': 'Ours',
    'DeepReach': 'SOTA',
    'Recovery_RL': 'SOTA',
    'PETS': 'SOTA',
    'CBF_QP': 'SOTA',
    'Shielded_RL': 'SOTA',
}

method_stats['category'] = method_stats['method'].map(categories)
category_colors = {'Baseline': 'gray', 'Ablation': 'blue', 'Ours': 'red', 'SOTA': 'green'}

for cat in ['Baseline', 'Ablation', 'Ours', 'SOTA']:
    cat_data = method_stats[method_stats['category'] == cat]
    ax.scatter(cat_data['computation_time_ms'], cat_data['success_rate'] * 100, 
               s=200, alpha=0.7, label=cat, color=category_colors[cat])
    
    for _, row in cat_data.iterrows():
        ax.annotate(row['method_display'], (row['computation_time_ms'], row['success_rate'] * 100), 
                    fontsize=8, ha='center', va='bottom')

ax.set_xlabel('Computation Time (ms)', fontsize=12)
ax.set_ylabel('Average Success Rate (%)', fontsize=12)
ax.set_title('Trade-off: Success Rate vs Computational Cost', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig6_tradeoff_scatter_extended.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 7: 综合性能排名 ============
print("生成 Fig 7: 综合性能排名...")
fig, ax = plt.subplots(figsize=(14, 8))

# 综合评分：成功率 50% + 低干预 25% + 高效率 25%
summary_norm = summary.copy()

# 归一化
for col in ['success_rate', 'interventions', 'computation_time_ms']:
    max_val = summary_norm[col].max()
    min_val = summary_norm[col].min()
    if max_val > min_val:
        summary_norm[f'{col}_norm'] = (summary_norm[col] - min_val) / (max_val - min_val)
    else:
        summary_norm[f'{col}_norm'] = 0.5

# 成功率越高越好，干预和时间越低越好
summary_norm['score'] = (
    summary_norm['success_rate_norm'] * 0.5 + 
    (1 - summary_norm['interventions_norm'].fillna(0)) * 0.25 +
    (1 - summary_norm['computation_time_ms_norm']) * 0.25
)

method_scores = summary_norm.groupby('method')['score'].mean().sort_values(ascending=True)
method_scores.index = [METHOD_NAMES.get(x, x) for x in method_scores.index]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(method_scores)))

bars = ax.barh(method_scores.index, method_scores.values * 100, color=colors)
ax.set_xlabel('Composite Score (Success×50% + LowIntervention×25% + Efficiency×25%)', fontsize=12)
ax.set_title('Overall Performance Ranking (Higher is Better)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)

for bar, val in zip(bars, method_scores.values):
    ax.text(val * 100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig7_composite_ranking_extended.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ Fig 8: 场景分类别成功率 ============
print("生成 Fig 8: 分类别成功率...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

categories = {
    'Perception': ['s1_lighting_drop', 's2_camera_occlusion', 's3_adversarial_patch'],
    'Dynamics': ['s4_payload_shift', 's5_joint_friction'],
    'Environment': ['s6_dynamic_crowd', 's7_narrow_corridor'],
    'Compound': ['s8_compound_hell'],
}

category_titles = {
    'Perception': 'Category 1: Perception & Cognitive Failures',
    'Dynamics': 'Category 2: Proprioceptive & Dynamics Shifts',
    'Environment': 'Category 3: Open-Environment Disturbances',
    'Compound': 'Category 4: Compound Extreme',
}

for idx, (cat, scenarios) in enumerate(categories.items()):
    ax = axes[idx // 2, idx % 2]
    
    cat_data = summary[summary['scenario'].isin(scenarios)]
    cat_avg = cat_data.groupby('method')['success_rate'].mean().sort_values(ascending=True)
    cat_avg.index = [METHOD_NAMES.get(x, x) for x in cat_avg.index]
    
    colors = plt.cm.RdYlGn(cat_avg.values)
    bars = ax.barh(cat_avg.index, cat_avg.values * 100, color=colors)
    
    ax.set_xlabel('Average Success Rate (%)', fontsize=11)
    ax.set_title(category_titles[cat], fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)
    
    for bar, val in zip(bars, cat_avg.values):
        ax.text(val * 100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%', va='center', fontsize=8)
    
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Embodied AI RTA: Performance by Failure Category', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig8_performance_by_category.png', dpi=200, bbox_inches='tight')
plt.close()

print("\n✅ 所有扩展版图生成完成!")
print(f"输出目录：{output_dir}")

# 生成统计摘要
print("\n" + "="*80)
print("统计摘要")
print("="*80)
print(f"总试验次数：{len(df)}")
print(f"方法数：{df['method'].nunique()}")
print(f"场景数：{df['scenario'].nunique()}")
print(f"每配置重复次数：30")

best_method = method_scores.idxmax()
print(f"\n最佳方法：{best_method} (综合得分：{method_scores.max()*100:.1f})")

best_scenario = summary.groupby('scenario')['success_rate'].mean().idxmax()
worst_scenario = summary.groupby('scenario')['success_rate'].mean().idxmin()
print(f"最高成功率场景：{SCENARIO_NAMES.get(best_scenario, best_scenario)}")
print(f"最具挑战场景：{SCENARIO_NAMES.get(worst_scenario, worst_scenario)}")

# 输出 Top 5 方法
print("\nTop 5 方法排名:")
top5 = method_scores.sort_values(ascending=False).head(5)
for i, (method, score) in enumerate(top5.items(), 1):
    print(f"  {i}. {method}: {score*100:.1f}")
