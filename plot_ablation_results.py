#!/usr/bin/env python3
"""消融实验结果可视化"""

import csv, json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("/home/vipuser/Embodied-RTA/outputs/ablation")
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 加载数据
with open(OUTPUT_DIR / "results.csv") as f:
    rows = list(csv.DictReader(f))

print(f'加载 {len(rows)} 条试验记录...')

# 按配置统计
configs = ["A0_Pure_VLA", "A1_R1_Only", "A2_R2_Only", "A3_R3_Only", 
           "A4_R1_R2", "A5_R1_R3", "A6_R2_R3", "A7_Full"]

stats = {}
for cfg in configs:
    subset = [r for r in rows if r['config'] == cfg]
    stats[cfg] = {
        'success_rate': np.mean([int(r['success']=='True') for r in subset]) * 100,
        'collision_rate': np.mean([int(r['collision']=='True') for r in subset]) * 100,
        'warning_rate': np.mean([int(r['warning']=='True') for r in subset]) * 100,
        'lead_time': np.mean([float(r['lead_time']) for r in subset if r['warning']=='True']),
        'runtime_ms': np.mean([float(r['runtime_ms']) for r in subset]),
    }

# 图 1: 成功率对比
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(configs)), [stats[c]['success_rate'] for c in configs], 
               color=['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#c0392b'])
plt.xticks(range(len(configs)), ['A0\nPure', 'A1\nR1', 'A2\nR2', 'A3\nR3', 'A4\nR1+R2', 'A5\nR1+R3', 'A6\nR2+R3', 'A7\nFull'], fontsize=9)
plt.ylabel('成功率 (%)', fontsize=12)
plt.title('消融实验：成功率对比', fontsize=14, fontweight='bold')
plt.ylim([0, 100])
for i, (bar, rate) in enumerate(zip(bars, [stats[c]['success_rate'] for c in configs])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_success_rate.png', dpi=150, bbox_inches='tight')
print('保存：fig1_success_rate.png')

# 图 2: 碰撞率对比
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(configs)), [stats[c]['collision_rate'] for c in configs],
               color=['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#c0392b'])
plt.xticks(range(len(configs)), ['A0\nPure', 'A1\nR1', 'A2\nR2', 'A3\nR3', 'A4\nR1+R2', 'A5\nR1+R3', 'A6\nR2+R3', 'A7\nFull'], fontsize=9)
plt.ylabel('碰撞率 (%)', fontsize=12)
plt.title('消融实验：碰撞率对比', fontsize=14, fontweight='bold')
plt.ylim([0, 30])
for i, (bar, rate) in enumerate(zip(bars, [stats[c]['collision_rate'] for c in configs])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_collision_rate.png', dpi=150, bbox_inches='tight')
print('保存：fig2_collision_rate.png')

# 图 3: 预警率对比
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(configs)), [stats[c]['warning_rate'] for c in configs],
               color=['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#c0392b'])
plt.xticks(range(len(configs)), ['A0\nPure', 'A1\nR1', 'A2\nR2', 'A3\nR3', 'A4\nR1+R2', 'A5\nR1+R3', 'A6\nR2+R3', 'A7\nFull'], fontsize=9)
plt.ylabel('预警率 (%)', fontsize=12)
plt.title('消融实验：预警率对比', fontsize=14, fontweight='bold')
plt.ylim([0, 80])
for i, (bar, rate) in enumerate(zip(bars, [stats[c]['warning_rate'] for c in configs])):
    if rate > 0:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_warning_rate.png', dpi=150, bbox_inches='tight')
print('保存：fig3_warning_rate.png')

# 图 4: 预警提前时间
plt.figure(figsize=(10, 6))
lead_times = [stats[c]['lead_time'] if stats[c]['warning_rate'] > 0 else 0 for c in configs]
bars = plt.bar(range(len(configs)), lead_times,
               color=['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#c0392b'])
plt.xticks(range(len(configs)), ['A0\nPure', 'A1\nR1', 'A2\nR2', 'A3\nR3', 'A4\nR1+R2', 'A5\nR1+R3', 'A6\nR2+R3', 'A7\nFull'], fontsize=9)
plt.ylabel('预警提前时间 (秒)', fontsize=12)
plt.title('消融实验：预警提前时间对比', fontsize=14, fontweight='bold')
plt.ylim([0, 1.5])
for i, (bar, lt) in enumerate(zip(bars, lead_times)):
    if lt > 0:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{lt:.2f}s', ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_lead_time.png', dpi=150, bbox_inches='tight')
print('保存：fig4_lead_time.png')

# 图 5: 运行时延
plt.figure(figsize=(10, 6))
runtimes = [stats[c]['runtime_ms'] for c in configs]
bars = plt.bar(range(len(configs)), runtimes,
               color=['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#c0392b'])
plt.xticks(range(len(configs)), ['A0\nPure', 'A1\nR1', 'A2\nR2', 'A3\nR3', 'A4\nR1+R2', 'A5\nR1+R3', 'A6\nR2+R3', 'A7\nFull'], fontsize=9)
plt.ylabel('运行时延 (毫秒)', fontsize=12)
plt.title('消融实验：运行时延对比', fontsize=14, fontweight='bold')
plt.ylim([0, 50])
for i, (bar, rt) in enumerate(zip(bars, runtimes)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{rt:.1f}ms', ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_runtime.png', dpi=150, bbox_inches='tight')
print('保存：fig5_runtime.png')

# 图 6: 综合对比 (成功率 vs 碰撞率 散点图)
plt.figure(figsize=(10, 8))
for i, cfg in enumerate(configs):
    s = stats[cfg]
    plt.scatter(s['collision_rate'], s['success_rate'], s=200, 
                label=cfg.replace('_', ' '), alpha=0.7,
                color=['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#c0392b'][i])
    plt.annotate(cfg, (s['collision_rate']+0.3, s['success_rate']), fontsize=9)

plt.xlabel('碰撞率 (%)', fontsize=12)
plt.ylabel('成功率 (%)', fontsize=12)
plt.title('消融实验：成功率 vs 碰撞率', fontsize=14, fontweight='bold')
plt.xlim([-1, 25])
plt.ylim([40, 95])
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_tradeoff.png', dpi=150, bbox_inches='tight')
print('保存：fig6_tradeoff.png')

print()
print('='*50)
print('所有图表已保存到:', FIG_DIR)
print('='*50)
