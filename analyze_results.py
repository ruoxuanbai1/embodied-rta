import json
from collections import defaultdict

with open('/root/rt1_experiment_outputs/experiment_results.json') as f:
    data = json.load(f)

print(f'总试验数：{len(data["config"])}')
print(f'平均干预次数：{sum(data["interventions"])/len(data["interventions"]):.1f}')
print(f'平均风险：{sum(data["avg_risk"])/len(data["avg_risk"]):.3f}')

# 按配置统计
config_stats = defaultdict(lambda: {'interventions': [], 'risk': []})
for i in range(len(data['config'])):
    cfg = data['config'][i]
    config_stats[cfg]['interventions'].append(data['interventions'][i])
    config_stats[cfg]['risk'].append(data['avg_risk'][i])

print('\n按配置统计 (前 7 个):')
for cfg in list(config_stats.keys())[:7]:
    stats = config_stats[cfg]
    avg_int = sum(stats['interventions']) / len(stats['interventions'])
    avg_risk = sum(stats['risk']) / len(stats['risk'])
    print(f'  {cfg:25s}: 干预={avg_int:.1f}次, 风险={avg_risk:.3f}')

# 按场景统计
scen_stats = defaultdict(lambda: {'total': 0})
for i in range(len(data['config'])):
    scen = data['scenario'][i]
    scen_stats[scen]['total'] += 1

print('\n按场景统计:')
for scen, stats in sorted(scen_stats.items()):
    print(f'  {scen}: {stats["total"]} 次试验')
