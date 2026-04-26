import json
from collections import defaultdict

with open('/root/rt1_experiment_outputs/experiment_results.json') as f:
    data = json.load(f)

scenario_stats = defaultdict(lambda: {'success': 0, 'total': 0})
for i in range(len(data['config'])):
    scen = data['scenario'][i]
    scenario_stats[scen]['total'] += 1
    succ = data['success'][i]
    if succ == True or succ == 'True':
        scenario_stats[scen]['success'] += 1

print('按场景统计:')
for scen, stats in sorted(scenario_stats.items()):
    rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f'  {scen}: {rate:.1f}% ({stats["success"]}/{stats["total"]})')
