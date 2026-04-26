#!/bin/bash
# RTA Experiment Progress Monitor

OUTPUT_DIR="/root/rt1_experiment_outputs"
LOG_FILE="$OUTPUT_DIR/progress_report.log"

echo "=== RTA Experiment Progress Report ===" | tee -a $LOG_FILE
echo "Time: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Process status
echo "=== Process Status ===" | tee -a $LOG_FILE
ps aux | grep 'rt1_full_experiment.py --output' | grep -v grep | head -3 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Results stats
echo "=== Results Statistics ===" | tee -a $LOG_FILE
if [ -f "$OUTPUT_DIR/experiment_results.json" ]; then
    python3 << PYEOF
import json
from collections import defaultdict

with open('$OUTPUT_DIR/experiment_results.json') as f:
    data = json.load(f)

total = len(data['config'])
scenarios = len(set(data['scenario']))
configs = len(set(data['config']))

print(f"Total trials completed: {total}")
print(f"Scenarios covered: {scenarios}/8")
print(f"Configs tested: {configs}/14")
print(f"Progress: {total/3360*100:.1f}%")
print("")

# Success rate by config
config_stats = defaultdict(lambda: {'success': 0, 'total': 0})
for i in range(len(data['config'])):
    cfg = data['config'][i]
    config_stats[cfg]['total'] += 1
    if data['success'][i] == True or data['success'][i] == 'True':
        config_stats[cfg]['success'] += 1

print("Success Rate by Configuration (top 5):")
sorted_configs = sorted(config_stats.items(), key=lambda x: x[1]['total'], reverse=True)
for cfg, stats in sorted_configs[:5]:
    rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"  {cfg:25s}: {rate:.1f}% ({stats['success']}/{stats['total']})")
PYEOF
else
    echo "Results file not found yet"
fi
echo "" | tee -a $LOG_FILE

# GPU status
echo "=== GPU Status ===" | tee -a $LOG_FILE
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Disk usage
echo "=== Output Directory ===" | tee -a $LOG_FILE
ls -lh $OUTPUT_DIR/ | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "========================================" | tee -a $LOG_FILE
