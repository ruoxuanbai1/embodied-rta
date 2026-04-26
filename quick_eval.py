#!/usr/bin/env python3
import json, os

# 阈值
S_logic_min = 0.55
D_ham_max = 0.35
D_ood_max = 0.5

data_dir = '/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL'

# 快速测试文件
test_files = [
    '/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL/B1_empty/F1_visual_noise/ep001_full_data.jsonl',
    '/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL/B1_empty/F2_visual_occlusion/ep002_full_data.jsonl',
    '/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL/B1_empty/normal/ep000_full_data.jsonl',
    '/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL/B2_static/F10_perception_dynamics/ep000_full_data.jsonl',
    '/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL/B2_static/F11_state_dynamics/ep000_full_data.jsonl',
    '/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL/B2_static/F13_full/ep000_full_data.jsonl',
]

# 统计
stats = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

print("处理文件中...")
for filepath in test_files:
    with open(filepath, 'r') as f:
        for line in f:
            step = json.loads(line)
            
            # R1 alarm
            r1_alarm = step["region1"]["alarm"]
            
            # R2 alarm (使用文件中已有的计算结果)
            r2_alarm = step["region2"]["alarm"]
            
            # R3 alarm
            scores = step["region3"]["scores"]
            m1 = scores.get("S_logic", 1.0) < S_logic_min
            m2 = scores.get("D_ham", 0.0) > D_ham_max
            m3 = scores.get("D_ood", 0.0) > D_ood_max
            r3_alarm = m1 or m2 or m3
            
            # Full alarm (OR logic)
            full_alarm = r1_alarm or r2_alarm or r3_alarm
            
            actual = step["ground_truth"]["actual_danger"]
            
            if actual and full_alarm:
                stats["tp"] += 1
            elif not actual and full_alarm:
                stats["fp"] += 1
            elif not actual and not full_alarm:
                stats["tn"] += 1
            else:
                stats["fn"] += 1

tp, fp, tn, fn = stats["tp"], stats["fp"], stats["tn"], stats["fn"]
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0

print("\n" + "="*60)
print("[快速评估结果] (6 集，新模型 + 新阈值)")
print("="*60)
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"TPR (Recall) = {tpr:.2%}")
print(f"FPR          = {fpr:.2%}")
print(f"Precision    = {precision:.2%}")
print(f"F1-Score     = {f1:.2%}")
print("="*60)
print("\n注：这是小样本快速估计，完整结果等待并行评估完成")
