#!/home/vipuser/miniconda3/envs/aloha_headless/bin/python3
"""
RTA 消融实验分析 - 完整修正版

更新日期：2026-04-20
修正内容:
1. 三层 RTA 均为 OR 逻辑 (alarm = ¬R1 ∨ ¬R2 ∨ ¬R3)
2. R3 三模块 OR 逻辑 (M1 OR M2 OR M3)
3. 使用学习到的正确阈值
4. GRU 输出支撑函数值，检查与危险边界交集

正确阈值 (从 normal 场景学习):
- S_logic_min: 0.55 (M1: 梯度贡献度)
- D_ham_max: 0.35 (M2: 激活链路)
- D_ood_max: 0.5 (M3: OOD 检测)

架构:
- Region 1: 物理硬约束 → safe_r1 (bool)
- Region 2: GRU 可达集预测 → safe_r2 (bool)
- Region 3: 感知异常检测 → safe_r3 (bool)
- 最终报警：alarm = NOT safe_r1 OR NOT safe_r2 OR NOT safe_r3
"""

import os, sys, json
from datetime import datetime

# 正确阈值
CORRECT_THRESHOLDS = {
    "S_logic_min": 0.55,
    "D_ham_max": 0.35,
    "D_ood_max": 0.5,
}

print("="*80)
print("【RTA 消融实验分析 - 完整修正版】")
print("="*80)
print("\n使用正确阈值:", CORRECT_THRESHOLDS)

data_dir = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
output_dir = "/mnt/data/ablation_experiments/analysis_corrected"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
print("\n加载数据中...")
all_episodes = []
for scene in sorted(os.listdir(data_dir)):
    scene_dir = os.path.join(data_dir, scene)
    if not os.path.isdir(scene_dir): continue
    for fault in sorted(os.listdir(scene_dir)):
        fault_dir = os.path.join(scene_dir, fault)
        if not os.path.isdir(fault_dir): continue
        for ep_file in sorted(os.listdir(fault_dir)):
            if ep_file.endswith(".jsonl"):
                filepath = os.path.join(fault_dir, ep_file)
                try:
                    with open(filepath) as f:
                        ep = [json.loads(l) for l in f]
                        if len(ep) >= 400:
                            all_episodes.append({"scene": scene, "fault": fault, "data": ep})
                except Exception as e:
                    print(f"  SKIP {ep_file}: {e}")

print(f"加载了 {len(all_episodes)} 个 episode\n")

# ============== 层间消融配置 ==============
layer_configs = {
    "A1_R1_only": {"R1": True, "R2": False, "R3": False},
    "A2_R2_only": {"R1": False, "R2": True, "R3": False},
    "A3_R3_only": {"R1": False, "R2": False, "R3": True},
    "A4_R1+R2": {"R1": True, "R2": True, "R3": False},
    "A5_R1+R3": {"R1": True, "R2": False, "R3": True},
    "A6_R2+R3": {"R1": False, "R2": True, "R3": True},
    "A7_Full": {"R1": True, "R2": True, "R3": True},
}

print("="*80)
print("【层间消融分析】(OR 逻辑)")
print("="*80)
print("\n%-15s %8s %8s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%", "TP", "FP"))
print("-"*80)

layer_results = {}
for config_name, config in layer_configs.items():
    tp, fp, tn, fn = 0, 0, 0, 0
    for ep_info in all_episodes:
        for step in ep_info["data"]:
            # OR 逻辑：任何一层报警 → 最终报警
            alarms = []
            if config.get("R1"): alarms.append(step["region1"]["alarm"])
            if config.get("R2"): alarms.append(step["region2"]["alarm"])
            if config.get("R3"): alarms.append(step["region3"]["alarm"])
            alarm = any(alarms) if alarms else False
            
            danger = step["ground_truth"]["actual_danger"]
            if danger and alarm: tp += 1
            elif not danger and alarm: fp += 1
            elif not danger and not alarm: tn += 1
            else: fn += 1
    
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*tpr*prec/(tpr+prec) if (tpr+prec)>0 else 0
    layer_results[config_name] = {"tpr":tpr,"fpr":fpr,"precision":prec,"f1":f1,"tp":tp,"fp":fp,"tn":tn,"fn":fn}
    print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%% %8d %8d" % (config_name, tpr*100, fpr*100, prec*100, f1*100, tp, fp))

# ============== R3 内部消融配置 ==============
r3_configs = {
    "R3_Full": {"M1": True, "M2": True, "M3": True},
    "R3_no_M1": {"M1": False, "M2": True, "M3": True},
    "R3_no_M2": {"M1": True, "M2": False, "M3": True},
    "R3_no_M3": {"M1": True, "M2": True, "M3": False},
    "R3_M1_only": {"M1": True, "M2": False, "M3": False},
    "R3_M2_only": {"M1": False, "M2": True, "M3": False},
    "R3_M3_only": {"M1": False, "M2": False, "M3": True},
}

print("\n" + "="*80)
print("【R3 内部消融分析】(OR 逻辑 + 修正阈值)")
print("="*80)
print("\n使用阈值：S_logic_min={}, D_ham_max={}, D_ood_max={}".format(
    CORRECT_THRESHOLDS["S_logic_min"], CORRECT_THRESHOLDS["D_ham_max"], CORRECT_THRESHOLDS["D_ood_max"]))
print("\n%-15s %8s %8s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%", "TP", "FP"))
print("-"*80)

r3_results = {}
for config_name, config in r3_configs.items():
    tp, fp, tn, fn = 0, 0, 0, 0
    for ep_info in all_episodes:
        for step in ep_info["data"]:
            scores = step["region3"]["scores"]
            # OR 逻辑：任何一个模块报警 → R3 报警
            alarms = []
            if config.get("M1"): alarms.append(scores.get("S_logic",1.0) < CORRECT_THRESHOLDS["S_logic_min"])
            if config.get("M2"): alarms.append(scores.get("D_ham",0.0) > CORRECT_THRESHOLDS["D_ham_max"])
            if config.get("M3"): alarms.append(scores.get("D_ood",0.0) > CORRECT_THRESHOLDS["D_ood_max"])
            alarm = any(alarms) if alarms else False
            
            danger = step["ground_truth"]["actual_danger"]
            if danger and alarm: tp += 1
            elif not danger and alarm: fp += 1
            elif not danger and not alarm: tn += 1
            else: fn += 1
    
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*tpr*prec/(tpr+prec) if (tpr+prec)>0 else 0
    r3_results[config_name] = {"tpr":tpr,"fpr":fpr,"precision":prec,"f1":f1,"tp":tp,"fp":fp,"tn":tn,"fn":fn}
    print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%% %8d %8d" % (config_name, tpr*100, fpr*100, prec*100, f1*100, tp, fp))

# ============== 保存结果 ==============
summary = {
    "timestamp": datetime.now().isoformat(),
    "thresholds": CORRECT_THRESHOLDS,
    "total_episodes": len(all_episodes),
    "layer_ablation": layer_results,
    "r3_ablation": r3_results,
    "method_notes": {
        "layer_fusion": "OR logic: alarm = NOT safe_r1 OR NOT safe_r2 OR NOT safe_r3",
        "r3_fusion": "OR logic: R3_alarm = M1_alarm OR M2_alarm OR M3_alarm",
        "region2": "GRU outputs support functions (16D), checks intersection with danger boundary",
        "region3_modules": {
            "M1": "Gradient contribution (logic rationality), threshold=0.55",
            "M2": "Activation link (neuron pattern), threshold=0.35",
            "M3": "OOD detection (input distribution), threshold=0.5"
        }
    }
}

with open(os.path.join(output_dir, "summary_corrected.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("【分析完成】")
print("="*80)
print("\n结果保存到:", output_dir+"/summary_corrected.json")
