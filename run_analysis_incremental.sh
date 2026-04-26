#!/bin/bash
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate aloha_headless
cd /root/act
python3 << 'PYEOF'
import os, json
from datetime import datetime

CORRECT_THRESHOLDS = {"S_logic_min": 0.55, "D_ham_max": 0.35, "D_ood_max": 0.5}
data_dir = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
output_dir = "/mnt/data/ablation_experiments/analysis_corrected"
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("【RTA 消融实验 - 修正版分析】(增量保存)")
print("="*80)
print("\n使用正确阈值:", CORRECT_THRESHOLDS, "\n")

layer_configs = {
    "A1_R1_only": {"R1": True, "R2": False, "R3": False},
    "A2_R2_only": {"R1": False, "R2": True, "R3": False},
    "A3_R3_only": {"R1": False, "R2": False, "R3": True},
    "A4_R1+R2": {"R1": True, "R2": True, "R3": False},
    "A5_R1+R3": {"R1": True, "R2": False, "R3": True},
    "A6_R2+R3": {"R1": False, "R2": True, "R3": True},
    "A7_Full": {"R1": True, "R2": True, "R3": True},
}

r3_configs = {
    "R3_Full": {"M1": True, "M2": True, "M3": True},
    "R3_no_M1": {"M1": False, "M2": True, "M3": True},
    "R3_no_M2": {"M1": True, "M2": False, "M3": True},
    "R3_no_M3": {"M1": True, "M2": True, "M3": False},
    "R3_M1_only": {"M1": True, "M2": False, "M3": False},
    "R3_M2_only": {"M1": False, "M2": True, "M3": False},
    "R3_M3_only": {"M1": False, "M2": False, "M3": True},
}

# 增量累加器
layer_tpfp = {name: {"tp":0,"fp":0,"tn":0,"fn":0} for name in layer_configs}
r3_tpfp = {name: {"tp":0,"fp":0,"tn":0,"fn":0} for name in r3_configs}

def save_results():
    """保存当前结果"""
    layer_results = {}
    for name, counts in layer_tpfp.items():
        tp,fp,tn,fn = counts["tp"],counts["fp"],counts["tn"],counts["fn"]
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1 = 2*tpr*prec/(tpr+prec) if (tpr+prec)>0 else 0
        layer_results[name] = {"tpr":tpr,"fpr":fpr,"precision":prec,"f1":f1,"tp":tp,"fp":fp,"tn":tn,"fn":fn}
    
    r3_results = {}
    for name, counts in r3_tpfp.items():
        tp,fp,tn,fn = counts["tp"],counts["fp"],counts["tn"],counts["fn"]
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1 = 2*tpr*prec/(tpr+prec) if (tpr+prec)>0 else 0
        r3_results[name] = {"tpr":tpr,"fpr":fpr,"precision":prec,"f1":f1,"tp":tp,"fp":fp,"tn":tn,"fn":fn}
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": CORRECT_THRESHOLDS,
        "layer_ablation": layer_results,
        "r3_ablation": r3_results
    }
    
    with open(os.path.join(output_dir, "summary_current.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # 打印当前结果
    print("\n" + "="*80)
    print("【当前结果】" + datetime.now().strftime("%H:%M:%S"))
    print("="*80)
    
    print("\n【层间消融】")
    print("%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
    print("-"*60)
    for name in layer_configs:
        r = layer_results[name]
        print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
    
    print("\n【R3 内部消融】")
    print("%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
    print("-"*60)
    for name in r3_configs:
        r = r3_results[name]
        print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
    
    print(f"\n✓ 结果已保存到 {output_dir}/summary_current.json")

total_eps = 0
for scene in os.listdir(data_dir):
    scene_dir = os.path.join(data_dir, scene)
    if not os.path.isdir(scene_dir): continue
    
    print(f"\n处理场景：{scene}")
    
    for fault in os.listdir(scene_dir):
        fault_dir = os.path.join(scene_dir, fault)
        if not os.path.isdir(fault_dir): continue
        
        for ep_file in os.listdir(fault_dir):
            if ep_file.endswith(".jsonl"):
                filepath = os.path.join(fault_dir, ep_file)
                try:
                    with open(filepath) as f:
                        ep = [json.loads(l) for l in f]
                        if len(ep) >= 400:
                            total_eps += 1
                            
                            # 处理这个 episode
                            for step in ep:
                                danger = step["ground_truth"]["actual_danger"]
                                
                                # 层间消融
                                for config_name, config in layer_configs.items():
                                    alarms = []
                                    if config.get("R1"): alarms.append(step["region1"]["alarm"])
                                    if config.get("R2"): alarms.append(step["region2"]["alarm"])
                                    if config.get("R3"): alarms.append(step["region3"]["alarm"])
                                    alarm = any(alarms) if alarms else False
                                    c = layer_tpfp[config_name]
                                    if danger and alarm: c["tp"]+=1
                                    elif not danger and alarm: c["fp"]+=1
                                    elif not danger and not alarm: c["tn"]+=1
                                    else: c["fn"]+=1
                                
                                # R3 内部消融
                                scores = step["region3"]["scores"]
                                for config_name, config in r3_configs.items():
                                    alarms = []
                                    if config.get("M1"): alarms.append(scores.get("S_logic",1.0) < CORRECT_THRESHOLDS["S_logic_min"])
                                    if config.get("M2"): alarms.append(scores.get("D_ham",0.0) > CORRECT_THRESHOLDS["D_ham_max"])
                                    if config.get("M3"): alarms.append(scores.get("D_ood",0.0) > CORRECT_THRESHOLDS["D_ood_max"])
                                    alarm = any(alarms) if alarms else False
                                    c = r3_tpfp[config_name]
                                    if danger and alarm: c["tp"]+=1
                                    elif not danger and alarm: c["fp"]+=1
                                    elif not danger and not alarm: c["tn"]+=1
                                    else: c["fn"]+=1
                            
                            # 每 5 个 episode 保存一次
                            if total_eps % 5 == 0:
                                save_results()
                                print(f"\n已处理 {total_eps} 个 episode...")
                
                except Exception as e:
                    print(f"  ⚠️ 跳过 {ep_file}: {e}")
    
    # 每个场景结束后保存
    print(f"\n✓ 场景 {scene} 完成，保存结果...")
    save_results()

# 最终保存
print("\n" + "="*80)
print("【全部完成】")
print("="*80)
print(f"总 episode 数：{total_eps}")
save_results()
print("\n最终结果：{}/summary_current.json".format(output_dir))
PYEOF
