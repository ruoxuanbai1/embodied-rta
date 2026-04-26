#!/home/vipuser/miniconda3/envs/aloha_headless/bin/python3
"""RTA 分析 - 流式处理版 (节省内存)"""
import os, sys, json
from datetime import datetime

CORRECT_THRESHOLDS = {"S_logic_min": 0.55, "D_ham_max": 0.35, "D_ood_max": 0.5}
data_dir = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
output_dir = "/mnt/data/ablation_experiments/analysis_corrected"
os.makedirs(output_dir, exist_ok=True)

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

layer_tpfp = {name: {"tp":0,"fp":0,"tn":0,"fn":0} for name in layer_configs}
r3_tpfp = {name: {"tp":0,"fp":0,"tn":0,"fn":0} for name in r3_configs}

print("="*80)
print("【RTA 分析 - 流式处理版】")
print("="*80)
print("\n阈值:", CORRECT_THRESHOLDS)
print("\n处理中...\n")

total_steps = 0
ep_count = 0

for scene in sorted(os.listdir(data_dir)):
    scene_dir = os.path.join(data_dir, scene)
    if not os.path.isdir(scene_dir): continue
    
    for fault in sorted(os.listdir(scene_dir)):
        fault_dir = os.path.join(scene_dir, fault)
        if not os.path.isdir(fault_dir): continue
        
        for ep_file in sorted(os.listdir(fault_dir)):
            if not ep_file.endswith(".jsonl"): continue
            
            filepath = os.path.join(fault_dir, ep_file)
            try:
                ep_count += 1
                ep_steps = 0
                
                with open(filepath) as f:
                    for line in f:
                        step = json.loads(line)
                        if not step or "ground_truth" not in step: continue
                        ep_steps += 1
                        total_steps += 1
                        
                        danger = step["ground_truth"].get("actual_danger", False)
                        
                        # 层间消融
                        for config_name, config in layer_configs.items():
                            alarms = []
                            if config.get("R1") and "region1" in step: alarms.append(step["region1"]["alarm"])
                            if config.get("R2") and "region2" in step: alarms.append(step["region2"]["alarm"])
                            if config.get("R3") and "region3" in step: alarms.append(step["region3"]["alarm"])
                            alarm = any(alarms) if alarms else False
                            
                            c = layer_tpfp[config_name]
                            if danger and alarm: c["tp"]+=1
                            elif not danger and alarm: c["fp"]+=1
                            elif not danger and not alarm: c["tn"]+=1
                            else: c["fn"]+=1
                        
                        # R3 内部消融
                        if "region3" in step and "scores" in step["region3"]:
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
                
                if ep_steps >= 400:
                    print(f"[{ep_count}] {scene}/{fault}/{ep_file}")
                    sys.stdout.flush()
                
                # 每 20 集保存一次
                if ep_count % 20 == 0:
                    save_results(layer_tpfp, r3_tpfp, ep_count, total_steps)
                    print(f"  → 已保存中间结果 ({ep_count}集, {total_steps}步)\n")
                    sys.stdout.flush()
                    
            except Exception as e:
                print(f"  SKIP {ep_file}: {e}")
                sys.stdout.flush()

print(f"\n处理完成：{ep_count}集，{total_steps}步")
print("计算最终结果...\n")
sys.stdout.flush()

# 最终保存
save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, final=True)

def save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, final=False):
    layer_results = {}
    for name, c in layer_tpfp.items():
        tp,fp,tn,fn = c["tp"],c["fp"],c["tn"],c["fn"]
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1 = 2*tpr*prec/(tpr+prec) if (tpr+prec)>0 else 0
        layer_results[name] = {"tpr":tpr,"fpr":fpr,"precision":prec,"f1":f1,"tp":tp,"fp":fp,"tn":tn,"fn":fn}
    
    r3_results = {}
    for name, c in r3_tpfp.items():
        tp,fp,tn,fn = c["tp"],c["fp"],c["tn"],c["fn"]
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1 = 2*tpr*prec/(tpr+prec) if (tpr+prec)>0 else 0
        r3_results[name] = {"tpr":tpr,"fpr":fpr,"precision":prec,"f1":f1,"tp":tp,"fp":fp,"tn":tn,"fn":fn}
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": CORRECT_THRESHOLDS,
        "episodes": ep_count,
        "steps": total_steps,
        "layer_ablation": layer_results,
        "r3_ablation": r3_results
    }
    
    fname = "summary_current.json" if not final else "summary_final.json"
    with open(os.path.join(output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2)
    
    if final:
        print("="*80)
        print("【层间消融结果】")
        print("="*80)
        print("\n%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
        print("-"*60)
        for name in layer_configs:
            r = layer_results[name]
            print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
        
        print("\n" + "="*80)
        print("【R3 内部消融结果】")
        print("="*80)
        print("\n%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
        print("-"*60)
        for name in r3_configs:
            r = r3_results[name]
            print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
        
        print(f"\n✓ 最终结果：{output_dir}/summary_final.json")
