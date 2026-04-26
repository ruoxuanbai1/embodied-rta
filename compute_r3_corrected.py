#!/home/vipuser/miniconda3/envs/aloha_headless/bin/python3
import os, json
from datetime import datetime

CORRECT_THRESHOLDS = {"S_logic_min": 0.55, "D_ham_max": 0.35, "D_ood_max": 0.5}
data_dir = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
output_dir = "/mnt/data/ablation_experiments/analysis_corrected"
os.makedirs(output_dir, exist_ok=True)

print("Loading episodes...")
all_episodes = []
for scene in os.listdir(data_dir):
    for fault in os.listdir(os.path.join(data_dir, scene)):
        fd = os.path.join(data_dir, scene, fault)
        if os.path.isdir(fd):
            for f in os.listdir(fd):
                if f.endswith(".jsonl"):
                    fp = os.path.join(fd, f)
                    try:
                        ep = [json.loads(l) for l in open(fp)]
                        if len(ep) >= 400:
                            all_episodes.append({"s": scene, "f": fault, "d": ep})
                    except: pass

print("Loaded", len(all_episodes), "episodes")

r3_configs = {
    "R3_Full": {"M1": True, "M2": True, "M3": True},
    "R3_no_M1": {"M1": False, "M2": True, "M3": True},
    "R3_no_M2": {"M1": True, "M2": False, "M3": True},
    "R3_no_M3": {"M1": True, "M2": True, "M3": False},
    "R3_M1_only": {"M1": True, "M2": False, "M3": False},
    "R3_M2_only": {"M1": False, "M2": True, "M3": False},
    "R3_M3_only": {"M1": False, "M2": False, "M3": True},
}

r3_tpfp = {name: {"tp":0,"fp":0,"tn":0,"fn":0} for name in r3_configs}
total_steps = 0

for i, ep_info in enumerate(all_episodes):
    for step in ep_info["d"]:
        total_steps += 1
        danger = step["ground_truth"]["actual_danger"]
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
    
    if (i+1) % 10 == 0:
        print("Processed", i+1, "/", len(all_episodes), "episodes...")
        r3_results = {}
        for name, counts in r3_tpfp.items():
            tp,fp,tn,fn = counts["tp"],counts["fp"],counts["tn"],counts["fn"]
            tpr = tp/(tp+fn) if (tp+fn)>0 else 0
            fpr = fp/(fp+tn) if (fp+tn)>0 else 0
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            f1 = 2*tpr*prec/(tpr+prec) if (tpr+prec)>0 else 0
            r3_results[name] = {"tpr":tpr,"fpr":fpr,"precision":prec,"f1":f1,"tp":tp,"fp":fp}
        with open(os.path.join(output_dir, "r3_current.json"), "w") as f:
            json.dump({"thresholds": CORRECT_THRESHOLDS, "total_steps": total_steps, "r3_ablation": r3_results}, f, indent=2)
        print("  Saved to r3_current.json")

print("\nComputing final results...")
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
    "total_episodes": len(all_episodes),
    "total_steps": total_steps,
    "r3_ablation": r3_results
}

with open(os.path.join(output_dir, "r3_final.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("【R3 内部消融 - 修正阈值结果】")
print("="*70)
print("\n%-15s %8s %8s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%", "TP", "FP"))
print("-"*70)
for name in r3_configs:
    r = r3_results[name]
    print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%% %8d %8d" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100, r["tp"], r["fp"]))

print("\n结果保存到:", output_dir+"/r3_final.json")
