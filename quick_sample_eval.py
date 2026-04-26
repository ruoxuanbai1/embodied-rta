#!/usr/bin/env python3
"""快速抽样评估 - 从已处理的 episode 中估算指标"""
import json, os, random

DD = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"

# 阈值
S_LOGIC_MIN = 0.55
D_HAM_MAX = 0.35
D_OOD_MAX = 0.5

def eval_episode(filepath):
    stats = {"tp":0, "fp":0, "tn":0, "fn":0, "steps":0}
    with open(filepath, 'r') as f:
        for line in f:
            s = json.loads(line)
            stats["steps"] += 1
            r1 = s["region1"]["alarm"]
            r2 = s["region2"]["alarm"]
            sc = s["region3"]["scores"]
            r3 = (sc.get("S_logic",1)<S_LOGIC_MIN or sc.get("D_ham",0)>D_HAM_MAX or sc.get("D_ood",0)>D_OOD_MAX)
            alarm = r1 or r2 or r3
            danger = s["ground_truth"]["actual_danger"]
            if danger and alarm: stats["tp"]+=1
            elif not danger and alarm: stats["fp"]+=1
            elif not danger and not alarm: stats["tn"]+=1
            else: stats["fn"]+=1
    return stats

def calc_metrics(st):
    tp,fp,tn,fn = st["tp"],st["fp"],st["tn"],st["fn"]
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*prec*tpr/(prec+tpr) if (prec+tpr)>0 else 0
    return {"tpr":tpr, "fpr":fpr, "prec":prec, "f1":f1, "steps":st["steps"]}

# 收集所有文件
files = []
for scene in sorted(os.listdir(DD)):
    sd = os.path.join(DD, scene)
    if not os.path.isdir(sd): continue
    for fault in sorted(os.listdir(sd)):
        fd = os.path.join(sd, fault)
        if not os.path.isdir(fd): continue
        for f in sorted(os.listdir(fd)):
            if f.endswith(".jsonl"):
                files.append(os.path.join(fd, f))

print("总文件数:", len(files))

# 抽样评估 (已处理的文件)
sample_size = min(20, len(files))
sampled = random.sample(files, sample_size)

print("抽样评估 %d 集...\n" % sample_size)

total = {"tp":0, "fp":0, "tn":0, "fn":0, "steps":0}
for i, fp in enumerate(sampled):
    st = eval_episode(fp)
    for k in total: total[k] += st[k]
    m = calc_metrics(st)
    parts = fp.split("/")
    print("  %2d/%d: %s/%s - TPR=%.1f%% FPR=%.1f%% F1=%.1f%% (%d步)" % (
        i+1, sample_size, parts[-3], parts[-2], m["tpr"]*100, m["fpr"]*100, m["f1"]*100, m["steps"]))

m = calc_metrics(total)
print("\n" + "="*60)
print("汇总结果 (%d集，%d步)" % (sample_size, total["steps"]))
print("="*60)
print("TP=%d FP=%d TN=%d FN=%d" % (total["tp"],total["fp"],total["tn"],total["fn"]))
print("TPR (Recall) = %.2f%%" % (m["tpr"]*100))
print("FPR          = %.2f%%" % (m["fpr"]*100))
print("Precision    = %.2f%%" % (m["prec"]*100))
print("F1-Score     = %.2f%%" % (m["f1"]*100))
print("="*60)
