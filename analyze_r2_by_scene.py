#!/usr/bin/env python3
import json, os

DD = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"

print("="*60)
print("Region 2 表现分析 (新 GRU 模型)")
print("="*60)

scenes = {}
for scene in os.listdir(DD):
    sd = DD+"/"+scene
    if not os.path.isdir(sd): continue
    scenes[scene] = {"tp":0,"fp":0,"tn":0,"fn":0}
    for fault in os.listdir(sd):
        fd = sd+"/"+fault
        if not os.path.isdir(fd): continue
        for f in os.listdir(fd):
            if f.endswith(".jsonl"):
                for line in open(fd+"/"+f):
                    s = json.loads(line)
                    r2_alarm = s["region2"]["alarm"]
                    danger = s["ground_truth"]["actual_danger"]
                    if danger and r2_alarm: scenes[scene]["tp"]+=1
                    elif not danger and r2_alarm: scenes[scene]["fp"]+=1
                    elif not danger and not r2_alarm: scenes[scene]["tn"]+=1
                    else: scenes[scene]["fn"]+=1

print("\n场景          步数     TPR     FPR")
print("-"*40)
for scene, st in sorted(scenes.items()):
    tp,fp,tn,fn = st["tp"],st["fp"],st["tn"],st["fn"]
    steps = tp+fp+tn+fn
    tpr = tp/(tp+fn)*100 if tp+fn>0 else 0
    fpr = fp/(fp+tn)*100 if fp+tn>0 else 0
    print("%-14s %5d   %5.1f%%   %5.1f%%" % (scene, steps, tpr, fpr))

print("\n"+"="*60)
