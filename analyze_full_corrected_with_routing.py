#!/home/vipuser/miniconda3/envs/aloha_headless/bin/python3
"""
RTA 分析 - 完整修正版 (使用原始数据中的 activations 计算 routing)

修正内容:
1. 从 region3["raw_data"]["activations"] 计算路由距离
2. R3 使用四模块：M1'(路由) OR M1(梯度) OR M2(激活) OR M3(OOD)
3. 使用正确阈值：S_logic_min=0.55, D_ham_max=0.35, D_ood_max=0.5, routing_tau=0.05
4. R2 使用数据中的 alarm（虽然全是 0，但方法正确）
5. 层间 OR 逻辑：alarm = NOT safe_r1 OR NOT safe_r2 OR NOT safe_r3
"""

import os, sys, json, numpy as np
from datetime import datetime

CORRECT_THRESHOLDS = {
    "S_logic_min": 0.55,
    "D_ham_max": 0.35,
    "D_ood_max": 0.5,
    "routing_tau": 0.05,
}

data_dir = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
output_dir = "/mnt/data/ablation_experiments/analysis_corrected"
os.makedirs(output_dir, exist_ok=True)

# 层间消融配置 (OR 逻辑)
layer_configs = {
    "A1_R1_only": {"R1": True, "R2": False, "R3": False},
    "A2_R2_only": {"R1": False, "R2": True, "R3": False},
    "A3_R3_only": {"R1": False, "R2": False, "R3": True},
    "A4_R1+R2": {"R1": True, "R2": True, "R3": False},
    "A5_R1+R3": {"R1": True, "R2": False, "R3": True},
    "A6_R2+R3": {"R1": False, "R2": True, "R3": True},
    "A7_Full": {"R1": True, "R2": True, "R3": True},
}

# R3 内部消融配置 (四模块 OR 逻辑)
r3_configs = {
    "R3_Full": {"M1_routing": True, "M1_gradient": True, "M2_hamming": True, "M3_ood": True},
    "R3_no_routing": {"M1_routing": False, "M1_gradient": True, "M2_hamming": True, "M3_ood": True},
    "R3_no_gradient": {"M1_routing": True, "M1_gradient": False, "M2_hamming": True, "M3_ood": True},
    "R3_no_hamming": {"M1_routing": True, "M1_gradient": True, "M2_hamming": False, "M3_ood": True},
    "R3_no_ood": {"M1_routing": True, "M1_gradient": True, "M2_hamming": True, "M3_ood": False},
    "R3_routing_only": {"M1_routing": True, "M1_gradient": False, "M2_hamming": False, "M3_ood": False},
    "R3_gradient_only": {"M1_routing": False, "M1_gradient": True, "M2_hamming": False, "M3_ood": False},
    "R3_hamming_only": {"M1_routing": False, "M1_gradient": False, "M2_hamming": True, "M3_ood": False},
    "R3_ood_only": {"M1_routing": False, "M1_gradient": False, "M2_hamming": False, "M3_ood": True},
}

layer_tpfp = {name: {"tp":0,"fp":0,"tn":0,"fn":0} for name in layer_configs}
r3_tpfp = {name: {"tp":0,"fp":0,"tn":0,"fn":0} for name in r3_configs}


def flatten(data):
    """递归展平嵌套列表"""
    if isinstance(data, (list, tuple)):
        result = []
        for item in data:
            result.extend(flatten(item))
        return result
    return [data]


def compute_routing_distance(acts):
    """从激活值计算路由距离"""
    distances = []
    layer_pairs = [
        ("encoder_layer0_ffn", "encoder_layer1_ffn"),
        ("encoder_layer1_ffn", "encoder_layer2_ffn"),
        ("encoder_layer2_ffn", "encoder_layer3_ffn"),
        ("encoder_layer3_ffn", "encoder_layer3_ffn"),
    ]
    
    for src, dst in layer_pairs:
        src_act = flatten(acts.get(src, []))
        dst_act = flatten(acts.get(dst, []))
        if not src_act or not dst_act:
            continue
        
        src_mean = sum(abs(x) for x in src_act) / len(src_act)
        dst_mean = sum(abs(x) for x in dst_act) / len(dst_act)
        prod = src_mean * dst_mean
        ref = 0.5
        dist = abs(prod - ref) / (ref + 1e-6)
        distances.append(dist)
    
    return sum(distances) / len(distances) if distances else 0.0


def save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, final=False):

print("="*80)
print("【RTA 分析 - 完整修正版 (含路由分析)】")
print("="*80)
print("\n阈值:", CORRECT_THRESHOLDS)
print("\n处理中...\n")
sys.stdout.flush()

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
                        
                        # 层间消融 (OR 逻辑)
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
                        
                        # R3 内部消融 (四模块 OR 逻辑)
                        if "region3" in step:
                            r3_data = step["region3"]
                            scores = r3_data.get("scores", {})
                            raw_data = r3_data.get("raw_data", {})
                            activations = raw_data.get("activations", {})
                            
                            # 计算路由距离 (从原始 activations)
                            routing_dist = compute_routing_distance(activations) if activations else 0.0
                            routing_alarm = routing_dist > CORRECT_THRESHOLDS["routing_tau"]
                            
                            # 四模块报警
                            m1_routing_alarm = routing_alarm
                            m1_gradient_alarm = scores.get("S_logic", 1.0) < CORRECT_THRESHOLDS["S_logic_min"]
                            m2_hamming_alarm = scores.get("D_ham", 0.0) > CORRECT_THRESHOLDS["D_ham_max"]
                            m3_ood_alarm = scores.get("D_ood", 0.0) > CORRECT_THRESHOLDS["D_ood_max"]
                            
                            for config_name, config in r3_configs.items():
                                alarms = []
                                if config.get("M1_routing"): alarms.append(m1_routing_alarm)
                                if config.get("M1_gradient"): alarms.append(m1_gradient_alarm)
                                if config.get("M2_hamming"): alarms.append(m2_hamming_alarm)
                                if config.get("M3_ood"): alarms.append(m3_ood_alarm)
                                alarm = any(alarms) if alarms else False
                                
                                c = r3_tpfp[config_name]
                                if danger and alarm: c["tp"]+=1
                                elif not danger and alarm: c["fp"]+=1
                                elif not danger and not alarm: c["tn"]+=1
                                else: c["fn"]+=1
                
                if ep_steps >= 400:
                    print(f"[{ep_count}] {scene}/{fault}/{ep_file}")
                    sys.stdout.flush()
                
                # 每 20 集保存一次中间结果
                if ep_count % 20 == 0:
                    save_results(layer_tpfp, r3_tpfp, ep_count, total_steps)
                    print(f"  → 已保存中间结果 ({ep_count}集，{total_steps}步)\n")
                    sys.stdout.flush()
                    
            except Exception as e:
                print(f"  SKIP {ep_file}: {e}")
                sys.stdout.flush()

print(f"\n处理完成：{ep_count}集，{total_steps}步")
print("计算最终结果...\n")
sys.stdout.flush()

# 最终保存
save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, final=True)
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
        "r3_ablation": r3_results,
        "method_notes": {
            "routing": "Computed from raw activations in region3['raw_data']['activations']",
            "layer_fusion": "OR logic: alarm = NOT safe_r1 OR NOT safe_r2 OR NOT safe_r3",
            "r3_fusion": "OR logic: R3_alarm = M1_routing OR M1_gradient OR M2_hamming OR M3_ood"
        }
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
        print("【R3 内部消融结果】(四模块)")
        print("="*80)
        print("\n%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
        print("-"*60)
        for name in r3_configs:
            r = r3_results[name]
            print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
        
        print(f"\n✓ 最终结果：{output_dir}/summary_final.json")
