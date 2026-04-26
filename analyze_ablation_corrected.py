#!/usr/bin/env python3
"""
RTA 消融实验分析 - 修正版 (Corrected Version)

更新日期：2026-04-20
核心修正:
1. 三层 RTA 均为 OR 逻辑 (不是 risk score 融合)
2. R3 三模块 OR 逻辑 (M1 OR M2 OR M3)
3. 使用学习到的正确阈值 (不是硬编码)

正确阈值 (从 normal 场景学习):
- S_logic_min: 0.55 (逻辑合理性阈值)
- D_ham_max: 0.35 (激活链路汉明距离阈值)
- D_ood_max: 0.5 (OOD 马氏距离阈值)

架构说明:
- Region 1: 物理硬约束 (关节限位、自碰撞、速度) → safe_r1 (bool)
- Region 2: 可达集预测 (GRU 支撑函数∩危险边界) → safe_r2 (bool)
- Region 3: 感知异常检测 (M1 梯度 OR M2 激活 OR M3 OOD) → safe_r3 (bool)
- 最终报警：alarm = ¬safe_r1 ∨ ¬safe_r2 ∨ ¬safe_r3

使用:
    python3 analyze_ablation_corrected.py --data-dir /mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL/
"""

import os, sys, json
from datetime import datetime

# ============== 正确阈值 (从 normal 场景学习) ==============

CORRECT_THRESHOLDS = {
    "S_logic_min": 0.55,   # M1: 梯度贡献度阈值 (5% 分位数)
    "D_ham_max": 0.35,     # M2: 激活链路阈值 (95% 分位数)
    "D_ood_max": 0.5,      # M3: OOD 检测阈值 (学习值)
}

print("="*80)
print("【RTA 消融实验分析 - 修正版】")
print("="*80)
print("\n使用正确阈值 (从 normal 场景学习):")
for k, v in CORRECT_THRESHOLDS.items():
    print(f"  {k}: {v}")
print()

# ============== 数据加载 ==============

def load_episode(filepath):
    """加载单个 episode 数据"""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def load_all_episodes(data_dir):
    """加载所有 episode 数据"""
    all_episodes = []
    
    for scene in os.listdir(data_dir):
        scene_dir = os.path.join(data_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        
        for fault in os.listdir(scene_dir):
            fault_dir = os.path.join(scene_dir, fault)
            if not os.path.isdir(fault_dir):
                continue
            
            for ep_file in os.listdir(fault_dir):
                if ep_file.endswith(".jsonl"):
                    filepath = os.path.join(fault_dir, ep_file)
                    try:
                        episode = load_episode(filepath)
                        if len(episode) >= 400:
                            all_episodes.append({
                                "scene": scene,
                                "fault": fault,
                                "file": filepath,
                                "data": episode
                            })
                    except Exception as e:
                        print(f"⚠️  加载失败 {filepath}: {e}")
    
    print(f"✓ 加载了 {len(all_episodes)} 个完整 episode")
    return all_episodes

# ============== 报警计算 (OR 逻辑) ==============

def compute_r3_alarm(scores, thresholds=None):
    """
    Region 3 报警计算 (OR 逻辑)
    
    任何一个模块超过阈值 → 报警
    
    Args:
        scores: dict with S_logic, D_ham, D_ood
        thresholds: 阈值 dict
    
    Returns:
        alarm: bool
        module_alarms: dict with M1, M2, M3
    """
    if thresholds is None:
        thresholds = CORRECT_THRESHOLDS
    
    s_logic = scores.get("S_logic", 1.0)
    d_ham = scores.get("D_ham", 0.0)
    d_ood = scores.get("D_ood", 0.0)
    
    # OR 逻辑：任何一个模块报警 → R3 报警
    m1_alarm = s_logic < thresholds["S_logic_min"]  # 逻辑不合理
    m2_alarm = d_ham > thresholds["D_ham_max"]       # 激活模式异常
    m3_alarm = d_ood > thresholds["D_ood_max"]       # OOD
    
    r3_alarm = m1_alarm or m2_alarm or m3_alarm
    
    return r3_alarm, {"M1": m1_alarm, "M2": m2_alarm, "M3": m3_alarm}

def compute_config_alarm(step, config_type, config, thresholds=None):
    """
    计算给定配置的报警
    
    Args:
        step: 单步数据 dict
        config_type: "layer" (层间消融) 或 "r3" (R3 内部消融)
        config: dict with boolean flags (e.g., {"R1": True, "R2": False, "R3": True})
        thresholds: R3 阈值 dict (可选)
    
    Returns:
        alarm: bool
    
    核心逻辑: OR 逻辑
    - 层间消融：R1 alarm OR R2 alarm OR R3 alarm
    - R3 内部：M1 alarm OR M2 alarm OR M3 alarm
    """
    if config_type == "layer":
        # 层间消融：OR 逻辑
        # 任何一层报警 → 最终报警
        alarms = []
        if config.get("R1"):
            alarms.append(step["region1"]["alarm"])
        if config.get("R2"):
            alarms.append(step["region2"]["alarm"])
        if config.get("R3"):
            alarms.append(step["region3"]["alarm"])
        return any(alarms) if alarms else False
    
    elif config_type == "r3":
        # R3 内部消融：基于分数和阈值的 OR 逻辑
        if thresholds is None:
            thresholds = CORRECT_THRESHOLDS
        
        scores = step["region3"]["scores"]
        alarms = []
        
        # M1: 梯度贡献度 (逻辑合理性)
        if config.get("M1"):
            alarms.append(scores.get("S_logic", 1.0) < thresholds["S_logic_min"])
        
        # M2: 激活链路 (神经元激活模式)
        if config.get("M2"):
            alarms.append(scores.get("D_ham", 0.0) > thresholds["D_ham_max"])
        
        # M3: OOD 检测 (输入分布)
        if config.get("M3"):
            alarms.append(scores.get("D_ood", 0.0) > thresholds["D_ood_max"])
        
        # OR 逻辑
        return any(alarms) if alarms else False
    
    return False

def analyze_config(all_episodes, config, config_type, config_name, thresholds=None):
    """分析单个配置的性能"""
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for ep_info in all_episodes:
        episode = ep_info["data"]
        
        for step in episode:
            alarm = compute_config_alarm(step, config_type, config, thresholds)
            actual_danger = step["ground_truth"]["actual_danger"]
            
            if actual_danger and alarm:
                tp += 1
            elif not actual_danger and alarm:
                fp += 1
            elif not actual_danger and not alarm:
                tn += 1
            else:  # actual_danger and not alarm
                fn += 1
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * tpr * precision / (tpr + precision) if (tpr + precision) > 0 else 0
    
    return {
        "config_name": config_name,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tpr, "fpr": fpr, "precision": precision, "f1": f1,
        "total_steps": tp + fp + tn + fn
    }

# ============== 层间消融分析 ==============

def analyze_layer_ablation(all_episodes, output_dir=None):
    """分析层间消融配置"""
    print("\n" + "="*80)
    print("【层间消融分析】(修正阈值)")
    print("="*80)
    
    layer_configs = {
        "A1_R1_only": {"R1": True, "R2": False, "R3": False},
        "A2_R2_only": {"R1": False, "R2": True, "R3": False},
        "A3_R3_only": {"R1": False, "R2": False, "R3": True},
        "A4_R1+R2": {"R1": True, "R2": True, "R3": False},
        "A5_R1+R3": {"R1": True, "R2": False, "R3": True},
        "A6_R2+R3": {"R1": False, "R2": True, "R3": True},
        "A7_Full": {"R1": True, "R2": True, "R3": True},
    }
    
    results = {}
    
    print("\n%-15s %8s %8s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Precision%", "F1%", "TP", "FP"))
    print("-"*80)
    
    for config_name, config in layer_configs.items():
        result = analyze_config(all_episodes, config, "layer", config_name)
        results[config_name] = result
        print("%-15s %7.1f%% %7.1f%% %8.1f%% %7.1f%% %8d %8d" % (
            config_name, result["tpr"]*100, result["fpr"]*100, 
            result["precision"]*100, result["f1"]*100, result["tp"], result["fp"]))
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "layer_ablation_corrected.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 结果保存到 {output_file}")
    
    return results

# ============== R3 内部消融分析 ==============

def analyze_r3_ablation(all_episodes, output_dir=None, thresholds=None):
    """分析 R3 内部消融配置"""
    print("\n" + "="*80)
    print("【R3 内部消融分析】(修正阈值)")
    print("="*80)
    
    if thresholds is None:
        thresholds = CORRECT_THRESHOLDS
    
    print(f"\n使用阈值：S_logic_min={thresholds['S_logic_min']}, D_ham_max={thresholds['D_ham_max']}, D_ood_max={thresholds['D_ood_max']}")
    
    r3_configs = {
        "R3_Full": {"M1": True, "M2": True, "M3": True},
        "R3_no_M1": {"M1": False, "M2": True, "M3": True},
        "R3_no_M2": {"M1": True, "M2": False, "M3": True},
        "R3_no_M3": {"M1": True, "M2": True, "M3": False},
        "R3_M1_only": {"M1": True, "M2": False, "M3": False},
        "R3_M2_only": {"M1": False, "M2": True, "M3": False},
        "R3_M3_only": {"M1": False, "M2": False, "M3": True},
    }
    
    results = {}
    
    print("\n%-15s %8s %8s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Precision%", "F1%", "TP", "FP"))
    print("-"*80)
    
    for config_name, config in r3_configs.items():
        result = analyze_config(all_episodes, config, "r3", config_name, thresholds)
        results[config_name] = result
        print("%-15s %7.1f%% %7.1f%% %8.1f%% %7.1f%% %8d %8d" % (
            config_name, result["tpr"]*100, result["fpr"]*100, 
            result["precision"]*100, result["f1"]*100, result["tp"], result["fp"]))
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "r3_ablation_corrected.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 结果保存到 {output_file}")
    
    return results

# ============== 主函数 ==============

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RTA 消融实验分析 (修正版)")
    parser.add_argument("--data-dir", required=True, help="数据目录")
    parser.add_argument("--output-dir", default="/mnt/data/ablation_experiments/analysis_corrected", help="输出目录")
    args = parser.parse_args()
    
    # 加载数据
    print(f"\n加载数据：{args.data_dir}")
    all_episodes = load_all_episodes(args.data_dir)
    
    # 分析
    layer_results = analyze_layer_ablation(all_episodes, args.output_dir)
    r3_results = analyze_r3_ablation(all_episodes, args.output_dir)
    
    # 保存汇总
    summary = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": CORRECT_THRESHOLDS,
        "total_episodes": len(all_episodes),
        "layer_ablation": layer_results,
        "r3_ablation": r3_results
    }
    
    output_file = os.path.join(args.output_dir, "summary_corrected.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ 汇总结果保存到 {output_file}")
    print("\n" + "="*80)
    print("【分析完成】")
    print("="*80)

if __name__ == "__main__":
    main()
