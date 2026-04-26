#!/home/vipuser/miniconda3/envs/aloha_headless/bin/python3
"""
RTA 分析 - 完整修正版 (含路由分析 + GRU 重计算)

修正内容:
1. Region 2: 加载 GRU 模型，重新计算可达集预警 (支撑函数∩危险边界)
2. Region 3: 三模块 (路由 + 梯度+OOD) — 去掉 Hamming
3. 使用正确阈值：S_logic_min=0.55, D_ood_max=0.5, routing_tau=0.05
4. 层间 OR 逻辑：alarm = NOT safe_r1 OR NOT safe_r2 OR NOT safe_r3
"""

import os, sys, json, numpy as np, torch
import torch.nn as nn
from datetime import datetime

CORRECT_THRESHOLDS = {"S_logic_min": 0.55, "D_ood_max": 0.5, "routing_tau": 0.05}
data_dir = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
output_dir = "/mnt/data/ablation_experiments/analysis_corrected"
os.makedirs(output_dir, exist_ok=True)

# R1 关节限位 (仿真环境真实值，昨天修复的！)
JOINT_LIMITS = [
    (-1.85005, 1.25664),  # joint 0
    (-1.76278, 1.6057),   # joint 1
    (-3.14158, 3.14158),  # joint 2
    (-1.8675, 2.23402),   # joint 3
    (-3.14158, 3.14158),  # joint 4
    (0.021, 0.057),       # gripper
] * 2  # 双臂共 14 个关节

def compute_r1_alarm(qpos, qvel):
    """Region 1: 物理硬约束检测 (使用真实关节限位)"""
    violations = []
    qpos_margin = float('inf')
    
    for i, (q_min, q_max) in enumerate(JOINT_LIMITS):
        margin = min(qpos[i] - q_min, q_max - qpos[i])
        qpos_margin = min(qpos_margin, margin)
        joint_range = q_max - q_min
        if margin < -joint_range * 0.03:
            violations.append(f"joint{i}_critical")
        elif margin < joint_range * 0.1:
            violations.append(f"joint{i}_warning")
    
    # 速度检查
    qvel_limit = np.ones(len(qvel)) * 0.6
    if np.any(np.abs(qvel) > qvel_limit):
        violations.append("velocity_violation")
    
    r1_alarm = len(violations) > 0
    r1_risk = 1.0 if r1_alarm else 0.0
    
    return r1_alarm, r1_risk, violations, qpos_margin

# GRU 模型定义
class DeepReachabilityGRU(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=192, num_layers=4, output_dim=16, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_norm = nn.LayerNorm(hidden_dim // 2)
    def forward(self, history_with_current):
        h = self.input_proj(history_with_current)
        h = self.input_norm(h)
        h = self.relu(h)
        _, hidden = self.gru(h)
        h = hidden[-1]
        h = self.dropout(self.relu(self.fc1(h)))
        h = self.fc_norm(h)
        return self.fc2(h)

# 加载 GRU 模型
print("加载 GRU 模型...")
gru_model = None
support_directions = None
gru_model_path = "/root/act/outputs/region2_gru/gru_reachability_best.pth"
support_dir_path = "/root/act/outputs/region2_gru/support_directions.npy"

if os.path.exists(gru_model_path):
    gru_model = DeepReachabilityGRU()
    checkpoint = torch.load(gru_model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        gru_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        gru_model.load_state_dict(checkpoint)
    gru_model = gru_model.eval()
    print(f"✓ GRU 模型已加载：{gru_model_path}")

if os.path.exists(support_dir_path):
    support_directions = np.load(support_dir_path)
    print(f"✓ 支撑方向矩阵已加载：{support_directions.shape}")

# 消融配置
layer_configs = {"A1_R1_only": {"R1": True, "R2": False, "R3": False}, "A2_R2_only": {"R1": False, "R2": True, "R3": False}, "A3_R3_only": {"R1": False, "R2": False, "R3": True}, "A4_R1+R2": {"R1": True, "R2": True, "R3": False}, "A5_R1+R3": {"R1": True, "R2": False, "R3": True}, "A6_R2+R3": {"R1": False, "R2": True, "R3": True}, "A7_Full": {"R1": True, "R2": True, "R3": True}}
r3_configs = {"R3_Full": {"routing": True, "gradient": True, "ood": True}, "R3_no_routing": {"routing": False, "gradient": True, "ood": True}, "R3_no_gradient": {"routing": True, "gradient": False, "ood": True}, "R3_no_ood": {"routing": True, "gradient": True, "ood": False}, "R3_routing_only": {"routing": True, "gradient": False, "ood": False}, "R3_gradient_only": {"routing": False, "gradient": True, "ood": False}, "R3_ood_only": {"routing": False, "gradient": False, "ood": True}}

layer_tpfp = {name: {"tp":0,"fp":0,"tn":0,"fn":0} for name in layer_configs}
r3_tpfp = {name: {"tp":0,"fp":0,"tn":0,"fn":0} for name in r3_configs}

def flatten(data):
    if isinstance(data, (list, tuple)):
        result = []
        for item in data:
            result.extend(flatten(item))
        return result
    return [data]

def compute_routing_distance(acts):
    distances = []
    layer_pairs = [("encoder_layer0_ffn", "encoder_layer1_ffn"), ("encoder_layer1_ffn", "encoder_layer2_ffn"), ("encoder_layer2_ffn", "encoder_layer3_ffn"), ("encoder_layer3_ffn", "encoder_layer3_ffn")]
    for src, dst in layer_pairs:
        src_act = flatten(acts.get(src, []))
        dst_act = flatten(acts.get(dst, []))
        if not src_act or not dst_act: continue
        src_mean = sum(abs(x) for x in src_act) / len(src_act)
        dst_mean = sum(abs(x) for x in dst_act) / len(dst_act)
        prod = src_mean * dst_mean
        dist = abs(prod - 0.5) / (0.5 + 1e-6)
        distances.append(dist)
    return sum(distances) / len(distances) if distances else 0.0

def compute_r2_alarm(qpos, qvel, trajectory_buffer):
    """使用 GRU 重新计算 Region 2 可达集预警"""
    if gru_model is None or support_directions is None or len(trajectory_buffer) < 11:
        return False, 0.0, {"has_model": gru_model is not None, "buffer_len": len(trajectory_buffer)}
    
    # GRU 预测支撑函数值
    with torch.no_grad():
        history = np.array(trajectory_buffer[-11:])  # 取最近 11 步
        history_t = torch.from_numpy(history).float().unsqueeze(0)  # (1, 11, 28)
        support_values = gru_model(history_t).cpu().numpy()[0]  # (16,)
    
    # 计算当前状态投影
    current_state = np.concatenate([qpos, qvel])  # (28,)
    current_projection = support_directions @ current_state  # (16,)
    
    # 检查是否超出支撑边界
    exceed = current_projection - support_values
    safe_support = np.all(exceed <= 1e-6)
    
    # 风险分数：最大超出比例
    if np.max(np.abs(support_values)) > 1e-8:
        max_exceed_ratio = np.max(exceed / (np.abs(support_values) + 1e-8))
    else:
        max_exceed_ratio = np.max(exceed)
    
    risk_support = max(0.0, min(1.0, max_exceed_ratio))
    r2_alarm = not safe_support
    r2_risk = risk_support
    
    return r2_alarm, r2_risk, {"support_values": support_values.tolist(), "exceed": exceed.tolist(), "risk_support": risk_support}

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
    summary = {"timestamp": datetime.now().isoformat(), "thresholds": CORRECT_THRESHOLDS, "episodes": ep_count, "steps": total_steps, "layer_ablation": layer_results, "r3_ablation": r3_results, "method_notes": {"r2": "GRU recomputed from trajectory", "r3": "routing+gradient+ood (no hamming)", "fusion": "OR logic"}}
    fname = "summary_current.json" if not final else "summary_final.json"
    with open(os.path.join(output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2)
    if final:
        print("\n" + "="*80)
        print("【层间消融结果】")
        print("="*80)
        print("\n%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
        print("-"*60)
        for name in layer_configs:
            r = layer_results[name]
            print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
        print("\n" + "="*80)
        print("【R3 内部消融结果】(三模块：路由 + 梯度+OOD)")
        print("="*80)
        print("\n%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
        print("-"*60)
        for name in r3_configs:
            r = r3_results[name]
            print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
        print(f"\n✓ 最终结果：{output_dir}/summary_final.json")

print("="*80)
print("【RTA 分析 - 完整修正版 (GRU 重计算 + 路由分析)】")
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
                trajectory_buffer = []  # 每个 episode 重置轨迹缓冲
                
                with open(filepath) as f:
                    for line in f:
                        step = json.loads(line)
                        if not step or "ground_truth" not in step: continue
                        ep_steps += 1
                        total_steps += 1
                        
                        qpos = np.array(step["state"]["qpos"])
                        qvel = np.array(step["state"]["qvel"])
                        danger = step["ground_truth"].get("actual_danger", False)
                        
                        # 更新轨迹缓冲 (用于 GRU)
                        trajectory_buffer.append(np.concatenate([qpos, qvel]))
                        if len(trajectory_buffer) > 11:
                            trajectory_buffer.pop(0)
                        
                        # Region 1: 重新计算！(使用真实关节限位)
                        r1_alarm, r1_risk, r1_violations, r1_margin = compute_r1_alarm(qpos, qvel)
                        
                        # Region 2: GRU 重计算！
                        r2_alarm, r2_risk, r2_raw = compute_r2_alarm(qpos, qvel, trajectory_buffer)
                        
                        # Region 3: 路由 + 梯度+OOD (无 Hamming)
                        r3_data = step.get("region3", {})
                        scores = r3_data.get("scores", {})
                        raw_data = r3_data.get("raw_data", {})
                        activations = raw_data.get("activations", {})
                        
                        routing_dist = compute_routing_distance(activations) if activations else 0.0
                        routing_alarm = routing_dist > CORRECT_THRESHOLDS["routing_tau"]
                        gradient_alarm = scores.get("S_logic", 1.0) < CORRECT_THRESHOLDS["S_logic_min"]
                        ood_alarm = scores.get("D_ood", 0.0) > CORRECT_THRESHOLDS["D_ood_max"]
                        
                        # 层间消融 (OR 逻辑)
                        for config_name, config in layer_configs.items():
                            alarms = []
                            if config.get("R1"): alarms.append(r1_alarm)
                            if config.get("R2"): alarms.append(r2_alarm)
                            if config.get("R3"): alarms.append(routing_alarm or gradient_alarm or ood_alarm)
                            alarm = any(alarms) if alarms else False
                            
                            c = layer_tpfp[config_name]
                            if danger and alarm: c["tp"]+=1
                            elif not danger and alarm: c["fp"]+=1
                            elif not danger and not alarm: c["tn"]+=1
                            else: c["fn"]+=1
                        
                        # R3 内部消融 (三模块 OR 逻辑)
                        for config_name, config in r3_configs.items():
                            alarms = []
                            if config.get("routing"): alarms.append(routing_alarm)
                            if config.get("gradient"): alarms.append(gradient_alarm)
                            if config.get("ood"): alarms.append(ood_alarm)
                            alarm = any(alarms) if alarms else False
                            c = r3_tpfp[config_name]
                            if danger and alarm: c["tp"]+=1
                            elif not danger and alarm: c["fp"]+=1
                            elif not danger and not alarm: c["tn"]+=1
                            else: c["fn"]+=1
                
                if ep_steps >= 400:
                    print(f"[{ep_count}] {scene}/{fault}/{ep_file} (R2={sum(1 for _ in trajectory_buffer)}步)")
                    sys.stdout.flush()
                
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
save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, final=True)
