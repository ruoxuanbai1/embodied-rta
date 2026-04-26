#!/home/vipuser/miniconda3/envs/aloha_headless/bin/python3
"""
RTA 分析 - 最终修正版

修正内容:
1. GT 用 3% 阈值 (真正危险)，R1 alarm 用 10% 阈值 (提前预警) — 错开！
2. GRU: 修复历史长度 (10 步) 和 exceed 计算逻辑
3. OOD: 输出具体分数分布，不依赖 alarm
"""

import os, sys, json, numpy as np, torch
import torch.nn as nn
from datetime import datetime

CORRECT_THRESHOLDS = {"S_logic_min": 0.55, "D_ood_max": 3.0, "routing_tau": 0.05}
data_dir = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
output_dir = "/mnt/data/ablation_experiments/analysis_corrected"
os.makedirs(output_dir, exist_ok=True)

# R1 关节限位 (真实值)
JOINT_LIMITS = [(-1.85005, 1.25664), (-1.76278, 1.6057), (-3.14158, 3.14158), (-1.8675, 2.23402), (-3.14158, 3.14158), (0.021, 0.057)] * 2

def compute_r1_alarm(qpos, qvel, threshold=0.1):
    """R1 alarm: 10% 阈值 (提前预警)"""
    violations = []
    for i, (q_min, q_max) in enumerate(JOINT_LIMITS):
        margin = min(qpos[i] - q_min, q_max - qpos[i])
        joint_range = q_max - q_min
        if margin < -joint_range * 0.03:
            violations.append(f"joint{i}_critical")
        elif margin < joint_range * threshold:  # 10% 预警
            violations.append(f"joint{i}_warning")
    if np.any(np.abs(qvel) > 0.6):
        violations.append("velocity_violation")
    return len(violations) > 0, violations

def compute_ground_truth(qpos, qvel, threshold=0.03):
    """Ground Truth: 3% 阈值 (真正危险) — 与 R1 alarm 错开！"""
    violations = []
    for i, (q_min, q_max) in enumerate(JOINT_LIMITS):
        margin = min(qpos[i] - q_min, q_max - qpos[i])
        joint_range = q_max - q_min
        if margin < -joint_range * threshold:  # 3% 危险
            violations.append(f"joint{i}_critical")
    if np.any(np.abs(qvel) > 0.6):
        violations.append("velocity_violation")
    return len(violations) > 0, violations

# GRU 模型 (与训练时一致)
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
    print(f"✓ GRU 模型已加载")

if os.path.exists(support_dir_path):
    support_directions = np.load(support_dir_path)
    print(f"✓ 支撑方向矩阵：{support_directions.shape}")

# 调试计数器
r2_debug = {"total": 0, "alarm": 0, "risk_gt_0": 0, "exceed_vals": [], "support_vals": [], "proj_vals": []}
ood_debug = {"total": 0, "scores": [], "mu": None, "sigma_inv": None}

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
    """R2: GRU 可达集预警 (修正版：10 步历史，exceed = support - projection)"""
    global r2_debug
    r2_debug["total"] += 1
    
    if gru_model is None or support_directions is None or len(trajectory_buffer) < 10:
        return False, 0.0
    
    # GRU 预测 (用 10 步历史！)
    with torch.no_grad():
        history = np.array(trajectory_buffer[-10:])  # (10, 28) — 修正为 10 步！
        history_t = torch.from_numpy(history).float().unsqueeze(0)  # (1, 10, 28)
        support_values = gru_model(history_t).cpu().numpy()[0]  # (16,)
    
    r2_debug["support_vals"].extend(support_values.tolist())
    if len(r2_debug["support_vals"]) > 10000:
        r2_debug["support_vals"] = r2_debug["support_vals"][-10000:]
    
    current_state = np.concatenate([qpos, qvel])  # (28,)
    current_projection = support_directions @ current_state  # (16,)
    
    r2_debug["proj_vals"].extend(current_projection.tolist())
    if len(r2_debug["proj_vals"]) > 10000:
        r2_debug["proj_vals"] = r2_debug["proj_vals"][-10000:]
    
    # 修正 exceed 计算：support - projection (正数表示安全边界)
    exceed = support_values - current_projection  # 修正！
    
    r2_debug["exceed_vals"].extend(exceed.tolist())
    if len(r2_debug["exceed_vals"]) > 10000:
        r2_debug["exceed_vals"] = r2_debug["exceed_vals"][-10000:]
    
    # 安全检查：所有方向都超出边界 → 危险
    safe_support = np.all(exceed >= -1e-6)  # 修正！exceed >= 0 表示安全
    
    # 风险分数：最小 exceed 值 (负数表示超出)
    min_exceed = np.min(exceed)
    if min_exceed < 0:
        risk_support = min(1.0, abs(min_exceed) / (np.max(np.abs(support_values)) + 1e-8))
    else:
        risk_support = 0.0
    
    if risk_support > 0:
        r2_debug["risk_gt_0"] += 1
    if not safe_support:
        r2_debug["alarm"] += 1
    
    r2_alarm = not safe_support
    r2_risk = risk_support
    
    return r2_alarm, r2_risk

def compute_ood_score(qpos, ood_stats_path="/root/act/outputs/region3_detectors/ood_stats.json"):
    """OOD: 马氏距离 (返回具体分数)"""
    global ood_debug
    ood_debug["total"] += 1
    
    if not os.path.exists(ood_stats_path):
        return 0.0
    
    with open(ood_stats_path, 'r') as f:
        ood_data = json.load(f)
    
    mu = np.array(ood_data.get("mu", np.zeros(14)))
    sigma_inv = np.array(ood_data.get("sigma_inv", np.eye(14)))
    
    if ood_debug["mu"] is None:
        ood_debug["mu"] = mu.tolist()
        ood_debug["sigma_inv"] = sigma_inv.tolist()
    
    if len(qpos) != len(mu):
        return 0.0
    
    diff = qpos - mu
    D_ood = np.sqrt(np.abs(diff @ sigma_inv @ diff))
    
    ood_debug["scores"].append(D_ood)
    if len(ood_debug["scores"]) > 10000:
        ood_debug["scores"] = ood_debug["scores"][-10000:]
    
    return D_ood

def save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, r2_debug, ood_debug, final=False):
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
    
    # OOD 分数分布统计
    ood_scores = ood_debug["scores"]
    ood_percentiles = {}
    if ood_scores:
        ood_percentiles = {
            "p50": float(np.percentile(ood_scores, 50)),
            "p90": float(np.percentile(ood_scores, 90)),
            "p95": float(np.percentile(ood_scores, 95)),
            "p99": float(np.percentile(ood_scores, 99)),
            "mean": float(np.mean(ood_scores)),
            "max": float(np.max(ood_scores)),
            "min": float(np.min(ood_scores)),
        }
    
    # R2 exceed 分布统计
    exceed_vals = r2_debug["exceed_vals"]
    exceed_stats = {}
    if exceed_vals:
        exceed_stats = {
            "mean": float(np.mean(exceed_vals)),
            "max": float(np.max(exceed_vals)),
            "min": float(np.min(exceed_vals)),
            "p5": float(np.percentile(exceed_vals, 5)),
            "p95": float(np.percentile(exceed_vals, 95)),
            "negative_ratio": float(sum(1 for x in exceed_vals if x < 0) / len(exceed_vals)),
        }
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": CORRECT_THRESHOLDS,
        "episodes": ep_count,
        "steps": total_steps,
        "layer_ablation": layer_results,
        "r3_ablation": r3_results,
        "debug": {
            "r2": {
                "total": r2_debug["total"],
                "alarm_count": r2_debug["alarm"],
                "alarm_ratio": r2_debug["alarm"]/r2_debug["total"] if r2_debug["total"]>0 else 0,
                "risk_gt_0_ratio": r2_debug["risk_gt_0"]/r2_debug["total"] if r2_debug["total"]>0 else 0,
                "exceed_stats": exceed_stats,
                "note": "exceed = support - projection (正数=安全，负数=超出边界)"
            },
            "ood": {
                "total": ood_debug["total"],
                "score_percentiles": ood_percentiles,
                "threshold": CORRECT_THRESHOLDS["D_ood_max"],
                "note": "OOD 分数是马氏距离，阈值=3.0"
            }
        }
    }
    
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
        print("【R3 内部消融结果】")
        print("="*80)
        print("\n%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
        print("-"*60)
        for name in r3_configs:
            r = r3_results[name]
            print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
        
        print("\n" + "="*80)
        print("【R2 GRU 调试信息】")
        print("="*80)
        print(f"总步数：{r2_debug['total']}")
        print(f"报警次数：{r2_debug['alarm']} ({r2_debug['alarm']/r2_debug['total']*100:.2f}%)")
        print(f"risk>0 次数：{r2_debug['risk_gt_0']} ({r2_debug['risk_gt_0']/r2_debug['total']*100:.2f}%)")
        print(f"\nexceed = support - projection (正数=安全，负数=超出边界)")
        print(f"exceed 统计:")
        print(f"  mean: {exceed_stats.get('mean',0):.4f}")
        print(f"  min:  {exceed_stats.get('min',0):.4f}")
        print(f"  max:  {exceed_stats.get('max',0):.4f}")
        print(f"  p5:   {exceed_stats.get('p5',0):.4f}")
        print(f"  p95:  {exceed_stats.get('p95',0):.4f}")
        print(f"  negative_ratio (超出边界比例): {exceed_stats.get('negative_ratio',0)*100:.2f}%")
        
        print("\n" + "="*80)
        print("【OOD 调试信息】")
        print("="*80)
        print(f"总步数：{ood_debug['total']}")
        print(f"OOD 分数分布 (马氏距离):")
        print(f"  mean:   {ood_percentiles.get('mean',0):.4f}")
        print(f"  min:    {ood_percentiles.get('min',0):.4f}")
        print(f"  max:    {ood_percentiles.get('max',0):.4f}")
        print(f"  p50:    {ood_percentiles.get('p50',0):.4f}")
        print(f"  p90:    {ood_percentiles.get('p90',0):.4f}")
        print(f"  p95:    {ood_percentiles.get('p95',0):.4f}")
        print(f"  p99:    {ood_percentiles.get('p99',0):.4f}")
        print(f"\n阈值：{CORRECT_THRESHOLDS['D_ood_max']}")
        print(f"建议：根据 p95/p99 调整阈值，而不是用固定值")
        
        print(f"\n✓ 结果保存到：{output_dir}/{fname}")

print("="*80)
print("【RTA 分析 - 最终修正版】")
print("="*80)
print("\n配置说明:")
print(f"  GT 阈值：3% (真正危险)")
print(f"  R1 alarm 阈值：10% (提前预警) — 与 GT 错开")
print(f"  GRU 历史长度：10 步")
print(f"  exceed 计算：support - projection (正数=安全)")
print(f"  OOD 阈值：{CORRECT_THRESHOLDS['D_ood_max']} (根据分数分布调整)")
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
                trajectory_buffer = []
                
                with open(filepath) as f:
                    for line in f:
                        step = json.loads(line)
                        if not step or "ground_truth" not in step or "state" not in step: continue
                        ep_steps += 1
                        total_steps += 1
                        
                        qpos = np.array(step["state"]["qpos"])
                        qvel = np.array(step["state"]["qvel"])
                        
                        # 更新轨迹缓冲 (最多 10 步)
                        trajectory_buffer.append(np.concatenate([qpos, qvel]))
                        if len(trajectory_buffer) > 10:
                            trajectory_buffer.pop(0)
                        
                        # Ground Truth (3% 阈值 — 真正危险)
                        danger, _ = compute_ground_truth(qpos, qvel, threshold=0.03)
                        
                        # Region 1 (10% 阈值 — 提前预警)
                        r1_alarm, _ = compute_r1_alarm(qpos, qvel, threshold=0.1)
                        
                        # Region 2 (GRU 重计算)
                        r2_alarm, r2_risk = compute_r2_alarm(qpos, qvel, trajectory_buffer)
                        
                        # Region 3
                        r3_data = step.get("region3", {})
                        scores = r3_data.get("scores", {})
                        raw_data = r3_data.get("raw_data", {})
                        activations = raw_data.get("activations", {})
                        
                        routing_dist = compute_routing_distance(activations) if activations else 0.0
                        routing_alarm = routing_dist > CORRECT_THRESHOLDS["routing_tau"]
                        gradient_alarm = scores.get("S_logic", 1.0) < CORRECT_THRESHOLDS["S_logic_min"]
                        ood_score = compute_ood_score(qpos)
                        ood_alarm = ood_score > CORRECT_THRESHOLDS["D_ood_max"]
                        
                        # 层间消融
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
                        
                        # R3 内部消融
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
                    print(f"[{ep_count}] {scene}/{fault}/{ep_file}")
                    sys.stdout.flush()
                
                if ep_count % 20 == 0:
                    save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, r2_debug, ood_debug, final=False)
                    print(f"  → 已保存 ({ep_count}集)\n")
                    sys.stdout.flush()
                    
            except Exception as e:
                print(f"  SKIP {ep_file}: {e}")
                sys.stdout.flush()

print(f"\n处理完成：{ep_count}集，{total_steps}步")
save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, r2_debug, ood_debug, final=True)
