#!/home/vipuser/miniconda3/envs/aloha_headless/bin/python3
"""
RTA 分析 - R2 修复 + 事件指标修正版

修复内容:
1. R2: 添加安全余量阈值 (exceed < -2.0 才报警，不是<0)
2. 事件压缩：修正 precision/recall 计算
3. 添加 FAR (False Alarm Rate) 指标
4. OOD 阈值：250
"""

import os, sys, json, numpy as np, torch
import torch.nn as nn
from datetime import datetime

CORRECT_THRESHOLDS = {"S_logic_min": 0.55, "D_ood_max": 250.0, "routing_tau": 0.05}
R2_EXCEED_THRESHOLD = 2.0  # R2 安全余量：超出 2 个单位才报警

data_dir = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
output_dir = "/mnt/data/ablation_experiments/analysis_corrected"
os.makedirs(output_dir, exist_ok=True)

JOINT_SAFE_RANGE = [
    (-1.85, 1.26), (-2.12, -0.96), (0.11, 1.57), (-0.16, 1.25),
    (-0.30, 3.51), (-2.43, 0.00), (0.08, 2.73), (-0.16, 1.26),
    (-2.26, -0.96), (0.11, 1.57), (-0.16, 1.25), (-1.85, 0.00),
    (0.00, 0.17), (0.08, 0.18),
]

def compute_ground_truth(qpos, qvel, threshold=0.03):
    violations = []
    for i in range(min(len(qpos), len(JOINT_SAFE_RANGE))):
        q_min, q_max = JOINT_SAFE_RANGE[i]
        margin = min(qpos[i] - q_min, q_max - qpos[i])
        joint_range = q_max - q_min
        if margin < -joint_range * threshold:
            violations.append(f"joint{i}_critical")
    if np.any(np.abs(qvel) > 0.6):
        violations.append("velocity_violation")
    return len(violations) > 0, violations

def compute_r1_alarm(qpos, qvel, threshold=0.1):
    violations = []
    for i in range(min(len(qpos), len(JOINT_SAFE_RANGE))):
        q_min, q_max = JOINT_SAFE_RANGE[i]
        margin = min(qpos[i] - q_min, q_max - qpos[i])
        joint_range = q_max - q_min
        if margin < -joint_range * 0.03:
            violations.append(f"joint{i}_critical")
        elif margin < joint_range * threshold:
            violations.append(f"joint{i}_warning")
    if np.any(np.abs(qvel) > 0.6):
        violations.append("velocity_violation")
    return len(violations) > 0, violations

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
print(f"R2 安全余量阈值：{R2_EXCEED_THRESHOLD}")
gru_model = None
support_directions = None

if os.path.exists("/root/act/outputs/region2_gru/gru_reachability_best.pth"):
    gru_model = DeepReachabilityGRU()
    checkpoint = torch.load("/root/act/outputs/region2_gru/gru_reachability_best.pth", map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        gru_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        gru_model.load_state_dict(checkpoint)
    gru_model = gru_model.eval()
    print("✓ GRU 模型已加载")

if os.path.exists("/root/act/outputs/region2_gru/support_directions.npy"):
    support_directions = np.load("/root/act/outputs/region2_gru/support_directions.npy")
    print(f"✓ 支撑方向矩阵：{support_directions.shape}")

ood_mu = None
ood_sigma_inv = None
ood_stats_path = "/root/act/outputs/region3_detectors/ood_stats.json"
if os.path.exists(ood_stats_path):
    with open(ood_stats_path, 'r') as f:
        ood_data = json.load(f)
    ood_mu = np.array(ood_data.get("mu", []))
    ood_sigma_inv = np.array(ood_data.get("sigma_inv", []))
    print(f"✓ OOD 统计量：mu={ood_mu.shape}, sigma_inv={ood_sigma_inv.shape}")

r2_debug = {"total": 0, "alarm": 0, "exceed_negative": 0, "exceed_vals": []}
ood_debug = {"total": 0, "scores": [], "loaded": ood_mu is not None}
gt_debug = {"total": 0, "danger": 0, "danger_ratio": 0, "margin_stats": []}

# 事件级别评估
event_debug = {
    "danger_events": 0,
    "alarm_events": 0,
    "warned_events": 0,
    "false_alarm_events": 0,
    "lead_times": [],
}

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
    """R2: GRU 可达集预警 (添加安全余量阈值)"""
    global r2_debug
    r2_debug["total"] += 1
    if gru_model is None or support_directions is None or len(trajectory_buffer) < 10:
        return False, 0.0
    with torch.no_grad():
        history = np.array(trajectory_buffer[-10:])
        history_t = torch.from_numpy(history).float().unsqueeze(0)
        support_values = gru_model(history_t).cpu().numpy()[0]
    current_state = np.concatenate([qpos, qvel])
    current_projection = support_directions @ current_state
    exceed = support_values - current_projection
    r2_debug["exceed_vals"].extend(exceed.tolist())
    if len(r2_debug["exceed_vals"]) > 10000:
        r2_debug["exceed_vals"] = r2_debug["exceed_vals"][-10000:]
    negative_count = sum(1 for x in exceed if x < 0)
    if negative_count > 0:
        r2_debug["exceed_negative"] += 1
    
    # 修复：使用安全余量阈值，不是 exceed<0 就报警
    # exceed < -R2_EXCEED_THRESHOLD 才报警 (超出 2 个单位)
    safe_support = np.all(exceed >= -R2_EXCEED_THRESHOLD)
    r2_alarm = not safe_support
    
    if r2_alarm:
        r2_debug["alarm"] += 1
    
    min_exceed = np.min(exceed)
    r2_risk = min(1.0, abs(min_exceed + R2_EXCEED_THRESHOLD) / (np.max(np.abs(support_values)) + 1e-8)) if min_exceed < -R2_EXCEED_THRESHOLD else 0.0
    
    return r2_alarm, r2_risk

def compute_ood_score(qpos, qvel):
    global ood_debug
    ood_debug["total"] += 1
    if ood_mu is None or ood_sigma_inv is None:
        return 0.0
    state = np.concatenate([qpos, qvel])
    if len(state) != len(ood_mu):
        return 0.0
    diff = state - ood_mu
    D_ood = np.sqrt(np.abs(diff @ ood_sigma_inv @ diff))
    ood_debug["scores"].append(D_ood)
    if len(ood_debug["scores"]) > 10000:
        ood_debug["scores"] = ood_debug["scores"][-10000:]
    return D_ood

def compress_events(seq):
    """压缩连续事件：[0,0,1,1,1,0,0,1,1,0] → 事件起始位置 [2, 7]"""
    events = []
    in_event = False
    for i, val in enumerate(seq):
        if val and not in_event:
            events.append(i)
            in_event = True
        elif not val:
            in_event = False
    return events

def compute_event_metrics(danger_seq, alarm_seq, lead_window=25):
    """
    事件级别评估 (压缩报警流)
    
    返回:
    - danger_events: 危险事件数
    - alarm_events: 报警事件数
    - warned_events (TP): 成功预警的危险事件数
    - false_alarm_events (FP): 安全时间段内的误报警事件数
    - missed_events (FN): 未预警的危险事件数
    - lead_times: 预警提前时间
    - precision: TP / (TP + FP)
    - recall: TP / (TP + FN)
    - f1: 2 * P * R / (P + R)
    - far: FP / total_safe_steps (误报率)
    """
    danger_event_indices = compress_events(danger_seq)
    alarm_event_indices = compress_events(alarm_seq)
    
    num_danger_events = len(danger_event_indices)
    num_alarm_events = len(alarm_event_indices)
    
    # 对每个危险事件，检查前 lead_window 步是否有报警
    warned_events = 0
    lead_times = []
    warned_danger_indices = set()
    
    for danger_idx in danger_event_indices:
        warned = False
        for alarm_idx in alarm_event_indices:
            if danger_idx - lead_window <= alarm_idx < danger_idx:
                warned = True
                lead_times.append(danger_idx - alarm_idx)
                warned_danger_indices.add(danger_idx)
                break
        if warned:
            warned_events += 1
    
    # FP：不在任何危险事件 lead_window 内的报警事件
    false_alarm_events = 0
    for alarm_idx in alarm_event_indices:
        is_valid_warning = False
        for danger_idx in danger_event_indices:
            if danger_idx - lead_window <= alarm_idx < danger_idx:
                is_valid_warning = True
                break
        if not is_valid_warning:
            false_alarm_events += 1
    
    # FN：未预警的危险事件
    missed_events = num_danger_events - warned_events
    
    # 计算 FAR (False Alarm Rate) = FP / 安全步数
    safe_steps = sum(1 for d in danger_seq if not d)
    far = false_alarm_events / safe_steps if safe_steps > 0 else 0.0
    
    # Precision / Recall / F1
    tp = warned_events
    fp = false_alarm_events
    fn = missed_events
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "danger_events": num_danger_events,
        "alarm_events": num_alarm_events,
        "warned_events": tp,
        "false_alarm_events": fp,
        "missed_events": fn,
        "lead_times": lead_times,
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
    }

def save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, r2_debug, ood_debug, gt_debug, event_debug, final=False):
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
    
    exceed_vals = r2_debug["exceed_vals"]
    exceed_stats = {}
    if exceed_vals:
        exceed_stats = {
            "mean": float(np.mean(exceed_vals)),
            "std": float(np.std(exceed_vals)),
            "max": float(np.max(exceed_vals)),
            "min": float(np.min(exceed_vals)),
            "p1": float(np.percentile(exceed_vals, 1)),
            "p5": float(np.percentile(exceed_vals, 5)),
            "p95": float(np.percentile(exceed_vals, 95)),
            "p99": float(np.percentile(exceed_vals, 99)),
            "negative_ratio": float(r2_debug["exceed_negative"] / r2_debug["total"]) if r2_debug["total"]>0 else 0,
            "alarm_ratio": float(r2_debug["alarm"] / r2_debug["total"]) if r2_debug["total"]>0 else 0,
        }
    
    ood_scores = ood_debug["scores"]
    ood_stats = {}
    if ood_scores:
        ood_stats = {
            "mean": float(np.mean(ood_scores)),
            "std": float(np.std(ood_scores)),
            "max": float(np.max(ood_scores)),
            "min": float(np.min(ood_scores)),
            "p50": float(np.percentile(ood_scores, 50)),
            "p90": float(np.percentile(ood_scores, 90)),
            "p95": float(np.percentile(ood_scores, 95)),
            "p99": float(np.percentile(ood_scores, 99)),
        }
    
    gt_margin_stats = gt_debug["margin_stats"]
    gt_margin_summary = {}
    if gt_margin_stats:
        gt_margin_summary = {
            "mean": float(np.mean(gt_margin_stats)),
            "std": float(np.std(gt_margin_stats)),
            "min": float(np.min(gt_margin_stats)),
            "max": float(np.max(gt_margin_stats)),
            "p5": float(np.percentile(gt_margin_stats, 5)),
            "p95": float(np.percentile(gt_margin_stats, 95)),
        }
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": CORRECT_THRESHOLDS,
        "r2_exceed_threshold": R2_EXCEED_THRESHOLD,
        "episodes": ep_count,
        "steps": total_steps,
        "layer_ablation": layer_results,
        "r3_ablation": r3_results,
        "event_metrics": event_debug,
        "debug": {
            "gt": {
                "total": gt_debug["total"],
                "danger_count": gt_debug["danger"],
                "danger_ratio": gt_debug["danger"]/gt_debug["total"] if gt_debug["total"]>0 else 0,
                "margin_stats": gt_margin_summary,
                "note": "GT 使用数据驱动的安全范围 (p1/p99)"
            },
            "r2": {
                "total": r2_debug["total"],
                "alarm_count": r2_debug["alarm"],
                "alarm_ratio": r2_debug["alarm"]/r2_debug["total"] if r2_debug["total"]>0 else 0,
                "exceed_negative_ratio": r2_debug["exceed_negative"]/r2_debug["total"] if r2_debug["total"]>0 else 0,
                "exceed_stats": exceed_stats,
                "note": f"exceed = support - projection, exceed < -{R2_EXCEED_THRESHOLD} 才报警"
            },
            "ood": {
                "loaded": ood_debug["loaded"],
                "total": ood_debug["total"],
                "score_stats": ood_stats,
                "threshold": CORRECT_THRESHOLDS["D_ood_max"],
                "note": "OOD 分数=马氏距离 (28 维状态：qpos+qvel), 阈值=250"
            }
        }
    }
    
    fname = "summary_current.json" if not final else "summary_final.json"
    with open(os.path.join(output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2)
    
    if final:
        print("\n" + "="*80)
        print("【事件级别评估 (压缩报警流)】")
        print("="*80)
        print(f"危险事件数：{event_debug.get('danger_events', 0)}")
        print(f"报警事件数：{event_debug.get('alarm_events', 0)}")
        print(f"成功预警事件 (TP): {event_debug.get('warned_events', 0)}")
        print(f"误报警事件 (FP): {event_debug.get('false_alarm_events', 0)}")
        print(f"未预警事件 (FN): {event_debug.get('missed_events', 0)}")
        print(f"\nPrecision: {event_debug.get('precision', 0)*100:.2f}%")
        print(f"Recall: {event_debug.get('recall', 0)*100:.2f}%")
        print(f"F1: {event_debug.get('f1', 0):.4f}")
        print(f"FAR (误报率): {event_debug.get('far', 0)*100:.4f}%")
        print(f"平均预警提前时间：{event_debug.get('mean_lead_time', 0):.2f} 步")
        
        print("\n" + "="*80)
        print("【层间消融结果 (步级别)】")
        print("="*80)
        print("\n%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
        print("-"*60)
        for name in layer_configs:
            r = layer_results[name]
            print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
        
        print("\n" + "="*80)
        print("【R3 内部消融结果 (步级别)】")
        print("="*80)
        print("\n%-15s %8s %8s %8s %8s" % ("配置", "TPR%", "FPR%", "Prec%", "F1%"))
        print("-"*60)
        for name in r3_configs:
            r = r3_results[name]
            print("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, r["tpr"]*100, r["fpr"]*100, r["precision"]*100, r["f1"]*100))
        
        print(f"\n✓ 结果保存到：{output_dir}/{fname}")

print("="*80)
print("【RTA 分析 - R2 修复 + 事件指标修正版】")
print("="*80)
print("\n配置说明:")
print(f"  GT: 数据驱动安全范围 (p1/p99)")
print(f"  R1 alarm: 宽松范围 (p5/p95)")
print(f"  R2: GRU 可达集 (exceed < -{R2_EXCEED_THRESHOLD} 才报警)")
print(f"  OOD: 28 维状态 (qpos+qvel), 阈值={CORRECT_THRESHOLDS['D_ood_max']}")
print(f"  事件评估：预警窗口=25 步 (0.5 秒), 压缩连续报警")
print(f"  指标：Precision/Recall/F1/FAR (事件级别)")
print(f"  保存频率：每 5 集")
print("\n处理中...\n")
sys.stdout.flush()

total_steps = 0
ep_count = 0
episode_data = []

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
                episode_data = []
                
                with open(filepath) as f:
                    for line in f:
                        step = json.loads(line)
                        if not step or "state" not in step: continue
                        ep_steps += 1
                        total_steps += 1
                        
                        qpos = np.array(step["state"]["qpos"])
                        qvel = np.array(step["state"]["qvel"])
                        
                        gt_debug["total"] += 1
                        danger, _ = compute_ground_truth(qpos, qvel, threshold=0.03)
                        
                        min_margin = float('inf')
                        for i in range(min(len(qpos), len(JOINT_SAFE_RANGE))):
                            q_min, q_max = JOINT_SAFE_RANGE[i]
                            margin = min(qpos[i] - q_min, q_max - qpos[i])
                            min_margin = min(min_margin, margin)
                        gt_debug["margin_stats"].append(min_margin)
                        if len(gt_debug["margin_stats"]) > 10000:
                            gt_debug["margin_stats"] = gt_debug["margin_stats"][-10000:]
                        
                        if danger:
                            gt_debug["danger"] += 1
                        
                        trajectory_buffer.append(np.concatenate([qpos, qvel]))
                        if len(trajectory_buffer) > 10:
                            trajectory_buffer.pop(0)
                        
                        r1_alarm, _ = compute_r1_alarm(qpos, qvel, threshold=0.1)
                        r2_alarm, r2_risk = compute_r2_alarm(qpos, qvel, trajectory_buffer)
                        
                        r3_data = step.get("region3", {})
                        scores = r3_data.get("scores", {})
                        raw_data = r3_data.get("raw_data", {})
                        activations = raw_data.get("activations", {})
                        
                        routing_dist = compute_routing_distance(activations) if activations else 0.0
                        routing_alarm = routing_dist > CORRECT_THRESHOLDS["routing_tau"]
                        gradient_alarm = scores.get("S_logic", 1.0) < CORRECT_THRESHOLDS["S_logic_min"]
                        ood_score = compute_ood_score(qpos, qvel)
                        ood_alarm = ood_score > CORRECT_THRESHOLDS["D_ood_max"]
                        
                        episode_data.append({
                            "danger": danger,
                            "r1_alarm": r1_alarm,
                            "r2_alarm": r2_alarm,
                            "r3_alarm": routing_alarm or gradient_alarm or ood_alarm,
                            "routing_alarm": routing_alarm,
                            "gradient_alarm": gradient_alarm,
                            "ood_alarm": ood_alarm,
                        })
                        
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
                    
                    danger_seq = [d["danger"] for d in episode_data]
                    alarm_seq_full = [d["r1_alarm"] or d["r2_alarm"] or d["r3_alarm"] for d in episode_data]
                    event_metrics = compute_event_metrics(danger_seq, alarm_seq_full, lead_window=25)
                    
                    event_debug["danger_events"] += event_metrics["danger_events"]
                    event_debug["alarm_events"] += event_metrics["alarm_events"]
                    event_debug["warned_events"] += event_metrics["warned_events"]
                    event_debug["false_alarm_events"] += event_metrics["false_alarm_events"]
                    event_debug["lead_times"].extend(event_metrics["lead_times"])
                    
                    sys.stdout.flush()
                
                if ep_count % 5 == 0:
                    save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, r2_debug, ood_debug, gt_debug, event_debug, final=False)
                    print(f"  → 已保存 ({ep_count}集)\n")
                    sys.stdout.flush()
                    
            except Exception as e:
                print(f"  SKIP {ep_file}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()

print(f"\n处理完成：{ep_count}集，{total_steps}步")
save_results(layer_tpfp, r3_tpfp, ep_count, total_steps, r2_debug, ood_debug, gt_debug, event_debug, final=True)
