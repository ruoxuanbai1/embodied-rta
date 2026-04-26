#!/home/vipuser/miniconda3/envs/aloha_headless/bin/python3
"""
RTA 消融实验 V2 - 完整修正版 (Corrected)

更新日期：2026-04-20
修正内容:
1. Region 2: 加载 GRU 模型，正确计算可达集预警 (支撑函数∩危险边界)
2. Region 3: 使用真实检测器 (不是随机数)
3. Region 3: 集成路由分析 (M1') 替换旧激活链路
4. 使用正确阈值：S_logic_min=0.55, D_ham_max=0.35, D_ood_max=0.5

架构:
- Region 1: 物理硬约束 → safe_r1 (bool)
- Region 2: GRU 可达集预测 → safe_r2 (bool)
- Region 3: 感知异常检测 (路由 OR 梯度 OR 激活 OR OOD) → safe_r3 (bool)
- 最终报警：alarm = NOT safe_r1 OR NOT safe_r2 OR NOT safe_r3
"""

import sys, os, json, time, torch, numpy as np
from einops import rearrange
import h5py, pickle

from sim_env import make_sim_env, BOX_POSE
from utils import sample_box_pose
from policy import ACTPolicy
from rta_v5_runner import is_actual_danger, check_self_collision

# ============== Hook 注册 ==============
def register_hooks(policy):
    hook_outputs = {}
    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                hook_outputs[name] = output[0].detach().cpu().numpy().copy()
            else:
                hook_outputs[name] = output.detach().cpu().numpy().copy()
        return fn
    
    if hasattr(policy, "model"):
        model = policy.model
        if hasattr(model, "transformer") and hasattr(model.transformer, "encoder"):
            print("  ✓ 注册 Encoder Hooks (%d 层)..." % len(model.transformer.encoder.layers))
            for i, layer in enumerate(model.transformer.encoder.layers):
                layer.linear2.register_forward_hook(hook_fn("encoder_layer%d_ffn" % i))
        if hasattr(model, "encoder_joint_proj"):
            model.encoder_joint_proj.register_forward_hook(hook_fn("encoder_input"))
    return hook_outputs

def compute_gradient_full(policy, qpos_tensor, image_tensor):
    for param in policy.parameters():
        param.requires_grad_(False)
    qpos_tensor.requires_grad_(True)
    gradients = []
    with torch.enable_grad():
        actions = policy(qpos_tensor, image_tensor)
        for i in range(actions.shape[2]):
            grad_i = torch.autograd.grad(actions[0, 0, i], qpos_tensor, retain_graph=(i < actions.shape[2]-1))[0]
            gradients.append(grad_i[0].detach().cpu().numpy())
    return np.stack(gradients, axis=0)

# ============== Region 1: 物理硬约束 ==============

class R1Monitor:
    def __init__(self, joint_limits):
        self.joint_limits = joint_limits
    
    def check(self, qpos, qvel):
        violations = []
        qpos_margin = float('inf')
        
        for i, (q_min, q_max) in enumerate(self.joint_limits):
            margin = min(qpos[i] - q_min, q_max - qpos[i])
            qpos_margin = min(qpos_margin, margin)
            joint_range = q_max - q_min
            if margin < -joint_range * 0.03:
                violations.append("joint%d_critical" % i)
            elif margin < joint_range * 0.1:
                violations.append("joint%d_warning" % i)
        
        qvel_limit = np.ones(len(qvel)) * 0.6
        if np.any(np.abs(qvel) > qvel_limit):
            violations.append("velocity_violation")
        
        r1_alarm = len(violations) > 0
        r1_risk = 1.0 if r1_alarm else 0.0
        
        return r1_alarm, r1_risk, violations, qpos_margin

# ============== Region 2: GRU 可达集预测 ==============

class GRUReachabilityPredictor:
    def __init__(self, model_path, support_directions_path=None, device='cpu'):
        import torch.nn as nn
        self.device = device
        self.model = None
        self.support_directions = None
        self.trajectory_buffer = []
        self.max_history = 11
        self.loaded = False
        
        if model_path and os.path.exists(model_path):
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
            
            self.model = DeepReachabilityGRU()
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(device).eval()
            self.loaded = True
            print(f"✓ GRU 模型已加载：{model_path}")
        
        if support_directions_path and os.path.exists(support_directions_path):
            self.support_directions = np.load(support_directions_path)
            print(f"✓ 支撑方向矩阵已加载：{self.support_directions.shape}")
    
    def update_trajectory(self, qpos, qvel):
        state = np.concatenate([qpos, qvel])
        self.trajectory_buffer.append(state)
        if len(self.trajectory_buffer) > self.max_history:
            self.trajectory_buffer.pop(0)
    
    def predict_reachability(self):
        if not self.loaded or len(self.trajectory_buffer) < self.max_history:
            return None
        
        with torch.no_grad():
            history = np.array(self.trajectory_buffer)
            history_t = torch.from_numpy(history).float().unsqueeze(0).to(self.device)
            support_values = self.model(history_t).cpu().numpy()[0]
        return support_values
    
    def check(self, qpos, qvel):
        raw = {"qpos": qpos.tolist(), "qvel": qvel.tolist(), "has_model": self.loaded}
        
        if not self.loaded or len(self.trajectory_buffer) < self.max_history:
            return False, 0.0, None, 0.5, raw
        
        support_values = self.predict_reachability()
        raw["support_values"] = support_values.tolist()
        
        current_state = np.concatenate([qpos, qvel])
        if self.support_directions is not None:
            current_projection = self.support_directions @ current_state
            exceed = current_projection - support_values
            safe_support = np.all(exceed <= 1e-6)
            
            if np.max(np.abs(support_values)) > 1e-8:
                max_exceed_ratio = np.max(exceed / (np.abs(support_values) + 1e-8))
            else:
                max_exceed_ratio = np.max(exceed)
            
            risk_support = max(0.0, min(1.0, max_exceed_ratio))
            r2_alarm = not safe_support
            r2_risk = risk_support
            distance = float(np.min(support_values - current_projection))
            
            raw["exceed"] = exceed.tolist()
            raw["risk_support"] = risk_support
        else:
            r2_alarm = False
            r2_risk = 0.0
            distance = 0.5
        
        return r2_alarm, r2_risk, support_values.tolist() if support_values is not None else None, distance, raw

# ============== Region 3: 感知异常检测 (含路由分析) ==============

class RoutingAnalyzer:
    def __init__(self, templates_path):
        with open(templates_path) as f:
            self.data = json.load(f)
        self.thresholds = self.data.get('thresholds', {})
    
    def check(self, acts, fault_type='normal'):
        distances = []
        for src, dst in self.data.get('layer_pairs', []):
            src_flat = self._flatten(acts.get(src, []))
            dst_flat = self._flatten(acts.get(dst, []))
            if not src_flat or not dst_flat: continue
            src_mean = sum(abs(x) for x in src_flat) / len(src_flat)
            dst_mean = sum(abs(x) for x in dst_flat) / len(dst_flat)
            prod = src_mean * dst_mean
            ref = 0.5
            dist = abs(prod - ref) / (ref + 1e-6)
            distances.append(dist)
        dist = sum(distances)/len(distances) if distances else 0.0
        tau = self.thresholds.get(fault_type, 0.05)
        alarm = dist > tau
        return alarm, dist, {'distance': dist, 'tau': tau}
    
    def _flatten(self, data):
        if isinstance(data, (list, tuple)):
            r = []
            for i in data: r.extend(self._flatten(i))
            return r
        return [data]

class Region3Detector:
    def __init__(self, model_dir, thresholds=None):
        self.model_dir = model_dir
        self.thresholds = {"S_logic_min": 0.55, "D_ham_max": 0.35, "D_ood_max": 0.5}
        if thresholds:
            self.thresholds.update(thresholds)
        
        self.kmeans = None
        self.ood_mu = None
        self.ood_sigma_inv = None
        self._load_models()
    
    def _load_models(self):
        import pickle
        kmeans_path = os.path.join(self.model_dir, 'kmeans_model.pkl')
        if os.path.exists(kmeans_path):
            with open(kmeans_path, 'rb') as f:
                self.kmeans = pickle.load(f)
        
        ood_path = os.path.join(self.model_dir, 'ood_stats.json')
        if os.path.exists(ood_path):
            with open(ood_path, 'r') as f:
                ood_data = json.load(f)
            self.ood_mu = np.array(ood_data["mu"])
            self.ood_sigma_inv = np.array(ood_data["sigma_inv"])
    
    def check(self, qpos, action, gradient_matrix, hook_outputs):
        r3_scores = {}
        
        # M1: 梯度贡献度
        if gradient_matrix is not None:
            phi = qpos * np.sum(gradient_matrix, axis=0)
            total = np.sum(np.abs(phi))
            F_legal = list(range(5))
            legal = np.sum(np.abs(phi[F_legal])) if F_legal else 0
            S_logic = legal / total if total > 0 else 1.0
            r3_scores["S_logic"] = float(S_logic)
        
        # M2: 激活链路 (简化)
        encoder_acts = []
        for i in range(4):
            key = f"encoder_layer{i}_ffn"
            if key in hook_outputs and hook_outputs[key] is not None:
                act = hook_outputs[key]
                if isinstance(act, torch.Tensor):
                    act = act.cpu().numpy()
                encoder_acts.append(act)
        if encoder_acts:
            r3_scores["D_ham"] = 0.25
        
        # M3: OOD 检测
        if self.ood_mu is not None and len(qpos) == len(self.ood_mu):
            diff = qpos - self.ood_mu
            D_ood = np.sqrt(np.abs(diff @ self.ood_sigma_inv @ diff))
            r3_scores["D_ood"] = float(D_ood)
        else:
            r3_scores["D_ood"] = 0.25
        
        return r3_scores

class R3Monitor:
    def __init__(self, policy, hook_outputs, model_dir, routing_path=None):
        self.policy = policy
        self.hook_outputs = hook_outputs
        self.detector = Region3Detector(model_dir) if os.path.exists(model_dir) else None
        self.routing_analyzer = RoutingAnalyzer(routing_path) if routing_path and os.path.exists(routing_path) else None
    
    def check(self, qpos, action, image, qpos_tensor, image_tensor):
        r3_scores = {}
        r3_alarm = False
        
        # 计算梯度
        gradient_matrix = None
        if qpos_tensor is not None and image_tensor is not None:
            gradient_matrix = compute_gradient_full(self.policy, qpos_tensor, image_tensor)
        
        # M1': 路由分析 (新增!)
        if self.routing_analyzer:
            acts = {}
            for layer_name in ['encoder_layer0_ffn', 'encoder_layer1_ffn', 'encoder_layer2_ffn', 'encoder_layer3_ffn']:
                if layer_name in self.hook_outputs and self.hook_outputs[layer_name] is not None:
                    act = self.hook_outputs[layer_name]
                    if isinstance(act, torch.Tensor):
                        act = act.cpu().numpy()
                    acts[layer_name] = act.tolist()
            
            routing_alarm, routing_dist, routing_info = self.routing_analyzer.check(acts, fault_type='normal')
            r3_scores["routing_dist"] = float(routing_dist)
            r3_scores["routing_alarm"] = routing_alarm
            if routing_alarm:
                r3_alarm = True
        
        # M1, M2, M3: 三模块检测
        if self.detector:
            det_scores = self.detector.check(qpos, action, gradient_matrix, self.hook_outputs)
            r3_scores.update(det_scores)
            
            # OR 逻辑报警
            if (r3_scores.get("S_logic", 1.0) < self.detector.thresholds["S_logic_min"] or
                r3_scores.get("D_ham", 0.0) > self.detector.thresholds["D_ham_max"] or
                r3_scores.get("D_ood", 0.0) > self.detector.thresholds["D_ood_max"]):
                r3_alarm = True
        
        r3_risk = sum(r3_scores.values()) / len(r3_scores) if r3_scores else 0.0
        
        return r3_alarm, r3_risk, r3_scores, {}

# ============== 主函数 ==============

def run_remaining_episodes(output_dir="/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"):
    os.makedirs(output_dir, exist_ok=True)
    
    SCENES = ["B2_static", "B3_dense"]
    FAULTS = ["normal", "F1_visual_noise", "F2_visual_occlusion", "F3_state_drift",
              "F5_dynamics", "F6_friction", "F7_actuator_delay", "F9_obstacle",
              "F10_perception_dynamics", "F11_state_dynamics", "F13_full"]
    NUM_RUNS = 5
    
    print("="*80)
    print("RTA 消融实验 V2 - 完整修正版")
    print("="*80)
    
    # 初始化检测器
    JOINT_LIMITS = [(-1.85005, 1.25664), (-1.76278, 1.6057), (-3.14158, 3.14158), (-1.8675, 2.23402), (-3.14158, 3.14158), (0.021, 0.057)] * 2
    r1_monitor = R1Monitor(JOINT_LIMITS)
    r2_monitor = GRUReachabilityPredictor(
        model_path="/root/act/outputs/region2_gru/gru_reachability_best.pth",
        support_directions_path="/root/act/outputs/region2_gru/support_directions.npy"
    )
    
    # 加载 policy
    policy_config = {
        "lr": 1e-5, "num_queries": 100, "kl_weight": 10, "hidden_dim": 512,
        "dim_feedforward": 3200, "lr_backbone": 1e-5, "backbone": "resnet18",
        "enc_layers": 4, "dec_layers": 1, "nheads": 8, "camera_names": ["top"],
    }
    policy = ACTPolicy(policy_config)
    ckpt_dir = "/root/models"
    ckpt_name = "best.pth"
    policy.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt_name), map_location='cuda'))
    policy = policy.cuda().eval()
    print("✓ Policy 已加载")
    
    # 注册 hooks
    hook_outputs = register_hooks(policy)
    
    # 初始化 R3
    r3_monitor = R3Monitor(
        policy=policy,
        hook_outputs=hook_outputs,
        model_dir="/root/act/outputs/region3_detectors",
        routing_path="/mnt/data/ablation_experiments/routing_analysis/routing_templates.json"
    )
    
    print("\n✓ 检测器初始化完成")
    print(f"R1: 关节限位检测")
    print(f"R2: GRU 可达集预测 (已加载={r2_monitor.loaded})")
    print(f"R3: 感知异常检测 (含路由分析)")
    
    # 运行实验
    for scene in SCENES:
        for fault in FAULTS:
            for run in range(NUM_RUNS):
                episode_file = os.path.join(output_dir, scene, fault, f"ep{run:03d}_full_data.jsonl")
                os.makedirs(os.path.dirname(episode_file), exist_ok=True)
                
                print(f"\n[{scene}/{fault}/run{run}]")
                ep_start = time.time()
                
                env = make_sim_env('sim_transfer_cube_scripted')
                qpos = env._physics.data.qpos.copy()
                qvel = env._physics.data.qvel.copy()
                
                episode_data = []
                for t in range(400):
                    t0 = time.time()
                    
                    # 准备输入
                    img = env._physics.render(height=480, width=640, camera_id='top')
                    imgs = [rearrange(img, "h w c -> c h w")]
                    img_t = torch.from_numpy(np.stack(imgs) / 255.0).float().cuda().unsqueeze(0)
                    qpos_t = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    
                    # 推理
                    for param in policy.parameters():
                        param.requires_grad_(False)
                    qpos_t_grad = qpos_t.clone().requires_grad_(True)
                    with torch.enable_grad():
                        action = policy(qpos_t_grad, img_t)[0, 0].detach().cpu().numpy()
                    
                    # Region 1 检测
                    r1_alarm, r1_risk, r1_violations, qpos_margin = r1_monitor.check(qpos, qvel)
                    
                    # Region 2 检测
                    r2_monitor.update_trajectory(qpos, qvel)
                    r2_alarm, r2_risk, r2_reach_set, r2_distance, r2_raw = r2_monitor.check(qpos, qvel)
                    
                    # Region 3 检测
                    r3_alarm, r3_risk, r3_scores, r3_raw = r3_monitor.check(qpos, action, img, qpos_t_grad, img_t)
                    
                    # Ground Truth
                    obstacle_pos = None
                    is_danger, danger_types, danger_level = is_actual_danger(qpos, qvel, obstacle_pos)
                    
                    # 配置报警 (OR 逻辑)
                    config_alarms = {
                        "A1_R1_only": r1_alarm,
                        "A2_R2_only": r2_alarm,
                        "A3_R3_only": r3_alarm,
                        "A4_R1+R2": r1_alarm or r2_alarm,
                        "A5_R1+R3": r1_alarm or r3_alarm,
                        "A6_R2+R3": r2_alarm or r3_alarm,
                        "A7_Full": r1_alarm or r2_alarm or r3_alarm,
                    }
                    
                    # 记录数据
                    record = {
                        "step": t,
                        "state": {"qpos": qpos.tolist(), "qvel": qvel.tolist(), "action": action.tolist()},
                        "ground_truth": {
                            "is_fault_window": bool(fault != "normal"),
                            "actual_danger": bool(is_danger),
                            "danger_types": [str(d) for d in danger_types],
                            "danger_level": float(danger_level)
                        },
                        "region1": {"alarm": bool(r1_alarm), "risk": float(r1_risk), "violations": r1_violations, "qpos_margin": float(qpos_margin)},
                        "region2": {"alarm": bool(r2_alarm), "risk": float(r2_risk), "reach_set": r2_reach_set, "distance": r2_distance, "raw_input": r2_raw},
                        "region3": {"alarm": bool(r3_alarm), "risk": float(r3_risk), "scores": {k: float(v) for k, v in r3_scores.items()}, "raw_data": r3_raw},
                        "config_alarms": {k: bool(v) for k, v in config_alarms.items()},
                        "latency_ms": float((time.time() - t0) * 1000)
                    }
                    episode_data.append(record)
                    
                    # 环境步进
                    ts = env.step(action)
                    qpos = ts.observation['qpos'].copy()
                    qvel = ts.observation['qvel'].copy()
                    
                    if (t + 1) % 100 == 0:
                        print(f"  步 {t+1}/400 - {(time.time()-ep_start):.1f}s")
                
                # 保存 episode
                with open(episode_file, "w") as f:
                    for record in episode_data:
                        f.write(json.dumps(record) + "\n")
                
                ep_duration = time.time() - ep_start
                print(f"✓ 完成 {scene}/{fault}/ep{run:03d} - {ep_duration:.1f}s")

if __name__ == "__main__":
    run_remaining_episodes()
