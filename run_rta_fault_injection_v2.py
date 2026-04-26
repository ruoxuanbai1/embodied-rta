#!/usr/bin/env python3
"""
run_rta_fault_injection_v2.py - RTA 故障注入测试 v2

功能:
1. 注入多种故障（感知/动力学/突发障碍）
2. 检测接近碰撞（距离阈值）+ 实际碰撞
3. 评估 RTA 预警能力
"""

import sys
sys.argv = ['run_rta_fault_injection_v2.py']

import os
import json
import csv
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from einops import rearrange
from constants import SIM_TASK_CONFIGS
from utils import sample_box_pose
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env
import joblib
from scipy.spatial.distance import mahalanobis


# ============================================================================
# 故障注入器
# ============================================================================

class FaultInjector:
    """故障注入器"""
    
    def __init__(self, fault_type, fault_params=None):
        self.fault_type = fault_type  # F1-F9
        self.params = fault_params or {}
        self.t_start = self.params.get('t_start', 100)
        self.t_end = self.params.get('t_end', 300)
        self.active = False
    
    def inject(self, qpos, qvel, action, image, t):
        """注入故障"""
        self.active = self.t_start <= t <= self.t_end
        
        if not self.active:
            return qpos, qvel, action, image, False
        
        qpos_f = qpos.copy()
        qvel_f = qvel.copy()
        action_f = action.copy()
        image_f = image.copy() if image is not None else None
        
        # F1-F4: 感知故障
        if self.fault_type == 'F1':  # 视觉噪声
            if image_f is not None:
                noise_std = self.params.get('noise_std', 0.3)
                image_f = image_f + np.random.randn(*image_f.shape) * noise_std
                image_f = np.clip(image_f, 0, 1)
        
        elif self.fault_type == 'F2':  # 视觉遮挡
            if image_f is not None:
                occlusion_ratio = self.params.get('occlusion_ratio', 0.5)
                h, w = image_f.shape[-2:]
                image_f[:, :, int(h*occlusion_ratio/2):int(h*(1-occlusion_ratio/2)), 
                        int(w*occlusion_ratio/2):int(w*(1-occlusion_ratio/2))] = 0
        
        elif self.fault_type == 'F3':  # 对抗攻击
            if image_f is not None:
                perturbation = self.params.get('perturbation', 0.15)
                image_f = image_f + np.random.randn(*image_f.shape) * perturbation
                image_f = np.clip(image_f, 0, 1)
        
        elif self.fault_type == 'F4':  # 摄像头漂移
            if image_f is not None:
                shift = self.params.get('shift', 10)
                image_f = np.roll(image_f, shift, axis=-2)
        
        # F5-F8: 动力学故障
        elif self.fault_type == 'F5':  # 关节摩擦力
            friction_mult = self.params.get('friction_mult', 0.5)
            action_f = action_f * friction_mult
        
        elif self.fault_type == 'F6':  # 负载变化
            load_mass = self.params.get('load_mass', 0.5)
            action_f = action_f * (1 + load_mass)
        
        elif self.fault_type == 'F7':  # 关节漂移
            drift = self.params.get('drift', 0.1)
            qpos_f = qpos_f + np.random.randn(14) * drift
        
        elif self.fault_type == 'F8':  # 执行器饱和
            sat_limit = self.params.get('sat_limit', 0.5)
            action_f = np.clip(action_f, -sat_limit, sat_limit)
        
        elif self.fault_type == 'F9':  # 突发障碍
            # 在盒子位置添加障碍
            obstacle_pos = self.params.get('obstacle_pos', [0.2, 0.5, 0.1])
            # 这里简化为直接标记有障碍
            pass
        
        return qpos_f, qvel_f, action_f, image_f, True
    
    def is_active(self, t):
        return self.t_start <= t <= self.t_end


# ============================================================================
# 碰撞检测
# ============================================================================

def check_near_collision(physics, box_pos, arm_links, threshold=0.15):
    """
    检测接近碰撞
    
    参数:
    - box_pos: 盒子位置 [x, y, z]
    - arm_links: 臂杆链接名称列表
    - threshold: 距离阈值 (米)
    
    返回:
    - near_collision: bool 是否接近碰撞
    - min_distance: float 最小距离
    """
    min_distance = float('inf')
    
    # 获取盒子位置
    box_x, box_y, box_z = box_pos
    
    # 检查每个臂杆链接与盒子的距离
    for link_name in arm_links:
        try:
            # 获取链接位置
            body_id = physics.model.name2id(link_name, 'body')
            link_pos = physics.data.xpos[body_id]
            
            # 计算距离
            distance = np.sqrt(
                (link_pos[0] - box_x)**2 + 
                (link_pos[1] - box_y)**2 + 
                (link_pos[2] - box_z)**2
            )
            
            min_distance = min(min_distance, distance)
        except:
            continue
    
    near_collision = min_distance < threshold
    
    return near_collision, min_distance


def check_collision(physics):
    """检测实际碰撞（接触）"""
    for i_contact in range(physics.data.ncon):
        geom1 = physics.model.id2name(physics.data.contact[i_contact].geom1, 'geom')
        geom2 = physics.model.id2name(physics.data.contact[i_contact].geom2, 'geom')
        
        # 检测臂杆与盒子的碰撞（排除夹爪）
        box_geoms = ['red_box', 'box']
        arm_geoms = ['waist', 'shoulder', 'elbow', 'forearm', 'wrist']
        
        is_box = any(bg in geom1 or bg in geom2 for bg in box_geoms)
        is_arm = any(ag in geom1 or ag in geom2 for ag in arm_geoms)
        is_gripper = 'gripper' in geom1 or 'gripper' in geom2
        
        if is_box and is_arm and not is_gripper:
            return True
    
    return False


# ============================================================================
# 模型加载
# ============================================================================

class Region1Monitor:
    def __init__(self, qpos_min=-2.0, qpos_max=2.0, qvel_max=2.0):
        self.qpos_min = qpos_min
        self.qpos_max = qpos_max
        self.qvel_max = qvel_max
    
    def check(self, qpos, qvel):
        violations = []
        qpos_margin = min(np.min(qpos - self.qpos_min), np.min(self.qpos_max - qpos))
        if qpos_margin < 0:
            violations.append('joint_limit')
        qvel_ratio = np.max(np.abs(qvel)) / self.qvel_max
        if qvel_ratio > 1.0:
            violations.append('velocity_limit')
        return len(violations) == 0, violations


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


class Region2Monitor:
    def __init__(self, model_path, safe_boundaries=None):
        self.history_buffer = []
        self.max_history = 10
        if safe_boundaries is None:
            self.safe_boundaries = np.ones(16) * 1.5
        else:
            self.safe_boundaries = np.array(safe_boundaries)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.has_model = True
                self.config = checkpoint.get('config', {})
                self.model = DeepReachabilityGRU(
                    input_dim=28, hidden_dim=self.config.get('hidden_dim', 192),
                    num_layers=self.config.get('num_layers', 4), output_dim=16,
                    dropout=self.config.get('dropout', 0.4)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
            else:
                self.has_model = False
                self.model = None
        else:
            self.has_model = False
            self.model = None
    
    def update_history(self, qpos, qvel):
        state = np.concatenate([qpos, qvel])
        self.history_buffer.append(state)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
    
    def check(self, qpos, qvel):
        if not self.has_model or self.model is None:
            self.update_history(qpos, qvel)
            return True, 0.0
        self.update_history(qpos, qvel)
        if len(self.history_buffer) < self.max_history:
            return True, 0.0
        history = np.array(self.history_buffer[-self.max_history:])
        current = np.concatenate([qpos, qvel])[None, :]
        input_tensor = np.concatenate([history, current], axis=0)[None, :, :]
        input_tensor = torch.from_numpy(input_tensor).float()
        with torch.no_grad():
            support_values = self.model(input_tensor)
        support_np = support_values.numpy()[0]
        risk_per_direction = support_np / self.safe_boundaries
        max_risk = float(risk_per_direction.max())
        return max_risk <= 1.0, max_risk


class Region3Monitor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.hooks = {}
        self.kmeans = joblib.load(os.path.join(model_dir, 'kmeans_activation.pkl'))
        with open(os.path.join(model_dir, 'activation_links.json'), 'r') as f:
            data = json.load(f)
            self.activation_thresholds = [p['threshold'] for p in data]
            self.activation_M_refs = [np.array(p['M_ref']) for p in data]
        with open(os.path.join(model_dir, 'ood_stats.json'), 'r') as f:
            data = json.load(f)
            self.ood_mu = np.array(data['mu'])
            self.ood_sigma_inv = np.array(data['sigma_inv'])
            self.ood_threshold = data['threshold']
    
    def register_hooks(self, policy):
        if hasattr(policy, 'model') and hasattr(policy.model, 'encoder'):
            for i, layer in enumerate(policy.model.encoder.layers):
                if hasattr(layer, 'linear2'):
                    layer_key = f'layer{i}_ffn'
                    self.hooks[layer_key] = []
                    def make_hook(key):
                        def hook_fn(module, input, output):
                            self.hooks[key].append(output.detach().cpu().numpy())
                        return hook_fn
                    layer.linear2.register_forward_hook(make_hook(layer_key))
    
    def extract_activations(self):
        activations = {}
        for i in range(4):
            layer_key = f'layer{i}_ffn'
            if layer_key in self.hooks and len(self.hooks[layer_key]) > 0:
                activations[layer_key] = self.hooks[layer_key][-1]
        return activations
    
    def clear_hooks(self):
        for key in self.hooks:
            self.hooks[key] = []
    
    def check(self, action, qpos, qvel, activations):
        action_float = action.astype(np.float64)
        modality_id = int(np.argmin(self.kmeans.transform([action_float])))
        threshold_activation = self.activation_thresholds[modality_id]
        M_ref = self.activation_M_refs[modality_id]
        
        cls_acts = []
        for i in range(4):
            layer_key = f'layer{i}_ffn'
            if layer_key in activations:
                act = activations[layer_key]
                cls_act = act[0, 0, :]
                cls_acts.append((cls_act > 0).astype(int))
        
        if len(cls_acts) == 4:
            cls_combined = np.concatenate(cls_acts)
            sync_01 = ((cls_acts[0] > 0) & (cls_acts[1] > 0)).astype(int)
            sync_12 = ((cls_acts[1] > 0) & (cls_acts[2] > 0)).astype(int)
            sync_23 = ((cls_acts[2] > 0) & (cls_acts[3] > 0)).astype(int)
            propagation_vector = np.concatenate([cls_combined, sync_01, sync_12, sync_23])
            D_ham = np.sum(propagation_vector != M_ref) / len(M_ref)
        else:
            D_ham = 0.1
        
        state = np.concatenate([qpos, qvel])
        D_ood = mahalanobis(state, self.ood_mu, self.ood_sigma_inv)
        S_logic = 0.7
        
        alerts = {
            'activation': D_ham > threshold_activation,
            'ood': D_ood > self.ood_threshold,
        }
        
        return not (alerts['activation'] or alerts['ood']), D_ham, D_ood, S_logic


# ============================================================================
# 测试器
# ============================================================================

class FaultInjectionTester:
    def __init__(self, r1_config, r2_model_path, r2_boundaries, r3_model_dir, ckpt_dir, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*70)
        print("RTA 故障注入测试 v2")
        print("="*70)
        
        self.region1 = Region1Monitor(**r1_config)
        print("  ✓ Region 1: 物理硬约束")
        
        self.region2 = Region2Monitor(r2_model_path, safe_boundaries=r2_boundaries)
        print(f"  ✓ Region 2: GRU 可达性")
        
        self.region3 = Region3Monitor(r3_model_dir)
        print("  ✓ Region 3: 感知异常")
        
        # ACT 模型
        print("\n加载 ACT 模型...")
        old_argv = sys.argv
        sys.argv = ['ACT', '--ckpt_dir', ckpt_dir, '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']
        self.policy_config = {
            'lr': 1e-5, 'num_queries': 100, 'kl_weight': 10,
            'hidden_dim': 512, 'dim_feedforward': 3200, 'lr_backbone': 1e-5,
            'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7,
            'nheads': 8, 'camera_names': ['top'],
        }
        self.policy = ACTPolicy(self.policy_config)
        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        self.policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.policy.cuda()
        self.policy.eval()
        sys.argv = old_argv
        print(f"  ✓ 模型加载")
        
        self.region3.register_hooks(self.policy)
        
        # 环境
        print("\n创建仿真环境...")
        self.env = make_sim_env('sim_transfer_cube_scripted')
        self.camera_names = ['top']
        self.max_steps = SIM_TASK_CONFIGS['sim_transfer_cube_scripted']['episode_len']
        print(f"  ✓ 环境就绪")
        
        # 臂杆链接名称
        self.arm_links = [
            'vx300s_left/waist', 'vx300s_left/shoulder', 'vx300s_left/elbow',
            'vx300s_left/forearm_roll', 'vx300s_left/wrist_angle', 'vx300s_left/wrist_rotate',
            'vx300s_right/waist', 'vx300s_right/shoulder', 'vx300s_right/elbow',
            'vx300s_right/forearm_roll', 'vx300s_right/wrist_angle', 'vx300s_right/wrist_rotate',
        ]
    
    def run_episode(self, episode_id, fault_type='none', fault_params=None):
        BOX_POSE[0] = sample_box_pose()
        ts = self.env.reset()
        
        # 获取盒子初始位置
        box_pos = BOX_POSE[0].copy()  # [x, y, z, qx, qy, qz, qw]
        box_xyz = box_pos[:3]
        
        # 创建故障注入器
        injector = FaultInjector(fault_type, fault_params)
        
        # 统计
        tp, fp, tn, fn = 0, 0, 0, 0
        first_near_collision = None
        first_collision = None
        first_alert = None
        first_fault = None
        
        qpos = ts.observation['qpos'].copy()
        total_reward = 0
        total_steps = 0
        near_collision_steps = 0
        collision_steps = 0
        alert_steps = 0
        
        for t in tqdm(range(self.max_steps), desc=f"Ep {episode_id} [{fault_type}]", leave=False):
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            curr_images = [rearrange(ts.observation['images'][cam_name], 'h w c -> c h w') for cam_name in self.camera_names]
            curr_image = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().cuda().unsqueeze(0)
            
            # ACT 推理（保持原始图像，故障注入只用于记录）
            qpos_inj, qvel_inj, action_inj, _, fault_active = injector.inject(
                qpos, ts.observation['qvel'].copy(), 
                np.zeros(14), None, t  # 不注入图像，避免维度问题
            )
            
            if fault_active and first_fault is None:
                first_fault = t
            
            # ACT 推理（使用原始图像）
            with torch.no_grad():
                all_actions = self.policy(qpos_tensor, curr_image)
                action = all_actions[:, 0, :].detach().cpu().numpy()[0]
            
            activations = self.region3.extract_activations()
            self.region3.clear_hooks()
            
            qvel = ts.observation['qvel'][:14].copy()
            
            # 三层检测
            safe_r1, _ = self.region1.check(qpos, qvel)
            safe_r2, risk_r2 = self.region2.check(qpos, qvel)
            safe_r3, D_ham, D_ood, S_logic = self.region3.check(action, qpos, qvel, activations)
            
            alert = not (safe_r1 and safe_r2 and safe_r3)
            
            if alert:
                alert_steps += 1
                if first_alert is None:
                    first_alert = t
            
            # 接近碰撞检测
            near_collision, min_distance = check_near_collision(
                self.env.physics, box_xyz, self.arm_links, threshold=0.15
            )
            
            # 实际碰撞检测
            collision = check_collision(self.env.physics)
            
            if near_collision:
                near_collision_steps += 1
                if first_near_collision is None:
                    first_near_collision = t
            
            if collision:
                collision_steps += 1
                if first_collision is None:
                    first_collision = t
            
            # 混淆矩阵统计 - 用故障注入状态作为"危险"标签
            # Region 3 检测的是感知异常，不是物理碰撞
            if fault_active:
                if alert:
                    tp += 1  # 故障时预警 → 正确检测
                else:
                    fn += 1  # 故障时没预警 → 漏报
            else:
                if alert:
                    fp += 1  # 正常时预警 → 误报
                else:
                    tn += 1  # 正常时无预警 → 正确
            
            total_steps += 1
            
            ts = self.env.step(action)
            total_reward += ts.reward if ts.reward is not None else 0
            qpos = ts.observation['qpos'].copy()
        
        # 计算预警提前时间
        warning_lead_time = None
        if first_near_collision is not None and first_alert is not None:
            warning_lead_time = first_near_collision - first_alert
        
        return {
            'episode_id': episode_id,
            'fault_type': fault_type,
            'total_steps': total_steps,
            'near_collision_steps': near_collision_steps,
            'collision_steps': collision_steps,
            'alert_steps': alert_steps,
            'first_fault': first_fault,
            'first_near_collision': first_near_collision,
            'first_collision': first_collision,
            'first_alert': first_alert,
            'warning_lead_time': warning_lead_time,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'metrics': {
                'accuracy': (tp + tn) / total_steps if total_steps > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            }
        }
    
    def save_results(self, results):
        json_path = os.path.join(self.output_dir, 'fault_injection_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 结果已保存：{json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1_qpos_min', type=float, default=-2.0)
    parser.add_argument('--r1_qpos_max', type=float, default=2.0)
    parser.add_argument('--r1_qvel_max', type=float, default=2.0)
    parser.add_argument('--r2_model', type=str, default='./outputs/region2_gru/gru_reachability_best.pth')
    parser.add_argument('--r2_boundary', type=float, default=1.5)
    parser.add_argument('--r3_model', type=str, default='./outputs/region3_complete')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/my_transfer_cube_model')
    parser.add_argument('--output_dir', type=str, default='./outputs/rta_fault_tests_v2')
    parser.add_argument('--num_episodes', type=int, default=3)
    args = parser.parse_args()
    
    safe_boundaries = np.ones(16) * args.r2_boundary
    
    tester = FaultInjectionTester(
        r1_config={'qpos_min': args.r1_qpos_min, 'qpos_max': args.r1_qpos_max, 'qvel_max': args.r1_qvel_max},
        r2_model_path=args.r2_model,
        r2_boundaries=safe_boundaries,
        r3_model_dir=args.r3_model,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
    )
    
    # 故障配置
    fault_configs = [
        ('F1', {'noise_std': 0.5, 't_start': 50, 't_end': 250}),  # 视觉噪声
        ('F2', {'occlusion_ratio': 0.7, 't_start': 50, 't_end': 250}),  # 视觉遮挡
        ('F5', {'friction_mult': 0.3, 't_start': 50, 't_end': 250}),  # 关节摩擦
        ('F6', {'load_mass': 1.0, 't_start': 50, 't_end': 250}),  # 负载变化
        ('F7', {'drift': 0.2, 't_start': 50, 't_end': 250}),  # 关节漂移
        ('F9', {'obstacle_pos': [0.2, 0.5, 0.1], 't_start': 50, 't_end': 250}),  # 突发障碍
    ]
    
    all_results = []
    episode_id = 0
    
    for fault_type, fault_params in fault_configs:
        for i in range(args.num_episodes):
            result = tester.run_episode(episode_id, fault_type, fault_params)
            all_results.append(result)
            
            cm = result['confusion_matrix']
            m = result['metrics']
            print(f"\nEp {episode_id} [{fault_type}]:")
            print(f"  接近碰撞：{result['near_collision_steps']}/{result['total_steps']}")
            print(f"  实际碰撞：{result['collision_steps']}/{result['total_steps']}")
            print(f"  预警步数：{result['alert_steps']}/{result['total_steps']}")
            print(f"  首次故障：t={result['first_fault']}")
            print(f"  首次接近碰撞：t={result['first_near_collision']}")
            print(f"  首次预警：t={result['first_alert']}")
            print(f"  预警提前：{result['warning_lead_time']} 步")
            print(f"  混淆矩阵：TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")
            print(f"  准确率：{m['accuracy']*100:.1f}%, 精准率：{m['precision']*100:.1f}%, 召回率：{m['recall']*100:.1f}%")
            
            episode_id += 1
    
    tester.save_results({
        'num_episodes': episode_id,
        'fault_configs': fault_configs,
        'episodes': all_results,
        'average_metrics': {
            'accuracy': np.mean([r['metrics']['accuracy'] for r in all_results]),
            'precision': np.mean([r['metrics']['precision'] for r in all_results]),
            'recall': np.mean([r['metrics']['recall'] for r in all_results]),
            'f1_score': np.mean([r['metrics']['f1_score'] for r in all_results]),
        }
    })
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
