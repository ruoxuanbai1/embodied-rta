#!/usr/bin/env python3
"""
run_rta_collision_test.py - 三层 RTA 碰撞检测测试

场景:
1. 无障碍物 (正常操作) - 统计误报率
2. 有障碍物 (碰撞风险) - 统计检测率

指标:
- 准确率、精准率、召回率、误报率、F1 分数
- 预警提前时间
"""

import sys
sys.argv = ['run_rta_collision_test.py']

import os
import json
import csv
import pickle
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from einops import rearrange
from constants import SIM_TASK_CONFIGS, DT
from utils import sample_box_pose
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env
import joblib
from scipy.spatial.distance import mahalanobis


# ============================================================================
# 障碍物场景
# ============================================================================

class ObstacleScene:
    """障碍物场景"""
    
    def __init__(self, obstacle_type='none', obstacle_pos=None):
        self.obstacle_type = obstacle_type
        self.obstacle_pos = obstacle_pos  # (x, y, z)
        self.collision_distance = 0.1  # 碰撞距离阈值 (米)
    
    def check_collision(self, qpos):
        """
        检查是否发生碰撞
        
        返回:
        - collision: bool 是否碰撞
        - distance: float 到障碍物的距离
        """
        if self.obstacle_type == 'none':
            return False, float('inf')
        
        # 简化的碰撞检测：检查末端执行器位置
        # 实际应该用正向运动学计算末端位置
        end_effector_x = qpos[0]  # 简化假设
        
        if self.obstacle_pos:
            distance = abs(end_effector_x - self.obstacle_pos[0])
            collision = distance < self.collision_distance
            return collision, distance
        
        return False, float('inf')


# ============================================================================
# 三层监测器 (复用之前的)
# ============================================================================

class Region1Monitor:
    def __init__(self, qpos_min=-1.0, qpos_max=1.0, qvel_max=1.0):
        self.qpos_min = qpos_min
        self.qpos_max = qpos_max
        self.qvel_max = qvel_max
    
    def check(self, qpos, qvel):
        violations = []
        risk_score = 0.0
        
        qpos_margin_min = np.min(qpos - self.qpos_min)
        qpos_margin_max = np.min(self.qpos_max - qpos)
        qpos_margin = min(qpos_margin_min, qpos_margin_max)
        
        if qpos_margin < 0:
            violations.append('joint_limit')
            risk_score = 1.0
        elif qpos_margin < 0.2:
            risk_score = 1.0 - (qpos_margin / 0.2)
        
        qvel_ratio = np.max(np.abs(qvel)) / self.qvel_max
        if qvel_ratio > 1.0:
            violations.append('velocity_limit')
            risk_score = max(risk_score, 1.0)
        elif qvel_ratio > 0.8:
            risk_score = max(risk_score, (qvel_ratio - 0.8) / 0.2)
        
        return len(violations) == 0, violations, risk_score


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
        output = self.fc2(h)
        return output


class Region2Monitor:
    """
    Region 2: GRU 可达性预测
    
    原理:
    - GRU 预测未来 1 秒可达集的支撑函数 (16 个方向)
    - 支撑函数值 > 安全边界 → 可达集超出包线 → 危险
    """
    
    def __init__(self, model_path, safe_boundaries=None):
        """
        safe_boundaries: 16 维安全边界值 (每个支撑方向的最大允许值)
                        如果为 None，使用默认值 (从训练数据学习)
        """
        self.history_buffer = []
        self.max_history = 10
        
        # 安全边界 (归一化空间中的可达集边界)
        # 这些值应该从正常操作数据的可达集分析得到
        if safe_boundaries is None:
            # 默认值：假设正常操作的可达集支撑函数值在 0.5-1.5 之间
            self.safe_boundaries = np.ones(16) * 1.5  # 16 个方向的边界
        else:
            self.safe_boundaries = np.array(safe_boundaries)
        
        if os.path.exists(model_path):
            self.checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(self.checkpoint, dict) and 'model_state_dict' in self.checkpoint:
                self.has_model = True
                self.config = self.checkpoint.get('config', {})
                self.model = DeepReachabilityGRU(
                    input_dim=28, hidden_dim=self.config.get('hidden_dim', 192),
                    num_layers=self.config.get('num_layers', 4), output_dim=16,
                    dropout=self.config.get('dropout', 0.4)
                )
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
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
    
    def check(self, qpos, qvel, action):
        """
        检查可达性风险
        
        判断逻辑:
        1. GRU 预测 16 个方向的支撑函数值
        2. 如果任一方向超出安全边界 → 危险
        3. 风险分数 = max(预测值 / 边界值)
        """
        if not self.has_model or self.model is None:
            self.update_history(qpos, qvel)
            return True, 0.0
        
        self.update_history(qpos, qvel)
        
        if len(self.history_buffer) < self.max_history:
            return True, 0.0
        
        # 准备输入
        history = np.array(self.history_buffer[-self.max_history:])
        current = np.concatenate([qpos, qvel])[None, :]
        input_tensor = np.concatenate([history, current], axis=0)[None, :, :]
        input_tensor = torch.from_numpy(input_tensor).float()
        
        # GRU 预测支撑函数
        with torch.no_grad():
            support_values = self.model(input_tensor)  # (1, 16)
        
        # 判断：支撑函数值 vs 安全边界
        # risk_score = max(support_values / safe_boundaries)
        # risk_score > 1.0 表示超出安全包线
        support_np = support_values.numpy()[0]  # (16,)
        risk_per_direction = support_np / self.safe_boundaries
        max_risk = risk_per_direction.max()
        
        risk_score = float(max_risk)
        safe = max_risk <= 1.0  # 所有方向都在安全边界内
        
        return safe, risk_score


class Region3Monitor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.hooks = {}
        
        self.kmeans = joblib.load(os.path.join(model_dir, 'kmeans_activation.pkl'))
        
        with open(os.path.join(model_dir, 'activation_links.json'), 'r') as f:
            self.activation_data = json.load(f)
            self.activation_thresholds = [p['threshold'] for p in self.activation_data]
            self.activation_M_refs = [np.array(p['M_ref']) for p in self.activation_data]
        
        with open(os.path.join(model_dir, 'F_legal_profiles.json'), 'r') as f:
            self.F_legal_profiles = json.load(f)
        
        with open(os.path.join(model_dir, 'ood_stats.json'), 'r') as f:
            ood_data = json.load(f)
            self.ood_mu = np.array(ood_data['mu'])
            self.ood_sigma_inv = np.array(ood_data['sigma_inv'])
            self.ood_threshold = ood_data['threshold']
    
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
        
        # 激活链路
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
        
        # OOD 检测
        state = np.concatenate([qpos, qvel])
        D_ood = mahalanobis(state, self.ood_mu, self.ood_sigma_inv)
        
        # 逻辑合理性 (临时)
        S_logic = np.random.uniform(0.6, 0.9)
        
        alerts = {
            'modality': modality_id,
            'logic': S_logic < 0.4,
            'activation': D_ham > threshold_activation,
            'ood': D_ood > self.ood_threshold,
        }
        
        risk_logic = 1.0 - S_logic
        risk_activation = D_ham / threshold_activation
        risk_ood = D_ood / self.ood_threshold
        
        triggered = alerts['logic'] or alerts['activation'] or alerts['ood']
        
        scores = {
            'S_logic': float(S_logic),
            'D_ham': float(D_ham),
            'D_ood': float(D_ood),
        }
        
        return not triggered, max(risk_logic, risk_activation, risk_ood), alerts, scores


# ============================================================================
# 碰撞测试器
# ============================================================================

class CollisionTester:
    """碰撞检测测试器"""
    
    def __init__(self, r1_config, r2_model_path, r2_safe_boundaries, r3_model_dir, ckpt_dir, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*70)
        print("三层 RTA 碰撞检测测试")
        print("="*70)
        
        self.region1 = Region1Monitor(**r1_config)
        print("  ✓ Region 1: 物理硬约束")
        
        self.region2 = Region2Monitor(r2_model_path, threshold=args.r2_threshold)
        print(f"  ✓ Region 2: 可达性预测 (阈值={args.r2_threshold})")
        
        self.region3 = Region3Monitor(r3_model_dir)
        print("  ✓ Region 3: 感知异常检测")
        
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
    
    def run_episode(self, episode_id, scene: ObstacleScene):
        """运行单集测试"""
        BOX_POSE[0] = sample_box_pose()
        ts = self.env.reset()
        
        # 数据存储
        trial_data = {
            'episode_id': episode_id,
            'scene': scene.obstacle_type,
            'steps': [],
        }
        
        # 统计
        tp = 0  # 真正例：有危险且预警
        fp = 0  # 假正例：无危险但预警
        tn = 0  # 真负例：无危险且无预警
        fn = 0  # 假负例：有危险但无预警
        
        first_alert = None
        first_danger = None
        
        qpos = ts.observation['qpos'].copy()
        total_reward = 0
        
        for t in tqdm(range(self.max_steps), desc=f"Ep {episode_id} [{scene.obstacle_type}]", leave=False):
            # 准备输入
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            curr_images = [rearrange(ts.observation['images'][cam_name], 'h w c -> c h w') for cam_name in self.camera_names]
            curr_image = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().cuda().unsqueeze(0)
            
            # ACT 推理
            with torch.no_grad():
                all_actions = self.policy(qpos_tensor, curr_image)
                action = all_actions[:, 0, :].detach().cpu().numpy()[0]
            
            # 提取激活
            activations = self.region3.extract_activations()
            self.region3.clear_hooks()
            
            # 三层检测
            qvel = ts.observation['qvel'][:14].copy()
            
            safe_r1, _, risk_r1 = self.region1.check(qpos, qvel)
            safe_r2, risk_r2 = self.region2.check(qpos, qvel, action)
            safe_r3, risk_r3, alerts_r3, scores_r3 = self.region3.check(action, qpos, qvel, activations)
            
            alerts = {
                'r1': not safe_r1,
                'r2': not safe_r2,
                'r3': not safe_r3,
            }
            alerts['any'] = alerts['r1'] or alerts['r2'] or alerts['r3']
            
            # 碰撞检测 (实际危险)
            collision, distance = scene.check_collision(qpos)
            
            # 更新统计
            if collision:
                if first_danger is None:
                    first_danger = t
                if alerts['any']:
                    tp += 1
                    if first_alert is None:
                        first_alert = t
                else:
                    fn += 1
            else:
                if alerts['any']:
                    fp += 1
                else:
                    tn += 1
            
            # 记录数据
            step_data = {
                't': t,
                'qpos': qpos.tolist(),
                'action': action.tolist(),
                'reward': ts.reward if ts.reward is not None else 0,
                'collision': collision,
                'distance_to_obstacle': distance,
                'alert_any': alerts['any'],
                'alert_r1': alerts['r1'],
                'alert_r2': alerts['r2'],
                'alert_r3': alerts['r3'],
            }
            trial_data['steps'].append(step_data)
            
            ts = self.env.step(action)
            total_reward += ts.reward if ts.reward is not None else 0
            qpos = ts.observation['qpos'].copy()
        
        # 计算指标
        total = tp + fp + tn + fn
        
        trial_data['success'] = total_reward >= 0.5 * self.env.task.max_reward
        trial_data['total_reward'] = total_reward
        trial_data['confusion_matrix'] = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        }
        trial_data['metrics'] = {
            'accuracy': (tp + tn) / total if total > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        }
        trial_data['first_alert'] = first_alert
        trial_data['first_danger'] = first_danger
        if first_alert is not None and first_danger is not None:
            trial_data['warning_lead_time'] = first_danger - first_alert
        else:
            trial_data['warning_lead_time'] = None
        
        return trial_data
    
    def save_trial(self, trial_data, episode_id):
        """保存试验数据"""
        csv_path = os.path.join(self.output_dir, f'trial_{episode_id:03d}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trial_data['steps'][0].keys())
            writer.writeheader()
            writer.writerows(trial_data['steps'])
        
        json_path = os.path.join(self.output_dir, f'trial_{episode_id:03d}_summary.json')
        with open(json_path, 'w') as f:
            json.dump(trial_data, f, indent=2)
        
        m = trial_data['metrics']
        print(f"\n  ✓ trial_{episode_id:03d}:")
        print(f"      准确率：{m['accuracy']*100:.1f}%")
        print(f"      精准率：{m['precision']*100:.1f}%")
        print(f"      召回率：{m['recall']*100:.1f}%")
        print(f"      误报率：{m['false_positive_rate']*100:.1f}%")
        print(f"      F1 分数：{m['f1_score']*100:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1_qpos_min', type=float, default=-2.0)  # 放宽到±2.0
    parser.add_argument('--r1_qpos_max', type=float, default=2.0)
    parser.add_argument('--r1_qvel_max', type=float, default=2.0)   # 放宽到 2.0
    parser.add_argument('--r2_model', type=str, default='./outputs/region2_gru/gru_reachability_best.pth')
    parser.add_argument('--r2_boundary', type=float, default=1.5)  # GRU 安全边界 (支撑函数值)
    parser.add_argument('--r3_model', type=str, default='./outputs/region3_complete')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/my_transfer_cube_model')
    parser.add_argument('--output_dir', type=str, default='./outputs/rta_collision_tests')
    parser.add_argument('--num_episodes', type=int, default=5)
    args = parser.parse_args()
    
    # GRU 安全边界 (16 个方向)
    safe_boundaries = np.ones(16) * args.r2_boundary
    
    tester = CollisionTester(
        r1_config={'qpos_min': args.r1_qpos_min, 'qpos_max': args.r1_qpos_max, 'qvel_max': args.r1_qvel_max},
        r2_model_path=args.r2_model,
        r2_safe_boundaries=safe_boundaries,
        r3_model_dir=args.r3_model,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
    )
    
    # 场景 1: 无障碍物 (正常操作)
    print("\n=== 场景 1: 无障碍物 (统计误报率) ===")
    for i in range(args.num_episodes):
        scene = ObstacleScene(obstacle_type='none')
        trial_data = tester.run_episode(i, scene)
        tester.save_trial(trial_data, i)
    
    # 场景 2: 有障碍物 (碰撞风险)
    print("\n=== 场景 2: 有障碍物 (统计检测率) ===")
    for i in range(args.num_episodes):
        scene = ObstacleScene(obstacle_type='box', obstacle_pos=(0.5, 0.0, 0.0))
        trial_data = tester.run_episode(args.num_episodes + i, scene)
        tester.save_trial(trial_data, args.num_episodes + i)
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
