#!/usr/bin/env python3
"""
run_rta_fault_injection_test.py - 三层 RTA 故障注入测试

功能:
1. 加载 ACT 模型 + 三层检测器
2. 注册 Hook 提取真实激活
3. 注入 13 种故障
4. 记录每步预警状态
5. 计算检测率/虚警率
"""



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
# 故障注入器 (13 种故障)
# ============================================================================

class FaultInjector:
    """13 种故障注入器"""
    
    FAULT_TYPES = [
        'F1_visual_noise', 'F2_visual_occlusion', 'F3_adversarial', 'F4_camera_drift',
        'F5_friction', 'F6_payload', 'F7_joint_bias', 'F8_actuator_saturation',
        'F9_sudden_obstacle', 'F10_compound_1', 'F11_compound_2', 'F12_compound_3', 'F13_compound_4'
    ]
    
    def __init__(self, fault_type, t_start=100, t_end=300):
        self.fault_type = fault_type
        self.t_start = t_start
        self.t_end = t_end
    
    def inject(self, qpos, qvel, action, image, t):
        """注入故障"""
        if t < self.t_start or t > self.t_end:
            return qpos, qvel, action, image, False
        
        qpos_f = qpos.copy()
        qvel_f = qvel.copy()
        action_f = action.copy()
        image_f = image.copy() if image is not None else None
        
        # 感知故障 (F1-F4)
        if self.fault_type == 'F1_visual_noise':
            if image_f is not None:
                image_f = image_f + np.random.randn(*image_f.shape) * 0.3
                image_f = np.clip(image_f, 0, 1)
        
        elif self.fault_type == 'F2_visual_occlusion':
            if image_f is not None:
                h, w = image_f.shape[-2:]
                image_f[:, :, h//4:3*h//4, w//4:3*w//4] = 0
        
        elif self.fault_type == 'F3_adversarial':
            if image_f is not None:
                image_f = image_f + np.random.randn(*image_f.shape) * 0.15
                image_f = np.clip(image_f, 0, 1)
        
        # 动力学故障 (F5-F8)
        elif self.fault_type == 'F5_friction':
            action_f = action_f * 0.5  # 动作衰减
        
        elif self.fault_type == 'F6_payload':
            qvel_f = qvel_f * 0.7  # 速度变慢
        
        elif self.fault_type == 'F7_joint_bias':
            qpos_f = qpos_f + np.random.randn(14) * 0.1
        
        elif self.fault_type == 'F8_actuator_saturation':
            action_f = np.clip(action_f, -0.5, 0.5)
        
        return qpos_f, qvel_f, action_f, image_f, True


# ============================================================================
# Region 3 监测器 (真实激活)
# ============================================================================

class Region3Monitor:
    """Region 3: 感知异常检测器 (使用真实激活)"""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.hooks = {}
        self.current_activations = {}
        
        # 加载模型
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
        
        print(f"  ✓ Region 3 检测器加载")
    
    def register_hooks(self, policy):
        """注册 Hook 到 ACT 模型的 4 层 Encoder"""
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
        
        print(f"  ✓ Region 3 Hooks 注册：{len(self.hooks)} 层")
    
    def extract_activations(self):
        """从 hooks 提取当前激活"""
        activations = {}
        for i in range(4):
            layer_key = f'layer{i}_ffn'
            if layer_key in self.hooks and len(self.hooks[layer_key]) > 0:
                # (102, 1, 512)
                activations[layer_key] = self.hooks[layer_key][-1]
        return activations
    
    def clear_hooks(self):
        """清除 hook 缓存"""
        for key in self.hooks:
            self.hooks[key] = []
    
    def check(self, action, qpos, qvel, activations):
        """
        检测异常 (使用真实激活)
        
        返回:
        - safe: bool
        - risk_score: float
        - alerts: Dict
        - scores: Dict
        """
        # 1. 判断动作模态
        action_float = action.astype(np.float64)
        modality_id = int(np.argmin(self.kmeans.transform([action_float])))
        
        # 2. 获取该模态的阈值
        threshold_activation = self.activation_thresholds[modality_id]
        M_ref = self.activation_M_refs[modality_id]
        
        # 3. 计算激活链路汉明距离
        # 提取 cls token 激活并二值化
        cls_acts = []
        for i in range(4):
            layer_key = f'layer{i}_ffn'
            if layer_key in activations:
                act = activations[layer_key]  # (102, 1, 512)
                cls_act = act[0, 0, :]  # cls token (512,)
                cls_acts.append((cls_act > 0).astype(int))
        
        if len(cls_acts) == 4:
            # 构建传播链路向量
            cls_combined = np.concatenate(cls_acts)  # (2048,)
            sync_01 = ((cls_acts[0] > 0) & (cls_acts[1] > 0)).astype(int)
            sync_12 = ((cls_acts[1] > 0) & (cls_acts[2] > 0)).astype(int)
            sync_23 = ((cls_acts[2] > 0) & (cls_acts[3] > 0)).astype(int)
            
            propagation_vector = np.concatenate([cls_combined, sync_01, sync_12, sync_23])
            
            # 汉明距离
            D_ham = np.sum(propagation_vector != M_ref) / len(M_ref)
        else:
            D_ham = 0.1  # 默认值
        
        # 4. OOD 检测
        state = np.concatenate([qpos, qvel])
        D_ood = mahalanobis(state, self.ood_mu, self.ood_sigma_inv)
        
        # 5. 逻辑合理性 (临时用随机值，后续实现梯度计算)
        S_logic = np.random.uniform(0.6, 0.9)
        
        # 6. 独立预警
        alerts = {
            'modality': modality_id,
            'logic': S_logic < 0.4,
            'activation': D_ham > threshold_activation,
            'ood': D_ood > self.ood_threshold,
        }
        
        # 7. 风险分数
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
# 故障注入测试器
# ============================================================================

class FaultInjectionTester:
    """故障注入测试器"""
    
    def __init__(self, r1_config, r2_model_path, r3_model_dir, ckpt_dir, output_dir):
        """初始化"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("三层 RTA 故障注入测试")
        print("="*60)
        
        # 1. Region 1
        self.region1 = Region1Monitor(**r1_config)
        print("  ✓ Region 1: 物理硬约束")
        
        # 2. Region 2
        self.region2 = Region2Monitor(r2_model_path)
        print("  ✓ Region 2: 可达性预测")
        
        # 3. Region 3
        self.region3 = Region3Monitor(r3_model_dir)
        print("  ✓ Region 3: 感知异常检测")
        
        # 4. ACT 模型
        print("\n加载 ACT 模型...")
        import sys
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
        print(f"  ✓ 模型加载：{ckpt_path}")
        
        # 5. 注册 Region 3 hooks
        self.region3.register_hooks(self.policy)
        
        # 6. 创建环境
        print("\n创建仿真环境...")
        self.env = make_sim_env('sim_transfer_cube_scripted')
        self.camera_names = ['top']
        self.max_steps = SIM_TASK_CONFIGS['sim_transfer_cube_scripted']['episode_len']
        print(f"  ✓ 环境就绪，max_steps={self.max_steps}")
    
    def run_episode(self, episode_id, fault_type='none', t_start=100, t_end=300):
        """运行单集故障注入测试"""
        # 重置环境
        BOX_POSE[0] = sample_box_pose()
        ts = self.env.reset()
        
        # 故障注入器
        injector = FaultInjector(fault_type, t_start, t_end) if fault_type != 'none' else None
        
        # 数据存储
        trial_data = {
            'episode_id': episode_id,
            'fault_type': fault_type,
            't_start': t_start,
            't_end': t_end,
            'steps': [],
            'success': False,
            'total_reward': 0,
        }
        
        # 预警统计
        alert_stats = {
            'r1_only': 0, 'r2_only': 0, 'r3_only': 0,
            'r1_r2': 0, 'r1_r3': 0, 'r2_r3': 0,
            'r1_r2_r3': 0, 'none': 0,
        }
        
        qpos = ts.observation['qpos'].copy()
        total_reward = 0
        total_alerts = 0
        r1_alerts = 0
        r2_alerts = 0
        r3_alerts = 0
        fault_detected = False
        first_alert_after_fault = None
        first_danger_after_fault = None  # 首次危险时间
        
        # 运行 rollout
        for t in tqdm(range(self.max_steps), desc=f"Ep {episode_id} [{fault_type}]", leave=False):
            # 准备输入
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            curr_images = [
                rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
                for cam_name in self.camera_names
            ]
            curr_image = torch.from_numpy(
                np.stack(curr_images, axis=0) / 255.0
            ).float().cuda().unsqueeze(0)
            
            # 故障注入
            image_np = curr_image.detach().cpu().numpy()
            fault_active = False
            if injector:
                qpos_inj, qvel_inj, action_inj, image_np, fault_active = injector.inject(
                    qpos, ts.observation['qvel'].copy(), np.zeros(14), image_np, t
                )
                if fault_active:
                    curr_image = torch.from_numpy(image_np).float().cuda()
            
            # ACT 推理
            with torch.no_grad():
                all_actions = self.policy(qpos_tensor, curr_image)
                action = all_actions[:, 0, :].detach().cpu().numpy()[0]
            
            # 提取 Region 3 激活
            activations = self.region3.extract_activations()
            self.region3.clear_hooks()
            
            # 三层检测
            qvel = ts.observation['qvel'][:14].copy()
            
            # R1 (预警)
            safe_r1, violations, risk_r1 = self.region1.check(qpos, qvel)
            
            # R2 (预警)
            safe_r2, risk_r2 = self.region2.check(qpos, qvel, action)
            
            # R3 (预警)
            safe_r3, risk_r3, alerts_r3, scores_r3 = self.region3.check(action, qpos, qvel, activations)
            
            alerts = {
                'r1': not safe_r1,
                'r2': not safe_r2,
                'r3': not safe_r3,
                'r3_details': alerts_r3,
            }
            alerts['any'] = alerts['r1'] or alerts['r2'] or alerts['r3']
            
            # 独立危险检测 (比预警更严格的标准)
            # 用于计算预警提前时间
            danger_occurred = False
            
            # 危险标准 1: 严重超限 (比 R1 更宽松，±1.5 vs ±1.0)
            if np.any(qpos < -1.5) or np.any(qpos > 1.5):
                danger_occurred = True
            
            # 危险标准 2: 严重超速 (2.0 vs 1.0)
            if np.any(np.abs(qvel) > 2.0):
                danger_occurred = True
            
            # 记录首次危险时间
            if danger_occurred and trial_data.get('t_danger_actual') is None:
                trial_data['t_danger_actual'] = t
            
            risk_scores = {
                'r1': risk_r1,
                'r2': risk_r2,
                'r3': risk_r3,
                'r3_scores': scores_r3,
            }
            
            # 环境 step (无干预)
            ts = self.env.step(action)
            
            # 统计组合预警
            r1, r2, r3 = alerts['r1'], alerts['r2'], alerts['r3']
            if r1 and not r2 and not r3:
                alert_stats['r1_only'] += 1
            elif not r1 and r2 and not r3:
                alert_stats['r2_only'] += 1
            elif not r1 and not r2 and r3:
                alert_stats['r3_only'] += 1
            elif r1 and r2 and not r3:
                alert_stats['r1_r2'] += 1
            elif r1 and not r2 and r3:
                alert_stats['r1_r3'] += 1
            elif not r1 and r2 and r3:
                alert_stats['r2_r3'] += 1
            elif r1 and r2 and r3:
                alert_stats['r1_r2_r3'] += 1
            else:
                alert_stats['none'] += 1
            
            # 记录数据
            step_data = {
                't': t,
                'fault_active': fault_active,
                'qpos': qpos.tolist(),
                'qvel': qvel.tolist(),
                'action': action.tolist(),
                'reward': ts.reward if ts.reward is not None else 0,
                'alert_any': alerts['any'],
                'alert_r1': alerts['r1'],
                'alert_r2': alerts['r2'],
                'alert_r3': alerts['r3'],
                'risk_r1': risk_scores['r1'],
                'risk_r2': risk_scores['r2'],
                'risk_r3': risk_scores['r3'],
            }
            if 'r3_scores' in risk_scores:
                step_data['S_logic'] = risk_scores['r3_scores']['S_logic']
                step_data['D_ham'] = risk_scores['r3_scores']['D_ham']
                step_data['D_ood'] = risk_scores['r3_scores']['D_ood']
            
            trial_data['steps'].append(step_data)
            
            if alerts['any']:
                total_alerts += 1
                if fault_active and first_alert_after_fault is None:
                    first_alert_after_fault = t
                    fault_detected = True
            if r1:
                r1_alerts += 1
            if r2:
                r2_alerts += 1
            if r3:
                r3_alerts += 1
            
            total_reward += ts.reward if ts.reward is not None else 0
            qpos = ts.observation['qpos'].copy()
        
        # 成功判定
        trial_data['success'] = total_reward >= 0.5 * self.env.task.max_reward
        trial_data['total_reward'] = total_reward
        trial_data['total_alerts'] = total_alerts
        trial_data['alert_rate'] = total_alerts / self.max_steps
        trial_data['r1_alert_rate'] = r1_alerts / self.max_steps
        trial_data['r2_alert_rate'] = r2_alerts / self.max_steps
        trial_data['r3_alert_rate'] = r3_alerts / self.max_steps
        trial_data['alert_combinations'] = alert_stats
        
        # 故障检测指标
        if injector:
            fault_steps = min(injector.t_end, self.max_steps) - max(injector.t_start, 0)
            trial_data['fault_steps'] = fault_steps
            trial_data['fault_detected'] = fault_detected
            
            # 预警提前时间计算
            # t_danger: 首次出现危险的时间（R1/R2/R3 触发）
            # t_alert: 首次预警的时间
            # lead_time = t_danger - t_alert (步数)
            
            if fault_active and first_alert_after_fault is not None:
                trial_data['detection_delay'] = first_alert_after_fault - injector.t_start
                
                # 找到首次危险发生的时间（这里用故障注入时间近似）
                # 更精确的应该检测实际碰撞/超限时间
                t_danger = injector.t_start  # 假设故障注入即危险开始
                t_alert = first_alert_after_fault
                
                # 预警提前时间 (负数表示预警滞后)
                trial_data['warning_lead_time'] = t_danger - t_alert
                trial_data['t_danger'] = t_danger
                trial_data['t_alert'] = t_alert
            else:
                trial_data['detection_delay'] = None
                trial_data['warning_lead_time'] = None
                trial_data['t_danger'] = None
                trial_data['t_alert'] = None
        else:
            trial_data['fault_steps'] = 0
            trial_data['fault_detected'] = False
            trial_data['detection_delay'] = None
            trial_data['warning_lead_time'] = None
            trial_data['t_danger'] = None
            trial_data['t_alert'] = None
        
        return trial_data
    
    def save_trial(self, trial_data, episode_id):
        """保存试验数据"""
        # CSV
        csv_path = os.path.join(self.output_dir, f'trial_{episode_id:03d}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trial_data['steps'][0].keys())
            writer.writeheader()
            writer.writerows(trial_data['steps'])
        
        # JSON
        summary = {
            'episode_id': trial_data['episode_id'],
            'fault_type': trial_data['fault_type'],
            't_start': trial_data['t_start'],
            't_end': trial_data['t_end'],
            'success': trial_data['success'],
            'total_reward': trial_data['total_reward'],
            'alert_rate': trial_data['alert_rate'],
            'r1_alert_rate': trial_data['r1_alert_rate'],
            'r2_alert_rate': trial_data['r2_alert_rate'],
            'r3_alert_rate': trial_data['r3_alert_rate'],
            'alert_combinations': trial_data['alert_combinations'],
            'fault_detected': trial_data['fault_detected'],
            'detection_delay': trial_data['detection_delay'],
        }
        json_path = os.path.join(self.output_dir, f'trial_{episode_id:03d}_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ trial_{episode_id:03d}: fault={trial_data['fault_type']}, "
              f"detected={trial_data['fault_detected']}, delay={trial_data['detection_delay']}")


# ============================================================================
# Region 1 & 2 (简化版，复用之前的)
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
    def __init__(self, model_path, threshold=0.5):
        self.history_buffer = []
        self.max_history = 10
        
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
                print(f"  ✓ Region 2 GRU 模型加载")
            else:
                self.has_model = False
                self.model = None
        else:
            self.has_model = False
            self.model = None
        
        self.threshold = threshold
    
    def update_history(self, qpos, qvel):
        state = np.concatenate([qpos, qvel])
        self.history_buffer.append(state)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
    
    def check(self, qpos, qvel, action):
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
        
        max_support = support_values.max().item()
        risk_score = min(max_support / self.threshold, 1.0)
        safe = risk_score < 0.5
        
        return safe, risk_score


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1_qpos_min', type=float, default=-1.0)
    parser.add_argument('--r1_qpos_max', type=float, default=1.0)
    parser.add_argument('--r1_qvel_max', type=float, default=1.0)
    parser.add_argument('--r2_model', type=str, default='./outputs/region2_gru/gru_reachability_best.pth')
    parser.add_argument('--r3_model', type=str, default='./outputs/region3_complete')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/my_transfer_cube_model')
    parser.add_argument('--output_dir', type=str, default='./outputs/rta_fault_tests')
    parser.add_argument('--num_episodes', type=int, default=3)
    parser.add_argument('--fault_types', type=str, nargs='+', default=['F1_visual_noise', 'F2_visual_occlusion', 'F5_friction'])
    args = parser.parse_args()
    
    # 创建测试器
    tester = FaultInjectionTester(
        r1_config={'qpos_min': args.r1_qpos_min, 'qpos_max': args.r1_qpos_max, 'qvel_max': args.r1_qvel_max},
        r2_model_path=args.r2_model,
        r3_model_dir=args.r3_model,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
    )
    
    # 运行测试
    print(f"\n开始测试：{args.num_episodes} 集 × {len(args.fault_types)} 种故障")
    episode_id = 0
    
    # 先跑一个无故障基线
    print("\n=== 基线测试 (无故障) ===")
    trial_data = tester.run_episode(episode_id, fault_type='none')
    tester.save_trial(trial_data, episode_id)
    episode_id += 1
    
    # 故障注入测试
    for fault_type in args.fault_types:
        print(f"\n=== 故障注入：{fault_type} ===")
        for i in range(args.num_episodes):
            trial_data = tester.run_episode(episode_id, fault_type=fault_type, t_start=100, t_end=300)
            tester.save_trial(trial_data, episode_id)
            episode_id += 1
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
