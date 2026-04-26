#!/usr/bin/env python3
"""
run_rta_three_layers_monitoring.py - 三层 RTA 纯预警测试 (无干预)

目的：评估三层检测器的预警能力，不施加任何干预

输出:
- 每步的预警状态 (R1/R2/R3)
- 预警率、虚警率
- 故障检测率 (注入故障时)
"""



import os
import json
import csv
import argparse
import pickle
import joblib
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

# ACT 相关
from einops import rearrange
from constants import SIM_TASK_CONFIGS, DT
from utils import sample_box_pose
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env


# ============================================================================
# Region 1: 物理硬约束 (只检测，不干预)
# ============================================================================

class Region1Monitor:
    """Region 1: 物理硬约束监测器"""
    
    def __init__(self, qpos_min=-1.0, qpos_max=1.0, qvel_max=1.0):
        self.qpos_min = qpos_min
        self.qpos_max = qpos_max
        self.qvel_max = qvel_max
    
    def check(self, qpos, qvel):
        """
        检查物理约束
        
        返回:
        - safe: bool 是否安全
        - violations: List[str] 违反的约束
        - risk_score: float 风险分数 (0-1)
        """
        violations = []
        risk_score = 0.0
        
        # 1. 关节限位检查
        qpos_margin_min = np.min(qpos - self.qpos_min)
        qpos_margin_max = np.min(self.qpos_max - qpos)
        qpos_margin = min(qpos_margin_min, qpos_margin_max)
        
        if qpos_margin < 0:
            violations.append('joint_limit')
            risk_score = 1.0
        elif qpos_margin < 0.2:
            risk_score = 1.0 - (qpos_margin / 0.2)  # 0-1
        
        # 2. 速度限制检查
        qvel_ratio = np.max(np.abs(qvel)) / self.qvel_max
        if qvel_ratio > 1.0:
            violations.append('velocity_limit')
            risk_score = max(risk_score, 1.0)
        elif qvel_ratio > 0.8:
            risk_score = max(risk_score, (qvel_ratio - 0.8) / 0.2)
        
        return len(violations) == 0, violations, risk_score


# ============================================================================
# Region 2: 可达性预测 (只检测，不干预)
# ============================================================================

class DeepReachabilityGRU(nn.Module):
    """深度 GRU 可达集预测模型 (4 层)"""
    
    def __init__(self, input_dim=28, hidden_dim=192, num_layers=4, 
                 output_dim=16, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
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
    """Region 2: GRU 可达性预测监测器"""
    
    def __init__(self, model_path, threshold=0.5):
        """加载 GRU 模型"""
        self.history_buffer = []  # 状态历史 (最多 10 步)
        self.max_history = 10
        
        if os.path.exists(model_path):
            self.checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(self.checkpoint, dict) and 'model_state_dict' in self.checkpoint:
                self.has_model = True
                self.config = self.checkpoint.get('config', {})
                
                # 加载模型
                self.model = DeepReachabilityGRU(
                    input_dim=28,
                    hidden_dim=self.config.get('hidden_dim', 192),
                    num_layers=self.config.get('num_layers', 4),
                    output_dim=16,
                    dropout=self.config.get('dropout', 0.4)
                )
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                self.model.eval()
                
                print(f"  ✓ Region 2 GRU 模型加载：{model_path}")
                print(f"    配置：hidden={self.config.get('hidden_dim', 192)}, layers={self.config.get('num_layers', 4)}")
            else:
                self.has_model = False
                self.model = None
                print(f"  ⚠ Region 2 模型格式未知")
        else:
            self.has_model = False
            self.model = None
            print(f"  ⚠ Region 2 模型不存在：{model_path}")
        
        self.threshold = threshold
    
    def update_history(self, qpos, qvel):
        """更新状态历史"""
        state = np.concatenate([qpos, qvel])  # (28,)
        self.history_buffer.append(state)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
    
    def check(self, qpos, qvel, action):
        """
        检查可达性风险
        
        返回:
        - safe: bool 是否安全
        - risk_score: float 风险分数 (0-1)
        """
        if not self.has_model or self.model is None:
            self.update_history(qpos, qvel)
            return True, 0.0
        
        # 更新历史
        self.update_history(qpos, qvel)
        
        # 需要至少 10 步历史才能预测
        if len(self.history_buffer) < self.max_history:
            return True, 0.0
        
        # 准备输入：(1, 11, 28) - 包括当前步
        history = np.array(self.history_buffer[-self.max_history:])  # (10, 28)
        current = np.concatenate([qpos, qvel])[None, :]  # (1, 28)
        input_tensor = np.concatenate([history, current], axis=0)[None, :, :]  # (1, 11, 28)
        input_tensor = torch.from_numpy(input_tensor).float()
        
        # GRU 预测
        with torch.no_grad():
            support_values = self.model(input_tensor)  # (1, 16)
        
        # 支撑函数值 > 0 表示可能越界
        # 风险分数 = max(support_values) / threshold
        max_support = support_values.max().item()
        risk_score = min(max_support / self.threshold, 1.0)
        
        safe = risk_score < 0.5
        
        return safe, risk_score


# ============================================================================
# Region 3: 感知异常检测 (只检测，不干预)
# ============================================================================

class Region3Monitor:
    """Region 3: 感知异常检测器"""
    
    def __init__(self, model_dir):
        """加载 Region 3 模型"""
        self.model_dir = model_dir
        
        # 1. 加载 KMeans
        self.kmeans = joblib.load(os.path.join(model_dir, 'kmeans_activation.pkl'))
        
        # 2. 加载激活链路阈值
        with open(os.path.join(model_dir, 'activation_links.json'), 'r') as f:
            self.activation_data = json.load(f)
            self.activation_thresholds = [p['threshold'] for p in self.activation_data]
            self.activation_M_refs = [np.array(p['M_ref']) for p in self.activation_data]
        
        # 3. 加载 F_legal
        with open(os.path.join(model_dir, 'F_legal_profiles.json'), 'r') as f:
            self.F_legal_profiles = json.load(f)
        
        # 4. 加载 OOD 统计量
        with open(os.path.join(model_dir, 'ood_stats.json'), 'r') as f:
            ood_data = json.load(f)
            self.ood_mu = np.array(ood_data['mu'])
            self.ood_sigma_inv = np.array(ood_data['sigma_inv'])
            self.ood_threshold = ood_data['threshold']
        
        print(f"  ✓ Region 3 检测器加载：{model_dir}")
    
    def check(self, action, qpos, qvel, gradient=None, activations=None):
        """
        检测异常
        
        返回:
        - safe: bool 是否安全
        - risk_score: float 风险分数 (0-1)
        - alerts: Dict 各模块预警状态
        - scores: Dict 各模块分数
        """
        from scipy.spatial.distance import mahalanobis
        
        # 1. 判断动作模态
        action_float = action.astype(np.float64)
        distances = self.kmeans.transform([action_float])[0]
        modality_id = int(np.argmin(distances))
        
        # 2. 获取该模态的阈值
        threshold_activation = self.activation_thresholds[modality_id]
        M_ref = self.activation_M_refs[modality_id]
        F_legal = self.F_legal_profiles[modality_id]
        
        # 3. 计算各模块分数
        # 激活链路：汉明距离 (临时用随机值，后续替换为真实计算)
        D_ham = np.random.uniform(0.05, 0.15)
        
        # OOD 检测：马氏距离
        state = np.concatenate([qpos, qvel])
        D_ood = mahalanobis(state, self.ood_mu, self.ood_sigma_inv)
        
        # 梯度贡献：逻辑合理性 (临时用随机值)
        S_logic = np.random.uniform(0.6, 0.9)
        
        # 4. 独立预警
        alerts = {
            'modality': modality_id,
            'logic': S_logic < 0.4,
            'activation': D_ham > threshold_activation,
            'ood': D_ood > self.ood_threshold,
        }
        
        # 5. 风险分数 (归一化到 0-1)
        risk_logic = 1.0 - S_logic  # 越低越危险
        risk_activation = D_ham / threshold_activation
        risk_ood = D_ood / self.ood_threshold
        
        # OR 逻辑：任一触发即不安全
        triggered = alerts['logic'] or alerts['activation'] or alerts['ood']
        
        scores = {
            'S_logic': float(S_logic),
            'D_ham': float(D_ham),
            'D_ood': float(D_ood),
            'risk_logic': float(risk_logic),
            'risk_activation': float(risk_activation),
            'risk_ood': float(risk_ood),
        }
        
        return not triggered, max(risk_logic, risk_activation, risk_ood), alerts, scores


# ============================================================================
# 三层监测器 (无干预)
# ============================================================================

class ThreeLayerMonitor:
    """完整三层监测器 (只检测，不干预)"""
    
    def __init__(self, r1_config, r2_model_path, r3_model_dir):
        """初始化三层监测器"""
        print("\n初始化三层监测器 (无干预)...")
        
        self.region1 = Region1Monitor(
            qpos_min=r1_config.get('qpos_min', -1.0),
            qpos_max=r1_config.get('qpos_max', 1.0),
            qvel_max=r1_config.get('qvel_max', 1.0),
        )
        print("  ✓ Region 1: 物理硬约束")
        
        self.region2 = Region2Monitor(r2_model_path)
        print("  ✓ Region 2: 可达性预测")
        
        self.region3 = Region3Monitor(r3_model_dir)
        print("  ✓ Region 3: 感知异常检测")
    
    def check(self, action, qpos, qvel, gradient=None, activations=None, physics=None):
        """
        三层检查 (无干预)
        
        返回:
        - alerts: Dict 各层预警状态
        - risk_scores: Dict 各层风险分数
        """
        alerts = {}
        risk_scores = {}
        
        # Region 1
        safe_r1, violations, risk_r1 = self.region1.check(qpos, qvel)
        alerts['r1'] = not safe_r1
        alerts['r1_violations'] = violations
        risk_scores['r1'] = risk_r1
        
        # Region 2
        safe_r2, risk_r2 = self.region2.check(qpos, qvel, action)
        alerts['r2'] = not safe_r2
        risk_scores['r2'] = risk_r2
        
        # Region 3
        safe_r3, risk_r3, alerts_r3, scores_r3 = self.region3.check(
            action, qpos, qvel, gradient, activations
        )
        alerts['r3'] = not safe_r3
        alerts['r3_details'] = alerts_r3
        risk_scores['r3'] = risk_r3
        risk_scores['r3_scores'] = scores_r3
        
        # 总体预警 (OR 逻辑)
        alerts['any'] = alerts['r1'] or alerts['r2'] or alerts['r3']
        
        return alerts, risk_scores


# ============================================================================
# 在线监测测试器
# ============================================================================

class MonitoringTester:
    """三层监测器在线测试器"""
    
    def __init__(self, monitor, ckpt_dir, output_dir):
        """初始化"""
        self.monitor = monitor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置 ACTPolicy 所需的命令行参数
        import sys
        old_argv = sys.argv
        sys.argv = ['ACTPolicy', '--ckpt_dir', ckpt_dir, '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']
        
        # 加载 ACT 模型
        print("\n加载 ACT 模型...")
        self.policy_config = {
            'lr': 1e-5, 'num_queries': 100, 'kl_weight': 10,
            'hidden_dim': 512, 'dim_feedforward': 3200, 'lr_backbone': 1e-5,
            'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7,
            'nheads': 8, 'camera_names': ['top'],
        }
        self.policy = ACTPolicy(self.policy_config)
        sys.argv = old_argv
        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        self.policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.policy.cuda()
        self.policy.eval()
        print(f"  ✓ 模型加载：{ckpt_path}")
        
        # 创建环境
        print("\n创建仿真环境...")
        self.env = make_sim_env('sim_transfer_cube_scripted')
        self.camera_names = ['top']
        self.max_steps = SIM_TASK_CONFIGS['sim_transfer_cube_scripted']['episode_len']
        print(f"  ✓ 环境就绪，max_steps={self.max_steps}")
    
    def run_episode(self, episode_id, fault_injector=None):
        """运行单集在线监测测试"""
        # 重置环境
        BOX_POSE[0] = sample_box_pose()
        ts = self.env.reset()
        
        # 数据存储
        trial_data = {
            'episode_id': episode_id,
            'fault_type': fault_injector.fault_id if fault_injector else 'none',
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
        
        # 运行 rollout (无干预!)
        for t in range(self.max_steps):
            # 准备输入
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            curr_images = [
                rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
                for cam_name in self.camera_names
            ]
            curr_image = torch.from_numpy(
                np.stack(curr_images, axis=0) / 255.0
            ).float().cuda().unsqueeze(0)
            
            # ACT 推理 (无干预，直接执行)
            with torch.no_grad():
                all_actions = self.policy(qpos_tensor, curr_image)
                action = all_actions[:, 0, :].detach().cpu().numpy()[0]
            
            # 三层监测 (只记录，不干预!)
            qvel = ts.observation['qvel'][:14].copy()
            gradient = np.zeros((14, 14))  # TODO: 计算真实梯度
            activations = {}  # TODO: 提取真实激活
            
            alerts, risk_scores = self.monitor.check(
                action, qpos, qvel, gradient, activations, self.env.physics
            )
            
            # 环境 step (执行原始动作，无干预!)
            ts = self.env.step(action)
            
            # 统计组合预警
            r1 = alerts['r1']
            r2 = alerts['r2']
            r3 = alerts['r3']
            
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
            
            # Region 3 详细分数
            if 'r3_scores' in risk_scores:
                step_data['S_logic'] = risk_scores['r3_scores']['S_logic']
                step_data['D_ham'] = risk_scores['r3_scores']['D_ham']
                step_data['D_ood'] = risk_scores['r3_scores']['D_ood']
            
            trial_data['steps'].append(step_data)
            
            if alerts['any']:
                total_alerts += 1
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
        
        # 各层预警率
        trial_data['r1_alert_rate'] = r1_alerts / self.max_steps
        trial_data['r2_alert_rate'] = r2_alerts / self.max_steps
        trial_data['r3_alert_rate'] = r3_alerts / self.max_steps
        
        # 组合预警统计
        trial_data['alert_combinations'] = alert_stats
        
        return trial_data
    
    def save_trial(self, trial_data, episode_id):
        """保存试验数据"""
        # CSV
        csv_path = os.path.join(self.output_dir, f'trial_{episode_id:03d}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trial_data['steps'][0].keys())
            writer.writeheader()
            writer.writerows(trial_data['steps'])
        
        # JSON 摘要 (包含完整指标)
        summary = {
            'episode_id': trial_data['episode_id'],
            'fault_type': trial_data['fault_type'],
            'success': trial_data['success'],
            'total_reward': trial_data['total_reward'],
            
            # 预警指标
            'total_alerts': trial_data['total_alerts'],
            'alert_rate': trial_data['alert_rate'],
            'r1_alert_rate': trial_data['r1_alert_rate'],
            'r2_alert_rate': trial_data['r2_alert_rate'],
            'r3_alert_rate': trial_data['r3_alert_rate'],
            
            # 组合预警统计
            'alert_combinations': trial_data['alert_combinations'],
        }
        json_path = os.path.join(self.output_dir, f'trial_{episode_id:03d}_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 打印详细指标
        print(f"  ✓ trial_{episode_id:03d}:")
        print(f"      总预警率：{trial_data['alert_rate']*100:.1f}%")
        print(f"      R1 预警率：{trial_data['r1_alert_rate']*100:.1f}%")
        print(f"      R2 预警率：{trial_data['r2_alert_rate']*100:.1f}%")
        print(f"      R3 预警率：{trial_data['r3_alert_rate']*100:.1f}%")
        print(f"      组合统计:")
        for combo, count in trial_data['alert_combinations'].items():
            if count > 0:
                print(f"        {combo}: {count} 步 ({count/self.max_steps*100:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1_qpos_min', type=float, default=-1.0)
    parser.add_argument('--r1_qpos_max', type=float, default=1.0)
    parser.add_argument('--r1_qvel_max', type=float, default=1.0)
    parser.add_argument('--r2_model', type=str, default='./outputs/region2_gru/gru_reachability_best.pth')
    parser.add_argument('--r3_model', type=str, default='./outputs/region3_complete')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/my_transfer_cube_model')
    parser.add_argument('--output_dir', type=str, default='./outputs/rta_monitoring_tests')
    parser.add_argument('--num_episodes', type=int, default=5)
    args = parser.parse_args()
    
    print("="*60)
    print("三层 RTA 监测器预警能力测试 (无干预)")
    print("="*60)
    
    # 创建三层监测器
    monitor = ThreeLayerMonitor(
        r1_config={
            'qpos_min': args.r1_qpos_min,
            'qpos_max': args.r1_qpos_max,
            'qvel_max': args.r1_qvel_max,
        },
        r2_model_path=args.r2_model,
        r3_model_dir=args.r3_model,
    )
    
    # 创建测试器
    tester = MonitoringTester(monitor, args.ckpt_dir, args.output_dir)
    
    # 运行测试
    print(f"\n开始测试：{args.num_episodes} 集 (无干预)")
    for ep_id in range(args.num_episodes):
        trial_data = tester.run_episode(ep_id)
        tester.save_trial(trial_data, ep_id)
    
    print("\n" + "="*60)
    print("测试完成! 输出目录:", args.output_dir)
    print("="*60)
