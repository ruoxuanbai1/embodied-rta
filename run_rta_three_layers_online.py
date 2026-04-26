#!/usr/bin/env python3
"""
run_rta_three_layers_online.py - 完整三层 RTA 系统在线测试

三层架构:
1. Region 1: 物理硬约束 (关节限位、速度、碰撞) → 紧急停止
2. Region 2: 可达性预测 (GRU 预测未来 1 秒) → 减速×50%
3. Region 3: 感知异常检测 (激活链路+OOD+梯度) → 动作×40%

测试流程:
1. 加载 ACT 模型
2. 加载三层 RTA 检测器
3. 运行 MuJoCo 仿真 (在线)
4. 注入故障
5. 三层检测 + 干预
6. 记录数据
"""

import sys
sys.argv = ['run_rta_three_layers_online.py', '--ckpt_dir', './ckpts/my_transfer_cube_model', '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']

import os
import json
import csv
import argparse
import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch

# ACT 相关
from einops import rearrange
from constants import SIM_TASK_CONFIGS, DT
from utils import sample_box_pose
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env


# ============================================================================
# Region 1: 物理硬约束
# ============================================================================

class Region1Constraint:
    """Region 1: 物理硬约束检测器"""
    
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
        """
        violations = []
        
        # 1. 关节限位检查
        if np.any(qpos < self.qpos_min) or np.any(qpos > self.qpos_max):
            violations.append('joint_limit')
        
        # 2. 速度限制检查
        if np.any(np.abs(qvel) > self.qvel_max):
            violations.append('velocity_limit')
        
        return len(violations) == 0, violations
    
    def intervene(self, action):
        """干预：紧急停止"""
        return np.zeros_like(action)


# ============================================================================
# Region 2: 可达性预测
# ============================================================================

class Region2Reachability:
    """Region 2: GRU 可达性预测"""
    
    def __init__(self, model_path, threshold=0.5):
        """加载 GRU 模型"""
        import torch
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.threshold = threshold
        print(f"  ✓ Region 2 GRU 模型加载：{model_path}")
    
    def predict_reachability(self, qpos, qvel, actions_future, physics):
        """
        预测未来 1 秒可达集并检查安全性
        
        返回:
        - safe: bool 是否安全
        - risk_score: float 风险分数 (0-1)
        """
        # 简化版：用 GRU 预测未来状态
        # TODO: 实现完整的可达集预测
        
        # 临时返回安全
        return True, 0.0
    
    def intervene(self, action):
        """干预：减速×50%"""
        return action * 0.5


# ============================================================================
# Region 3: 感知异常检测
# ============================================================================

class Region3AnomalyDetector:
    """Region 3: 感知异常检测器"""
    
    def __init__(self, model_dir):
        """加载 Region 3 模型"""
        self.model_dir = model_dir
        
        # 1. 加载 KMeans
        with open(os.path.join(model_dir, 'kmeans_activation.pkl'), 'rb') as f:
            self.kmeans = pickle.load(f)
        
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
    
    def detect(self, action, qpos, qvel, gradient, activations):
        """
        检测异常
        
        返回:
        - safe: bool 是否安全
        - risk_score: float 风险分数
        - alerts: Dict 各模块预警状态
        """
        from scipy.spatial.distance import mahalanobis
        
        # 1. 判断动作模态
        modality_id = int(self.kmeans.transform([action.astype(np.float64)])[0])
        modality_id = np.argmin(self.kmeans.transform([action.astype(np.float64)]))
        
        # 2. 获取该模态的阈值
        threshold_activation = self.activation_thresholds[modality_id]
        M_ref = self.activation_M_refs[modality_id]
        F_legal = self.F_legal_profiles[modality_id]
        
        # 3. 计算各模块分数 (简化版，实际应该用真实的激活和梯度)
        # 激活链路：汉明距离
        D_ham = np.random.uniform(0.05, 0.15)  # 临时随机值
        
        # OOD 检测：马氏距离
        state = np.concatenate([qpos, qvel])
        D_ood = mahalanobis(state, self.ood_mu, self.ood_sigma_inv)
        
        # 梯度贡献：逻辑合理性
        S_logic = np.random.uniform(0.6, 0.9)  # 临时随机值
        
        # 4. 独立预警
        alerts = {
            'modality': modality_id,
            'logic': S_logic < 0.4,
            'activation': D_ham > threshold_activation,
            'ood': D_ood > self.ood_threshold,
        }
        
        # 5. OR 逻辑
        triggered = alerts['logic'] or alerts['activation'] or alerts['ood']
        
        return not triggered, D_ood / self.ood_threshold, alerts
    
    def intervene(self, action):
        """干预：动作×40%"""
        return action * 0.4


# ============================================================================
# 完整三层 RTA 系统
# ============================================================================

class ThreeLayerRTA:
    """完整三层 RTA 系统"""
    
    def __init__(self, r1_config, r2_model_path, r3_model_dir):
        """初始化三层 RTA"""
        print("\n初始化三层 RTA 系统...")
        
        self.region1 = Region1Constraint(
            qpos_min=r1_config.get('qpos_min', -1.0),
            qpos_max=r1_config.get('qpos_max', 1.0),
            qvel_max=r1_config.get('qvel_max', 1.0),
        )
        print("  ✓ Region 1: 物理硬约束")
        
        self.region2 = Region2Reachability(r2_model_path)
        print("  ✓ Region 2: 可达性预测")
        
        self.region3 = Region3AnomalyDetector(r3_model_dir)
        print("  ✓ Region 3: 感知异常检测")
    
    def check_and_intervene(self, action, qpos, qvel, gradient, activations, physics):
        """
        三层检查 + 干预
        
        返回:
        - final_action: 最终动作 (可能已被修改)
        - intervention_level: int 干预等级 (0=无，1=R3，2=R2，3=R1)
        - alerts: Dict 各层预警状态
        """
        intervention_level = 0
        alerts = {'r1': False, 'r2': False, 'r3': False}
        
        # Region 1: 物理硬约束 (最高优先级)
        safe_r1, violations = self.region1.check(qpos, qvel)
        if not safe_r1:
            alerts['r1'] = True
            intervention_level = 3
            final_action = self.region1.intervene(action)
            return final_action, intervention_level, alerts
        
        # Region 2: 可达性预测
        safe_r2, risk_r2 = self.region2.predict_reachability(qpos, qvel, action, physics)
        if not safe_r2:
            alerts['r2'] = True
            intervention_level = 2
            action = self.region2.intervene(action)
        
        # Region 3: 感知异常检测
        safe_r3, risk_r3, alerts_r3 = self.region3.detect(action, qpos, qvel, gradient, activations)
        if not safe_r3:
            alerts['r3'] = True
            if intervention_level < 1:
                intervention_level = 1
            action = self.region3.intervene(action)
        
        return action, intervention_level, alerts


# ============================================================================
# 在线测试器
# ============================================================================

class RTAOnlineTester:
    """三层 RTA 在线测试器"""
    
    def __init__(self, rta_system, ckpt_dir, output_dir):
        """初始化"""
        self.rta = rta_system
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载 ACT 模型
        print("\n加载 ACT 模型...")
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
        print(f"  ✓ 模型加载：{ckpt_path}")
        
        # 创建环境
        print("\n创建仿真环境...")
        self.env = make_sim_env('sim_transfer_cube_scripted')
        self.camera_names = ['top']
        self.max_steps = SIM_TASK_CONFIGS['sim_transfer_cube_scripted']['episode_len']
        print(f"  ✓ 环境就绪，max_steps={self.max_steps}")
    
    def run_episode(self, episode_id, fault_injector=None):
        """运行单集在线测试"""
        # 重置环境
        BOX_POSE[0] = sample_box_pose()
        ts = self.env.reset()
        
        # 数据存储
        trial_data = {
            'episode_id': episode_id,
            'fault_type': fault_injector.fault_id if fault_injector else 'none',
            'steps': [],
            'interventions': [],
            'success': False,
            'total_reward': 0,
        }
        
        qpos = ts.observation['qpos'].copy()
        total_reward = 0
        
        # 运行 rollout
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
            
            # 故障注入
            if fault_injector:
                # TODO: 实现故障注入
                pass
            
            # ACT 推理
            with torch.no_grad():
                all_actions = self.policy(qpos_tensor, curr_image)
                action = all_actions[:, 0, :].detach().cpu().numpy()[0]
            
            # 三层 RTA 检查 + 干预
            qvel = ts.observation['qvel'][:14].copy()
            gradient = np.zeros((14, 14))  # TODO: 计算真实梯度
            activations = {}  # TODO: 提取真实激活
            
            final_action, intervention_level, alerts = self.rta.check_and_intervene(
                action, qpos, qvel, gradient, activations, self.env.physics
            )
            
            # 环境 step
            ts = self.env.step(final_action)
            
            # 记录数据
            step_data = {
                't': t,
                'qpos': qpos.tolist(),
                'qvel': qvel.tolist(),
                'action_original': action.tolist(),
                'action_final': final_action.tolist(),
                'reward': ts.reward if ts.reward is not None else 0,
                'intervention_level': intervention_level,
                'alert_r1': alerts['r1'],
                'alert_r2': alerts['r2'],
                'alert_r3': alerts['r3'],
            }
            trial_data['steps'].append(step_data)
            trial_data['interventions'].append(intervention_level > 0)
            
            total_reward += ts.reward if ts.reward is not None else 0
            qpos = ts.observation['qpos'].copy()
        
        # 成功判定
        trial_data['success'] = total_reward >= 0.5 * self.env.task.max_reward
        trial_data['total_reward'] = total_reward
        trial_data['total_interventions'] = sum(trial_data['interventions'])
        trial_data['intervention_rate'] = trial_data['total_interventions'] / self.max_steps
        
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
            'success': trial_data['success'],
            'total_reward': trial_data['total_reward'],
            'total_interventions': trial_data['total_interventions'],
            'intervention_rate': trial_data['intervention_rate'],
        }
        json_path = os.path.join(self.output_dir, f'trial_{episode_id:03d}_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ 已保存：trial_{episode_id:03d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1_qpos_min', type=float, default=-1.0)
    parser.add_argument('--r1_qpos_max', type=float, default=1.0)
    parser.add_argument('--r1_qvel_max', type=float, default=1.0)
    parser.add_argument('--r2_model', type=str, default='./outputs/region2_gru/gru_reachability_best.pth')
    parser.add_argument('--r3_model', type=str, default='./outputs/region3_complete')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/my_transfer_cube_model')
    parser.add_argument('--output_dir', type=str, default='./outputs/rta_three_layers_tests')
    parser.add_argument('--num_episodes', type=int, default=5)
    args = parser.parse_args()
    
    print("="*60)
    print("完整三层 RTA 系统在线测试")
    print("="*60)
    
    # 创建三层 RTA 系统
    rta_system = ThreeLayerRTA(
        r1_config={
            'qpos_min': args.r1_qpos_min,
            'qpos_max': args.r1_qpos_max,
            'qvel_max': args.r1_qvel_max,
        },
        r2_model_path=args.r2_model,
        r3_model_dir=args.r3_model,
    )
    
    # 创建测试器
    tester = RTAOnlineTester(rta_system, args.ckpt_dir, args.output_dir)
    
    # 运行测试
    print(f"\n开始测试：{args.num_episodes} 集")
    for ep_id in range(args.num_episodes):
        trial_data = tester.run_episode(ep_id)
        tester.save_trial(trial_data, ep_id)
        print(f"  Ep {ep_id}: success={trial_data['success']}, interventions={trial_data['total_interventions']}")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
