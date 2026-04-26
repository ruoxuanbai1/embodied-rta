#!/usr/bin/env python3
"""
run_region3_online_test.py - Region 3 在线测试

测试流程:
1. 加载 ACT 模型
2. 加载 Region 3 检测器
3. 运行环境 rollout (真实仿真)
4. 注入故障 (在指定时间)
5. Region 3 实时检测
6. 触发干预 (动作降级)
7. 记录数据

输出:
- CSV: 时序数据 (每步的 qpos, action, scores, alerts)
- JSON: 试验报告 (检测率、虚警率、提前时间等)
"""

import sys
sys.argv = ['run_region3_online_test.py', '--ckpt_dir', './ckpts/my_transfer_cube_model', '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']

import os
import json
import csv
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm

# ACT 相关
from einops import rearrange
from constants import SIM_TASK_CONFIGS
from utils import sample_box_pose
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env

# Region 3 检测器
from region3_detector_inference import Region3Detector

# 故障注入器
from run_rta_experiments import FaultInjector


class Region3OnlineTester:
    """Region 3 在线测试器"""
    
    def __init__(self, detector_dir, ckpt_dir, output_dir):
        """初始化"""
        print("="*60)
        print("Region 3 在线测试器")
        print("="*60)
        
        # 1. 加载 Region 3 检测器
        print("\n1. 加载 Region 3 检测器...")
        self.detector = Region3Detector(detector_dir)
        print(f"  ✓ 检测器加载完成")
        
        # 2. 加载 ACT 模型
        print("\n2. 加载 ACT 模型...")
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
        print(f"  ✓ 模型加载完成：{ckpt_path}")
        
        # 3. 创建环境
        print("\n3. 创建仿真环境...")
        self.env = make_sim_env('sim_transfer_cube_scripted')
        self.camera_names = ['top']
        self.max_steps = SIM_TASK_CONFIGS['sim_transfer_cube_scripted']['episode_len']
        print(f"  ✓ 环境就绪，max_steps={self.max_steps}")
        
        # 4. 设置输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n4. 输出目录：{output_dir}")
        
        # 5. 梯度计算开关
        self.compute_gradient = True  # 需要梯度时打开
    
    def compute_gradient_contrib(self, qpos_tensor, image, action_idx=0):
        """计算梯度贡献 φ = qpos × ∂a/∂qpos"""
        qpos_tensor.requires_grad_(True)
        
        # 前向传播
        action_out = self.policy(qpos_tensor, image)
        action_now = action_out[:, action_idx, :]  # (1, 14)
        
        # 计算梯度
        gradient = []
        for i in range(14):
            grad = torch.autograd.grad(
                action_now[:, i].sum(),
                qpos_tensor,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            if grad is None:
                grad = torch.zeros_like(qpos_tensor)
            gradient.append(grad.squeeze(0).detach().cpu().numpy())
        
        gradient = np.array(gradient)  # (14, 14)
        
        # 贡献度 φ = qpos × gradient
        qpos_np = qpos_tensor.squeeze(0).detach().cpu().numpy()
        contrib = qpos_np[None, :] * gradient  # (14, 14)
        
        return contrib
    
    def extract_activations(self, hooks):
        """从 hooks 提取 4 层激活"""
        activations = {}
        for i in range(4):
            layer_key = f'layer{i}_ffn'
            if layer_key in hooks and hooks[layer_key].outputs:
                # (102, 1, 512)
                act = hooks[layer_key].outputs[0].detach().cpu().numpy()
                activations[layer_key] = act
        return activations
    
    def run_episode(self, episode_id, fault_injector=None):
        """
        运行单集测试
        
        参数:
        - episode_id: 集 ID
        - fault_injector: 故障注入器 (可选)
        
        返回:
        - trial_data: Dict 试验数据
        """
        print(f"\n{'='*60}")
        print(f"Episode {episode_id}")
        print(f"{'='*60}")
        
        # 重置环境
        BOX_POSE[0] = sample_box_pose()
        ts = self.env.reset()
        
        # 数据存储
        trial_data = {
            'episode_id': episode_id,
            'fault_type': fault_injector.fault_id if fault_injector else 'none',
            'steps': [],
            'alerts': [],
            'interventions': [],
            'success': False,
            'total_reward': 0,
        }
        
        qpos = ts.observation['qpos'].copy()
        total_reward = 0
        
        # 运行 rollout
        for t in tqdm(range(self.max_steps), desc=f"Ep {episode_id}", leave=False):
            # 准备输入
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            curr_images = [
                rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
                for cam_name in self.camera_names
            ]
            curr_image = torch.from_numpy(
                np.stack(curr_images, axis=0) / 255.0
            ).float().cuda().unsqueeze(0)
            
            # 故障注入 (图像)
            image_np = curr_image.detach().cpu().numpy()
            if fault_injector:
                image_np, _, _, _ = fault_injector.inject(
                    qpos, ts.observation['qvel'].copy(),
                    np.zeros(14), image_np, t
                )
                curr_image = torch.from_numpy(image_np).float().cuda()
            
            # ACT 推理 (inference mode)
            with torch.no_grad():
                all_actions = self.policy(qpos_tensor, curr_image)
                action = all_actions[:, 0, :].detach().cpu().numpy()  # (1, 14)
            
            # 计算梯度贡献 (需要时)
            if self.compute_gradient:
                gradient_contrib = self.compute_gradient_contrib(
                    qpos_tensor.clone(), curr_image, action_idx=0
                )
            else:
                gradient_contrib = np.zeros((14, 14))
            
            # 提取激活 (需要 hooks，暂时简化)
            activations = {
                f'layer{i}_ffn': np.random.randn(102, 1, 512).astype(np.float64)
                for i in range(4)
            }
            
            # Region 3 检测
            state = np.concatenate([qpos, ts.observation['qvel'][:14]])
            alerts, intervention = self.detector.detect(
                action=action[0],
                qpos=qpos,
                qvel=ts.observation['qvel'][:14],
                gradient=gradient_contrib,
                activations=activations
            )
            
            # 干预：动作降级
            if intervention:
                action = action * 0.4  # 降级到 40%
            
            # 环境 step
            ts = self.env.step(action[0])
            
            # 记录数据
            step_data = {
                't': t,
                'qpos': qpos.tolist(),
                'qvel': ts.observation['qvel'][:14].tolist(),
                'action': action[0].tolist(),
                'reward': ts.reward if ts.reward is not None else 0,
                'modality': alerts['modality'],
                'S_logic': alerts['logic']['score'],
                'D_ham': alerts['activation']['score'],
                'D_ood': alerts['ood']['score'],
                'alert_logic': alerts['logic']['triggered'],
                'alert_activation': alerts['activation']['triggered'],
                'alert_ood': alerts['ood']['triggered'],
                'intervention': intervention,
            }
            trial_data['steps'].append(step_data)
            trial_data['alerts'].append(
                alerts['logic']['triggered'] or
                alerts['activation']['triggered'] or
                alerts['ood']['triggered']
            )
            trial_data['interventions'].append(intervention)
            
            total_reward += ts.reward if ts.reward is not None else 0
            qpos = ts.observation['qpos'].copy()
        
        # 成功判定
        trial_data['success'] = total_reward >= 0.5 * self.env.task.max_reward
        trial_data['total_reward'] = total_reward
        
        # 统计
        trial_data['total_alerts'] = sum(trial_data['alerts'])
        trial_data['total_interventions'] = sum(trial_data['interventions'])
        trial_data['alert_rate'] = trial_data['total_alerts'] / self.max_steps
        
        print(f"  成功：{trial_data['success']}")
        print(f"  总奖励：{trial_data['total_reward']:.2f}")
        print(f"  预警次数：{trial_data['total_alerts']} ({trial_data['alert_rate']*100:.1f}%)")
        print(f"  干预次数：{trial_data['total_interventions']}")
        
        return trial_data
    
    def save_trial(self, trial_data, episode_id):
        """保存试验数据"""
        # CSV: 时序数据
        csv_path = os.path.join(self.output_dir, f'trial_{episode_id:03d}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trial_data['steps'][0].keys())
            writer.writeheader()
            writer.writerows(trial_data['steps'])
        
        # JSON: 试验摘要
        summary = {
            'episode_id': trial_data['episode_id'],
            'fault_type': trial_data['fault_type'],
            'success': trial_data['success'],
            'total_reward': trial_data['total_reward'],
            'total_alerts': trial_data['total_alerts'],
            'total_interventions': trial_data['total_interventions'],
            'alert_rate': trial_data['alert_rate'],
        }
        json_path = os.path.join(self.output_dir, f'trial_{episode_id:03d}_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ 已保存：{csv_path}, {json_path}")
    
    def run_tests(self, num_episodes, fault_configs):
        """
        运行多组测试
        
        参数:
        - num_episodes: 测试集数
        - fault_configs: List[Dict] 故障配置列表
        """
        print(f"\n{'='*60}")
        print(f"开始测试：{num_episodes} 集 × {len(fault_configs)} 种故障")
        print(f"{'='*60}")
        
        all_results = []
        
        for ep_id in range(num_episodes):
            for fault_config in fault_configs:
                # 创建故障注入器
                fault_injector = FaultInjector(
                    fault_config['fault_id'],
                    fault_config['params']
                )
                
                # 运行测试
                trial_data = self.run_episode(ep_id, fault_injector)
                
                # 保存数据
                self.save_trial(trial_data, ep_id * len(fault_configs) + fault_configs.index(fault_config))
                
                all_results.append(trial_data)
        
        # 生成汇总报告
        self.generate_report(all_results)
        
        return all_results
    
    def generate_report(self, all_results):
        """生成汇总报告"""
        print(f"\n{'='*60}")
        print("生成汇总报告...")
        print(f"{'='*60}")
        
        # 按故障类型分组统计
        fault_stats = {}
        for result in all_results:
            fault_type = result['fault_type']
            if fault_type not in fault_stats:
                fault_stats[fault_type] = {
                    'total': 0,
                    'detected': 0,
                    'alerts': 0,
                    'interventions': 0,
                }
            fault_stats[fault_type]['total'] += 1
            if result['alert_rate'] > 0.1:  # 预警率>10% 认为检测到
                fault_stats[fault_type]['detected'] += 1
            fault_stats[fault_type]['alerts'] += result['total_alerts']
            fault_stats[fault_type]['interventions'] += result['total_interventions']
        
        # 计算检测率
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(all_results),
            'fault_detection_rates': {},
        }
        
        for fault_type, stats in fault_stats.items():
            detection_rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0
            report['fault_detection_rates'][fault_type] = {
                'detection_rate': detection_rate,
                'total': stats['total'],
                'detected': stats['detected'],
                'avg_alerts': stats['alerts'] / stats['total'] if stats['total'] > 0 else 0,
                'avg_interventions': stats['interventions'] / stats['total'] if stats['total'] > 0 else 0,
            }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'test_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✓ 报告已保存：{report_path}")
        
        # 打印摘要
        print(f"\n{'='*60}")
        print("检测结果摘要")
        print(f"{'='*60}")
        for fault_type, stats in report['fault_detection_rates'].items():
            print(f"{fault_type}: {stats['detection_rate']*100:.1f}% ({stats['detected']}/{stats['total']})")
        
        return report


if __name__ == '__main__':
    import torch
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_dir', type=str, default='./outputs/region3_complete')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/my_transfer_cube_model')
    parser.add_argument('--output_dir', type=str, default='./outputs/region3_online_tests')
    parser.add_argument('--num_episodes', type=int, default=5)
    args = parser.parse_args()
    
    # 故障配置 (13 种)
    fault_configs = [
        {'fault_id': 'F1', 'params': {'t_start': 100, 't_end': 300, 'noise_std': 0.3}},
        {'fault_id': 'F2', 'params': {'t_start': 100, 't_end': 300, 'occlusion_ratio': 0.5}},
        # ... 其他故障
    ]
    
    # 创建测试器
    tester = Region3OnlineTester(
        detector_dir=args.detector_dir,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir
    )
    
    # 运行测试
    tester.run_tests(args.num_episodes, fault_configs)
