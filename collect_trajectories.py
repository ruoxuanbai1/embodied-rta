#!/usr/bin/env python3
"""
ACT 轨迹数据采集脚本

快速采集正常和故障轨迹数据
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

from lerobot.policies.act.modeling_act import ACTPolicy
from simple_env import SimpleNavigationEnv


class TrajectoryCollector:
    """轨迹采集器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 加载 ACT 模型
        print('加载 ACT 预训练模型...')
        self.model = ACTPolicy.from_pretrained('lerobot/act_aloha_sim_transfer_cube_human')
        self.model.eval()
        self.model.to(device)
        
        # Hook
        self.hook_a = None
        self.hook_b = None
        self._register_hooks()
        
        # 输出目录
        self.output_dir = Path('/root/rt1_trajectory_data')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _register_hooks(self):
        """注册 Hook"""
        def hook_a_fn(module, input, output):
            self.hook_a = output.detach()
        
        def hook_b_fn(module, input, output):
            self.hook_b = output.detach()
        
        self.model.model.encoder.layers[-1].linear2.register_forward_hook(hook_a_fn)
        self.model.model.decoder.layers[-1].linear2.register_forward_hook(hook_b_fn)
    
    def collect_episode(
        self,
        env,
        fault_type='normal',
        max_steps=200
    ) -> dict:
        """采集单条轨迹"""
        states = []
        actions = []
        hook_as = []
        hook_bs = []
        
        state = env.reset()
        
        for step in range(max_steps):
            # 获取观测
            obs = env.get_observation_dict()
            
            # ACT 推理 (select 模式)
            with torch.no_grad():
                # 使用 select 方法而不是直接 forward
                output = self.model.select_action(obs)
                action = output[0].cpu().numpy() if isinstance(output, tuple) else output.cpu().numpy()
            
            # 记录数据
            states.append(state.copy())
            actions.append(action.copy())
            
            # Hook 数据 (如果有)
            if self.hook_a is not None:
                hook_as.append(self.hook_a[0].cpu().numpy().copy())
            if self.hook_b is not None:
                hook_bs.append(self.hook_b[0].cpu().numpy().copy())
            
            # 环境步进
            state, reward, done, info = env.step(action)
            
            if done:
                break
        
        return {
            'fault_type': fault_type,
            'timestamp': datetime.now().isoformat(),
            'states': np.array(states),
            'actions': np.array(actions),
            'hook_a': np.array(hook_as) if hook_as else None,
            'hook_b': np.array(hook_bs) if hook_bs else None,
        }
    
    def collect_normal(self, n_episodes=50):
        """采集正常轨迹"""
        print(f'\n采集正常轨迹：{n_episodes} 条')
        
        for i in range(n_episodes):
            env = SimpleNavigationEnv(scenario='B2')
            traj = self.collect_episode(env, fault_type='normal')
            
            # 保存
            save_path = self.output_dir / f'traj_{i:04d}_normal.npz'
            np.savez_compressed(save_path, **traj)
            
            if (i + 1) % 10 == 0:
                print(f'  已采集 {i+1}/{n_episodes} 条')
        
        print(f'完成 {n_episodes} 条正常轨迹')
    
    def collect_fault(self, fault_name, n_episodes=20):
        """采集故障轨迹"""
        print(f'\n采集故障轨迹：{fault_name} × {n_episodes} 条')
        
        for i in range(n_episodes):
            env = SimpleNavigationEnv(scenario='B2')
            
            # 注入故障 (第 10 步开始)
            env.fault_type = fault_name
            env.fault_active = True
            env.fault_start_step = 10
            env.fault_duration = 50
            
            traj = self.collect_episode(env, fault_type=fault_name)
            
            # 保存
            traj_id = len(list(self.output_dir.glob('*.npz')))
            save_path = self.output_dir / f'traj_{traj_id:04d}_{fault_name}.npz'
            np.savez_compressed(save_path, **traj)
            
            if (i + 1) % 5 == 0:
                print(f'  已采集 {i+1}/{n_episodes} 条')
        
        print(f'完成 {fault_name} × {n_episodes} 条')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-normal', type=int, default=50)
    parser.add_argument('--n-fault', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='/root/rt1_trajectory_data')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('ACT 轨迹数据采集')
    print('=' * 60)
    
    collector = TrajectoryCollector(device=args.device)
    
    # 采集正常数据
    collector.collect_normal(args.n_normal)
    
    # 采集故障数据
    fault_types = [
        '光照突变', '传感器噪声', '执行器退化',
        '负载突变', '动态闯入',
    ]
    
    for fault in fault_types:
        collector.collect_fault(fault, args.n_fault)
    
    # 统计
    n_files = len(list(collector.output_dir.glob('*.npz')))
    print(f'\n✅ 采集完成! 共 {n_files} 条轨迹')
    print(f'保存位置：{collector.output_dir}')
