#!/usr/bin/env python3
"""
ACT 轨迹数据采集 - 使用 ACT 原始接口

不修改 ACT 模型，只提供正确的观测格式
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig


class ACTTrajectoryCollector:
    """ACT 轨迹采集器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 加载 ACT 模型和配置
        print('加载 ACT 预训练模型...')
        config = ACTConfig()
        print(f'ACT 配置：dim={config.dim}, n_encoder_layers={config.n_encoder_layers}')
        
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
        """注册 Hook 到 Encoder/Decoder 最后层 FFN"""
        def hook_a_fn(module, input, output):
            self.hook_a = output.detach().cpu()
        
        def hook_b_fn(module, input, output):
            self.hook_b = output.detach().cpu()
        
        # 节点 A: Encoder 最后层 linear2
        self.model.model.encoder.layers[-1].linear2.register_forward_hook(hook_a_fn)
        
        # 节点 B: Decoder 最后层 linear2
        self.model.model.decoder.layers[-1].linear2.register_forward_hook(hook_b_fn)
        
        print('Hook 已注册')
    
    def create_observation(self, state: np.ndarray) -> dict:
        """
        创建 ACT 观测字典
        
        根据 ACT 配置创建正确的观测格式
        """
        # ACT 需要的观测格式 (根据 config)
        # observation.state: (B, state_dim)
        # observation.images.top: (B, C, H, W)
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # 创建符合 ACT 要求的图像观测
        # ACT 使用 ResNet18 backbone，需要 3×256×256 或 3×480×640
        dummy_image = torch.randn(1, 3, 256, 256, device=self.device) * 0.1
        
        return {
            'observation.state': state_tensor,
            'observation.images.top': dummy_image,
        }
    
    def collect_episode(
        self,
        state_dim: int = 14,
        action_dim: int = 14,
        fault_type: str = 'normal',
        max_steps: int = 200
    ) -> dict:
        """
        采集单条轨迹
        
        使用 ACT 模型生成动作
        """
        states = []
        actions = []
        hook_as = []
        hook_bs = []
        
        # 初始状态
        state = np.zeros(state_dim)
        state[2] = 1.0  # z 高度
        
        for step in range(max_steps):
            # 创建观测
            obs = self.create_observation(state)
            
            # ACT 推理
            with torch.no_grad():
                # 使用 forward 生成动作
                output = self.model(obs)
                action = output['action'][0, 0].cpu().numpy()
            
            # 记录数据
            states.append(state.copy())
            actions.append(action.copy())
            
            if self.hook_a is not None:
                hook_as.append(self.hook_a.numpy().copy().flatten())
            if self.hook_b is not None:
                hook_bs.append(self.hook_b.numpy().copy().flatten())
            
            # 简单动力学模型
            dt = 0.02
            state[0:3] += action[0:3] * dt  # 位置
            state[3:6] = action[0:3]  # 速度
            state[6:9] += action[3:6] * dt  # 姿态
            
            # 检查目标到达
            dist_to_goal = np.linalg.norm(state[0:3] - np.array([10.0, 0.0, 1.0]))
            if dist_to_goal < 0.5 or step >= max_steps - 1:
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
            traj = self.collect_episode(fault_type='normal')
            
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
            # 添加状态噪声模拟故障
            state_noise = np.random.randn(14) * 0.2
            traj = self.collect_episode(fault_type=fault_name)
            
            # 在状态上添加噪声
            traj['states'] += state_noise
            
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
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('ACT 轨迹数据采集 (原始接口)')
    print('=' * 60)
    
    collector = ACTTrajectoryCollector(device=args.device)
    
    # 采集正常数据
    collector.collect_normal(args.n_normal)
    
    # 采集故障数据
    fault_types = ['光照突变', '传感器噪声', '执行器退化', '负载突变', '动态闯入']
    
    for fault in fault_types:
        collector.collect_fault(fault, args.n_fault)
    
    # 统计
    n_files = len(list(collector.output_dir.glob('*.npz')))
    print(f'\n✅ 采集完成! 共 {n_files} 条轨迹')
    print(f'保存位置：{collector.output_dir}')
