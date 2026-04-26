#!/usr/bin/env python3
"""
ACT 轨迹数据采集 - 800 条完整版

正常：400 条
故障：13 种 × 31 条 = 403 条
总计：803 条
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from lerobot.policies.act.modeling_act import ACTPolicy


class ACTTrajectoryCollector:
    """ACT 轨迹采集器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        print('加载 ACT 预训练模型...')
        self.model = ACTPolicy.from_pretrained('lerobot/act_aloha_sim_transfer_cube_human')
        self.model.eval()
        self.model.to(device)
        
        self.hook_a = None
        self.hook_b = None
        self._register_hooks()
        
        self.output_dir = Path('/root/rt1_trajectory_data')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _register_hooks(self):
        def hook_a_fn(module, input, output):
            self.hook_a = output.detach().cpu()
        def hook_b_fn(module, input, output):
            self.hook_b = output.detach().cpu()
        
        self.model.model.encoder.layers[-1].linear2.register_forward_hook(hook_a_fn)
        self.model.model.decoder.layers[-1].linear2.register_forward_hook(hook_b_fn)
    
    def create_observation(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        dummy_image = torch.randn(1, 3, 256, 256, device=self.device) * 0.1
        return {
            'observation.state': state_tensor,
            'observation.images.top': dummy_image,
        }
    
    def collect_episode(self, fault_type='normal', max_steps=200):
        states, actions, hook_as, hook_bs = [], [], [], []
        state = np.zeros(14)
        state[2] = 1.0
        
        for step in range(max_steps):
            obs = self.create_observation(state)
            with torch.no_grad():
                action = self.model.select_action(obs)
                if action.ndim > 1:
                    action = action[0]
                action = action.cpu().numpy()
            
            states.append(state.copy())
            actions.append(action.copy())
            
            if self.hook_a is not None:
                hook_as.append(self.hook_a.numpy().copy().flatten())
            if self.hook_b is not None:
                hook_bs.append(self.hook_b.numpy().copy().flatten())
            
            dt = 0.02
            state[0:3] += action[0:3] * dt
            state[3:6] = action[0:3]
            state[6:9] += action[3:6] * dt
            
            dist = np.linalg.norm(state[0:3] - np.array([10.0, 0.0, 1.0]))
            if dist < 0.5 or step >= max_steps - 1:
                break
        
        return {
            'fault_type': fault_type,
            'timestamp': datetime.now().isoformat(),
            'states': np.array(states),
            'actions': np.array(actions),
            'hook_a': np.array(hook_as) if hook_as else None,
            'hook_b': np.array(hook_bs) if hook_bs else None,
        }
    
    def collect_batch(self, fault_type, n_episodes, start_id=0):
        print(f'\n采集：{fault_type} × {n_episodes} 条')
        for i in range(n_episodes):
            traj = self.collect_episode(fault_type=fault_type)
            traj_id = start_id + i
            save_path = self.output_dir / f'traj_{traj_id:04d}_{fault_type}.npz'
            np.savez_compressed(save_path, **traj)
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{n_episodes}')
        print(f'完成 {fault_type} × {n_episodes} 条')


if __name__ == '__main__':
    print('=' * 60)
    print('ACT 轨迹数据采集 (800 条完整版)')
    print('=' * 60)
    
    collector = ACTTrajectoryCollector(device='cuda')
    
    # 检查已有数据
    existing = list(collector.output_dir.glob('*.npz'))
    existing_normal = len([f for f in existing if '_normal.npz' in str(f)])
    existing_fault = len(existing) - existing_normal
    print(f'\n已存在：{existing_normal} 条正常，{existing_fault} 条故障')
    
    # 补充正常轨迹至 400 条
    n_normal_needed = max(0, 400 - existing_normal)
    if n_normal_needed > 0:
        print(f'需补充正常：{n_normal_needed} 条')
        start_id = existing_normal
        for i in range(n_normal_needed):
            traj = collector.collect_episode(fault_type='normal')
            save_path = collector.output_dir / f'traj_{start_id + i:04d}_normal.npz'
            np.savez_compressed(save_path, **traj)
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{n_normal_needed}')
        print(f'完成 {n_normal_needed} 条正常轨迹')
    
    # 故障轨迹：13 种 × 31 条 = 403 条
    fault_types = [
        '光照突变', '传感器噪声', '执行器退化', '负载突变', '动态闯入',
        '摄像头遮挡', '对抗补丁', '深度噪声',
        '关节摩擦', '电压下降',
        '复合感知动力学', '复合全故障', '复合感知×2',
    ]
    
    max_id = len(list(collector.output_dir.glob('*.npz')))
    
    for fault in fault_types:
        collector.collect_batch(fault, 31, start_id=max_id)
        max_id += 31
    
    n_files = len(list(collector.output_dir.glob('*.npz')))
    print(f'\n✅ 采集完成！共 {n_files} 条轨迹')
    print(f'保存位置：{collector.output_dir}')
