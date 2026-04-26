#!/usr/bin/env python3
"""
ACT 轨迹数据采集器

在仿真环境中运行 ACT 预训练模型，收集:
- 状态序列
- 动作序列
- Hook 激活 (节点 A + 节点 B)
- 故障标签

用于训练 Region 2 GRU 和 Region 3 检测器
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from lerobot.policies.act.modeling_act import ACTPolicy


class ACTTrajectoryCollector:
    """
    ACT 轨迹数据采集器
    
    收集的数据用于:
    1. Region 2 GRU 训练 (状态 + 动作序列)
    2. Region 3 阈值学习 (正常 vs 故障轨迹)
    3. Region 3 掩码库构建 (激活模式)
    4. Region 3 关键特征集挖掘 (SHAP 分析)
    """
    
    def __init__(
        self,
        model_id: str = 'lerobot/act_aloha_sim_transfer_cube_human',
        device: str = 'cuda',
        output_dir: str = '/root/rt1_trajectory_data'
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载 ACT 预训练模型
        print(f'加载 ACT 模型：{model_id}')
        self.model = ACTPolicy.from_pretrained(model_id)
        self.model.eval()
        self.model.to(device)
        
        # Hook 挂载点
        self.hook_a: Optional[torch.Tensor] = None  # 节点 A (Encoder FFN)
        self.hook_b: Optional[torch.Tensor] = None  # 节点 B (Decoder FFN)
        
        self._register_hooks()
        
        # 数据缓存
        self.trajectories = []
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'hook_a': [],
            'hook_b': [],
            'images': [],
            'fault_type': None,
            'timestamp': datetime.now().isoformat(),
        }
    
    def _register_hooks(self):
        """注册 Hook 捕获激活"""
        
        def hook_a_fn(module, input, output):
            self.hook_a = output.detach()
        
        def hook_b_fn(module, input, output):
            self.hook_b = output.detach()
        
        # 节点 A: Encoder 最后层 FFN
        encoder_layer = self.model.model.encoder.layers[-1]
        encoder_layer.linear2.register_forward_hook(hook_a_fn)
        
        # 节点 B: Decoder 最后层 FFN
        decoder_layer = self.model.model.decoder.layers[-1]
        decoder_layer.linear2.register_forward_hook(hook_b_fn)
        
        print('Hook 已注册:')
        print(f'  节点 A: model.encoder.layers.{len(self.model.model.encoder.layers)-1}.linear2')
        print(f'  节点 B: model.decoder.layers.{len(self.model.model.decoder.layers)-1}.linear2')
    
    def reset(self, fault_type: Optional[str] = None):
        """开始新轨迹采集"""
        if self.current_trajectory['states']:
            self._save_trajectory()
        
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'hook_a': [],
            'hook_b': [],
            'images': [],
            'fault_type': fault_type,
            'timestamp': datetime.now().isoformat(),
        }
    
    def step(self, state: np.ndarray, image: Optional[np.ndarray] = None):
        """
        执行一步并采集数据
        
        Args:
            state: (state_dim,) 本体状态
            image: (H, W, 3) 视觉观测 (可选)
        
        Returns:
            action: (action_dim,) ACT 输出动作
        """
        # 预处理
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # 构造 ACT 输入
        batch_obs = {
            'observation.state': state_tensor,
        }
        
        if image is not None:
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            image_tensor = image_tensor.to(self.device)
            batch_obs['observation.images.top'] = image_tensor
        
        # ACT 前向传播
        with torch.no_grad():
            output = self.model(batch_obs)
        
        action = output['action'][0, 0].cpu().numpy()  # 取第一个动作
        
        # 记录数据
        self.current_trajectory['states'].append(state.copy())
        self.current_trajectory['actions'].append(action.copy())
        
        if self.hook_a is not None:
            self.current_trajectory['hook_a'].append(self.hook_a[0].cpu().numpy().copy())
        if self.hook_b is not None:
            self.current_trajectory['hook_b'].append(self.hook_b[0].cpu().numpy().copy())
        
        if image is not None:
            self.current_trajectory['images'].append(image.copy())
        
        return action
    
    def _save_trajectory(self):
        """保存当前轨迹"""
        if not self.current_trajectory['states']:
            return
        
        trajectory_id = len(self.trajectories)
        fault_type = self.current_trajectory['fault_type'] or 'normal'
        
        # 转换为 numpy 数组
        traj_data = {
            'id': trajectory_id,
            'fault_type': fault_type,
            'timestamp': self.current_trajectory['timestamp'],
            'states': np.array(self.current_trajectory['states']),
            'actions': np.array(self.current_trajectory['actions']),
            'hook_a': np.array(self.current_trajectory['hook_a']) if self.current_trajectory['hook_a'] else None,
            'hook_b': np.array(self.current_trajectory['hook_b']) if self.current_trajectory['hook_b'] else None,
        }
        
        # 保存为 npz
        save_path = self.output_dir / f'trajectory_{trajectory_id:04d}_{fault_type}.npz'
        np.savez_compressed(save_path, **traj_data)
        
        self.trajectories.append(traj_data)
        print(f'保存轨迹 [{trajectory_id:04d}] {fault_type}: {len(traj_data["states"])} 步')
        
        # 清空当前轨迹
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'hook_a': [],
            'hook_b': [],
            'images': [],
            'fault_type': None,
            'timestamp': datetime.now().isoformat(),
        }
    
    def collect_normal_data(self, env, n_episodes: int = 50, max_steps: int = 200):
        """
        收集正常运行数据
        
        Args:
            env: 仿真环境
            n_episodes: 轨迹数量
            max_steps: 每轨迹最大步数
        """
        print(f'\n收集正常数据：{n_episodes} 条轨迹')
        
        for ep in range(n_episodes):
            self.reset(fault_type='normal')
            state = env.reset()
            
            for step in range(max_steps):
                action = self.step(state)
                state, reward, done, info = env.step(action)
                
                if done:
                    break
            
            self._save_trajectory()
        
        print(f'完成 {n_episodes} 条正常轨迹')
    
    def collect_fault_data(self, env, fault_injector, n_per_fault: int = 20, max_steps: int = 200):
        """
        收集故障数据
        
        Args:
            env: 仿真环境
            fault_injector: 故障注入器
            n_per_fault: 每种故障的轨迹数
            max_steps: 每轨迹最大步数
        """
        fault_types = fault_injector.get_fault_types()
        
        for fault_type in fault_types:
            print(f'\n收集故障数据：{fault_type} × {n_per_fault} 条')
            
            for i in range(n_per_fault):
                self.reset(fault_type=fault_type)
                state = env.reset()
                
                # 注入故障
                fault_injector.inject(fault_type)
                
                for step in range(max_steps):
                    action = self.step(state)
                    state, reward, done, info = env.step(action)
                    
                    if done:
                        break
                
                # 清除故障
                fault_injector.clear()
                
                self._save_trajectory()
        
        print(f'完成 {len(fault_types) * n_per_fault} 条故障轨迹')
    
    def get_statistics(self) -> Dict:
        """获取数据统计"""
        normal_count = sum(1 for t in self.trajectories if t['fault_type'] == 'normal')
        fault_count = len(self.trajectories) - normal_count
        
        return {
            'total_trajectories': len(self.trajectories),
            'normal': normal_count,
            'fault': fault_count,
            'output_dir': str(self.output_dir),
        }


class SimpleFaultInjector:
    """简单故障注入器"""
    
    def __init__(self):
        self.active_fault = None
        self.fault_types = [
            'lighting_drop',      # 光照突变
            'sensor_noise',       # 传感器噪声
            'actuator_degradation',  # 执行器退化
            'dynamic_obstacle',   # 动态障碍物
        ]
    
    def get_fault_types(self) -> List[str]:
        return self.fault_types
    
    def inject(self, fault_type: str):
        self.active_fault = fault_type
        # 实际实现需要根据环境修改
    
    def clear(self):
        self.active_fault = None


if __name__ == '__main__':
    print('=== ACT 轨迹数据采集器 ===')
    print('此脚本用于从 ACT 预训练模型收集轨迹数据')
    print('')
    print('使用方法:')
    print('1. 在完整试验运行器中集成此采集器')
    print('2. 运行试验时自动收集轨迹')
    print('3. 使用 collect_statistics.py 分析数据')
    print('')
    
    # 测试模型加载
    device = 'cuda'
    collector = ACTTrajectoryCollector(device=device)
    
    print(f'\\n✅ 模型加载成功')
    print(f'Hook 已注册，准备采集轨迹')
