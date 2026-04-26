#!/usr/bin/env python3
"""
Region 2 GRU 训练数据生成

流程:
1. 收集/生成轨迹数据 (5000 正常 + 2000 故障)
2. 添加扰动 (控制 + 状态 + 参数)
3. 动力学方程推演可达集
4. 计算支撑函数 (上下界)
5. 保存训练数据 (70,000 样本)
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime

# 支撑变量定义 (与 Region 1 一致)
SUPPORT_VARIABLES = [
    'base_x', 'base_y',      # 底盘位置
    'base_v', 'base_ω',       # 底盘速度
    'ee_x', 'ee_y', 'ee_z',   # 末端位置
    'arm_dq_0', 'arm_dq_1', 'arm_dq_2', 'arm_dq_3',  # 关节速度
    'arm_dq_4', 'arm_dq_5', 'arm_dq_6',
    'zmp_x', 'zmp_y',         # ZMP 位置
]

N_VARIABLES = len(SUPPORT_VARIABLES)  # 16
N_OUTPUTS = N_VARIABLES * 2  # 32 (min + max)


class FetchDynamics:
    """Fetch 移动机械臂动力学模型"""
    
    def __init__(self):
        # 底盘参数
        self.mass = 25.0  # kg
        self.length = 0.6  # m
        self.width = 0.6  # m
        self.v_max = 1.0  # m/s
        self.ω_max = 1.5  # rad/s
        self.a_max = 1.0  # m/s²
        self.α_max = 2.0  # rad/s²
        
        # 机械臂参数
        self.arm_length = 0.4  # m (简化)
        self.torque_limits = [50, 50, 30, 30, 20, 20, 10]  # Nm
        
        # 仿真参数
        self.dt = 0.02  # 50Hz
    
    def rollout(self, x0: Dict, u_seq: List[Dict], 
                horizon: float, n_samples: int = 100,
                disturbance: bool = True) -> List[Dict]:
        """
        动力学推演可达集
        
        参数:
            x0: 初始状态
            u_seq: 控制序列
            horizon: 预测时域 (秒)
            n_samples: 蒙特卡洛采样数
            disturbance: 是否添加扰动
        
        返回:
            final_states: 最终状态列表
        """
        n_steps = int(horizon / self.dt)
        final_states = []
        
        for sample in range(n_samples):
            x = x0.copy()
            
            # 添加初始状态扰动
            if disturbance:
                x = self._perturb_state(x, sample)
            
            for step in range(n_steps):
                # 获取控制输入
                if step < len(u_seq):
                    u = u_seq[step].copy()
                else:
                    u = {'v': 0, 'ω': 0, 'τ': np.zeros(7)}
                
                # 添加控制扰动
                if disturbance:
                    u = self._perturb_control(u, sample, step)
                
                # 动力学更新
                x = self._step(x, u)
            
            final_states.append(x)
        
        return final_states
    
    def _step(self, x: Dict, u: Dict) -> Dict:
        """单步动力学更新"""
        x_new = x.copy()
        
        # 底盘运动学 (Unicycle 模型)
        base = x['base'].copy()
        base[0] += base[3] * np.cos(base[2]) * self.dt  # x
        base[1] += base[3] * np.sin(base[2]) * self.dt  # y
        base[2] += base[4] * self.dt                     # θ
        base[3] += u['v'] * self.a_max * self.dt        # v
        base[4] += u['ω'] * self.α_max * self.dt        # ω
        
        # 速度限制
        base[3] = np.clip(base[3], -self.v_max, self.v_max)
        base[4] = np.clip(base[4], -self.ω_max, self.ω_max)
        
        x_new['base'] = base
        
        # 机械臂简化动力学
        arm_dq = x['arm_dq'].copy()
        for i in range(7):
            # 关节加速度
            dqdt = u['τ'][i] / self.torque_limits[i] * 10.0
            arm_dq[i] += dqdt * self.dt
            arm_dq[i] = np.clip(arm_dq[i], -2.0, 2.0)
        
        x_new['arm_dq'] = arm_dq
        
        # 更新末端位置 (简化运动学)
        x_new['ee_x'] = x['base'][0] + self.arm_length * np.cos(x['arm_q'][0])
        x_new['ee_y'] = x['base'][1] + self.arm_length * np.sin(x['arm_q'][0])
        x_new['ee_z'] = 0.3 + x['arm_q'][0] * 0.5
        
        # ZMP 计算
        com_z = 0.5
        ax = u['v'] * self.a_max
        x_new['zmp_x'] = x['base'][0] - (com_z / 9.81) * ax
        x_new['zmp_y'] = x['base'][1] - (com_z / 9.81) * (u['ω'] * self.α_max)
        
        return x_new
    
    def _perturb_state(self, x: Dict, seed: int) -> Dict:
        """添加状态扰动"""
        np.random.seed(seed)
        
        x_new = x.copy()
        x_new['base'] = x['base'] + np.random.randn(5) * 0.1
        x_new['arm_q'] = x['arm_q'] + np.random.randn(7) * 0.05
        x_new['arm_dq'] = x['arm_dq'] + np.random.randn(7) * 0.1
        
        return x_new
    
    def _perturb_control(self, u: Dict, seed: int, step: int) -> Dict:
        """添加控制扰动"""
        np.random.seed(seed * 100 + step)
        
        u_new = u.copy()
        u_new['v'] = u['v'] + np.random.randn() * 0.1
        u_new['ω'] = u['ω'] + np.random.randn() * 0.1
        u_new['τ'] = u['τ'] + np.random.randn(7) * 5
        
        return u_new


def generate_trajectory(n_steps: int = 500, fault_type: str = None) -> Tuple[List[Dict], List[Dict]]:
    """
    生成单条轨迹
    
    参数:
        n_steps: 轨迹长度
        fault_type: 故障类型 (None = 正常)
    
    返回:
        states: 状态序列
        controls: 控制序列
    """
    dynamics = FetchDynamics()
    
    # 初始状态
    x0 = {
        'base': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        'arm_q': np.zeros(7),
        'arm_dq': np.zeros(7),
        'ee_x': 0.4, 'ee_y': 0.0, 'ee_z': 0.3,
        'zmp_x': 0.0, 'zmp_y': 0.0,
    }
    
    states = [x0]
    controls = []
    
    for step in range(n_steps):
        # 生成目标导向的控制
        target_x = 10.0
        current_x = states[-1]['base'][0]
        
        # 比例控制
        v_cmd = np.clip((target_x - current_x) / 5.0, -1, 1)
        ω_cmd = np.sin(step / 50.0) * 0.5
        
        # 机械臂控制 (简化)
        τ_cmd = np.random.randn(7) * 10
        
        # 故障注入
        u = {'v': v_cmd, 'ω': ω_cmd, 'τ': τ_cmd}
        
        if fault_type == 'lighting_drop' and step == 250:
            # 光照突变 → 视觉特征噪声 → 动作错误
            u['v'] *= -1
            u['τ'] *= -1
        
        elif fault_type == 'payload_shift' and step == 200:
            # 负载突变 → 动力学改变
            dynamics.mass += 2.0
        
        elif fault_type == 'joint_friction' and step == 150:
            # 摩擦激增
            u['τ'] *= 0.5
        
        controls.append(u)
        
        # 动力学更新
        x_new = dynamics._step(states[-1], u)
        states.append(x_new)
    
    return states, controls


def compute_support_function(final_states: List[Dict]) -> np.ndarray:
    """
    从可达点云计算支撑函数 (上下界)
    
    返回:
        support: (32,) 数组 [min_0, max_0, min_1, max_1, ...]
    """
    support = np.zeros(N_OUTPUTS)
    
    # 变量名到状态字典键的映射
    var_map = {
        'base_x': ('base', 0),
        'base_y': ('base', 1),
        'base_v': ('base', 3),
        'base_ω': ('base', 4),
        'ee_x': 'ee_x',
        'ee_y': 'ee_y',
        'ee_z': 'ee_z',
        'arm_dq_0': ('arm_dq', 0),
        'arm_dq_1': ('arm_dq', 1),
        'arm_dq_2': ('arm_dq', 2),
        'arm_dq_3': ('arm_dq', 3),
        'arm_dq_4': ('arm_dq', 4),
        'arm_dq_5': ('arm_dq', 5),
        'arm_dq_6': ('arm_dq', 6),
        'zmp_x': 'zmp_x',
        'zmp_y': 'zmp_y',
    }
    
    for i, var in enumerate(SUPPORT_VARIABLES):
        key = var_map[var]
        if isinstance(key, tuple):
            values = [s[key[0]][key[1]] for s in final_states]
        else:
            values = [s.get(key, 0.0) for s in final_states]
        
        support[i * 2] = np.min(values)      # min
        support[i * 2 + 1] = np.max(values)  # max
    
    return support


def generate_training_dataset(n_normal: int = 5000, 
                              n_fault: int = 2000,
                              n_samples: int = 100,
                              horizon: float = 1.0) -> Dict:
    """
    生成完整训练数据集
    
    参数:
        n_normal: 正常轨迹数
        n_fault: 故障轨迹数
        n_samples: 每条轨迹的蒙特卡洛采样数
        horizon: 预测时域 (秒)
    
    返回:
        dataset: {
            'inputs': (N, 10, 19) 状态序列
            'outputs': (N, 32) 支撑函数
        }
    """
    dynamics = FetchDynamics()
    
    inputs = []
    outputs = []
    
    fault_types = [None, 'lighting_drop', 'payload_shift', 'joint_friction']
    
    print(f"开始生成训练数据...")
    print(f"  正常轨迹：{n_normal}")
    print(f"  故障轨迹：{n_fault}")
    print(f"  蒙特卡洛采样：{n_samples}/轨迹")
    print(f"  预测时域：{horizon}s")
    
    total_trajectories = n_normal + n_fault
    
    for traj_id in range(total_trajectories):
        # 确定故障类型
        if traj_id < n_normal:
            fault_type = None
        else:
            fault_type = fault_types[(traj_id - n_normal) % len(fault_types)]
        
        # 生成轨迹
        states, controls = generate_trajectory(n_steps=500, fault_type=fault_type)
        
        # 采样状态序列
        for t in range(10, len(states) - int(horizon / dynamics.dt)):
            # 输入：过去 10 帧状态
            state_seq = []
            for i in range(10):
                s = states[t - 10 + i]
                state_vec = np.concatenate([
                    s['base'][:2],      # base_x, base_y
                    s['base'][3:],      # base_v, base_ω
                    [s['ee_x'], s['ee_y'], s['ee_z']],
                    s['arm_dq'],
                    [s['zmp_x'], s['zmp_y']],
                ])
                state_seq.append(state_vec)
            
            state_seq = np.array(state_seq)  # (10, 19)
            
            # 输出：未来可达集支撑函数
            u_seq = controls[t:t + int(horizon / dynamics.dt)]
            final_states = dynamics.rollout(states[t], u_seq, horizon, n_samples)
            support = compute_support_function(final_states)  # (32,)
            
            inputs.append(state_seq)
            outputs.append(support)
        
        if (traj_id + 1) % 100 == 0:
            print(f"  进度：{traj_id + 1}/{total_trajectories} 轨迹")
    
    dataset = {
        'inputs': np.array(inputs),      # (N, 10, 19)
        'outputs': np.array(outputs),    # (N, 32)
        'metadata': {
            'n_trajectories': total_trajectories,
            'n_samples': n_samples,
            'horizon': horizon,
            'n_variables': N_VARIABLES,
            'generated_at': datetime.now().isoformat(),
        }
    }
    
    print(f"\n数据集生成完成!")
    print(f"  输入形状：{dataset['inputs'].shape}")
    print(f"  输出形状：{dataset['outputs'].shape}")
    print(f"  总样本数：{len(inputs)}")
    
    return dataset


def save_dataset(dataset: Dict, output_path: str):
    """保存数据集"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(output_path,
             inputs=dataset['inputs'],
             outputs=dataset['outputs'],
             metadata=json.dumps(dataset['metadata']))
    
    print(f"数据集已保存：{output_path}")


if __name__ == '__main__':
    # 生成训练数据
    dataset = generate_training_dataset(
        n_normal=5000,
        n_fault=2000,
        n_samples=100,
        horizon=1.0
    )
    
    # 保存
    save_dataset(dataset, '/home/admin/.openclaw/workspace/Embodied-RTA/reachability/gru_training_data.npz')
    
    # 验证
    print("\n验证数据集...")
    print(f"  输入范围：[{dataset['inputs'].min():.3f}, {dataset['inputs'].max():.3f}]")
    print(f"  输出范围：[{dataset['outputs'].min():.3f}, {dataset['outputs'].max():.3f}]")
    print(f"  支撑变量数：{N_VARIABLES}")
    print(f"  输出维度：{N_OUTPUTS}")
