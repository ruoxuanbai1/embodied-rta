#!/usr/bin/env python3
"""
基于 OpenVLA 真实轨迹生成 Region 2 训练数据

输入：
- reachability/openvla_trajectories_normal.pkl (正常轨迹)
- reachability/openvla_trajectories_faults.pkl (故障轨迹)

输出：
- reachability/openvla_reachability_dataset.h5 (训练数据)

数据规格:
- 输入：(N, 10, 19) - 10 帧状态序列
- 输出：(N, 32) - 16 变量的 min+max 支撑函数
"""

import sys
import numpy as np
import pickle
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# 添加路径
PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT))

print("="*80)
print("Region 2: 基于 OpenVLA 真实轨迹生成训练数据")
print("="*80)

# ============== 配置 ==============

WINDOW_SIZE = 10      # 输入窗口大小 (10 帧)
HORIZON = 1.0         # 预测时域 (秒)
N_SAMPLES = 100       # 蒙特卡洛采样数
DT = 0.02             # 控制频率 50Hz

# 支撑变量 (16 个)
SUPPORT_VARIABLES = [
    'base_x', 'base_y',      # 底盘位置
    'base_v', 'base_omega',  # 底盘速度
    'base_a',                # 底盘加速度 (从动作推断)
    'ee_x', 'ee_y', 'ee_z',  # 末端位置 (运动学计算)
    'arm_dq_0', 'arm_dq_1', 'arm_dq_2', 'arm_dq_3',  # 关节速度
    'arm_dq_4', 'arm_dq_5', 'arm_dq_6',
    'zmp_x', 'zmp_y',        # ZMP 位置
]

N_VARIABLES = len(SUPPORT_VARIABLES)  # 16
N_OUTPUTS = N_VARIABLES * 2  # 32 (min + max)

# ============== 加载轨迹数据 ==============

print("\n[1/5] 加载 OpenVLA 真实轨迹...")

with open(PROJECT_ROOT / 'reachability' / 'openvla_trajectories_normal.pkl', 'rb') as f:
    normal_trajs = pickle.load(f)

with open(PROJECT_ROOT / 'reachability' / 'openvla_trajectories_faults.pkl', 'rb') as f:
    fault_trajs = pickle.load(f)

print(f"  正常轨迹：{len(normal_trajs)} 条")
print(f"  故障轨迹：{len(fault_trajs)} 条")
print(f"  总计：{len(normal_trajs) + len(fault_trajs)} 条")

# ============== 动力学模型 (用于可达集推演) ==============

class FetchDynamics:
    """Fetch 移动机械臂简化动力学 (用于蒙特卡洛推演)"""
    
    def __init__(self):
        self.dt = DT
        self.v_max = 1.0
        self.ω_max = 1.5
        self.a_max = 1.0
        self.α_max = 2.0
    
    def state_to_vector(self, state: Dict) -> np.ndarray:
        """状态字典 → 向量 (19 维)"""
        base = state['base']  # [x, y, θ, v, ω]
        arm_dq = state['arm_dq']  # [dq1...dq7]
        
        # 计算末端位置 (简化运动学)
        ee_x = base[0] + 0.4 * np.cos(base[2])
        ee_y = base[1] + 0.4 * np.sin(base[2])
        ee_z = 0.3 + base[0] * 0.1  # 简化
        
        # ZMP 计算
        zmp_x = base[0] - (0.5 / 9.81) * base[3] * self.a_max
        zmp_y = base[1] - (0.5 / 9.81) * base[4] * self.α_max
        
        return np.array([
            base[0], base[1],       # x, y
            base[3], base[4],       # v, ω
            0.0,                    # a (从动作计算)
            ee_x, ee_y, ee_z,       # 末端位置
            arm_dq[0], arm_dq[1], arm_dq[2], arm_dq[3],  # 关节速度
            arm_dq[4], arm_dq[5], arm_dq[6],
            zmp_x, zmp_y,           # ZMP
        ])
    
    def rollout(self, x0: np.ndarray, u_seq: List[Dict], 
                n_samples: int = 100) -> np.ndarray:
        """
        蒙特卡洛可达集推演
        
        参数:
            x0: 初始状态向量 (19 维)
            u_seq: 控制序列
            n_samples: 采样数
        
        返回:
            final_states: (n_samples, 19) 最终状态矩阵
        """
        n_steps = len(u_seq)
        final_states = []
        
        for sample in range(n_samples):
            x = x0.copy()
            
            # 添加初始状态扰动
            x += np.random.randn(19) * 0.05
            
            for step in range(n_steps):
                u = u_seq[step]
                
                # 添加控制扰动
                v_perturb = u['v'] + np.random.randn() * 0.1
                ω_perturb = u['omega'] + np.random.randn() * 0.1
                
                # 底盘运动学更新
                x[0] += x[2] * np.cos(x[1]) * self.dt  # x
                x[1] += x[2] * np.sin(x[1]) * self.dt  # y (θ在索引 2 是简化)
                x[2] += v_perturb * self.a_max * self.dt  # v
                x[3] += ω_perturb * self.α_max * self.dt  # ω
                
                # 速度限制
                x[2] = np.clip(x[2], -self.v_max, self.v_max)
                x[3] = np.clip(x[3], -self.ω_max, self.ω_max)
                
                # 加速度 (从速度变化计算)
                x[4] = v_perturb * self.a_max
            
            final_states.append(x)
        
        return np.array(final_states)


def compute_support_function(final_states: np.ndarray) -> np.ndarray:
    """
    从可达点云计算支撑函数 (上下界)
    
    参数:
        final_states: (n_samples, 19) 最终状态矩阵
    
    返回:
        support: (32,) 支撑函数 [min_0, max_0, min_1, max_1, ...]
    """
    support = np.zeros(N_OUTPUTS)
    
    for i in range(N_VARIABLES):
        support[i * 2] = np.min(final_states[:, i])      # min
        support[i * 2 + 1] = np.max(final_states[:, i])  # max
    
    return support


# ============== 生成训练数据 ==============

print("\n[2/5] 生成训练数据...")

dynamics = FetchDynamics()
training_data = []

# 处理所有轨迹
all_trajs = normal_trajs + fault_trajs

for traj_idx, traj in enumerate(all_trajs):
    states = traj['states']
    actions = traj['actions']
    
    # 滑动窗口 (窗口大小=10)
    for window_start in range(0, len(states) - WINDOW_SIZE, 5):  # 步长=5
        # 输入：过去 10 帧状态
        window_states = states[window_start:window_start + WINDOW_SIZE]
        window_actions = actions[window_start:window_start + WINDOW_SIZE]
        
        # 转换为向量
        input_seq = np.array([dynamics.state_to_vector(s) for s in window_states])
        
        # 最后一帧作为初始状态
        x0 = input_seq[-1].copy()
        
        # 蒙特卡洛可达集推演
        final_states = dynamics.rollout(x0, window_actions, n_samples=N_SAMPLES)
        
        # 计算支撑函数
        support = compute_support_function(final_states)
        
        # 保存训练样本
        training_data.append({
            'input': input_seq,    # (10, 19)
            'output': support,     # (32,)
            'traj_id': traj_idx,
            'window_start': window_start,
        })
    
    # 进度汇报
    if (traj_idx + 1) % 20 == 0:
        print(f"  进度：{traj_idx + 1}/{len(all_trajs)} 轨迹 | "
              f"已生成 {len(training_data)} 样本")

print(f"  ✅ 共生成 {len(training_data)} 训练样本")

# ============== 数据统计 ==============

print("\n[3/5] 数据统计...")

inputs = np.array([d['input'] for d in training_data])
outputs = np.array([d['output'] for d in training_data])

print(f"  输入形状：{inputs.shape}")  # (N, 10, 19)
print(f"  输出形状：{outputs.shape}")  # (N, 32)

# 支撑变量统计
print(f"\n  支撑变量范围:")
for i in range(N_VARIABLES):
    var_name = SUPPORT_VARIABLES[i]
    min_val = outputs[:, i * 2].min()
    max_val = outputs[:, i * 2 + 1].max()
    print(f"    {var_name}: [{min_val:.3f}, {max_val:.3f}]")

# ============== 保存 HDF5 ==============

print("\n[4/5] 保存 HDF5 文件...")

output_file = PROJECT_ROOT / 'reachability' / 'openvla_reachability_dataset.h5'

with h5py.File(output_file, 'w') as f:
    # 数据
    f.create_dataset('inputs', data=inputs, compression='gzip')
    f.create_dataset('outputs', data=outputs, compression='gzip')
    
    # 元数据
    f.attrs['n_samples'] = len(training_data)
    f.attrs['window_size'] = WINDOW_SIZE
    f.attrs['horizon'] = HORIZON
    f.attrs['n_variables'] = N_VARIABLES
    f.attrs['n_outputs'] = N_OUTPUTS
    f.attrs['support_variables'] = SUPPORT_VARIABLES
    f.attrs['creation_time'] = datetime.now().isoformat()
    
    # 变量名
    var_dtype = h5py.special_dtype(vlen=str)
    var_names = f.create_dataset('variable_names', (N_VARIABLES,), dtype=var_dtype)
    for i, name in enumerate(SUPPORT_VARIABLES):
        var_names[i] = name

print(f"✅ 数据已保存：{output_file}")

# 文件大小
file_size_mb = output_file.stat().st_size / 1024 / 1024
print(f"  文件大小：{file_size_mb:.1f} MB")

# ============== 验证数据 ==============

print("\n[5/5] 验证数据...")

with h5py.File(output_file, 'r') as f:
    inputs_loaded = f['inputs'][:]
    outputs_loaded = f['outputs'][:]
    
    assert inputs_loaded.shape == inputs.shape, "输入形状不匹配"
    assert outputs_loaded.shape == outputs.shape, "输出形状不匹配"
    
    # 随机检查几个样本
    for i in range(5):
        idx = np.random.randint(len(training_data))
        assert np.allclose(inputs_loaded[idx], inputs[idx]), f"样本 {idx} 输入不匹配"
        assert np.allclose(outputs_loaded[idx], outputs[idx]), f"样本 {idx} 输出不匹配"

print("✅ 数据验证通过!")

# ============== 总结 ==============

print("\n" + "="*80)
print("Region 2 训练数据生成完成!")
print("="*80)
print(f"""
数据来源:
  - OpenVLA 真实轨迹：{len(all_trajs)} 条
  - 训练样本：{len(training_data)} 个

数据规格:
  - 输入：(N, 10, 19) - 10 帧状态序列
  - 输出：(N, 32) - 16 变量×2 (min+max)

支撑变量 (16 个):
  {', '.join(SUPPORT_VARIABLES)}

输出文件:
  {output_file}

下一步:
  python3 reachability/train_gru_reachability.py
  (使用生成的数据训练 GRU 可达性预测模型)
""")
print("="*80)
