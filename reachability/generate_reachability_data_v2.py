#!/usr/bin/env python3
"""
Region 2 GRU 训练数据生成 (v2 - 基于 OpenVLA 真实轨迹)

流程:
1. 加载 OpenVLA 收集的真实轨迹
2. 在轨迹上添加扰动（控制/状态/参数）
3. 用动力学模型推演可达集
4. 计算支撑函数作为标签
5. 保存训练数据

版本：2.0 (2026-03-31)
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*80)
print("Region 2 GRU 训练数据生成 (v2 - OpenVLA 真实轨迹)")
print(f"开始时间：{datetime.now()}")
print("="*80)

# ============ 配置 ============

TRAJECTORY_DIR = PROJECT_ROOT / 'reachability' / 'openvla_trajectories'
OUTPUT_PATH = PROJECT_ROOT / 'reachability' / 'gru_training_data_v2.npz'

# 支撑变量 (16 维)
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

# 扰动配置
PERTURBATIONS = {
    'state_noise_std': 0.05,      # 状态噪声标准差 (5%)
    'control_noise_std': 0.1,     # 控制噪声标准差 (10%)
    'param_noise_std': 0.02,      # 参数噪声标准差 (2%)
    'n_monte_carlo': 50,          # 蒙特卡洛采样数
}

# 预测时域
PREDICTION_HORIZON = 1.0  # 秒
DT = 0.02  # 50Hz


class FetchDynamics:
    """Fetch 移动机械臂动力学模型"""
    
    def __init__(self, nominal_params: Dict = None):
        # 标称参数
        self.params = nominal_params or {
            'mass': 25.0,
            'length': 0.6,
            'width': 0.6,
            'arm_length': 0.4,
        }
        
        # 仿真参数
        self.dt = DT
    
    def perturb_params(self, seed: int) -> 'FetchDynamics':
        """创建扰动后的动力学模型"""
        np.random.seed(seed)
        
        perturbed = {
            'mass': self.params['mass'] * (1 + np.random.randn() * PERTURBATIONS['param_noise_std']),
            'length': self.params['length'] * (1 + np.random.randn() * PERTURBATIONS['param_noise_std']),
            'width': self.params['width'] * (1 + np.random.randn() * PERTURBATIONS['param_noise_std']),
            'arm_length': self.params['arm_length'] * (1 + np.random.randn() * PERTURBATIONS['param_noise_std']),
        }
        
        return FetchDynamics(perturbed)
    
    def rollout(self, x0: np.ndarray, u_seq: np.ndarray, 
                horizon: float = PREDICTION_HORIZON) -> np.ndarray:
        """
        动力学推演
        
        参数:
            x0: 初始状态 (16,)
            u_seq: 控制序列 (T, 2) [v, ω]
            horizon: 预测时域
        
        返回:
            final_states: 最终状态 (N, 16)
        """
        n_steps = int(horizon / self.dt)
        
        # 简化动力学模型
        # TODO: 实现完整的 Fetch 动力学
        
        x = x0.copy()
        for step in range(n_steps):
            # 底盘运动学
            if step < len(u_seq):
                v, ω = u_seq[step]
            else:
                v, ω = 0.0, 0.0
            
            x[0] += v * np.cos(x[2]) * self.dt  # base_x
            x[1] += v * np.sin(x[2]) * self.dt  # base_y
            x[2] += ω * self.dt                  # base_θ
            x[3] = v                             # base_v
            x[4] = ω                             # base_ω
            
            # TODO: 机械臂动力学
        
        return x
    
    def compute_reachable_set(self, x0: np.ndarray, u_seq: np.ndarray,
                               n_samples: int = None) -> np.ndarray:
        """
        计算可达集 (蒙特卡洛方法)
        
        参数:
            x0: 初始状态
            u_seq: 控制序列
            n_samples: 采样数
        
        返回:
            reachable_states: 可达状态 (N, 16)
        """
        n_samples = n_samples or PERTURBATIONS['n_monte_carlo']
        
        final_states = []
        
        for i in range(n_samples):
            # 扰动初始状态
            x0_perturbed = x0 + np.random.randn(len(x0)) * PERTURBATIONS['state_noise_std']
            
            # 扰动控制序列
            u_perturbed = u_seq + np.random.randn(*u_seq.shape) * PERTURBATIONS['control_noise_std']
            
            # 扰动动力学参数
            model = self.perturb_params(seed=i)
            
            # 推演
            x_final = model.rollout(x0_perturbed, u_perturbed)
            final_states.append(x_final)
        
        return np.array(final_states)


def compute_support_function(states: np.ndarray) -> np.ndarray:
    """
    计算支撑函数 (可达集上下界)
    
    参数:
        states: 状态样本 (N, 16)
    
    返回:
        support: 支撑函数 (32,) [min_0, max_0, min_1, max_1, ...]
    """
    support = []
    
    for i in range(states.shape[1]):
        support.append(states[:, i].min())
        support.append(states[:, i].max())
    
    return np.array(support)


def process_trajectory(traj_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    处理单条轨迹
    
    参数:
        traj_path: 轨迹文件路径
    
    返回:
        inputs: 输入序列列表 [(10, 16), ...]
        outputs: 输出标签列表 [(32,), ...]
    """
    data = np.load(traj_path, allow_pickle=True)
    states = data['states']  # (T, 16)
    
    # 加载元数据
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    inputs = []
    outputs = []
    
    dynamics = FetchDynamics()
    
    # 滑动窗口处理
    window_size = 10  # 过去 10 帧
    horizon = PREDICTION_HORIZON
    n_steps = int(horizon / DT)
    
    for t in range(window_size, len(states) - n_steps):
        # 输入：过去 10 帧状态
        input_seq = states[t-window_size:t]  # (10, 16)
        
        # 提取控制序列 (从动作中估计)
        # TODO: 从动作中提取 v, ω
        u_seq = np.zeros((n_steps, 2))
        for i in range(n_steps):
            if t+i < len(states) - 1:
                dx = states[t+i+1, 0] - states[t+i, 0]
                dy = states[t+i+1, 1] - states[t+i, 1]
                v = np.sqrt(dx**2 + dy**2) / DT
                u_seq[i, 0] = np.clip(v, -1.0, 1.0)
                u_seq[i, 1] = 0.0  # 简化
        
        # 计算可达集
        x0 = states[t]
        reachable = dynamics.compute_reachable_set(x0, u_seq)
        
        # 计算支撑函数
        support = compute_support_function(reachable)
        
        inputs.append(input_seq)
        outputs.append(support)
    
    return inputs, outputs, metadata


def main():
    """主函数"""
    
    # 检查轨迹目录
    if not TRAJECTORY_DIR.exists():
        print(f"❌ 轨迹目录不存在：{TRAJECTORY_DIR}")
        print("请先运行：collect_openvla_trajectories.py")
        sys.exit(1)
    
    # 查找所有轨迹文件
    traj_files = list(TRAJECTORY_DIR.glob('traj_*.npz'))
    
    if len(traj_files) == 0:
        print(f"❌ 未找到轨迹文件：{TRAJECTORY_DIR}")
        sys.exit(1)
    
    print(f"\n找到 {len(traj_files)} 条轨迹")
    print(f"输出路径：{OUTPUT_PATH}")
    print(f"扰动配置：{PERTURBATIONS}")
    print()
    
    # 并行处理轨迹
    all_inputs = []
    all_outputs = []
    all_metadata = []
    
    n_workers = min(8, os.cpu_count())
    print(f"使用 {n_workers} 个工作进程并行处理\n")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_traj = {
            executor.submit(process_trajectory, traj): traj 
            for traj in traj_files
        }
        
        for i, future in enumerate(as_completed(future_to_traj)):
            traj_path = future_to_traj[future]
            
            try:
                inputs, outputs, metadata = future.result()
                all_inputs.extend(inputs)
                all_outputs.extend(outputs)
                all_metadata.append(metadata)
                
                # 进度报告
                if (i + 1) % 10 == 0:
                    print(f"进度：{i+1}/{len(traj_files)} | "
                          f"样本数：{len(all_inputs):,}")
                
            except Exception as e:
                print(f"处理 {traj_path.name} 失败：{e}")
    
    # 转换为 numpy 数组
    X = np.array(all_inputs)  # (N, 10, 16)
    y = np.array(all_outputs)  # (N, 32)
    
    print(f"\n{'='*80}")
    print("训练数据生成完成!")
    print(f"{'='*80}")
    print(f"输入形状：{X.shape}")
    print(f"输出形状：{y.shape}")
    print(f"总样本数：{len(X):,}")
    print(f"输出路径：{OUTPUT_PATH}")
    
    # 保存
    print("\n保存数据...")
    np.savez(
        OUTPUT_PATH,
        X=X,
        y=y,
        metadata={
            'n_trajectories': len(traj_files),
            'n_samples': len(X),
            'n_variables': N_VARIABLES,
            'n_outputs': N_OUTPUTS,
            'window_size': 10,
            'prediction_horizon': PREDICTION_HORIZON,
            'perturbations': PERTURBATIONS,
            'generated_at': datetime.now().isoformat(),
        }
    )
    
    print(f"✅ 数据已保存到：{OUTPUT_PATH}")
    
    # 验证
    print("\n验证数据...")
    loaded = np.load(OUTPUT_PATH)
    print(f"  X: {loaded['X'].shape}")
    print(f"  y: {loaded['y'].shape}")
    print(f"  输入范围：[{loaded['X'].min():.3f}, {loaded['X'].max():.3f}]")
    print(f"  输出范围：[{loaded['y'].min():.3f}, {loaded['y'].max():.3f}]")
    
    return X, y


if __name__ == '__main__':
    X, y = main()
