#!/usr/bin/env python3
"""
收集 OpenVLA-7B 真实轨迹数据

场景：工作空间中随机分布障碍物
任务：OpenVLA 通过视觉感知，避开障碍物到达目标点
记录：图像 + 状态 + 动作 + 障碍物位置

输出：
- reachability/openvla_trajectories_normal.pkl (正常场景)
- reachability/openvla_trajectories_faults.pkl (故障场景)
"""

import sys
import os
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# 添加路径
PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'agents'))
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))

print("="*80)
print("OpenVLA-7B 真实轨迹数据收集")
print("="*80)

# ============== 配置 ==============

# 场景配置
N_SCENES = 100  # 100 个随机场景
SCENE_TYPES = ['empty', 'sparse', 'dense']  # 场景类型
MAX_STEPS = 500  # 每场景最大步数

# 故障配置
FAULT_TYPES = [
    None,  # 正常
    'lighting_drop',      # 光照突变
    'occlusion',          # 摄像头遮挡
    'adversarial_patch',  # 对抗补丁
    'payload_shift',      # 负载突变
    'joint_friction',     # 关节摩擦
]

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / 'reachability'
OUTPUT_DIR.mkdir(exist_ok=True)

# ============== 加载模型和环境 ==============

print("\n[1/4] 加载 OpenVLA-7B 模型...")
from openvla_agent import OpenVLAAgent

vla = OpenVLAAgent(model_path='/data/models/openvla-7b', device='cuda')
print("✅ OpenVLA-7B 加载成功")

print("\n[2/4] 加载视觉环境...")
from fetch_env_vision import FetchMobileEnvWithVision

env = FetchMobileEnvWithVision(render=False, use_pybullet=False)
print("✅ 环境加载成功")

# ============== 数据收集 ==============

print("\n[3/4] 开始收集轨迹数据...")
print(f"  场景数：{N_SCENES}")
print(f"  场景类型：{SCENE_TYPES}")
print(f"  故障类型：{len(FAULT_TYPES)} 种")

all_trajectories = {
    'normal': [],  # 正常轨迹
    'faults': [],  # 故障轨迹
}

for scene_idx in range(N_SCENES):
    # 随机选择场景类型
    scene_type = np.random.choice(SCENE_TYPES)
    
    # 随机种子 (确保可复现)
    seed = scene_idx * 100
    
    for fault_idx, fault_type in enumerate(FAULT_TYPES):
        # 重置环境
        obs = env.reset(seed=seed, scene_type=scene_type)
        
        # 轨迹数据
        trajectory = {
            'scene_id': f'{scene_idx:03d}_{scene_type}',
            'fault_type': fault_type,
            'seed': seed,
            'images': [],       # OpenVLA 看到的 RGB 图像
            'states': [],       # 完整状态
            'actions': [],      # OpenVLA 输出的动作
            'obstacles': [],    # 障碍物位置
            'success': False,   # 是否成功到达
            'collision': False, # 是否碰撞
            'steps': 0,         # 实际步数
        }
        
        # 故障注入配置
        fault_info = None
        if fault_type is not None:
            fault_info = {
                'active': True,
                'type': fault_type,
                'params': {
                    'intensity': 0.7 if fault_type == 'lighting_drop' else None,
                    'mask_ratio': 0.3 if fault_type == 'occlusion' else None,
                    'noise_std': 50 if fault_type == 'depth_noise' else None,
                }
            }
            # 故障注入时间 (随机 100-200 步后)
            fault_start_step = np.random.randint(100, 200)
            fault_duration = np.random.randint(50, 150)
        
        # 运行轨迹
        instruction = "navigate to the green goal while avoiding obstacles"
        
        for step in range(MAX_STEPS):
            # 获取当前观测
            image = obs['image']  # (224, 224, 3) RGB
            state = {
                'base': obs['base'].copy(),      # [x, y, θ, v, ω]
                'arm_q': obs['arm_q'].copy(),    # [q1-q7]
                'arm_dq': obs['arm_dq'].copy(),  # [dq1-dq7]
                'obstacles': [o.copy() for o in obs['obstacles']],
            }
            
            # 故障注入时机
            if fault_info and step >= fault_start_step and step < fault_start_step + fault_duration:
                current_fault = fault_info
            else:
                current_fault = None
            
            # OpenVLA 推理
            action = vla.get_action(image, instruction)
            
            # 记录数据
            trajectory['images'].append(image.copy())
            trajectory['states'].append(state)
            trajectory['actions'].append(action.copy())
            
            # 环境步进
            obs, reward, done, info = env.step(action, fault_info=current_fault)
            
            # 检查终止
            if info.get('collision', False):
                trajectory['collision'] = True
                trajectory['steps'] = step + 1
                break
            
            if info.get('reached_goal', False):
                trajectory['success'] = True
                trajectory['steps'] = step + 1
                break
            
            if done:
                trajectory['steps'] = step + 1
                break
        
        # 保存轨迹
        if fault_type is None:
            all_trajectories['normal'].append(trajectory)
        else:
            all_trajectories['faults'].append(trajectory)
        
        # 进度汇报
        if (scene_idx + 1) % 10 == 0:
            normal_count = len(all_trajectories['normal'])
            fault_count = len(all_trajectories['faults'])
            success_rate = sum(1 for t in all_trajectories['normal'] if t['success']) / max(1, normal_count)
            print(f"  进度：{scene_idx + 1}/{N_SCENES} 场景 | "
                  f"正常：{normal_count} | 故障：{fault_count} | "
                  f"成功率：{success_rate:.1%}")

# ============== 统计分析 ==============

print("\n[4/4] 统计分析...")

# 正常轨迹统计
normal_count = len(all_trajectories['normal'])
normal_success = sum(1 for t in all_trajectories['normal'] if t['success'])
normal_collision = sum(1 for t in all_trajectories['normal'] if t['collision'])
normal_avg_steps = np.mean([t['steps'] for t in all_trajectories['normal']])

print(f"\n正常轨迹:")
print(f"  总数：{normal_count}")
print(f"  成功：{normal_success} ({normal_success/normal_count:.1%})")
print(f"  碰撞：{normal_collision} ({normal_collision/normal_count:.1%})")
print(f"  平均步数：{normal_avg_steps:.1f}")

# 故障轨迹统计
fault_count = len(all_trajectories['faults'])
fault_success = sum(1 for t in all_trajectories['faults'] if t['success'])
fault_collision = sum(1 for t in all_trajectories['faults'] if t['collision'])

print(f"\n故障轨迹:")
print(f"  总数：{fault_count}")
print(f"  成功：{fault_success} ({fault_success/fault_count:.1%})")
print(f"  碰撞：{fault_collision} ({fault_collision/fault_count:.1%})")

# 按故障类型统计
print(f"\n按故障类型:")
for fault_type in FAULT_TYPES[1:]:
    type_trajs = [t for t in all_trajectories['faults'] if t['fault_type'] == fault_type]
    type_success = sum(1 for t in type_trajs if t['success'])
    print(f"  {fault_type}: {type_success}/{len(type_trajs)} ({type_success/len(type_trajs):.1%})")

# ============== 保存数据 ==============

print(f"\n保存数据...")

# 保存完整轨迹
output_file_normal = OUTPUT_DIR / 'openvla_trajectories_normal.pkl'
output_file_faults = OUTPUT_DIR / 'openvla_trajectories_faults.pkl'

with open(output_file_normal, 'wb') as f:
    pickle.dump(all_trajectories['normal'], f)
print(f"✅ 正常轨迹：{output_file_normal} ({len(all_trajectories['normal'])} 条)")

with open(output_file_faults, 'wb') as f:
    pickle.dump(all_trajectories['faults'], f)
print(f"✅ 故障轨迹：{output_file_faults} ({len(all_trajectories['faults'])} 条)")

# 保存元数据
metadata = {
    'collection_time': datetime.now().isoformat(),
    'n_scenes': N_SCENES,
    'scene_types': SCENE_TYPES,
    'fault_types': FAULT_TYPES,
    'max_steps': MAX_STEPS,
    'normal_count': normal_count,
    'fault_count': fault_count,
    'normal_success_rate': normal_success / normal_count,
    'fault_success_rate': fault_success / fault_count,
}

with open(OUTPUT_DIR / 'trajectory_metadata.json', 'w') as f:
    import json
    json.dump(metadata, f, indent=2)
print(f"✅ 元数据：{OUTPUT_DIR / 'trajectory_metadata.json'}")

# ============== 数据格式说明 ==============

print("\n" + "="*80)
print("数据格式说明:")
print("="*80)
print("""
每条轨迹包含:
{
    'scene_id': '001_sparse',          # 场景 ID
    'fault_type': 'lighting_drop',      # 故障类型 (None=正常)
    'seed': 100,                        # 随机种子
    'images': [                         # OpenVLA 看到的 RGB 图像
        (224, 224, 3) uint8,            # 每帧图像
        ...
    ],
    'states': [                         # 状态序列
        {
            'base': [x, y, θ, v, ω],    # 底盘状态 (5 维)
            'arm_q': [q1...q7],         # 关节角度 (7 维)
            'arm_dq': [dq1...dq7],      # 关节速度 (7 维)
            'obstacles': [...]          # 障碍物列表
        },
        ...
    ],
    'actions': [                        # OpenVLA 输出动作
        {'v': 0.5, 'omega': 0.1, 'tau': [...]},
        ...
    ],
    'obstacles': [...],                 # 障碍物配置
    'success': True,                    # 是否成功
    'collision': False,                 # 是否碰撞
    'steps': 234,                       # 实际步数
}

用途:
- Region 2: 从 states + actions 生成可达集训练数据
- Region 3: 从 images + states 学习激活掩码和阈值
- 试验分析: 成功率、碰撞率、轨迹可视化
""")

print("="*80)
print("✅ 数据收集完成!")
print("="*80)
