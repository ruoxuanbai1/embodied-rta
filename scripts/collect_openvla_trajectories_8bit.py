#!/usr/bin/env python3
"""
OpenVLA-7B 轨迹收集 (8bit 量化版)

场景：工作空间中随机分布障碍物
任务：OpenVLA 通过视觉感知，避开障碍物到达目标点
记录：图像 + 状态 + 动作
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
print("OpenVLA-7B (8bit) 真实轨迹数据收集")
print("="*80)

# ============== 配置 ==============

N_SCENES = 50       # 50 个随机场景 (先小规模测试)
SCENE_TYPES = ['empty', 'sparse', 'dense']
MAX_STEPS = 300     # 每场景最大步数
FAULT_TYPES = [None, 'lighting_drop', 'occlusion']  # 简化故障集

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / 'reachability'
OUTPUT_DIR.mkdir(exist_ok=True)

# ============== 加载模型和环境 ==============

print("\n[1/3] 加载 OpenVLA-7B (8bit)...")
from openvla_agent_8bit import OpenVLAAgent

vla = OpenVLAAgent(model_path='/data/models/openvla-7b', device='cuda')
print("✅ OpenVLA-7B 加载成功")

print("\n[2/3] 加载视觉环境...")
from fetch_env_vision import FetchMobileEnvWithVision

env = FetchMobileEnvWithVision(render=False, use_pybullet=False)
print("✅ 环境加载成功")

# ============== 数据收集 ==============

print("\n[3/3] 开始收集轨迹数据...")

all_trajectories = {'normal': [], 'faults': []}

for scene_idx in range(N_SCENES):
    scene_type = np.random.choice(SCENE_TYPES)
    seed = scene_idx * 100
    
    for fault_idx, fault_type in enumerate(FAULT_TYPES):
        obs = env.reset(seed=seed, scene_type=scene_type)
        
        trajectory = {
            'scene_id': f'{scene_idx:03d}_{scene_type}',
            'fault_type': fault_type,
            'seed': seed,
            'images': [],
            'states': [],
            'actions': [],
            'success': False,
            'collision': False,
            'steps': 0,
        }
        
        fault_info = None
        if fault_type is not None:
            fault_info = {
                'active': True,
                'type': fault_type,
                'params': {'intensity': 0.7, 'mask_ratio': 0.3},
            }
            fault_start = np.random.randint(50, 100)
            fault_duration = np.random.randint(30, 80)
        
        instruction = "navigate to the green goal"
        
        for step in range(MAX_STEPS):
            image = obs['image']
            state = {
                'base': obs['base'].copy(),
                'arm_q': obs['arm_q'].copy(),
                'arm_dq': obs['arm_dq'].copy(),
                'obstacles': [o.copy() for o in obs['obstacles']],
            }
            
            # 故障注入
            current_fault = None
            if fault_info and step >= fault_start and step < fault_start + fault_duration:
                current_fault = fault_info
            
            # OpenVLA 推理
            action = vla.get_action(image, instruction)
            
            # 记录
            trajectory['images'].append(image.copy())
            trajectory['states'].append(state)
            trajectory['actions'].append(action.copy())
            
            # 步进
            obs, reward, done, info = env.step(action, fault_info=current_fault)
            
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
        
        # 保存
        if fault_type is None:
            all_trajectories['normal'].append(trajectory)
        else:
            all_trajectories['faults'].append(trajectory)
        
        # 进度
        if (scene_idx + 1) % 5 == 0:
            n_normal = len(all_trajectories['normal'])
            n_faults = len(all_trajectories['faults'])
            success_rate = sum(1 for t in all_trajectories['normal'] if t['success']) / max(1, n_normal)
            print(f"  进度：{scene_idx+1}/{N_SCENES} | 正常:{n_normal} | 故障:{n_faults} | 成功率:{success_rate:.1%}")

# ============== 统计 ==============

print("\n统计...")
normal_count = len(all_trajectories['normal'])
normal_success = sum(1 for t in all_trajectories['normal'] if t['success'])
fault_count = len(all_trajectories['faults'])
fault_success = sum(1 for t in all_trajectories['faults'] if t['success'])

print(f"  正常轨迹：{normal_count} (成功:{normal_success}, {normal_success/normal_count:.1%})")
print(f"  故障轨迹：{fault_count} (成功:{fault_success}, {fault_success/fault_count:.1%})")

# ============== 保存 ==============

print("\n保存数据...")

with open(OUTPUT_DIR / 'openvla_trajectories_normal.pkl', 'wb') as f:
    pickle.dump(all_trajectories['normal'], f)
print(f"✅ 正常轨迹：{OUTPUT_DIR / 'openvla_trajectories_normal.pkl'}")

with open(OUTPUT_DIR / 'openvla_trajectories_faults.pkl', 'wb') as f:
    pickle.dump(all_trajectories['faults'], f)
print(f"✅ 故障轨迹：{OUTPUT_DIR / 'openvla_trajectories_faults.pkl'}")

metadata = {
    'collection_time': datetime.now().isoformat(),
    'n_scenes': N_SCENES,
    'scene_types': SCENE_TYPES,
    'fault_types': FAULT_TYPES,
    'max_steps': MAX_STEPS,
    'normal_count': normal_count,
    'fault_count': fault_count,
    'normal_success_rate': normal_success / normal_count,
}

import json
with open(OUTPUT_DIR / 'trajectory_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ 元数据：{OUTPUT_DIR / 'trajectory_metadata.json'}")

print("\n" + "="*80)
print("✅ 数据收集完成!")
print("="*80)
