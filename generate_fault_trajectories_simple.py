#!/usr/bin/env python3
"""
快速生成故障轨迹数据 (简化版)

方法：复制 normal 轨迹，添加不同类型的扰动和故障标签
用于 Region 3 阈值学习的初步验证

故障类型 (8 种 × 50 条 = 400 条):
1. F1_lighting: 状态加噪声
2. F2_occlusion: 状态部分置零
3. F3_adversarial: 状态加对抗噪声
4. F4_payload: 状态放大 1.5 倍
5. F5_friction: 动作衰减
6. F6_dynamic: 状态截断
7. F7_sensor: 状态加大噪声
8. F8_compound: 多重扰动
"""

import numpy as np, glob
from pathlib import Path
import random

print("="*60)
print("故障轨迹生成 (简化版)")
print("="*60)

SRC_DIR = Path("/mnt/data/aloha_act_500_optimized")
DST_DIR = Path("/mnt/data/aloha_fault_trajectories")
DST_DIR.mkdir(parents=True, exist_ok=True)

# 加载 normal 轨迹
normal_files = sorted(glob.glob(str(SRC_DIR / "traj_*_normal.npz")))
print(f"找到 {len(normal_files)} 条 normal 轨迹")

# 故障类型
fault_types = [
    "F1_lighting", "F2_occlusion", "F3_adversarial", "F4_payload",
    "F5_friction", "F6_dynamic", "F7_sensor", "F8_compound"
]

fault_id = 0
for ft in fault_types:
    print(f"\n生成 {ft}...")
    for i in range(50):
        # 随机选择一个 normal 轨迹
        src_file = random.choice(normal_files)
        data = np.load(src_file)
        
        states = data["states"].copy()
        actions = data["actions"].copy()
        gradients = data["gradients"].copy() if "gradients" in data.files else np.zeros_like(states)
        hook_a = data["hook_a"].copy() if "hook_a" in data.files else None
        hook_b = data["hook_b"].copy() if "hook_b" in data.files else None
        
        # 注入故障 (从第 50 步开始)
        inject_step = 50
        
        if ft == "F1_lighting":
            states[inject_step:] += np.random.randn(*states[inject_step:].shape) * 0.1
        elif ft == "F2_occlusion":
            states[inject_step:, :7] *= 0.5  # 前 7 维减半
        elif ft == "F3_adversarial":
            states[inject_step:] += np.random.randn(*states[inject_step:].shape) * 0.15
        elif ft == "F4_payload":
            states[inject_step:] *= 1.5
        elif ft == "F5_friction":
            actions[inject_step:] *= 0.6
        elif ft == "F6_dynamic":
            states[inject_step:] = np.clip(states[inject_step:], -5, 5)
        elif ft == "F7_sensor":
            states[inject_step:] += np.random.randn(*states[inject_step:].shape) * 0.2
        elif ft == "F8_compound":
            states[inject_step:] = states[inject_step:] * 1.3 + np.random.randn(*states[inject_step:].shape) * 0.15
            actions[inject_step:] *= 0.7
        
        # 保存
        out_data = {
            "fault_type": ft,
            "states": states.astype(np.float32),
            "actions": actions.astype(np.float32),
            "gradients": gradients.astype(np.float32),
        }
        if hook_a is not None:
            out_data["hook_a"] = hook_a
        if hook_b is not None:
            out_data["hook_b"] = hook_b
        
        out_file = DST_DIR / f"fault_{fault_id:04d}_{ft}.npz"
        np.savez_compressed(str(out_file), **out_data)
        fault_id += 1
    
    print(f"  ✅ 完成 50 条")

print()
print("="*60)
print(f"✅ 生成完成！")
print(f"故障轨迹：{fault_id} 条")
print(f"输出目录：{DST_DIR}")
print("="*60)
