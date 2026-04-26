#!/usr/bin/env python3
"""
S4 场景修复验证测试
运行 S4 场景 30 次，验证 ZMP 稳定性修复效果
"""

import sys
sys.path.insert(0, '/home/vipuser/Embodied-RTA')
sys.path.insert(0, '/home/vipuser/Embodied-RTA/envs')

from fetch_env_extended import FetchMobileEnv
import numpy as np

print("="*60)
print("S4 场景修复验证测试")
print("="*60)

# 测试参数
NUM_RUNS = 30
scenario = 's4_payload_shift'

env = FetchMobileEnv('/home/vipuser/Embodied-RTA/configs/fetch_params.yaml')

successes = 0
zmp_failures = 0
collisions = 0

for run in range(NUM_RUNS):
    np.random.seed(run)
    obs = env.reset(scenario=scenario, seed=run)
    
    success = True
    zmp_failure = False
    collision = False
    
    for step in range(500):  # 10 秒测试
        # 简单动作：向前移动
        action = {
            'v': 0.3,  # 中等速度
            'ω': 0.0,
            'τ': np.zeros(7)
        }
        
        obs, reward, done, info = env.step(action)
        
        if info.get('collision'):
            collision = True
            success = False
            break
        
        if not info.get('zmp_stable', True):
            zmp_failure = True
            success = False
            break
        
        if done:
            if info.get('violations'):
                success = False
            break
    
    if success:
        successes += 1
    if zmp_failure:
        zmp_failures += 1
    if collision:
        collisions += 1
    
    if (run + 1) % 10 == 0:
        print(f"  进度：{run+1}/{NUM_RUNS}, 当前成功率：{successes/(run+1)*100:.1f}%")

print("\n" + "="*60)
print("测试结果")
print("="*60)
print(f"总试验次数：{NUM_RUNS}")
print(f"成功次数：{successes}")
print(f"成功率：{successes/NUM_RUNS*100:.1f}%")
print(f"ZMP 失败次数：{zmp_failures}")
print(f"碰撞次数：{collisions}")

if successes / NUM_RUNS > 0.5:
    print("\n✅ S4 场景修复成功！成功率 > 50%")
else:
    print("\n⚠️ S4 场景仍需调整，成功率 < 50%")
