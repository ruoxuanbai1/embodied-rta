#!/usr/bin/env python3
"""
S4 场景 ZMP 调试
分析 ZMP 计算的具体数值
"""

import sys
sys.path.insert(0, '/home/vipuser/Embodied-RTA')
sys.path.insert(0, '/home/vipuser/Embodied-RTA/envs')

from fetch_env_extended import FetchMobileEnv
import numpy as np

print("="*70)
print("S4 场景 ZMP 调试分析")
print("="*70)

env = FetchMobileEnv('/home/vipuser/Embodied-RTA/configs/fetch_params.yaml')
obs = env.reset(scenario='s4_payload_shift', seed=42)

print(f"\n环境参数:")
print(f"  底盘长度：{env.length} m")
print(f"  底盘质量：{env.mass} kg")
print(f"  初始重心高度：{env.com_height} m")
print(f"  ZMP 安全裕度：{env.zmp_safety_margin} m")
print(f"  支撑边界：{(env.length/2) - env.zmp_safety_margin:.3f} m")

print(f"\n场景配置:")
scenario_config = env.scenario_library['s4_payload_shift']
print(f"  负载质量：{scenario_config.params['payload_mass']} kg")
print(f"  重心偏移：{scenario_config.params['com_shift']}")
print(f"  注入时间：{scenario_config.injection_time} s")

# 运行并记录 ZMP 数据
print(f"\n运行测试并记录 ZMP 数据:")
print("-"*70)

zmp_data = []

for step in range(400):  # 8 秒
    t = step * env.dt
    
    # 动作
    action = {
        'v': 0.3,
        'ω': 0.0,
        'τ': np.zeros(7)
    }
    
    obs, reward, done, info = env.step(action)
    
    # 计算 ZMP
    base = env.state['base']
    com = env.state['com_position']
    ax = base[3] * env.a_max
    zmp_x = com[0] - (com[2] / env.gravity) * ax
    support_margin = (env.length / 2) - env.zmp_safety_margin
    
    zmp_data.append({
        'time': t,
        'base_x': base[0],
        'base_v': base[3],
        'ax': ax,
        'com_x': com[0],
        'com_z': com[2],
        'zmp_x': zmp_x,
        'support_margin': support_margin,
        'zmp_stable': info.get('zmp_stable', True),
        'payload': env.state['payload_mass'],
        'fault_active': env.fault_active
    })
    
    if step % 50 == 0 or (step > 0 and env.fault_active and step < 100):
        print(f"t={t:.2f}s: base_v={base[3]:.2f} m/s, ax={ax:.2f} m/s², "
              f"com_z={com[2]:.2f} m, zmp_x={zmp_x:.3f} m, "
              f"margin={support_margin:.3f} m, "
              f"stable={info.get('zmp_stable', True)}, "
              f"payload={env.state['payload_mass']:.1f} kg, "
              f"fault={env.fault_active}")
    
    if done:
        print(f"\n试验在 t={t:.2f}s 终止")
        print(f"  碰撞：{info.get('collision', False)}")
        print(f"  ZMP 失稳：{not info.get('zmp_stable', True)}")
        print(f"  约束违反：{info.get('violations', False)}")
        break

# 分析
print(f"\n" + "="*70)
print("ZMP 数据分析")
print("="*70)

# 找到故障注入前后的 ZMP 统计
before_fault = [d for d in zmp_data if not d['fault_active']]
after_fault = [d for d in zmp_data if d['fault_active']]

if before_fault:
    print(f"\n故障注入前 ({len(before_fault)} 步):")
    print(f"  ZMP_x 范围：[{min(d['zmp_x'] for d in before_fault):.3f}, {max(d['zmp_x'] for d in before_fault):.3f}] m")
    print(f"  平均 ZMP_x: {np.mean([d['zmp_x'] for d in before_fault]):.3f} m")

if after_fault:
    print(f"\n故障注入后 ({len(after_fault)} 步):")
    print(f"  ZMP_x 范围：[{min(d['zmp_x'] for d in after_fault):.3f}, {max(d['zmp_x'] for d in after_fault):.3f}] m")
    print(f"  平均 ZMP_x: {np.mean([d['zmp_x'] for d in after_fault]):.3f} m")

print(f"\n支撑边界：±{support_margin:.3f} m")

# 诊断
print(f"\n" + "="*70)
print("诊断")
print("="*70)

if after_fault:
    max_zmp = max(d['zmp_x'] for d in after_fault)
    if max_zmp > support_margin:
        print(f"⚠️  ZMP 超出边界：{max_zmp:.3f} > {support_margin:.3f}")
        print(f"   超出量：{max_zmp - support_margin:.3f} m ({(max_zmp/support_margin - 1)*100:.1f}%)")
        
        # 分析原因
        max_com_z = max(d['com_z'] for d in after_fault)
        max_ax = max(abs(d['ax']) for d in after_fault)
        print(f"\n可能原因:")
        print(f"   1. 重心高度：{max_com_z:.2f} m (过高会增加 ZMP 偏移)")
        print(f"   2. 最大加速度：{max_ax:.2f} m/s²")
        print(f"   3. 负载质量：{env.state['payload_mass']:.1f} kg")
        
        # 建议
        print(f"\n建议调整:")
        print(f"   - 降低负载质量到 {scenario_config.params['payload_mass'] * 0.5:.1f} kg")
        print(f"   - 或限制最大加速度到 {env.a_max * 0.5:.1f} m/s²")
        print(f"   - 或进一步放宽 ZMP 安全裕度到 0.01 m")
