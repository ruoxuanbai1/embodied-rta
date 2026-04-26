#!/usr/bin/env python3
"""
OpenVLA-7B 推理测试 (修复版)
"""

import sys
import numpy as np
sys.path.insert(0, '/home/vipuser/Embodied-RTA')

print("="*60)
print("OpenVLA-7B 推理测试")
print("="*60)

print("\n加载 OpenVLA-7B...")
from agents.openvla_agent import OpenVLAAgent

vla = OpenVLAAgent(model_path='/data/models/openvla-7b', device='cuda')
print("✅ 模型加载成功")

print("\n创建测试图像...")
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
instruction = 'navigate to the goal'

print("执行推理...")
try:
    action = vla.get_action(test_image, instruction)
    print(f'✅ 推理成功!')
    print(f'   动作：v={action["v"]:.4f}, omega={action["omega"]:.4f}')
    print(f'   扭矩：{action["tau"].shape}')
except Exception as e:
    print(f'❌ 推理失败：{e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n连续推理 3 次测试稳定性...")
for i in range(3):
    action = vla.get_action(test_image, instruction)
    print(f'  第{i+1}次：v={action["v"]:.4f}, omega={action["omega"]:.4f}')

print("\n" + "="*60)
print("✅ 所有测试通过！")
print("="*60)
