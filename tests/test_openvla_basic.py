#!/usr/bin/env python3
"""
OpenVLA-7B 基础功能测试

验证：
1. 模型能否加载
2. 能否接收图像 + 语言指令
3. 能否输出动作
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("OpenVLA-7B 基础功能测试")
print("="*60)

# 1. 导入并加载模型
print("\n[1/3] 加载 OpenVLA-7B 模型...")
from agents.openvla_agent import OpenVLAAgent

vla = OpenVLAAgent(model_path="/data/models/openvla-7b", device="cuda")
print("✅ 模型加载成功")

# 2. 创建测试图像 (模拟相机输入)
print("\n[2/3] 创建测试图像...")
# 生成随机 RGB 图像 (224x224)
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
print(f"  图像形状：{test_image.shape}")
print(f"  图像类型：{test_image.dtype}")

# 3. 测试推理
print("\n[3/3] 测试 OpenVLA 推理...")
instruction = "避开障碍物到达终点"
print(f"  语言指令：{instruction}")

action = vla.get_action(test_image, instruction)

print(f"\n✅ 推理成功!")
print(f"  输出动作:")
print(f"    v (线速度): {action['v']:.4f} m/s")
print(f"    ω (角速度): {action['omega']:.4f} rad/s")
print(f"    τ (扭矩): {action['tau'].shape}")

# 4. 多次测试稳定性
print("\n[额外测试] 连续推理 5 次...")
for i in range(5):
    action = vla.get_action(test_image, instruction)
    print(f"  第{i+1}次：v={action['v']:.4f}, ω={action['omega']:.4f}")

print("\n" + "="*60)
print("✅ 所有测试通过！OpenVLA-7B 可以正常使用")
print("="*60)
