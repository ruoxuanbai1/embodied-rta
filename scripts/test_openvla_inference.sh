#!/bin/bash
# OpenVLA-7B 推理测试 (修复版)

echo "=========================================="
echo "  OpenVLA-7B 推理测试"
echo "  transformers==4.40.1"
echo "=========================================="

cd /home/vipuser/Embodied-RTA
PYTHON=/home/vipuser/miniconda3/bin/python

$PYTHON -c "
import sys
import numpy as np
sys.path.insert(0, '/home/vipuser/Embodied-RTA')

print('加载 OpenVLA-7B...')
from agents.openvla_agent import OpenVLAAgent

vla = OpenVLAAgent(model_path='/data/models/openvla-7b', device='cuda')
print('✅ 模型加载成功')

print('创建测试图像...')
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
instruction = 'navigate to the goal'

print('执行推理...')
action = vla.get_action(test_image, instruction)

print(f'✅ 推理成功!')
print(f'   动作：v={action[\"v\"]:.4f}, omega={action[\"omega\"]:.4f}')
print(f'   扭矩：{action[\"tau\"].shape}')

# 多次推理测试稳定性
print('连续推理 3 次...')
for i in range(3):
    action = vla.get_action(test_image, instruction)
    print(f'  第{i+1}次：v={action[\"v\"]:.4f}, omega={action[\"omega\"]:.4f}')

print('')
print('✅ 所有测试通过！')
"
