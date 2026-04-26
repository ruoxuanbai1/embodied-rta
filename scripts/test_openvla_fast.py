#!/usr/bin/env python3
"""OpenVLA-7B 快速推理测试"""

import sys
sys.path.insert(0, '/home/vipuser/Embodied-RTA')

print('Loading OpenVLA-7B...')
from agents.openvla_agent import OpenVLAAgent

vla = OpenVLAAgent('/data/models/openvla-7b', 'cuda')
print('✅ Model loaded!')

import numpy as np
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
print('Running inference...')

action = vla.get_action(img, 'navigate to the goal')
print(f'✅ Success! v={action["v"]:.4f}, omega={action["omega"]:.4f}')
