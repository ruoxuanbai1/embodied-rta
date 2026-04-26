#!/usr/bin/env python3
"""测试 Region 3 检测器"""

import torch
import sys
sys.path.insert(0, '/root/Embodied-RTA')

from rta_region3 import Region3Detector

print('=== 测试 Region 3 五模块检测器 ===')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = Region3Detector(state_dim=14, action_dim=14, hidden_dim=512, device=device)

# 模拟输入
B = 2
state = torch.randn(B, 14, device=device)
action_logits = torch.randn(B, 14, device=device)
hook_a = torch.randn(B, 512, device=device)
hook_b = torch.randn(B, 512, device=device)
visual_features = torch.randn(B, 512, device=device)

# 测试 1: 不带 SHAP 模块 (action=None)
print('\n[测试 1] 不带 SHAP 模块')
result = detector(
    state=state,
    action_logits=action_logits,
    hook_a=hook_a,
    hook_b=hook_b,
    visual_features=visual_features,
    action=None,
    action_type=0,
)

print(f"风险分数：{result['risk_score'].mean().item():.3f}")
print(f"风险等级：{result['risk_level']}")
print(f"模块分数:")
for name, score in result['module_scores'].items():
    if isinstance(score[0], list):
        print(f"  {name}: {score[0][0]:.3f}")
    else:
        print(f"  {name}: {score[0]:.3f}")

# 测试 2: 数据收集模式
print('\n[测试 2] 数据收集模式')
result_collect = detector(
    state=state,
    action_logits=action_logits,
    hook_a=hook_a,
    hook_b=hook_b,
    visual_features=visual_features,
    action=None,
    collect_mode=True,
)
print(f"统计数据已更新")

print('\n✅ Region 3 检测器测试通过!')
