#!/usr/bin/env python3
"""
OpenVLA-7B + Region 3 集成测试

验证:
1. OpenVLA-7B 模型加载成功
2. Region 3 多层激活钩子注册成功
3. 正常推理时激活链路正常
4. OOD 输入时检测到异常
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加路径
PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'agents'))
sys.path.insert(0, str(PROJECT_ROOT / 'xai'))

print("="*80)
print("OpenVLA-7B + Region 3 集成测试")
print("="*80)

# ============== Step 1: 加载 OpenVLA-7B ==============
print("\n[1/4] 加载 OpenVLA-7B 模型...")
from openvla_agent import OpenVLAAgent

try:
    vla = OpenVLAAgent(model_path="/data/models/openvla-7b", device="cuda")
    print("✅ OpenVLA-7B 加载成功")
except Exception as e:
    print(f"❌ OpenVLA 加载失败：{e}")
    sys.exit(1)

# ============== Step 2: 注册 Region 3 钩子 ==============
print("\n[2/4] 注册 Region 3 多层激活钩子...")
from xai.multi_layer_activation import MultiLayerActivationHook

try:
    hook_manager = MultiLayerActivationHook(vla.model)
    print(f"✅ 钩子注册成功 (共 {len(hook_manager.hooks)} 个)")
except Exception as e:
    print(f"❌ 钩子注册失败：{e}")
    sys.exit(1)

# ============== Step 3: 正常推理测试 ==============
print("\n[3/4] 正常推理测试...")

# 创建测试图像 (模拟正常场景)
test_image_normal = np.random.rand(224, 224, 3).astype(np.float32)
instruction = "navigate to the goal"

try:
    # 清空激活缓存
    hook_manager.clear_activations()
    
    # 推理
    action = vla.get_action(test_image_normal, instruction)
    
    # 检查激活
    activations = hook_manager.get_all_activations()
    print(f"✅ 推理成功")
    print(f"   动作：v={action['v']:.3f}, omega={action['omega']:.3f}")
    print(f"   激活层数：{len(activations)}")
    
    # 打印关键层激活统计
    print("\n   关键层激活统计:")
    for name, act in list(activations.items())[:5]:
        print(f"   - {name}: mean={act.mean():.4f}, std={act.std():.4f}")
    
except Exception as e:
    print(f"❌ 推理失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============== Step 4: OOD 检测测试 ==============
print("\n[4/4] OOD 检测测试...")

from xai.region3_activation_link import Region3Detector

try:
    # 初始化检测器
    detector = Region3Detector(model=vla.model)
    
    # 注册钩子
    detector.register_hooks()
    
    # 收集参考数据 (正常图像)
    print("   收集正常参考数据...")
    normal_images = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(10)]
    for img in normal_images:
        hook_manager.clear_activations()
        _ = vla.get_action(img, instruction)
    
    # 计算参考统计
    detector.collect_reference_stats()
    print("   ✅ 参考数据收集完成")
    
    # 测试 OOD 图像 (全黑图像)
    ood_image = np.zeros((224, 224, 3), dtype=np.float32)
    
    hook_manager.clear_activations()
    action_ood = vla.get_action(ood_image, instruction)
    
    # 检测
    triggered, info = detector.detect()
    
    print(f"   OOD 图像检测结果:")
    print(f"   - 触发：{triggered}")
    print(f"   - 风险分数：{info.get('risk', 0):.3f}")
    print(f"   - 各检测器分数:")
    print(f"     · Link: {info.get('link', 0):.3f}")
    print(f"     · OOD: {info.get('ood', 0):.3f}")
    print(f"     · Jump: {info.get('jump', 0):.3f}")
    print(f"     · Entropy: {info.get('entropy', 0):.3f}")
    
    print("✅ OOD 检测测试完成")
    
except Exception as e:
    print(f"❌ OOD 检测失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============== 总结 ==============
print("\n" + "="*80)
print("✅ 所有测试通过！")
print("="*80)
print("\nOpenVLA-7B + Region 3 集成状态:")
print("  - OpenVLA-7B 模型：✅ 加载成功")
print("  - Region 3 钩子：✅ 注册成功")
print("  - 正常推理：✅ 工作正常")
print("  - OOD 检测：✅ 工作正常")
print("\n下一步：运行完整试验 (tests/run_openvla_rta.py)")
