#!/bin/bash
# OpenVLA-7B + Region 3 集成测试 (vipuser conda 环境)

echo "=========================================="
echo "  OpenVLA-7B + Region 3 集成测试"
echo "  Python: /home/vipuser/miniconda3/bin/python"
echo "=========================================="

cd /home/vipuser/Embodied-RTA
PYTHON=/home/vipuser/miniconda3/bin/python

# Step 1: 测试 OpenVLA 加载
echo ""
echo "[1/3] 测试 OpenVLA-7B 加载..."
$PYTHON -c "
from agents.openvla_agent import OpenVLAAgent
print('Loading OpenVLA-7B...')
vla = OpenVLAAgent(model_path='/data/models/openvla-7b', device='cuda')
print('✅ OpenVLA-7B 加载成功')
"

if [ $? -ne 0 ]; then
    echo "❌ OpenVLA 加载失败"
    exit 1
fi

# Step 2: 测试 Region 3 钩子注册
echo ""
echo "[2/3] 测试 Region 3 钩子注册..."
$PYTHON -c "
import sys
sys.path.insert(0, '/home/vipuser/Embodied-RTA')
from xai.multi_layer_activation import MultiLayerActivationHook
from agents.openvla_agent import OpenVLAAgent

print('Loading OpenVLA...')
vla = OpenVLAAgent(model_path='/data/models/openvla-7b', device='cuda')

print('Registering hooks...')
hook_manager = MultiLayerActivationHook(vla.model)
print(f'✅ 钩子注册成功：{len(hook_manager.hooks)} 个钩子')
"

if [ $? -ne 0 ]; then
    echo "❌ Region 3 钩子注册失败"
    exit 1
fi

# Step 3: 测试完整推理 + 激活捕获
echo ""
echo "[3/3] 测试推理 + 激活捕获..."
$PYTHON -c "
import sys
import numpy as np
sys.path.insert(0, '/home/vipuser/Embodied-RTA')
from xai.multi_layer_activation import MultiLayerActivationHook
from agents.openvla_agent import OpenVLAAgent

print('Loading OpenVLA...')
vla = OpenVLAAgent(model_path='/data/models/openvla-7b', device='cuda')

print('Registering hooks...')
hook_manager = MultiLayerActivationHook(vla.model)

print('Running inference...')
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
instruction = 'navigate to the goal'

hook_manager.clear_activations()
action = vla.get_action(test_image, instruction)

activations = hook_manager.get_all_activations()
print(f'✅ 推理成功')
print(f'   动作：v={action[\"v\"]:.4f}, omega={action[\"omega\"]:.4f}')
print(f'   激活层数：{len(activations)}')

# 打印前 5 层激活统计
print('   关键层激活:')
for i, (name, act) in enumerate(list(activations.items())[:5]):
    print(f'   - {name}: mean={act.mean():.4f}, std={act.std():.4f}')
"

if [ $? -ne 0 ]; then
    echo "❌ 推理测试失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 所有测试通过！"
echo "=========================================="
echo ""
echo "OpenVLA-7B + Region 3 集成状态:"
echo "  ✅ OpenVLA-7B 模型加载成功"
echo "  ✅ Region 3 多层激活钩子注册成功"
echo "  ✅ 推理 + 激活捕获工作正常"
echo ""
echo "下一步：运行完整试验"
echo "  cd /home/vipuser/Embodied-RTA"
echo "  $PYTHON tests/run_openvla_rta.py"
