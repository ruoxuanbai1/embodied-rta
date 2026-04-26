#!/usr/bin/env python3
import sys
sys.argv = ['debug_hook_call.py', '--ckpt_dir', './ckpts/my_transfer_cube_model', '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']

import torch
import numpy as np
from einops import rearrange
from constants import SIM_TASK_CONFIGS
from utils import sample_box_pose
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env

print("="*70)
print("Debug Hook Call")
print("="*70)

# 加载模型
print("\n加载模型...")
policy_config = {
    'lr': 1e-5, 'num_queries': 100, 'kl_weight': 10,
    'hidden_dim': 512, 'dim_feedforward': 3200, 'lr_backbone': 1e-5,
    'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7,
    'nheads': 8, 'camera_names': ['top'],
}
policy = ACTPolicy(policy_config)
ckpt_path = '/root/act/ckpts/my_transfer_cube_model/policy_best.ckpt'
policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
policy.cuda()
policy.eval()
print("✅ 模型加载完成")

# 检查模型结构
print("\n" + "="*70)
print("模型结构检查")
print("="*70)
model = policy.model
print(f"model type: {type(model)}")
print(f"has encoder: {hasattr(model, 'encoder')}, encoder={model.encoder}")
print(f"has decoder: {hasattr(model, 'decoder')}, decoder={model.decoder}")

if hasattr(model, 'encoder') and model.encoder is not None:
    print(f"\nEncoder layers: {len(model.encoder.layers)}")
    for i, layer in enumerate(model.encoder.layers):
        print(f"  Layer {i}: {type(layer)}")
        print(f"    has linear2: {hasattr(layer, 'linear2')}")
        if hasattr(layer, 'linear2'):
            print(f"    linear2: {layer.linear2}")

# 注册 hook（带调试）
print("\n" + "="*70)
print("注册 hooks (带调试)")
print("="*70)

class DebugHook:
    def __init__(self, name):
        self.name = name
        self.call_count = 0
        self.last_output = None
        self.handle = None
    
    def hook_fn(self, module, input, output):
        self.call_count += 1
        self.last_output = output.detach().cpu()
        print(f"  [HOOK {self.name}] Called! output shape={output.shape}")
    
    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)

hooks = {}
for i, layer in enumerate(model.encoder.layers):
    if hasattr(layer, 'linear2'):
        hooks[f"layer{i}_ffn"] = DebugHook(f"layer{i}_ffn")
        hooks[f"layer{i}_ffn"].register(layer.linear2)
        print(f"✅ Registered: layer{i}_ffn -> {layer.linear2}")

# 创建环境并运行
print("\n" + "="*70)
print("运行推理")
print("="*70)
env = make_sim_env('sim_transfer_cube_scripted')
BOX_POSE[0] = sample_box_pose()
ts = env.reset()

camera_names = ['top']
qpos = ts.observation['qpos'].copy()
qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
curr_images = [rearrange(ts.observation['images'][cam_name], 'h w c -> c h w') for cam_name in camera_names]
curr_image = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().cuda().unsqueeze(0)

print(f"\n输入：qpos={qpos_tensor.shape}, image={curr_image.shape}")
print("运行 policy 前向传播...")

all_actions = policy(qpos_tensor, curr_image)
print(f"输出：{all_actions.shape}")

# 检查 hook 调用
print("\n" + "="*70)
print("Hook 调用统计")
print("="*70)
for key, hook in hooks.items():
    print(f"{key}: 调用次数={hook.call_count}")
    if hook.last_output is not None:
        out = hook.last_output
        nonzero = (out!=0).sum()/out.numel()*100
        print(f"  ✅ shape={out.shape}, 非零={nonzero:.1f}%, mean={out.mean():.6f}")
    else:
        print(f"  ❌ 无输出")
