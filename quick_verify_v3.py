#!/usr/bin/env python3
"""临时验证：检查正在收集的 v3 数据"""
import sys
import os
sys.argv = ['quick_verify_v3.py', '--ckpt_dir', './ckpts/my_transfer_cube_model', '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']
sys.path.insert(0, '/root/act')

import torch
import numpy as np
from einops import rearrange
from constants import SIM_TASK_CONFIGS
from utils import sample_box_pose
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env

class NetworkHook:
    def __init__(self, name):
        self.name = name
        self.outputs = []
        self.handle = None
    
    def hook_fn(self, module, input, output):
        self.outputs.append(output.detach().cpu())
    
    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
    
    def get_output(self, idx=-1):
        return self.outputs[idx] if self.outputs and idx < len(self.outputs) else None
    
    def clear(self):
        self.outputs = []

print("="*70)
print("快速验证激活数据 (1 集)")
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

# 注册 hooks
print("\n注册 hooks...")
hooks = {}
model = policy.model
if hasattr(model, 'encoder') and model.encoder is not None:
    for i, layer in enumerate(model.encoder.layers):
        if hasattr(layer, 'linear2'):
            hooks[f"layer{i}_ffn"] = NetworkHook(f"layer{i}_ffn")
            hooks[f"layer{i}_ffn"].register(layer.linear2)

if hasattr(model, 'decoder') and model.decoder is not None:
    last_layer = model.decoder.layers[-1]
    if hasattr(last_layer, 'linear2'):
        hooks['decoder'] = NetworkHook('decoder')
        hooks['decoder'].register(last_layer.linear2)

print(f"✅ 注册 hooks: {list(hooks.keys())}")

# 创建环境
print("\n创建环境...")
env = make_sim_env('sim_transfer_cube_scripted')
BOX_POSE[0] = sample_box_pose()
ts = env.reset()
print("✅ 环境就绪")

# 运行一步
print("\n运行推理...")
camera_names = ['top']
qpos = ts.observation['qpos'].copy()
qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
curr_images = [rearrange(ts.observation['images'][cam_name], 'h w c -> c h w') for cam_name in camera_names]
curr_image = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().cuda().unsqueeze(0)

# 不用 no_grad，让 hook 能捕获输出
all_actions = policy(qpos_tensor, curr_image)
action = all_actions[:, 0, :].detach().cpu().numpy()

print(f"✅ 推理完成，action shape: {action.shape}")

# 检查激活数据
print("\n" + "="*70)
print("激活数据验证")
print("="*70)

all_good = True
for key, hook in hooks.items():
    if hook.outputs:
        out = hook.outputs[0]
        nonzero_pct = (out!=0).sum()/out.numel()*100
        print(f"✅ {key}: shape={out.shape}, 非零={nonzero_pct:.1f}%, mean={out.mean():.6f}")
    else:
        print(f"❌ {key}: 输出为空!")
        all_good = False

print("\n" + "="*70)
if all_good:
    print("🎉 所有 hooks 都捕获到非空激活数据!")
else:
    print("⚠️ 部分 hooks 输出为空")
print("="*70)
