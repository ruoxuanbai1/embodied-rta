#!/usr/bin/env python3
"""
collect_rta_encoder.py - RTA 数据收集 (Encoder 全层激活)
记录：encoder 输入层 + layer0-3 FFN 输出
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
import sys
sys.argv = ['test', '--ckpt_dir', './ckpts/my_transfer_cube_model', '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']

import torch
import numpy as np
import pickle
from tqdm import tqdm
from einops import rearrange
from datetime import datetime
from constants import SIM_TASK_CONFIGS
from utils import sample_box_pose
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env

class NetworkHook:
    def __init__(self, name):
        self.name = name
        self.output = None
        self.handle = None
    def hook_fn(self, module, input, output):
        self.output = output.detach()
    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
    def get_output(self):
        return self.output
    def clear(self):
        self.output = None

def register_hooks(policy):
    hooks = {}
    model = policy.model
    
    # Encoder 输入层 (qpos 投影)
    if hasattr(model, 'encoder_joint_proj'):
        hooks['encoder_input'] = NetworkHook('encoder_input')
        hooks['encoder_input'].register(model.encoder_joint_proj)
        print(f"  ✓ Hook: encoder_input")
    
    # Encoder 每一层 FFN 输出 (layer0-3)
    if hasattr(model, 'encoder') and model.encoder is not None:
        for i, layer in enumerate(model.encoder.layers):
            if hasattr(layer, 'linear2'):
                hooks[f'encoder_layer{i}_ffn'] = NetworkHook(f'encoder_layer{i}_ffn')
                hooks[f'encoder_layer{i}_ffn'].register(layer.linear2)
                print(f"  ✓ Hook: encoder_layer{i}_ffn")
    
    return hooks

def main():
    output_dir = './data/rta_training_full'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("RTA Data Collection (Encoder 全层)")
    print("="*60)
    
    policy_config = {
        'lr': 1e-5, 'num_queries': 100, 'kl_weight': 10,
        'hidden_dim': 512, 'dim_feedforward': 3200, 'lr_backbone': 1e-5,
        'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7,
        'nheads': 8, 'camera_names': ['top'],
    }
    policy = ACTPolicy(policy_config)
    policy.load_state_dict(torch.load('./ckpts/my_transfer_cube_model/policy_best.ckpt', map_location='cpu'))
    policy.cuda()
    policy.eval()
    print(f"✓ 模型：{sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")
    
    hooks = register_hooks(policy)
    env = make_sim_env('sim_transfer_cube_scripted')
    print(f"✓ 环境：sim_transfer_cube_scripted")
    
    print("\n开始收集 (50 集)...")
    for ep in tqdm(range(50), desc="Collecting"):
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()
        
        episode_data = {
            'episode_id': ep,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'qpos': [], 'qvel': [], 'action': [], 'reward': [],
            'encoder_activations': {
                'input': [],
                'layer0': [],
                'layer1': [],
                'layer2': [],
                'layer3': [],
            },
            'input_qpos': [],
            'output_action': [],
            'gradient': [],
            'gradient_steps': [],
        }
        
        qpos = ts.observation['qpos'].copy()
        
        for t in range(400):
            qpos_tensor = torch.from_numpy(qpos.copy()).float().cuda().unsqueeze(0)
            curr_images = [rearrange(ts.observation['images'][cam], 'h w c -> c h w') for cam in ['top']]
            image_tensor = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().cuda().unsqueeze(0)
            
            for hook in hooks.values():
                hook.clear()
            
            with torch.set_grad_enabled(False):
                all_actions = policy(qpos_tensor, image_tensor)
            
            # 提取所有 Encoder 层激活 (5 层)
            hook_map = {
                'encoder_input': 'input',
                'encoder_layer0_ffn': 'layer0',
                'encoder_layer1_ffn': 'layer1',
                'encoder_layer2_ffn': 'layer2',
                'encoder_layer3_ffn': 'layer3',
            }
            for hook_name, short_name in hook_map.items():
                if hook_name in hooks and hooks[hook_name].output is not None:
                    out = hooks[hook_name].output
                    if len(out.shape) >= 2:
                        act = out[0].mean(dim=0).cpu().numpy()
                        episode_data['encoder_activations'][short_name].append(act)
                    else:
                        episode_data['encoder_activations'][short_name].append(np.zeros(512))
                else:
                    episode_data['encoder_activations'][short_name].append(np.zeros(512))
            
            action = all_actions[0, t % 100].cpu().numpy()
            episode_data['input_qpos'].append(qpos.copy())
            episode_data['output_action'].append(action)
            
            # 每 10 步计算梯度
            if t % 10 == 0:
                eps = 1e-4
                qpos_np = qpos.copy()
                grad = np.zeros((14, 14))
                with torch.no_grad():
                    base_action = policy(qpos_tensor, image_tensor)[0, 0].cpu().numpy()
                for j in range(14):
                    qpos_pert = qpos_np.copy()
                    qpos_pert[j] += eps
                    qpos_pert_t = torch.from_numpy(qpos_pert).float().cuda().unsqueeze(0)
                    with torch.no_grad():
                        pert_action = policy(qpos_pert_t, image_tensor)[0, 0].cpu().numpy()
                    grad[:, j] = (pert_action - base_action) / eps
                episode_data['gradient'].append(grad)
                episode_data['gradient_steps'].append(t)
            
            ts = env.step(action)
            episode_data['qpos'].append(qpos.copy())
            episode_data['qvel'].append(ts.observation['qvel'][:14].copy() if len(ts.observation['qvel']) > 14 else np.zeros(14))
            episode_data['action'].append(action)
            episode_data['reward'].append(ts.reward if ts.reward is not None else 0.0)
            qpos = ts.observation['qpos'].copy()
        
        episode_data['success'] = ts.reward >= 0.5 * env.task.max_reward
        
        # 转 numpy
        for key in ['qpos', 'qvel', 'action', 'reward', 'input_qpos', 'output_action']:
            if episode_data[key]:
                episode_data[key] = np.array(episode_data[key])
        for k in episode_data['encoder_activations']:
            if episode_data['encoder_activations'][k]:
                episode_data['encoder_activations'][k] = np.array(episode_data['encoder_activations'][k])
        if episode_data['gradient']:
            episode_data['gradient'] = np.array(episode_data['gradient'])
        
        # 保存
        output_file = os.path.join(output_dir, f'ep{ep:03d}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(episode_data, f)
        
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ✓ Ep {ep+1}/50 已保存")
    
    print("\n✅ 完成!")

if __name__ == '__main__':
    main()
