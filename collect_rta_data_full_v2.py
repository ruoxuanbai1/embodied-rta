#!/usr/bin/env python3
"""
collect_rta_data_full.py - 完整 RTA 数据收集 (最终版 v2)

重要配置:
- export MUJOCO_GL=egl (无显示器环境)
- 每集单独保存 (避免数据丢失)
- 三层激活链路 (encoder_input + enc_layer3 + dec_layer6)
"""
import os
os.environ['MUJOCO_GL'] = 'egl'  # ✅ 无显示器环境

import sys
sys.argv = [
    'collect_rta_data_full.py',
    '--ckpt_dir', './ckpts/my_transfer_cube_model',
    '--task_name', 'sim_transfer_cube_scripted',
    '--num_episodes', '50',
    '--output_dir', './data/rta_training_full',
]

import torch
import numpy as np
import pickle
from tqdm import tqdm
from einops import rearrange
from datetime import datetime
from constants import DT, SIM_TASK_CONFIGS
from utils import sample_box_pose
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env

def post_process(action):
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.squeeze(action)
    action = np.clip(action, -1, 1)
    return action

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
    """注册所有需要的 Hook"""
    hooks = {}
    
    if hasattr(policy, 'model'):
        model = policy.model
        
        # 1. Encoder 输入层
        if hasattr(model, 'encoder_joint_proj'):
            hooks['encoder_input'] = NetworkHook('encoder_input')
            hooks['encoder_input'].register(model.encoder_joint_proj)
        
        # 2. Encoder 最后层 (layer3) FFN
        if hasattr(model, 'encoder') and model.encoder is not None:
            if len(model.encoder.layers) > 3:
                layer3 = model.encoder.layers[3]
                if hasattr(layer3, 'linear2'):
                    hooks['encoder_layer3_ffn'] = NetworkHook('encoder_layer3_ffn')
                    hooks['encoder_layer3_ffn'].register(layer3.linear2)
        
        # 3. Decoder 最后层 (layer6) FFN
        if hasattr(model, 'decoder') and model.decoder is not None:
            if len(model.decoder.layers) > 6:
                layer6 = model.decoder.layers[6]
                if hasattr(layer6, 'linear2'):
                    hooks['decoder_layer6_ffn'] = NetworkHook('decoder_layer6_ffn')
                    hooks['decoder_layer6_ffn'].register(layer6.linear2)
        
        # 4. Backbone (ResNet) - 用于 OOD
        if hasattr(model, 'backbones') and model.backbones is not None:
            backbone = model.backbones[0]
            if hasattr(backbone, 'body') and hasattr(backbone.body, 'layer4'):
                hooks['backbone'] = NetworkHook('backbone')
                hooks['backbone'].register(backbone.body.layer4[-1])
    
    return hooks

def compute_gradient(policy, qpos_tensor, image_tensor, device='cuda'):
    """计算梯度矩阵 ∂a/∂qpos (14×14)"""
    eps = 1e-4
    qpos_np = qpos_tensor.cpu().numpy()[0]
    
    with torch.no_grad():
        base_output = policy(qpos_tensor, image_tensor)
        if isinstance(base_output, dict):
            base_action = base_output['action'][0, 0]
        else:
            base_action = base_output[0, 0]
    
    gradient = np.zeros((14, 14))
    
    for j in range(14):
        qpos_perturbed = qpos_np.copy()
        qpos_perturbed[j] += eps
        
        qpos_tensor_pert = torch.from_numpy(qpos_perturbed).float().to(device).unsqueeze(0)
        
        with torch.no_grad():
            pert_output = policy(qpos_tensor_pert, image_tensor)
            if isinstance(pert_output, dict):
                pert_action = pert_output['action'][0, 0]
            else:
                pert_action = pert_output[0, 0]
        
        gradient[:, j] = (pert_action.cpu().numpy() - base_action.cpu().numpy()) / eps
    
    return gradient

def save_episode(episode_data, output_dir):
    """保存单集数据"""
    ep_id = episode_data['episode_id']
    output_file = os.path.join(output_dir, f'ep{ep_id:03d}.pkl')
    
    with open(output_file, 'wb') as f:
        pickle.dump(episode_data, f)
    
    return output_file

def main():
    # 配置
    ckpt_dir = './ckpts/my_transfer_cube_model'
    task_name = 'sim_transfer_cube_scripted'
    num_episodes = 50
    output_dir = './data/rta_training_full'
    gradient_step = 10
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("RTA Data Collection - Full Version v2 (每集保存)")
    print("="*80)
    print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', 'Not set')}")
    print(f"Model: {ckpt_dir}")
    print(f"Task: {task_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Output: {output_dir}")
    print(f"Save: Per episode (not batch)")
    print("="*80)
    
    # 加载模型
    print("\n【1】加载 ACT 模型...")
    policy_config = {
        'lr': 1e-5, 'num_queries': 100, 'kl_weight': 10,
        'hidden_dim': 512, 'dim_feedforward': 3200, 'lr_backbone': 1e-5,
        'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7,
        'nheads': 8, 'camera_names': ['top'],
    }
    policy = ACTPolicy(policy_config)
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    policy.cuda()
    policy.eval()
    
    print(f"  ✓ 模型参数：{sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")
    
    # 注册 Hook
    print("\n【2】注册 Hook...")
    hooks = register_hooks(policy)
    print(f"  ✓ Hook: {list(hooks.keys())}")
    
    # 创建环境
    print("\n【3】创建仿真环境...")
    env = make_sim_env(task_name)
    print(f"  ✓ 任务：{task_name}")
    
    # 数据收集
    print("\n【4】开始收集数据 (每集单独保存)...")
    
    for ep in tqdm(range(num_episodes), desc="Collecting"):
        # 初始化
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()
        
        episode_data = {
            'episode_id': ep,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'qpos': [],
            'qvel': [],
            'action': [],
            'reward': [],
            'backbone_feature': [],
            'activation_pattern': [],
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
            
            # 提取 Hook 数据
            if 'backbone' in hooks and hooks['backbone'].output is not None:
                feat = hooks['backbone'].output
                if len(feat.shape) == 4:
                    feat = feat.mean(dim=(2, 3))
                episode_data['backbone_feature'].append(feat[0].cpu().numpy())
            
            # 三层激活链路
            activation_layers = []
            for hook_name in ['encoder_input', 'encoder_layer3_ffn', 'decoder_layer6_ffn']:
                if hook_name in hooks and hooks[hook_name].output is not None:
                    out = hooks[hook_name].output
                    if len(out.shape) >= 2:
                        activation_layers.append(out[0].mean(dim=0).cpu().numpy())
            
            if len(activation_layers) == 3:
                episode_data['activation_pattern'].append(np.stack(activation_layers, axis=0))
            
            action = post_process(all_actions[:, t % 100, :])
            episode_data['input_qpos'].append(qpos.copy())
            episode_data['output_action'].append(all_actions[0, t % 100].cpu().numpy())
            
            if t % gradient_step == 0:
                episode_data['gradient'].append(compute_gradient(policy, qpos_tensor, image_tensor))
                episode_data['gradient_steps'].append(t)
            
            ts = env.step(action)
            episode_data['qpos'].append(qpos.copy())
            episode_data['qvel'].append(ts.observation['qvel'][:14].copy() if len(ts.observation['qvel']) > 14 else np.zeros(14))
            episode_data['action'].append(action.copy())
            episode_data['reward'].append(ts.reward if ts.reward is not None else 0.0)
            
            qpos = ts.observation['qpos'].copy()
        
        episode_data['success'] = ts.reward >= 0.5 * env.task.max_reward
        
        # 转换为 numpy
        for key in ['qpos', 'qvel', 'action', 'reward', 'backbone_feature', 'input_qpos', 'output_action']:
            if episode_data[key]:
                episode_data[key] = np.array(episode_data[key])
        if episode_data['activation_pattern']:
            episode_data['activation_pattern'] = np.array(episode_data['activation_pattern'])
        if episode_data['gradient']:
            episode_data['gradient'] = np.array(episode_data['gradient'])
        
        # ✅ 立即保存本集数据
        output_file = save_episode(episode_data, output_dir)
        print(f"\n  ✓ Ep {ep+1}/{num_episodes} 已保存：{os.path.basename(output_file)}")
        print(f"    success={episode_data['success']}, backbone={episode_data['backbone_feature'].shape}, activation={episode_data['activation_pattern'].shape if episode_data['activation_pattern'] is not None else 'N/A'}")
    
    print("\n" + "="*80)
    print("✅ 数据收集完成！")
    print("="*80)

if __name__ == '__main__':
    main()
