#!/usr/bin/env python3
"""
Collect RTA Data v3.1 - Fixed: save per episode, less memory
"""
import sys
sys.argv = ['collect_rta_data_v3.py', '--ckpt_dir', './ckpts/my_transfer_cube_model', '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']

import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
from einops import rearrange
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/my_transfer_cube_model')
    parser.add_argument('--task_name', type=str, default='sim_transfer_cube_scripted')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./data/rta_training_v3')
    args, unknown = parser.parse_known_args()
    
    ckpt_dir = args.ckpt_dir
    task_name = args.task_name
    num_episodes = args.num_episodes
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("RTA Data Collection v3.1 (Training Mode)")
    print("="*60)
    
    print("\nLoading model...")
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
    print("number of parameters: %.2fM" % (sum(p.numel() for p in policy.parameters()) / 1e6))
    
    print("\nRegistering hooks...")
    hooks = {}
    model = policy.model
    if hasattr(model, 'encoder') and model.encoder is not None:
        for i, layer in enumerate(model.encoder.layers):
            if hasattr(layer, 'linear2'):
                hooks[f"layer{i}_ffn"] = NetworkHook(f"layer{i}_ffn")
                hooks[f"layer{i}_ffn"].register(layer.linear2)
    print(f"Registered hooks: {list(hooks.keys())}")
    
    print("\nCreating environment...")
    env = make_sim_env(task_name)
    camera_names = ['top']
    
    for ep in tqdm(range(num_episodes), desc="Collecting"):
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()
        
        qpos_list = []
        qvel_list = []
        action_list = []
        reward_list = []
        layer_activations = {f"layer{i}_ffn": [] for i in range(4)}
        gradient_contrib_list = []  # 梯度贡献度 (14, 14) per step
        
        qpos = ts.observation['qpos'].copy()
        
        for t in range(400):
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_tensor.requires_grad_(True)  # 需要梯度
            curr_images = [rearrange(ts.observation['images'][cam_name], 'h w c -> c h w') for cam_name in camera_names]
            curr_image = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().cuda().unsqueeze(0)
            
            # Training mode to get encoder activations + gradients
            dummy_actions = torch.randn(1, 100, 14).cuda()
            is_pad = torch.zeros(1, 100, dtype=torch.bool).cuda()
            
            # Forward with gradients
            _ = policy(qpos_tensor, curr_image, actions=dummy_actions, is_pad=is_pad)
            
            # Extract activations
            for i in range(4):
                layer_key = f"layer{i}_ffn"
                if layer_key in hooks and hooks[layer_key].outputs:
                    layer_activations[layer_key].append(hooks[layer_key].get_output().numpy())
            
            # Compute gradient contribution: φ = qpos × ∂a/∂qpos
            # Get action output for gradient computation
            with torch.set_grad_enabled(True):
                action_out = policy(qpos_tensor, curr_image, actions=dummy_actions, is_pad=is_pad)
                action_now = action_out[:, 0, :]  # (1, 14) - first action
                
                # Compute gradient for each action dimension
                gradient_contrib = []
                for action_dim in range(14):
                    grad = torch.autograd.grad(
                        action_now[:, action_dim].sum(),
                        qpos_tensor,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    if grad is None:
                        grad = torch.zeros_like(qpos_tensor)
                    # φ_ij = qpos_j × ∂a_i/∂qpos_j
                    contrib = qpos_tensor * grad  # (1, 14)
                    gradient_contrib.append(contrib.squeeze(0).detach().cpu().numpy())
            
            gradient_contrib = np.array(gradient_contrib)  # (14, 14)
            gradient_contrib_list.append(gradient_contrib)
            
            for hook in hooks.values():
                hook.clear()
            
            # Get action for env step (inference mode)
            with torch.no_grad():
                all_actions = policy(qpos_tensor, curr_image).detach().cpu().numpy()
            action = post_process(all_actions[:, t % 100, :])
            
            ts = env.step(action)
            
            qpos_list.append(qpos.copy())
            qvel_list.append(ts.observation['qvel'][:14].copy() if len(ts.observation['qvel']) > 14 else np.zeros(14))
            action_list.append(action.copy())
            reward_list.append(ts.reward if ts.reward is not None else 0.0)
            
            qpos = ts.observation['qpos'].copy()
        
        success = ts.reward >= 0.5 * env.task.max_reward
        
        episode_data = {
            'qpos': np.array(qpos_list),
            'qvel': np.array(qvel_list),
            'action': np.array(action_list),
            'reward': np.array(reward_list),
            'success': success,
            'layer_activations': {k: np.array(v) if v else None for k, v in layer_activations.items()},
            'gradient_contrib': np.array(gradient_contrib_list),  # (T, 14, 14)
        }
        
        # Save per episode
        ep_path = os.path.join(output_dir, f'ep{ep:03d}.pkl')
        with open(ep_path, 'wb') as f:
            pickle.dump(episode_data, f)
        
        l0_shape = layer_activations['layer0_ffn'][0].shape if layer_activations['layer0_ffn'] else None
        print(f"  Ep {ep+1}: success={success}, layer0 shape={l0_shape}")
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"Output: {output_dir}/ep*.pkl ({num_episodes} files)")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
