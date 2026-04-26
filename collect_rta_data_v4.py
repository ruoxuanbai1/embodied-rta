#!/usr/bin/env python3
"""
Collect RTA Data v4 - Hook backbone features (inference mode)
"""
import sys
sys.argv = ['collect_rta_data_v4.py', '--ckpt_dir', './ckpts/my_transfer_cube_model', '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']

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
        if isinstance(output, (list, tuple)):
            feat = output[0] if len(output) > 0 else None
        elif isinstance(output, dict):
            feat = output.get('feat', output.get('0', None))
        else:
            feat = output
        if feat is not None:
            feat_pooled = feat.mean(dim=(2, 3)) if len(feat.shape) == 4 else feat
            self.outputs.append(feat_pooled.detach().cpu())
    
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
    parser.add_argument('--output_dir', type=str, default='./data/rta_training_v4')
    args, unknown = parser.parse_known_args()
    
    ckpt_dir = args.ckpt_dir
    task_name = args.task_name
    num_episodes = args.num_episodes
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("RTA Data Collection v4 (Hook Backbone)")
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
    print("number of parameters: %.2fM" % (sum(p.num.numel() for p in policy.parameters()) / 1e6))
    
    print("\nRegistering backbone hook...")
    backbone_hook = NetworkHook('backbone')
    if policy.model.backbones:
        backbone_hook.register(policy.model.backbones[0])
        print(f"✅ Registered hook on backbone")
    else:
        print("❌ No backbones found!")
        return
    
    print("\nCreating environment...")
    env = make_sim_env(task_name)
    camera_names = ['top']
    
    all_data = []
    
    for ep in tqdm(range(num_episodes), desc="Collecting"):
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()
        
        qpos_list = []
        qvel_list = []
        action_list = []
        reward_list = []
        backbone_features = []
        
        qpos = ts.observation['qpos'].copy()
        
        for t in range(400):
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            curr_images = [rearrange(ts.observation['images'][cam_name], 'h w c -> c h w') for cam_name in camera_names]
            curr_image = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().cuda().unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                all_actions = policy(qpos_tensor, curr_image)
            action = post_process(all_actions[:, t % 100, :].detach().cpu().numpy())
            
            # Extract backbone feature
            if backbone_hook.outputs:
                backbone_features.append(backbone_hook.get_output().numpy())
            backbone_hook.clear()
            
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
            'backbone_features': np.array(backbone_features) if backbone_features else None,
        }
        all_data.append(episode_data)
        
        bf_shape = backbone_features[0].shape if backbone_features else None
        print(f"  Ep {ep+1}: success={success}, backbone shape={bf_shape}")
    
    print("\nSaving...")
    with open(os.path.join(output_dir, 'backbone_features_v4.pkl'), 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"Output: {output_dir}/backbone_features_v4.pkl")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
