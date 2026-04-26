#!/usr/bin/env python3
"""
Collect RTA Data v3 - Fixed activation extraction
Collects: trajectories, activations (all layers), gradients
"""
import sys
sys.argv = ['collect_rta_data_v3.py', '--ckpt_dir', './ckpts/my_transfer_cube_model', '--policy_class', 'ACT', '--task_name', 'sim_transfer_cube_scripted', '--seed', '0', '--num_epochs', '1']

import torch
import numpy as np
import os
import pickle
import h5py
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


class BackboneFeatureExtractor:
    def __init__(self, backbone_module):
        self.backbone = backbone_module
        self.features = []
        self._register_hook()
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, (list, tuple)):
                feat = output[0][0] if isinstance(output[0], (list, tuple)) else output[0]
            elif isinstance(output, dict):
                feat = output.get('feat', output.get('0', None))
            else:
                feat = output
            if feat is None:
                return
            feat_pooled = feat.mean(dim=(2, 3)) if len(feat.shape) == 4 else feat
            self.features.append(feat_pooled.detach().cpu())
        self.handle = self.backbone.register_forward_hook(hook_fn)
    
    def get_feature(self, idx=-1):
        return self.features[idx] if self.features and idx < len(self.features) else None
    
    def clear(self):
        self.features = []

def register_hooks(policy):
    hooks = {}
    backbone_extractor = None
    
    if hasattr(policy, 'model'):
        model = policy.model
        
        if hasattr(model, 'encoder') and model.encoder is not None:
            if hasattr(model.encoder, 'layers') and len(model.encoder.layers) > 0:
                for i, encoder_layer in enumerate(model.encoder.layers):
                    if hasattr(encoder_layer, 'linear2'):
                        hooks[f"layer{i}_ffn"] = NetworkHook(f"layer{i}_ffn")
                        hooks[f"layer{i}_ffn"].register(encoder_layer.linear2)
        
        if hasattr(model, 'decoder') and model.decoder is not None:
            if hasattr(model.decoder, 'layers') and len(model.decoder.layers) > 0:
                decoder_layer = model.decoder.layers[-1]
                if hasattr(decoder_layer, 'linear2'):
                    hooks['decoder'] = NetworkHook('decoder')
                    hooks['decoder'].register(decoder_layer.linear2)
        
        if hasattr(model, 'backbones') and model.backbones is not None:
            backbone_extractor = BackboneFeatureExtractor(model.backbones[0])
    
    return hooks, backbone_extractor

def compute_gradient(policy, qpos, image, raw_action):
    action_out = policy(qpos, image)
    action_now = action_out[:, 0, :]
    
    qpos_contributions = []
    for i in range(14):
        grad = torch.autograd.grad(
            action_now[:, i].sum(),
            qpos,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]
        
        if grad is None:
            grad = torch.zeros_like(qpos)
        
        contribution = qpos * grad
        qpos_contributions.append(contribution.squeeze(0).detach().cpu().numpy())
    
    return np.stack(qpos_contributions, axis=0)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/my_transfer_cube_model')
    parser.add_argument('--task_name', type=str, default='sim_transfer_cube_scripted')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./data/rta_training_v3')
    parser.add_argument('--temporal_agg', action='store_true')
    args, unknown = parser.parse_known_args()
    
    ckpt_dir = args.ckpt_dir
    task_name = args.task_name
    num_episodes = args.num_episodes
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("RTA Data Collection v3 (Fixed Activation Extraction)")
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
    hooks, backbone_extractor = register_hooks(policy)
    print(f"Registered hooks: {list(hooks.keys())}")
    
    print("\nCreating environment...")
    env = make_sim_env(task_name)
    camera_names = ['top']
    query_frequency = 100
    
    all_data = []
    
    for ep in tqdm(range(num_episodes), desc="Collecting"):
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()
        
        qpos_list = []
        qvel_list = []
        action_list = []
        reward_list = []
        gradient_contributions = []
        layer_activations = {f"layer{i}_ffn": [] for i in range(4)}
        decoder_activations = []
        
        qpos = ts.observation['qpos'].copy()
        
        for t in range(400):
            qpos_numpy = qpos.copy()
            qpos_tensor = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
            
            curr_images = [rearrange(ts.observation['images'][cam_name], 'h w c -> c h w') for cam_name in camera_names]
            curr_image = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().cuda().unsqueeze(0)
            
            # Inference: get action
            # Training mode forward to get encoder activations (need full action sequence)
            # Generate dummy actions for the full sequence
            dummy_actions = torch.randn(1, 100, 14).cuda()
            is_pad = torch.zeros(1, 100, dtype=torch.bool).cuda()
            with torch.set_grad_enabled(False):
                _ = policy(qpos_tensor, curr_image, actions=dummy_actions, is_pad=is_pad)  # training mode
            
            # Extract activations
            for i in range(4):
                layer_key = f"layer{i}_ffn"
                if layer_key in hooks and hooks[layer_key].outputs:
                    layer_activations[layer_key].append(hooks[layer_key].get_output().numpy())
            
            # Clear hooks for next step
            for hook in hooks.values():
                hook.clear()
            
            # Now get actual action for env step (inference mode)
            with torch.no_grad():
                all_actions = policy(qpos_tensor, curr_image).detach().cpu().numpy()
            action = post_process(all_actions[:, t % 100, :])
            
            if 'decoder' in hooks and hooks['decoder'].outputs:
                decoder_activations.append(hooks['decoder'].get_output().numpy())
            
            for hook in hooks.values():
                hook.clear()
            
            ts = env.step(action)
            
            qpos_list.append(qpos.copy())
            qvel_list.append(ts.observation['qvel'][:14].copy() if len(ts.observation['qvel']) > 14 else np.zeros(14))
            action_list.append(action.copy())
            reward_list.append(ts.reward if ts.reward is not None else 0.0)
        
        success = ts.reward >= 0.5 * env.task.max_reward
        
        episode_data = {
            'qpos': np.array(qpos_list),
            'qvel': np.array(qvel_list),
            'action': np.array(action_list),
            'reward': np.array(reward_list),
            'success': success,
            'layer_activations': {k: np.array(v) if v else None for k, v in layer_activations.items()},
            'decoder_act': np.array(decoder_activations) if decoder_activations else None,
        }
        all_data.append(episode_data)
        
        print(f"  Ep {ep+1}: success={success}, layer0 shape={layer_activations['layer0_ffn'][0].shape if layer_activations['layer0_ffn'] else None}")
    
    print("\nSaving...")
    with open(os.path.join(output_dir, 'activations_v3.pkl'), 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"Output: {output_dir}/activations_v3.pkl")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
