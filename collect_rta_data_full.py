#!/usr/bin/env python3
"""
collect_rta_data_full.py - 完整 RTA 数据收集 (最终版)

收集内容:
1. 轨迹数据：qpos, qvel, action, reward, success
2. OOD 特征：ResNet Backbone 输出 (512 维)
3. 激活链路：三层联合模式 (encoder_input + enc_layer3 + dec_layer6) → (3, 512)
4. ACT 输入输出：input_qpos, output_action (用于模态聚类)
5. 梯度：∂a/∂qpos (每 10 步计算一次)
"""
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
import os
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
        
        # 1. Encoder 输入层 (qpos 投影后)
        if hasattr(model, 'encoder_joint_proj'):
            hooks['encoder_input'] = NetworkHook('encoder_input')
            hooks['encoder_input'].register(model.encoder_joint_proj)
        
        # 2. Encoder 最后层 (layer3) FFN 输出
        if hasattr(model, 'encoder') and model.encoder is not None:
            if len(model.encoder.layers) > 3:
                layer3 = model.encoder.layers[3]
                if hasattr(layer3, 'linear2'):
                    hooks['encoder_layer3_ffn'] = NetworkHook('encoder_layer3_ffn')
                    hooks['encoder_layer3_ffn'].register(layer3.linear2)
        
        # 3. Decoder 最后层 (layer6) FFN 输出
        if hasattr(model, 'decoder') and model.decoder is not None:
            if len(model.decoder.layers) > 6:
                layer6 = model.decoder.layers[6]
                if hasattr(layer6, 'linear2'):
                    hooks['decoder_layer6_ffn'] = NetworkHook('decoder_layer6_ffn')
                    hooks['decoder_layer6_ffn'].register(layer6.linear2)
        
        # 4. Backbone (ResNet) 输出 - 用于 OOD
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

def main():
    # 配置
    ckpt_dir = './ckpts/my_transfer_cube_model'
    task_name = 'sim_transfer_cube_scripted'
    num_episodes = 50
    output_dir = './data/rta_training_full'
    gradient_step = 10
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("RTA Data Collection - Full Version (三层激活链路)")
    print("="*80)
    print(f"Model: {ckpt_dir}")
    print(f"Task: {task_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Output: {output_dir}")
    print(f"Gradient computation: every {gradient_step} steps")
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
    policy.eval()  # ✅ 评估模式，不训练
    
    num_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"  ✓ 模型参数：{num_params:.2f}M")
    print(f"  ✓ 模式：eval (不更新参数)")
    
    # 注册 Hook
    print("\n【2】注册 Hook...")
    hooks = register_hooks(policy)
    print(f"  ✓ Hook 列表：{list(hooks.keys())}")
    
    # 创建环境
    print("\n【3】创建仿真环境...")
    env = make_sim_env(task_name)
    camera_names = ['top']
    print(f"  ✓ 任务：{task_name}")
    print(f"  ✓ 相机：{camera_names}")
    
    # 数据收集
    print("\n【4】开始收集数据...")
    all_data = []
    
    for ep in tqdm(range(num_episodes), desc="Collecting episodes"):
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
            'backbone_feature': [],  # OOD (512 维)
            'activation_pattern': [],  # 三层联合激活 (3, 512)
            'input_qpos': [],  # ACT 输入
            'output_action': [],  # ACT 输出
            'gradient': [],  # 梯度 (每 10 步)
            'gradient_steps': [],
        }
        
        qpos = ts.observation['qpos'].copy()
        
        for t in range(400):  # 8 秒 @ 50Hz
            # 准备输入
            qpos_tensor = torch.from_numpy(qpos.copy()).float().cuda().unsqueeze(0)
            curr_images = [rearrange(ts.observation['images'][cam], 'h w c -> c h w') for cam in camera_names]
            image_tensor = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().cuda().unsqueeze(0)
            
            # 清空 Hook
            for hook in hooks.values():
                hook.clear()
            
            # 前向传播 (触发 Hook)
            with torch.set_grad_enabled(False):
                all_actions = policy(qpos_tensor, image_tensor)  # (1, 100, 14)
            
            # 提取 Hook 数据
            # 1. Backbone (ResNet 输出)
            if 'backbone' in hooks and hooks['backbone'].output is not None:
                feat = hooks['backbone'].output
                if len(feat.shape) == 4:
                    feat = feat.mean(dim=(2, 3))  # Global average pooling
                episode_data['backbone_feature'].append(feat[0].cpu().numpy())
            
            # 2. 三层激活链路 (平均 pooling)
            activation_layers = []
            
            if 'encoder_input' in hooks and hooks['encoder_input'].output is not None:
                enc_in = hooks['encoder_input'].output
                if len(enc_in.shape) >= 2:
                    enc_in_pooled = enc_in[0].mean(dim=0)  # (512,)
                    activation_layers.append(enc_in_pooled.cpu().numpy())
            
            if 'encoder_layer3_ffn' in hooks and hooks['encoder_layer3_ffn'].output is not None:
                enc_l3 = hooks['encoder_layer3_ffn'].output
                if len(enc_l3.shape) >= 2:
                    enc_l3_pooled = enc_l3[0].mean(dim=0)  # (512,)
                    activation_layers.append(enc_l3_pooled.cpu().numpy())
            
            if 'decoder_layer6_ffn' in hooks and hooks['decoder_layer6_ffn'].output is not None:
                dec_l6 = hooks['decoder_layer6_ffn'].output
                if len(dec_l6.shape) >= 2:
                    dec_l6_pooled = dec_l6[0].mean(dim=0)  # (512,)
                    activation_layers.append(dec_l6_pooled.cpu().numpy())
            
            if len(activation_layers) == 3:
                episode_data['activation_pattern'].append(np.stack(activation_layers, axis=0))  # (3, 512)
            
            # 获取动作
            action = post_process(all_actions[:, t % 100, :])
            
            # 记录输入输出
            episode_data['input_qpos'].append(qpos.copy())
            episode_data['output_action'].append(all_actions[0, t % 100].cpu().numpy())
            
            # 定期计算梯度
            if t % gradient_step == 0:
                grad = compute_gradient(policy, qpos_tensor, image_tensor)
                episode_data['gradient'].append(grad)
                episode_data['gradient_steps'].append(t)
            
            # 执行动作
            ts = env.step(action)
            
            # 记录状态
            episode_data['qpos'].append(qpos.copy())
            episode_data['qvel'].append(ts.observation['qvel'][:14].copy() if len(ts.observation['qvel']) > 14 else np.zeros(14))
            episode_data['action'].append(action.copy())
            episode_data['reward'].append(ts.reward if ts.reward is not None else 0.0)
            
            qpos = ts.observation['qpos'].copy()
        
        # 检查成功
        episode_data['success'] = ts.reward >= 0.5 * env.task.max_reward
        
        # 转换为 numpy 数组
        for key in ['qpos', 'qvel', 'action', 'reward', 'backbone_feature', 'input_qpos', 'output_action']:
            if episode_data[key]:
                episode_data[key] = np.array(episode_data[key])
        
        if episode_data['activation_pattern']:
            episode_data['activation_pattern'] = np.array(episode_data['activation_pattern'])
        
        if episode_data['gradient']:
            episode_data['gradient'] = np.array(episode_data['gradient'])
        
        all_data.append(episode_data)
        
        # 验证数据
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"\n  Ep {ep+1}/{num_episodes}:")
            print(f"    success={episode_data['success']}")
            print(f"    qpos: {episode_data['qpos'].shape}")
            print(f"    backbone: {episode_data['backbone_feature'].shape}")
            print(f"    activation_pattern: {episode_data['activation_pattern'].shape if episode_data['activation_pattern'] is not None else 'N/A'}")
            print(f"    gradient: {len(episode_data['gradient'])} steps")
    
    # 保存数据
    print("\n【5】保存数据...")
    output_file = os.path.join(output_dir, 'rta_data_full.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"  ✓ 保存至：{output_file}")
    print(f"  ✓ 总集数：{len(all_data)}")
    
    # 统计信息
    print("\n【6】数据统计")
    total_steps = sum(len(ep['qpos']) for ep in all_data)
    success_count = sum(1 for ep in all_data if ep['success'])
    print(f"  总步数：{total_steps}")
    print(f"  成功集数：{success_count} / {len(all_data)}")
    
    print("\n" + "="*80)
    print("✅ 数据收集完成!")
    print("="*80)

if __name__ == '__main__':
    main()
