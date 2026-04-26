#!/usr/bin/env python3
"""
train_region3_gradient_ood.py - Region 3 模块 1+3: 梯度贡献 + OOD 检测训练

数据源：旧数据 (500 条正常轨迹)
- /mnt/data/aloha_act_500_optimized/ (梯度贡献)
- /mnt/data/aloha_act_500_optimized/ (OOD 输入)

模块 1: 决策因子贡献度
- 贡献度 φ = 状态 × 梯度 (泰勒展开)
- 学习每个模态的关键特征集 F_legal

模块 3: OOD 检测
- 马氏距离 D_ood = mahalanobis(state, μ, Σ_inv)
- 阈值：99% 分位数
"""

import numpy as np
import pickle
import json
import os
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def load_gradient_ood_data(data_dir):
    """加载梯度和 OOD 数据"""
    print(f"📦 加载梯度 + OOD 数据：{data_dir}")
    
    # 加载梯度贡献
    gradient_path = os.path.join(data_dir, 'gradient_contrib.pkl')
    with open(gradient_path, 'rb') as f:
        gradient_data = pickle.load(f)
    print(f"✓ 梯度贡献：{len(gradient_data)} 集")
    
    # 加载 OOD 输入
    ood_path = os.path.join(data_dir, 'ood_inputs.pkl')
    with open(ood_path, 'rb') as f:
        ood_data = pickle.load(f)
    print(f"✓ OOD 输入：{len(ood_data)} 集")
    
    # 加载轨迹
    trajectories_path = os.path.join(data_dir, 'trajectories.hdf5')
    import h5py
    trajectories = []
    with h5py.File(trajectories_path, 'r') as f:
        for ep_id in range(len(f.keys())):
            grp = f[f'episode_{ep_id}']
            traj = {
                'qpos': grp['qpos'][:],
                'qvel': grp['qvel'][:],
                'action': grp['action'][:],
            }
            trajectories.append(traj)
    print(f"✓ 轨迹：{len(trajectories)} 集")
    
    return trajectories, gradient_data, ood_data


def cluster_modalities(trajectories, K=8, seed=42):
    """动作模态聚类"""
    print("\n📊 动作模态聚类...")
    
    all_actions = np.concatenate([traj['action'] for traj in trajectories], axis=0)
    print(f"  总动作数：{len(all_actions)}")
    
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10, max_iter=300)
    all_labels = kmeans.fit_predict(all_actions)
    
    print(f"  聚类完成，K={K}")
    for k in range(K):
        count = np.sum(all_labels == k)
        print(f"    模态 {k}: {count} 样本 ({count/len(all_actions)*100:.1f}%)")
    
    # 分割回每集
    all_modality_labels = []
    action_idx = 0
    for traj in trajectories:
        T = len(traj['action'])
        labels = all_labels[action_idx:action_idx+T]
        all_modality_labels.append(labels)
        action_idx += T
    
    return kmeans, all_modality_labels


def learn_key_features(gradient_data, trajectories, all_modality_labels, K=8):
    """
    学习每个模态的关键决策因子 F_legal
    
    贡献度 φ_ij = qpos_j × ∂a_i/∂qpos_j
    """
    print("\n📊 学习关键决策因子 (F_legal)...")
    
    F_legal_profiles = []
    
    for k in range(K):
        contribs_k = []
        
        for ep_idx, (grad_data, modality_labels, traj) in enumerate(zip(gradient_data, all_modality_labels, trajectories)):
            mask = (modality_labels == k)
            qpos = traj['qpos']
            
            for t in range(len(mask)):
                if mask[t]:
                    qpos_t = qpos[t]
                    gradient = grad_data['qpos_contrib'][t]
                    contrib = qpos_t[None, :] * gradient
                    contribs_k.append(contrib)
        
        if len(contribs_k) == 0:
            print(f"  ⚠️ 模态 {k} 无样本")
            F_legal_profiles.append(None)
            continue
        
        contribs_k = np.array(contribs_k)
        avg_contrib = np.mean(contribs_k, axis=0)
        
        # 对每个动作，找出贡献最大的前 3 个状态
        F_legal = []
        for action_dim in range(14):
            contributions = avg_contrib[action_dim, :]
            top_features = np.argsort(np.abs(contributions))[-3:].tolist()
            F_legal.append(top_features)
        
        F_legal_profiles.append(F_legal)
        print(f"  ✓ 模态 {k}: {len(contribs_k)} 样本")
    
    return F_legal_profiles


def learn_ood_stats(ood_data):
    """学习 OOD 统计量"""
    print("\n📊 学习 OOD 统计量 (马氏距离)...")
    
    all_states = np.concatenate([ood['states'] for ood in ood_data], axis=0)
    print(f"  总状态数：{len(all_states)}")
    
    μ = np.mean(all_states, axis=0)
    Σ = np.cov(all_states.T)
    
    reg = 1e-5 * np.eye(Σ.shape[0])
    Σ_inv = np.linalg.inv(Σ + reg)
    print(f"  状态维度：{len(μ)}")
    
    print(f"  计算马氏距离...")
    D_ood_normal = []
    for i in tqdm(range(len(all_states)), desc="Mahalanobis"):
        d = mahalanobis(all_states[i], μ, Σ_inv)
        D_ood_normal.append(d)
    
    threshold = np.percentile(D_ood_normal, 99)
    print(f"  OOD 阈值 (99% 分位数): {threshold:.4f}")
    print(f"  马氏距离统计：min={min(D_ood_normal):.4f}, max={max(D_ood_normal):.4f}, mean={np.mean(D_ood_normal):.4f}")
    
    ood_stats = {
        'mu': μ.tolist(),
        'sigma_inv': Σ_inv.tolist(),
        'threshold': float(threshold),
        'D_ood_normal': {
            'min': float(min(D_ood_normal)),
            'max': float(max(D_ood_normal)),
            'mean': float(np.mean(D_ood_normal)),
            'std': float(np.std(D_ood_normal)),
        }
    }
    
    return ood_stats


def train_gradient_ood_module(data_dir, output_dir, config=None):
    """训练梯度 + OOD 模块"""
    if config is None:
        config = {'K': 8, 'seed': 42}
    
    print("=" * 60)
    print("Region 3 模块 1+3: 梯度贡献 + OOD 检测")
    print("=" * 60)
    
    # 1. 加载数据
    trajectories, gradient_data, ood_data = load_gradient_ood_data(data_dir)
    
    # 2. 动作聚类
    kmeans, all_modality_labels = cluster_modalities(
        trajectories, K=config['K'], seed=config['seed']
    )
    
    # 3. 学习关键特征
    F_legal_profiles = learn_key_features(gradient_data, trajectories, all_modality_labels, K=config['K'])
    
    # 4. 学习 OOD 统计量
    ood_stats = learn_ood_stats(ood_data)
    
    # 5. 保存结果
    print("\n💾 保存结果...")
    os.makedirs(output_dir, exist_ok=True)
    
    import joblib
    joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_gradient_ood.pkl'))
    print(f"  ✓ kmeans_gradient_ood.pkl")
    
    with open(os.path.join(output_dir, 'F_legal_profiles.json'), 'w') as f:
        json.dump(F_legal_profiles, f, indent=2)
    print(f"  ✓ F_legal_profiles.json")
    
    with open(os.path.join(output_dir, 'ood_stats.json'), 'w') as f:
        json.dump(ood_stats, f, indent=2)
    print(f"  ✓ ood_stats.json")
    
    training_summary = {
        'modules': ['gradient_contribution', 'ood_detection'],
        'K': config['K'],
        'num_episodes': len(trajectories),
        'ood_threshold': ood_stats['threshold'],
    }
    with open(os.path.join(output_dir, 'gradient_ood_summary.json'), 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"  ✓ gradient_ood_summary.json")
    
    print("\n" + "=" * 60)
    print("✅ 梯度 + OOD 模块训练完成!")
    print("=" * 60)
    
    return {
        'kmeans': kmeans,
        'F_legal_profiles': F_legal_profiles,
        'ood_stats': ood_stats,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/aloha_act_500_optimized')
    parser.add_argument('--output_dir', type=str, default='./outputs/region3_gradient_ood')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train_gradient_ood_module(
        args.data_dir,
        args.output_dir,
        config={'K': args.K, 'seed': args.seed}
    )
