#!/usr/bin/env python3
"""
train_region3_v3.py - Region 3: 三模块异常检测训练 (适配 v3.1 数据格式)

v3.1 数据格式:
- 每集一个 pickle 文件：ep000.pkl, ep001.pkl, ...
- 每个文件包含:
  - qpos: (T, 14)
  - qvel: (T, 14)
  - action: (T, 14)
  - reward: (T,)
  - success: bool
  - layer_activations: Dict
    - layer0_ffn: (T, 102, 1, 512)
    - layer1_ffn: (T, 102, 1, 512)
    - layer2_ffn: (T, 102, 1, 512)
    - layer3_ffn: (T, 102, 1, 512)

三模块:
1. 决策因子贡献度 (梯度分析) → S_logic
2. 激活链路 (4 层 Encoder FFN 激活) → D_act
3. OOD 检测 (马氏距离) → D_ood
"""

import numpy as np
import pickle
import json
import os
import glob
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 数据加载 (v3.1 格式)
# ============================================================================

def load_region3_data_v3(data_dir):
    """
    加载 Region 3 v3.1 训练数据
    
    返回:
    - trajectories: List[Dict] - 轨迹数据
    - activations: List[Dict] - 激活数据 (4 层)
    - all_states: np.array - 所有状态 (用于 OOD)
    """
    print(f"📦 加载 Region 3 v3.1 数据：{data_dir}")
    
    ep_files = sorted(glob.glob(os.path.join(data_dir, 'ep*.pkl')))
    print(f"  找到 {len(ep_files)} 集")
    
    trajectories = []
    activations = []
    all_states = []
    
    for ep_file in tqdm(ep_files, desc="Loading"):
        with open(ep_file, 'rb') as f:
            ep_data = pickle.load(f)
        
        traj = {
            'qpos': ep_data['qpos'],  # (T, 14)
            'qvel': ep_data['qvel'],  # (T, 14)
            'action': ep_data['action'],  # (T, 14)
            'reward': ep_data['reward'],  # (T,)
            'success': ep_data['success'],
        }
        trajectories.append(traj)
        
        # 提取激活数据
        act = {
            'layer0_ffn': ep_data['layer_activations'].get('layer0_ffn'),
            'layer1_ffn': ep_data['layer_activations'].get('layer1_ffn'),
            'layer2_ffn': ep_data['layer_activations'].get('layer2_ffn'),
            'layer3_ffn': ep_data['layer_activations'].get('layer3_ffn'),
        }
        activations.append(act)
        
        # 收集状态 (qpos + qvel)
        qpos = ep_data['qpos']
        qvel = ep_data['qvel']
        states = np.concatenate([qpos, qvel], axis=1)  # (T, 28)
        all_states.append(states)
    
    all_states_concat = np.concatenate(all_states, axis=0)
    
    print(f"✓ 轨迹数据：{len(trajectories)} 集")
    print(f"✓ 激活数据：{len(activations)} 集")
    print(f"✓ 状态数据：{all_states_concat.shape} (总帧数)")
    
    return trajectories, activations, all_states_concat


# ============================================================================
# 动作模态聚类
# ============================================================================

def cluster_modalities(trajectories, K=8, seed=42):
    """对动作进行 K-Means 聚类"""
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


# ============================================================================
# 模块 1: OOD 统计量学习
# ============================================================================

def learn_ood_stats(all_states):
    """学习 OOD 检测的马氏距离统计量"""
    print("\n📊 学习 OOD 统计量 (马氏距离)...")
    
    print(f"  总状态数：{len(all_states)}")
    print(f"  状态维度：{all_states.shape[1]}")
    
    # 均值和协方差
    μ = np.mean(all_states, axis=0)
    Σ = np.cov(all_states.T)
    
    # 正则化求逆
    reg = 1e-5 * np.eye(Σ.shape[0])
    Σ_inv = np.linalg.inv(Σ + reg)
    
    # 计算所有正常样本的马氏距离
    print(f"  计算马氏距离...")
    D_ood_normal = []
    for i in tqdm(range(len(all_states)), desc="Mahalanobis"):
        d = mahalanobis(all_states[i], μ, Σ_inv)
        D_ood_normal.append(d)
    
    # 阈值：99% 分位数
    threshold = np.percentile(D_ood_normal, 99)
    print(f"  OOD 阈值 (99% 分位数): {threshold:.4f}")
    print(f"  马氏距离统计：min={min(D_ood_normal):.4f}, "
          f"max={max(D_ood_normal):.4f}, mean={np.mean(D_ood_normal):.4f}")
    
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


# ============================================================================
# 模块 2: 激活链路学习 (4 层 Encoder)
# ============================================================================

def learn_activation_profiles(activations, all_modality_labels, K=8):
    """
    学习每个模态的标准激活连接掩码 (4 层 Encoder FFN 输出)
    
    使用汉明距离：先二值化激活 (>0=1, ≤0=0)，然后学习标准激活模式
    """
    print("\n📊 学习激活链路掩码 (4 层 Encoder, 汉明距离)...")
    
    activation_profiles = []
    
    for k in range(K):
        # 收集模态 k 的所有二值激活
        layer_binary_acts_k = {f'layer{i}_ffn': [] for i in range(4)}
        
        for ep_idx, (act_data, modality_labels) in enumerate(zip(activations, all_modality_labels)):
            mask = (modality_labels == k)
            
            for t in range(len(mask)):
                if mask[t]:
                    for i in range(4):
                        layer_key = f'layer{i}_ffn'
                        layer_act = act_data.get(layer_key)
                        if layer_act is not None and t < len(layer_act):
                            # layer_act[t]: (102, 1, 512)
                            # 取 cls token (位置 0) 的激活：(512,)
                            cls_act = layer_act[t][0, 0, :]  # (512,)
                            # 二值化：>0 = 1 (激活), ≤0 = 0 (不激活)
                            binary_act = (cls_act > 0).astype(int)  # (512,) 0/1
                            layer_binary_acts_k[layer_key].append(binary_act)
        
        # 通过投票确定标准激活掩码 (>50% 样本激活的神经元)
        profile = {}
        for i in range(4):
            layer_key = f'layer{i}_ffn'
            binary_acts = layer_binary_acts_k[layer_key]
            
            if len(binary_acts) > 0:
                binary_array = np.array(binary_acts)  # (N, 512)
                # 投票：>50% 样本激活的神经元 = 1
                M_ref = (np.mean(binary_array, axis=0) > 0.5).astype(int)  # (512,)
                profile[layer_key] = {
                    'M_ref': M_ref.tolist(),
                    'n_samples': len(binary_acts),
                    'activation_rate': np.mean(binary_array, axis=0).tolist(),  # 每个神经元的激活频率
                }
                print(f"  ✓ 模态 {k} - {layer_key}: {len(binary_acts)} 样本，激活率={np.mean(M_ref)*100:.1f}%")
            else:
                profile[layer_key] = None
                print(f"  ⚠️ 模态 {k} - {layer_key}: 无样本")
        
        activation_profiles.append(profile)
    
    return activation_profiles


# ============================================================================
# 风险分数计算
# ============================================================================

def extract_cls_acts(layer_activations):
    """
    从 layer_activations 提取 cls token 的二值激活
    
    输入：layer_activations = Dict
        - layer0_ffn: (T, 102, 1, 512)
        - layer1_ffn: (T, 102, 1, 512)
        ...
    
    输出：cls_acts = Dict
        - layer0_ffn: (T, 512) 二值
        - layer1_ffn: (T, 512) 二值
        ...
    """
    cls_acts = {}
    for i in range(4):
        layer_key = f'layer{i}_ffn'
        layer_act = layer_activations.get(layer_key)
        if layer_act is not None:
            # 取 cls token (位置 0): (T, 512)
            cls_act = layer_act[:, 0, 0, :]
            # 二值化
            cls_acts[layer_key] = (cls_act > 0).astype(int)
        else:
            cls_acts[layer_key] = None
    return cls_acts


def compute_activation_risk(curr_binary, M_ref):
    """
    计算激活风险分数 (汉明距离)
    
    curr_binary: (512,) 当前二值激活
    M_ref: (512,) 标准激活掩码
    
    D_ham = 不同的神经元数量 / 总神经元数
    """
    if M_ref is None or curr_binary is None:
        return 0.5
    
    distance = np.sum(curr_binary != M_ref) / len(M_ref)
    return distance


def compute_ood_risk(state, ood_stats):
    """
    计算 OOD 风险分数 (马氏距离)
    
    状态变量是连续物理量，用马氏距离考虑协方差
    """
    μ = np.array(ood_stats['mu'])
    Σ_inv = np.array(ood_stats['sigma_inv'])
    threshold = ood_stats['threshold']
    
    d = mahalanobis(state, μ, Σ_inv)
    
    # 归一化到 [0, 1]: d/threshold
    risk = d / threshold
    risk = min(max(risk, 0.0), 1.0)
    
    return risk


# ============================================================================
# 训练主函数
# ============================================================================

def train_region3_v3(data_dir, output_dir, config=None):
    """训练 Region 3 三模块检测器 (v3.1 数据格式)"""
    if config is None:
        config = {
            'K': 8,
            'seed': 42,
        }
    
    print("=" * 60)
    print("Region 3: 三模块异常检测训练 (v3.1)")
    print("=" * 60)
    print(f"配置：{json.dumps(config, indent=2)}")
    
    # 1. 加载数据
    trajectories, activations, all_states = load_region3_data_v3(data_dir)
    
    # 2. 动作模态聚类
    kmeans, all_modality_labels = cluster_modalities(
        trajectories, K=config['K'], seed=config['seed']
    )
    
    # 3. 学习 OOD 统计量
    ood_stats = learn_ood_stats(all_states)
    
    # 4. 学习激活链路模式
    activation_profiles = learn_activation_profiles(activations, all_modality_labels, K=config['K'])
    
    # 5. 保存结果
    print("\n💾 保存训练结果...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 KMeans 模型
    import joblib
    joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_model.pkl'))
    print(f"  ✓ kmeans_model.pkl")
    
    # 保存 OOD 统计量
    with open(os.path.join(output_dir, 'ood_stats.json'), 'w') as f:
        json.dump(ood_stats, f, indent=2)
    print(f"  ✓ ood_stats.json")
    
    # 保存激活模式
    with open(os.path.join(output_dir, 'activation_profiles.json'), 'w') as f:
        json.dump(activation_profiles, f, indent=2)
    print(f"  ✓ activation_profiles.json")
    
    # 保存训练摘要
    training_summary = {
        'K': config['K'],
        'seed': config['seed'],
        'num_episodes': len(trajectories),
        'total_frames': len(all_states),
        'ood_threshold': ood_stats['threshold'],
    }
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"  ✓ training_summary.json")
    
    print("\n" + "=" * 60)
    print("✅ Region 3 训练完成!")
    print(f"输出目录：{output_dir}")
    print("=" * 60)
    
    return {
        'kmeans': kmeans,
        'ood_stats': ood_stats,
        'activation_profiles': activation_profiles,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/rta_training_v3')
    parser.add_argument('--output_dir', type=str, default='./outputs/region3_v3')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    config = {
        'K': args.K,
        'seed': args.seed,
    }
    
    train_region3_v3(args.data_dir, args.output_dir, config)
