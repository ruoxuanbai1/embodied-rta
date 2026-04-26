#!/usr/bin/env python3
"""
train_region3_activation_v2.py - Region 3 模块 2: 激活链路训练 (传播链路版本)

核心思想:
- 检测 4 层 Encoder 之间的激活传播模式
- 正常数据学习标准传播链路 M_ref
- 异常检测：当前链路 vs 标准链路的汉明距离

传播链路表示:
方法 1: 外积表示层间连接
  connection_01 = outer(layer0_act, layer1_act)  # (512, 512)
  connection_12 = outer(layer1_act, layer2_act)
  connection_23 = outer(layer2_act, layer3_act)
  总链路 = concat(connection_01, connection_12, connection_23)  # (3×512×512)

方法 2: 4 层联合激活模式
  combined_act = concat(layer0_act, layer1_act, layer2_act, layer3_act)  # (4×512)
  学习联合激活的相关性矩阵

方法 3 (简化): 4 层激活的一致性
  检查 4 层激活是否"一致" (都激活或都不激活)
"""

import numpy as np
import pickle
import json
import os
import glob
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


def load_activation_data(data_dir, fault_data_dir=None):
    """加载激活数据"""
    print(f"📦 加载激活数据：{data_dir}")
    
    ep_files = sorted(glob.glob(os.path.join(data_dir, 'ep*.pkl')))
    print(f"  找到 {len(ep_files)} 集 (正常)")
    
    trajectories = []
    activations = []
    
    for ep_file in tqdm(ep_files, desc="Loading normal"):
        with open(ep_file, 'rb') as f:
            ep_data = pickle.load(f)
        trajectories.append({'action': ep_data['action']})
        activations.append({
            f'layer{i}_ffn': ep_data['layer_activations'].get(f'layer{i}_ffn')
            for i in range(4)
        })
    
    # 加载故障数据
    fault_activations = []
    if fault_data_dir and os.path.exists(fault_data_dir):
        fault_files = sorted(glob.glob(os.path.join(fault_data_dir, 'ep*.pkl')))
        print(f"  找到 {len(fault_files)} 集 (故障)")
        for ep_file in tqdm(fault_files, desc="Loading fault"):
            with open(ep_file, 'rb') as f:
                ep_data = pickle.load(f)
            fault_activations.append({
                f'layer{i}_ffn': ep_data['layer_activations'].get(f'layer{i}_ffn')
                for i in range(4)
            })
    
    print(f"✓ 正常：{len(trajectories)} 集，故障：{len(fault_activations)} 集")
    return trajectories, activations, fault_activations


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
    
    all_modality_labels = []
    action_idx = 0
    for traj in trajectories:
        T = len(traj['action'])
        labels = all_labels[action_idx:action_idx+T]
        all_modality_labels.append(labels)
        action_idx += T
    
    return kmeans, all_modality_labels


def extract_cls_acts(layer_activations, t):
    """提取 t 时刻 4 层的 cls token 激活"""
    cls_acts = []
    for i in range(4):
        layer_key = f'layer{i}_ffn'
        layer_act = layer_activations.get(layer_key)
        if layer_act is not None and t < len(layer_act):
            # (102, 1, 512) → cls token (位置 0) → (512,)
            cls_act = layer_act[t][0, 0, :]
            cls_acts.append(cls_act)
    
    if len(cls_acts) == 4:
        return np.concatenate(cls_acts)  # (2048,)
    return None


def compute_propagation_link(cls_acts):
    """
    计算传播链路表示
    
    cls_acts: (2048,) = concat(layer0, layer1, layer2, layer3), 每层 512 维
    
    返回:
    - 二值化联合激活模式 (2048,)
    - 或层间外积连接 (简化为向量)
    """
    # 方法 1: 直接使用联合激活 (二值化)
    binary_combined = (cls_acts > 0).astype(int)  # (2048,)
    
    # 方法 2: 计算层间相关性 (简化版)
    # layer0_corr = layer0 * layer1 (逐元素相乘，表示同步激活)
    layer0 = cls_acts[0:512]
    layer1 = cls_acts[512:1024]
    layer2 = cls_acts[1024:1536]
    layer3 = cls_acts[1536:2048]
    
    # 层间同步激活模式
    sync_01 = (layer0 > 0) & (layer1 > 0)  # (512,)
    sync_12 = (layer1 > 0) & (layer2 > 0)
    sync_23 = (layer2 > 0) & (layer3 > 0)
    
    # 组合：联合激活 + 同步模式
    propagation_vector = np.concatenate([
        binary_combined,      # (2048,)
        sync_01.astype(int),  # (512,)
        sync_12.astype(int),  # (512,)
        sync_23.astype(int),  # (512,)
    ])  # (3584,)
    
    return propagation_vector


def learn_activation_links(activations, all_modality_labels, K=8, fault_activations=None):
    """
    学习每个模态的标准激活传播链路
    
    对每个模态:
    1. 收集所有时刻的传播链路向量
    2. 投票确定标准链路 M_ref
    3. 计算正常样本的汉明距离分布
    4. 学习阈值 (分位数或 Youden)
    """
    print("\n📊 学习激活传播链路 (汉明距离 + 阈值学习)...")
    
    M_ref_profiles = []
    
    for k in range(K):
        # 收集模态 k 的所有传播链路
        propagation_vectors_k = []
        propagation_vectors_fault_k = []
        
        # 正常样本
        for ep_idx, (act_data, modality_labels) in enumerate(zip(activations, all_modality_labels)):
            mask = (modality_labels == k)
            T = len(act_data['layer0_ffn'])
            
            for t in range(T):
                if mask[t]:
                    cls_acts = extract_cls_acts(act_data, t)
                    if cls_acts is not None:
                        prop_vec = compute_propagation_link(cls_acts)
                        propagation_vectors_k.append(prop_vec)
        
        # 故障样本
        if fault_activations:
            for act_data in fault_activations:
                T = len(act_data['layer0_ffn'])
                for t in range(T):
                    cls_acts = extract_cls_acts(act_data, t)
                    if cls_acts is not None:
                        prop_vec = compute_propagation_link(cls_acts)
                        propagation_vectors_fault_k.append(prop_vec)
        
        if len(propagation_vectors_k) > 0:
            prop_array = np.array(propagation_vectors_k)  # (N, 3584)
            
            # 投票确定标准传播链路 (>50% 样本激活的位 = 1)
            M_ref = (np.mean(prop_array, axis=0) > 0.5).astype(int)  # (3584,)
            
            # 计算正常样本的汉明距离分布
            distances_normal = []
            for pv in propagation_vectors_k:
                d = np.sum(pv != M_ref) / len(M_ref)
                distances_normal.append(d)
            
            # 故障样本的汉明距离
            distances_fault = []
            for pv in propagation_vectors_fault_k:
                d = np.sum(pv != M_ref) / len(M_ref)
                distances_fault.append(d)
            
            # 阈值学习
            if len(distances_fault) > 0:
                # Youden 指数优化
                y_true = [0] * len(distances_normal) + [1] * len(distances_fault)
                y_scores = distances_normal + distances_fault
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                youden = tpr - fpr
                optimal_idx = np.argmax(youden)
                threshold = thresholds[optimal_idx]
                auc_score = auc(fpr, tpr)
                method = f'Youden (AUC={auc_score:.3f})'
            else:
                # 分位数法
                threshold = np.percentile(distances_normal, 95)
                method = '95% 分位数 (无故障数据)'
            
            profile = {
                'M_ref': M_ref.tolist(),
                'threshold': float(threshold),
                'method': method,
                'n_normal': len(propagation_vectors_k),
                'n_fault': len(propagation_vectors_fault_k),
                'D_ham_normal': {
                    'mean': float(np.mean(distances_normal)),
                    'std': float(np.std(distances_normal)),
                    'p95': float(np.percentile(distances_normal, 95)),
                    'p99': float(np.percentile(distances_normal, 99)),
                },
                'dim': len(M_ref),  # 3584
                'description': '联合激活 (2048) + 层间同步 (1536)',
            }
            
            print(f"  ✓ 模态 {k}: {len(propagation_vectors_k)} 正常，{len(propagation_vectors_fault_k)} 故障，"
                  f"阈值={threshold:.4f} ({method})")
        else:
            profile = None
            print(f"  ⚠️ 模态 {k}: 无样本")
        
        M_ref_profiles.append(profile)
    
    return M_ref_profiles


def train_activation_module(data_dir, output_dir, config=None, fault_data_dir=None):
    """训练激活链路模块"""
    if config is None:
        config = {'K': 8, 'seed': 42}
    
    print("=" * 60)
    print("Region 3 模块 2: 激活链路训练 (传播链路)")
    print("=" * 60)
    
    # 1. 加载数据
    trajectories, activations, fault_activations = load_activation_data(data_dir, fault_data_dir)
    
    # 2. 动作聚类
    kmeans, all_modality_labels = cluster_modalities(trajectories, K=config['K'], seed=config['seed'])
    
    # 3. 学习传播链路
    M_ref_profiles = learn_activation_links(
        activations, all_modality_labels, K=config['K'], fault_activations=fault_activations
    )
    
    # 4. 保存结果
    print("\n💾 保存结果...")
    os.makedirs(output_dir, exist_ok=True)
    
    import joblib
    joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_activation.pkl'))
    
    with open(os.path.join(output_dir, 'activation_links.json'), 'w') as f:
        json.dump(M_ref_profiles, f, indent=2)
    
    summary = {
        'module': 'activation_link_propagation',
        'K': config['K'],
        'num_normal': len(trajectories),
        'num_fault': len(fault_activations) if fault_activations else 0,
        'layers': 4,
        'dim_per_layer': 512,
        'propagation_dim': 3584,  # 2048 + 3×512
        'method': 'Hamming distance on propagation vector',
    }
    with open(os.path.join(output_dir, 'activation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ kmeans_activation.pkl")
    print(f"  ✓ activation_links.json")
    print(f"  ✓ activation_summary.json")
    
    print("\n" + "=" * 60)
    print("✅ 激活链路模块训练完成!")
    print("=" * 60)
    
    return {'kmeans': kmeans, 'M_ref_profiles': M_ref_profiles}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/rta_training_v3')
    parser.add_argument('--fault_data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs/region3_activation_v2')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train_activation_module(
        args.data_dir, args.output_dir,
        config={'K': args.K, 'seed': args.seed},
        fault_data_dir=args.fault_data_dir
    )
