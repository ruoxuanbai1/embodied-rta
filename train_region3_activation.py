#!/usr/bin/env python3
"""
train_region3_activation.py - Region 3 模块 2: 激活链路训练

数据源：v3.1 新收集的激活数据 (ep*.pkl)
阈值学习：从正常数据中分位数学习 (可扩展到 Youden 指数)
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
    """加载激活数据（正常 + 可选故障）"""
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
    
    # 加载故障数据（如果有）
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


def learn_thresholds_youden(distances_normal, distances_fault):
    """使用 Youden 指数学习最优阈值"""
    if len(distances_fault) == 0:
        return np.percentile(distances_normal, 95), '95% 分位数 (无故障数据)'
    
    # 准备标签
    y_true = [0] * len(distances_normal) + [1] * len(distances_fault)
    y_scores = distances_normal + distances_fault
    
    # ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Youden 指数最大化
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]
    
    tpr_opt = tpr[optimal_idx]
    fpr_opt = fpr[optimal_idx]
    
    print(f"    AUC={roc_auc:.4f}, 最优阈值={optimal_threshold:.4f}")
    print(f"    检测率 (TPR)={tpr_opt*100:.1f}%, 虚警率 (FPR)={fpr_opt*100:.1f}%")
    
    return optimal_threshold, f'Youden (AUC={roc_auc:.3f})'


def learn_activation_masks(activations, all_modality_labels, K=8, fault_activations=None):
    """学习激活掩码和阈值"""
    print("\n📊 学习激活链路掩码 (汉明距离 + 阈值学习)...")
    
    M_ref_profiles = []
    
    for k in range(K):
        layer_data_k = {f'layer{i}_ffn': [] for i in range(4)}
        layer_fault_data_k = {f'layer{i}_ffn': [] for i in range(4)} if fault_activations else None
        
        # 收集正常样本
        for ep_idx, (act_data, modality_labels) in enumerate(zip(activations, all_modality_labels)):
            mask = (modality_labels == k)
            for t in range(len(mask)):
                if mask[t]:
                    for i in range(4):
                        layer_key = f'layer{i}_ffn'
                        layer_act = act_data.get(layer_key)
                        if layer_act is not None and t < len(layer_act):
                            cls_act = layer_act[t][0, 0, :]
                            layer_data_k[layer_key].append((cls_act > 0).astype(int))
        
        # 收集故障样本
        if fault_activations:
            for ep_idx, act_data in enumerate(fault_activations):
                for t in range(len(act_data.get('layer0_ffn', []))):
                    for i in range(4):
                        layer_key = f'layer{i}_ffn'
                        layer_act = act_data.get(layer_key)
                        if layer_act is not None and t < len(layer_act):
                            cls_act = layer_act[t][0, 0, :]
                            layer_fault_data_k[layer_key].append((cls_act > 0).astype(int))
        
        profile = {}
        for i in range(4):
            layer_key = f'layer{i}_ffn'
            binary_acts = layer_data_k[layer_key]
            
            if len(binary_acts) > 0:
                binary_array = np.array(binary_acts)
                M_ref = (np.mean(binary_array, axis=0) > 0.5).astype(int)
                
                # 正常样本的汉明距离
                distances_normal = [np.sum(ba != M_ref) / len(M_ref) for ba in binary_acts]
                
                # 故障样本的汉明距离
                distances_fault = []
                if layer_fault_data_k and len(layer_fault_data_k[layer_key]) > 0:
                    for ba in layer_fault_data_k[layer_key]:
                        d = np.sum(ba != M_ref) / len(M_ref)
                        distances_fault.append(d)
                
                # 阈值学习
                threshold, method = learn_thresholds_youden(distances_normal, distances_fault)
                
                profile[layer_key] = {
                    'M_ref': M_ref.tolist(),
                    'threshold': float(threshold),
                    'method': method,
                    'n_normal': len(binary_acts),
                    'n_fault': len(distances_fault),
                    'D_ham_normal': {
                        'mean': float(np.mean(distances_normal)),
                        'std': float(np.std(distances_normal)),
                        'p95': float(np.percentile(distances_normal, 95)),
                        'p99': float(np.percentile(distances_normal, 99)),
                    }
                }
                print(f"  ✓ 模态 {k} - {layer_key}: {len(binary_acts)} 正常，{len(distances_fault)} 故障，阈值={threshold:.4f} ({method})")
            else:
                profile[layer_key] = None
        
        M_ref_profiles.append(profile)
    
    return M_ref_profiles


def train_activation_module(data_dir, output_dir, config=None, fault_data_dir=None):
    """训练激活链路模块"""
    if config is None:
        config = {'K': 8, 'seed': 42}
    
    print("=" * 60)
    print("Region 3 模块 2: 激活链路训练")
    print("=" * 60)
    
    # 1. 加载数据
    trajectories, activations, fault_activations = load_activation_data(data_dir, fault_data_dir)
    
    # 2. 动作聚类
    kmeans, all_modality_labels = cluster_modalities(trajectories, K=config['K'], seed=config['seed'])
    
    # 3. 学习激活掩码和阈值
    M_ref_profiles = learn_activation_masks(
        activations, all_modality_labels, K=config['K'], fault_activations=fault_activations
    )
    
    # 4. 保存结果
    print("\n💾 保存结果...")
    os.makedirs(output_dir, exist_ok=True)
    
    import joblib
    joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_activation.pkl'))
    
    with open(os.path.join(output_dir, 'activation_masks.json'), 'w') as f:
        json.dump(M_ref_profiles, f, indent=2)
    
    summary = {
        'module': 'activation_link',
        'K': config['K'],
        'num_normal': len(trajectories),
        'num_fault': len(fault_activations) if fault_activations else 0,
        'layers': 4,
        'dim_per_layer': 512,
    }
    with open(os.path.join(output_dir, 'activation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ kmeans_activation.pkl")
    print(f"  ✓ activation_masks.json")
    print(f"  ✓ activation_summary.json")
    
    print("\n" + "=" * 60)
    print("✅ 激活链路模块训练完成!")
    print("=" * 60)
    
    return {'kmeans': kmeans, 'M_ref_profiles': M_ref_profiles}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/rta_training_v3')
    parser.add_argument('--fault_data_dir', type=str, default=None, help='故障数据目录 (可选)')
    parser.add_argument('--output_dir', type=str, default='./outputs/region3_activation')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train_activation_module(
        args.data_dir, args.output_dir,
        config={'K': args.K, 'seed': args.seed},
        fault_data_dir=args.fault_data_dir
    )
