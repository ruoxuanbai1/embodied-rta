#!/usr/bin/env python3
"""
Region 3 参数学习 - 快速版

从 ACT 轨迹数据中学习:
1. 掩码库 (激活链路标准模式)
2. 关键特征集 (特征敏感度)
3. 阈值 (熵+OOD+ 跳变 + 汉明距离)
"""

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime


def load_valid_trajectories(trajectory_dir):
    """加载有效轨迹"""
    trajectory_files = list(Path(trajectory_dir).glob('*.npz'))
    normal_trajs = []
    fault_trajs = []
    invalid = 0
    
    for traj_file in trajectory_files:
        try:
            data = np.load(traj_file)
            fault_type = 'normal'
            if 'fault_type' in data:
                ft = data['fault_type']
                if isinstance(ft, np.ndarray):
                    fault_type = ft.item() if ft.ndim == 0 else str(ft[0])
                else:
                    fault_type = str(ft)
            
            traj = {
                'file': str(traj_file),
                'fault_type': fault_type,
                'states': data['states'],
                'actions': data['actions'],
                'hook_a': data.get('hook_a', None),
                'hook_b': data.get('hook_b', None),
            }
            
            if 'normal' in fault_type:
                normal_trajs.append(traj)
            else:
                fault_trajs.append(traj)
        except:
            invalid += 1
    
    print(f'加载轨迹：{len(normal_trajs)} 正常，{len(fault_trajs)} 故障，跳过 {invalid} 个损坏文件')
    return {'normal': normal_trajs, 'fault': fault_trajs}


def learn_mask_library(trajectories, n_clusters=5):
    """学习激活链路掩码库"""
    print('\n[1/3] 学习掩码库...')
    
    # 收集所有激活
    all_masks = []
    for traj in trajectories:
        if traj['hook_a'] is not None:
            hook_a = traj['hook_a']
            # 二值化
            masks = (hook_a > 0).astype(np.int32)
            all_masks.append(masks)
    
    if not all_masks:
        print('  ⚠️ 无 hook 数据，使用默认掩码库')
        return {'default': np.random.randint(0, 2, (5, 512)).tolist()}
    
    all_masks = np.vstack(all_masks)
    print(f'  收集 {len(all_masks)} 个激活掩码')
    
    # 简化：使用 K-Means 聚类
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(all_masks)
    
    # 保存聚类中心 (二值化)
    cluster_centers = (kmeans.cluster_centers_ > 0.5).astype(np.int32).tolist()
    
    print(f'  聚类得到 {n_clusters} 个标准掩码')
    return {'default': cluster_centers}


def learn_legal_features(trajectories, top_k=5):
    """学习关键特征集"""
    print('\n[2/3] 学习关键特征集...')
    
    all_states = []
    all_actions = []
    
    for traj in trajectories:
        all_states.append(traj['states'])
        all_actions.append(traj['actions'])
    
    all_states = np.vstack(all_states)
    all_actions = np.vstack(all_actions)
    
    # 计算相关性
    correlations = []
    for i in range(all_states.shape[1]):
        try:
            corr = np.abs(np.corrcoef(all_states[:, i], all_actions[:, 0])[0, 1])
            correlations.append((i, corr if not np.isnan(corr) else 0))
        except:
            correlations.append((i, 0))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    legal_features = [int(c[0]) for c in correlations[:top_k]]
    
    print(f'  关键特征：{legal_features}')
    print(f'  相关性：{[f"{c[1]:.3f}" for c in correlations[:top_k]]}')
    
    return {'default': legal_features}


def learn_thresholds(normal_trajs, fault_trajs):
    """学习各模块阈值"""
    print('\n[3/3] 学习阈值...')
    
    # 计算统计量
    def compute_stats(trajs):
        entropy_vals = []
        ood_vals = []
        jump_vals = []
        hamming_vals = []
        
        for traj in trajs:
            actions = traj['actions']
            states = traj['states']
            
            # 熵
            for action in actions:
                prob = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
                entropy = -np.sum(prob * np.log(prob + 1e-8))
                entropy_vals.append(entropy)
            
            # OOD (马氏距离简化)
            state_mean = states.mean(axis=0)
            state_std = states.std(axis=0) + 1e-6
            for state in states:
                mahalanobis = np.sqrt(np.sum(((state - state_mean) / state_std) ** 2))
                ood_vals.append(mahalanobis)
            
            # 跳变
            for i in range(1, len(states)):
                cos_sim = np.dot(states[i], states[i-1]) / (np.linalg.norm(states[i]) * np.linalg.norm(states[i-1]) + 1e-8)
                jump = (1.0 - cos_sim) / 2.0
                jump_vals.append(jump)
            
            # 汉明距离
            if traj.get('hook_a') is not None:
                masks = (traj['hook_a'] > 0).astype(np.int32)
                mean_mask = (masks.mean(axis=0) > 0.5).astype(np.int32)
                for mask in masks:
                    hamming = np.sum(mask != mean_mask)
                    hamming_vals.append(hamming)
        
        return {
            'entropy': np.array(entropy_vals),
            'ood': np.array(ood_vals),
            'jump': np.array(jump_vals),
            'hamming': np.array(hamming_vals) if hamming_vals else np.array([0.3 * 512]),
        }
    
    normal_stats = compute_stats(normal_trajs)
    fault_stats = compute_stats(fault_trajs) if fault_trajs else normal_stats
    
    # 计算阈值 (95 百分位)
    thresholds = {
        'entropy': float(np.percentile(normal_stats['entropy'], 95)),
        'ood': float(np.percentile(normal_stats['ood'], 95)),
        'jump': float(np.percentile(normal_stats['jump'], 95)),
        'hamming': float(np.percentile(normal_stats['hamming'], 90)),
    }
    
    print(f'  熵阈值：{thresholds["entropy"]:.3f}')
    print(f'  OOD 阈值：{thresholds["ood"]:.3f}')
    print(f'  跳变阈值：{thresholds["jump"]:.3f}')
    print(f'  汉明距离阈值：{thresholds["hamming"]:.1f}')
    
    return thresholds


if __name__ == '__main__':
    print('=' * 60)
    print('Region 3 参数学习 (快速版)')
    print('=' * 60)
    
    # 加载轨迹
    data = load_valid_trajectories('/root/rt1_trajectory_data')
    
    if not data['normal']:
        print('\n❌ 错误：未找到正常轨迹')
    else:
        # 学习掩码库
        mask_library = learn_mask_library(data['normal'])
        
        # 学习特征集
        legal_features = learn_legal_features(data['normal'])
        
        # 学习阈值
        thresholds = learn_thresholds(data['normal'], data['fault'])
        
        # 保存结果
        results = {
            'mask_library': mask_library,
            'legal_feature_sets': legal_features,
            'thresholds': thresholds,
        }
        
        output_path = '/root/Embodied-RTA/region3_learned_params.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\n✅ Region 3 参数学习完成!')
        print(f'保存至：{output_path}')
        print(f'时间：{datetime.now().isoformat()}')
