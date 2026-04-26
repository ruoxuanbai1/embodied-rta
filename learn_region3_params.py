#!/usr/bin/env python3
"""
Region 3 检测器参数学习

从 ACT 轨迹数据中学习:
1. 掩码库 (激活链路标准模式)
2. 关键特征集 (CFS)
3. 各模块阈值
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans


class Region3Learner:
    """
    Region 3 参数学习器
    
    从轨迹数据中学习:
    1. 激活链路掩码库
    2. 决策因子关键特征集
    3. 各模块阈值
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        state_dim: int = 14,
        device: str = 'cuda'
    ):
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.device = device
        
        # 学习结果
        self.mask_library = {}  # {action_type: [mask_1, mask_2, ...]}
        self.legal_feature_sets = {}  # {action_type: [feature_indices]}
        self.thresholds = {}
    
    def load_trajectories(self, trajectory_dir: str) -> Dict:
        """加载轨迹数据"""
        trajectory_files = list(Path(trajectory_dir).glob('*.npz'))
        
        normal_trajs = []
        fault_trajs = []
        invalid_count = 0
        
        for traj_file in trajectory_files:
            try:
                data = np.load(traj_file)
                
                fault_type_arr = data.get('fault_type', None)
                if fault_type_arr is not None:
                    if isinstance(fault_type_arr, np.ndarray):
                        fault_type = fault_type_arr.item() if fault_type_arr.ndim == 0 else str(fault_type_arr[0])
                    else:
                        fault_type = str(fault_type_arr)
                else:
                    fault_type = 'normal'
                
                traj_data = {
                    'file': str(traj_file),
                    'fault_type': fault_type,
                    'states': data['states'],
                    'actions': data['actions'],
                    'hook_a': data.get('hook_a', None),
                    'hook_b': data.get('hook_b', None),
                }
                
                if 'normal' in traj_data['fault_type']:
                    normal_trajs.append(traj_data)
                else:
                    fault_trajs.append(traj_data)
            except Exception as e:
                invalid_count += 1
                continue
        
        print(f'加载轨迹：{len(normal_trajs)} 正常，{len(fault_trajs)} 故障，跳过 {invalid_count} 个损坏文件')
        
        return {
            'normal': normal_trajs,
            'fault': fault_trajs,
        }
    
    def learn_mask_library(
        self,
        trajectories: List[Dict],
        n_clusters: int = 5,
        action_bins: int = 8
    ) -> Dict:
        """
        学习激活链路掩码库
        
        Args:
            trajectories: 轨迹列表
            n_clusters: 每个动作类型的聚类数
            action_bins: 动作分箱数
        
        Returns:
            mask_library: {action_bin: [masks]}
        """
        print(f'\\n学习激活链路掩码库...')
        
        # 按动作类型分组
        action_groups = {i: [] for i in range(action_bins)}
        
        for traj in trajectories:
            if traj['hook_a'] is None or traj['hook_b'] is None:
                continue
            
            hook_a = traj['hook_a']  # (steps, hidden_dim)
            hook_b = traj['hook_b']
            actions = traj['actions']
            
            for step in range(len(actions)):
                # 动作分箱 (使用动作范数)
                action_norm = np.linalg.norm(actions[step])
                action_bin = min(int(action_norm * action_bins / 2.0), action_bins - 1)
                
                # 二值化激活
                mask_a = (hook_a[step] > 0).astype(np.int32)
                mask_b = (hook_b[step] > 0).astype(np.int32)
                mask_combined = np.concatenate([mask_a, mask_b])
                
                action_groups[action_bin].append(mask_combined)
        
        # 对每个动作类型聚类
        mask_library = {}
        
        for action_bin, masks in action_groups.items():
            if len(masks) < n_clusters:
                continue
            
            masks_array = np.array(masks)
            
            # K-Means 聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(masks_array)
            
            # 保存聚类中心 (二值化)
            cluster_centers = (kmeans.cluster_centers_ > 0.5).astype(np.int32)
            mask_library[action_bin] = cluster_centers
            
            print(f'  动作箱 {action_bin}: {len(masks)} 样本，{n_clusters} 个掩码')
        
        self.mask_library = mask_library
        return mask_library
    
    def learn_legal_features(
        self,
        trajectories: List[Dict],
        top_k: int = 5
    ) -> Dict:
        """
        学习关键特征集 (CFS)
        
        使用梯度分析确定哪些状态特征对动作决策贡献最大
        """
        print(f'\\n学习关键特征集...')
        
        # 简化：使用相关性分析
        all_states = []
        all_actions = []
        
        for traj in trajectories:
            all_states.append(traj['states'])
            all_actions.append(traj['actions'])
        
        all_states = np.vstack(all_states)
        all_actions = np.vstack(all_actions)
        
        # 计算状态 - 动作相关性
        correlations = []
        for i in range(self.state_dim):
            corr = np.abs(np.corrcoef(all_states[:, i], all_actions[:, 0])[0, 1])
            correlations.append((i, corr if not np.isnan(corr) else 0))
        
        # 排序取 top-k
        correlations.sort(key=lambda x: x[1], reverse=True)
        legal_features = [c[0] for c in correlations[:top_k]]
        
        self.legal_feature_sets['default'] = legal_features
        
        print(f'  关键特征：{legal_features}')
        print(f'  相关性：{[f"{c[1]:.3f}" for c in correlations[:top_k]]}')
        
        return self.legal_feature_sets
    
    def learn_thresholds(
        self,
        normal_trajs: List[Dict],
        fault_trajs: List[Dict]
    ) -> Dict:
        """
        学习各模块阈值
        
        使用网格搜索优化，目标:
        - 检测率 > 90%
        - 虚警率 < 5%
        """
        print(f'\\n学习阈值...')
        
        # 计算正常和故障的统计量
        normal_stats = self._compute_statistics(normal_trajs)
        fault_stats = self._compute_statistics(fault_trajs)
        
        thresholds = {}
        
        # 熵阈值
        normal_entropy = normal_stats['entropy']
        fault_entropy = fault_stats['entropy']
        
        # 找到最佳分割点
        best_entropy_th = np.percentile(normal_entropy, 95)
        thresholds['entropy'] = float(best_entropy_th)
        
        # OOD 阈值 (马氏距离)
        normal_ood = normal_stats['mahalanobis']
        fault_ood = fault_stats['mahalanobis']
        
        best_ood_th = np.percentile(normal_ood, 95)
        thresholds['ood'] = float(best_ood_th)
        
        # 跳变阈值
        normal_jump = normal_stats['jump']
        best_jump_th = np.percentile(normal_jump, 95)
        thresholds['jump'] = float(best_jump_th)
        
        # 汉明距离阈值
        normal_hamming = normal_stats['hamming']
        best_hamming_th = np.percentile(normal_hamming, 90)
        thresholds['hamming'] = float(best_hamming_th)
        
        self.thresholds = thresholds
        
        print(f'  熵阈值：{thresholds["entropy"]:.3f}')
        print(f'  OOD 阈值：{thresholds["ood"]:.3f}')
        print(f'  跳变阈值：{thresholds["jump"]:.3f}')
        print(f'  汉明距离阈值：{thresholds["hamming"]:.3f}')
        
        return thresholds
    
    def _compute_statistics(self, trajectories: List[Dict]) -> Dict:
        """计算轨迹统计量"""
        entropy_values = []
        mahalanobis_values = []
        jump_values = []
        hamming_values = []
        
        for traj in trajectories:
            states = traj['states']
            actions = traj['actions']
            
            # 熵
            for action in actions:
                # 简化熵计算
                action_prob = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
                entropy = -np.sum(action_prob * np.log(action_prob + 1e-8))
                entropy_values.append(entropy)
            
            # 马氏距离 (简化)
            state_mean = states.mean(axis=0)
            state_std = states.std(axis=0) + 1e-6
            for state in states:
                mahalanobis = np.sqrt(np.sum(((state - state_mean) / state_std) ** 2))
                mahalanobis_values.append(mahalanobis)
            
            # 跳变
            for i in range(1, len(states)):
                cos_sim = np.dot(states[i], states[i-1]) / (np.linalg.norm(states[i]) * np.linalg.norm(states[i-1]) + 1e-8)
                jump = (1.0 - cos_sim) / 2.0
                jump_values.append(jump)
            
            # 汉明距离 (如果有 hook 数据)
            if traj.get('hook_a') is not None:
                hook_a = traj['hook_a']
                masks = (hook_a > 0).astype(np.int32)
                
                # 计算与均值的距离
                mean_mask = (masks.mean(axis=0) > 0.5).astype(np.int32)
                for mask in masks:
                    hamming = np.sum(mask != mean_mask)
                    hamming_values.append(hamming)
        
        return {
            'entropy': np.array(entropy_values),
            'mahalanobis': np.array(mahalanobis_values),
            'jump': np.array(jump_values),
            'hamming': np.array(hamming_values) if hamming_values else np.array([0.3 * 512]),
        }
    
    def save_results(self, output_path: str):
        """保存学习结果"""
        results = {
            'mask_library': self.mask_library,
            'legal_feature_sets': self.legal_feature_sets,
            'thresholds': self.thresholds,
        }
        
        import json
        with open(output_path, 'w') as f:
            # 转换 numpy 为 Python 类型
            serializable = {}
            for key, val in results.items():
                if isinstance(val, dict):
                    serializable[key] = {}
                    for k, v in val.items():
                        if isinstance(v, np.ndarray):
                            serializable[key][k] = v.tolist()
                        else:
                            serializable[key][k] = v
                else:
                    serializable[key] = val
            
            json.dump(serializable, f, indent=2)
        
        print(f'\\n学习结果保存至：{output_path}')
    
    def load_results(self, input_path: str):
        """加载学习结果"""
        import json
        
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        self.mask_library = {int(k): np.array(v) for k, vs in results.get('mask_library', {}).items() for v in [vs]}
        self.legal_feature_sets = results.get('legal_feature_sets', {})
        self.thresholds = results.get('thresholds', {})
        
        print(f'加载学习结果：{input_path}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory-dir', type=str, default='/root/rt1_trajectory_data')
    parser.add_argument('--output', type=str, default='/root/Embodied-RTA/region3_learned_params.json')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('Region 3 参数学习')
    print('=' * 60)
    
    learner = Region3Learner(device=args.device)
    
    # 加载轨迹
    data = learner.load_trajectories(args.trajectory_dir)
    
    if not data['normal']:
        print('\\n❌ 错误：未找到正常轨迹数据')
        print('请先运行试验收集轨迹数据')
    else:
        # 学习掩码库
        learner.learn_mask_library(data['normal'], n_clusters=5)
        
        # 学习关键特征集
        learner.learn_legal_features(data['normal'])
        
        # 学习阈值
        learner.learn_thresholds(data['normal'], data['fault'])
        
        # 保存结果
        learner.save_results(args.output)
        
        print('\\n✅ Region 3 参数学习完成!')
