#!/usr/bin/env python3
"""
Region 3: 多层激活链路检测器 - 学习版

通过收集正常和故障场景的激活数据，学习：
1. 各层激活的参考分布 (均值/方差)
2. 关键特征维度 (PCA)
3. 最优阈值 (在验证集上优化)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
import json

class Region3Learner:
    """Region 3 学习器"""
    
    def __init__(self, model, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations = {name: [] for name in layer_names}
        self.stats = {}
        self.thresholds = {}
        self.pca_components = None
    
    def collect_activations(self, dataloader, label: str):
        """收集激活数据"""
        self.model.eval()
        print(f'Collecting activations for {label}...')
        
        with torch.no_grad():
            for batch in dataloader:
                # 前向传播并记录激活
                _ = self.model(batch)
                # 这里需要实现钩子来捕获各层激活
                # 简化示例
                pass
        
        print(f'Collected {len(self.activations[self.layer_names[0]])} samples')
    
    def compute_statistics(self):
        """计算各层激活统计"""
        print('Computing statistics...')
        for name in self.layer_names:
            acts = np.concatenate(self.activations[name])
            self.stats[name] = {
                'mean': np.mean(acts, axis=0),
                'std': np.std(acts, axis=0) + 1e-8,
                'min': np.min(acts, axis=0),
                'max': np.max(acts, axis=0),
            }
        print('Statistics computed')
    
    def identify_key_features(self, n_components: int = 50):
        """通过 PCA 识别关键特征"""
        from sklearn.decomposition import PCA
        
        print(f'Identifying key features (n_components={n_components})...')
        
        # 合并所有层的激活
        all_acts = []
        for name in self.layer_names:
            acts = np.concatenate(self.activations[name])
            all_acts.append(acts.reshape(acts.shape[0], -1))
        
        X = np.hstack(all_acts)
        
        # PCA
        pca = PCA(n_components=n_components)
        pca.fit(X)
        
        self.pca_components = pca.components_
        explained_var = np.sum(pca.explained_variance_ratio_)
        
        print(f'PCA: {n_components} components explain {explained_var*100:.1f}% variance')
    
    def optimize_thresholds(self, normal_data, fault_data):
        """在验证集上优化阈值"""
        print('Optimizing thresholds...')
        
        best_thresholds = {}
        best_f1 = 0
        
        # 网格搜索最优阈值
        for link_th in [0.2, 0.3, 0.4]:
            for ood_th in [2.0, 3.0, 4.0]:
                for jump_th in [0.4, 0.5, 0.6]:
                    for entropy_th in [1.0, 1.3, 1.5]:
                        # 计算检测率和虚警率
                        detected = 0
                        false_alarm = 0
                        
                        for sample in fault_data:
                            score = self.compute_risk(sample, link_th, ood_th, jump_th, entropy_th)
                            if score > 0.4:
                                detected += 1
                        
                        for sample in normal_data:
                            score = self.compute_risk(sample, link_th, ood_th, jump_th, entropy_th)
                            if score > 0.4:
                                false_alarm += 1
                        
                        detection_rate = detected / len(fault_data)
                        false_alarm_rate = false_alarm / len(normal_data)
                        f1 = 2 * detection_rate * (1 - false_alarm_rate) / (detection_rate + 1 - false_alarm_rate + 1e-8)
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_thresholds = {
                                'link': link_th,
                                'ood': ood_th,
                                'jump': jump_th,
                                'entropy': entropy_th,
                                'risk': 0.4
                            }
        
        self.thresholds = best_thresholds
        print(f'Best thresholds: {best_thresholds}')
        print(f'Best F1: {best_f1:.3f}')
    
    def compute_risk(self, sample, link_th, ood_th, jump_th, entropy_th):
        """计算风险分数"""
        # 简化实现
        return 0.5
    
    def save(self, output_path: Path):
        """保存学习结果"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.savez(output_path / 'stats.npz', **self.stats)
        
        if self.pca_components is not None:
            np.save(output_path / 'pca_components.npy', self.pca_components)
        
        with open(output_path / 'thresholds.json', 'w') as f:
            json.dump(self.thresholds, f, indent=2)
        
        print(f'Saved to {output_path}')
    
    @classmethod
    def load(cls, model, layer_names, load_path: Path):
        """加载学习结果"""
        learner = cls(model, layer_names)
        
        stats = np.load(load_path / 'stats.npz', allow_pickle=True)
        learner.stats = {k: stats[k] for k in stats.files}
        
        if (load_path / 'pca_components.npy').exists():
            learner.pca_components = np.load(load_path / 'pca_components.npy')
        
        with open(load_path / 'thresholds.json') as f:
            learner.thresholds = json.load(f)
        
        print(f'Loaded from {load_path}')
        return learner


if __name__ == '__main__':
    print('Region 3 Learner - 需要真实 OpenVLA 模型和数据')
    print('当前为示例代码，需要集成到完整流程中')
