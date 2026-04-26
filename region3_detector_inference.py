#!/usr/bin/env python3
"""
region3_detector_inference.py - Region 3 完整检测器推理示例

使用流程:
1. 加载训练好的模型和阈值
2. 对每一帧:
   a. 判断动作模态 (KMeans)
   b. 使用该模态的阈值
   c. 计算三个模块分数
   d. 独立预警 (任一触发 → 干预)
"""

import numpy as np
import pickle
import json
import joblib
from scipy.spatial.distance import mahalanobis


class Region3Detector:
    def __init__(self, model_dir='./outputs/region3_complete'):
        """加载训练好的模型和阈值"""
        print(f"加载 Region 3 检测器：{model_dir}")
        
        # 1. 加载 KMeans 模型 (激活链路)
        self.kmeans_activation = joblib.load(f'{model_dir}/kmeans_activation.pkl')
        
        # 2. 加载 KMeans 模型 (梯度 + OOD)
        self.kmeans_gradient_ood = joblib.load(f'{model_dir}/kmeans_gradient_ood.pkl')
        
        # 3. 加载激活链路阈值 (8 模态)
        with open(f'{model_dir}/activation_links.json', 'r') as f:
            activation_data = json.load(f)
            self.activation_thresholds = [
                profile['threshold'] for profile in activation_data
            ]
            self.activation_M_refs = [
                np.array(profile['M_ref']) for profile in activation_data
            ]
            print(f"  ✓ 激活链路：{len(self.activation_thresholds)} 模态阈值")
        
        # 4. 加载梯度贡献 F_legal (8 模态)
        with open(f'{model_dir}/F_legal_profiles.json', 'r') as f:
            gradient_data = json.load(f)
            self.F_legal_profiles = gradient_data
            print(f"  ✓ 梯度贡献：{len(gradient_data)} 模态")
        
        # 5. 加载 OOD 统计量
        with open(f'{model_dir}/ood_stats.json', 'r') as f:
            ood_data = json.load(f)
            self.ood_mu = np.array(ood_data['mu'])
            self.ood_sigma_inv = np.array(ood_data['sigma_inv'])
            self.ood_threshold = ood_data['threshold']
            print(f"  ✓ OOD 检测：阈值={self.ood_threshold:.4f}")
        
        # 固定阈值
        self.logic_threshold = 0.4  # 逻辑合理性阈值
        
        print("✅ Region 3 检测器加载完成")
    
    def predict_modality(self, action):
        """判断当前动作属于哪个模态"""
        action_array = np.array(action, dtype=np.float64).reshape(1, -1)
        # 使用 transform 计算到各聚类中心的距离，取最近的
        distances = self.kmeans_activation.transform(action_array)[0]
        return np.argmin(distances)
    
    def compute_hamming_distance(self, current_activations, M_ref):
        """
        计算汉明距离
        
        current_activations: 当前 4 层激活 (需要转换为传播链路向量)
        M_ref: 标准传播链路 (3584 维)
        """
        # 提取 4 层 cls token 激活
        cls_acts = []
        for i in range(4):
            layer_act = current_activations.get(f'layer{i}_ffn')
            if layer_act is not None:
                # 假设传入的是单帧：(102, 1, 512) → 取 cls token
                cls_act = layer_act[0, 0, :]  # (512,)
                cls_acts.append(cls_act)
        
        if len(cls_acts) != 4:
            return 0.5  # 异常时返回默认值
        
        # 构建传播链路向量
        cls_combined = np.concatenate(cls_acts)  # (2048,)
        binary_combined = (cls_combined > 0).astype(int)
        
        # 层间同步模式
        layer0 = cls_combined[0:512]
        layer1 = cls_combined[512:1024]
        layer2 = cls_combined[1024:1536]
        layer3 = cls_combined[1536:2048]
        
        sync_01 = ((layer0 > 0) & (layer1 > 0)).astype(int)
        sync_12 = ((layer1 > 0) & (layer2 > 0)).astype(int)
        sync_23 = ((layer2 > 0) & (layer3 > 0)).astype(int)
        
        # 组合
        propagation_vector = np.concatenate([
            binary_combined,
            sync_01,
            sync_12,
            sync_23,
        ])  # (3584,)
        
        # 汉明距离
        distance = np.sum(propagation_vector != M_ref) / len(M_ref)
        return distance
    
    def compute_logic_score(self, qpos, gradient, F_legal):
        """
        计算逻辑合理性分数
        
        qpos: (14,) 当前状态
        gradient: (14, 14) 梯度 ∂a/∂qpos
        F_legal: 该模态的关键特征集 (list of list)
        """
        if F_legal is None:
            return 0.5
        
        # 贡献度 φ = qpos × gradient
        contrib = qpos[None, :] * gradient  # (14, 14)
        
        total_contrib = np.abs(contrib).sum()
        
        # 合法特征贡献
        legal_contrib = 0.0
        for action_dim in range(14):
            for state_dim in F_legal[action_dim]:
                legal_contrib += np.abs(contrib[action_dim, state_dim])
        
        S_logic = legal_contrib / (total_contrib + 1e-8)
        return S_logic
    
    def compute_ood_distance(self, state):
        """计算 OOD 马氏距离"""
        return mahalanobis(state, self.ood_mu, self.ood_sigma_inv)
    
    def detect(self, action, qpos, qvel, gradient, activations):
        """
        完整检测流程
        
        输入:
        - action: (14,) 当前动作
        - qpos: (14,) 当前位置
        - qvel: (14,) 当前速度
        - gradient: (14, 14) 梯度 ∂a/∂qpos
        - activations: Dict 4 层激活
        
        输出:
        - alerts: Dict 各模块预警状态
        - intervention: bool 是否需要干预
        """
        # 步骤 1: 判断动作模态
        modality_id = self.predict_modality(action)
        
        # 步骤 2: 获取该模态的阈值和参数
        threshold_activation = self.activation_thresholds[modality_id]
        M_ref = self.activation_M_refs[modality_id]
        F_legal = self.F_legal_profiles[modality_id]
        
        # 步骤 3: 计算各模块分数
        D_ham = self.compute_hamming_distance(activations, M_ref)
        D_ood = self.compute_ood_distance(np.concatenate([qpos, qvel]))  # (28,)
        S_logic = self.compute_logic_score(qpos, gradient, F_legal)
        
        # 步骤 4: 独立预警
        alerts = {
            'modality': int(modality_id),
            'logic': {
                'score': float(S_logic),
                'threshold': self.logic_threshold,
                'triggered': bool(S_logic < self.logic_threshold),
            },
            'activation': {
                'score': float(D_ham),
                'threshold': float(threshold_activation),
                'triggered': bool(D_ham > threshold_activation),
            },
            'ood': {
                'score': float(D_ood),
                'threshold': float(self.ood_threshold),
                'triggered': bool(D_ood > self.ood_threshold),
            },
        }
        
        # 干预逻辑：OR (任一触发)
        intervention = (
            alerts['logic']['triggered'] or
            alerts['activation']['triggered'] or
            alerts['ood']['triggered']
        )
        
        return alerts, intervention


# 使用示例
if __name__ == '__main__':
    # 1. 加载检测器
    detector = Region3Detector('./outputs/region3_complete')
    
    # 2. 模拟输入 (使用 float64)
    action = np.random.randn(14).astype(np.float64)
    qpos = np.random.randn(14).astype(np.float64)
    qvel = np.random.randn(14).astype(np.float64)
    gradient = np.random.randn(14, 14).astype(np.float64)
    activations = {
        f'layer{i}_ffn': np.random.randn(102, 1, 512).astype(np.float64)
        for i in range(4)
    }
    
    # 3. 检测
    alerts, intervention = detector.detect(action, qpos, qvel, gradient, activations)
    
    # 4. 输出结果
    print("\n" + "="*60)
    print("Region 3 检测结果")
    print("="*60)
    print(f"动作模态：{alerts['modality']}")
    print(f"\n模块 1 - 逻辑合理性:")
    print(f"  S_logic = {alerts['logic']['score']:.4f}")
    print(f"  阈值 = {alerts['logic']['threshold']:.4f}")
    print(f"  预警 = {'⚠️ 是' if alerts['logic']['triggered'] else '✅ 否'}")
    
    print(f"\n模块 2 - 激活链路:")
    print(f"  D_ham = {alerts['activation']['score']:.4f}")
    print(f"  阈值 = {alerts['activation']['threshold']:.4f} (模态{alerts['modality']})")
    print(f"  预警 = {'⚠️ 是' if alerts['activation']['triggered'] else '✅ 否'}")
    
    print(f"\n模块 3 - OOD 检测:")
    print(f"  D_ood = {alerts['ood']['score']:.4f}")
    print(f"  阈值 = {alerts['ood']['threshold']:.4f}")
    print(f"  预警 = {'⚠️ 是' if alerts['ood']['triggered'] else '✅ 否'}")
    
    print(f"\n{'='*60}")
    print(f"干预 = {'⚠️ 是 (动作×40%)' if intervention else '✅ 否 (正常执行)'}")
    print(f"{'='*60}")
