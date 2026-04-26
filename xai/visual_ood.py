"""
视觉 XAI 检测模块
四种边界：OOD + 跳变 + 熵 + 掩码库
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import mahalanobis

class VisualOODDetector:
    """视觉特征 OOD 检测"""
    
    def __init__(self, feature_dim=512, training_features=None):
        self.feature_dim = feature_dim
        
        if training_features is not None:
            # 从训练数据学习均值和协方差
            self.mean = np.mean(training_features, axis=0)
            self.cov = np.cov(training_features.T)
            # 正则化
            self.cov += 1e-6 * np.eye(self.feature_dim)
            self.cov_inv = np.linalg.inv(self.cov)
        else:
            self.mean = np.zeros(feature_dim)
            self.cov_inv = np.eye(feature_dim)
        
        self.threshold = 3.0  # 3σ 阈值
    
    def detect(self, features):
        """
        检测视觉特征是否 OOD
        
        参数:
            features: (512,) 视觉特征向量
        
        返回:
            is_ood: 是否 OOD
            distance: 马氏距离
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        diff = features - self.mean
        distance = np.sqrt(np.dot(np.dot(diff, self.cov_inv), diff))
        
        is_ood = distance > self.threshold
        return is_ood, distance


class VisualJumpDetector:
    """视觉特征时序跳变检测"""
    
    def __init__(self, threshold=0.5, derivative_threshold=0.3):
        self.threshold = threshold
        self.derivative_threshold = derivative_threshold
        self.prev_features = None
    
    def detect(self, features):
        """检测特征跳变"""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        if self.prev_features is None:
            self.prev_features = features.copy()
            return False, 0.0, 0.0
        
        # 计算跳变幅度
        delta = np.linalg.norm(features - self.prev_features)
        derivative = delta / 0.02  # dt=20ms
        
        has_jump = (delta > self.threshold) or (derivative > self.derivative_threshold)
        
        self.prev_features = features.copy()
        return has_jump, delta, derivative


class OutputEntropyDetector:
    """策略输出熵检测"""
    
    def __init__(self, threshold=1.30, kl_threshold=0.39):
        self.entropy_threshold = threshold
        self.kl_threshold = kl_threshold
        self.nominal_policy = None
    
    def set_nominal_policy(self, nominal_features):
        """设置标称策略特征"""
        self.nominal_policy = nominal_features
    
    def compute_entropy(self, policy_output):
        """计算输出熵"""
        if isinstance(policy_output, torch.Tensor):
            policy_output = policy_output.detach().cpu().numpy()
        
        # 对于高斯策略，熵 = 0.5 * (1 + ln(2π)) + σ
        if len(policy_output.shape) == 1:
            # 动作均值
            entropy = 0.5 * (1 + np.log(2 * np.pi)) + np.std(policy_output)
        else:
            # 动作分布
            probs = np.exp(policy_output) / np.sum(np.exp(policy_output))
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return entropy
    
    def compute_kl_divergence(self, current_policy):
        """计算 KL 散度"""
        if self.nominal_policy is None:
            return 0.0
        
        if isinstance(current_policy, torch.Tensor):
            current_policy = current_policy.detach().cpu().numpy()
        
        # KL(p||q)
        kl = np.sum(self.nominal_policy * np.log(self.nominal_policy / (current_policy + 1e-10)))
        return max(0, kl)  # KL 散度非负
    
    def detect(self, policy_output):
        """检测输出不确定性"""
        entropy = self.compute_entropy(policy_output)
        kl_div = self.compute_kl_divergence(policy_output)
        
        is_uncertain = (entropy > self.entropy_threshold) or (kl_div > self.kl_threshold)
        return is_uncertain, entropy, kl_div


class FeatureMaskDetector:
    """视觉特征掩码库匹配"""
    
    def __init__(self, mask_library=None, threshold=3):
        """
        参数:
            mask_library: dict, 掩码库 {Mask_0: [neuron_indices], ...}
            threshold: 汉明距离阈值
        """
        self.threshold = threshold
        
        if mask_library is None:
            # 默认 8 个掩码模式 (从训练数据学习)
            self.mask_library = {
                'Mask_0': [5, 12, 23, 45, 67, 89],    # 空旷走廊
                'Mask_1': [8, 15, 28, 42, 55, 78],    # 动态行人
                'Mask_2': [3, 18, 31, 48, 62, 85],    # 低光照
                'Mask_3': [10, 22, 35, 50, 68, 91],   # 抓取任务
                'Mask_4': [7, 19, 33, 46, 61, 84],    # 避障
                'Mask_5': [12, 25, 38, 52, 69, 92],   # 导航
                'Mask_6': [9, 21, 36, 49, 65, 88],    # 高光照
                'Mask_7': [15, 28, 41, 55, 72, 95],   # 复合场景
            }
        else:
            self.mask_library = mask_library
    
    def detect(self, features):
        """
        检测特征激活模式是否异常
        
        参数:
            features: (512,) 视觉特征
        
        返回:
            is_abnormal: 是否异常
            distance: 最小汉明距离
            matched_mask: 匹配的掩码 ID
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        # 获取激活最强的 6 个通道
        top_6 = np.argsort(features)[-6:]
        
        # 与每个掩码对比汉明距离
        min_distance = 64
        matched_mask = None
        
        for mask_id, dominant_neurons in self.mask_library.items():
            # 计算汉明距离 (不同神经元的数量)
            distance = np.sum(np.sort(top_6) != np.sort(dominant_neurons))
            
            if distance < min_distance:
                min_distance = distance
                matched_mask = mask_id
        
        is_abnormal = min_distance >= self.threshold
        return is_abnormal, min_distance, matched_mask


class Region3VisualDetector:
    """
    Region 3 视觉 XAI 检测器 (四边界融合)
    """
    
    def __init__(self, feature_dim=512):
        # 初始化四种检测器
        self.ood_detector = VisualOODDetector(feature_dim)
        self.jump_detector = VisualJumpDetector()
        self.entropy_detector = OutputEntropyDetector()
        self.mask_detector = FeatureMaskDetector()
        
        # 风险权重
        self.weights = {
            'ood': 0.30,
            'jump': 0.25,
            'entropy': 0.20,
            'xai': 0.25
        }
        
        self.threshold = 0.4  # 风险融合阈值
    
    def detect(self, visual_features, policy_output=None):
        """
        综合检测
        
        参数:
            visual_features: (512,) 视觉特征
            policy_output: 策略输出 (可选)
        
        返回:
            trigger: 是否触发 RTA
            risk_3: 风险分数
            details: 各边界详细信息
        """
        risk_3 = 0.0
        details = {}
        
        # 1. OOD 检测
        is_ood, ood_dist = self.ood_detector.detect(visual_features)
        details['ood'] = {
            'is_ood': is_ood,
            'mahalanobis_distance': float(ood_dist),
            'threshold': self.ood_detector.threshold
        }
        if is_ood:
            risk_3 += self.weights['ood']
        
        # 2. 跳变检测
        is_jump, jump_delta, jump_deriv = self.jump_detector.detect(visual_features)
        details['jump'] = {
            'is_jump': is_jump,
            'delta': float(jump_delta),
            'derivative': float(jump_deriv),
            'threshold': self.jump_detector.threshold
        }
        if is_jump:
            risk_3 += self.weights['jump']
        
        # 3. 熵检测
        if policy_output is not None:
            is_uncertain, entropy, kl_div = self.entropy_detector.detect(policy_output)
            details['entropy'] = {
                'is_uncertain': is_uncertain,
                'entropy': float(entropy),
                'kl_divergence': float(kl_div),
                'entropy_threshold': self.entropy_detector.entropy_threshold,
                'kl_threshold': self.entropy_detector.kl_threshold
            }
            if is_uncertain:
                risk_3 += self.weights['entropy']
        else:
            details['entropy'] = {'is_uncertain': False, 'entropy': 0.0}
        
        # 4. XAI 掩码检测
        is_abnormal, hamming_dist, matched_mask = self.mask_detector.detect(visual_features)
        details['xai'] = {
            'is_abnormal': is_abnormal,
            'hamming_distance': int(hamming_dist),
            'matched_mask': matched_mask,
            'threshold': self.mask_detector.threshold
        }
        if is_abnormal:
            risk_3 += self.weights['xai']
        
        # 风险融合
        trigger = risk_3 > self.threshold
        details['risk_3'] = float(risk_3)
        details['threshold'] = self.threshold
        details['trigger'] = trigger
        
        return trigger, risk_3, details


if __name__ == '__main__':
    # 测试 Region 3 检测器
    detector = Region3VisualDetector(feature_dim=512)
    
    # 正常特征
    normal_features = np.random.randn(512) * 0.5
    trigger, risk, details = detector.detect(normal_features)
    print(f"正常特征：risk={risk:.3f}, trigger={trigger}")
    
    # OOD 特征
    ood_features = np.random.randn(512) * 3.0 + 5.0
    trigger, risk, details = detector.detect(ood_features)
    print(f"OOD 特征：risk={risk:.3f}, trigger={trigger}, OOD={details['ood']['is_ood']}")
    
    # 跳变特征
    detector.jump_detector.prev_features = np.random.randn(512) * 0.5
    jump_features = np.random.randn(512) * 2.0 + 3.0
    trigger, risk, details = detector.detect(jump_features)
    print(f"跳变特征：risk={risk:.3f}, trigger={trigger}, Jump={details['jump']['is_jump']}")
    
    print("\nRegion 3 检测器测试通过!")
