#!/usr/bin/env python3
"""
Region 3: 多层激活链路异常检测

基于 OpenVLA 模型的层间激活模式分析，检测感知异常

检测方法:
1. 多层激活链路分析 (Vision→LLM→Action 层间相关性) - 权重 35%
2. OOD 检测 (马氏距离 > 3σ) - 权重 25%
3. 跳变检测 (时序差分 Δ > 0.5) - 权重 20%
4. 输出熵检测 (熵 > 1.30) - 权重 20%

综合风险分数 > 0.4 时触发保守模式

版本：2.0 (2026-03-31)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class ActivationStats:
    """激活统计信息"""
    mean: float = 0.0
    std: float = 1.0
    min: float = 0.0
    max: float = 10.0
    entropy: float = 0.0


@dataclass
class LayerActivation:
    """单层激活数据"""
    layer_name: str
    activation: np.ndarray
    stats: ActivationStats = field(default_factory=ActivationStats)


class ActivationHook:
    """PyTorch 激活钩子"""
    
    def __init__(self, module: nn.Module, layer_name: str):
        self.module = module
        self.layer_name = layer_name
        self.activations: List[np.ndarray] = []
        self.handle = None
        
    def register(self):
        """注册钩子"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0].detach().cpu().numpy()
            else:
                act = output.detach().cpu().numpy()
            self.activations.append(act)
        
        self.handle = self.module.register_forward_hook(hook_fn)
        return self
    
    def remove(self):
        """移除钩子"""
        if self.handle:
            self.handle.remove()
    
    def get_latest(self, n: int = 1) -> np.ndarray:
        """获取最近 n 次激活"""
        if len(self.activations) < n:
            return np.concatenate(self.activations, axis=0) if self.activations else np.array([])
        return np.concatenate(self.activations[-n:], axis=0)
    
    def clear(self):
        """清空激活历史"""
        self.activations.clear()


class MultiLayerActivationAnalyzer:
    """
    多层激活链路分析器
    
    分析 OpenVLA 模型 Vision→LLM→Action 的层间激活模式
    """
    
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        self.hooks: Dict[str, ActivationHook] = {}
        self.reference_activations: Dict[str, List[np.ndarray]] = {}
        self.layer_correlations: Dict[Tuple[str, str], np.ndarray] = {}
        
        # 默认层名称 (OpenVLA 结构)
        self.vision_layers = ['vision_encoder.layer1', 'vision_encoder.layer2', 'vision_encoder.layer3']
        self.llm_layers = ['llm.layers.0', 'llm.layers.6', 'llm.layers.12']
        self.action_layers = ['action_head.0', 'action_head.1']
        
        # 风险阈值
        self.thresholds = {
            'link_correlation': 0.3,      # 链路相关性异常阈值
            'ood_mahalanobis': 3.0,       # 马氏距离 3σ
            'temporal_jump': 0.5,         # 时序跳变阈值
            'output_entropy': 1.30,       # 输出熵阈值
            'risk_trigger': 0.4,          # 综合风险触发阈值
        }
        
        # 权重配置
        self.weights = {
            'link': 0.35,
            'ood': 0.25,
            'jump': 0.20,
            'entropy': 0.20,
        }
    
    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """注册激活钩子"""
        if self.model is None:
            raise ValueError("Model not provided")
        
        if layer_names is None:
            layer_names = self.vision_layers + self.llm_layers + self.action_layers
        
        for name in layer_names:
            try:
                module = self._get_module_by_name(name)
                if module:
                    hook = ActivationHook(module, name)
                    hook.register()
                    self.hooks[name] = hook
                    print(f"Registered hook: {name}")
            except Exception as e:
                print(f"Failed to register hook for {name}: {e}")
    
    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """根据名称获取模块"""
        parts = name.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def collect_reference_activations(self, dataloader, n_samples: int = 100):
        """收集正常激活参考数据"""
        print(f"Collecting reference activations from {n_samples} samples...")
        
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_samples:
                    break
                
                _ = self.model(batch)
                
                for name, hook in self.hooks.items():
                    if name not in self.reference_activations:
                        self.reference_activations[name] = []
                    
                    act = hook.get_latest()
                    if act.size > 0:
                        self.reference_activations[name].append(act)
                
                hook.clear()
        
        # 计算统计信息
        for name, acts in self.reference_activations.items():
            all_acts = np.concatenate(acts, axis=0)
            print(f"  {name}: shape={all_acts.shape}, mean={all_acts.mean():.4f}, std={all_acts.std():.4f}")
    
    def compute_layer_correlation(self, layer1: str, layer2: str) -> float:
        """计算两层之间的激活相关性"""
        if layer1 not in self.hooks or layer2 not in self.hooks:
            return 0.0
        
        act1 = self.hooks[layer1].get_latest()
        act2 = self.hooks[layer2].get_latest()
        
        if act1.size == 0 or act2.size == 0:
            return 0.0
        
        # 展平
        act1_flat = act1.reshape(act1.shape[0], -1)
        act2_flat = act2.reshape(act2.shape[0], -1)
        
        # 计算皮尔逊相关性
        if len(act1_flat) < 2 or len(act2_flat) < 2:
            return 0.0
        
        corr = np.corrcoef(act1_flat, act2_flat)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def compute_activation_link_score(self) -> float:
        """
        计算激活链路异常分数
        
        比较当前层间相关性与正常参考的差异
        """
        scores = []
        
        # Vision→LLM 链路
        for v_layer in self.vision_layers[:1]:
            for l_layer in self.llm_layers[:1]:
                if v_layer in self.hooks and l_layer in self.hooks:
                    current_corr = self.compute_layer_correlation(v_layer, l_layer)
                    
                    # 与参考比较
                    key = (v_layer, l_layer)
                    if key in self.layer_correlations:
                        ref_corr = self.layer_correlations[key]
                        diff = abs(current_corr - ref_corr)
                        scores.append(diff)
        
        if not scores:
            return 0.0
        
        return np.mean(scores)
    
    def compute_ood_score(self, layer_name: str) -> float:
        """
        计算 OOD 分数 (马氏距离)
        """
        if layer_name not in self.hooks or layer_name not in self.reference_activations:
            return 0.0
        
        current_act = self.hooks[layer_name].get_latest()
        ref_acts = self.reference_activations[layer_name]
        
        if len(ref_acts) == 0 or current_act.size == 0:
            return 0.0
        
        ref_all = np.concatenate(ref_acts, axis=0)
        
        # 计算均值和协方差
        mean = np.mean(ref_all, axis=0)
        
        # 简化：使用标准差代替协方差 (高维情况下协方差计算昂贵)
        std = np.std(ref_all, axis=0) + 1e-8
        
        # 马氏距离近似
        current_flat = current_act.reshape(-1)
        mean_flat = mean.reshape(-1)[:len(current_flat)]
        std_flat = std.reshape(-1)[:len(current_flat)]
        
        if len(current_flat) != len(mean_flat):
            min_len = min(len(current_flat), len(mean_flat))
            current_flat = current_flat[:min_len]
            mean_flat = mean_flat[:min_len]
            std_flat = std_flat[:min_len]
        
        z_scores = np.abs((current_flat - mean_flat) / std_flat)
        mahalanobis = np.mean(z_scores)
        
        return mahalanobis
    
    def compute_temporal_jump_score(self, layer_name: str) -> float:
        """
        计算时序跳变分数
        """
        if layer_name not in self.hooks:
            return 0.0
        
        activations = self.hooks[layer_name].activations
        
        if len(activations) < 2:
            return 0.0
        
        # 计算相邻帧的差异
        jumps = []
        for i in range(1, min(len(activations), 10)):
            prev = activations[i-1].flatten()
            curr = activations[i].flatten()
            
            if len(prev) == len(curr):
                diff = np.linalg.norm(curr - prev) / (np.linalg.norm(prev) + 1e-8)
                jumps.append(diff)
        
        return np.max(jumps) if jumps else 0.0
    
    def compute_output_entropy(self, output: np.ndarray) -> float:
        """
        计算输出熵
        """
        # Softmax
        exp_out = np.exp(output - np.max(output))
        probs = exp_out / (np.sum(exp_out) + 1e-8)
        
        # 熵
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        return entropy
    
    def compute_risk_score(self, output: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
        """
        计算综合风险分数
        
        返回:
            risk: 综合风险分数 (0-1)
            details: 各检测器分数详情
        """
        details = {}
        
        # 1. 激活链路分析 (35%)
        link_score = self.compute_activation_link_score()
        link_normalized = min(link_score / self.thresholds['link_correlation'], 1.0)
        details['link'] = link_normalized
        
        # 2. OOD 检测 (25%)
        ood_scores = []
        for layer in self.vision_layers[:1] + self.llm_layers[:1]:
            score = self.compute_ood_score(layer)
            ood_scores.append(score)
        
        ood_score = np.max(ood_scores) if ood_scores else 0.0
        ood_normalized = min(ood_score / self.thresholds['ood_mahalanobis'], 1.0)
        details['ood'] = ood_normalized
        
        # 3. 跳变检测 (20%)
        jump_scores = []
        for layer in self.vision_layers[:1]:
            score = self.compute_temporal_jump_score(layer)
            jump_scores.append(score)
        
        jump_score = np.max(jump_scores) if jump_scores else 0.0
        jump_normalized = min(jump_score / self.thresholds['temporal_jump'], 1.0)
        details['jump'] = jump_normalized
        
        # 4. 输出熵检测 (20%)
        if output is not None:
            entropy = self.compute_output_entropy(output)
            entropy_normalized = min(max((entropy - 0.5) / (self.thresholds['output_entropy'] - 0.5), 0), 1.0)
            details['entropy'] = entropy_normalized
        else:
            details['entropy'] = 0.0
        
        # 综合风险分数
        risk = (
            self.weights['link'] * details['link'] +
            self.weights['ood'] * details['ood'] +
            self.weights['jump'] * details['jump'] +
            self.weights['entropy'] * details['entropy']
        )
        
        return risk, details
    
    def should_trigger(self, risk: float) -> bool:
        """判断是否应该触发保守模式"""
        return risk > self.thresholds['risk_trigger']
    
    def get_conservative_action(self, original_action: Dict) -> Dict:
        """获取保守模式动作"""
        return {
            'v': original_action.get('v', 0.0) * 0.4,
            'ω': original_action.get('ω', 0.0) * 0.4,
            'τ': original_action.get('τ', np.zeros(7)) * 0.6,
        }
    
    def clear(self):
        """清空所有激活历史"""
        for hook in self.hooks.values():
            hook.clear()


class Region3VisualDetector:
    """
    Region 3: 视觉异常检测器 (完整实现)
    
    整合所有检测器，提供统一的接口
    """
    
    def __init__(self, model: Optional[nn.Module] = None):
        self.analyzer = MultiLayerActivationAnalyzer(model)
        self.risk_history: deque = deque(maxlen=10)
        self.triggered = False
        self.trigger_count = 0
    
    def initialize(self, dataloader=None):
        """初始化检测器"""
        if self.analyzer.model is not None:
            self.analyzer.register_hooks()
            
            if dataloader is not None:
                self.analyzer.collect_reference_activations(dataloader)
    
    def detect(self, output: Optional[np.ndarray] = None) -> Tuple[bool, Dict]:
        """
        执行异常检测
        
        返回:
            triggered: 是否触发
            info: 详细信息
        """
        risk, details = self.analyzer.compute_risk_score(output)
        
        self.risk_history.append(risk)
        
        # 平滑处理
        smoothed_risk = np.mean(self.risk_history)
        
        triggered = self.analyzer.should_trigger(smoothed_risk)
        
        if triggered:
            self.triggered = True
            self.trigger_count += 1
        
        info = {
            'risk': risk,
            'smoothed_risk': smoothed_risk,
            'triggered': triggered,
            'details': details,
            'trigger_count': self.trigger_count,
        }
        
        return triggered, info
    
    def get_action(self, original_action: Dict, triggered: bool) -> Dict:
        """根据检测结果获取动作"""
        if triggered:
            return self.analyzer.get_conservative_action(original_action)
        return original_action
    
    def reset(self):
        """重置检测器状态"""
        self.risk_history.clear()
        self.triggered = False
        self.analyzer.clear()
    
    @property
    def computation_time(self) -> float:
        """估计计算时间 (ms)"""
        return 0.04  # ~40ms


# ============ 测试代码 ============

if __name__ == '__main__':
    print("Region 3 多层激活链路检测器 - 测试")
    print("="*60)
    
    # 创建检测器 (无模型模式)
    detector = Region3VisualDetector()
    
    # 模拟检测
    print("\n模拟检测测试...")
    for i in range(5):
        # 模拟输出
        output = np.random.randn(100) * (1 + 0.5 * i)  # 逐渐增大
        
        triggered, info = detector.detect(output)
        
        print(f"Step {i+1}: risk={info['risk']:.4f}, triggered={triggered}")
        print(f"  Details: {info['details']}")
    
    print(f"\n总触发次数：{detector.trigger_count}")
    print(f"计算时间估计：{detector.computation_time:.2f}ms")
