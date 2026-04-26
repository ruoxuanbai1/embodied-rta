#!/usr/bin/env python3
"""
Region 3: 多层激活链路异常检测

基于 OpenVLA 模型的层间激活模式分析

检测方法:
1. 多层激活链路分析 (Vision→LLM→Action 层间相关性) - 权重 35%
2. OOD 检测 (马氏距离 > 3σ) - 权重 25%
3. 跳变检测 (时序差分 Δ > 0.5) - 权重 20%
4. 输出熵检测 (熵 > 1.30) - 权重 20%

综合风险分数 > 0.4 时触发保守模式
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from collections import deque
import time

class ActivationHook:
    """PyTorch 激活钩子"""
    def __init__(self, module, name):
        self.name = name
        self.activations = []
        self.handle = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            act = output[0].detach().cpu().numpy()
        else:
            act = output.detach().cpu().numpy()
        self.activations.append(act)
        if len(self.activations) > 100:
            self.activations = self.activations[-100:]
    
    def get_latest(self):
        return self.activations[-1] if self.activations else None
    
    def remove(self):
        self.handle.remove()


class Region3Detector:
    """
    Region 3: 多层激活链路检测器
    """
    def __init__(self, model=None):
        self.model = model
        self.hooks = {}
        self.ref_stats = {}  # 正常激活统计
        self.risk_history = deque(maxlen=10)
        
        # 权重配置
        self.weights = {'link': 0.35, 'ood': 0.25, 'jump': 0.20, 'entropy': 0.20}
        self.thresholds = {'link': 0.3, 'ood': 3.0, 'jump': 0.5, 'entropy': 1.30, 'risk': 0.4}
    
    def register_hooks(self, layer_names: List[str]):
        """注册激活钩子"""
        if self.model is None:
            return
        for name in layer_names:
            try:
                parts = name.split('.')
                module = self.model
                for p in parts:
                    module = getattr(module, p)
                self.hooks[name] = ActivationHook(module, name)
            except:
                pass
    
    def collect_reference(self, dataloader, n_samples=100):
        """收集正常激活参考数据"""
        print('Collecting reference activations...')
        self.model.eval()
        all_acts = {name: [] for name in self.hooks}
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_samples:
                    break
                _ = self.model(batch)
                for name, hook in self.hooks.items():
                    act = hook.get_latest()
                    if act is not None:
                        all_acts[name].append(act.flatten())
        
        # 计算统计
        for name, acts in all_acts.items():
            if acts:
                all_acts_flat = np.concatenate(acts)
                self.ref_stats[name] = {
                    'mean': np.mean(all_acts_flat),
                    'std': np.std(all_acts_flat) + 1e-8
                }
        print(f'Collected stats for {len(self.ref_stats)} layers')
    
    def compute_link_score(self) -> float:
        """计算激活链路异常分数"""
        if len(self.hooks) < 2:
            return 0.0
        
        scores = []
        hook_names = list(self.hooks.keys())
        for i in range(len(hook_names)-1):
            act1 = self.hooks[hook_names[i]].get_latest()
            act2 = self.hooks[hook_names[i+1]].get_latest()
            if act1 is not None and act2 is not None:
                flat1 = act1.flatten()[:1000]
                flat2 = act2.flatten()[:1000]
                if len(flat1) == len(flat2):
                    corr = np.corrcoef(flat1, flat2)[0,1]
                    if not np.isnan(corr):
                        scores.append(abs(corr))
        
        return np.mean(scores) if scores else 0.0
    
    def compute_ood_score(self) -> float:
        """计算 OOD 分数 (马氏距离)"""
        scores = []
        for name, hook in self.hooks.items():
            act = hook.get_latest()
            if act is not None and name in self.ref_stats:
                z = abs(act.flatten().mean() - self.ref_stats[name]['mean']) / self.ref_stats[name]['std']
                scores.append(min(z / 3.0, 1.0))
        return np.max(scores) if scores else 0.0
    
    def compute_jump_score(self) -> float:
        """计算时序跳变分数"""
        scores = []
        for hook in self.hooks.values():
            acts = hook.activations
            if len(acts) >= 2:
                diff = np.linalg.norm(acts[-1].flatten() - acts[-2].flatten())
                scores.append(min(diff / 0.5, 1.0))
        return np.max(scores) if scores else 0.0
    
    def compute_entropy_score(self, output_logits) -> float:
        """计算输出熵分数"""
        if output_logits is None:
            return 0.0
        probs = np.exp(output_logits - output_logits.max()) / np.sum(np.exp(output_logits - output_logits.max()))
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return min(max((entropy - 0.5) / (self.thresholds['entropy'] - 0.5), 0), 1.0)
    
    def detect(self, output_logits=None) -> Tuple[bool, Dict]:
        """执行异常检测"""
        link = self.compute_link_score()
        ood = self.compute_ood_score()
        jump = self.compute_jump_score()
        entropy = self.compute_entropy_score(output_logits)
        
        risk = (self.weights['link'] * link + 
                self.weights['ood'] * ood + 
                self.weights['jump'] * jump + 
                self.weights['entropy'] * entropy)
        
        self.risk_history.append(risk)
        smoothed = np.mean(self.risk_history)
        
        triggered = smoothed > self.thresholds['risk']
        
        info = {
            'risk': float(risk),
            'smoothed_risk': float(smoothed),
            'triggered': triggered,
            'link': float(link),
            'ood': float(ood),
            'jump': float(jump),
            'entropy': float(entropy)
        }
        
        return triggered, info
    
    def get_conservative_action(self, action: Dict) -> Dict:
        """获取保守模式动作"""
        return {
            'v': action.get('v', 0) * 0.4,
            'omega': action.get('omega', 0) * 0.4,
            'tau': action.get('tau', np.zeros(7)) * 0.6
        }
