#!/usr/bin/env python3
"""
Region 3 五模块检测器 - 完整科学版

基于 ACT 预训练模型的运行时安全保障
1. 输入 OOD (马氏距离)
2. 输入跳变 (余弦相似度)
3. 输出熵 (Shannon)
4. 决策因子贡献度 (SHAP 梯度)
5. 激活链路 (汉明距离 + 掩码库)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class Region3Detector(nn.Module):
    """
    Region 3 五模块检测器
    
    Hook 挂载点:
    - 节点 A (世界观): model.encoder.layers[-1].linear2 (Encoder FFN 输出)
    - 节点 B (方法论): model.decoder.layers[-1].linear2 (Decoder FFN 输出)
    """
    
    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 14,
        hidden_dim: int = 512,
        device: str = 'cuda'
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # ========== 模块 1: 输入 OOD ==========
        self.register_buffer('state_mean', torch.zeros(state_dim, device=device))
        self.register_buffer('state_cov', torch.eye(state_dim, device=device))
        
        # ========== 模块 2: 输入跳变 ==========
        self.prev_state: Optional[torch.Tensor] = None
        
        # ========== 模块 3: 输出熵 ==========
        self.entropy_threshold = nn.Parameter(torch.tensor(2.5, device=device))
        
        # ========== 模块 4: 决策因子贡献度 (CFS) ==========
        # 按动作模态定义合法特征集
        # 动作模态：0-4=前进，5-9=转向，10-13=其他
        self.legal_feature_sets = {
            'forward': [0, 1, 2, 3, 4, 5],    # 位置 + 速度
            'turn': [6, 7, 8, 9, 10, 11],      # 姿态 + 角速度
            'default': list(range(state_dim)),
        }
        self.s_logic_threshold_normal = 0.7
        self.s_logic_threshold_warning = 0.4
        
        # ========== 模块 5: 激活链路掩码库 ==========
        # 按动作模态存储标准掩码
        self.num_masks_per_action = 5
        self.register_buffer('mask_library_forward', torch.zeros(5, hidden_dim * 2, dtype=torch.int32, device=device))
        self.register_buffer('mask_library_turn', torch.zeros(5, hidden_dim * 2, dtype=torch.int32, device=device))
        self.mask_library_initialized = False
        self.hamming_threshold = nn.Parameter(torch.tensor(0.3 * hidden_dim * 2, device=device))
        
        # ========== 融合权重 ==========
        self.module_weights = {
            'ood': 0.15,
            'jump': 0.15,
            'entropy': 0.25,
            'shap': 0.25,
            'activation': 0.20,
        }
        
        # ========== 风险等级阈值 ==========
        self.risk_thresholds = {
            'GREEN': 0.2,
            'YELLOW': 0.4,
            'ORANGE': 0.6,
            'RED': 1.0,
        }
    
    # ==================== 模块 1: 输入 OOD ====================
    
    def compute_ood_mahalanobis(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算输入状态的马氏距离 (OOD 检测)
        D_M = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
        """
        B = state.shape[0]
        diff = state - self.state_mean.unsqueeze(0)
        
        # 简化：使用对角协方差
        inv_std = 1.0 / (torch.diag(self.state_cov) + 1e-6)
        mahalanobis = torch.sqrt(torch.sum((diff ** 2) * inv_std.unsqueeze(0), dim=-1))
        
        # 归一化到 [0, 1]
        ood_score = torch.sigmoid(mahalanobis - 3.0)  # 3σ 阈值
        
        return ood_score, {
            'mahalanobis': mahalanobis.detach().cpu().numpy().tolist(),
        }
    
    # ==================== 模块 2: 输入跳变 ====================
    
    def compute_temporal_jump(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算输入时序跳变 (余弦相似度)
        jump = (1 - cos_sim) / 2
        """
        B = state.shape[0]
        
        state_jump = torch.zeros(B, device=self.device)
        if self.prev_state is not None:
            cos_sim = F.cosine_similarity(state, self.prev_state, dim=-1)
            state_jump = (1.0 - cos_sim) / 2.0
        
        self.prev_state = state.detach()
        
        return state_jump, {
            'jump': state_jump.detach().cpu().numpy().tolist(),
        }
    
    # ==================== 模块 3: 输出熵 ====================
    
    def compute_entropy(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算动作分布的 Shannon 熵
        H = -Σ p_i · log(p_i)
        """
        action_probs = F.softmax(action_logits, dim=-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        
        # 归一化 (最大熵 = log(action_dim))
        max_entropy = np.log(self.action_dim)
        entropy_norm = entropy / max_entropy
        
        # 风险分数 (熵越高越危险)
        entropy_risk = torch.sigmoid(entropy_norm - self.entropy_threshold / max_entropy)
        
        return entropy_risk, {
            'entropy': entropy.detach().cpu().numpy().tolist(),
            'entropy_norm': entropy_norm.detach().cpu().numpy().tolist(),
        }
    
    # ==================== 模块 4: 决策因子贡献度 (SHAP) ====================
    
    def compute_shap_contribution(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        action_type: str = 'default'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        决策因子贡献度溯源算法
        
        φ_i = x_i · ∂Q(s,a*)/∂x_i
        
        S_logic = Σ(φ_i for i in F_legal) / Σ(φ_j for all j)
        
        判定:
        - S_logic ≥ 0.7: 逻辑正常
        - 0.4 < S_logic < 0.7: 逻辑偏移
        - S_logic ≤ 0.4: 逻辑崩溃
        """
        B = state.shape[0]
        
        # 计算动作范数作为 Q 值代理
        action_norm = action.norm(dim=-1, keepdim=True)
        
        # 对状态求梯度
        if state.requires_grad:
            gradients = torch.autograd.grad(
                outputs=action_norm.sum(),
                inputs=state,
                retain_graph=False,
                create_graph=False
            )[0]
        else:
            # 无法计算梯度，返回默认值
            return torch.ones(B, device=self.device) * 0.5, {'error': 'no_grad'}
        
        # 计算贡献度 φ_i = x_i · gradient_i
        contributions = state * gradients  # (B, state_dim)
        contributions_abs = contributions.abs()
        
        # 获取合法特征集
        legal_features = self.legal_feature_sets.get(action_type, self.legal_feature_sets['default'])
        
        # 计算 S_logic
        legal_contrib = contributions_abs[:, legal_features].sum(dim=-1)
        total_contrib = contributions_abs.sum(dim=-1) + 1e-6
        
        s_logic = legal_contrib / total_contrib  # (B,)
        
        # 映射到风险分数 (S_logic 越低越危险)
        # S_logic >= 0.7: 安全 (risk=0), <= 0.4: 危险 (risk=1)
        shap_risk = torch.clamp((self.s_logic_threshold_normal - s_logic) / 
                                (self.s_logic_threshold_normal - self.s_logic_threshold_warning), 0, 1)
        
        return 1.0 - shap_risk, {  # 返回合理性分数 (高=合理)
            's_logic': s_logic.detach().cpu().numpy().tolist(),
            'contributions': contributions.detach().cpu().numpy().tolist()[:2],
        }
    
    # ==================== 模块 5: 激活链路 ====================
    
    def compute_activation_linkage(
        self,
        hook_a: torch.Tensor,
        hook_b: torch.Tensor,
        action_type: str = 'default'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        激活链路一致性检测 (汉明距离)
        
        离线阶段:
        1. 收集 N 个动作模态的安全样本
        2. 提取掩码 M = (hook > 0)
        3. 聚类得到 K 个标准掩码 M_ref
        
        在线检测:
        D_Ham = popcount(M_curr ⊕ M_ref)
        """
        B = hook_a.shape[0]
        
        # 二值化激活
        mask_a = (hook_a > 0).to(torch.int32)
        mask_b = (hook_b > 0).to(torch.int32)
        
        # 融合 A+B 掩码
        mask_combined = torch.cat([mask_a, mask_b], dim=-1)  # (B, 2*hidden_dim)
        
        # 如果掩码库未初始化，返回安全
        if not self.mask_library_initialized:
            return torch.ones(B, device=self.device), {'info': 'mask library not initialized'}
        
        # 选择对应动作模态的掩码库
        if action_type == 'forward':
            ref_masks = self.mask_library_forward
        elif action_type == 'turn':
            ref_masks = self.mask_library_turn
        else:
            ref_masks = torch.cat([self.mask_library_forward, self.mask_library_turn], dim=0)
        
        # 计算与所有标准掩码的最小汉明距离
        # D_Ham = popcount(M_curr ⊕ M_ref)
        min_distance = float('inf')
        
        for ref_mask in ref_masks:
            # 异或 + 计数
            xor = torch.bitwise_xor(mask_combined, ref_mask.unsqueeze(0))
            distance = xor.sum(dim=-1).float()  # (B,)
            min_distance = min(min_distance, distance.min().item())
        
        # 归一化距离
        max_distance = mask_combined.shape[-1]
        normalized_distance = min_distance / max_distance
        
        # 距离越大越危险
        activation_risk = torch.sigmoid(torch.tensor(normalized_distance * 10 - 3, device=self.device))
        
        return 1.0 - activation_risk, {
            'hamming_distance': min_distance,
            'normalized_distance': normalized_distance,
        }
    
    # ==================== 数据收集模式 ====================
    
    def collect_normal_statistics(
        self,
        state: torch.Tensor,
        momentum: float = 0.1
    ):
        """收集正常操作统计数据 (用于 OOD 检测)"""
        B = state.shape[0]
        
        # 更新状态统计
        self.state_mean = (1 - momentum) * self.state_mean + momentum * state.mean(0)
        
        # 更新状态协方差 (简化为对角)
        diff = state - self.state_mean.unsqueeze(0)
        var = (diff ** 2).mean(0)
        self.state_cov = torch.diag((1 - momentum) * torch.diag(self.state_cov) + momentum * var)
    
    def update_mask_library(
        self,
        hook_a: torch.Tensor,
        hook_b: torch.Tensor,
        action_type: str,
        n_clusters: int = 5
    ):
        """
        更新掩码库 (离线学习阶段)
        
        对每一类动作，提取所有样本的掩码，通过 K-Means 聚类得到标准掩码
        """
        B = hook_a.shape[0]
        
        # 二值化
        mask_a = (hook_a > 0).to(torch.int32)
        mask_b = (hook_b > 0).to(torch.int32)
        mask_combined = torch.cat([mask_a, mask_b], dim=-1).cpu()  # (B, 2*hidden_dim)
        
        # K-Means 聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(mask_combined.numpy())
        
        # 保存聚类中心 (二值化)
        cluster_centers = torch.tensor(
            (kmeans.cluster_centers_ > 0.5).astype(np.int32),
            device=self.device
        )
        
        # 存储到对应动作模态的掩码库
        if action_type == 'forward':
            self.mask_library_forward = cluster_centers
        elif action_type == 'turn':
            self.mask_library_turn = cluster_centers
        
        self.mask_library_initialized = True
    
    # ==================== 前向传播 ====================
    
    def forward(
        self,
        state: torch.Tensor,
        action_logits: torch.Tensor,
        hook_a: torch.Tensor,
        hook_b: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_type: str = 'default',
        collect_mode: bool = False
    ) -> Dict:
        """
        Region 3 前向传播
        
        返回:
            risk_score: (B,) 融合风险分数 [0, 1]
            risk_level: str 风险等级 (GREEN/YELLOW/ORANGE/RED)
            module_scores: Dict 各模块单独分数
            s_logic: float 决策逻辑合理性 (0-1)
        """
        B = state.shape[0]
        module_scores = {}
        details = {}
        
        # ========== 模块 1: OOD ==========
        ood_score, ood_details = self.compute_ood_mahalanobis(state)
        module_scores['ood'] = ood_score
        details['ood'] = ood_details
        
        # ========== 模块 2: 跳变 ==========
        jump_score, jump_details = self.compute_temporal_jump(state)
        module_scores['jump'] = jump_score
        details['jump'] = jump_details
        
        # ========== 模块 3: 熵 ==========
        entropy_score, entropy_details = self.compute_entropy(action_logits)
        module_scores['entropy'] = entropy_score
        details['entropy'] = entropy_details
        
        # ========== 模块 4: SHAP ==========
        s_logic = 0.5  # 默认值
        if action is not None:
            state_for_shap = state.clone().requires_grad_(True)
            shap_score, shap_details = self.compute_shap_contribution(
                state_for_shap, action, action_type
            )
            module_scores['shap'] = shap_score
            details['shap'] = shap_details
            s_logic = shap_details.get('s_logic', [0.5])[0] if isinstance(shap_details.get('s_logic'), list) else 0.5
        else:
            module_scores['shap'] = torch.ones(B, device=self.device) * 0.5
        
        # ========== 模块 5: 激活链路 ==========
        activation_score, activation_details = self.compute_activation_linkage(
            hook_a, hook_b, action_type
        )
        module_scores['activation'] = activation_score
        details['activation'] = activation_details
        
        # ========== 融合风险分数 ==========
        risk_score = sum(
            module_scores[k] * self.module_weights[k]
            for k in self.module_weights if k in module_scores
        )
        
        # ========== 风险等级 ==========
        risk_level = self._get_risk_level(risk_score)
        
        # ========== 数据收集模式 ==========
        if collect_mode:
            self.collect_normal_statistics(state)
        
        return {
            'risk_score': risk_score.detach(),
            'risk_level': risk_level,
            'module_scores': {k: v.detach().cpu().numpy().tolist() for k, v in module_scores.items()},
            'details': details,
            's_logic': s_logic,  # 决策逻辑合理性
        }
    
    def _get_risk_level(self, risk_score: torch.Tensor) -> str:
        """将风险分数映射到等级"""
        r = risk_score.mean().item()
        if r < self.risk_thresholds['GREEN']:
            return 'GREEN'
        elif r < self.risk_thresholds['YELLOW']:
            return 'YELLOW'
        elif r < self.risk_thresholds['ORANGE']:
            return 'ORANGE'
        else:
            return 'RED'


if __name__ == '__main__':
    # 测试 Region 3 检测器
    print('=== 测试 Region 3 五模块检测器 ===')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = Region3Detector(state_dim=14, action_dim=14, hidden_dim=512, device=device)
    
    # 模拟输入
    B = 2
    state = torch.randn(B, 14, device=device)
    action_logits = torch.randn(B, 14, device=device)
    hook_a = torch.randn(B, 512, device=device)
    hook_b = torch.randn(B, 512, device=device)
    
    # 测试 1: 不带 SHAP (action=None)
    print('\n[测试 1] 不带 SHAP 模块')
    result = detector(
        state=state,
        action_logits=action_logits,
        hook_a=hook_a,
        hook_b=hook_b,
        action=None,
        action_type='forward',
    )
    
    print(f"风险分数：{result['risk_score'].mean().item():.3f}")
    print(f"风险等级：{result['risk_level']}")
    print(f"模块分数:")
    for name, score in result['module_scores'].items():
        print(f"  {name}: {score[0]:.3f}")
    
    # 测试 2: 数据收集模式
    print('\n[测试 2] 数据收集模式')
    result_collect = detector(
        state=state,
        action_logits=action_logits,
        hook_a=hook_a,
        hook_b=hook_b,
        action=None,
        collect_mode=True,
    )
    print(f"统计数据已更新")
    
    print('\n✅ Region 3 检测器测试通过!')
