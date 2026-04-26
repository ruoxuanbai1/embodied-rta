"""
Region 3: Cognitive Layer - Five-Module Perception Anomaly Detector

基于 ACT 预训练模型的五模块检测架构:
1. 输入 OOD (马氏距离)
2. 输入跳变 (余弦相似度)
3. 输出熵 (Shannon)
4. 决策因子贡献度 (轻量化 SHAP/梯度溯源)
5. 激活链路 (汉明距离 + 掩码库)

Hook 挂载点:
- 节点 A (世界观): model.encoder.layers.3.linear2 (Encoder 最后层 FFN)
- 节点 B (方法论): model.decoder.layers.0.linear2 (Decoder 最后层 FFN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class Region3Detector(nn.Module):
    """
    Region 3 五模块检测器
    
    输入:
        - images: (B, 3, H, W) 视觉观测
        - state: (B, state_dim) 本体状态
        - action_logits: (B, action_dim) ACT 输出动作分布
        - hook_a: (B, dim) 节点 A 激活 (Encoder FFN 输出)
        - hook_b: (B, dim) 节点 B 激活 (Decoder FFN 输出)
    
    输出:
        - risk_score: (B,) 融合风险分数 [0, 1]
        - risk_level: str 风险等级 (GREEN/YELLOW/ORANGE/RED)
        - module_scores: Dict 各模块单独分数
    """
    
    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 14,
        hidden_dim: int = 512,  # Transformer hidden dim
        device: str = 'cuda'
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # ========== 模块 1: 输入 OOD 检测 ==========
        # 马氏距离需要正常数据的统计量
        self.register_buffer('state_mean', torch.zeros(state_dim, device=device))
        self.register_buffer('state_cov', torch.eye(state_dim, device=device))
        self.register_buffer('visual_mean', torch.zeros(512, device=device))  # ResNet18 feature dim
        self.register_buffer('visual_cov', torch.eye(512, device=device))
        
        # ========== 模块 2: 输入跳变检测 ==========
        self.prev_state: Optional[torch.Tensor] = None
        self.prev_visual: Optional[torch.Tensor] = None
        self.state_jump_buffer = []
        
        # ========== 模块 3: 输出熵检测 ==========
        # 熵阈值 (需要学习)
        self.entropy_threshold = nn.Parameter(torch.tensor(2.5))
        
        # ========== 模块 4: 决策因子贡献度 (SHAP-Lite) ==========
        # 关键特征集 (CFS) - 按动作模态定义
        self.legal_feature_sets = {
            'pitch': ['qpos_0', 'qpos_1', 'qpos_2', 'qvel_0'],  # 示例
            'roll': ['qpos_3', 'qpos_4', 'qvel_1'],
            'yaw': ['qpos_5', 'qvel_2'],
            'default': list(range(state_dim)),  # 默认全部特征
        }
        
        # ========== 模块 5: 激活链路掩码库 ==========
        # 离线学习得到的标准掩码 (每个动作模态 K 个)
        self.num_masks_per_action = 5
        self.register_buffer('mask_library', torch.zeros(
            action_dim, self.num_masks_per_action, hidden_dim, device=device
        ))
        self.mask_library_initialized = False
        
        # 汉明距离阈值 (需要学习)
        self.hamming_threshold = nn.Parameter(torch.tensor(0.3 * hidden_dim))
        
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
    
    # ==================== 模块 1: 输入 OOD 检测 ====================
    
    def compute_ood_mahalanobis(
        self,
        state: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算输入状态的马氏距离 (OOD 检测)
        
        Args:
            state: (B, state_dim) 本体状态
            visual_features: (B, 512) 视觉特征 (可选)
        
        Returns:
            ood_score: (B,) OOD 分数
            details: 详细信息
        """
        B = state.shape[0]
        
        # 状态马氏距离
        diff = state - self.state_mean.unsqueeze(0)
        # 简化：使用对角协方差
        inv_std = 1.0 / (torch.diag(self.state_cov) + 1e-6)
        mahalanobis_state = torch.sqrt(torch.sum((diff ** 2) * inv_std.unsqueeze(0), dim=-1))
        
        # 归一化到 [0, 1] (使用 sigmoid)
        ood_state = torch.sigmoid(mahalanobis_state - 3.0)  # 阈值 3σ
        
        # 视觉 OOD (如果有视觉特征)
        ood_visual = torch.zeros(B, device=self.device)
        if visual_features is not None:
            diff_vis = visual_features - self.visual_mean.unsqueeze(0)
            inv_std_vis = 1.0 / (torch.diag(self.visual_cov) + 1e-6)
            mahalanobis_vis = torch.sqrt(torch.sum((diff_vis ** 2) * inv_std_vis.unsqueeze(0), dim=-1))
            ood_visual = torch.sigmoid(mahalanobis_vis - 3.0)
        
        # 融合
        ood_score = 0.5 * ood_state + 0.5 * ood_visual
        
        return ood_score, {
            'mahalanobis_state': mahalanobis_state.detach().cpu().numpy().tolist(),
            'mahalanobis_visual': mahalanobis_vis.detach().cpu().numpy().tolist() if visual_features is not None else None,
        }
    
    # ==================== 模块 2: 输入跳变检测 ====================
    
    def compute_temporal_jump(
        self,
        state: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算输入时序跳变 (余弦相似度)
        
        Args:
            state: (B, state_dim) 当前状态
            visual_features: (B, 512) 当前视觉特征
        
        Returns:
            jump_score: (B,) 跳变分数 (0=平滑，1=跳变)
            details: 详细信息
        """
        B = state.shape[0]
        
        # 状态跳变
        state_jump = torch.zeros(B, device=self.device)
        if self.prev_state is not None:
            # 余弦相似度
            cos_sim = F.cosine_similarity(state, self.prev_state, dim=-1)
            state_jump = (1.0 - cos_sim) / 2.0  # 归一化到 [0, 1]
        
        # 视觉跳变
        visual_jump = torch.zeros(B, device=self.device)
        if self.prev_visual is not None and visual_features is not None:
            cos_sim_vis = F.cosine_similarity(visual_features, self.prev_visual, dim=-1)
            visual_jump = (1.0 - cos_sim_vis) / 2.0
        
        # 更新缓存
        self.prev_state = state.detach()
        if visual_features is not None:
            self.prev_visual = visual_features.detach()
        
        # 融合
        jump_score = 0.5 * state_jump + 0.5 * visual_jump
        
        return jump_score, {
            'state_jump': state_jump.detach().cpu().numpy().tolist(),
            'visual_jump': visual_jump.detach().cpu().numpy().tolist(),
        }
    
    # ==================== 模块 3: 输出熵检测 ====================
    
    def compute_entropy(
        self,
        action_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算动作分布的 Shannon 熵
        
        Args:
            action_logits: (B, action_dim) 动作 logits
        
        Returns:
            entropy_score: (B,) 熵分数 (归一化)
            details: 详细信息
        """
        # Softmax 概率
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Shannon 熵
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        
        # 归一化 (最大熵 = log(action_dim))
        max_entropy = np.log(self.action_dim)
        entropy_norm = entropy / max_entropy
        
        # 使用学习阈值计算风险
        entropy_risk = torch.sigmoid(entropy_norm - self.entropy_threshold / max_entropy)
        
        return entropy_risk, {
            'entropy': entropy.detach().cpu().numpy().tolist(),
            'entropy_norm': entropy_norm.detach().cpu().numpy().tolist(),
        }
    
    # ==================== 模块 4: 决策因子贡献度 (SHAP-Lite) ====================
    
    def compute_shap_contribution(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        model: Optional[nn.Module] = None,
        action_type: str = 'default'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        轻量化 SHAP - 计算状态特征对动作决策的贡献度
        
        φ_i = x_i · ∂Q(s,a*)/∂x_i
        
        Args:
            state: (B, state_dim) 输入状态 (requires_grad=True)
            action: (B, action_dim) 输出动作
            model: ACT 模型 (用于反向传播)
            action_type: 动作模态 (用于选择 CFS)
        
        Returns:
            shap_score: (B,) 逻辑合理性分数 (0=不合理，1=合理)
            details: 详细信息
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
            # 如果 state 没有 requires_grad，返回默认值
            return torch.ones(B, device=self.device), {'error': 'state does not require_grad'}
        
        # 计算贡献度 φ_i = x_i · gradient_i
        contributions = state * gradients  # (B, state_dim)
        
        # 获取合法特征集
        legal_features = self.legal_feature_sets.get(action_type, self.legal_feature_sets['default'])
        
        # 计算合法特征贡献度占比
        if isinstance(legal_features[0], int):
            # 数字索引
            legal_contrib = contributions[:, legal_features].sum(dim=-1)
        else:
            # 特征名 (需要映射到索引，这里简化处理)
            legal_contrib = contributions.sum(dim=-1)  # 简化：使用全部
        
        total_contrib = contributions.abs().sum(dim=-1)
        
        # S_logic = 合法贡献 / 总贡献
        s_logic = legal_contrib.abs() / (total_contrib + 1e-6)
        
        # 映射到风险分数 (S_logic 越低越危险)
        # S_logic >= 0.7: 安全, <= 0.4: 危险
        shap_risk = torch.clamp((0.7 - s_logic) / 0.3, 0, 1)
        
        return 1.0 - shap_risk, {  # 返回合理性分数 (高=合理)
            's_logic': s_logic.detach().cpu().numpy().tolist(),
            'contributions': contributions.detach().cpu().numpy().tolist()[:2],  # 前 2 个样本
        }
    
    # ==================== 模块 5: 激活链路检测 ====================
    
    def compute_activation_linkage(
        self,
        hook_a: torch.Tensor,
        hook_b: torch.Tensor,
        action_type: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        激活链路一致性检测 (汉明距离)
        
        Args:
            hook_a: (B, hidden_dim) 节点 A 激活
            hook_b: (B, hidden_dim) 节点 B 激活
            action_type: 动作类型索引 (用于选择掩码)
        
        Returns:
            activation_score: (B,) 一致性分数 (1=一致，0=不一致)
            details: 详细信息
        """
        B = hook_a.shape[0]
        
        # 二值化激活 (符号函数)
        mask_a = (hook_a > 0).to(torch.int32)
        mask_b = (hook_b > 0).to(torch.int32)
        
        # 融合 A+B 掩码
        mask_combined = torch.cat([mask_a, mask_b], dim=-1)  # (B, 2*hidden_dim)
        
        # 如果掩码库未初始化，使用当前激活作为参考
        if not self.mask_library_initialized:
            # 初始化掩码库
            self.mask_library = mask_combined.unsqueeze(0).unsqueeze(0)  # (1, 1, 2*hidden_dim)
            self.mask_library_initialized = True
            return torch.ones(B, device=self.device), {'info': 'mask library initialized'}
        
        # 计算与掩码库的最小汉明距离
        # D_Ham = popcount(M_curr ⊕ M_ref)
        min_distance = torch.tensor(float('inf'), device=self.device)
        
        if action_type is not None and action_type < self.mask_library.shape[0]:
            # 使用特定动作类型的掩码
            ref_masks = self.mask_library[action_type]  # (K, 2*hidden_dim)
        else:
            # 使用全部掩码
            ref_masks = self.mask_library.view(-1, self.hidden_dim * 2)
        
        for ref_mask in ref_masks:
            # 异或 + 计数
            xor = torch.bitwise_xor(mask_combined, ref_mask.unsqueeze(0))
            distance = xor.sum(dim=-1).float()  # (B,)
            min_distance = torch.min(min_distance, distance.min())
        
        # 归一化距离
        max_distance = mask_combined.shape[-1]
        normalized_distance = min_distance / max_distance
        
        # 距离越大越危险
        activation_risk = torch.sigmoid(torch.tensor(normalized_distance * 10 - 3))
        
        return 1.0 - activation_risk, {
            'hamming_distance': min_distance,
            'normalized_distance': normalized_distance,
        }
    
    # ==================== 数据收集模式 ====================
    
    def collect_normal_statistics(
        self,
        state: torch.Tensor,
        visual_features: torch.Tensor,
        momentum: float = 0.1
    ):
        """
        收集正常操作统计数据 (用于 OOD 检测)
        
        Args:
            state: (B, state_dim) 状态数据
            visual_features: (B, 512) 视觉特征
            momentum: 滑动平均动量
        """
        B = state.shape[0]
        
        # 更新状态统计
        self.state_mean = (1 - momentum) * self.state_mean + momentum * state.mean(0)
        
        # 更新状态协方差 (简化为对角)
        diff = state - self.state_mean.unsqueeze(0)
        var = (diff ** 2).mean(0)
        self.state_cov = torch.diag((1 - momentum) * torch.diag(self.state_cov) + momentum * var)
        
        # 更新视觉统计
        self.visual_mean = (1 - momentum) * self.visual_mean + momentum * visual_features.mean(0)
        diff_vis = visual_features - self.visual_mean.unsqueeze(0)
        var_vis = (diff_vis ** 2).mean(0)
        self.visual_cov = torch.diag((1 - momentum) * torch.diag(self.visual_cov) + momentum * var_vis)
    
    def update_mask_library(
        self,
        hook_a: torch.Tensor,
        hook_b: torch.Tensor,
        action_type: int,
        n_samples: int = 100
    ):
        """
        更新掩码库 (离线学习阶段)
        
        Args:
            hook_a: (B, hidden_dim) 节点 A 激活
            hook_b: (B, hidden_dim) 节点 B 激活
            action_type: 动作类型
            n_samples: 每个动作类型保存的样本数
        """
        B = hook_a.shape[0]
        
        # 二值化
        mask_a = (hook_a > 0).to(torch.int32)
        mask_b = (hook_b > 0).to(torch.int32)
        mask_combined = torch.cat([mask_a, mask_b], dim=-1)  # (B, 2*hidden_dim)
        
        # 扩展掩码库维度
        if self.mask_library.shape[0] <= action_type:
            new_size = action_type + 1
            new_library = torch.zeros(
                new_size, self.num_masks_per_action, self.hidden_dim * 2,
                device=self.device
            )
            new_library[:self.mask_library.shape[0]] = self.mask_library
            self.mask_library = new_library
        
        # 保存样本 (简单平均)
        if B >= n_samples:
            self.mask_library[action_type] = mask_combined[:n_samples].unsqueeze(0)
        else:
            # 重复填充
            repeated = mask_combined.repeat(n_samples // B + 1, 1)[:n_samples]
            self.mask_library[action_type] = repeated.unsqueeze(0)
        
        self.mask_library_initialized = True
    
    # ==================== 前向传播 ====================
    
    def forward(
        self,
        state: torch.Tensor,
        action_logits: torch.Tensor,
        hook_a: torch.Tensor,
        hook_b: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        action_type: Optional[int] = None,
        collect_mode: bool = False
    ) -> Dict:
        """
        Region 3 前向传播
        
        Args:
            state: (B, state_dim) 本体状态
            action_logits: (B, action_dim) ACT 输出动作 logits
            hook_a: (B, hidden_dim) 节点 A 激活 (Encoder FFN)
            hook_b: (B, hidden_dim) 节点 B 激活 (Decoder FFN)
            visual_features: (B, 512) 视觉特征 (可选)
            action: (B, action_dim) ACT 输出动作 (用于 SHAP)
            action_type: 动作类型索引
            collect_mode: 是否为数据收集模式
        
        Returns:
            result: Dict 包含风险分数、等级、各模块详情
        """
        B = state.shape[0]
        module_scores = {}
        details = {}
        
        # ========== 模块 1: OOD ==========
        ood_score, ood_details = self.compute_ood_mahalanobis(state, visual_features)
        module_scores['ood'] = ood_score
        details['ood'] = ood_details
        
        # ========== 模块 2: 跳变 ==========
        jump_score, jump_details = self.compute_temporal_jump(state, visual_features)
        module_scores['jump'] = jump_score
        details['jump'] = jump_details
        
        # ========== 模块 3: 熵 ==========
        entropy_score, entropy_details = self.compute_entropy(action_logits)
        module_scores['entropy'] = entropy_score
        details['entropy'] = entropy_details
        
        # ========== 模块 4: SHAP ==========
        if action is not None:
            # 创建需要梯度的 state 副本
            state_for_shap = state.clone().requires_grad_(True)
            shap_score, shap_details = self.compute_shap_contribution(
                state_for_shap, action, action_type=action_type
            )
            module_scores['shap'] = shap_score
            details['shap'] = shap_details
        else:
            module_scores['shap'] = torch.ones(B, device=self.device) * 0.5  # 默认中间值
        
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
            self.collect_normal_statistics(state, visual_features)
        
        return {
            'risk_score': risk_score.detach(),
            'risk_level': risk_level,
            'module_scores': {k: v.detach().cpu().numpy().tolist() for k, v in module_scores.items()},
            'details': details,
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
    visual_features = torch.randn(B, 512, device=device)
    action = torch.randn(B, 14, device=device)
    
    # 前向传播
    result = detector(
        state=state,
        action_logits=action_logits,
        hook_a=hook_a,
        hook_b=hook_b,
        visual_features=visual_features,
        action=action,
        action_type=0,
    )
    
    print(f"风险分数：{result['risk_score'].item():.3f}")
    print(f"风险等级：{result['risk_level']}")
    print(f"模块分数:")
    for name, score in result['module_scores'].items():
        print(f"  {name}: {score[0]:.3f}")
    
    print('\\n✅ Region 3 检测器测试通过!')
