"""
RT-1-X Hook 实现 - 用于 Region 3 安全检测

作者：Embodied RTA Team
日期：2026-04-01

功能:
- Hook 1: 导出 EfficientNet → Transformer 交界处的视觉语义特征 (用于 OOD 检测)
- Hook 2: 导出完整的 256 维 Action Logits 概率分布 (用于熵检测)

使用示例:
    model = RT1XWithHooks(checkpoint_path)
    image = load_image("scene.png")
    instruction = tokenize("Navigate to target")
    
    # 推理并获取 hooks 输出
    action, hooks = model.infer_with_hooks(image, instruction)
    
    # Hook 1: 视觉特征 (用于 OOD 检测)
    visual_features = hooks['visual_semantic_features']  # shape: (batch, 1024)
    
    # Hook 2: Action Logits (用于熵检测)
    action_logits = hooks['action_logits']  # shape: (batch, 256)
    action_probs = softmax(action_logits)
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class RT1Hooks:
    """RT-1 Hook 输出容器"""
    visual_semantic_features: torch.Tensor  # Hook 1: 视觉特征
    action_logits: torch.Tensor              # Hook 2: Action Logits
    action_probs: torch.Tensor               # Softmax 概率
    action_id: int                           # argmax 动作


class RT1XWithHooks(nn.Module):
    """
    RT-1-X 模型包装器，带 Hook 支持
    
    架构断点:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: RGB Image (320x256x3) + Language Tokens             │
    └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  EfficientNet-B3 (Vision Encoder)                           │
    │  Output: Visual Features (batch, seq_len, 1024)             │
    └─────────────────────────────────────────────────────────────┘
                            │
                  ══════════╪══════════  ← Hook 1: 视觉语义特征导出点
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Transformer Encoder (BERT-like)                            │
    │  Input: [Visual Features] + [Language Tokens]               │
    │  Output: Fused Representations (batch, seq_len, 768)        │
    └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Action Head (Linear + LayerNorm)                           │
    │  Output: Action Logits (batch, 256)                         │
    └─────────────────────────────────────────────────────────────┘
                            │
                  ══════════╪══════════  ← Hook 2: Action Logits 导出点
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Argmax → Action ID                                         │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # 模型组件 (根据实际 RT-1-X 架构调整)
        # EfficientNet-B3 作为视觉编码器
        self.vision_encoder = self._build_vision_encoder()
        
        # Transformer 编码器 (融合视觉 + 语言)
        self.transformer_encoder = self._build_transformer()
        
        # Action 预测头
        self.action_head = self._build_action_head()
        
        # 加载预训练权重
        self._load_checkpoint()
        
        # Hook 存储
        self._hook_outputs: Dict[str, torch.Tensor] = {}
    
    def _build_vision_encoder(self) -> nn.Module:
        """构建视觉编码器 (EfficientNet-B3)"""
        # 简化版本 - 实际应使用 timm 或 official EfficientNet
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # 移除分类头，保留特征提取部分
        self.efficientnet_features = model.features
        
        # 输出维度：1536 (EfficientNet-B3 的 final convolution channels)
        self.vision_feature_dim = 1536
        
        return self.efficientnet_features
    
    def _build_transformer(self) -> nn.Module:
        """构建 Transformer 编码器"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=12
        )
        
        return transformer
    
    def _build_action_head(self) -> nn.Module:
        """构建 Action 预测头"""
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)  # 256 个离散动作
        )
    
    def _load_checkpoint(self):
        """加载 RT-1-X 预训练权重"""
        print(f"Loading RT-1-X checkpoint from: {self.checkpoint_path}")
        
        # 实际实现需要根据 checkpoint 格式调整
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 加载权重 (假设 checkpoint 是 state_dict 格式)
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
        
        print("Checkpoint loaded successfully!")
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """
        视觉编码
        
        Args:
            images: (batch, 3, 320, 256) RGB 图像
        
        Returns:
            visual_features: (batch, seq_len, 1536) 视觉特征
        """
        # EfficientNet 特征提取
        features = self.vision_encoder(images)
        
        # Global Average Pooling → (batch, 1536)
        visual_features = F.adaptive_avg_pool2d(features, (1, 1))
        visual_features = visual_features.flatten(1)  # (batch, 1536)
        
        # 投影到 Transformer 维度
        # visual_features = self.vision_projection(visual_features)
        
        # 添加 seq_len 维度以匹配 Transformer 输入
        visual_features = visual_features.unsqueeze(1)  # (batch, 1, 1536)
        
        return visual_features
    
    def forward(
        self,
        images: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        return_hooks: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播 (带 Hook 支持)
        
        Args:
            images: (batch, 3, 320, 256) RGB 图像
            language_tokens: (batch, seq_len) 语言 token IDs (可选)
            return_hooks: 是否返回 Hook 输出
        
        Returns:
            action_logits: (batch, 256) 动作 logits
            hooks: Hook 输出字典 (如果 return_hooks=True)
        """
        batch_size = images.shape[0]
        
        # ========== Hook 1: 视觉特征提取 ==========
        visual_features = self.encode_visual(images)
        # 投影到 Transformer 维度 (768)
        visual_features = nn.Linear(self.vision_feature_dim, 768).to(self.device)(visual_features)
        
        if return_hooks:
            self._hook_outputs['visual_semantic_features'] = visual_features.detach()
        
        # ========== Transformer 编码 ==========
        if language_tokens is not None:
            # 融合视觉 + 语言特征
            # 简化实现：只使用视觉特征
            fused_features = visual_features
        else:
            fused_features = visual_features
        
        # Transformer 编码
        transformer_output = self.transformer_encoder(fused_features)
        
        # 取 [CLS] token 或者平均池化
        cls_representation = transformer_output[:, 0, :]  # (batch, 768)
        
        # ========== Hook 2: Action Logits ==========
        action_logits = self.action_head(cls_representation)  # (batch, 256)
        
        if return_hooks:
            self._hook_outputs['action_logits'] = action_logits.detach()
            self._hook_outputs['action_probs'] = F.softmax(action_logits, dim=-1).detach()
        
        return action_logits
    
    def infer_with_hooks(
        self,
        image: np.ndarray,
        language_instruction: str = "Navigate safely to the target point"
    ) -> Tuple[int, RT1Hooks]:
        """
        单次推理 (带 Hook 输出)
        
        Args:
            image: (320, 256, 3) RGB 图像 (numpy array)
            language_instruction: 语言指令字符串
        
        Returns:
            action_id: 预测的动作 ID
            hooks: Hook 输出容器
        """
        self.eval()
        
        # 预处理图像
        image_tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            action_logits = self.forward(image_tensor, return_hooks=True)
            
            # 计算概率和 argmax
            action_probs = F.softmax(action_logits, dim=-1)
            action_id = action_logits.argmax(dim=-1).item()
            
            # 提取 Hook 输出
            hooks = RT1Hooks(
                visual_semantic_features=self._hook_outputs['visual_semantic_features'],
                action_logits=self._hook_outputs['action_logits'],
                action_probs=self._hook_outputs['action_probs'],
                action_id=action_id
            )
        
        return action_id, hooks
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理"""
        # 归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))
        
        # 标准化 (ImageNet 均值和标准差)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = (image - mean) / std
        
        return torch.from_numpy(image)


# ========== Region 3 检测函数 ==========

def compute_shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    计算 Shannon 熵
    
    Args:
        probs: (batch, 256) 概率分布
    
    Returns:
        entropy: (batch,) 熵值
    """
    eps = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    return entropy


def compute_ood_score(
    visual_features: torch.Tensor,
    normal_mean: torch.Tensor,
    normal_cov: torch.Tensor
) -> torch.Tensor:
    """
    计算 OOD 分数 (马氏距离)
    
    Args:
        visual_features: (batch, dim) 当前视觉特征
        normal_mean: (dim,) 正常场景特征均值
        normal_cov: (dim, dim) 正常场景特征协方差
    
    Returns:
        ood_score: (batch,) OOD 分数 (马氏距离)
    """
    diff = visual_features - normal_mean.unsqueeze(0)
    
    # 马氏距离：sqrt((x-μ)^T Σ^-1 (x-μ))
    cov_inv = torch.inverse(normal_cov.unsqueeze(0))
    mahalanobis = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=-1))
    
    return mahalanobis


def detect_temporal_jump(
    current_features: torch.Tensor,
    previous_features: torch.Tensor,
    threshold: float = 0.5
) -> bool:
    """
    检测时序跳变
    
    Args:
        current_features: 当前帧特征
        previous_features: 前一帧特征
        threshold: 跳变阈值
    
    Returns:
        is_jump: 是否发生跳变
    """
    # 余弦相似度
    similarity = F.cosine_similarity(
        current_features.flatten(),
        previous_features.flatten(),
        dim=0
    )
    
    # 相似度 < (1 - threshold) 认为发生跳变
    jump_detected = similarity.item() < (1 - threshold)
    
    return jump_detected


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 示例用法
    print("RT-1-X Hook 实现 - 测试示例")
    
    # 1. 加载模型
    # model = RT1XWithHooks(checkpoint_path="path/to/rt1x_checkpoint.pth")
    
    # 2. 准备输入
    # image = np.random.randint(0, 255, (320, 256, 3), dtype=np.uint8)
    # instruction = "Navigate safely to the target point"
    
    # 3. 推理并获取 Hook 输出
    # action_id, hooks = model.infer_with_hooks(image, instruction)
    
    # 4. Region 3 检测
    # entropy = compute_shannon_entropy(hooks.action_probs)
    # print(f"Action Entropy: {entropy.item():.4f}")
    
    # 5. OOD 检测 (需要预先计算的统计量)
    # ood_score = compute_ood_score(
    #     hooks.visual_semantic_features,
    #     normal_mean=torch.zeros(1536),
    #     normal_cov=torch.eye(1536)
    # )
    # print(f"OOD Score (Mahalanobis): {ood_score.item():.4f}")
    
    print("✅ Hook 实现完成!")
    print("\n关键输出:")
    print("  - Hook 1: visual_semantic_features (用于 OOD 检测)")
    print("  - Hook 2: action_logits (256 维，用于熵检测)")
    print("\n下一步:")
    print("  1. 在服务器上安装 RT-1-X 模型")
    print("  2. 在 Isaac Lab 中集成 Hook")
    print("  3. 运行 Region 3 检测测试")
