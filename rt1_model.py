"""
RT-1 (Robotics Transformer) Implementation with RTA Hooks
Architecture: EfficientNet-B3 + Transformer Encoder + Action Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from typing import Dict, List, Tuple, Optional
import numpy as np


class RT1WithHooks(nn.Module):
    """
    RT-1 Architecture with Region 3 Hook Points:
    - Hook 1: Visual features (EfficientNet output) → OOD detection
    - Hook 2: Action logits → Entropy detection
    """
    
    def __init__(
        self,
        vocab_size: int = 512,
        action_dim: int = 256,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 64,
        image_size: Tuple[int, int] = (256, 320),
    ):
        super().__init__()
        
        # Vision Encoder: EfficientNet-B3
        self.image_size = image_size
        efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.vision_encoder = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool,
            efficientnet.classifier[:1]  # Up to dropout
        )
        
        # Get EfficientNet output dim (1536 for B3)
        self.vision_feature_dim = 1536
        
        # Language Embedding
        self.language_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Projection to transformer dim
        self.vision_proj = nn.Linear(self.vision_feature_dim, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action Head
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, action_dim),
        )
        
        # Hook storage
        self.hook1_visual_features: Optional[torch.Tensor] = None
        self.hook2_action_logits: Optional[torch.Tensor] = None
        
    def forward(
        self,
        images: torch.Tensor,
        language_tokens: torch.Tensor,
        return_hooks: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) RGB images
            language_tokens: (B, seq_len) tokenized language
            return_hooks: Whether to store hook outputs
            
        Returns:
            Dict with action_logits, action_probs, action_id, and optional hooks
        """
        B = images.shape[0]
        
        # Vision encoding
        visual_features = self.vision_encoder(images)  # (B, 3072, 1, 1)
        
        # Flatten to (B, 3072)
        visual_features = torch.flatten(visual_features, start_dim=1)
        
        # Hook 1: Visual features (for OOD detection)
        if return_hooks:
            self.hook1_visual_features = visual_features.detach()
        
        # Project to embed dim
        vision_embed = self.vision_proj(visual_features)  # (B, 768)
        vision_embed = vision_embed.unsqueeze(1)  # (B, 1, 768)
        
        # Language encoding
        lang_embed = self.language_embed(language_tokens)  # (B, seq_len, 768)
        
        # Concatenate vision + language
        combined = torch.cat([vision_embed, lang_embed], dim=1)  # (B, 1+seq_len, 768)
        
        # Transformer
        transformer_out = self.transformer(combined)  # (B, 1+seq_len, 768)
        
        # Take first token (vision-guided) for action prediction
        action_features = transformer_out[:, 0, :]  # (B, 768)
        
        # Action head
        action_logits = self.action_head(action_features)  # (B, 256)
        
        # Hook 2: Action logits (for entropy detection)
        if return_hooks:
            self.hook2_action_logits = action_logits.detach()
        
        # Softmax probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        action_id = action_probs.argmax(dim=-1)
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'action_id': action_id,
            'visual_features': self.hook1_visual_features if return_hooks else None,
            'action_logits_raw': self.hook2_action_logits if return_hooks else None,
        }


class Region3Detector(nn.Module):
    """
    Region 3: Cognitive Layer - Perception Anomaly Detection
    Multi-detector fusion: Entropy + OOD + Temporal Jump
    """
    
    def __init__(
        self,
        vision_feature_dim: int = 1536,
        action_dim: int = 256,
        thresholds: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        # OOD detector: Mahalanobis distance in visual feature space
        self.vision_feature_dim = vision_feature_dim
        self.register_buffer('normal_mean', torch.zeros(vision_feature_dim))
        self.register_buffer('normal_cov', torch.eye(vision_feature_dim))
        
        # Action dim for entropy
        self.action_dim = action_dim
        
        # Default thresholds (should be learned from data!)
        self.thresholds = thresholds or {
            'entropy': 2.5,      # nats
            'ood_mahalanobis': 3.0,  # sigma
            'temporal_jump': 0.5,    # cosine similarity
        }
        
        # Weights for fusion
        self.weights = {
            'entropy': 0.40,
            'ood': 0.35,
            'temporal': 0.25,
        }
        
        # Temporal buffer
        self.prev_action_logits: Optional[torch.Tensor] = None
        
    def compute_entropy(self, action_logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of action distribution"""
        action_probs = F.softmax(action_logits, dim=-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        return entropy  # (B,)
    
    def compute_ood_mahalanobis(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance for OOD detection"""
        B = visual_features.shape[0]
        diff = visual_features - self.normal_mean.unsqueeze(0)  # (B, D)
        
        # Simplified: use diagonal covariance
        inv_cov = 1.0 / (self.normal_cov.diag() + 1e-6)  # (D,)
        mahalanobis = torch.sqrt(torch.sum((diff ** 2) * inv_cov.unsqueeze(0), dim=-1))
        return mahalanobis  # (B,)
    
    def compute_temporal_jump(self, action_logits: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between consecutive action logits"""
        if self.prev_action_logits is None:
            self.prev_action_logits = action_logits.detach()
            return torch.zeros(action_logits.shape[0], device=action_logits.device)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(
            action_logits, 
            self.prev_action_logits, 
            dim=-1
        )
        
        # Jump = 1 - similarity
        jump = 1.0 - cos_sim
        
        # Update buffer
        self.prev_action_logits = action_logits.detach()
        
        return jump  # (B,)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        action_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute risk scores for Region 3
        """
        # Individual detectors
        entropy = self.compute_entropy(action_logits)
        ood_score = self.compute_ood_mahalanobis(visual_features)
        temporal_jump = self.compute_temporal_jump(action_logits)
        
        # Normalize scores to [0, 1]
        entropy_norm = torch.sigmoid(entropy - self.thresholds['entropy'])
        ood_norm = torch.sigmoid(ood_score - self.thresholds['ood_mahalanobis'])
        temporal_norm = torch.sigmoid(temporal_jump - self.thresholds['temporal_jump'])
        
        # Fused risk score
        risk_score = (
            self.weights['entropy'] * entropy_norm +
            self.weights['ood'] * ood_norm +
            self.weights['temporal'] * temporal_norm
        )
        
        return {
            'risk_score': risk_score,  # (B,) in [0, 1]
            'entropy': entropy,
            'ood_score': ood_score,
            'temporal_jump': temporal_jump,
            'risk_level': self._get_risk_level(risk_score),
        }
    
    def _get_risk_level(self, risk_score: torch.Tensor) -> List[str]:
        """Map risk score to level"""
        levels = []
        for r in risk_score.detach().cpu().numpy():
            if r < 0.2:
                levels.append('GREEN')
            elif r < 0.4:
                levels.append('YELLOW')
            elif r < 0.6:
                levels.append('ORANGE')
            else:
                levels.append('RED')
        return levels
    
    def update_normal_statistics(
        self,
        visual_features: torch.Tensor,
        action_logits: torch.Tensor
    ):
        """Update normal operation statistics (for threshold learning)"""
        # Running statistics update
        momentum = 0.1
        self.normal_mean = (1 - momentum) * self.normal_mean + momentum * visual_features.mean(0)
        
        # Update covariance (simplified diagonal)
        diff = visual_features - self.normal_mean.unsqueeze(0)
        var = (diff ** 2).mean(0)
        self.normal_cov = torch.diag((1 - momentum) * self.normal_cov.diag() + momentum * var)


if __name__ == '__main__':
    # Test the model
    print('Testing RT-1 with Hooks...')
    
    model = RT1WithHooks()
    detector = Region3Detector()
    
    # Dummy input
    B = 2
    images = torch.randn(B, 3, 256, 320)
    lang_tokens = torch.randint(0, 512, (B, 10))
    
    # Forward pass
    output = model(images, lang_tokens)
    
    print(f"Input: images={images.shape}, lang={lang_tokens.shape}")
    print(f"Output: action_logits={output['action_logits'].shape}")
    print(f"Hook 1 (visual): {output['visual_features'].shape}")
    print(f"Hook 2 (logits): {output['action_logits_raw'].shape}")
    
    # Region 3 detection
    risk = detector(output['visual_features'], output['action_logits'])
    print(f"Risk score: {risk['risk_score']}")
    print(f"Risk levels: {risk['risk_level']}")
    print(f"Entropy: {risk['entropy']}")
    
    print('\n✅ RT-1 model with Region 3 hooks ready!')
