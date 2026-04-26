#!/usr/bin/env python3
"""
GRU 可达集预测模型训练 - 精简版

模型规模：~180K 参数
- GRU: 192 hidden, 2 层
- 双头输出：支撑值 (32 维) + 危险概率 (1 维)
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pickle

# ========== 配置 ==========
DATA_DIR = "/root/act/retrain_data"
OUTPUT_DIR = "/root/act/outputs/region2_gru_retrained"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型超参数 (中等规模 - 平衡性能与过拟合)
HIDDEN_DIM = 256      # 增加容量
NUM_LAYERS = 2        # 保持 2 层 (10 步短期依赖足够)
DROPOUT = 0.3
SUPPORT_DIM = 28  # 与支撑方向矩阵一致

# 训练超参数
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 100
DANGER_WEIGHT = 5.0   # 危险样本权重
DANGER_POS_WEIGHT = 3.0  # 分类正样本权重
TRAIN_PERTURB_PROB = 0.5  # 数据增强概率
NOISE_STD = 0.02      # 2% 噪声

# ========== 数据集 ==========
class PerturbedDataset(Dataset):
    def __init__(self, histories, targets, labels, indices, perturb_prob=TRAIN_PERTURB_PROB):
        self.histories = histories[indices]
        self.targets = targets[indices]
        self.labels = labels[indices]
        self.perturb_prob = perturb_prob
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        history = self.histories[idx]
        target = self.targets[idx]
        label = self.labels[idx]
        
        # 数据增强：50% 概率加噪声
        if np.random.random() < self.perturb_prob:
            noise = np.random.randn(*history.shape).astype(np.float32) * NOISE_STD
            history = history + noise
        
        return (
            torch.FloatTensor(history),
            torch.FloatTensor(target),
            torch.FloatTensor([label])
        )

# ========== 模型架构 ==========
class GRUReachability(nn.Module):
    def __init__(self, state_dim=28, hidden_dim=192, num_layers=2, support_dim=32, dropout=0.3):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # GRU 编码器
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 共享表示
        self.shared = nn.Linear(hidden_dim, hidden_dim // 2)
        self.shared_norm = nn.LayerNorm(hidden_dim // 2)
        
        # 头 1: 可达集预测 (支撑值)
        self.reach_head = nn.Linear(hidden_dim // 2, support_dim)
        
        # 头 2: 危险概率
        self.danger_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, history):
        # history: (B, 10, 28)
        h = self.input_proj(history)
        h = self.input_norm(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        _, hidden = self.gru(h)
        h = hidden[-1]  # (B, hidden)
        
        h = self.shared(h)
        h = self.shared_norm(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        reach_pred = self.reach_head(h)      # (B, support_dim)
        danger_logit = self.danger_head(h)   # (B, 1)
        
        return reach_pred, danger_logit

# ========== 损失函数 ==========
class ReachabilityLoss(nn.Module):
    def __init__(self, danger_weight=DANGER_WEIGHT, danger_pos_weight=DANGER_POS_WEIGHT):
        super().__init__()
        self.danger_weight = danger_weight
        self.danger_pos_weight = danger_pos_weight
    
    def forward(self, reach_pred, reach_target, danger_logit, danger_target):
        # 支撑值损失：Huber
        reach_loss = nn.functional.huber_loss(reach_pred, reach_target, delta=1.0)
        
        # 危险分类损失：加权 BCE
        danger_pred = torch.sigmoid(danger_logit)
        weights = torch.where(
            danger_target > 0.5,
            self.danger_pos_weight,
            1.0
        )
        danger_loss = nn.functional.binary_cross_entropy(
            danger_pred, danger_target, weight=weights
        )
        
        # 总损失
        loss = reach_loss + self.danger_weight * danger_loss
        
        return loss, reach_loss, danger_loss

# ========== 训练函数 ==========
def train():
    print("="*60)
    print("【GRU 重训练 - 精简版】")
    print("="*60)
    print(f"模型：GRU({HIDDEN_DIM}, {NUM_LAYERS}层), Dropout={DROPOUT}")
    print(f"支撑维度：{SUPPORT_DIM}")
    print(f"学习率：{LEARNING_RATE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"危险权重：×{DANGER_WEIGHT}, 正样本权重：×{DANGER_POS_WEIGHT}")
    print(f"数据增强：{TRAIN_PERTURB_PROB*100:.0f}% 概率，噪声σ={NOISE_STD}")
    print()
    
    # 加载数据
    print("加载数据...")
    histories = np.load(os.path.join(DATA_DIR, 'gru_train_history.npy'))
    targets = np.load(os.path.join(DATA_DIR, 'gru_train_support_target.npy'))
    labels = np.load(os.path.join(DATA_DIR, 'gru_train_danger_label.npy'))
    train_idx = np.load(os.path.join(DATA_DIR, 'train_idx.npy'))
    val_idx = np.load(os.path.join(DATA_DIR, 'val_idx.npy'))
    
    print(f"  训练集：{len(train_idx)} 样本")
    print(f"  验证集：{len(val_idx)} 样本")
    print(f"  危险样本比例：{np.mean(labels)*100:.1f}%")
    
    # 数据加载器
    train_dataset = PerturbedDataset(histories, targets, labels, train_idx)
    val_dataset = PerturbedDataset(histories, targets, labels, val_idx, perturb_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备：{device}")
    
    model = GRUReachability(
        state_dim=28,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        support_dim=SUPPORT_DIM,
        dropout=DROPOUT
    ).to(device)
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量：{n_params:,} ({n_params/1e6:.2f}M)")
    
    criterion = ReachabilityLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # 检查 checkpoint
    checkpoint_path = os.path.join(OUTPUT_DIR, 'training_checkpoint.pkl')
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_j = -1
    
    if os.path.exists(checkpoint_path):
        print(f"\n发现 checkpoint，恢复训练...")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        best_val_j = checkpoint['best_val_j']
        print(f"  恢复到 epoch {start_epoch}")
    
    # 训练循环
    print("\n开始训练...\n")
    
    for epoch in range(start_epoch, EPOCHS):
        # 训练
        model.train()
        train_loss_sum = 0.0
        train_reach_loss_sum = 0.0
        train_danger_loss_sum = 0.0
        
        for history, target, danger_flag in train_loader:
            history = history.to(device)
            target = target.to(device)
            danger_flag = danger_flag.to(device)
            
            optimizer.zero_grad()
            
            reach_pred, danger_logit = model(history)
            
            loss, reach_loss, danger_loss = criterion(reach_pred, target, danger_logit, danger_flag)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_reach_loss_sum += reach_loss.item()
            train_danger_loss_sum += danger_loss.item()
        
        n_batches = len(train_loader)
        train_loss = train_loss_sum / n_batches
        train_reach_loss = train_reach_loss_sum / n_batches
        train_danger_loss = train_danger_loss_sum / n_batches
        
        # 验证
        model.eval()
        val_loss_sum = 0.0
        all_danger_pred = []
        all_danger_flag = []
        
        with torch.no_grad():
            for history, target, danger_flag in val_loader:
                history = history.to(device)
                target = target.to(device)
                danger_flag = danger_flag.to(device)
                
                reach_pred, danger_logit = model(history)
                
                loss, reach_loss, danger_loss = criterion(reach_pred, target, danger_logit, danger_flag)
                
                val_loss_sum += loss.item()
                
                danger_pred = torch.sigmoid(danger_logit)
                all_danger_pred.extend(danger_pred.cpu().numpy().flatten())
                all_danger_flag.extend(danger_flag.cpu().numpy().flatten())
        
        val_loss = val_loss_sum / len(val_loader)
        
        # 计算 Youden J
        all_danger_pred = np.array(all_danger_pred)
        all_danger_flag = np.array(all_danger_flag)
        
        thresholds = np.percentile(all_danger_pred, np.arange(10, 91, 5))
        best_j = -1
        best_thresh = 0.5
        for thresh in thresholds:
            pred = all_danger_pred >= thresh
            tp = np.sum((pred) & (all_danger_flag == 1))
            fp = np.sum((pred) & (all_danger_flag == 0))
            tn = np.sum((~pred) & (all_danger_flag == 0))
            fn = np.sum((~pred) & (all_danger_flag == 1))
            tpr = tp/(tp+fn) if (tp+fn)>0 else 0
            fpr = fp/(fp+tn) if (fp+tn)>0 else 0
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_thresh = thresh
        
        scheduler.step(best_j)
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS}: "
              f"train_loss={train_loss:.4f} (reach={train_reach_loss:.4f}, danger={train_danger_loss:.4f}), "
              f"val_loss={val_loss:.4f}, val_J={best_j:.3f} (@thresh={best_thresh:.2f}), "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_j': best_val_j,
            }, os.path.join(OUTPUT_DIR, 'gru_retrained_best.pth'))
            print(f"  ✓ 保存最佳损失模型 (val_loss={best_val_loss:.4f})")
        
        if best_j > best_val_j:
            best_val_j = best_j
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_j': best_val_j,
            }, os.path.join(OUTPUT_DIR, 'gru_retrained_best_j.pth'))
            print(f"  ✓ 保存最佳 Youden 模型 (val_J={best_val_j:.3f})")
        
        # 定期保存 checkpoint (每 5 个 epoch)
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_j': best_val_j,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  → 保存 checkpoint")
    
    # 清理 checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("\n" + "="*60)
    print("✓ 训练完成！")
    print("="*60)
    print(f"最佳验证损失：{best_val_loss:.4f}")
    print(f"最佳 Youden J: {best_val_j:.3f}")
    print(f"\n模型保存位置:")
    print(f"  {OUTPUT_DIR}/gru_retrained_best.pth")
    print(f"  {OUTPUT_DIR}/gru_retrained_best_j.pth")

if __name__ == '__main__':
    train()
