#!/usr/bin/env python3
"""
Region 2 GRU 可达性预测模型训练

架构:
输入 (10, 19) → GRU(128)×2 → FC(64) → ReLU → FC(32) → 输出

训练数据：70,000 样本 (来自动力学推演 + 扰动)
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os


class ReachabilityGRU(nn.Module):
    """可达性预测 GRU 模型"""
    
    def __init__(self, input_dim=19, hidden_dim=128, 
                 num_layers=2, output_dim=32):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x: (Batch, 10, 19)
        out, h_n = self.gru(x)  # out: (B, 10, 128), h_n: (2, B, 128)
        
        # 取最后一帧隐藏状态
        out = h_n[-1]  # (B, 128)
        
        # 全连接层
        out = self.fc1(out)  # (B, 64)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (B, 32)
        
        return out


class AsymmetricHuberLoss(nn.Module):
    """非对称 Huber 损失
    
    欠预测 (pred < true) 惩罚更重，因为可达集过小会导致危险
    """
    
    def __init__(self, under_weight=2.0, over_weight=1.0, delta=1.0):
        super().__init__()
        self.under_weight = under_weight
        self.over_weight = over_weight
        self.delta = delta
    
    def forward(self, pred, target):
        error = target - pred
        
        # Huber 损失
        is_small = error.abs() <= self.delta
        small_loss = 0.5 * error ** 2
        large_loss = self.delta * (error.abs() - 0.5 * self.delta)
        
        loss = torch.where(is_small, small_loss, large_loss)
        
        # 非对称加权
        weights = torch.where(error > 0, 
                              self.under_weight,  # 欠预测
                              self.over_weight)   # 过预测
        
        loss = loss * weights
        
        return loss.mean()


def load_training_data(data_path: str) -> tuple:
    """加载训练数据"""
    data = np.load(data_path)
    inputs = torch.FloatTensor(data['inputs'])
    outputs = torch.FloatTensor(data['outputs'])
    
    # 数据归一化
    inputs_mean = inputs.mean()
    inputs_std = inputs.std()
    inputs = (inputs - inputs_mean) / (inputs_std + 1e-6)
    
    outputs_mean = outputs.mean()
    outputs_std = outputs.std()
    outputs = (outputs - outputs_mean) / (outputs_std + 1e-6)
    
    return inputs, outputs, {
        'inputs_mean': inputs_mean,
        'inputs_std': inputs_std,
        'outputs_mean': outputs_mean,
        'outputs_std': outputs_std,
    }


def train_gru(data_path: str, output_dir: str, 
              epochs: int = 100, batch_size: int = 64,
              lr: float = 0.001):
    """训练 GRU 模型"""
    
    print("="*60)
    print("Region 2 GRU 可达性预测模型训练")
    print("="*60)
    
    # 加载数据
    print("\n加载训练数据...")
    inputs, outputs, norm_params = load_training_data(data_path)
    print(f"  输入形状：{inputs.shape}")
    print(f"  输出形状：{outputs.shape}")
    print(f"  归一化参数已保存")
    
    # 划分训练/验证集
    n_train = int(0.9 * len(inputs))
    train_inputs = inputs[:n_train]
    train_outputs = outputs[:n_train]
    val_inputs = inputs[n_train:]
    val_outputs = outputs[n_train:]
    
    print(f"  训练集：{len(train_inputs)} 样本")
    print(f"  验证集：{len(val_inputs)} 样本")
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_outputs)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReachabilityGRU().to(device)
    
    print(f"\n模型架构:")
    print(f"  输入：(Batch, 10, 19)")
    print(f"  GRU: 2 层 × 128 隐藏单元")
    print(f"  输出：(Batch, 32)")
    print(f"  设备：{device}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # 损失函数
    criterion = AsymmetricHuberLoss(under_weight=2.0, over_weight=1.0)
    
    # 训练循环
    print(f"\n开始训练 (epochs={epochs}, batch_size={batch_size})...")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        
        for batch_inputs, batch_outputs in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_outputs)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss_sum += loss.item()
            n_batches += 1
        
        train_loss = train_loss_sum / n_batches
        
        # 验证
        model.eval()
        val_loss_sum = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_outputs in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)
                
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_outputs)
                
                val_loss_sum += loss.item()
                n_batches += 1
        
        val_loss = val_loss_sum / n_batches
        
        # 学习率调整
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'norm_params': norm_params,
                'history': history,
            }, os.path.join(output_dir, 'gru_reachability_best.pt'))
        
        # 进度报告
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, "
                  f"lr={current_lr:.6f}")
    
    # 保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'norm_params': norm_params,
        'history': history,
    }, os.path.join(output_dir, 'gru_reachability_final.pt'))
    
    print(f"\n训练完成!")
    print(f"  最佳验证损失：{best_val_loss:.6f}")
    print(f"  最终验证损失：{val_loss:.6f}")
    print(f"  模型已保存：{output_dir}/gru_reachability_*.pt")
    
    return model, history


if __name__ == '__main__':
    # 训练参数
    DATA_PATH = '/home/vipuser/Embodied-RTA/reachability/gru_training_data.npz'
    OUTPUT_DIR = '/home/vipuser/Embodied-RTA/reachability/'
    
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 0.001
    
    # 开始训练
    model, history = train_gru(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )
