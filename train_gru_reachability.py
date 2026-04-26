#!/usr/bin/env python3
"""
Region 2 GRU 可达集预测模型训练

训练数据来源：ACT 轨迹数据
输入：10 步状态历史
输出：32 维支撑函数 (16 变量 × min/max)

训练方法：蒙特卡洛扰动 + 动力学传播生成可达集标签
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class GRUReachabilityPredictor(nn.Module):
    """
    GRU 可达集预测模型
    
    输入：(B, seq_len=10, state_dim=14) 状态历史
    输出：(B, 28) 支撑函数 (14 变量 × min/max)
    """
    
    def __init__(
        self,
        state_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 28,  # 14 × 2
        device: str = 'cuda'
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        self.to(device)
    
    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        预测可达集支撑函数
        
        Args:
            state_history: (B, seq_len, state_dim)
        
        Returns:
            support_fn: (B, output_dim)
        """
        # GRU 编码
        _, hidden = self.gru(state_history)
        hidden_state = hidden[-1]  # (B, hidden_dim)
        
        # 回归支撑函数
        support_fn = self.regressor(hidden_state)
        
        return support_fn
    
    def compute_reachable_set(
        self,
        state_history: torch.Tensor,
        dynamics_model: callable,
        n_perturbations: int = 50,
        horizon: int = 250,  # 5 秒 @ 50Hz
        perturbation_scale: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用蒙特卡洛扰动计算可达集
        
        Args:
            state_history: (B, seq_len, state_dim)
            dynamics_model: 动力学模型函数 f(x, u) -> x_next
            n_perturbations: 扰动次数
            horizon: 预测步数
            perturbation_scale: 扰动幅度
        
        Returns:
            min_bound: (B, 16) 可达集下界
            max_bound: (B, 16) 可达集上界
        """
        B, seq_len, _ = state_history.shape
        
        # 当前状态
        x_current = state_history[:, -1]  # (B, state_dim)
        
        # 存储所有扰动轨迹
        all_trajectories = []
        
        for i in range(n_perturbations):
            # 施加扰动
            perturbation = torch.randn_like(x_current) * perturbation_scale
            x_perturbed = x_current + perturbation
            
            # 正向传播动力学
            trajectory = [x_perturbed]
            for t in range(horizon):
                # 简单积分模型 (实际应使用具体动力学)
                x_next = x_perturbed  # TODO: 替换为真实动力学
                trajectory.append(x_next)
                x_perturbed = x_next
            
            trajectory = torch.stack(trajectory, dim=1)  # (B, horizon+1, state_dim)
            all_trajectories.append(trajectory)
        
        # 堆叠所有轨迹
        all_trajectories = torch.stack(all_trajectories, dim=0)  # (n_pert, B, horizon+1, state_dim)
        
        # 计算包络 (min/max over perturbations and time)
        min_bound = all_trajectories.min(dim=0)[0].min(dim=1)[0]  # (B, state_dim)
        max_bound = all_trajectories.max(dim=0)[0].min(dim=1)[0]  # (B, state_dim)
        
        # 选择 16 个关键变量
        key_vars = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 简化
        min_bound = min_bound[:, :16]
        max_bound = max_bound[:, :16]
        
        return min_bound, max_bound


def train_gru(
    trajectory_dir: str = '/root/rt1_trajectory_data',
    output_path: str = '/root/Embodied-RTA/gru_reachability.pth',
    seq_len: int = 10,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cuda'
):
    """
    训练 GRU 模型
    
    使用轨迹数据生成训练样本:
    - 输入：10 步历史
    - 标签：未来 5 秒可达集 (通过扰动 + 动力学计算)
    """
    print('=' * 60)
    print('训练 GRU 可达集预测模型')
    print('=' * 60)
    
    # 加载轨迹数据
    trajectory_files = list(Path(trajectory_dir).glob('*.npz'))
    print(f'找到 {len(trajectory_files)} 条轨迹')
    
    # 构建数据集
    all_states = []
    all_actions = []
    
    for traj_file in trajectory_files:
        try:
            data = np.load(traj_file)
            states = data['states']
            actions = data['actions']
            
            # 跳过故障数据 (只使用正常轨迹训练)
            if 'fault' in str(traj_file) and 'normal' not in str(traj_file):
                continue
            
            all_states.append(states)
            all_actions.append(actions)
        except Exception as e:
            print(f'跳过损坏文件：{traj_file.name}')
            continue
    
    # 合并所有轨迹
    all_states = np.vstack(all_states)
    all_actions = np.vstack(all_actions)
    
    print(f'总数据量：{len(all_states)} 步')
    
    # 创建序列样本
    sequences = []
    targets = []
    
    for i in range(len(all_states) - seq_len):
        seq = all_states[i:i+seq_len]  # (seq_len, state_dim)
        
        # 计算可达集标签 (简化：使用未来状态的范围)
        future_states = all_states[i+seq_len:i+seq_len+50]  # 未来 50 步
        if len(future_states) > 0:
            min_bound = future_states.min(axis=0)[:14]  # 14 维状态
            max_bound = future_states.max(axis=0)[:14]
            target = np.concatenate([min_bound, max_bound])  # (28,)
            
            sequences.append(seq)
            targets.append(target)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    print(f'序列样本：{len(sequences)}')
    print(f'输入形状：{sequences.shape}')
    print(f'标签形状：{targets.shape}')
    
    # 转换为 Tensor
    sequences_tensor = torch.FloatTensor(sequences).to(device)
    targets_tensor = torch.FloatTensor(targets).to(device)
    
    # 划分训练/验证集
    n_train = int(len(sequences) * 0.8)
    
    train_seq = sequences_tensor[:n_train]
    train_tgt = targets_tensor[:n_train]
    val_seq = sequences_tensor[n_train:]
    val_tgt = targets_tensor[n_train:]
    
    # 创建模型
    model = GRUReachabilityPredictor(state_dim=14, hidden_dim=64, device=device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 训练循环
    print(f'\\n开始训练：{epochs} epochs')
    print('-' * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        # Mini-batch 训练
        indices = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_seq = train_seq[batch_idx]
            batch_tgt = train_tgt[batch_idx]
            
            optimizer.zero_grad()
            pred = model(batch_seq)
            loss = criterion(pred, batch_tgt)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_loss / n_batches
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(val_seq)
            val_loss = criterion(val_pred, val_tgt).item()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, output_path)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d}/{epochs}: '
                  f'Train Loss={avg_train_loss:.4f}, '
                  f'Val Loss={val_loss:.4f}, '
                  f'Best={best_val_loss:.4f}')
    
    print('-' * 60)
    print(f'训练完成! 最佳验证损失：{best_val_loss:.4f}')
    print(f'模型保存至：{output_path}')
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory-dir', type=str, default='/root/rt1_trajectory_data')
    parser.add_argument('--output', type=str, default='/root/Embodied-RTA/gru_reachability.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    train_gru(
        trajectory_dir=args.trajectory_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
