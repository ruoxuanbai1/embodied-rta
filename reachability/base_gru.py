"""
底盘 GRU 可达性预测模型
输入：过去 10 帧状态序列
输出：未来 2D 可达包络的 8 维支撑函数
"""

import torch
import torch.nn as nn
import numpy as np

class BaseReachabilityGRU(nn.Module):
    """底盘可达性 GRU 模型"""
    
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=8):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        # x: (Batch, 10, 5) - 过去 10 帧底盘状态
        out, _ = self.gru(x)  # (B, 10, 64)
        out = out[:, -1, :]   # 取最后一帧 (B, 64)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)   # (B, 8) - 8 维支撑函数
        return out


class ArmReachabilityGRU(nn.Module):
    """机械臂末端可达性 GRU 模型"""
    
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, output_dim=6):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        # x: (Batch, 10, 14) - 过去 10 帧机械臂状态
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)   # (B, 6) - 6 维支撑函数 (x,y,z 正负方向)
        return out


if __name__ == '__main__':
    # 测试模型
    base_gru = BaseReachabilityGRU()
    arm_gru = ArmReachabilityGRU()
    
    # 底盘测试
    x_base = torch.randn(2, 10, 5)
    y_base = base_gru(x_base)
    print(f"底盘输入：{x_base.shape}, 输出：{y_base.shape}")
    
    # 机械臂测试
    x_arm = torch.randn(2, 10, 14)
    y_arm = arm_gru(x_arm)
    print(f"机械臂输入：{x_arm.shape}, 输出：{y_arm.shape}")
    
    print("GRU 模型测试通过!")
