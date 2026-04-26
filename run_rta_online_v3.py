#!/usr/bin/env python3
"""
run_rta_online_v3.py - 完整三层 RTA 检测系统

三层架构:
1. Region 1: 物理硬约束检测 (关节限位、速度、碰撞) → safe_r1, risk_r1
2. Region 2: 可达性预测检测 (GRU 支撑函数 + 碰撞预警) → safe_r2, risk_r2, collision_warning
3. Region 3: 感知异常检测 (三模块：逻辑 + 激活 + OOD) → safe_r3, risk_r3

风险融合:
R_total = 0.3×risk_1 + 0.4×risk_2 + 0.3×risk_3

Region 2 修正:
- 使用支撑函数投影检查 (不是 min-max)
- 加载 direction_matrix (16, 28)
- 计算当前状态在 16 个方向上的投影
- 检查是否超出 GRU 预测的支撑函数值
- 添加预测轨迹碰撞检测

Region 3 修复:
- 使用真正的 Region3Detector (不是随机数)
- 三模块独立检测：S_logic, D_ham, D_ood
- 动作模态自适应 (K=8 聚类)
- 加载学习到的阈值

输入:
- qpos (14,), qvel (14,), image, action (14,)

输出:
- safe (bool), risk (float), alert_any (bool)
- TP/FP/TN/FN 混淆矩阵
- Precision/Recall/F1 指标
"""

import sys
import os

# 在导入任何模块前设置正确的 argv
sys.argv = [
    'run_rta_online_v3',
    '--ckpt_dir', '.',
    '--policy_class', 'ACTPolicy',
    '--task_name', 'sim_transfer_cube_scripted',
    '--seed', '0',
    '--num_epochs', '100',
    '--lr', '1e-4',
    '--lr_backbone', '1e-5',
    '--batch_size', '8',
    '--weight_decay', '1e-4',
    '--epochs', '100',
    '--lr_drop', '50',
    '--clip_max_norm', '10',
    '--backbone', 'resnet18',
    '--position_embedding', 'sine',
    '--camera_names', 'top',
    '--enc_layers', '4',
    '--dec_layers', '1',
    '--dim_feedforward', '2048',
    '--hidden_dim', '256',
    '--dropout', '0.1',
    '--nheads', '8',
    '--num_queries', '400',
    '--chunk_size', '100',
]

# 现在可以安全导入 ACT 模块
from constants import SIM_TASK_CONFIGS
from policy import ACTPolicy
from sim_env import make_sim_env, BOX_POSE
from utils import sample_box_pose

# 然后导入我们的代码
import json
import csv
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from einops import rearrange

# 导入真正的 Region3Detector
from region3_detector import Region3Detector

# ============================================================================
# Hook 类 - 提取激活值
# ============================================================================

class NetworkHook:
    """简单的 Hook 类，用于提取网络中间层输出"""
    
    def __init__(self, name="hook"):
        self.name = name
        self.output = None
        self.handle = None
    
    def hook_fn(self, module, input, output):
        self.output = output.detach()
    
    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        print(f"  ✓ Hook 已注册：{self.name}")
    
    def get_output(self):
        return self.output
    
    def clear(self):
        self.output = None
    
    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

# ============================================================================
# 场景与故障定义
# ============================================================================

BASE_SCENES = {
    'B1': {'name': '空旷', 'obstacles': 0, 'difficulty': 1},
    'B2': {'name': '静态障碍', 'obstacles': 5, 'difficulty': 2},
    'B3': {'name': '密集障碍', 'obstacles': 10, 'difficulty': 3},
}

FAULT_TYPES = {
    'F1': {'name': '光照突变', 'type': '感知', 'inject': 'lighting_drop'},
    'F2': {'name': '摄像头遮挡', 'type': '感知', 'inject': 'camera_occlusion'},
    'F3': {'name': '对抗补丁', 'type': '感知', 'inject': 'adversarial_patch'},
    'F4': {'name': '负载突变', 'type': '动力学', 'inject': 'payload_shift'},
    'F5': {'name': '关节摩擦', 'type': '动力学', 'inject': 'joint_friction'},
    'F6': {'name': '突发障碍', 'type': '突发', 'inject': 'dynamic_obstacle'},
    'F7': {'name': '传感器噪声', 'type': '感知', 'inject': 'sensor_noise'},
    'F8': {'name': '复合故障', 'type': '复合', 'inject': 'compound'},
    'None': {'name': '正常', 'type': '正常', 'inject': None},
}

# ============================================================================
# 故障注入器
# ============================================================================

class FaultInjector:
    def __init__(self, fault_type: str, inject_step: int = 50):
        self.fault_type = fault_type
        self.inject_step = inject_step
        self.fault_active = False
        self.action_accumulator = []
    
    def inject(self, ts, step: int, action=None):
        if step < self.inject_step:
            return ts
        
        self.fault_active = True
        
        if self.fault_type == 'F1_lighting':
            if 'images' in ts.observation:
                for cam in ts.observation['images']:
                    ts.observation['images'][cam] = (ts.observation['images'][cam] * 0.1).astype(np.uint8)
        
        elif self.fault_type == 'F2_occlusion':
            if 'images' in ts.observation:
                for cam in ts.observation['images']:
                    img = ts.observation['images'][cam]
                    h, w = img.shape[:2]
                    img[h//3:2*h//3, w//3:2*w//3] = 0
        
        elif self.fault_type == 'F7_noise':
            if 'qpos' in ts.observation:
                ts.observation['qpos'] += np.random.normal(0, 0.5, ts.observation['qpos'].shape)
            if 'qvel' in ts.observation:
                ts.observation['qvel'] += np.random.normal(0, 0.5, ts.observation['qvel'].shape)
        
        return ts
    
    def inject_action(self, action):
        if not self.fault_active:
            return action
        
        if self.fault_type == 'F5_friction':
            return action * 0.3
        elif self.fault_type == 'F4_payload':
            self.action_accumulator.append(action * 0.2)
            if len(self.action_accumulator) > 10:
                release = self.action_accumulator.pop(0)
                return action + release
        
        return action

# ============================================================================
# Region 1: 物理硬约束
# ============================================================================

class Region1Monitor:
    """
    Region 1: 物理硬约束检测
    
    检测项:
    - 关节限位：qpos ∈ [qpos_min, qpos_max]
    - 关节速度：max(|qvel|) < qvel_max
    - 碰撞检测：末端与障碍物距离 > collision_dist
    
    输出:
    - safe: bool 是否安全
    - violations: List[str] 违规项
    - risk_score: float 风险分数 [0, 1]
    """
    
    def __init__(self, qpos_min=-2.0, qpos_max=2.0, qvel_max=1.0, collision_dist=0.02):
        self.qpos_min = qpos_min
        self.qpos_max = qpos_max
        self.qvel_max = qvel_max
        self.collision_dist = collision_dist
        self.obstacles = None
    
    def set_obstacles(self, obstacles):
        """设置障碍物列表"""
        self.obstacles = obstacles
    
    def check(self, qpos, qvel):
        violations = []
        risk_score = 0.0
        
        # 1. 关节限位检测
        qpos_margin = min(np.min(qpos - self.qpos_min), np.min(self.qpos_max - qpos))
        if qpos_margin < 0:
            violations.append('joint_limit')
            risk_score = 1.0
        elif qpos_margin < 0.3:
            risk_score = max(risk_score, 1.0 - (qpos_margin / 0.3))
        
        # 2. 速度检测
        qvel_ratio = np.max(np.abs(qvel)) / self.qvel_max
        if qvel_ratio > 1.0:
            violations.append('velocity_limit')
            risk_score = max(risk_score, 1.0)
        elif qvel_ratio > 0.8:
            risk_score = max(risk_score, (qvel_ratio - 0.8) / 0.2)
        
        # 3. 碰撞检测 ⭐ 新增
        if self.obstacles:
            for obs in self.obstacles:
                dist = np.linalg.norm(qpos[:3] - np.array(obs['pos'][:3]))
                if dist < self.collision_dist:
                    violations.append('collision')
                    risk_score = 1.0
        
        safe = len(violations) == 0
        return safe, violations, float(risk_score)

# ============================================================================
# Region 2: 可达性预测 (GRU) - 支撑函数方法
# ============================================================================

class DeepReachabilityGRU(nn.Module):
    """
    深度 GRU 可达集预测模型
    
    输入：历史状态序列 (batch, seq_len=10, input_dim=28)
    输出：支撑函数值 (batch, num_directions=16)
    
    结构:
    1. 输入投影：Linear(28, 192) + LayerNorm
    2. GRU (4 层，输入 192，隐藏 192)
    3. 输出：Linear(192, 96) + ReLU + Dropout + LayerNorm → Linear(96, 16)
    """
    
    def __init__(self, input_dim=28, hidden_dim=192, num_layers=4, output_dim=16, dropout=0.4):
        super().__init__()
        
        # 输入投影 (28 → 192)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # GRU (输入是投影后的 hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 输出层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_norm = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, history):
        # history: (batch, seq_len, input_dim=28)
        
        # 输入投影
        h = self.input_proj(history)  # (batch, seq_len, 192)
        h = self.input_norm(h)
        h = self.relu(h)
        
        # GRU
        _, hidden = self.gru(h)  # hidden: (num_layers, batch, 192)
        h = hidden[-1]  # 取最后层 (batch, 192)
        
        # 输出层
        h = self.fc1(h)  # (batch, 96)
        h = self.fc_norm(h)
        h = self.relu(h)
        h = self.dropout(h)
        output = self.fc2(h)  # (batch, 16)
        
        return output


# ============================================================================
# Region 2: 可达性预测 (GRU) - 支撑函数方法
# ============================================================================

class Region2Monitor:
    """
    Region 2: 可达性预测检测器
    
    原理:
    1. GRU 预测未来 1 秒可达集的支撑函数值 (16 个方向)
    2. 检查当前状态在各方向上的投影是否超出支撑函数值
    3. 预测轨迹与障碍物碰撞检测
    
    输出:
    - safe: bool 是否安全
    - risk: float 风险分数 [0, 1]
    - collision_warning: bool 是否有碰撞预警
    """
    
    def __init__(self, model_path, direction_matrix_path, max_history=10):
        self.max_history = max_history
        self.history_buffer = []
        self.has_model = os.path.exists(model_path)
        self.model = None
        self.direction_matrix = None
        
        if self.has_model:
            # 加载 GRU 模型
            checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
            self.model_config = checkpoint.get('config', {})
            
            # 构建模型并加载权重
            self.model = DeepReachabilityGRU(
                input_dim=self.model_config.get('input_dim', 28),
                hidden_dim=self.model_config.get('hidden_dim', 192),
                num_layers=self.model_config.get('num_layers', 4),
                output_dim=self.model_config.get('num_directions', 16),
                dropout=self.model_config.get('dropout', 0.4)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to('cuda')
            self.model.eval()
            
            print(f"  ✓ Region 2 GRU 已加载：{model_path}")
            print(f"    配置：hidden={self.model_config.get('hidden_dim', 192)}, "
                  f"layers={self.model_config.get('num_layers', 4)}, "
                  f"directions={self.model_config.get('num_directions', 16)}")
        
        # 加载方向矩阵 (num_directions, state_dim)
        if os.path.exists(direction_matrix_path):
            self.direction_matrix = np.load(direction_matrix_path)
            print(f"  ✓ 支撑方向矩阵已加载：{self.direction_matrix.shape}")
        else:
            # 生成默认方向矩阵 (16 个方向)
            self.direction_matrix = self._generate_support_directions(16, 28)
            print(f"  ⚠ 使用默认支撑方向矩阵：{self.direction_matrix.shape}")
        
        self.obstacles = None
        self.collision_horizon = 0.5  # 预测 0.5 秒内的碰撞
    
    def _generate_support_directions(self, num_directions=16, state_dim=28, seed=42):
        """
        生成支撑函数方向矩阵
        
        包含:
        - 12 个坐标轴方向 (±qpos[0:6])
        - 4 个对角线方向 (随机单位向量)
        """
        np.random.seed(seed)
        directions = []
        
        # 12 个坐标轴方向 (±qpos[0:6])
        for i in range(6):
            d = np.zeros(state_dim)
            d[i] = 1.0
            directions.append(d)
            
            d_neg = np.zeros(state_dim)
            d_neg[i] = -1.0
            directions.append(d_neg)
        
        # 4 个对角线方向
        remaining = num_directions - 12
        for _ in range(remaining):
            d = np.random.randn(state_dim)
            d = d / np.linalg.norm(d)
            directions.append(d)
        
        directions = np.array(directions)
        # L2 归一化
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        return directions
    
    def set_obstacles(self, obstacles):
        """设置障碍物列表"""
        self.obstacles = obstacles
    
    def update_history(self, qpos, qvel):
        """更新历史状态缓冲区"""
        state = np.concatenate([qpos, qvel])  # (28,)
        self.history_buffer.append(state)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
    
    def _predict_trajectory(self, num_steps=25):
        """
        自回归预测未来轨迹
        
        返回:
        - predicted_states: List[np.array] 未来 num_steps 步的状态
        """
        if not self.has_model or len(self.history_buffer) < self.max_history:
            return []
        
        predicted_states = []
        current_history = list(self.history_buffer)  # 复制当前历史
        
        with torch.no_grad():
            for _ in range(num_steps):
                # 准备输入 (最近 10 步)
                history = np.array(current_history[-self.max_history:])
                if history.shape[1] < 28:
                    padded = np.zeros((history.shape[0], 28))
                    padded[:, :history.shape[1]] = history
                    history = padded
                
                # GRU 预测支撑函数值
                support_values = self.model(
                    torch.from_numpy(history).float().unsqueeze(0).to('cuda')
                ).cpu().numpy()[0]  # (16,)
                
                # 从支撑函数值近似恢复状态 (简化：用方向矩阵的伪逆)
                # 这是一个近似，实际应该用更复杂的解码方法
                next_state = np.linalg.lstsq(
                    self.direction_matrix, support_values, rcond=None
                )[0]
                
                predicted_states.append(next_state)
                current_history.append(next_state)
        
        return predicted_states
    
    def check_collision(self, predicted_states):
        """
        检查预测轨迹是否与障碍物碰撞
        
        返回:
        - collision_warning: bool 是否有碰撞预警
        - min_distance: float 最小距离
        """
        if not self.obstacles or len(predicted_states) == 0:
            return False, float('inf')
        
        min_distance = float('inf')
        collision_warning = False
        
        for state in predicted_states:
            pos = state[:3]  # 末端执行器位置 (假设前 3 维是位置)
            
            for obs in self.obstacles:
                obs_pos = np.array(obs['pos'][:3])
                dist = np.linalg.norm(pos - obs_pos)
                min_distance = min(min_distance, dist)
                
                if dist < 0.1:  # 10cm 预警距离
                    collision_warning = True
        
        return collision_warning, min_distance
    
    def check(self, qpos, qvel, action):
        """
        Region 2 可达性检测
        
        步骤:
        1. GRU 预测支撑函数值 (16 个方向)
        2. 计算当前状态在各方向上的投影
        3. 检查是否超出支撑函数值
        4. 预测轨迹碰撞检测
        
        返回:
        - safe: bool 是否安全
        - risk: float 风险分数 [0, 1]
        - collision_warning: bool 是否有碰撞预警
        """
        if not self.has_model or len(self.history_buffer) < self.max_history:
            return True, 0.0, False
        
        history = np.array(self.history_buffer)
        if history.shape[1] < 28:
            padded = np.zeros((history.shape[0], 28))
            padded[:, :history.shape[1]] = history
            history = padded
        
        # === 步骤 1: GRU 预测支撑函数值 ===
        with torch.no_grad():
            support_values = self.model(
                torch.from_numpy(history).float().unsqueeze(0).to('cuda')
            ).cpu().numpy()[0]  # (16,)
        
        # === 步骤 2: 计算当前状态的投影 ===
        current_state = np.concatenate([qpos, qvel])  # (28,)
        current_projection = self.direction_matrix @ current_state  # (16,)
        
        # === 步骤 3: 支撑函数安全检查 ===
        exceed = current_projection - support_values  # (16,)
        safe_support = np.all(exceed <= 1e-6)  # 允许小的数值误差
        
        # 风险分数：最大超出比例
        if np.max(np.abs(support_values)) > 1e-8:
            max_exceed_ratio = np.max(exceed / (np.abs(support_values) + 1e-8))
        else:
            max_exceed_ratio = np.max(exceed)
        
        risk_support = max(0.0, min(1.0, max_exceed_ratio))
        
        # === 步骤 4: 碰撞检测 ===
        predicted_states = self._predict_trajectory(num_steps=25)  # 预测 0.5 秒 (25 步@50Hz)
        collision_warning, min_dist = self.check_collision(predicted_states)
        
        # 碰撞风险
        if collision_warning:
            risk_collision = max(0.0, min(1.0, 1.0 - min_dist / 0.1))
        else:
            risk_collision = 0.0
        
        # === 步骤 5: 融合风险 ===
        safe = safe_support and not collision_warning
        risk = max(risk_support, risk_collision)
        
        return safe, float(risk), collision_warning

# ============================================================================
# RTA 融合中心
# ============================================================================

class RTAFusionCenter:
    def __init__(self, weights=(0.3, 0.4, 0.3)):
        self.weights = weights
    
    def fuse(self, risk_r1, risk_r2, risk_r3):
        return float(0.3*risk_r1 + 0.4*risk_r2 + 0.3*risk_r3)
    
    def decide_intervention(self, risk_total):
        if risk_total >= 0.6:
            return 'takeover', 0.0
        elif risk_total >= 0.4:
            return 'slowdown', 0.5
        elif risk_total >= 0.2:
            return 'slowdown', 0.8
        else:
            return 'none', 1.0

# ============================================================================
# RTA 指标统计器（带时间窗口）
# ============================================================================

class RTAMetrics:
    """
    RTA 指标统计器（带预警时间窗口）
    
    核心逻辑:
    - 以危险事件为中心，检查预警是否在危险前 n 步内
    - 窗口内预警 → TP
    - 窗口外预警 → FP
    - 无预警的危险 → FN
    - 既无危险也无预警 → TN
    """
    
    def __init__(self, warning_window=25):
        self.warning_window = warning_window  # 预警窗口大小（步数）
        self.danger_steps = []
        self.alert_steps = []
        self.latencies = []
        self.total_steps = 0
        
        # 分层统计
        self.r1_alerts = 0
        self.r2_alerts = 0
        self.r3_alerts = 0
        
        # 分模块统计
        self.r1_joint = 0
        self.r1_velocity = 0
        self.r1_collision = 0
        self.r2_support = 0
        self.r2_collision = 0
        self.r3_logic = 0
        self.r3_activation = 0
        self.r3_ood = 0
        
        # 风险分数历史
        self.risk_r1_history = []
        self.risk_r2_history = []
        self.risk_r3_history = []
        
        # 详细日志（用于诊断 FP 来源）
        self.detailed_logs = []
    
    def record(self, step, safe_r1, risk_r1, violations_r1,
                     safe_r2, risk_r2, collision_warning_r2,
                     safe_r3, risk_r3, anomaly_modules_r3,
                     dangers, latency_ms):
        """记录单个时间步"""
        self.total_steps += 1
        self.latencies.append(latency_ms)
        
        # 风险分数历史
        self.risk_r1_history.append(risk_r1)
        self.risk_r2_history.append(risk_r2)
        self.risk_r3_history.append(risk_r3)
        
        # 判断危险
        is_danger = len(dangers) > 0
        if is_danger:
            self.danger_steps.append(step)
        
        # 判断各层预警
        alert_r1 = (not safe_r1) or (risk_r1 > 0.5)
        alert_r2 = (not safe_r2) or (risk_r2 > 0.5)
        alert_r3 = (not safe_r3) or (risk_r3 > 0.5)
        is_alert = alert_r1 or alert_r2 or alert_r3
        
        if is_alert:
            self.alert_steps.append(step)
        
        # 分层统计
        if alert_r1:
            self.r1_alerts += 1
            if 'joint_limit' in violations_r1:
                self.r1_joint += 1
            if 'velocity_limit' in violations_r1:
                self.r1_velocity += 1
            if 'collision' in violations_r1:
                self.r1_collision += 1
        
        if alert_r2:
            self.r2_alerts += 1
            if risk_r2 > 0.5:
                self.r2_support += 1
            if collision_warning_r2:
                self.r2_collision += 1
        
        if alert_r3:
            self.r3_alerts += 1
            if 'logic' in anomaly_modules_r3:
                self.r3_logic += 1
            if 'activation' in anomaly_modules_r3:
                self.r3_activation += 1
            if 'ood' in anomaly_modules_r3:
                self.r3_ood += 1
        
        # 详细日志
        self.detailed_logs.append({
            'step': step,
            'is_danger': is_danger,
            'is_alert': is_alert,
            'alert_r1': alert_r1,
            'alert_r2': alert_r2,
            'alert_r3': alert_r3,
        })
    
    def _compute_confusion_matrix(self):
        """带时间窗口的混淆矩阵计算"""
        TP = 0
        FN = 0
        FP = 0
        
        # 步骤 1: 为每个危险事件检查是否有预警在窗口内
        for d_step in self.danger_steps:
            window_start = max(0, d_step - self.warning_window)
            window_end = d_step - 1
            
            has_alert = any(
                window_start <= a_step <= window_end 
                for a_step in self.alert_steps
            )
            
            if has_alert:
                TP += 1
            else:
                FN += 1
        
        # 步骤 2: 计算 FP（预警但不在任何危险窗口内）
        for a_step in self.alert_steps:
            in_any_window = False
            
            for d_step in self.danger_steps:
                window_start = max(0, d_step - self.warning_window)
                window_end = d_step - 1
                
                if window_start <= a_step <= window_end:
                    in_any_window = True
                    break
            
            if not in_any_window:
                FP += 1
        
        # 步骤 3: 计算 TN
        meaningful_steps = set(self.danger_steps) | set(self.alert_steps)
        TN = self.total_steps - len(meaningful_steps)
        
        return TP, TN, FP, FN
    
    def compute_metrics(self):
        """计算完整指标"""
        TP, TN, FP, FN = self._compute_confusion_matrix()
        
        # 基础指标
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 预警提前时间
        lead_times = []
        for d_step in self.danger_steps:
            window_start = max(0, d_step - self.warning_window)
            window_end = d_step - 1
            
            alerts_in_window = [
                a_step for a_step in self.alert_steps 
                if window_start <= a_step <= window_end
            ]
            
            if alerts_in_window:
                first_alert = min(alerts_in_window)
                lead_steps = d_step - first_alert
                lead_times.append(lead_steps / 50.0)  # 转换为秒
        
        # 密度
        danger_ratio = len(self.danger_steps) / self.total_steps
        alert_ratio = len(self.alert_steps) / self.total_steps
        
        return {
            # 混淆矩阵
            'tp': TP, 'tn': TN, 'fp': FP, 'fn': FN,
            'precision': precision, 'recall': recall, 'fpr': fpr, 'f1_score': f1,
            
            # 预警提前时间
            'avg_lead_time_s': np.mean(lead_times) if lead_times else None,
            'max_lead_time_s': np.max(lead_times) if lead_times else None,
            'num_danger_events': len(self.danger_steps),
            'num_covered_events': TP,
            
            # 时延
            'avg_latency_ms': np.mean(self.latencies),
            'max_latency_ms': np.max(self.latencies),
            
            # 密度
            'danger_ratio': danger_ratio,
            'alert_ratio': alert_ratio,
            'total_steps': self.total_steps,
            
            # 分层统计
            'r1_alerts': self.r1_alerts,
            'r2_alerts': self.r2_alerts,
            'r3_alerts': self.r3_alerts,
            'r1_alert_ratio': self.r1_alerts / self.total_steps,
            'r2_alert_ratio': self.r2_alerts / self.total_steps,
            'r3_alert_ratio': self.r3_alerts / self.total_steps,
            
            # 分模块统计
            'r1_joint_ratio': self.r1_joint / self.total_steps,
            'r1_velocity_ratio': self.r1_velocity / self.total_steps,
            'r1_collision_ratio': self.r1_collision / self.total_steps,
            'r2_support_ratio': self.r2_support / self.total_steps,
            'r2_collision_ratio': self.r2_collision / self.total_steps,
            'r3_logic_ratio': self.r3_logic / self.total_steps,
            'r3_activation_ratio': self.r3_activation / self.total_steps,
            'r3_ood_ratio': self.r3_ood / self.total_steps,
            
            # 风险分数统计
            'risk_r1_mean': np.mean(self.risk_r1_history),
            'risk_r2_mean': np.mean(self.risk_r2_history),
            'risk_r3_mean': np.mean(self.risk_r3_history),
        }
    
    def diagnose(self):
        """诊断分析"""
        diagnosis = []
        metrics = self.compute_metrics()
        
        # 1. 危险密度
        if metrics['danger_ratio'] > 0.3:
            diagnosis.append(f"⚠️ 危险密度过高：{metrics['danger_ratio']*100:.1f}%")
        
        # 2. 预警密度
        if metrics['alert_ratio'] > 0.5:
            diagnosis.append(f"⚠️ 预警密度过高：{metrics['alert_ratio']*100:.1f}%")
            
            if metrics['r1_alert_ratio'] > 0.3:
                diagnosis.append(f"  → Region 1 报警过多：{metrics['r1_alert_ratio']*100:.1f}%")
                if metrics['r1_joint_ratio'] > 0.2:
                    diagnosis.append(f"    - 关节限位：{metrics['r1_joint_ratio']*100:.1f}%")
                if metrics['r1_velocity_ratio'] > 0.2:
                    diagnosis.append(f"    - 速度限制：{metrics['r1_velocity_ratio']*100:.1f}%")
                if metrics['r1_collision_ratio'] > 0.2:
                    diagnosis.append(f"    - 碰撞检测：{metrics['r1_collision_ratio']*100:.1f}%")
            
            if metrics['r2_alert_ratio'] > 0.3:
                diagnosis.append(f"  → Region 2 报警过多：{metrics['r2_alert_ratio']*100:.1f}%")
                if metrics['r2_support_ratio'] > 0.2:
                    diagnosis.append(f"    - 支撑函数超出：{metrics['r2_support_ratio']*100:.1f}%")
                if metrics['r2_collision_ratio'] > 0.2:
                    diagnosis.append(f"    - 碰撞预警：{metrics['r2_collision_ratio']*100:.1f}%")
            
            if metrics['r3_alert_ratio'] > 0.3:
                diagnosis.append(f"  → Region 3 报警过多：{metrics['r3_alert_ratio']*100:.1f}%")
                if metrics['r3_logic_ratio'] > 0.2:
                    diagnosis.append(f"    - 逻辑异常：{metrics['r3_logic_ratio']*100:.1f}%")
                if metrics['r3_activation_ratio'] > 0.2:
                    diagnosis.append(f"    - 激活异常：{metrics['r3_activation_ratio']*100:.1f}%")
                if metrics['r3_ood_ratio'] > 0.2:
                    diagnosis.append(f"    - OOD 异常：{metrics['r3_ood_ratio']*100:.1f}%")
        
        # 3. 虚警率
        if metrics['fpr'] > 0.3:
            diagnosis.append(f"🔴 虚警率过高：{metrics['fpr']*100:.1f}%")
        
        # 4. 预警提前时间
        if metrics['avg_lead_time_s'] is not None:
            if metrics['avg_lead_time_s'] < 0.1:
                diagnosis.append(f"⚠️ 预警提前时间过短：{metrics['avg_lead_time_s']*1000:.0f}ms")
            elif metrics['avg_lead_time_s'] >= 0.5:
                diagnosis.append(f"✅ 预警提前时间优秀：{metrics['avg_lead_time_s']*1000:.0f}ms")
        
        # 5. 漏报
        if metrics['num_danger_events'] - metrics['num_covered_events'] > 0:
            diagnosis.append(f"⚠️ 漏报 {metrics['num_danger_events'] - metrics['num_covered_events']} 次危险事件")
        
        return diagnosis if diagnosis else ["✅ 无显著问题"]


# ============================================================================
# 危险检测器 (真正的物理危险)
# ============================================================================

class DangerDetector:
    def __init__(self, velocity_limit=5.0, collision_dist=0.02, skip_steps=20):
        self.velocity_limit = velocity_limit
        self.collision_dist = collision_dist
        self.skip_steps = skip_steps
        self.step_count = 0
        self.obstacles = None
    
    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
    
    def check_all(self, qpos, qvel):
        self.step_count += 1
        
        if self.step_count < self.skip_steps:
            return [], float('inf')
        
        dangers = []
        min_distance = float('inf')
        
        # 碰撞检测
        if self.obstacles:
            for obs in self.obstacles:
                dist = np.linalg.norm(qpos[:3] - np.array(obs['pos'][:3]))
                if dist < self.collision_dist:
                    dangers.append('collision_obstacle')
                min_distance = min(min_distance, dist)
        else:
            min_distance = 1.0
        
        # 速度检测
        max_vel = np.max(np.abs(qvel))
        if max_vel > self.velocity_limit:
            dangers.append('velocity_violation')
        
        return dangers, min_distance

# ============================================================================
# 梯度计算
# ============================================================================

def compute_gradient(policy, qpos_tensor, image_tensor, device='cuda'):
    """
    计算梯度矩阵 ∂a/∂qpos
    
    使用数值梯度近似：
    ∂a_i/∂qpos_j ≈ (a_i(qpos + ε*e_j) - a_i(qpos)) / ε
    """
    eps = 1e-4
    qpos_np = qpos_tensor.cpu().numpy()[0]  # (14,)
    
    with torch.no_grad():
        base_output = policy(qpos_tensor, image_tensor)
        if isinstance(base_output, dict):
            base_action = base_output['action'][0]  # (14,)
        else:
            base_action = base_output[0]
    
    gradient = np.zeros((14, 14))  # (action_dim, state_dim)
    
    for j in range(14):
        qpos_perturbed = qpos_np.copy()
        qpos_perturbed[j] += eps
        
        qpos_tensor_pert = torch.from_numpy(qpos_perturbed).float().to(device).unsqueeze(0)
        
        with torch.no_grad():
            pert_output = policy(qpos_tensor_pert, image_tensor)
            if isinstance(pert_output, dict):
                pert_action = pert_output['action'][0]
            else:
                pert_action = pert_output[0]
        
        gradient[:, j] = (pert_action.cpu().numpy() - base_action.cpu().numpy()) / eps
    
    return gradient

# ============================================================================
# 在线测试器 (集成真正的 Region3Detector)
# ============================================================================

class OnlineRTATester:
    def __init__(self, policy, r1_config, r2_model_path, r2_direction_matrix_path, r3_model_dir, output_dir, device='cuda', warning_window=25):
        self.policy = policy
        self.device = device
        self.region1 = Region1Monitor(**r1_config)
        self.region2 = Region2Monitor(r2_model_path, r2_direction_matrix_path)
        
        # 使用真正的 Region3Detector
        self.region3 = Region3Detector(r3_model_dir, load_learned_thresholds=True)
        
        self.fusion = RTAFusionCenter()
        self.danger_detector = DangerDetector()
        self.output_dir = output_dir
        self.warning_window = warning_window
        os.makedirs(output_dir, exist_ok=True)
        
        # 注册 Hook 用于提取激活值
        self.encoder_hook = NetworkHook('encoder_last_layer')
        encoder_last_layer = self.policy.model.encoder.layers[-1]
        self.encoder_hook.register(encoder_last_layer.linear2)
        
        # ACT 没有 decoder，设为 None
        self.decoder_hook = None
        
        self.all_steps = []
        self.all_trials = []
    
    def run_trial(self, scene_id, fault_id, trial_id, max_steps=250):
        """运行单次试验（纯检测，无干预）"""
        scene = BASE_SCENES.get(scene_id, BASE_SCENES['B1'])
        fault = FAULT_TYPES.get(fault_id, FAULT_TYPES['None'])
        
        env = make_sim_env('sim_transfer_cube_scripted')
        fault_injector = FaultInjector(fault_id, inject_step=50) if fault_id != 'None' else None
        
        # 设置障碍物
        if scene['obstacles'] > 0:
            obstacles = self._generate_obstacles(scene['obstacles'])
            if hasattr(env.task, '_set_obstacles'):
                env.task._set_obstacles(obstacles)
            self.danger_detector.set_obstacles(obstacles)
            self.region1.set_obstacles(obstacles)
            self.region2.set_obstacles(obstacles)
        else:
            self.danger_detector.set_obstacles(None)
            self.region1.set_obstacles(None)
            self.region2.set_obstacles(None)
        
        qpos = sample_box_pose()
        qpos = np.append(qpos, BOX_POSE[0])
        
        # 使用 RTAMetrics 记录数据
        metrics_tracker = RTAMetrics(warning_window=self.warning_window)
        
        t_first_alert = None
        t_first_danger = None
        trial_steps = []
        
        for step in range(max_steps):
            start_time = time.time()
            
            qpos_obs = qpos[:14].copy()
            qvel_obs = qpos[14:28].copy() if len(qpos) > 14 else np.zeros(14)
            
            qpos_tensor = torch.from_numpy(qpos_obs).float().to(self.device).unsqueeze(0)
            curr_images = [rearrange(ts.observation['images'][cam], 'h w c -> c h w') 
                          for cam in ['top']]
            image_tensor = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().to(self.device).unsqueeze(0)
            
            self.encoder_hook.clear()
            
            with torch.no_grad():
                all_actions = self.policy(qpos_tensor, image_tensor)
                action_vla = all_actions[:, 0].detach().cpu().numpy()[0]
            
            encoder_act = self.encoder_hook.get_output()
            if encoder_act is not None:
                if len(encoder_act.shape) > 2:
                    encoder_act = encoder_act[0, -1, :].cpu().numpy()
                elif len(encoder_act.shape) == 2:
                    encoder_act = encoder_act[0, :].cpu().numpy()
                else:
                    encoder_act = encoder_act[0].cpu().numpy()
            else:
                encoder_act = np.zeros(512)
            
            decoder_act = np.zeros(128)
            
            gradient = compute_gradient(self.policy, qpos_tensor, image_tensor, self.device)
            
            self.region2.update_history(qpos_obs, qvel_obs)
            
            # Region 1 检测（传入 violations 用于诊断）
            safe_r1, violations_r1, risk_r1 = self.region1.check(qpos_obs, qvel_obs)
            
            # Region 2 检测
            safe_r2, risk_r2, collision_warning_r2 = self.region2.check(qpos_obs, qvel_obs, action_vla)
            
            # Region 3 检测
            region3_result = self.region3.detect(
                qpos=qpos_obs,
                qvel=qvel_obs,
                action=action_vla,
                gradient=gradient,
                encoder_act=encoder_act,
                decoder_act=decoder_act
            )
            
            safe_r3 = not region3_result['is_anomaly']
            risk_r3 = self._compute_region3_risk(region3_result)
            anomaly_modules_r3 = region3_result['anomaly_modules']
            
            # 预警判断（OR 逻辑）
            alert_any = (not safe_r1 or risk_r1 > 0.5) or \
                        (not safe_r2 or risk_r2 > 0.5) or \
                        (not safe_r3 or risk_r3 > 0.5)
            
            if alert_any:
                if t_first_alert is None:
                    t_first_alert = step
            
            # ⭐ 不加干预，纯检测模式
            action_executed = action_vla
            
            # 故障注入
            if fault_injector:
                action_executed = fault_injector.inject_action(action_executed)
            
            # 危险检测（Ground Truth）
            dangers, min_dist = self.danger_detector.check_all(qpos_obs, qvel_obs)
            
            if len(dangers) > 0:
                if t_first_danger is None:
                    t_first_danger = step
            
            # ⭐ 使用 RTAMetrics 记录（窗口化混淆矩阵）
            metrics_tracker.record(
                step=step,
                safe_r1=safe_r1, risk_r1=risk_r1, violations_r1=violations_r1,
                safe_r2=safe_r2, risk_r2=risk_r2, collision_warning_r2=collision_warning_r2,
                safe_r3=safe_r3, risk_r3=risk_r3, anomaly_modules_r3=anomaly_modules_r3,
                dangers=dangers,
                latency_ms=latency_ms
            )
            
            # 保存详细数据
            step_data = {
                'episode': trial_id,
                'step': step,
                'scene': scene_id,
                'fault': fault_id,
                'risk_r1': risk_r1,
                'risk_r2': risk_r2,
                'risk_r3': risk_r3,
                'alert_any': int(alert_any),
                'dangers': '|'.join(dangers) if dangers else '',
                'distance': min_dist,
                'modality_id': region3_result['modality_id'],
                'S_logic': region3_result['S_logic'],
                'D_ham': region3_result['D_ham'],
                'D_ood_norm': region3_result['D_ood_norm'],
            }
            trial_steps.append(step_data)
            
            ts = env.step(action_executed)
            qpos = ts.observation['qpos'].copy()
        
        # 计算窗口化混淆矩阵指标
        metrics = metrics_tracker.compute_metrics()
        
        # 添加试验基本信息
        metrics['episode_id'] = trial_id
        metrics['scene'] = scene_id
        metrics['fault'] = fault_id
        metrics['t_first_alert'] = t_first_alert
        metrics['t_first_danger'] = t_first_danger
        
        # 诊断分析
        diagnosis = metrics_tracker.diagnose()
        metrics['diagnosis'] = diagnosis
        
        self.all_steps.extend(trial_steps)
        self.all_trials.append(metrics)
        
        return metrics
    
    def _compute_region3_risk(self, result):
        """从 Region3Detector 结果计算风险分数"""
        anomaly_modules = len(result['anomaly_modules'])
        
        if anomaly_modules == 0:
            return 0.1
        elif anomaly_modules == 1:
            return 0.4
        elif anomaly_modules == 2:
            return 0.7
        else:
            return 1.0
    
    def _generate_obstacles(self, num_obstacles):
        obstacles = []
        for i in range(num_obstacles):
            obstacles.append({
                'pos': [0.3 + np.random.uniform(-0.2, 0.2),
                       np.random.uniform(-0.3, 0.3),
                       0.05],
                'size': [0.05, 0.05, 0.1]
            })
        return obstacles
    
    def save_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = os.path.join(self.output_dir, f'trial_{timestamp}')
        
        with open(f'{base_path}.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'step', 'scene', 'fault', 'risk_r1', 'risk_r2', 'risk_r3',
                'risk_total', 'alert_any', 'intervention', 'dangers', 'distance', 'reward',
                'modality_id', 'S_logic', 'D_ham', 'D_ood_norm'
            ])
            writer.writeheader()
            writer.writerows(self.all_steps)
        
        summary_path = f'{base_path}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(self.all_trials, f, indent=2)
        
        print(f"\n✓ 结果已保存：{base_path}.*")
        
        return base_path

# ============================================================================
# 主函数
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备：{device}")
    
    ckpt_dir = './ckpts/my_transfer_cube_model'
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    
    print(f"\n📦 加载 ACT 策略：{ckpt_path}")
    
    # ACTPolicy 使用 args_override 字典
    # 根据 checkpoint 推断配置：dec_layers=7, hidden_dim=512, dim_feedforward=3200, num_queries=100
    args_override = {
        'enc_layers': 4,
        'dec_layers': 7,          # checkpoint 有 7 层 decoder
        'dim_feedforward': 3200,   # 根据 checkpoint 推断
        'hidden_dim': 512,         # 根据 checkpoint 推断
        'dropout': 0.1,
        'nheads': 8,
        'num_queries': 100,        # checkpoint 的 query_embed 是 (100, 512)
        'qpos_noise_std': 0.0,
        'kl_weight': 10.0,
        'num_actions': 14,
        'input_dim': 28,
        'camera_names': ['top'],
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'weight_decay': 1e-4,
        'lr_drop': 50,
        'clip_max_norm': 10,
        'backbone': 'resnet18',
        'position_embedding': 'sine',
    }
    
    policy = ACTPolicy(args_override)
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint)
    policy.to(device)
    policy.eval()
    
    print(f"  ✓ 策略已加载")
    
    r1_config = {
        'qpos_min': -2.0,
        'qpos_max': 2.0,
        'qvel_max': 1.0,
    }
    
    # Region 2: GRU 模型和方向矩阵
    r2_model_path = './outputs/region2_gru/gru_reachability_best.pth'
    r2_direction_matrix_path = './outputs/region2_gru/support_directions.npy'
    
    # Region 3: 检测器目录
    r3_model_dir = './outputs/region3_detectors'
    
    output_dir = './outputs/rta_online_v3_real_region3'
    
    tester = OnlineRTATester(
        policy=policy,
        r1_config=r1_config,
        r2_model_path=r2_model_path,
        r2_direction_matrix_path=r2_direction_matrix_path,
        r3_model_dir=r3_model_dir,
        output_dir=output_dir,
        device=device
    )
    
    print(f"\n🧪 开始测试...")
    
    test_configs = [
        ('B2', 'F1', 0),
        ('B2', 'F1', 1),
        ('B2', 'F2', 0),
    ]
    
    for scene, fault, trial in test_configs:
        print(f"\n=== 试验 {trial}: {scene}_{fault} ===")
        metrics = tester.run_trial(scene, fault, trial)
        
        print(f"  成功：{'✅' if metrics['success'] else '❌'}")
        print(f"  碰撞：{'🛡️ 无' if not metrics['collision'] else '💥 有'}")
        print(f"  预警：{metrics['total_alerts']} 次 (首次 t={metrics['t_first_alert']})")
        print(f"  危险：首次 t={metrics['t_first_danger']}")
        print(f"  指标：P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1_score']:.2f}")
        print(f"  时延：{metrics['avg_latency_ms']:.1f}ms (max {metrics['max_latency_ms']:.1f}ms)")
    
    tester.save_results()
    
    print("\n" + "="*60)
    print("✅ 测试完成!")
    print("="*60)

if __name__ == '__main__':
    main()
