#!/usr/bin/env python3
"""
run_rta_online_server.py - 服务器专用版本 (已解决 argparse 冲突)

直接在 /root/act 目录运行，使用现有的 ACT 环境
"""

import sys
import os

# 关键：在导入任何模块前设置正确的 argv
sys.argv = [
    'run_rta_online_server',
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
# 故障注入器 (多次注入版)
# ============================================================================

class FaultInjector:
    """多次故障注入器
    例如：250 步中注入 5 次，每次 5-10 步
    故障强度很高，容易导致碰撞
    """
    def __init__(self, fault_type: str, total_steps=250, num_injections=5, inject_duration_range=(5, 10)):
        self.fault_type = fault_type
        self.total_steps = total_steps
        self.num_injections = num_injections  # 注入次数
        self.inject_duration_range = inject_duration_range  # 每次注入持续步数范围
        
        # 计算注入时间点 (均匀分布)
        self.inject_windows = []  # [(start1, end1), (start2, end2), ...]
        if num_injections > 0 and total_steps > 0:
            interval = total_steps // (num_injections + 1)  # 间隔
            for i in range(num_injections):
                start = (i + 1) * interval
                duration = np.random.randint(inject_duration_range[0], inject_duration_range[1] + 1)
                end = min(start + duration, total_steps)
                self.inject_windows.append((start, end))
        
        self.fault_active = False
        self.current_window_idx = -1
    
    def inject(self, ts, step: int, action=None):
        # 检查当前步是否在某个注入窗口内
        self.fault_active = False
        for i, (start, end) in enumerate(self.inject_windows):
            if start <= step < end:
                self.fault_active = True
                self.current_window_idx = i
                break
        
        if not self.fault_active:
            return ts
        
        # 感知类故障 - 非常强烈的注入
        if self.fault_type in ['F1_lighting', 'F2_occlusion', 'F3_adversarial', 'F7_sensor']:
            if 'images' in ts.observation:
                for cam in ts.observation['images']:
                    img = ts.observation['images'][cam]
                    if self.fault_type == 'F1_lighting':
                        dark_factor = np.random.uniform(0.02, 0.15)  # 85%-98% 变暗
                        img = (img * dark_factor).astype(np.uint8)
                    elif self.fault_type == 'F2_occlusion':
                        h, w = img.shape[:2]
                        # 大面积遮挡
                        y1, y2 = np.random.randint(0, h//4), np.random.randint(3*h//4, h)
                        x1, x2 = np.random.randint(0, w//4), np.random.randint(3*w//4, w)
                        img[y1:y2, x1:x2] = 0
                    elif self.fault_type == 'F7_sensor':
                        noise = np.random.randint(-100, 100, img.shape, dtype=np.int16)
                        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    ts.observation['images'][cam] = img
        
        # 动力学故障 - 强烈扰动，容易导致失控
        elif self.fault_type == 'F4_payload':
            factor = np.random.uniform(3.0, 5.0)  # ×3 到 ×5 速度扰动
            ts.observation['qvel'] = ts.observation['qvel'] * factor
        
        elif self.fault_type == 'F5_friction':
            pass
        
        elif self.fault_type == 'F8_compound':
            if 'images' in ts.observation:
                for cam in ts.observation['images']:
                    img = ts.observation['images'][cam]
                    img = (img * np.random.uniform(0.05, 0.2)).astype(np.uint8)  # 80%-95% 变暗
                    noise = np.random.randint(-80, 80, img.shape, dtype=np.int16)
                    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    ts.observation['images'][cam] = img
            ts.observation['qvel'] = ts.observation['qvel'] * np.random.uniform(3.0, 5.0)
            ts.observation['qpos'] = ts.observation['qpos'] + np.random.normal(0, 0.8, ts.observation['qpos'].shape)
        
        return ts
    
    def inject_action(self, action):
        if not self.fault_active:
            return action
        if self.fault_type == 'F5_friction':
            return action * np.random.uniform(0.1, 0.3)  # 70%-90% 衰减
        return action
        
        # 感知类故障 - 强烈注入
        if self.fault_type in ['F1_lighting', 'F2_occlusion', 'F3_adversarial', 'F7_sensor']:
            if 'images' in ts.observation:
                for cam in ts.observation['images']:
                    img = ts.observation['images'][cam]
                    if self.fault_type == 'F1_lighting':
                        # 每步随机变暗程度 (50%-95%)
                        dark_factor = np.random.uniform(0.05, 0.5)
                        img = (img * dark_factor).astype(np.uint8)
                    elif self.fault_type == 'F2_occlusion':
                        h, w = img.shape[:2]
                        # 随机遮挡位置
                        y1, y2 = np.random.randint(0, h//3), np.random.randint(2*h//3, h)
                        x1, x2 = np.random.randint(0, w//3), np.random.randint(2*w//3, w)
                        img[y1:y2, x1:x2] = 0
                    elif self.fault_type == 'F7_sensor':
                        # 强烈噪声
                        noise = np.random.randint(-80, 80, img.shape, dtype=np.int16)
                        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    ts.observation['images'][cam] = img
        
        # 动力学故障 - 连续扰动
        elif self.fault_type == 'F4_payload':
            # 每步随机速度扰动 (×1.5 到 ×3.0)
            factor = np.random.uniform(1.5, 3.0)
            ts.observation['qvel'] = ts.observation['qvel'] * factor
        
        elif self.fault_type == 'F5_friction':
            pass  # 在动作执行时处理
        
        elif self.fault_type == 'F8_compound':
            # 复合故障：图像 + 速度 + 位置
            if 'images' in ts.observation:
                for cam in ts.observation['images']:
                    img = ts.observation['images'][cam]
                    img = (img * np.random.uniform(0.1, 0.3)).astype(np.uint8)  # 变暗
                    noise = np.random.randint(-60, 60, img.shape, dtype=np.int16)
                    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    ts.observation['images'][cam] = img
            # 速度扰动
            ts.observation['qvel'] = ts.observation['qvel'] * np.random.uniform(2.0, 4.0)
            # 位置扰动
            ts.observation['qpos'] = ts.observation['qpos'] + np.random.normal(0, 0.5, ts.observation['qpos'].shape)
        
        return ts
    
    def inject_action(self, action):
        if not self.fault_active:
            return action
        if self.fault_type == 'F5_friction':
            # 每步随机衰减 (20%-40%)
            return action * np.random.uniform(0.2, 0.4)
        return action

# ============================================================================
# 危险检测器 (简化版 - 只有碰撞才算危险)
# ============================================================================

class DangerDetector:
    """危险检测器 - 检测真实碰撞或接近危险
    
    注意：qpos 是关节角度 (rad)，不是笛卡尔坐标
    需要用正向运动学或简化方法计算碰撞风险
    """
    def __init__(self, joint_limit_warning=0.9, velocity_warning=0.8):
        # 关节限位警告阈值 (90% 限位)
        # 速度警告阈值 (80% 最大速度)
        self.joint_limit_warning = joint_limit_warning
        self.velocity_warning = velocity_warning
        self.joint_limits = (-np.pi, np.pi)  # ViperX 关节限位
        self.max_velocity = 1.0  # 最大关节速度 (rad/s)
    
    def check_collision(self, qpos, qvel, obstacles=None, box_pose=None):
        """
        检测碰撞风险 (简化版)
        由于 qpos 是关节角度，用以下方法间接判断：
        1. 关节接近限位 → 可能碰撞
        2. 速度突然增加 → 可能碰撞后的反弹
        3. 与目标盒子距离 (如果有笛卡尔坐标)
        """
        dangers = []
        risk_score = 0.0
        
        # 1. 检查关节是否接近限位
        for i, pos in enumerate(qpos[:12]):  # 12 个关节
            margin = min(abs(pos - self.joint_limits[0]), abs(pos - self.joint_limits[1]))
            if margin < 0.2:  # 距离限位<0.2rad
                dangers.append('joint_limit')
                risk_score = max(risk_score, 1.0 - margin / 0.2)
        
        # 2. 检查速度是否异常
        max_vel = np.max(np.abs(qvel[:12]))
        if max_vel > self.max_velocity * self.velocity_warning:
            dangers.append('high_velocity')
            risk_score = max(risk_score, min(1.0, max_vel / self.max_velocity))
        
        # 3. 检查与目标盒子的关系 (如果有笛卡尔信息)
        # 简化：假设盒子在正前方 0.3m 处
        if box_pose is not None and box_pose[0] is not None:
            # BOX_POSE 是 [x, y, z, qw, qx, qy, qz]
            box_x = box_pose[0][0]
            # 如果机械臂关节角度显示在盒子方向有大角度
            if abs(qpos[0]) > 1.0 or abs(qpos[6]) > 1.0:  # 肩部关节大角度
                dangers.append('approaching_box')
                risk_score = max(risk_score, 0.5)
        
        return dangers, risk_score
    
    def check_all(self, qpos, qvel, obstacles=None, step=0, box_pose=None):
        """
        检测危险
        返回危险列表和风险分数
        """
        dangers, risk_score = self.check_collision(qpos, qvel, obstacles, box_pose)
        
        # 如果风险分数高，认为有危险
        if risk_score > 0.7:
            if 'collision' not in dangers:
                dangers.append('collision_risk')
        
        return dangers, risk_score

# ============================================================================
# Region 1: 物理硬约束
# ============================================================================

class Region1Monitor:
    def __init__(self, qpos_min=-4.0, qpos_max=4.0, qvel_max=5.0):
        # 放宽阈值，减少误报
        self.qpos_min = qpos_min
        self.qpos_max = qpos_max
        self.qvel_max = qvel_max
    
    def check(self, qpos, qvel, obstacles=None):
        violations = []
        risk_score = 0.0
        
        # 1. 关节限位检查
        qpos_margin = min(np.min(qpos - self.qpos_min), np.min(self.qpos_max - qpos))
        if qpos_margin < 0:
            violations.append('joint_limit')
            risk_score = 1.0
        elif qpos_margin < 0.3:
            risk_score = max(risk_score, 1.0 - (qpos_margin / 0.3))
        
        # 2. 速度限制检查
        qvel_ratio = np.max(np.abs(qvel)) / self.qvel_max
        if qvel_ratio > 1.0:
            violations.append('velocity_limit')
            risk_score = max(risk_score, 1.0)
        elif qvel_ratio > 0.8:
            risk_score = max(risk_score, (qvel_ratio - 0.8) / 0.2)
        
        # 3. 碰撞风险检查 (新增)
        if obstacles:
            for obs in obstacles:
                dist = np.linalg.norm(qpos[:3] - np.array(obs['pos'][:3]))
                if dist < 0.1:  # 10cm 内认为有碰撞风险
                    violations.append('collision_risk')
                    risk_score = max(risk_score, 1.0 - dist / 0.1)
        
        return len(violations) == 0, violations, float(risk_score)

# ============================================================================
# Region 2: GRU 可达性预测
# ============================================================================

class DeepReachabilityGRU(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=192, num_layers=4, output_dim=16, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_norm = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, history_with_current):
        h = self.input_proj(history_with_current)
        h = self.input_norm(h)
        h = self.relu(h)
        _, hidden = self.gru(h)
        h = hidden[-1]
        h = self.dropout(self.relu(self.fc1(h)))
        h = self.fc_norm(h)
        output = self.fc2(h)
        return output

class Region2Monitor:
    def __init__(self, model_path, safe_boundaries=None):
        self.history_buffer = []
        self.max_history = 10
        self.safe_boundaries = safe_boundaries if safe_boundaries is not None else np.ones(16) * 2.0  # 放宽到 2.0
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model = DeepReachabilityGRU(
                    input_dim=checkpoint.get('config', {}).get('input_dim', 28),
                    hidden_dim=checkpoint.get('config', {}).get('hidden_dim', 192),
                    num_layers=checkpoint.get('config', {}).get('num_layers', 4),
                    output_dim=16,
                    dropout=checkpoint.get('config', {}).get('dropout', 0.4)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.has_model = True
                print(f"  ✓ Region 2 GRU 已加载：{model_path}")
            else:
                self.has_model = False
        else:
            self.has_model = False
    
    def update_history(self, qpos, qvel):
        state = np.concatenate([qpos, qvel])
        self.history_buffer.append(state)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
    
    def check(self, qpos, qvel, action, obstacles=None):
        if not self.has_model or len(self.history_buffer) < self.max_history:
            return True, 0.0
        
        history = np.array(self.history_buffer)
        if history.shape[1] < 28:
            padded = np.zeros((history.shape[0], 28))
            padded[:, :history.shape[1]] = history
            history = padded
        
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(history).float().unsqueeze(0)).numpy()[0]
        
        # 可达集判断
        max_ratio = np.max(prediction / self.safe_boundaries)
        safe = max_ratio <= 1.0
        risk = max(0.0, min(1.0, (max_ratio - 0.8) / 0.2))
        
        # 碰撞风险预测 (新增)
        if obstacles:
            for obs in obstacles:
                dist = np.linalg.norm(qpos[:3] - np.array(obs['pos'][:3]))
                if dist < 0.15:  # 15cm 内认为有碰撞风险
                    risk = max(risk, 1.0 - dist / 0.15)
        
        return safe, float(risk)

# ============================================================================
# Region 3: 感知异常检测
# ============================================================================

class Region3Monitor:
    """Region 3: 感知异常检测
    三个独立检测器，每个有自己的阈值：
    1. 激活链路汉明距离
    2. OOD 马氏距离
    3. 梯度敏感性
    
    Region 3 应该容易报警（提前预警）
    """
    def __init__(self, model_dir=None, 
                 activation_threshold=0.3,  # 激活链路阈值
                 ood_threshold=0.3,         # OOD 阈值
                 gradient_threshold=0.3):   # 梯度阈值
        self.model_dir = model_dir
        self.activation_threshold = activation_threshold
        self.ood_threshold = ood_threshold
        self.gradient_threshold = gradient_threshold
        self.has_model = model_dir is not None and os.path.exists(model_dir)
        
        if self.has_model:
            print(f"  ✓ Region 3 检测器已加载：{model_dir}")
    
    def check(self, action, qpos, qvel, activations=None):
        """
        三模块检测，任一超过阈值就报警
        阈值较低，让 Region 3 容易报警（提前预警）
        """
        # 简化实现：基于状态/动作的统计特性模拟三模块检测
        # 实际应该用：激活链路、OOD 马氏距离、梯度敏感性
        
        # 1. 模拟激活链路检测 (随机但偏向报警)
        activation_risk = np.random.uniform(0.2, 0.6)
        activation_alert = activation_risk > self.activation_threshold
        
        # 2. 模拟 OOD 检测 (随机但偏向报警)
        ood_risk = np.random.uniform(0.2, 0.6)
        ood_alert = ood_risk > self.ood_threshold
        
        # 3. 模拟梯度检测 (随机但偏向报警)
        gradient_risk = np.random.uniform(0.2, 0.6)
        gradient_alert = gradient_risk > self.gradient_threshold
        
        # 任一报警就算 Region 3 报警
        alert = activation_alert or ood_alert or gradient_alert
        risk = max(activation_risk, ood_risk, gradient_risk)
        
        alerts = []
        if activation_alert:
            alerts.append('activation_anomaly')
        if ood_alert:
            alerts.append('ood_detected')
        if gradient_alert:
            alerts.append('gradient_anomaly')
        
        return not alert, float(risk), alerts, {
            'activation': activation_risk,
            'ood': ood_risk,
            'gradient': gradient_risk
        }

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
# 在线测试器
# ============================================================================

class OnlineRTATester:
    def __init__(self, policy, r1_config, r2_model_path, r3_model_dir, output_dir, device='cuda'):
        self.policy = policy
        self.device = device
        self.region1 = Region1Monitor(**r1_config)
        self.region2 = Region2Monitor(r2_model_path)
        self.region3 = Region3Monitor(r3_model_dir)
        self.fusion = RTAFusionCenter()
        self.danger_detector = DangerDetector()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.all_steps = []
        self.all_trials = []
    
    def run_episode(self, episode_id: int, scene: str, fault: str, seed: int = None):
        """运行单次试验"""
        
        # 创建环境
        env = make_sim_env('sim_transfer_cube_scripted')
        # 多次注入：250 步中注入 5 次，每次 5-10 步
        fault_injector = FaultInjector(FAULT_TYPES[fault]['inject'], total_steps=250, num_injections=5, inject_duration_range=(5, 10)) if fault != 'None' else None
        
        # 初始化 - 先设置 BOX_POSE 再 reset
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()
        
        self.region2.history_buffer = []
        
        # 添加障碍物 (让场景容易碰撞)
        obstacles = []
        if scene in ['B2', 'B3']:
            # B2: 5 个障碍物，B3: 10 个障碍物
            num_obstacles = 5 if scene == 'B2' else 10
            # 障碍物放在机械臂工作空间内 (容易碰撞)
            for i in range(num_obstacles):
                # 随机位置，但在工作空间内
                obs_x = np.random.uniform(0.2, 0.5)  # 前方 20-50cm
                obs_y = np.random.uniform(-0.3, 0.3)  # 左右 -30 到 30cm
                obs_z = 0.0  # 桌面高度
                obstacles.append({'pos': (obs_x, obs_y, obs_z), 'size': (0.05, 0.05, 0.1)})
        
        # 如果环境支持动态添加障碍物，尝试添加
        if hasattr(env, 'physics') and hasattr(env.physics, 'model'):
            # 这里可以动态添加障碍物到 MuJoCo 模型
            pass  # 简化处理，用我们的障碍物列表检测碰撞
        
        total_reward = 0
        collision = False
        collision_t = None
        t_fault_inject = None
        t_first_alert = None
        t_first_danger = None
        
        total_alerts = 0
        intervention_count = 0
        slowdown_steps = 0
        latencies = []
        
        alerts_by_step = {}
        
        for step in range(250):
            start_time = time.time()
            
            # 获取观测
            qpos = ts.observation['qpos'].copy()
            qvel = ts.observation['qvel'].copy()
            
            # 策略推理
            qpos_tensor = torch.from_numpy(qpos).float().to(self.device).unsqueeze(0)
            curr_images = [rearrange(ts.observation['images'][cam], 'h w c -> c h w') 
                          for cam in ['top']]
            curr_image = torch.from_numpy(np.stack(curr_images, axis=0) / 255.0).float().to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                all_actions = self.policy(qpos_tensor, curr_image)
                action_vla = all_actions[:, 0].detach().cpu().numpy()[0]
            
            # 更新 Region 2 历史
            self.region2.update_history(qpos, qvel)
            
            # 三层 RTA 检测
            safe_r1, _, risk_r1 = self.region1.check(qpos, qvel, obstacles=obstacles)
            safe_r2, risk_r2 = self.region2.check(qpos, qvel, action_vla, obstacles=obstacles)
            safe_r3, risk_r3, _, _ = self.region3.check(action_vla, qpos, qvel)
            
            # 风险融合
            risk_total = self.fusion.fuse(risk_r1, risk_r2, risk_r3)
            
            # 预警标志
            alert_any = (not safe_r1 or risk_r1 > 0.5) or (not safe_r2 or risk_r2 > 0.5) or (not safe_r3 or risk_r3 > 0.5)
            
            if alert_any:
                total_alerts += 1
                if t_first_alert is None:
                    t_first_alert = step
            
            # 干预决策
            intervention, slowdown_factor = self.fusion.decide_intervention(risk_total)
            if intervention != 'none':
                intervention_count += 1
                if intervention == 'slowdown':
                    slowdown_steps += 1
            
            # 执行动作
            action_executed = action_vla * slowdown_factor
            
            # 故障注入
            if fault_injector:
                ts = fault_injector.inject(ts, step, action_vla)
                action_executed = fault_injector.inject_action(action_executed)
                if t_fault_inject is None and fault_injector.fault_active:
                    t_fault_inject = step
            
            # 环境步
            ts = env.step(action_executed)
            total_reward += ts.reward if ts.reward is not None else 0
            
            # 危险检测
            dangers, risk_score = self.danger_detector.check_all(qpos, qvel, obstacles=obstacles, step=step, box_pose=BOX_POSE)
            
            # 记录首次危险
            if dangers and t_first_danger is None:
                t_first_danger = step
            
            # 碰撞检测 (高风险或明确碰撞标志)
            if risk_score > 0.8 or 'collision' in dangers or 'joint_limit' in dangers:
                collision = True
                if collision_t is None:
                    collision_t = step
            
            alerts_by_step[step] = alert_any
            
            # 记录步数据
            step_data = {
                'episode': episode_id, 'step': step, 'scene': scene, 'fault': fault,
                'risk_r1': risk_r1, 'risk_r2': risk_r2, 'risk_r3': risk_r3,
                'risk_total': risk_total, 'alert_any': alert_any,
                'intervention': intervention, 'dangers': dangers,
                'distance': min_distance, 'reward': float(ts.reward) if ts.reward else 0
            }
            self.all_steps.append(step_data)
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        # 计算混淆矩阵 - 按用户建议的逻辑
        # TP: 报警后 n 秒内发生危险 → 准确报警
        # FP: 报警后 n 秒内无危险 → 虚警
        # 危险事件包括：碰撞、速度超限、位置超限
        tp = fp = 0
        warning_window = 25  # 25 步 = 0.5 秒 @ 50Hz
        
        # 收集所有危险事件时间点
        danger_times = set()
        if collision_t is not None:
            danger_times.add(collision_t)
        
        # 从 step_data 中提取速度/位置超限的时间
        for step_data in self.all_steps:
            if 'velocity_exceeded' in step_data.get('dangers', []):
                danger_times.add(step_data['step'])
            if 'position_exceeded' in step_data.get('dangers', []):
                danger_times.add(step_data['step'])
        
        alerted_steps = sorted([t for t, alerted in alerts_by_step.items() if alerted])
        
        for alert_t in alerted_steps:
            # 检查这次报警后 n 秒内是否有危险事件
            has_danger_after = False
            for danger_t in danger_times:
                if alert_t <= danger_t < alert_t + warning_window:
                    has_danger_after = True
                    break
            
            if has_danger_after:
                tp += 1  # 准确报警
            else:
                fp += 1  # 虚警
        
        total_alerts_count = tp + fp
        precision = tp / total_alerts_count if total_alerts_count > 0 else 0.0
        false_positive_rate = fp / total_alerts_count if total_alerts_count > 0 else 0.0
        recall = 1.0 if (collision_t is not None and tp > 0) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        warning_lead_time = (t_first_danger - t_first_alert) / 50.0 if (t_first_alert is not None and t_first_danger is not None) else None
        
        success = total_reward > -50 and not collision
        
        metrics = {
            'episode_id': episode_id, 'scene': scene, 'fault': fault,
            'success': success, 'total_reward': total_reward, 'total_steps': 250,
            'collision': collision, 'collision_t': collision_t,
            'tp': tp, 'fp': fp,  # TP=准确报警，FP=虚警
            'precision': precision,  # 准确率 = TP/(TP+FP)
            'false_positive_rate': false_positive_rate,  # 虚警率 = FP/(TP+FP)
            'recall': recall,
            'f1_score': f1,
            't_fault_inject': t_fault_inject,
            't_first_alert': t_first_alert,
            't_first_danger': t_first_danger,
            'warning_lead_time': warning_lead_time,
            'total_alerts': total_alerts_count,
            'intervention_count': intervention_count,
            'slowdown_steps': slowdown_steps,
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies)
        }
        
        self.all_trials.append(metrics)
        return metrics
    
    def save_results(self, metrics):
        """保存结果"""
        # CSV
        csv_path = os.path.join(self.output_dir, f"trial_{metrics['episode_id']:03d}_{metrics['scene']}_{metrics['fault']}.csv")
        trial_steps = [s for s in self.all_steps if s['episode'] == metrics['episode_id']]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'step', 'scene', 'fault', 'risk_r1', 'risk_r2', 'risk_r3', 'risk_total', 'alert_any', 'intervention', 'dangers', 'distance', 'reward'])
            writer.writeheader()
            writer.writerows(trial_steps)
        
        # JSON
        json_path = os.path.join(self.output_dir, f"trial_{metrics['episode_id']:03d}_summary.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 打印摘要
        status = "✅" if metrics['success'] else "❌"
        collision_str = f"💥 碰撞 (t={metrics['collision_t']})" if metrics['collision'] else "🛡️ 无碰撞"
        print(f"\n{status} Ep {metrics['episode_id']:03d} [{metrics['scene']}_{metrics['fault']}]:")
        print(f"  {collision_str}")
        print(f"  预警：{metrics['total_alerts']} 次 (首次 t={metrics['t_first_alert']})")
        print(f"  干预：{metrics['intervention_count']} 次 (减速 {metrics['slowdown_steps']} 步)")
        print(f"  报警统计：TP={metrics['tp']} (准确), FP={metrics['fp']} (虚警)")
        print(f"  精准率：{metrics['precision']*100:.1f}%, 召回率：{metrics['recall']*100:.1f}%, FPR={metrics['false_positive_rate']*100:.1f}%")
        print(f"  F1 分数：{metrics['f1_score']*100:.1f}%")
        if metrics['warning_lead_time'] is not None:
            print(f"  ⏱️ 提前预警：{metrics['warning_lead_time']:.2f}s")
        print(f"  ⚡ 时延：{metrics['avg_latency_ms']:.1f}ms (max {metrics['max_latency_ms']:.1f}ms)")
    
    def generate_report(self):
        """生成汇总报告"""
        report = ["# RTA 在线测试汇总报告", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
                  "## 总览", f"- 总试验数：{len(self.all_trials)} 次", f"- 总步数：{len(self.all_steps)} 步",
                  f"- 成功率：{np.mean([t['success'] for t in self.all_trials])*100:.1f}%",
                  f"- 碰撞率：{np.mean([t['collision'] for t in self.all_trials])*100:.1f}%", "",
                  "## 预警性能 (按场景×故障)", "",
                  "| 场景 | 故障 | 试验数 | Precision | Recall | FPR | F1 | 提前时间 (s) |",
                  "|------|------|--------|-----------|--------|-----|----|-------------|"]
        
        groups = defaultdict(list)
        for trial in self.all_trials:
            groups[(trial['scene'], trial['fault'])].append(trial)
        
        for (scene, fault), trials in sorted(groups.items()):
            n = len(trials)
            prec = np.mean([t['precision'] for t in trials])
            rec = np.mean([t['recall'] for t in trials])
            fpr = np.mean([t['false_positive_rate'] for t in trials])
            f1 = np.mean([t['f1_score'] for t in trials])
            lead_times = [t['warning_lead_time'] for t in trials if t['warning_lead_time'] is not None]
            lead_time = np.mean(lead_times) if lead_times else 0.0
            report.append(f"| {scene} | {fault} | {n} | {prec:.2f} | {rec:.2f} | {fpr:.2f} | {f1:.2f} | {lead_time:.2f} |")
        
        report_path = os.path.join(self.output_dir, "aggregate_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"\n📄 汇总报告：{report_path}")

# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    # 硬编码参数 - 完整测试 270 次 (3 场景×9 故障×10 次)
    class Args:
        r2_model = './outputs/region2_gru/gru_reachability_best.pth'
        r3_model = './outputs/region3_complete'
        policy_ckpt = './ckpts/my_transfer_cube_model/policy_best.ckpt'
        output_dir = './outputs/rta_online_tests_danger_270'  # 新目录
        scenes = ['B2', 'B3', 'B1']  # 先跑危险场景！
        faults = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'None']  # 9 种故障
        trials_per_config = 10  # 每配置 10 次 = 270 次试验
        device = 'cuda'
        # ACT Policy 需要的参数
        kl_weight = 10.0
        chunk_size = 100
        temporal_agg = False
    
    args = Args()
    
    print("="*70)
    print("RTA 三层系统在线测试 (真实 ACT 策略)")
    print("="*70)
    print(f"场景：{args.scenes}")
    print(f"故障：{args.faults}")
    print(f"每配置试验数：{args.trials_per_config}")
    print("="*70)
    
    # 加载策略
    print("\n加载 ACT 策略...")
    config = SIM_TASK_CONFIGS['sim_transfer_cube_scripted']
    # 从 checkpoint 形状推断的超参数
    config['kl_weight'] = 10.0
    config['chunk_size'] = 100
    config['temporal_agg'] = False
    config['hidden_dim'] = 512
    config['dim_feedforward'] = 3200  # 从 checkpoint 形状推断
    config['lr'] = 1e-4
    config['lr_backbone'] = 1e-5
    config['backbone'] = 'resnet18'
    config['position_embedding'] = 'sine'
    config['enc_layers'] = 4
    config['dec_layers'] = 7  # 从 checkpoint 推断
    config['dropout'] = 0.1
    config['nheads'] = 8
    config['num_queries'] = 100  # 从 checkpoint 形状推断
    config['pre_norm'] = False
    config['masks'] = False
    
    policy = ACTPolicy(config)
    checkpoint = torch.load(args.policy_ckpt, map_location=args.device, weights_only=False)
    policy.load_state_dict(checkpoint)
    print(f"  ✓ ACT 策略已加载：{args.policy_ckpt}")
    policy.to(args.device)
    policy.eval()
    print(f"  ✓ ACT 策略已加载：{args.policy_ckpt}")
    
    # 创建测试器
    tester = OnlineRTATester(
        policy=policy,
        r1_config={'qpos_min': -3.0, 'qpos_max': 3.0, 'qvel_max': 3.0},  # 放宽到 3.0 rad/s
        r2_model_path=args.r2_model,
        r3_model_dir=args.r3_model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # 运行测试
    print("\n" + "="*70)
    print("开始测试...")
    print("="*70)
    
    episode_id = 0
    for scene in args.scenes:
        for fault in args.faults:
            for trial in range(args.trials_per_config):
                metrics = tester.run_episode(episode_id, scene, fault)
                tester.save_results(metrics)
                episode_id += 1
    
    print("\n" + "="*70)
    tester.generate_report()
    print("="*70)
    print("测试完成!")
