#!/usr/bin/env python3
"""
run_rta_online_test.py - 三层 RTA 在线测试脚本 (集成实际环境)

功能:
1. 完整场景矩阵 (B1-B3 × F1-F8)
2. 多类型危险事件检测 (碰撞/自碰撞/超限/任务失败)
3. 危险窗口机制 (危险前 N 步算正样本)
4. 实时统计 + CSV/JSON/Markdown 报告
5. 集成 ALOHA 仿真环境和 ACT 策略
"""

import sys
import os

# 必须在导入任何项目模块前设置 sys.argv，避免 constants.py 的 argparse 退出
sys.argv = [
    'run_rta_online_test',
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
import json
import csv
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from einops import rearrange

# 保存原始命令行参数 (在导入 constants 之前会被覆盖)
_ORIGINAL_ARGS = sys.argv[1:] if len(sys.argv) > 1 else []

# 导入实际环境和策略
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.aloha_sim import ALOHASimulationEnv

# ============================================================================
# 场景与故障定义
# ============================================================================

BASE_SCENES = {
    'B1': {'name': '空旷', 'obstacles': 0, 'difficulty': 1, 'scene_id': 'empty'},
    'B2': {'name': '静态障碍', 'obstacles': 5, 'difficulty': 2, 'scene_id': 'static'},
    'B3': {'name': '密集障碍', 'obstacles': 10, 'difficulty': 3, 'scene_id': 'dense'},
}

FAULT_TYPES = {
    'F1': {'name': '光照突变', 'type': '感知', 'fault_id': 'F1_lighting'},
    'F2': {'name': '摄像头遮挡', 'type': '感知', 'fault_id': 'F2_occlusion'},
    'F3': {'name': '对抗补丁', 'type': '感知', 'fault_id': 'F3_adversarial'},
    'F4': {'name': '负载突变', 'type': '动力学', 'fault_id': 'F4_payload'},
    'F5': {'name': '关节摩擦', 'type': '动力学', 'fault_id': 'F5_friction'},
    'F6': {'name': '突发障碍', 'type': '突发', 'fault_id': 'F6_dynamic'},
    'F7': {'name': '传感器噪声', 'type': '感知', 'fault_id': 'F7_sensor'},
    'F8': {'name': '复合故障', 'type': '复合', 'fault_id': 'F8_compound'},
    'None': {'name': '正常', 'type': '正常', 'fault_id': None},
}

DANGER_TYPES = [
    'collision_obstacle',      # 碰撞障碍物
    'collision_table',         # 碰撞桌面
    'self_collision',          # 自碰撞
    'workspace_violation',     # 工作空间超限
    'velocity_violation',      # 速度超限
    'acceleration_violation',  # 加速度超限
]


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class StepData:
    """单步数据"""
    episode: int
    step: int
    scene: str
    fault: str
    qpos: List[float]
    qvel: List[float]
    action_vla: List[float]
    action_executed: List[float]
    risk_r1: float
    risk_r2: float
    risk_r3: float
    risk_total: float
    alert_r1: bool
    alert_r2: bool
    alert_r3: bool
    alert_any: bool
    intervention: str
    slowdown_factor: float
    dangers: List[str]
    distance_to_obstacle: float
    reward: float


@dataclass
class DangerEvent:
    """危险事件"""
    t: int
    danger_type: str
    severity: float
    details: Dict


@dataclass
class TrialMetrics:
    """单次试验指标"""
    episode_id: int
    scene: str
    fault: str
    success: bool
    total_reward: float
    total_steps: int
    danger_events: List[DangerEvent]
    collision: bool
    collision_t: Optional[int]
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    false_positive_rate: float
    f1_score: float
    t_fault_inject: Optional[int]
    t_first_alert: Optional[int]
    t_first_danger: Optional[int]
    warning_lead_time: Optional[float]
    total_alerts: int
    intervention_count: int
    slowdown_steps: int
    stop_steps: int
    avg_inference_latency_ms: float
    max_inference_latency_ms: float


# ============================================================================
# 故障注入器
# ============================================================================

class FaultInjector:
    """故障注入器"""
    
    def __init__(self, fault_type: str, inject_step: int = 50):
        self.fault_type = fault_type
        self.inject_step = inject_step
        self.fault_active = False
    
    def inject(self, state: np.ndarray, action: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """注入故障"""
        if step < self.inject_step:
            return state, action
        
        self.fault_active = True
        
        if self.fault_type == 'F1_lighting':
            # 光照突变 - 影响视觉 (简化：在状态上加噪声模拟感知误差)
            state = state + np.random.normal(0, 0.1, state.shape)
        
        elif self.fault_type == 'F2_occlusion':
            # 相机遮挡 - 状态估计误差
            state = state + np.random.normal(0, 0.15, state.shape)
        
        elif self.fault_type == 'F3_adversarial':
            # 对抗补丁 - 状态扰动
            state = state + np.random.normal(0, 0.2, state.shape)
        
        elif self.fault_type == 'F4_payload':
            # 负载突变 - 动作效果增强
            action = action * 1.3
        
        elif self.fault_type == 'F5_friction':
            # 关节摩擦 - 动作衰减
            action = action * 0.6
        
        elif self.fault_type == 'F6_dynamic':
            # 动态障碍 - 状态约束变化
            state = np.clip(state, -2.5, 2.5)
        
        elif self.fault_type == 'F7_sensor':
            # 传感器噪声
            state = state + np.random.normal(0, 0.2, state.shape)
        
        elif self.fault_type == 'F8_compound':
            # 复合故障：F1 + F4 + F5
            state = state + np.random.normal(0, 0.1, state.shape)
            action = action * 1.3 * 0.6
        
        return state, action


# ============================================================================
# 策略模型 (简化 ACT)
# ============================================================================

class SimpleACTPolicy(nn.Module):
    """简化 ACT 策略 - 用于测试"""
    
    def __init__(self, state_dim=14, action_dim=14, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state, images=None):
        """前向传播"""
        # state: (B, state_dim)
        action = self.network(state)
        return action.unsqueeze(1)  # (B, 1, action_dim)


class RealACTPolicy(nn.Module):
    """真实 ACT 策略包装器"""
    
    def __init__(self, ckpt_path, device='cpu'):
        super().__init__()
        self.device = device
        self.ckpt_path = ckpt_path
        
        # 延迟导入，避免循环依赖
        try:
            from policy import ACTPolicy
            from constants import SIM_TASK_CONFIGS
            
            # 创建策略
            config = SIM_TASK_CONFIGS['sim_transfer_cube_scripted']
            self.policy = ACTPolicy(config)
            
            # 加载检查点
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy.to(device)
            self.policy.eval()
            
            self.has_real_policy = True
            print(f"  ✓ 真实 ACT 策略已加载：{ckpt_path}")
            
        except Exception as e:
            print(f"  ⚠ 真实 ACT 策略加载失败：{e}")
            print(f"  ℹ 使用简化策略替代")
            self.has_real_policy = False
            self.policy = SimpleACTPolicy().to(device)
    
    def forward(self, state, images=None):
        """前向传播"""
        if self.has_real_policy:
            # 真实 ACT 需要图像输入
            # 简化：用零图像替代
            if images is None:
                # 创建伪图像 (2 个相机，3 通道，480x640)
                batch_size = state.shape[0] if len(state.shape) > 1 else 1
                images = torch.zeros(batch_size, 2, 3, 480, 640).to(self.device)
            
            # ACT 推理
            all_actions = self.policy(state, images)
            return all_actions
        else:
            # 简化策略
            action = self.policy(state)
            return action


# ============================================================================
# 危险检测器
# ============================================================================

class DangerDetector:
    """多类型危险事件检测器"""
    
    def __init__(self, workspace_bounds=None, velocity_limit=0.6, accel_limit=1.0):
        self.workspace_bounds = workspace_bounds or {
            'x': (-0.5, 0.5),
            'y': (-0.5, 0.5),
            'z': (-0.1, 0.5),
        }
        self.velocity_limit = velocity_limit
        self.accel_limit = accel_limit
        self.prev_qvel = None
    
    def check_collision(self, state, obstacles=None):
        """碰撞检测"""
        dangers = []
        min_distance = float('inf')
        
        # 用正向运动学计算末端位置
        ee_pos = self._forward_kinematics(state)
        
        # 障碍物碰撞
        if obstacles:
            for obs in obstacles:
                obs_pos = np.array(obs['pos'])
                obs_size = np.array(obs['size'])
                dist = np.max(np.abs(ee_pos[:2] - obs_pos[:2]) - obs_size[:2])
                if dist < 0.05:
                    dangers.append('collision_obstacle')
                min_distance = min(min_distance, dist)
        else:
            min_distance = 1.0
        
        # 桌面碰撞 (z < 0)
        if ee_pos[2] < -0.02:
            dangers.append('collision_table')
        
        return dangers, min_distance
    
    def check_self_collision(self, state):
        """自碰撞检测"""
        dangers = []
        # 简化：检查左右臂关节角度是否冲突
        left_arm = state[:6]
        right_arm = state[6:12]
        
        if np.any(np.abs(left_arm - right_arm) > 2.5):
            dangers.append('self_collision')
        
        return dangers
    
    def check_workspace(self, state):
        """工作空间检测"""
        dangers = []
        ee_pos = self._forward_kinematics(state)
        
        if not (self.workspace_bounds['x'][0] <= ee_pos[0] <= self.workspace_bounds['x'][1]):
            dangers.append('workspace_violation')
        if not (self.workspace_bounds['y'][0] <= ee_pos[1] <= self.workspace_bounds['y'][1]):
            dangers.append('workspace_violation')
        if not (self.workspace_bounds['z'][0] <= ee_pos[2] <= self.workspace_bounds['z'][1]):
            dangers.append('workspace_violation')
        
        return dangers
    
    def check_velocity(self, qvel):
        """速度检测"""
        dangers = []
        if np.max(np.abs(qvel)) > self.velocity_limit:
            dangers.append('velocity_violation')
        return dangers
    
    def check_acceleration(self, qvel):
        """加速度检测"""
        dangers = []
        if self.prev_qvel is not None:
            accel = np.abs(np.array(qvel) - np.array(self.prev_qvel))
            if np.max(accel) > self.accel_limit:
                dangers.append('acceleration_violation')
        self.prev_qvel = qvel.copy()
        return dangers
    
    def _forward_kinematics(self, state):
        """简化的正向运动学"""
        # 左臂
        left_ee = np.array([
            0.3 * np.sin(state[0]) * np.cos(state[1]),
            0.3 * np.sin(state[1]),
            0.3 * np.cos(state[0]) * np.cos(state[1])
        ])
        # 右臂
        right_ee = np.array([
            0.3 * np.sin(state[6]) * np.cos(state[7]),
            0.3 * np.sin(state[7]),
            0.3 * np.cos(state[6]) * np.cos(state[7])
        ])
        return (left_ee + right_ee) / 2
    
    def check_all(self, state, qvel, obstacles=None):
        """综合检测所有危险类型"""
        all_dangers = []
        
        coll_dangers, min_dist = self.check_collision(state, obstacles)
        all_dangers.extend(coll_dangers)
        all_dangers.extend(self.check_self_collision(state))
        all_dangers.extend(self.check_workspace(state))
        all_dangers.extend(self.check_velocity(qvel))
        all_dangers.extend(self.check_acceleration(qvel))
        
        return list(set(all_dangers)), min_dist


# ============================================================================
# 危险窗口管理器
# ============================================================================

class DangerWindowManager:
    """管理危险时间窗口"""
    
    def __init__(self, window_steps=25):
        self.window_steps = window_steps
        self.danger_windows = []
    
    def register_danger(self, t: int, danger_type: str, duration: int = 10):
        """注册危险事件"""
        window_start = max(0, t - self.window_steps)
        window_end = t + duration
        self.danger_windows.append((window_start, window_end, danger_type))
    
    def is_in_danger_window(self, t: int) -> Tuple[bool, List[str]]:
        """判断是否处于危险窗口"""
        in_window = False
        active_dangers = []
        
        for start, end, dtype in self.danger_windows:
            if start <= t <= end:
                in_window = True
                active_dangers.append(dtype)
        
        return in_window, list(set(active_dangers))
    
    def compute_confusion_matrix(self, alerts_by_step: Dict[int, bool]) -> Dict:
        """计算混淆矩阵"""
        tp = fp = tn = fn = 0
        
        for t, alerted in alerts_by_step.items():
            in_danger, _ = self.is_in_danger_window(t)
            
            if alerted and in_danger:
                tp += 1
            elif alerted and not in_danger:
                fp += 1
            elif not alerted and in_danger:
                fn += 1
            else:
                tn += 1
        
        return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


# ============================================================================
# Region 1: 物理硬约束
# ============================================================================

class Region1Monitor:
    def __init__(self, qpos_min=-3.0, qpos_max=3.0, qvel_max=0.6):
        self.qpos_min = qpos_min
        self.qpos_max = qpos_max
        self.qvel_max = qvel_max
    
    def check(self, qpos, qvel):
        violations = []
        risk_score = 0.0
        
        qpos_margin = min(np.min(qpos - self.qpos_min), np.min(self.qpos_max - qpos))
        if qpos_margin < 0:
            violations.append('joint_limit')
            risk_score = 1.0
        elif qpos_margin < 0.3:
            risk_score = max(risk_score, 1.0 - (qpos_margin / 0.3))
        
        qvel_ratio = np.max(np.abs(qvel)) / self.qvel_max
        if qvel_ratio > 1.0:
            violations.append('velocity_limit')
            risk_score = max(risk_score, 1.0)
        elif qvel_ratio > 0.8:
            risk_score = max(risk_score, (qvel_ratio - 0.8) / 0.2)
        
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
    def __init__(self, model_path=None, safe_boundaries=None):
        self.history_buffer = []
        self.max_history = 10
        self.safe_boundaries = safe_boundaries if safe_boundaries is not None else np.ones(16) * 1.5
        
        self.has_model = False
        self.model = None
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.has_model = True
                    self.config = checkpoint.get('config', {})
                    self.model = DeepReachabilityGRU(
                        input_dim=self.config.get('input_dim', 28),
                        hidden_dim=self.config.get('hidden_dim', 192),
                        num_layers=self.config.get('num_layers', 4),
                        output_dim=16,
                        dropout=self.config.get('dropout', 0.4)
                    )
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                    print(f"  ✓ Region 2 模型已加载：{model_path}")
            except Exception as e:
                print(f"  ⚠ Region 2 模型加载失败：{e}")
    
    def update_history(self, qpos, qvel):
        # qpos: 12 维 (双臂各 6 关节), qvel: 12 维
        # 连接成 24 维状态
        state = np.concatenate([qpos[:12], qvel[:12]])
        self.history_buffer.append(state)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
    
    def check(self, qpos, qvel, action):
        if not self.has_model or len(self.history_buffer) < self.max_history:
            # 无模型时用简单启发式
            max_vel = np.max(np.abs(qvel))
            risk = min(1.0, max_vel / 0.6)
            return risk < 0.7, float(risk)
        
        history = np.array(self.history_buffer)
        
        # 模型期望输入维度：(B, seq_len, input_dim)
        # input_dim = 28 (根据模型配置)
        # 当前历史：(10, 24) - 需要填充到 28 维
        if history.shape[1] < 28:
            # 用零填充到 28 维
            padded = np.zeros((history.shape[0], 28))
            padded[:, :history.shape[1]] = history
            history = padded
        
        history_tensor = torch.from_numpy(history).float().unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(history_tensor).numpy()[0]
        
        max_ratio = np.max(prediction / self.safe_boundaries)
        safe = max_ratio <= 1.0
        risk = max(0.0, min(1.0, (max_ratio - 0.8) / 0.2))
        
        return safe, float(risk)


# ============================================================================
# Region 3: 感知异常检测
# ============================================================================

class Region3Monitor:
    def __init__(self, model_dir=None, threshold=0.5):
        self.model_dir = model_dir
        self.threshold = threshold
        self.has_model = model_dir is not None and os.path.exists(model_dir)
        
        if self.has_model:
            print(f"  ✓ Region 3 模型目录：{model_dir}")
    
    def check(self, action, qpos, qvel, state=None):
        """
        简化实现：用状态/动作的统计特性模拟异常检测
        实际应该用激活链路 + OOD + 梯度
        """
        if not self.has_model:
            # 无模型时用简单启发式
            state_norm = np.linalg.norm(state) if state is not None else 0
            action_norm = np.linalg.norm(action)
            
            # 异常分数
            anomaly_score = 0.0
            if state_norm > 5.0:
                anomaly_score += 0.3
            if action_norm > 1.0:
                anomaly_score += 0.3
            anomaly_score += np.random.uniform(0, 0.2)
            
            risk = min(1.0, anomaly_score)
            safe = risk < self.threshold
            return safe, float(risk), [], {'anomaly_score': risk}
        
        # 有模型时加载实际检测器
        # 这里是简化版本
        risk = np.random.uniform(0, 0.3)
        safe = risk < self.threshold
        return safe, float(risk), [], {}


# ============================================================================
# RTA 融合中心
# ============================================================================

class RTAFusionCenter:
    def __init__(self, weights=(0.3, 0.4, 0.3)):
        self.weights = weights
    
    def fuse(self, risk_r1, risk_r2, risk_r3):
        """融合三层风险"""
        risk_total = (
            self.weights[0] * risk_r1 +
            self.weights[1] * risk_r2 +
            self.weights[2] * risk_r3
        )
        return float(risk_total)
    
    def decide_intervention(self, risk_total):
        """根据总风险决定干预措施"""
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
    def __init__(self, r1_config, r2_model_path, r3_model_dir, output_dir, device='cpu'):
        self.device = device
        self.region1 = Region1Monitor(**r1_config)
        self.region2 = Region2Monitor(r2_model_path)
        self.region3 = Region3Monitor(r3_model_dir)
        self.fusion = RTAFusionCenter()
        self.danger_detector = DangerDetector()
        self.danger_window_mgr = DangerWindowManager(window_steps=25)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.all_steps = []
        self.all_trials = []
    
    def create_policy(self, ckpt_path=None):
        """创建策略模型"""
        if ckpt_path and os.path.exists(ckpt_path):
            # 尝试加载真实 ACT 策略
            policy = RealACTPolicy(ckpt_path, self.device)
        else:
            # 使用简化策略
            policy = SimpleACTPolicy(state_dim=14, action_dim=14, hidden_dim=256).to(self.device)
            print(f"  ℹ 使用随机初始化策略 (无 ckpt: {ckpt_path})")
        
        policy.eval()
        return policy
    
    def run_episode(self, episode_id: int, scene: str, fault: str, 
                    policy: nn.Module, seed: int = None) -> TrialMetrics:
        """运行单次试验"""
        
        # 创建环境
        scene_id = BASE_SCENES.get(scene, {}).get('scene_id', 'empty')
        fault_id = FAULT_TYPES.get(fault, {}).get('fault_id')
        
        env = ALOHASimulationEnv(scene=scene_id, fault_type=fault_id, seed=seed)
        fault_injector = FaultInjector(fault_id, inject_step=50) if fault_id else None
        
        # 初始化
        state = env.reset()
        self.region2.history_buffer = []
        self.danger_window_mgr.danger_windows = []
        
        step_data_list = []
        alerts_by_step = {}
        danger_events = []
        
        total_reward = 0
        collision = False
        collision_t = None
        t_fault_inject = None
        t_first_alert = None
        t_first_danger = None
        
        total_alerts = 0
        intervention_count = 0
        slowdown_steps = 0
        stop_steps = 0
        latencies = []
        
        for step in range(250):
            start_time = time.time()
            
            # 获取观测
            obs = env.get_observation()
            qpos = obs[:12]
            qvel = obs[12:]
            
            # 策略推理
            state_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                all_actions = policy(state_tensor)
                action_vla = all_actions[0, 0].cpu().numpy()
            
            # 更新 Region 2 历史
            self.region2.update_history(qpos, qvel)
            
            # 三层 RTA 检测
            safe_r1, _, risk_r1 = self.region1.check(qpos, qvel)
            safe_r2, risk_r2 = self.region2.check(qpos, qvel, action_vla)
            safe_r3, risk_r3, _, _ = self.region3.check(action_vla, qpos, qvel, obs)
            
            # 风险融合
            risk_total = self.fusion.fuse(risk_r1, risk_r2, risk_r3)
            
            # 预警标志
            alert_r1 = not safe_r1 or risk_r1 > 0.5
            alert_r2 = not safe_r2 or risk_r2 > 0.5
            alert_r3 = not safe_r3 or risk_r3 > 0.5
            alert_any = alert_r1 or alert_r2 or alert_r3
            
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
                obs, action_executed = fault_injector.inject(obs, action_executed, step)
                if t_fault_inject is None and fault_injector.fault_active:
                    t_fault_inject = step
            
            # 环境步
            next_state, reward, done, info = env.step(action_executed)
            total_reward += reward
            
            # 危险检测
            obstacles = env.obstacles if hasattr(env, 'obstacles') else None
            dangers, min_distance = self.danger_detector.check_all(obs, qvel, obstacles)
            
            # 注册危险事件
            if dangers:
                if t_first_danger is None:
                    t_first_danger = step
                for d in dangers:
                    danger_events.append(DangerEvent(t=step, danger_type=d, severity=0.8, details={}))
                    self.danger_window_mgr.register_danger(step, d, duration=10)
                
                if 'collision_obstacle' in dangers or 'collision_table' in dangers:
                    collision = True
                    if collision_t is None:
                        collision_t = step
            
            alerts_by_step[step] = alert_any
            
            # 记录步数据
            step_data = StepData(
                episode=episode_id,
                step=step,
                scene=scene,
                fault=fault,
                qpos=qpos.tolist(),
                qvel=qvel.tolist(),
                action_vla=action_vla.tolist(),
                action_executed=action_executed.tolist(),
                risk_r1=float(risk_r1),
                risk_r2=float(risk_r2),
                risk_r3=float(risk_r3),
                risk_total=float(risk_total),
                alert_r1=alert_r1,
                alert_r2=alert_r2,
                alert_r3=alert_r3,
                alert_any=alert_any,
                intervention=intervention,
                slowdown_factor=float(slowdown_factor),
                dangers=dangers,
                distance_to_obstacle=float(min_distance),
                reward=float(reward)
            )
            step_data_list.append(step_data)
            self.all_steps.append(asdict(step_data))
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
            
            if done:
                break
        
        # 计算混淆矩阵
        cm = self.danger_window_mgr.compute_confusion_matrix(alerts_by_step)
        tp, fp, tn, fn = cm['tp'], cm['fp'], cm['tn'], cm['fn']
        
        # 计算指标
        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        warning_lead_time = None
        if t_first_alert is not None and t_first_danger is not None:
            warning_lead_time = (t_first_danger - t_first_alert) / 50.0
        
        success = total_reward > -50 and not collision
        
        metrics = TrialMetrics(
            episode_id=episode_id,
            scene=scene,
            fault=fault,
            success=success,
            total_reward=total_reward,
            total_steps=step + 1,
            danger_events=danger_events,
            collision=collision,
            collision_t=collision_t,
            tp=tp, fp=fp, tn=tn, fn=fn,
            precision=precision,
            recall=recall,
            false_positive_rate=fpr,
            f1_score=f1,
            t_fault_inject=t_fault_inject,
            t_first_alert=t_first_alert,
            t_first_danger=t_first_danger,
            warning_lead_time=warning_lead_time,
            total_alerts=total_alerts,
            intervention_count=intervention_count,
            slowdown_steps=slowdown_steps,
            stop_steps=stop_steps,
            avg_inference_latency_ms=np.mean(latencies),
            max_inference_latency_ms=np.max(latencies)
        )
        
        self.all_trials.append(metrics)
        return metrics
    
    def save_trial_csv(self, metrics: TrialMetrics):
        """保存 CSV"""
        filename = f"trial_{metrics.episode_id:03d}_{metrics.scene}_{metrics.fault}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        trial_steps = [s for s in self.all_steps if s['episode'] == metrics.episode_id]
        
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['episode', 'step', 'scene', 'fault', 'risk_r1', 'risk_r2', 
                         'risk_r3', 'risk_total', 'alert_any', 'intervention', 
                         'dangers', 'distance_to_obstacle', 'reward']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for step in trial_steps:
                row = {k: step[k] for k in fieldnames}
                row['dangers'] = '|'.join(row['dangers']) if row['dangers'] else ''
                writer.writerow(row)
    
    def save_trial_json(self, metrics: TrialMetrics):
        """保存 JSON"""
        filename = f"trial_{metrics.episode_id:03d}_summary.json"
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            'episode_id': int(metrics.episode_id),
            'scene': str(metrics.scene),
            'fault': str(metrics.fault),
            'success': bool(metrics.success),
            'total_reward': float(metrics.total_reward),
            'total_steps': int(metrics.total_steps),
            'collision': bool(metrics.collision),
            'collision_t': int(metrics.collision_t) if metrics.collision_t is not None else None,
            'confusion_matrix': {'tp': int(metrics.tp), 'fp': int(metrics.fp), 'tn': int(metrics.tn), 'fn': int(metrics.fn)},
            'metrics': {
                'precision': float(metrics.precision), 
                'recall': float(metrics.recall), 
                'false_positive_rate': float(metrics.false_positive_rate), 
                'f1_score': float(metrics.f1_score)
            },
            'timing': {
                't_fault_inject': int(metrics.t_fault_inject) if metrics.t_fault_inject is not None else None,
                't_first_alert': int(metrics.t_first_alert) if metrics.t_first_alert is not None else None,
                't_first_danger': int(metrics.t_first_danger) if metrics.t_first_danger is not None else None,
                'warning_lead_time': float(metrics.warning_lead_time) if metrics.warning_lead_time is not None else None
            },
            'intervention': {
                'total_alerts': int(metrics.total_alerts), 
                'intervention_count': int(metrics.intervention_count), 
                'slowdown_steps': int(metrics.slowdown_steps), 
                'stop_steps': int(metrics.stop_steps)
            },
            'performance': {
                'avg_inference_latency_ms': float(metrics.avg_inference_latency_ms), 
                'max_inference_latency_ms': float(metrics.max_inference_latency_ms)
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_trial_summary(self, metrics: TrialMetrics):
        """打印摘要"""
        status = "✅" if metrics.success else "❌"
        collision_str = f"💥 碰撞 (t={metrics.collision_t})" if metrics.collision else "🛡️ 无碰撞"
        
        print(f"\n{status} Ep {metrics.episode_id:03d} [{metrics.scene}_{metrics.fault}]:")
        print(f"  {collision_str}")
        print(f"  预警：{metrics.total_alerts} 次 (首次 t={metrics.t_first_alert})")
        print(f"  干预：{metrics.intervention_count} 次 (减速 {metrics.slowdown_steps} 步)")
        print(f"  混淆矩阵：TP={metrics.tp}, FP={metrics.fp}, TN={metrics.tn}, FN={metrics.fn}")
        print(f"  精准率：{metrics.precision*100:.1f}%, 召回率：{metrics.recall*100:.1f}%, FPR={metrics.false_positive_rate*100:.1f}%")
        print(f"  F1 分数：{metrics.f1_score*100:.1f}%")
        if metrics.warning_lead_time is not None:
            print(f"  ⏱️ 提前预警：{metrics.warning_lead_time:.2f}s")
        print(f"  ⚡ 时延：{metrics.avg_inference_latency_ms:.1f}ms (max {metrics.max_inference_latency_ms:.1f}ms)")
    
    def generate_aggregate_report(self) -> str:
        """生成汇总报告"""
        report = []
        report.append("# RTA 在线测试汇总报告")
        report.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## 总览")
        report.append(f"- 总试验数：{len(self.all_trials)} 次")
        report.append(f"- 总步数：{len(self.all_steps)} 步")
        report.append(f"- 成功率：{np.mean([t.success for t in self.all_trials])*100:.1f}%")
        report.append(f"- 碰撞率：{np.mean([t.collision for t in self.all_trials])*100:.1f}%")
        report.append("")
        report.append("## 预警性能 (按场景×故障)")
        report.append("")
        report.append("| 场景 | 故障 | 试验数 | Precision | Recall | FPR | F1 | 提前时间 (s) |")
        report.append("|------|------|--------|-----------|--------|-----|----|-------------|")
        
        groups = defaultdict(list)
        for trial in self.all_trials:
            groups[(trial.scene, trial.fault)].append(trial)
        
        for (scene, fault), trials in sorted(groups.items()):
            n = len(trials)
            prec = np.mean([t.precision for t in trials])
            rec = np.mean([t.recall for t in trials])
            fpr = np.mean([t.false_positive_rate for t in trials])
            f1 = np.mean([t.f1_score for t in trials])
            lead_times = [t.warning_lead_time for t in trials if t.warning_lead_time is not None]
            lead_time = np.mean(lead_times) if lead_times else 0.0
            report.append(f"| {scene} | {fault} | {n} | {prec:.2f} | {rec:.2f} | {fpr:.2f} | {f1:.2f} | {lead_time:.2f} |")
        
        report.append("")
        report_path = os.path.join(self.output_dir, "aggregate_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\n📄 汇总报告：{report_path}")
        return '\n'.join(report)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    # 忽略外部传入的参数，使用我们自己的参数
    # 因为 constants.py 的 argparse 会吞掉所有参数
    sys.argv = [
        'run_rta_online_test',
        '--r1_qpos_min', '-3.0',
        '--r1_qpos_max', '3.0',
        '--r1_qvel_max', '0.6',
        '--r2_model', './outputs/region2_gru/gru_reachability_best.pth',
        '--r3_model', './outputs/region3_complete',
        '--policy_ckpt', './ckpts/my_transfer_cube_model/policy_best.ckpt',
        '--output_dir', './outputs/rta_online_tests',
        '--scenes', 'B1',
        '--faults', 'None',
        '--trials_per_config', '1',
        '--device', 'cpu',
    ]
    
    parser = argparse.ArgumentParser(description='RTA 在线测试')
    parser.add_argument('--r1_qpos_min', type=float, default=-3.0)
    parser.add_argument('--r1_qpos_max', type=float, default=3.0)
    parser.add_argument('--r1_qvel_max', type=float, default=0.6)
    parser.add_argument('--r2_model', type=str, default='./outputs/region2_gru/gru_reachability_best.pth')
    parser.add_argument('--r3_model', type=str, default='./outputs/region3_complete')
    parser.add_argument('--policy_ckpt', type=str, default='./ckpts/policy_best.ckpt')
    parser.add_argument('--output_dir', type=str, default='./outputs/rta_online_tests')
    parser.add_argument('--scenes', type=str, nargs='+', default=['B1', 'B2', 'B3'])
    parser.add_argument('--faults', type=str, nargs='+', default=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'None'])
    parser.add_argument('--trials_per_config', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    print("="*70)
    print("RTA 三层系统在线测试 (集成实际环境)")
    print("="*70)
    print(f"场景：{args.scenes}")
    print(f"故障：{args.faults}")
    print(f"每配置试验数：{args.trials_per_config}")
    print(f"输出目录：{args.output_dir}")
    print("="*70)
    
    tester = OnlineRTATester(
        r1_config={'qpos_min': args.r1_qpos_min, 'qpos_max': args.r1_qpos_max, 'qvel_max': args.r1_qvel_max},
        r2_model_path=args.r2_model,
        r3_model_dir=args.r3_model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    policy = tester.create_policy(args.policy_ckpt)
    
    print("\n" + "="*70)
    print("开始测试...")
    print("="*70)
    
    episode_id = 0
    for scene in args.scenes:
        for fault in args.faults:
            for trial in range(args.trials_per_config):
                metrics = tester.run_episode(episode_id, scene, fault, policy, seed=episode_id)
                tester.save_trial_csv(metrics)
                tester.save_trial_json(metrics)
                tester.print_trial_summary(metrics)
                episode_id += 1
    
    print("\n" + "="*70)
    tester.generate_aggregate_report()
    print("="*70)
    print("测试完成!")
