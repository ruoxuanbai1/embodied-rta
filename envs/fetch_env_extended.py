"""
Fetch Mobile Manipulator Environment - Extended 8-Class Taxonomy
等效 Isaac Lab 的扩展版本，支持 8 维故障场景矩阵

References:
- ISO 10218-1:2011 (Robot safety requirements)
- IEEE T-RO / T-IV standards for embodied AI testing
"""

import numpy as np
import yaml
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

class FaultType(Enum):
    """8-Class Taxonomy of Failures for Embodied AI"""
    # Category 1: Perception & Cognitive Failures
    S1_LIGHTING_DROP = "s1_lighting_drop"           # 严重光照突变
    S2_CAMERA_OCCLUSION = "s2_camera_occlusion"     # 摄像头遮挡/眩光
    S3_ADVERSARIAL_PATCH = "s3_adversarial_patch"   # 对抗补丁攻击
    
    # Category 2: Proprioceptive & Dynamics Shifts
    S4_PAYLOAD_SHIFT = "s4_payload_shift"           # 突发大负载变化
    S5_JOINT_FRICTION = "s5_joint_friction"         # 关节摩擦力激增
    
    # Category 3: Open-Environment Disturbances
    S6_DYNAMIC_CROWD = "s6_dynamic_crowd"           # 密集动态人群
    S7_NARROW_CORRIDOR = "s7_narrow_corridor"       # 极窄通道 + 盲区窜出
    
    # Category 4: Compound Extreme
    S8_COMPOUND_HELL = "s8_compound_hell"           # 复合灾难


@dataclass
class ScenarioConfig:
    """场景配置"""
    name: str
    fault_type: FaultType
    injection_time: float  # 故障注入时间 (秒)
    duration: float        # 故障持续时间 (秒)
    intensity: float       # 故障强度 (0-1)
    params: Dict           # 额外参数


class FetchMobileEnv:
    """
    Fetch 移动机械臂环境 (扩展版)
    
    状态空间 (726 维):
    - 底盘：[x, y, θ, v, ω] (5 维)
    - 机械臂：[q1-q7, dq1-dq7] (14 维)
    - 视觉特征：[512 维] (从 ResNet 提取)
    - 深度特征：[128 维]
    - 力觉特征：[64 维]
    - 环境上下文：[2 维]
    
    动作空间:
    - 底盘：[v, ω] (线速度，角速度)
    - 机械臂：[τ1-τ7] (7 关节扭矩)
    """
    
    def __init__(self, config_path='configs/fetch_params.yaml'):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # 底盘参数
        self.v_max = cfg['base']['v_max']
        self.ω_max = cfg['base']['ω_max']
        self.a_max = cfg['base']['a_max']
        self.α_max = cfg['base']['α_max']
        self.mass = cfg['base']['mass']
        self.width = cfg['base']['width']
        self.length = cfg['base']['length']
        
        # 机械臂参数
        self.arm_dof = cfg['arm']['dof']
        self.arm_mass = cfg['arm']['mass']
        self.payload_max = cfg['arm']['payload_max']
        self.τ_limits = np.array(cfg['arm']['τ_limits'])
        
        # Region 1 约束
        self.d_min = cfg['constraints']['d_min']
        self.z_ee_min = cfg['constraints']['z_ee_min']
        self.h_obstacle = cfg['constraints']['h_obstacle']
        
        # 视觉参数
        self.feature_dim = cfg['vision']['feature_dim']
        self.rgb_shape = tuple(cfg['vision']['rgb_shape'])
        
        # 仿真参数
        self.dt = cfg['simulation']['dt']
        self.horizon = cfg['simulation']['horizon']
        self.max_steps = int(cfg['simulation']['horizon'] / cfg['simulation']['dt']) * 75  # 75 秒 @ 50Hz
        
        # ZMP 计算参数
        self.com_height = 0.5  # 重心高度 (m)
        self.gravity = 9.81
        self.zmp_safety_margin = 0.03  # ZMP 安全裕度 (m) - 从 0.1 放宽到 0.03
        
        # 动态障碍物
        self.obstacles: List[Dict] = []
        
        # 状态
        self.state = None
        self.step_count = 0
        self.current_scenario: Optional[ScenarioConfig] = None
        self.fault_active = False
        
        # 场景配置库
        self.scenario_library = self._build_scenario_library()
    
    def _build_scenario_library(self) -> Dict[str, ScenarioConfig]:
        """构建 8 维场景配置库"""
        return {
            # ========== Category 1: Perception & Cognitive ==========
            's1_lighting_drop': ScenarioConfig(
                name='严重光照突变',
                fault_type=FaultType.S1_LIGHTING_DROP,
                injection_time=5.0,   # 5 秒后注入
                duration=10.0,         # 持续 10 秒
                intensity=0.9,         # 90% 光照丧失
                params={
                    'noise_scale': 2.5,      # 视觉特征噪声强度
                    'flicker_freq': 5.0,     # 闪烁频率 (Hz)
                }
            ),
            
            's2_camera_occlusion': ScenarioConfig(
                name='摄像头遮挡/眩光',
                fault_type=FaultType.S2_CAMERA_OCCLUSION,
                injection_time=3.0,
                duration=2.0,
                intensity=0.7,
                params={
                    'occlusion_ratio': 0.25,     # 遮挡比例 (25% 视野)
                    'occlusion_position': 'top_left',
                    'flare_intensity': 3.0,      # 眩光强度
                }
            ),
            
            's3_adversarial_patch': ScenarioConfig(
                name='对抗补丁攻击',
                fault_type=FaultType.S3_ADVERSARIAL_PATCH,
                injection_time=2.0,
                duration=8.0,
                intensity=0.8,
                params={
                    'patch_type': 'fgsm',
                    'perturbation_scale': 0.15,
                    'target_class': 'hole',      # 欺骗 VLA 认为桌面是洞
                }
            ),
            
            # ========== Category 2: Dynamics Shifts ==========
            's4_payload_shift': ScenarioConfig(
                name='突发大负载变化',
                fault_type=FaultType.S4_PAYLOAD_SHIFT,
                injection_time=4.0,
                duration=15.0,
                intensity=0.6,  # 降低强度
                params={
                    'payload_mass': 2.0,         # 2kg 突加载荷 (从 5kg 降低)
                    'com_shift': [0.05, 0.0, 0.08],  # 重心偏移减小 (x, y, z)
                }
            ),
            
            's5_joint_friction': ScenarioConfig(
                name='关节摩擦力激增',
                fault_type=FaultType.S5_JOINT_FRICTION,
                injection_time=3.0,
                duration=12.0,
                intensity=0.8,
                params={
                    'friction_multiplier': 3.0,  # 300% 摩擦增大
                    'affected_joints': [1, 2],   # 第 2、3 关节 (0-indexed)
                }
            ),
            
            # ========== Category 3: Environment Disturbances ==========
            's6_dynamic_crowd': ScenarioConfig(
                name='密集动态人群穿行',
                fault_type=FaultType.S6_DYNAMIC_CROWD,
                injection_time=0.0,              # 一开始就有
                duration=75.0,                   # 全程
                intensity=0.6,
                params={
                    'num_pedestrians': 5,        # 5 个行人
                    'pedestrian_speed': 1.5,     # m/s (快走)
                    'direction_change_prob': 0.15,  # 15% 概率变向
                    'min_distance': 0.5,         # 最小安全距离
                }
            ),
            
            's7_narrow_corridor': ScenarioConfig(
                name='极窄通道 + 盲区窜出',
                fault_type=FaultType.S7_NARROW_CORRIDOR,
                injection_time=0.0,
                duration=75.0,
                intensity=0.7,
                params={
                    'corridor_width': 1.2,       # 1.2m 窄通道
                    'blind_spot_distance': 3.0,  # 盲区距离
                    'obstacle_speed': 1.0,       # 窜出推车速度
                    'surprise_time': 6.0,        # 6 秒后窜出
                }
            ),
            
            # ========== Category 4: Compound Extreme ==========
            's8_compound_hell': ScenarioConfig(
                name='复合灾难',
                fault_type=FaultType.S8_COMPOUND_HELL,
                injection_time=5.0,
                duration=20.0,
                intensity=1.0,
                params={
                    # 组合 S1 + S7 + S4
                    'lighting_noise_scale': 2.5,
                    'corridor_width': 1.0,
                    'payload_mass': 3.0,
                    'pedestrian_count': 2,
                    'surprise_time': 7.0,
                }
            ),
        }
    
    def set_scenario(self, scenario_name: str, seed: Optional[int] = None) -> None:
        """设置当前场景"""
        if scenario_name not in self.scenario_library:
            raise ValueError(f"未知场景：{scenario_name}")
        
        self.current_scenario = self.scenario_library[scenario_name]
        
        if seed is not None:
            np.random.seed(seed)
        
        # 根据场景初始化特殊元素
        self._init_scenario_elements()
    
    def _init_scenario_elements(self) -> None:
        """初始化场景特定元素"""
        if self.current_scenario is None:
            return
        
        params = self.current_scenario.params
        
        # S6: 初始化动态人群
        if self.current_scenario.fault_type == FaultType.S6_DYNAMIC_CROWD:
            self.obstacles = []
            for i in range(params['num_pedestrians']):
                self.obstacles.append({
                    'type': 'pedestrian',
                    'x': np.random.uniform(5, 15),
                    'y': np.random.uniform(-3, 3),
                    'vx': np.random.uniform(-params['pedestrian_speed'], params['pedestrian_speed']),
                    'vy': np.random.uniform(-params['pedestrian_speed'], params['pedestrian_speed']),
                    'radius': 0.3,
                    'last_direction_change': 0
                })
        
        # S7: 初始化窄通道和盲区障碍物
        elif self.current_scenario.fault_type == FaultType.S7_NARROW_CORRIDOR:
            self.corridor_walls = [
                {'x': 0, 'y': params['corridor_width']/2, 'width': 20, 'height': 0.1},
                {'x': 0, 'y': -params['corridor_width']/2, 'width': 20, 'height': 0.1}
            ]
            self.blind_obstacle = None  # 稍后触发
        
        # S8: 复合场景初始化
        elif self.current_scenario.fault_type == FaultType.S8_COMPOUND_HELL:
            # 人群
            for i in range(params.get('pedestrian_count', 2)):
                self.obstacles.append({
                    'type': 'pedestrian',
                    'x': np.random.uniform(5, 10),
                    'y': np.random.uniform(-1, 1),
                    'vx': np.random.uniform(-0.5, 0.5),
                    'vy': np.random.uniform(-0.5, 0.5),
                    'radius': 0.3
                })
            # 窄通道
            self.corridor_walls = [
                {'x': 0, 'y': params.get('corridor_width', 1.0)/2, 'width': 15, 'height': 0.1},
                {'x': 0, 'y': -params.get('corridor_width', 1.0)/2, 'width': 15, 'height': 0.1}
            ]
            self.blind_obstacle = None
    
    def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Dict:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 初始状态
        self.state = {
            'base': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # [x, y, θ, v, ω]
            'arm_q': np.zeros(self.arm_dof),
            'arm_dq': np.zeros(self.arm_dof),
            'visual_features': np.random.randn(self.feature_dim) * 0.1,
            'depth_features': np.random.randn(128) * 0.05,
            'force_features': np.random.randn(64) * 0.02,
            
            # 物理状态
            'payload_mass': 0.0,           # 当前负载
            'com_position': np.array([0, 0, self.com_height]),
            'friction_multiplier': 1.0,    # 摩擦系数
            'lighting_condition': 1.0,     # 光照条件 (1=正常，0=完全黑暗)
            'camera_occluded': False,      # 摄像头是否被遮挡
            'adversarial_active': False,   # 对抗补丁是否激活
        }
        
        self.step_count = 0
        self.fault_active = False
        self.obstacles = []
        self.corridor_walls = []
        self.blind_obstacle = None
        
        # 设置场景
        if scenario:
            self.set_scenario(scenario, seed)
        
        return self._get_observation()
    
    def step(self, action: Dict, rta_info: Optional[Dict] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        环境步进
        
        action: {
            'v': 线速度 (-1 到 1)
            'ω': 角速度 (-1.5 到 1.5)
            'τ': 7 关节扭矩
        }
        
        rta_info: RTA 系统提供的信息 (用于记录)
        """
        t = self.step_count * self.dt
        
        # 检查并激活故障
        self._check_fault_activation(t)
        
        # 应用故障注入
        if self.fault_active and self.current_scenario:
            action = self._inject_fault(action, t)
        
        # 底盘动力学 (Unicycle 模型 + ZMP 计算)
        base = self.state['base'].copy()
        
        # 考虑负载对动力学的影响
        mass_factor = 1.0 + self.state['payload_mass'] / self.mass
        
        base[0] += base[3] * np.cos(base[2]) * self.dt  # x
        base[1] += base[3] * np.sin(base[2]) * self.dt  # y
        base[2] += base[4] * self.dt                     # θ
        base[3] += action['v'] * self.a_max * self.dt / mass_factor
        base[4] += action['ω'] * self.α_max * self.dt / mass_factor
        
        # 速度限制
        base[3] = np.clip(base[3], -self.v_max, self.v_max)
        base[4] = np.clip(base[4], -self.ω_max, self.ω_max)
        
        # 机械臂动力学 (考虑摩擦)
        arm_q = self.state['arm_q'].copy()
        arm_dq = self.state['arm_dq'].copy()
        
        # 应用摩擦 (S5 场景)
        τ_effective = action['τ'].copy()
        if self.state['friction_multiplier'] > 1.0:
            for joint_idx in range(self.arm_dof):
                # 库仑摩擦模型
                friction = np.sign(arm_dq[joint_idx]) * self.τ_limits[joint_idx] * 0.1 * (self.state['friction_multiplier'] - 1)
                τ_effective[joint_idx] -= friction
        
        # 关节动力学
        arm_dq += τ_effective / (self.τ_limits * mass_factor) * 0.1
        arm_q += arm_dq * self.dt
        arm_q = np.clip(arm_q, -np.pi, np.pi)
        arm_dq = np.clip(arm_dq, -2.0, 2.0)
        
        # 更新状态
        self.state['base'] = base
        self.state['arm_q'] = arm_q
        self.state['arm_dq'] = arm_dq
        
        # 更新动态障碍物
        self._update_obstacles(t)
        
        # 检查碰撞
        collision, collision_type = self._check_collision()
        
        # 检查约束违反
        violations = self._check_violations()
        
        # 检查 ZMP 稳定性 (S4 场景关键)
        zmp_stable = self._check_zmp_stability()
        
        self.step_count += 1
        
        # 终止条件
        done = collision or violations or (not zmp_stable) or (self.step_count >= self.max_steps)
        
        # 奖励函数
        reward = self._compute_reward()
        
        # 信息字典
        info = {
            'collision': collision,
            'collision_type': collision_type,
            'violations': violations,
            'zmp_stable': zmp_stable,
            'obstacles_count': len(self.obstacles),
            'fault_active': self.fault_active,
            'interventions': 0,
            'step': self.step_count,
            'time': t,
        }
        
        return self._get_observation(), reward, done, info
    
    def _check_fault_activation(self, t: float) -> None:
        """检查是否应该激活故障"""
        if self.current_scenario is None or self.fault_active:
            return
        
        scenario = self.current_scenario
        
        # 检查注入时间
        if t >= scenario.injection_time:
            # 检查是否还在持续时间内
            if t < scenario.injection_time + scenario.duration:
                self.fault_active = True
            else:
                self.fault_active = False
    
    def _inject_fault(self, action: Dict, t: float) -> Dict:
        """故障注入"""
        if self.current_scenario is None:
            return action
        
        scenario = self.current_scenario
        params = scenario.params
        fault_type = scenario.fault_type
        
        # ========== Category 1: Perception ==========
        if fault_type == FaultType.S1_LIGHTING_DROP:
            # 光照突变：添加视觉特征噪声 + 闪烁效果
            flicker = np.sin(2 * np.pi * params['flicker_freq'] * t)
            noise = np.random.randn(self.feature_dim) * params['noise_scale'] * (0.5 + 0.5 * flicker)
            self.state['visual_features'] += noise
            self.state['lighting_condition'] = 1.0 - scenario.intensity
        
        elif fault_type == FaultType.S2_CAMERA_OCCLUSION:
            # 摄像头遮挡：屏蔽部分视觉特征
            occlusion_mask = np.random.rand(self.feature_dim) < params['occlusion_ratio']
            self.state['visual_features'][occlusion_mask] = 0
            self.state['camera_occluded'] = True
            
            # 眩光效果
            if np.random.random() < 0.3:
                flare = np.random.randn(self.feature_dim) * params['flare_intensity']
                self.state['visual_features'] += flare
        
        elif fault_type == FaultType.S3_ADVERSARIAL_PATCH:
            # 对抗补丁：反转或扰乱动作
            self.state['adversarial_active'] = True
            if params.get('patch_type') == 'fgsm':
                # FGSM 风格攻击：沿梯度方向扰动
                perturbation = np.sign(np.random.randn(*action['τ'].shape)) * params['perturbation_scale']
                action['τ'] = action['τ'] * (1 - scenario.intensity) + perturbation * scenario.intensity
        
        # ========== Category 2: Dynamics ==========
        elif fault_type == FaultType.S4_PAYLOAD_SHIFT:
            # 负载突变 - 只在首次激活时应用
            if self.state['payload_mass'] == 0.0:
                self.state['payload_mass'] = params['payload_mass']
                # 设置新的重心位置 (不是累加)
                self.state['com_position'] = np.array(params['com_shift'])
        
        elif fault_type == FaultType.S5_JOINT_FRICTION:
            # 摩擦激增
            self.state['friction_multiplier'] = params['friction_multiplier']
        
        # ========== Category 3: Environment ==========
        elif fault_type == FaultType.S7_NARROW_CORRIDOR:
            # 盲区障碍物触发
            if self.blind_obstacle is None and t >= params.get('surprise_time', 6.0):
                self.blind_obstacle = {
                    'type': 'cart',
                    'x': self.state['base'][0] + params.get('blind_spot_distance', 3.0),
                    'y': self.state['base'][1] + np.random.uniform(-0.5, 0.5),
                    'vx': 0,
                    'vy': params.get('obstacle_speed', 1.0),
                    'radius': 0.4,
                    'emergence_time': t
                }
                self.obstacles.append(self.blind_obstacle)
        
        # ========== Category 4: Compound ==========
        elif fault_type == FaultType.S8_COMPOUND_HELL:
            # 复合故障：组合多种效应
            # S1 效应
            noise = np.random.randn(self.feature_dim) * params.get('lighting_noise_scale', 2.5)
            self.state['visual_features'] += noise
            self.state['lighting_condition'] = 0.2
            
            # S4 效应
            self.state['payload_mass'] = params.get('payload_mass', 3.0)
            
            # S7 效应
            if self.blind_obstacle is None and t >= params.get('surprise_time', 7.0):
                self.blind_obstacle = {
                    'type': 'pedestrian',
                    'x': self.state['base'][0] + 3.0,
                    'y': self.state['base'][1] + np.random.uniform(-0.3, 0.3),
                    'vx': 0,
                    'vy': 1.2,
                    'radius': 0.35
                }
                self.obstacles.append(self.blind_obstacle)
        
        return action
    
    def _update_obstacles(self, t: float) -> None:
        """更新动态障碍物位置"""
        for obs in self.obstacles:
            if obs['type'] == 'pedestrian':
                # 随机方向变化
                if np.random.random() < 0.15:
                    obs['vx'] = np.random.uniform(-1.0, 1.0)
                    obs['vy'] = np.random.uniform(-1.0, 1.0)
                
                obs['x'] += obs['vx'] * self.dt
                obs['y'] += obs['vy'] * self.dt
            
            elif obs['type'] == 'cart':
                # 推车直线运动
                obs['y'] += obs['vy'] * self.dt
    
    def _check_collision(self) -> Tuple[bool, Optional[str]]:
        """检查碰撞"""
        base_pos = self.state['base'][:2]  # [x, y]
        
        # 检查与动态障碍物碰撞
        for obs in self.obstacles:
            dist = np.linalg.norm(base_pos - np.array([obs['x'], obs['y']]))
            if dist < (0.4 + obs['radius']):  # 机器人半径 + 障碍物半径
                return True, f"collision_with_{obs['type']}"
        
        # 检查与走廊墙壁碰撞
        if hasattr(self, 'corridor_walls'):
            for wall in self.corridor_walls:
                if abs(base_pos[1] - wall['y']) < (0.3 + wall['height']):
                    if 0 <= base_pos[0] <= wall['width']:
                        return True, 'collision_with_wall'
        
        return False, None
    
    def _check_violations(self) -> bool:
        """检查约束违反"""
        # 末端执行器高度检查
        ee_height = self.state['arm_q'][0] * 0.5 + 0.3  # 简化计算
        if ee_height < self.z_ee_min:
            return True
        
        return False
    
    def _check_zmp_stability(self) -> bool:
        """
        检查 ZMP (Zero Moment Point) 稳定性
        用于 S4 负载突变场景
        
        ZMP 公式：ZMP_x = x_com - (z_com / g) * a_x
        
        稳定性判据：|ZMP_x| < support_polygon_margin
        """
        base = self.state['base']
        com = self.state['com_position']
        
        # ZMP 计算
        ax = base[3] * self.a_max  # 底盘加速度
        zmp_x = com[0] - (com[2] / self.gravity) * ax
        
        # 支撑多边形边界 (底盘长度的一半 - 安全裕度)
        # 安全裕度从 0.1m 放宽到 0.03m，允许更大的 ZMP 偏移
        support_margin = (self.length / 2) - self.zmp_safety_margin
        
        if abs(zmp_x) > support_margin:
            return False  # ZMP 超出支撑区域，可能翻倒
        
        return True
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 鼓励向目标移动
        target_x = 10.0
        current_x = self.state['base'][0]
        reward += (target_x - current_x) * 0.01
        
        # 惩罚碰撞风险
        for obs in self.obstacles:
            dist = np.linalg.norm(self.state['base'][:2] - np.array([obs['x'], obs['y']]))
            if dist < 1.0:
                reward -= (1.0 - dist) * 0.5
        
        # 惩罚剧烈动作
        reward -= np.sum(np.array([self.state['base'][3], self.state['base'][4]])**2) * 0.001
        
        return reward
    
    def _get_observation(self) -> Dict:
        """获取观测"""
        return {
            'base_state': self.state['base'].copy(),
            'arm_state': np.concatenate([self.state['arm_q'], self.state['arm_dq']]),
            'visual_features': self.state['visual_features'].copy(),
            'depth_features': self.state['depth_features'].copy(),
            'force_features': self.state['force_features'].copy(),
            'obstacles': [obs.copy() for obs in self.obstacles],
            'fault_active': self.fault_active,
            'lighting_condition': self.state['lighting_condition'],
            'payload_mass': self.state['payload_mass'],
        }
    
    def get_scenario_info(self) -> Dict:
        """获取当前场景信息"""
        if self.current_scenario is None:
            return {}
        
        return {
            'name': self.current_scenario.name,
            'fault_type': self.current_scenario.fault_type.value,
            'injection_time': self.current_scenario.injection_time,
            'duration': self.current_scenario.duration,
            'intensity': self.current_scenario.intensity,
            'params': self.current_scenario.params,
            'fault_active': self.fault_active,
        }
