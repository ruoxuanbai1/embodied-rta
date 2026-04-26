#!/usr/bin/env python3
"""
安全备份控制器 (Safe Fallback Controller)

三级干预策略:
- Region 1 触发 → 紧急刹车 (固定动作)
- Region 2 触发 → 解析 + 缩放 (TTC 动态调整)
- Region 3 触发 → 保守模式 (固定比例)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class InterventionInfo:
    """干预信息"""
    intervention_type: str       # 干预类型
    scale_v: float               # 速度缩放因子
    scale_ω: float               # 角速度缩放因子
    scale_τ: float               # 扭矩缩放因子
    min_ttc: float               # 最小 TTC (秒)
    critical_obstacle: Optional[Dict]  # 关键障碍物


class SafeFallbackController:
    """安全备份控制器"""
    
    def __init__(self, env_params: Dict = None):
        """
        初始化安全控制器
        
        参数:
            env_params: 环境参数 (可选)
        """
        # TTC 阈值 (秒)
        self.ttc_safe = 3.0       # 安全
        self.ttc_warning = 2.0    # 轻度危险
        self.ttc_danger = 1.0     # 中度危险
        # <1.0 秒 = 紧急危险
        
        # 缩放因子配置
        self.scales = {
            'none': (1.0, 1.0, 1.0),        # 无干预
            'mild_slowdown': (0.7, 0.7, 0.8),    # 轻度减速
            'moderate_slowdown': (0.4, 0.4, 0.5),  # 中度减速
            'emergency_slowdown': (0.2, 0.2, 0.3), # 紧急减速
            'conservative': (0.4, 0.4, 0.6),      # 保守模式 (Region 3)
            'emergency_stop': (-0.3, 0.0, 0.0),   # 紧急刹车 (Region 1)
        }
        
        self.env_params = env_params or {}
    
    def decide_action(self, vla_action: Dict, rta_level: int, 
                      risk_info: Dict = None) -> Tuple[Dict, InterventionInfo]:
        """
        根据 RTA 级别决定安全动作
        
        参数:
            vla_action: VLA 原始动作 {'v', 'ω', 'τ'}
            rta_level: RTA 触发级别 (0=无，1/2/3=对应层级)
            risk_info: 风险信息 (可达集、障碍物等)
        
        返回:
            safe_action: 安全动作
            info: 干预信息
        """
        if rta_level == 0:
            # 无危险：执行原动作
            info = InterventionInfo(
                intervention_type='none',
                scale_v=1.0, scale_ω=1.0, scale_τ=1.0,
                min_ttc=float('inf'),
                critical_obstacle=None
            )
            return vla_action.copy(), info
        
        elif rta_level == 1:
            # Region 1 触发：紧急危险 → 立即刹车
            safe_action = self.emergency_stop()
            info = InterventionInfo(
                intervention_type='emergency_stop',
                scale_v=-0.3, scale_ω=0.0, scale_τ=0.0,
                min_ttc=0.0,
                critical_obstacle=None
            )
            return safe_action, info
        
        elif rta_level == 2:
            # Region 2 触发：预测危险 → 解析 + 缩放 (TTC 动态调整)
            return self.ttc_based_scaling(vla_action, risk_info)
        
        elif rta_level == 3:
            # Region 3 触发：感知异常 → 保守模式
            safe_action = self.conservative_mode(vla_action)
            info = InterventionInfo(
                intervention_type='conservative',
                scale_v=0.4, scale_ω=0.4, scale_τ=0.6,
                min_ttc=float('inf'),
                critical_obstacle=None
            )
            return safe_action, info
        
        else:
            # 未知级别：执行原动作
            return vla_action.copy(), InterventionInfo(
                intervention_type='unknown',
                scale_v=1.0, scale_ω=1.0, scale_τ=1.0,
                min_ttc=float('inf'),
                critical_obstacle=None
            )
    
    def emergency_stop(self) -> Dict:
        """
        紧急刹车 (Region 1)
        
        动作:
        - 底盘：反向加速 -0.3m/s²
        - 机械臂：保持当前扭矩 (零)
        """
        return {
            'v': -0.3,           # 反向减速
            'ω': 0.0,            # 停止转向
            'τ': np.zeros(7)     # 机械臂放松
        }
    
    def ttc_based_scaling(self, vla_action: Dict, risk_info: Dict
                          ) -> Tuple[Dict, InterventionInfo]:
        """
        根据 TTC 动态调整缩放因子 (Region 2)
        
        参数:
            vla_action: VLA 原始动作
            risk_info: 风险信息
                - reachable_set: GRU 预测的可达集
                - obstacles: 障碍物列表
        
        返回:
            safe_action: 安全动作
            info: 干预信息
        """
        obstacles = risk_info.get('obstacles', [])
        reachable_set = risk_info.get('reachable_set', {})
        
        # 1. 找到最危险的障碍物 (最小 TTC)
        min_ttc = float('inf')
        critical_obs = None
        
        for obs in obstacles:
            ttc = self._estimate_ttc(vla_action, obs)
            if ttc < min_ttc:
                min_ttc = ttc
                critical_obs = obs
        
        # 2. 根据 TTC 动态调整缩放因子
        if min_ttc > self.ttc_safe:
            # 安全：执行原动作
            scale_v, scale_ω, scale_τ = self.scales['none']
            intervention_type = 'none'
        
        elif min_ttc > self.ttc_warning:
            # 轻度危险：轻微减速 (70%)
            scale_v, scale_ω, scale_τ = self.scales['mild_slowdown']
            intervention_type = 'mild_slowdown'
        
        elif min_ttc > self.ttc_danger:
            # 中度危险：明显减速 (40%)
            scale_v, scale_ω, scale_τ = self.scales['moderate_slowdown']
            intervention_type = 'moderate_slowdown'
        
        else:
            # 紧急危险：接近刹车 (20%)
            scale_v, scale_ω, scale_τ = self.scales['emergency_slowdown']
            intervention_type = 'emergency_slowdown'
        
        # 3. 检查速度是否在可达集内
        v_pred_max = reachable_set.get('base_v_max', 1.0)
        ω_pred_max = reachable_set.get('base_ω_max', 1.5)
        
        # 如果预测速度超限，进一步缩放
        if abs(vla_action['v'] * scale_v) > v_pred_max:
            scale_v *= (v_pred_max / abs(vla_action['v'] + 1e-6))
        
        if abs(vla_action['ω'] * scale_ω) > ω_pred_max:
            scale_ω *= (ω_pred_max / abs(vla_action['ω'] + 1e-6))
        
        # 4. 返回安全动作
        safe_action = {
            'v': vla_action['v'] * scale_v,
            'ω': vla_action['ω'] * scale_ω,
            'τ': vla_action['τ'] * scale_τ,
        }
        
        # 5. 干预信息
        info = InterventionInfo(
            intervention_type=intervention_type,
            scale_v=scale_v,
            scale_ω=scale_ω,
            scale_τ=scale_τ,
            min_ttc=min_ttc,
            critical_obstacle=critical_obs
        )
        
        return safe_action, info
    
    def conservative_mode(self, vla_action: Dict) -> Dict:
        """
        保守模式 (Region 3)
        
        感知异常时使用：低速 + 高安全裕度
        """
        scale_v, scale_ω, scale_τ = self.scales['conservative']
        
        return {
            'v': vla_action['v'] * scale_v,
            'ω': vla_action['ω'] * scale_ω,
            'τ': vla_action['τ'] * scale_τ
        }
    
    def _estimate_ttc(self, action: Dict, obstacle: Dict) -> float:
        """
        估计碰撞时间 (Time To Collision)
        
        简化计算：假设直线运动
        
        参数:
            action: 当前动作
            obstacle: 障碍物
        
        返回:
            ttc: 碰撞时间 (秒)，inf 表示无碰撞风险
        """
        # 机器人到障碍物的距离
        dist = self._distance_to_obstacle(action, obstacle)
        
        # 机器人速度在障碍物方向上的投影
        v_rel = self._relative_velocity(action, obstacle)
        
        if v_rel > 0:
            ttc = dist / v_rel
        else:
            ttc = float('inf')  # 远离障碍物
        
        return ttc
    
    def _distance_to_obstacle(self, action: Dict, obstacle: Dict) -> float:
        """
        计算机器人到障碍物的距离
        
        简化：假设机器人是点，障碍物是圆/盒子
        """
        # 机器人位置 (局部坐标系原点)
        robot_pos = np.array([0.0, 0.0])
        
        # 障碍物位置
        if isinstance(obstacle, dict):
            obs_pos = np.array(obstacle.get('position', [0, 0])[:2])
            obs_type = obstacle.get('type', 'cylinder')
            if obs_type == 'cylinder':
                obs_radius = obstacle.get('size', [0.3])[0]
            else:
                obs_size = obstacle.get('size', [0.5, 0.5, 1.0])
                obs_radius = max(obs_size[0], obs_size[1]) / 2
        else:
            # Obstacle 对象
            obs_pos = obstacle.position[:2]
            if obstacle.type == 'cylinder':
                obs_radius = obstacle.size[0]
            else:
                obs_radius = max(obstacle.size[0], obstacle.size[1]) / 2
        
        # 计算距离
        dist = np.linalg.norm(obs_pos - robot_pos) - obs_radius
        
        return max(0, dist)
    
    def _relative_velocity(self, action: Dict, obstacle: Dict) -> float:
        """
        计算机器人相对于障碍物的速度在障碍物方向上的投影
        """
        # 机器人速度
        v_robot = np.array([
            action.get('v', 0),
            action.get('ω', 0) * 0.3  # 简化：角速度转换为线速度
        ])
        
        # 障碍物速度
        if isinstance(obstacle, dict):
            v_obs = np.array(obstacle.get('velocity', [0, 0])[:2])
            obs_pos = np.array(obstacle.get('position', [1, 0])[:2])
        else:
            v_obs = obstacle.velocity[:2] if hasattr(obstacle, 'velocity') else np.array([0, 0])
            obs_pos = obstacle.position[:2]
        
        # 相对速度
        v_rel = v_robot - v_obs
        
        # 投影到障碍物方向
        obs_norm = np.linalg.norm(obs_pos) + 1e-6
        obs_dir = obs_pos / obs_norm
        
        return np.dot(v_rel, obs_dir)


# 测试
if __name__ == '__main__':
    controller = SafeFallbackController()
    
    # 测试 VLA 动作
    vla_action = {
        'v': 0.8,
        'ω': 0.5,
        'τ': np.ones(7) * 10
    }
    
    # 测试障碍物
    obstacle = {
        'type': 'cylinder',
        'position': [3.0, 0.5, 0],
        'velocity': [-1.0, 0, 0],
        'size': [0.4, 1.7]
    }
    
    # 测试不同 TTC 情况
    print("=== Safe Fallback Controller 测试 ===\n")
    
    # 情况 1: 安全 (TTC > 3s)
    risk_info = {'obstacles': [obstacle], 'reachable_set': {'base_v_max': 1.0, 'base_ω_max': 1.5}}
    action, info = controller.decide_action(vla_action, rta_level=2, risk_info=risk_info)
    print(f"TTC 安全: intervention={info.intervention_type}, scales=({info.scale_v:.2f}, {info.scale_ω:.2f}, {info.scale_τ:.2f})")
    
    # 情况 2: 轻度危险
    obstacle_close = {'type': 'cylinder', 'position': [2.0, 0.2, 0], 'velocity': [-1.5, 0, 0], 'size': [0.4, 1.7]}
    risk_info = {'obstacles': [obstacle_close], 'reachable_set': {'base_v_max': 1.0, 'base_ω_max': 1.5}}
    action, info = controller.decide_action(vla_action, rta_level=2, risk_info=risk_info)
    print(f"TTC 轻度：intervention={info.intervention_type}, scales=({info.scale_v:.2f}, {info.scale_ω:.2f}, {info.scale_τ:.2f})")
    
    # 情况 3: Region 1 紧急刹车
    action, info = controller.decide_action(vla_action, rta_level=1)
    print(f"Region 1:   intervention={info.intervention_type}, action=({action['v']:.2f}, {action['ω']:.2f})")
    
    # 情况 4: Region 3 保守模式
    action, info = controller.decide_action(vla_action, rta_level=3)
    print(f"Region 3:   intervention={info.intervention_type}, scales=({info.scale_v:.2f}, {info.scale_ω:.2f}, {info.scale_τ:.2f})")
    
    print("\n✅ 测试完成!")
