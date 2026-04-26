#!/usr/bin/env python3
"""
Region 1: 完整物理硬包线约束检查

包含:
- 位置约束 (末端高度、防撞距离、工作空间)
- 速度约束 (底盘、关节)
- 加速度约束 (底盘、关节)
- 扭矩约束 (7 关节)
- ZMP 稳定性判据
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConstraintLimits:
    """约束限制"""
    # 位置约束
    ee_z_min: float = 0.05       # 末端最低高度 (m)
    ee_z_max: float = 2.0        # 末端最高高度 (m)
    collision_dist: float = 0.15 # 防撞距离 (m)
    workspace_x: Tuple[float, float] = (-5.0, 15.0)
    workspace_y: Tuple[float, float] = (-5.0, 5.0)
    workspace_z: Tuple[float, float] = (0.0, 2.5)
    
    # 速度约束
    base_v_max: float = 1.0      # 底盘线速度 (m/s)
    base_ω_max: float = 1.5      # 底盘角速度 (rad/s)
    joint_v_max: List[float] = None  # 7 关节速度限制 (rad/s)
    
    # 加速度约束
    base_a_max: float = 1.0      # 底盘线加速度 (m/s²)
    base_α_max: float = 2.0      # 底盘角加速度 (rad/s²)
    joint_a_max: List[float] = None  # 7 关节加速度限制 (rad/s²)
    
    # 扭矩约束
    torque_limits: List[float] = None  # 7 关节扭矩限制 (Nm)
    
    # ZMP 约束
    zmp_margin: float = 0.03     # ZMP 安全裕度 (m)
    base_length: float = 0.6     # 底盘长度 (m)
    base_width: float = 0.6      # 底盘宽度 (m)
    
    def __post_init__(self):
        if self.joint_v_max is None:
            self.joint_v_max = [2.0] * 7
        if self.joint_a_max is None:
            self.joint_a_max = [5.0] * 7
        if self.torque_limits is None:
            self.torque_limits = [50, 50, 30, 30, 20, 20, 10]


class Region1Checker:
    """Region 1 完整约束检查器"""
    
    def __init__(self, limits: ConstraintLimits = None):
        self.limits = limits or ConstraintLimits()
        self.dt = 0.02  # 50Hz 控制频率
    
    def check_all_constraints(self, state: Dict, action: Dict, 
                              obstacles: List[Dict] = None) -> List[str]:
        """
        完整约束检查
        
        参数:
            state: 当前状态字典
            action: 控制输入字典
            obstacles: 障碍物列表
        
        返回:
            violations: 违反列表，空表示无违反
        """
        violations = []
        
        # 1. 位置约束
        violations.extend(self.check_position(state, obstacles))
        
        # 2. 速度约束 (当前 + 预测)
        violations.extend(self.check_velocity(state, action))
        
        # 3. 加速度约束
        violations.extend(self.check_acceleration(action))
        
        # 4. 扭矩约束
        violations.extend(self.check_torque(action))
        
        # 5. ZMP 稳定性
        violations.extend(self.check_zmp(state, action))
        
        # 6. 工作空间边界
        violations.extend(self.check_workspace(state))
        
        return violations
    
    def check_position(self, state: Dict, obstacles: List[Dict] = None) -> List[str]:
        """位置约束检查"""
        violations = []
        
        # 末端高度检查
        ee_height = self._compute_ee_height(state)
        if ee_height < self.limits.ee_z_min:
            violations.append(f"EE_TOO_LOW: {ee_height:.3f} < {self.limits.ee_z_min}m")
        if ee_height > self.limits.ee_z_max:
            violations.append(f"EE_TOO_HIGH: {ee_height:.3f} > {self.limits.ee_z_max}m")
        
        # 防撞距离检查
        if obstacles:
            base_pos = state['base'][:2]  # [x, y]
            for obs in obstacles:
                dist = np.linalg.norm(base_pos - np.array([obs['x'], obs['y']]))
                if dist < self.limits.collision_dist:
                    violations.append(f"COLLISION_RISK: dist={dist:.3f} < {self.limits.collision_dist}m")
        
        return violations
    
    def check_velocity(self, state: Dict, action: Dict) -> List[str]:
        """速度约束检查 (当前值 + 预测值)"""
        violations = []
        
        # 底盘速度预测
        v_pred = state['base_v'] + action.get('v', 0) * self.limits.base_a_max * self.dt
        if abs(v_pred) > self.limits.base_v_max:
            violations.append(f"BASE_V_EXCEEDED: {v_pred:.3f} > {self.limits.base_v_max}m/s")
        
        ω_pred = state['base_ω'] + action.get('ω', 0) * self.limits.base_α_max * self.dt
        if abs(ω_pred) > self.limits.base_ω_max:
            violations.append(f"BASE_OMEGA_EXCEEDED: {ω_pred:.3f} > {self.limits.base_ω_max}rad/s")
        
        # 关节速度预测
        arm_dq = state.get('arm_dq', np.zeros(7))
        τ = action.get('τ', np.zeros(7))
        for i in range(7):
            dq_pred = arm_dq[i] + τ[i] / self.limits.torque_limits[i] * self.dt
            if abs(dq_pred) > self.limits.joint_v_max[i]:
                violations.append(f"JOINT_{i}_V_EXCEEDED: {dq_pred:.3f} > {self.limits.joint_v_max[i]}rad/s")
        
        return violations
    
    def check_acceleration(self, action: Dict) -> List[str]:
        """加速度约束检查"""
        violations = []
        
        # 底盘加速度
        a_cmd = action.get('v', 0) * self.limits.base_a_max
        if abs(a_cmd) > self.limits.base_a_max:
            violations.append(f"BASE_A_EXCEEDED: {a_cmd:.3f} > {self.limits.base_a_max}m/s²")
        
        α_cmd = action.get('ω', 0) * self.limits.base_α_max
        if abs(α_cmd) > self.limits.base_α_max:
            violations.append(f"BASE_ALPHA_EXCEEDED: {α_cmd:.3f} > {self.limits.base_α_max}rad/s²")
        
        return violations
    
    def check_torque(self, action: Dict) -> List[str]:
        """扭矩约束检查"""
        violations = []
        
        τ = action.get('τ', np.zeros(7))
        for i in range(7):
            if abs(τ[i]) > self.limits.torque_limits[i]:
                violations.append(f"JOINT_{i}_TORQUE_EXCEEDED: {τ[i]:.3f} > {self.limits.torque_limits[i]}Nm")
        
        return violations
    
    def check_zmp(self, state: Dict, action: Dict) -> List[str]:
        """ZMP 稳定性检查"""
        violations = []
        
        # 计算 ZMP
        com = state.get('com_position', np.array([0, 0, 0.5]))
        base_acc = action.get('v', 0) * self.limits.base_a_max
        
        zmp_x = com[0] - (com[2] / 9.81) * base_acc
        zmp_y = com[1] - (com[2] / 9.81) * (action.get('ω', 0) * self.limits.base_α_max)
        
        # 支撑边界
        x_margin = self.limits.base_length / 2 - self.limits.zmp_margin
        y_margin = self.limits.base_width / 2 - self.limits.zmp_margin
        
        if abs(zmp_x) > x_margin:
            violations.append(f"ZMP_X_UNSTABLE: {zmp_x:.3f} > {x_margin}m")
        if abs(zmp_y) > y_margin:
            violations.append(f"ZMP_Y_UNSTABLE: {zmp_y:.3f} > {y_margin}m")
        
        return violations
    
    def check_workspace(self, state: Dict) -> List[str]:
        """工作空间边界检查"""
        violations = []
        
        base_pos = state['base']
        
        if not (self.limits.workspace_x[0] <= base_pos[0] <= self.limits.workspace_x[1]):
            violations.append(f"WORKSPACE_X_EXCEEDED: x={base_pos[0]:.3f}")
        if not (self.limits.workspace_y[0] <= base_pos[1] <= self.limits.workspace_y[1]):
            violations.append(f"WORKSPACE_Y_EXCEEDED: y={base_pos[1]:.3f}")
        
        return violations
    
    def _compute_ee_height(self, state: Dict) -> float:
        """计算机器人末端高度 (简化模型)"""
        arm_q = state.get('arm_q', np.zeros(7))
        # 简化：基于第一个关节角度估算
        ee_height = arm_q[0] * 0.5 + 0.3
        return ee_height
    
    def get_support_variables(self) -> Dict[str, str]:
        """
        返回 Region 2 需要预测的支撑变量
        
        这些变量的上下界将由 GRU 预测
        """
        return {
            # 底盘位置 (2D)
            'base_x': '底盘 X 位置 (m)',
            'base_y': '底盘 Y 位置 (m)',
            
            # 底盘速度
            'base_v': '底盘线速度 (m/s)',
            'base_ω': '底盘角速度 (rad/s)',
            
            # 末端位置 (3D)
            'ee_x': '末端 X 位置 (m)',
            'ee_y': '末端 Y 位置 (m)',
            'ee_z': '末端 Z 位置 (m)',
            
            # 关节速度 (7 个)
            'arm_dq_0': '关节 1 速度 (rad/s)',
            'arm_dq_1': '关节 2 速度 (rad/s)',
            'arm_dq_2': '关节 3 速度 (rad/s)',
            'arm_dq_3': '关节 4 速度 (rad/s)',
            'arm_dq_4': '关节 5 速度 (rad/s)',
            'arm_dq_5': '关节 6 速度 (rad/s)',
            'arm_dq_6': '关节 7 速度 (rad/s)',
            
            # ZMP 位置
            'zmp_x': 'ZMP X 位置 (m)',
            'zmp_y': 'ZMP Y 位置 (m)',
        }
    
    def get_support_dim(self) -> int:
        """返回支撑变量维度 (每个变量需要预测 min+max)"""
        return len(self.get_support_variables()) * 2  # 15 × 2 = 30 维


# 测试
if __name__ == '__main__':
    checker = Region1Checker()
    
    # 测试状态
    state = {
        'base': np.array([0.0, 0.0, 0.0, 0.5, 0.3]),  # [x, y, θ, v, ω]
        'base_v': 0.5,
        'base_ω': 0.3,
        'arm_q': np.zeros(7),
        'arm_dq': np.zeros(7),
        'com_position': np.array([0, 0, 0.5]),
    }
    
    # 测试动作
    action = {
        'v': 0.8,  # 大加速度
        'ω': 1.0,
        'τ': np.ones(7) * 40,  # 大扭矩
    }
    
    # 测试障碍物
    obstacles = [{'x': 0.1, 'y': 0.1, 'radius': 0.2}]
    
    # 运行检查
    violations = checker.check_all_constraints(state, action, obstacles)
    
    print("Region 1 约束检查结果:")
    if violations:
        for v in violations:
            print(f"  ❌ {v}")
    else:
        print("  ✅ 无违反")
    
    print(f"\n支撑变量维度：{checker.get_support_dim()}")
    print("支撑变量:")
    for var, desc in checker.get_support_variables().items():
        print(f"  - {var}: {desc}")
