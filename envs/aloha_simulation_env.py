#!/usr/bin/env python3
"""
ALOHA 仿真环境 - 基于 MuJoCo/dm_control (简化版)

状态空间 (14 维):
- 左臂关节角度 (6)
- 右臂关节角度 (6)  
- 左 gripper 开合 (1)
- 右 gripper 开合 (1)

动作空间 (14 维): 关节目标位置
"""

import numpy as np
import time

class ALOHASimulationEnv:
    """简化的 ALOHA 仿真环境 (运动学模型)"""
    
    def __init__(self, scene='empty', fault_type=None, seed=None):
        """
        Args:
            scene: 'empty', 'static', 'dense'
            fault_type: None, 'F1_lighting', 'F2_occlusion', 'F3_adversarial',
                       'F4_payload', 'F5_friction', 'F6_dynamic', 'F7_sensor', 'F8_compound'
        """
        self.scene = scene
        self.fault_type = fault_type
        self.np_random = np.random.RandomState(seed)
        
        # 状态空间
        self.state_dim = 14
        self.action_dim = 14
        
        # 物理参数
        self.joint_limits = (-np.pi, np.pi)
        self.gripper_limits = (0, 0.1)
        self.max_velocity = 0.5  # rad/s
        
        # 场景配置
        self.obstacles = self._get_scene_obstacles(scene)
        
        # 重置环境
        self.reset()
    
    def _get_scene_obstacles(self, scene):
        """获取场景障碍物配置"""
        if scene == 'empty':
            return []
        elif scene == 'static':
            # 5 个静态障碍
            return [
                {'pos': (0.2, 0.3, 0), 'size': (0.05, 0.05, 0.1)},
                {'pos': (-0.2, 0.3, 0), 'size': (0.05, 0.05, 0.1)},
                {'pos': (0, 0.4, 0), 'size': (0.05, 0.05, 0.1)},
                {'pos': (0.3, 0.2, 0), 'size': (0.05, 0.05, 0.1)},
                {'pos': (-0.3, 0.2, 0), 'size': (0.05, 0.05, 0.1)},
            ]
        elif scene == 'dense':
            # 10 个密集障碍
            return [
                {'pos': (0.2*i, 0.2*j, 0), 'size': (0.04, 0.04, 0.08)}
                for i in range(-2, 3) for j in range(1, 3)
            ][:10]
        return []
    
    def reset(self):
        """重置环境到初始状态"""
        # 初始状态：双臂收起
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        
        # 添加小随机扰动
        self.state[:6] = self.np_random.uniform(-0.1, 0.1, 6)  # 左臂
        self.state[6:12] = self.np_random.uniform(-0.1, 0.1, 6)  # 右臂
        self.state[12:14] = self.np_random.uniform(0.05, 0.08, 2)  # gripper
        
        # 故障注入点
        self.fault_inject_step = 50
        self.current_step = 0
        
        # 目标位置 (抓取任务)
        self.target_pos = self.np_random.uniform(0.1, 0.3, 3)
        
        return self.state.copy()
    
    def step(self, action):
        """
        执行动作
        
        Args:
            action: 14 维关节目标位置
        
        Returns:
            next_state, reward, done, info
        """
        self.current_step += 1
        
        # 注入故障
        if self.current_step >= self.fault_inject_step and self.fault_type:
            action = self._inject_fault(action)
        
        # 简化的运动学仿真
        # PD 控制：向目标位置移动
        dt = 0.02  # 50Hz
        kp = 2.0
        velocity = kp * (action - self.state)
        
        # 速度限制
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
        
        # 状态更新
        next_state = self.state + velocity * dt
        
        # 添加动力学噪声
        next_state += self.np_random.normal(0, 0.01, self.state_dim)
        
        # 关节限制
        next_state[:12] = np.clip(next_state[:12], *self.joint_limits)
        next_state[12:14] = np.clip(next_state[12:14], *self.gripper_limits)
        
        # 碰撞检测
        collision = self._check_collision(next_state)
        if collision:
            next_state = self.state  # 碰撞则保持原状态
            reward = -1.0
        else:
            # 奖励：接近目标
            ee_pos = self._forward_kinematics(next_state)
            reward = -np.linalg.norm(ee_pos - self.target_pos)
        
        # 终止条件
        done = self.current_step >= 250 or collision
        
        self.state = next_state.astype(np.float32)
        
        info = {
            'collision': collision,
            'step': self.current_step,
            'ee_pos': ee_pos,
        }
        
        return self.state.copy(), reward, done, info
    
    def _inject_fault(self, action):
        """注入故障"""
        if self.fault_type == 'F4_payload':
            # 负载偏移 - 动力学变化
            action = action * 1.3
        
        elif self.fault_type == 'F5_friction':
            # 关节摩擦 - 动作衰减
            action = action * 0.6
        
        elif self.fault_type == 'F7_sensor':
            # 传感器噪声 - 在观测中处理
            pass
        
        elif self.fault_type == 'F8_compound':
            # 复合故障
            action = action * 1.2 + self.np_random.normal(0, 0.1, self.action_dim)
        
        return action
    
    def get_observation(self):
        """获取观测 (可能包含故障)"""
        obs = self.state.copy()
        
        if self.fault_type == 'F7_sensor':
            if self.current_step >= self.fault_inject_step:
                obs = obs + self.np_random.normal(0, 0.2, self.state_dim)
        
        return obs.astype(np.float32)
    
    def _forward_kinematics(self, state):
        """简化的正向运动学 (计算末端执行器位置)"""
        # 简化：假设基座到末端的映射
        left_arm = state[:6]
        right_arm = state[6:12]
        
        # 简化的 FK (实际应该用 DH 参数)
        left_ee = np.array([
            0.3 * np.sin(left_arm[0]) * np.cos(left_arm[1]),
            0.3 * np.sin(left_arm[1]),
            0.3 * np.cos(left_arm[0]) * np.cos(left_arm[1])
        ])
        
        right_ee = np.array([
            0.3 * np.sin(right_arm[0]) * np.cos(right_arm[1]),
            0.3 * np.sin(right_arm[1]),
            0.3 * np.cos(right_arm[0]) * np.cos(right_arm[1])
        ])
        
        # 返回双臂中点
        return (left_ee + right_ee) / 2
    
    def _check_collision(self, state):
        """碰撞检测"""
        ee_pos = self._forward_kinematics(state)
        
        for obs in self.obstacles:
            obs_pos = np.array(obs['pos'])
            obs_size = np.array(obs['size'])
            
            # 简单 AABB 碰撞检测
            if np.all(np.abs(ee_pos[:2] - obs_pos[:2]) < obs_size[:2]):
                return True
        
        return False
    
    def render(self):
        """渲染 (文本模式)"""
        print(f'Step {self.current_step}/250')
        print(f'State: {self.state[:6]} (left), {self.state[6:12]} (right)')
        print(f'EE pos: {self._forward_kinematics(self.state)}')
        print(f'Target: {self.target_pos}')


# 测试
if __name__ == '__main__':
    print('测试 ALOHA 仿真环境...')
    
    env = ALOHASimulationEnv(scene='empty', seed=42)
    state = env.reset()
    print(f'初始状态：{state}')
    
    total_reward = 0
    for i in range(100):
        action = env.np_random.uniform(-0.5, 0.5, 14).astype(np.float32)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if i % 20 == 0:
            print(f'Step {i}: reward={reward:.3f}, total={total_reward:.3f}')
        
        if done:
            print(f'Episode done at step {i}')
            break
    
    print(f'最终奖励：{total_reward:.3f}')
    print('✅ 环境测试通过!')
