#!/usr/bin/env python3
"""
简化仿真环境

用于快速测试和数据采集，不依赖 Isaac Lab
模拟 Fetch 机器人导航任务
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class SimpleNavigationEnv:
    """
    简化导航环境
    
    状态空间 (14 维):
    - 位置：x, y, z (3)
    - 速度：vx, vy, vz (3)
    - 姿态：roll, pitch, yaw (3)
    - 角速度：wx, wy, wz (3)
    - 本体：battery, load (2)
    
    动作空间 (14 维):
    - 线速度：vx, vy, vz (3)
    - 角速度：wx, wy, wz (3)
    - 关节：joint1-8 (8)
    """
    
    def __init__(
        self,
        scenario: str = 'B1',
        fault_type: Optional[str] = None,
        device: str = 'cuda'
    ):
        self.device = device
        self.scenario = scenario
        self.fault_type = fault_type
        
        # 场景配置
        self.config = self._get_scenario_config(scenario)
        
        # 状态
        self.state = None
        self.goal = None
        self.obstacles = []
        self.step_count = 0
        self.max_steps = 500
        
        # 故障状态
        self.fault_active = False
        self.fault_start_step = 0
        self.fault_duration = 0
        
        # 视觉特征 (简化)
        self.visual_feature_dim = 512
    
    def _get_scenario_config(self, scenario: str) -> Dict:
        """获取场景配置"""
        configs = {
            'B1': {'obstacles': 0, 'narrow': False, 'goal_dist': 10.0},
            'B2': {'obstacles': 5, 'narrow': False, 'goal_dist': 10.0},
            'B3': {'obstacles': 10, 'narrow': False, 'goal_dist': 10.0},
            'B4': {'obstacles': 2, 'narrow': True, 'gap': 0.8, 'goal_dist': 10.0},
        }
        return configs.get(scenario, configs['B1'])
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.step_count = 0
        
        # 初始状态 (原点)
        self.state = np.zeros(14)
        self.state[2] = 1.0  # z = 1m (机器人高度)
        
        # 目标位置
        goal_dist = self.config['goal_dist']
        self.goal = np.array([goal_dist, 0.0, 1.0])
        
        # 生成障碍物
        self.obstacles = self._generate_obstacles()
        
        # 重置故障
        self.fault_active = False
        
        return self.state.copy()
    
    def _generate_obstacles(self) -> list:
        """生成障碍物"""
        obstacles = []
        n_obstacles = self.config['obstacles']
        
        for i in range(n_obstacles):
            # 随机位置
            x = np.random.uniform(2, 8)
            y = np.random.uniform(-3, 3)
            z = 0.5
            
            # 随机尺寸
            size = np.random.uniform(0.3, 0.8)
            
            obstacles.append({
                'pos': np.array([x, y, z]),
                'size': size,
                'type': 'static'
            })
        
        # 窄通道场景
        if self.config.get('narrow'):
            gap = self.config.get('gap', 0.8)
            obstacles.append({
                'pos': np.array([5.0, -gap/2, 0.5]),
                'size': 2.0,
                'type': 'wall'
            })
            obstacles.append({
                'pos': np.array([5.0, gap/2, 0.5]),
                'size': 2.0,
                'type': 'wall'
            })
        
        return obstacles
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        环境步进
        
        Args:
            action: (14,) 动作
        
        Returns:
            state: 新状态
            reward: 奖励
            done: 是否结束
            info: 信息
        """
        self.step_count += 1
        
        # 应用动作 (简单积分)
        dt = 0.02  # 50Hz
        
        # 更新速度
        self.state[3:6] = action[0:3] * 0.5  # 线速度
        self.state[10:13] = action[3:6] * 0.5  # 角速度
        
        # 更新位置
        self.state[0] += self.state[3] * dt
        self.state[1] += self.state[4] * dt
        self.state[2] += self.state[5] * dt
        
        # 更新姿态
        self.state[6:9] += self.state[10:13] * dt
        
        # 故障影响
        if self.fault_active:
            self._apply_fault()
        
        # 检查碰撞
        collision = self._check_collision()
        
        # 检查目标到达
        dist_to_goal = np.linalg.norm(self.state[0:3] - self.goal)
        goal_reached = dist_to_goal < 0.5
        
        # 奖励
        reward = -dist_to_goal * 0.1
        if goal_reached:
            reward += 10.0
        if collision:
            reward -= 5.0
        
        # 结束条件
        done = goal_reached or collision or (self.step_count >= self.max_steps)
        
        info = {
            'goal_reached': goal_reached,
            'collision': collision,
            'dist_to_goal': dist_to_goal,
            'step': self.step_count,
        }
        
        return self.state.copy(), reward, done, info
    
    def _check_collision(self) -> bool:
        """检查碰撞"""
        robot_radius = 0.3
        
        # 检查障碍物
        for obs in self.obstacles:
            dist = np.linalg.norm(self.state[0:3] - obs['pos'])
            if dist < robot_radius + obs['size']:
                return True
        
        # 检查边界
        if abs(self.state[0]) > 15 or abs(self.state[1]) > 5:
            return True
        
        return False
    
    def _apply_fault(self):
        """应用故障影响"""
        if self.fault_type is None:
            return
        
        # 不同故障类型的影响
        if '光照' in self.fault_type or '视觉' in self.fault_type:
            # 视觉故障：添加状态噪声
            self.state += np.random.randn(14) * 0.1
        
        elif '噪声' in self.fault_type:
            # 传感器噪声
            self.state += np.random.randn(14) * 0.2
        
        elif '执行器' in self.fault_type or '摩擦' in self.fault_type:
            # 执行器退化：动作衰减
            pass  # 在 step 中处理
        
        elif '负载' in self.fault_type:
            # 负载突变：动力学变化
            self.state[3:6] *= 0.7
    
    def inject_fault(self, fault_type: str, start_step: int, duration: int):
        """注入故障"""
        if self.step_count >= start_step and not self.fault_active:
            self.fault_type = fault_type
            self.fault_active = True
            self.fault_start_step = self.step_count
            self.fault_duration = duration
    
    def clear_fault(self):
        """清除故障"""
        if self.fault_active and (self.step_count - self.fault_start_step) >= self.fault_duration:
            self.fault_active = False
            self.fault_type = None
    
    def get_visual_features(self) -> torch.Tensor:
        """获取视觉特征 (简化) - 返回 ResNet18 特征维度"""
        # 生成 512 维特征 (模拟 ResNet18 输出)
        feature = np.random.randn(self.visual_feature_dim).astype(np.float32) * 0.1
        
        # 用状态调制特征
        feature[:14] += self.state * 0.5
        
        return torch.FloatTensor(feature).to(self.device)
    
    def get_observation_dict(self) -> Dict:
        """获取观测字典 (ACT 格式)"""
        # ACT 需要 BCHW 格式的图像
        # 简化：使用随机图像 + 状态
        dummy_image = torch.randn(3, 256, 256, device=self.device) * 0.1
        
        return {
            'observation.state': torch.FloatTensor(self.state).unsqueeze(0).to(self.device),
            'observation.images.top': dummy_image.unsqueeze(0),
        }


if __name__ == '__main__':
    # 测试环境
    print('=== 测试简化导航环境 ===')
    
    env = SimpleNavigationEnv(scenario='B2')
    state = env.reset()
    
    print(f'初始状态：{state}')
    print(f'目标位置：{env.goal}')
    print(f'障碍物数量：{len(env.obstacles)}')
    
    # 运行几步
    for i in range(10):
        action = np.random.randn(14) * 0.5
        state, reward, done, info = env.step(action)
        print(f'Step {i}: pos={state[0:3]}, reward={reward:.2f}, done={done}')
        
        if done:
            break
    
    print('\\n✅ 环境测试通过!')
