"""
Fetch Mobile Manipulator Environment
等效 Isaac Lab 的简化版本，保持接口一致
"""

import numpy as np
import yaml
from typing import Dict, Tuple, Optional

class FetchMobileEnv:
    """
    Fetch 移动机械臂环境
    
    状态空间:
    - 底盘：[x, y, θ, v, ω] (5 维)
    - 机械臂：[q1-q7, dq1-dq7] (14 维)
    - 视觉特征：[512 维] (从 ResNet 提取)
    
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
        
        # 机械臂参数
        self.τ_limits = np.array(cfg['arm']['τ_limits'])
        
        # Region 1 约束
        self.d_min = cfg['constraints']['d_min']
        self.z_ee_min = cfg['constraints']['z_ee_min']
        
        # 仿真参数
        self.dt = cfg['simulation']['dt']
        self.max_steps = 1500  # 30 秒 @ 50Hz
        
        # 障碍物 (动态行人)
        self.obstacles = []
        
        # 状态
        self.state = None
        self.step_count = 0
    
    def reset(self, seed: Optional[int] = None) -> Dict:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 初始状态
        self.state = {
            'base': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # [x, y, θ, v, ω]
            'arm_q': np.zeros(7),  # 关节角度
            'arm_dq': np.zeros(7),  # 关节速度
            'visual_features': np.random.randn(512) * 0.1  # 模拟视觉特征
        }
        
        self.step_count = 0
        self.obstacles = []
        
        return self._get_observation()
    
    def step(self, action: Dict, fault_info: Optional[Dict] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        环境步进
        
        action: {
            'v': 线速度 (-1 到 1)
            'ω': 角速度 (-1.5 到 1.5)
            'τ': 7 关节扭矩
        }
        """
        # 应用故障注入
        if fault_info and fault_info.get('active', False):
            action = self._inject_fault(action, fault_info)
        
        # 底盘动力学 (Unicycle 模型)
        base = self.state['base'].copy()
        base[0] += base[3] * np.cos(base[2]) * self.dt  # x
        base[1] += base[3] * np.sin(base[2]) * self.dt  # y
        base[2] += base[4] * self.dt                     # θ
        base[3] += action['v'] * self.a_max * self.dt   # v
        base[4] += action['ω'] * self.α_max * self.dt   # ω
        
        # 速度限制
        base[3] = np.clip(base[3], -self.v_max, self.v_max)
        base[4] = np.clip(base[4], -self.ω_max, self.ω_max)
        
        # 机械臂简化动力学
        arm_q = self.state['arm_q'] + action['τ'] / self.τ_limits * 0.1
        arm_q = np.clip(arm_q, -np.pi, np.pi)
        
        # 更新状态
        self.state['base'] = base
        self.state['arm_q'] = arm_q
        
        # 更新动态障碍物 (行人)
        self._update_obstacles()
        
        # 检查碰撞
        collision = self._check_collision()
        
        # 检查约束违反
        violations = self._check_violations()
        
        self.step_count += 1
        
        # 终止条件
        done = collision or violations or (self.step_count >= self.max_steps)
        
        # 奖励 (鼓励向目标移动，惩罚碰撞风险)
        reward = self._compute_reward()
        
        info = {
            'collision': collision,
            'violations': violations,
            'obstacles': len(self.obstacles)
        }
        
        return self._get_observation(), reward, done, info
    
    def _inject_fault(self, action: Dict, fault_info: Dict) -> Dict:
        """故障注入"""
        fault_type = fault_info.get('type', None)
        
        if fault_type == 'lighting_ood':
            # 光照致盲：添加视觉特征噪声
            self.state['visual_features'] += np.random.randn(512) * 2.0
        
        elif fault_type == 'adversarial_patch':
            # 对抗补丁：欺骗动作输出
            action['τ'] *= -1  # 反转机械臂动作
        
        elif fault_type == 'dynamic_human':
            # 动态行人：添加突发障碍物
            if np.random.random() < 0.1:
                self.obstacles.append({
                    'x': self.state['base'][0] + np.random.uniform(0.5, 2.0),
                    'y': self.state['base'][1] + np.random.uniform(-1.0, 1.0),
                    'vx': np.random.uniform(-0.5, 0.5),
                    'vy': np.random.uniform(-0.5, 0.5),
                    'radius': 0.3
                })
        
        return action
    
    def _update_obstacles(self):
        """更新动态障碍物位置"""
        for obs in self.obstacles:
            obs['x'] += obs['vx'] * self.dt
            obs['y'] += obs['vy'] * self.dt
    
    def _check_collision(self) -> bool:
        """检查碰撞"""
        base_x, base_y = self.state['base'][0], self.state['base'][1]
        
        for obs in self.obstacles:
            dist = np.sqrt((base_x - obs['x'])**2 + (base_y - obs['y'])**2)
            if dist < self.d_min + obs['radius']:
                return True
        
        return False
    
    def _check_violations(self) -> bool:
        """检查约束违反"""
        base = self.state['base']
        
        # 速度超限
        if abs(base[3]) > self.v_max * 1.1:
            return True
        if abs(base[4]) > self.ω_max * 1.1:
            return True
        
        return False
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        # 鼓励向目标移动
        target_x, target_y = 10.0, 0.0
        dist_to_target = np.sqrt((self.state['base'][0] - target_x)**2 + 
                                  (self.state['base'][1] - target_y)**2)
        
        reward = -0.01 * dist_to_target
        
        # 惩罚碰撞风险
        for obs in self.obstacles:
            dist = np.sqrt((self.state['base'][0] - obs['x'])**2 + 
                          (self.state['base'][1] - obs['y'])**2)
            if dist < 1.0:
                reward -= 1.0 / (dist + 0.1)
        
        return reward
    
    def _get_observation(self) -> Dict:
        """获取观测"""
        return {
            'base_state': self.state['base'].copy(),
            'arm_state': np.concatenate([self.state['arm_q'], self.state['arm_dq']]),
            'visual_features': self.state['visual_features'].copy(),
            'obstacles': self.obstacles.copy()
        }
    
    def add_obstacle(self, x: float, y: float, vx: float = 0, vy: float = 0):
        """添加障碍物"""
        self.obstacles.append({
            'x': x, 'y': y, 'vx': vx, 'vy': vy, 'radius': 0.3
        })


if __name__ == '__main__':
    # 测试环境
    env = FetchMobileEnv('configs/fetch_params.yaml')
    obs = env.reset(seed=42)
    
    print("初始观测:")
    print(f"  底盘状态：{obs['base_state']}")
    print(f"  机械臂状态：{obs['arm_state'].shape}")
    print(f"  视觉特征：{obs['visual_features'].shape}")
    
    # 运行几步
    for i in range(10):
        action = {'v': 0.5, 'ω': 0.1, 'τ': np.zeros(7)}
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={done}")
    
    print("\n环境测试通过!")
