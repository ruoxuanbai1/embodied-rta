#!/usr/bin/env python3
"""ALOHA 仿真环境 - 简化运动学模型"""
import numpy as np

class ALOHASimulationEnv:
    def __init__(self, scene='empty', fault_type=None, seed=None):
        self.scene = scene
        self.fault_type = fault_type
        self.np_random = np.random.RandomState(seed)
        self.state_dim = 14
        self.action_dim = 14
        self.joint_limits = (-np.pi, np.pi)
        self.gripper_limits = (0, 0.1)
        self.max_velocity = 0.5
        self.obstacles = self._get_scene_obstacles(scene)
        self.reset()
    
    def _get_scene_obstacles(self, scene):
        if scene == 'empty': return []
        elif scene == 'static':
            return [{'pos': (0.2*i, 0.2*j, 0), 'size': (0.05, 0.05, 0.1)} for i,j in [(-1,1),(1,1),(0,2),(-2,1),(2,1)]]
        elif scene == 'dense':
            return [{'pos': (0.15*i, 0.15*j, 0), 'size': (0.04, 0.04, 0.08)} for i in range(-3,4) for j in range(1,3)][:10]
        return []
    
    def reset(self):
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.state[:6] = self.np_random.uniform(-0.1, 0.1, 6)
        self.state[6:12] = self.np_random.uniform(-0.1, 0.1, 6)
        self.state[12:14] = self.np_random.uniform(0.05, 0.08, 2)
        self.fault_inject_step = 50
        self.current_step = 0
        self.target_pos = self.np_random.uniform(0.1, 0.3, 3)
        return self.state.copy()
    
    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.fault_inject_step and self.fault_type:
            action = self._inject_fault(action)
        dt = 0.02
        kp = 2.0
        velocity = kp * (action - self.state)
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
        next_state = self.state + velocity * dt + self.np_random.normal(0, 0.01, self.state_dim)
        next_state[:12] = np.clip(next_state[:12], *self.joint_limits)
        next_state[12:14] = np.clip(next_state[12:14], *self.gripper_limits)
        collision = self._check_collision(next_state)
        if collision:
            next_state = self.state
            reward = -1.0
        else:
            ee_pos = self._forward_kinematics(next_state)
            reward = -np.linalg.norm(ee_pos - self.target_pos)
        done = self.current_step >= 250 or collision
        self.state = next_state.astype(np.float32)
        return self.state.copy(), reward, done, {'collision': collision, 'step': self.current_step}
    
    def _inject_fault(self, action):
        if self.fault_type == 'F4_payload': action = action * 1.3
        elif self.fault_type == 'F5_friction': action = action * 0.6
        elif self.fault_type == 'F8_compound': action = action * 1.2 + self.np_random.normal(0, 0.1, self.action_dim)
        return action
    
    def get_observation(self):
        obs = self.state.copy()
        if self.fault_type == 'F7_sensor' and self.current_step >= self.fault_inject_step:
            obs = obs + self.np_random.normal(0, 0.2, self.state_dim)
        return obs.astype(np.float32)
    
    def _forward_kinematics(self, state):
        left_ee = np.array([0.3*np.sin(state[0])*np.cos(state[1]), 0.3*np.sin(state[1]), 0.3*np.cos(state[0])*np.cos(state[1])])
        right_ee = np.array([0.3*np.sin(state[6])*np.cos(state[7]), 0.3*np.sin(state[7]), 0.3*np.cos(state[6])*np.cos(state[7])])
        return (left_ee + right_ee) / 2
    
    def _check_collision(self, state):
        ee_pos = self._forward_kinematics(state)
        for obs in self.obstacles:
            if np.all(np.abs(ee_pos[:2] - np.array(obs['pos'])[:2]) < np.array(obs['size'])[:2]):
                return True
        return False

if __name__ == '__main__':
    print('测试 ALOHA 环境...')
    env = ALOHASimulationEnv(scene='empty', seed=42)
    state = env.reset()
    print('初始状态:', state[:6], '(left arm)')
    total_r = 0
    for i in range(50):
        action = env.np_random.uniform(-0.5, 0.5, 14).astype(np.float32)
        s, r, d, _ = env.step(action)
        total_r += r
        if d: break
    print(f'50 步总奖励：{total_r:.3f} ✅')
