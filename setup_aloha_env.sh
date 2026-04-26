#!/bin/bash
# ALOHA 仿真环境配置脚本

echo "========================================"
echo "ALOHA 仿真环境配置"
echo "========================================"

# 1. 安装基础依赖
echo "[1/5] 安装基础依赖..."
pip3 install mujoco==3.1.3
pip3 install dm_control==1.0.14
pip3 install gymnasium==0.29.0
pip3 install opencv-python-headless
pip3 install h5py
pip3 install einops

# 2. 安装 ALOHA 仿真
echo "[2/5] 安装 ALOHA 仿真..."
cd /home/vipuser
git clone https://github.com/tonyzhaozh/act.git || true
cd act
pip3 install -e .

# 3. 验证安装
echo "[3/5] 验证安装..."
python3 << 'EOF'
import mujoco
import dm_control
import gymnasium
print(f'MuJoCo: {mujoco.__version__}')
print(f'dm_control: {dm_control.__version__}')
print(f'gymnasium: {gymnasium.__version__}')

# 测试 ALOHA 环境导入
try:
    from envs.aloha_env import ALOHAEnv
    print('✅ ALOHA 环境导入成功')
except Exception as e:
    print(f'⚠️ ALOHA 环境导入失败：{e}')
EOF

# 4. 创建 ALOHA 仿真环境脚本
echo "[4/5] 创建 ALOHA 仿真环境..."
mkdir -p /home/vipuser/Embodied-RTA/envs

cat > /home/vipuser/Embodied-RTA/envs/aloha_simulation_env.py << 'PYEOF'
#!/usr/bin/env python3
"""
ALOHA 仿真环境 - 基于 MuJoCo/dm_control

状态空间 (14 维):
- 左臂关节角度 (6)
- 右臂关节角度 (6)
- 左 gripper 开合 (1)
- 右 gripper 开合 (1)

动作空间 (14 维):
- 左臂关节目标 (6)
- 右臂关节目标 (6)
- 左 gripper 目标 (1)
- 右 gripper 目标 (1)
"""

import numpy as np
from dm_control import mujoco, composer
from dm_control.suite import base
import time

class ALOHASimulationEnv:
    def __init__(self, scene='empty', fault_type=None, render=False):
        """
        初始化 ALOHA 仿真环境
        
        Args:
            scene: 'empty', 'static', 'dense'
            fault_type: None 或故障类型
            fault_type: None, 'F1_lighting', 'F2_occlusion', etc.
        """
        self.scene = scene
        self.fault_type = fault_type
        self.render = render
        
        # 加载 ALOHA MJCF 模型
        self.model = self._load_aloha_model()
        self.physics = mujoco.Physics.from_mujoco_model(self.model)
        
        # 状态空间
        self.state_dim = 14
        self.action_dim = 14
        
        # 重置环境
        self.reset()
    
    def _load_aloha_model(self):
        """加载 ALOHA 机械臂 MJCF 模型"""
        # ALOHA 双臂机械臂配置
        aloha_mjcf = """
        <mujoco model="aloha">
          <worldbody>
            <light name="top" pos="0 0 2"/>
            <body name="base" pos="0 0 0">
              <!-- 左臂 -->
              <body name="left_arm" pos="-0.3 0 0">
                <joint name="left_joint_1" type="hinge" axis="0 0 1" pos="0 0 0"/>
                <joint name="left_joint_2" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <joint name="left_joint_3" type="hinge" axis="0 0 1" pos="0 0 0"/>
                <joint name="left_joint_4" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <joint name="left_joint_5" type="hinge" axis="0 0 1" pos="0 0 0"/>
                <joint name="left_joint_6" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <body name="left_gripper" pos="0 0 0.3">
                  <joint name="left_gripper_open" type="slide" axis="1 0 0" pos="0 0 0"/>
                </body>
              </body>
              
              <!-- 右臂 -->
              <body name="right_arm" pos="0.3 0 0">
                <joint name="right_joint_1" type="hinge" axis="0 0 1" pos="0 0 0"/>
                <joint name="right_joint_2" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <joint name="right_joint_3" type="hinge" axis="0 0 1" pos="0 0 0"/>
                <joint name="right_joint_4" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <joint name="right_joint_5" type="hinge" axis="0 0 1" pos="0 0 0"/>
                <joint name="right_joint_6" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <body name="right_gripper" pos="0 0 0.3">
                  <joint name="right_gripper_open" type="slide" axis="1 0 0" pos="0 0 0"/>
                </body>
              </body>
              
              <!-- 桌面 -->
              <body name="table" pos="0 0 -0.5">
                <geom type="box" size="0.5 0.5 0.05" rgba="0.5 0.3 0.1 1"/>
              </body>
              
              <!-- 目标物体 -->
              <body name="target_cube" pos="0 0.3 -0.4">
                <geom type="box" size="0.03 0.03 0.03" rgba="0.2 0.8 0.2 1"/>
                <joint type="free"/>
              </body>
            </body>
          </worldbody>
          
          <!-- 障碍物 (根据场景) -->
        </mujoco>
        """
        
        return mujoco.MjModel.from_xml_string(aloha_mjcf)
    
    def reset(self):
        """重置环境"""
        self.physics.reset()
        
        # 随机初始状态
        self.state = np.random.uniform(-0.1, 0.1, self.state_dim).astype(np.float32)
        
        # 设置故障注入点
        self.fault_inject_step = 50
        self.current_step = 0
        
        return self.state.copy()
    
    def step(self, action):
        """执行动作"""
        self.current_step += 1
        
        # 注入故障
        if self.current_step >= self.fault_inject_step and self.fault_type:
            action = self._inject_fault(action)
        
        # 简化的动力学仿真
        # 实际应该调用 physics.step()
        next_state = self.state + action * 0.02 + np.random.normal(0, 0.01, self.state_dim)
        
        # 限制状态范围
        next_state = np.clip(next_state, -1, 1)
        
        # 奖励 (接近目标)
        reward = -np.linalg.norm(next_state[:6] - 0.5)  # 简化
        
        done = self.current_step >= 250
        
        self.state = next_state.astype(np.float32)
        
        return self.state.copy(), reward, done, {}
    
    def _inject_fault(self, action):
        """注入故障"""
        if self.fault_type == 'F4_payload':
            # 负载偏移 - 动作放大
            action = action * 1.5
        elif self.fault_type == 'F5_friction':
            # 关节摩擦 - 动作衰减
            action = action * 0.6
        elif self.fault_type == 'F7_sensor':
            # 传感器噪声 - 状态加噪声 (在 step 中处理)
            pass
        
        return action
    
    def get_observation(self):
        """获取观测 (包括可能的故障)"""
        obs = self.state.copy()
        
        if self.fault_type == 'F1_lighting':
            # 光照变化不影响状态，但影响视觉
            pass
        elif self.fault_type == 'F7_sensor':
            # 传感器噪声
            if self.current_step >= self.fault_inject_step:
                obs = obs + np.random.normal(0, 0.2, self.state_dim)
        
        return obs
    
    def render(self):
        """渲染"""
        if self.render:
            self.physics.render()


# 测试
if __name__ == '__main__':
    env = ALOHASimulationEnv(scene='empty')
    state = env.reset()
    print(f'初始状态：{state}')
    
    for i in range(10):
        action = np.random.uniform(-0.1, 0.1, 14).astype(np.float32)
        next_state, reward, done, _ = env.step(action)
        print(f'Step {i}: reward={reward:.3f}, done={done}')
PYEOF

echo "[5/5] 配置完成!"
echo ""
echo "========================================"
echo "下一步:"
echo "1. 运行此脚本：bash setup_aloha_env.sh"
echo "2. 测试环境：python3 envs/aloha_simulation_env.py"
echo "3. 开始收集训练数据"
echo "========================================"
