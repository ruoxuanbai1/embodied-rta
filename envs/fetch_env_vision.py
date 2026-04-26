#!/usr/bin/env python3
"""
Fetch 移动机械臂环境 (支持视觉输入)

改进:
1. 集成 PyBullet 物理引擎，提供真实 RGB 图像
2. 支持 OpenVLA 图像输入接口
3. 故障注入 (光照突变/遮挡/对抗补丁)
"""

import numpy as np
import yaml
from typing import Dict, Tuple, Optional, List
from pathlib import Path

# 尝试导入 PyBullet (可选)
try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False
    print("警告：PyBullet 未安装，使用简化视觉模型")

# 尝试导入 OpenCV (用于图像处理)
try:
    import cv2
    HAS_CV = True
except ImportError:
    HAS_CV = False


class FetchMobileEnvWithVision:
    """
    Fetch 移动机械臂环境 (带视觉)
    
    状态空间:
    - 底盘：[x, y, θ, v, ω] (5 维)
    - 机械臂：[q1-q7, dq1-dq7] (14 维)
    - 视觉：RGB 图像 (224×224×3)
    
    动作空间:
    - 底盘：[v, ω] (线速度，角速度)
    - 机械臂：[τ1-τ7] (7 关节扭矩)
    """
    
    def __init__(self, config_path='configs/fetch_params.yaml', 
                 render=False, use_pybullet=True):
        """
        参数:
            render: 是否显示 GUI
            use_pybullet: 是否使用 PyBullet (否则用简化模型)
        """
        # 加载配置
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            # 默认配置
            cfg = {
                'base': {'v_max': 1.0, 'ω_max': 1.5, 'a_max': 1.0, 'α_max': 2.0},
                'arm': {'τ_limits': [50, 50, 30, 30, 20, 20, 10]},
                'constraints': {'d_min': 0.15, 'z_ee_min': 0.05},
                'simulation': {'dt': 0.02}
            }
        
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
        
        # 视觉配置
        self.image_size = 224
        self.use_pybullet = use_pybullet and HAS_PYBULLET
        self.render = render
        self.physics_client = None
        self.robot_id = None
        
        # 障碍物
        self.obstacles = []
        self.obstacle_ids = []
        
        # 故障状态
        self.fault_active = False
        self.fault_type = None
        self.fault_params = {}
        
        # 状态
        self.state = None
        self.step_count = 0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        # 初始化 PyBullet
        if self.use_pybullet:
            self._init_pybullet()
    
    def _init_pybullet(self):
        """初始化 PyBullet 物理引擎"""
        if self.render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 加载地面
        p.loadURDF("plane.urdf")
        
        # 加载 Fetch 机器人 (需要 fetch_description 包)
        # 如果找不到，使用简化模型
        try:
            self.robot_id = p.loadURDF(
                "fetch_description/fetch.urdf",
                [0, 0, 0],
                useFixedBase=False
            )
        except:
            # 简化：用立方体代替
            self.robot_id = p.loadURDF("cube_small.urdf", [0, 0, 0.5])
            print("警告：使用简化机器人模型")
        
        print(f"✅ PyBullet 初始化完成 (robot_id={self.robot_id})")
    
    def reset(self, seed: Optional[int] = None, 
              scene_type: str = 'empty') -> Dict:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 重置 PyBullet 状态
        if self.use_pybullet:
            p.resetSimulation()
            p.loadURDF("plane.urdf")
            
            # 重新加载机器人
            try:
                self.robot_id = p.loadURDF(
                    "fetch_description/fetch.urdf",
                    [0, 0, 0],
                    useFixedBase=False
                )
            except:
                self.robot_id = p.loadURDF("cube_small.urdf", [0, 0, 0.5])
        
        # 重置状态
        self.state = {
            'base': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # [x, y, θ, v, ω]
            'arm_q': np.zeros(7),
            'arm_dq': np.zeros(7),
        }
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.step_count = 0
        
        # 加载场景
        self._load_scene(scene_type)
        
        # 重置故障
        self.fault_active = False
        
        return self._get_observation()
    
    def _load_scene(self, scene_type: str):
        """加载场景 (empty/sparse/dense)"""
        # 清除旧障碍物
        for obs_id in self.obstacle_ids:
            if p.isValid(obs_id):
                p.removeBody(obs_id)
        self.obstacle_ids = []
        self.obstacles = []
        
        if scene_type == 'empty':
            # 无障碍物
            pass
        
        elif scene_type == 'sparse':
            # 5 个随机障碍物
            for i in range(5):
                x = np.random.uniform(3, 8)
                y = np.random.uniform(-2, 2)
                self._add_obstacle(x, y)
        
        elif scene_type == 'dense':
            # 10 个随机障碍物
            for i in range(10):
                x = np.random.uniform(2, 9)
                y = np.random.uniform(-2.5, 2.5)
                self._add_obstacle(x, y)
        
        elif scene_type == 'narrow':
            # 窄通道 (两面墙)
            self._add_wall(x=5, y_min=-0.4, y_max=-0.3)  # 左墙
            self._add_wall(x=5, y_min=0.3, y_max=0.4)    # 右墙
    
    def _add_obstacle(self, x: float, y: float, size: float = 0.4):
        """添加障碍物"""
        if self.use_pybullet:
            obs_id = p.loadURDF("cube_small.urdf", [x, y, 0.5])
            self.obstacle_ids.append(obs_id)
        
        self.obstacles.append({'x': x, 'y': y, 'z': 0.5, 'size': size})
    
    def _add_wall(self, x: float, y_min: float, y_max: float):
        """添加墙"""
        y = (y_min + y_max) / 2
        self._add_obstacle(x, y, size=0.1)
    
    def get_camera_image(self) -> np.ndarray:
        """
        获取相机 RGB 图像 (224×224×3)
        
        相机安装在机器人头部，朝前
        """
        if self.use_pybullet and p.isConnected():
            # PyBullet 渲染
            return self._get_pybullet_image()
        else:
            # 简化：生成合成图像
            return self._get_synthetic_image()
    
    def _get_pybullet_image(self) -> np.ndarray:
        """从 PyBullet 获取渲染图像"""
        # 相机参数
        camera_distance = 0.5
        camera_yaw = self.robot_yaw * 180 / np.pi
        camera_pitch = -20
        camera_roll = 0
        
        # 计算相机目标位置
        target_x = self.robot_x + 0.5 * np.cos(self.robot_yaw)
        target_y = self.robot_y + 0.5 * np.sin(self.robot_yaw)
        target_z = 1.0
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[target_x, target_y, target_z],
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=camera_roll,
            upAxisIndex=2
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )
        
        # 渲染
        img = p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # 提取 RGB (去掉 alpha 通道)
        rgb = np.array(img[2][:, :, :3], dtype=np.uint8)
        
        return rgb
    
    def _get_synthetic_image(self) -> np.ndarray:
        """
        生成合成图像 (无 PyBullet 时使用)
        
        包含:
        - 地面 (灰色)
        - 目标点 (绿色)
        - 障碍物 (红色)
        - 机器人位置标记 (蓝色)
        """
        # 创建空白图像
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 128
        
        # 绘制地面网格 (简化)
        for i in range(0, self.image_size, 20):
            img[i, :, :] = 100
            img[:, i, :] = 100
        
        # 绘制目标点 (绿色，在图像中心上方)
        goal_x = int(self.image_size * 0.5)
        goal_y = int(self.image_size * 0.3)
        cv2.circle(img, (goal_x, goal_y), 10, (0, 255, 0), -1)
        
        # 绘制障碍物 (红色)
        for obs in self.obstacles:
            # 将世界坐标转换为图像坐标 (简化投影)
            obs_x = int((obs['x'] - self.robot_x) / 15.0 * self.image_size + self.image_size / 2)
            obs_y = int((obs['y'] - self.robot_y) / 15.0 * self.image_size + self.image_size / 2)
            
            # 限制在图像范围内
            obs_x = np.clip(obs_x, 0, self.image_size - 1)
            obs_y = np.clip(obs_y, 0, self.image_size - 1)
            
            size = int(obs['size'] * 50)
            cv2.circle(img, (obs_x, obs_y), size, (0, 0, 255), -1)
        
        # 应用故障效果
        if self.fault_active:
            img = self._apply_fault_to_image(img)
        
        return img
    
    def _apply_fault_to_image(self, img: np.ndarray) -> np.ndarray:
        """应用故障效果到图像"""
        if self.fault_type == 'lighting_drop':
            # 光照突变 (变暗)
            intensity = self.fault_params.get('intensity', 0.5)
            img = (img * (1 - intensity)).astype(np.uint8)
        
        elif self.fault_type == 'occlusion':
            # 摄像头遮挡 (黑色矩形)
            h, w = img.shape[:2]
            mask_ratio = self.fault_params.get('mask_ratio', 0.3)
            mask_h = int(h * mask_ratio)
            mask_w = int(w * mask_ratio)
            x = np.random.randint(0, w - mask_w)
            y = np.random.randint(0, h - mask_h)
            img[y:y+mask_h, x:x+mask_w, :] = 0
        
        elif self.fault_type == 'adversarial_patch':
            # 对抗补丁 (随机噪声块)
            h, w = img.shape[:2]
            patch_size = 32
            x = np.random.randint(0, w - patch_size)
            y = np.random.randint(0, h - patch_size)
            patch = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)
            img[y:y+patch_size, x:x+patch_size, :] = patch
        
        elif self.fault_type == 'depth_noise':
            # 深度噪声 (高斯噪声)
            noise_std = self.fault_params.get('noise_std', 50)
            noise = np.random.normal(0, noise_std, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def step(self, action: Dict, fault_info: Optional[Dict] = None) -> Tuple[Dict, float, bool, Dict]:
        """环境步进"""
        # 应用故障注入
        if fault_info and fault_info.get('active', False):
            self.fault_active = True
            self.fault_type = fault_info.get('type')
            self.fault_params = fault_info.get('params', {})
            action = self._inject_fault(action, fault_info)
        else:
            self.fault_active = False
        
        # 底盘动力学 (Unicycle 模型)
        base = self.state['base'].copy()
        base[0] += base[3] * np.cos(base[2]) * self.dt  # x
        base[1] += base[3] * np.sin(base[2]) * self.dt  # y
        base[2] += base[4] * self.dt                     # θ
        base[3] += action.get('v', 0) * self.a_max * self.dt   # v
        base[4] += action.get('ω', 0) * self.α_max * self.dt   # ω
        
        # 速度限制
        base[3] = np.clip(base[3], -self.v_max, self.v_max)
        base[4] = np.clip(base[4], -self.ω_max, self.ω_max)
        
        # 更新机器人位置 (PyBullet)
        if self.use_pybullet and p.isConnected() and p.isValid(self.robot_id):
            self.robot_x = base[0]
            self.robot_y = base[1]
            self.robot_yaw = base[2]
            
            # 更新 PyBullet 中机器人位置
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [self.robot_x, self.robot_y, 0.5],
                [0, 0, np.sin(self.robot_yaw/2), np.cos(self.robot_yaw/2)]
            )
        
        # 机械臂简化动力学
        arm_q = self.state['arm_q'] + action.get('τ', np.zeros(7)) / self.τ_limits * 0.1
        arm_q = np.clip(arm_q, -np.pi, np.pi)
        
        # 更新状态
        self.state['base'] = base
        self.state['arm_q'] = arm_q
        
        self.step_count += 1
        
        # 计算奖励 (接近目标 + 避免碰撞)
        reward = self._compute_reward()
        
        # 检查终止条件
        done = self._check_done()
        
        # 信息
        info = {
            'step': self.step_count,
            'collision': self._check_collision(),
            'zmp_stable': self._check_zmp_stable(),
            'reached_goal': self._check_goal_reached(),
            'fault_active': self.fault_active,
        }
        
        return self._get_observation(), reward, done, info
    
    def _inject_fault(self, action: Dict, fault_info: Dict) -> Dict:
        """注入故障到动作"""
        fault_type = fault_info.get('type')
        
        if fault_type == 'lighting_drop':
            # 光照突变 → 视觉特征噪声 → 动作错误
            action = action.copy()
            action['v'] *= -1
            action['τ'] = -action.get('τ', np.zeros(7))
        
        elif fault_type == 'payload_shift':
            # 负载突变 → 动力学改变 (动作效果减弱)
            action = action.copy()
            action['v'] *= 0.7
            action['τ'] = action.get('τ', np.zeros(7)) * 0.7
        
        elif fault_type == 'joint_friction':
            # 摩擦激增
            action = action.copy()
            action['τ'] = action.get('τ', np.zeros(7)) * 0.5
        
        return action
    
    def _get_observation(self) -> Dict:
        """获取观测"""
        return {
            'image': self.get_camera_image(),  # RGB 图像
            'base': self.state['base'].copy(),
            'arm_q': self.state['arm_q'].copy(),
            'arm_dq': self.state['arm_dq'].copy(),
            'obstacles': self.obstacles.copy(),
            'step': self.step_count,
        }
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        # 目标位置
        goal_x = 10.0
        
        # 距离奖励 (越近越好)
        dist_reward = -abs(self.state['base'][0] - goal_x)
        
        # 碰撞惩罚
        if self._check_collision():
            dist_reward -= 10.0
        
        # ZMP 不稳定惩罚
        if not self._check_zmp_stable():
            dist_reward -= 5.0
        
        return dist_reward
    
    def _check_done(self) -> bool:
        """检查是否终止"""
        # 到达目标
        if self._check_goal_reached():
            return True
        
        # 碰撞
        if self._check_collision():
            return True
        
        # ZMP 不稳定 (摔倒)
        if not self._check_zmp_stable():
            return True
        
        # 超时
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def _check_collision(self) -> bool:
        """检查碰撞"""
        for obs in self.obstacles:
            dist = np.sqrt((obs['x'] - self.robot_x)**2 + (obs['y'] - self.robot_y)**2)
            if dist < self.d_min:
                return True
        return False
    
    def _check_zmp_stable(self) -> bool:
        """检查 ZMP 稳定性 (简化)"""
        # 简化：假设始终稳定
        return True
    
    def _check_goal_reached(self) -> bool:
        """检查是否到达目标"""
        goal_x = 10.0
        return abs(self.state['base'][0] - goal_x) < 0.5
    
    def close(self):
        """关闭环境"""
        if self.use_pybullet and p.isConnected():
            p.disconnect()
    
    def __del__(self):
        self.close()


# 测试
if __name__ == '__main__':
    print("测试 FetchMobileEnvWithVision...")
    
    env = FetchMobileEnvWithVision(render=True, use_pybullet=False)
    
    for scene in ['empty', 'sparse', 'dense']:
        print(f"\n测试场景：{scene}")
        obs = env.reset(scene_type=scene)
        
        img = obs['image']
        print(f"  图像形状：{img.shape}")
        print(f"  图像类型：{img.dtype}")
        
        # 保存测试图像
        import cv2
        cv2.imwrite(f'test_scene_{scene}.png', img)
        print(f"  已保存：test_scene_{scene}.png")
    
    env.close()
    print("\n✅ 测试完成")
