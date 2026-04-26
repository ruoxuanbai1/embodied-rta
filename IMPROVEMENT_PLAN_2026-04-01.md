# Embodied-RTA 关键问题改进方案

**更新时间**: 2026-04-01 06:28  
**问题来源**: 用户审查反馈

---

## 🔴 问题 1: 仿真环境缺少真实图像输入

### 现状
- `fetch_env.py` 使用随机视觉特征：`np.random.randn(512) * 0.1`
- **没有 RGB 图像渲染** → OpenVLA 无法接收真实图像输入
- **无法验证 Region 3 的视觉效果**

### 改进方案

#### 方案 A: 集成 PyBullet/Mujoco 渲染 (推荐)
```python
import pybullet as p
import pybullet_data

class FetchMobileEnvWithVision:
    def __init__(self, render=True):
        # 启动 PyBullet
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)  # 无头模式
        
        # 加载 Fetch 机器人模型
        self.robot = p.loadURDF("fetch_description/fetch.urdf")
        
        # 加载场景 (办公室/走廊)
        p.loadURDF("plane.urdf")
        
    def get_camera_image(self) -> np.ndarray:
        """获取相机 RGB 图像 (224×224×3)"""
        # 相机安装在机器人头部
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[self.robot_x, self.robot_y, 1.0],
            distance=0.5,
            yaw=self.robot_yaw,
            pitch=-20,
            roll=0,
            upAxisIndex=2
        )
        
        # 渲染图像
        img = p.getCameraImage(
            width=224, height=224,
            viewMatrix=view_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # 返回 RGB (去掉 alpha 通道)
        return img[2][:, :, :3]
    
    def get_depth_image(self) -> np.ndarray:
        """获取深度图像"""
        img = p.getCameraImage(...)
        return img[3]  # 深度通道
```

**依赖**: `pip install pybullet`

#### 方案 B: 使用 Isaac Lab/Gym (如果已安装)
```python
from isaacgym import gymapi, gymtorch

# Isaac 已有完整视觉渲染
```

#### 方案 C: 简化版 - 使用预渲染图像 + 合成 (快速验证)
```python
class PseudoVisionEnv:
    def __init__(self):
        # 加载真实场景图像库
        self.scene_library = load_scene_images('scenes/')
        
    def get_camera_image(self, state):
        # 根据机器人位置选择最接近的场景
        scene = self._select_nearest_scene(state)
        
        # 叠加障碍物 (简单合成)
        if self.obstacles:
            scene = self._overlay_obstacles(scene, self.obstacles)
        
        # 添加故障效果 (光照突变/遮挡/对抗补丁)
        if self.fault_active:
            scene = self._apply_fault(scene, self.fault_type)
        
        return scene
```

**推荐**: 先用方案 C 快速验证，再迁移到方案 A/B

---

## 🔴 问题 2: Region 2 训练数据需要基于 OpenVLA 真实轨迹

### 现状
- `generate_training_data.py` 使用**合成轨迹** (简化动力学模型)
- **不是 OpenVLA 真实输出** → 训练数据分布不匹配

### 改进方案

#### 两阶段训练流程

**阶段 1: 收集 OpenVLA 真实轨迹**
```python
# scripts/collect_openvla_trajectories.py
from agents.openvla_agent import OpenVLAAgent
from envs.fetch_env_vision import FetchMobileEnvWithVision

vla = OpenVLAAgent('/data/models/openvla-7b')
env = FetchMobileEnvWithVision(render=False)

trajectories = []

# 正常场景收集 (1000 条轨迹)
for episode in range(1000):
    obs = env.reset()
    trajectory = {'states': [], 'actions': [], 'images': []}
    
    for step in range(500):
        # 获取相机图像
        image = env.get_camera_image()
        
        # OpenVLA 推理
        action = vla.get_action(
            image=image,
            instruction="navigate to the goal"
        )
        
        # 记录
        trajectory['states'].append(env.state.copy())
        trajectory['actions'].append(action)
        trajectory['images'].append(image)
        
        # 环境步进
        obs, reward, done, info = env.step(action)
        
        if done:
            break
    
    trajectories.append(trajectory)

# 保存
import pickle
with open('reachability/openvla_trajectories_normal.pkl', 'wb') as f:
    pickle.dump(trajectories, f)

print(f"收集完成：{len(trajectories)} 条轨迹")
```

**阶段 2: 故障注入 + 可达集推演**
```python
# reachability/generate_training_data_v2.py (基于真实轨迹)
import pickle

# 加载真实轨迹
with open('openvla_trajectories_normal.pkl', 'rb') as f:
    normal_trajs = pickle.load(f)

# 故障场景收集 (500 条轨迹 × 5 故障类型)
fault_types = ['lighting_drop', 'occlusion', 'payload_shift', 
               'joint_friction', 'dynamic_intruder']

fault_trajs = []
for fault in fault_types:
    for episode in range(100):
        traj = inject_fault(normal_trajs[episode % len(normal_trajs)], fault)
        fault_trajs.append(traj)

# 蒙特卡洛可达集推演 (基于真实轨迹)
training_data = []
for traj in normal_trajs + fault_trajs:
    for window in sliding_window(traj['states'], window_size=10):
        # 从真实状态推演可达集
        reachable_set = monte_carlo_rollout(
            x0=window[-1],
            u_seq=traj['actions'][:10],
            n_samples=100,
            horizon=1.0
        )
        
        # 计算支撑函数
        support = compute_support_function(reachable_set)
        
        # 保存训练样本
        training_data.append({
            'input': window,      # (10, 19) 状态序列
            'output': support     # (32,) 支撑函数
        })

# 保存训练数据
save_training_data(training_data, 'reachability_dataset_v2.h5')
```

**关键改进**:
- ✅ 输入状态来自 OpenVLA 真实轨迹
- ✅ 动作分布匹配 OpenVLA 输出
- ✅ 故障注入在真实轨迹基础上进行

---

## 🔴 问题 3: Region 3 掩码库/特征/阈值需从 OpenVLA 真实数据学习

### 现状
- 掩码库是**手工定义**的 (8 个参考掩码)
- 阈值是基于**经验设置**的 (马氏距离 3σ, 熵 1.30 等)
- **不是从 OpenVLA 真实激活学习**

### 改进方案

#### Step 1: 收集 OpenVLA 正常激活数据
```python
# xai/collect_reference_activations.py
from agents.openvla_agent import OpenVLAAgent
from xai.multi_layer_activation import MultiLayerActivationHook
from envs.fetch_env_vision import FetchMobileEnvWithVision

vla = OpenVLAAgent('/data/models/openvla-7b')
hook_manager = MultiLayerActivationHook(vla.model)
env = FetchMobileEnvWithVision(render=False)

# 收集正常场景激活 (1000 次推理)
all_activations = {name: [] for name in hook_manager.hooks}

for episode in range(1000):
    obs = env.reset()
    image = env.get_camera_image()
    
    hook_manager.clear_activations()
    _ = vla.get_action(image, "navigate to the goal")
    
    # 保存各层激活
    for name, act in hook_manager.get_all_activations().items():
        all_activations[name].append(act.cpu().numpy())

# 计算统计 (均值、方差、协方差矩阵)
reference_stats = {}
for name, acts in all_activations.items():
    acts_flat = np.concatenate([a.flatten() for a in acts], axis=0)
    reference_stats[name] = {
        'mean': np.mean(acts_flat, axis=0),
        'std': np.std(acts_flat, axis=0),
        'cov': np.cov(acts_flat.T),  # 协方差矩阵 (用于马氏距离)
        'percentile_95': np.percentile(acts_flat, 95, axis=0),
        'percentile_05': np.percentile(acts_flat, 5, axis=0),
    }

# 保存
import pickle
with open('xai/openvla_reference_stats.pkl', 'wb') as f:
    pickle.dump(reference_stats, f)

print("✅ 参考统计计算完成")
```

#### Step 2: 学习掩码库 (关键特征组合)
```python
# xai/learn_activation_masks.py
from sklearn.feature_selection import mutual_info_classif

# 加载正常 + 故障激活数据
normal_acts = load_activations('normal/')  # 1000 次正常推理
fault_acts = load_activations('faults/')   # 500 次故障推理

# 对每一层，学习区分正常/故障的关键神经元
masks = {}
for layer_name in all_layers:
    # 正常激活
    normal_flat = np.array([a[layer_name].flatten() for a in normal_acts])
    
    # 故障激活
    fault_flat = np.array([a[layer_name].flatten() for a in fault_acts])
    
    # 标签
    X = np.concatenate([normal_flat, fault_flat], axis=0)
    y = np.concatenate([np.zeros(len(normal_flat)), np.ones(len(fault_flat))])
    
    # 互信息特征选择 (找出最能区分正常/故障的神经元)
    mi_scores = mutual_info_classif(X, y)
    
    # 选择 Top-K 关键神经元
    top_k_indices = np.argsort(mi_scores)[-K:]  # K=50
    
    # 保存掩码
    masks[layer_name] = {
        'key_neurons': top_k_indices,
        'weights': mi_scores[top_k_indices],
        'normal_mean': np.mean(normal_flat[:, top_k_indices], axis=0),
        'normal_std': np.std(normal_flat[:, top_k_indices], axis=0),
    }

# 保存掩码库
with open('xai/openvla_activation_masks.pkl', 'wb') as f:
    pickle.dump(masks, f)
```

#### Step 3: 学习预警阈值 (基于 ROC 曲线)
```python
# xai/learn_thresholds.py
from sklearn.metrics import roc_curve, auc

# 加载正常 + 故障风险分数
normal_risks = load_risks('normal/')  # 1000 次正常推理的风险分数
fault_risks = load_risks('faults/')   # 500 次故障推理的风险分数

# 标签
y_true = np.concatenate([np.zeros(len(normal_risks)), np.ones(len(fault_risks))])
y_scores = np.concatenate([normal_risks, fault_risks])

# ROC 曲线
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 选择最优阈值 (Youden's J 统计量)
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold = thresholds[optimal_idx]

print(f"最优阈值：{optimal_threshold:.3f}")
print(f"对应 TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f}")
print(f"AUC: {roc_auc:.3f}")

# 保存
with open('xai/openvla_thresholds.pkl', 'wb') as f:
    pickle.dump({
        'optimal_threshold': optimal_threshold,
        'tpr': tpr[optimal_idx],
        'fpr': fpr[optimal_idx],
        'auc': roc_auc,
    }, f)
```

---

## 🔴 问题 4: 需要对比试验与消融试验

### 完整试验设计

#### 消融试验 (Ablation Study)
验证三层 RTA 各自贡献：

| 配置 | Region 1 | Region 2 | Region 3 | 说明 |
|------|----------|----------|----------|------|
| **Pure_VLA** | ❌ | ❌ | ❌ | 无保护基线 |
| **R1_Only** | ✅ | ❌ | ❌ | 仅硬约束 |
| **R2_Only** | ❌ | ✅ | ❌ | 仅可达性预测 |
| **R3_Only** | ❌ | ❌ | ✅ | 仅感知异常检测 |
| **R1+R2** | ✅ | ✅ | ❌ | 物理安全 |
| **R1+R3** | ✅ | ❌ | ✅ | 感知 + 硬约束 |
| **R2+R3** | ❌ | ✅ | ✅ | 预测 + 感知 |
| **Ours_Full** | ✅ | ✅ | ✅ | 完整三层 |

#### 对比试验 (Baseline Comparison)
与现有方法对比：

| 方法 | 来源 | 说明 |
|------|------|------|
| **Recovery_RL** | UC Berkeley 2021 | 安全恢复强化学习 |
| **CBF-QP** | Ames et al. 2017 | 控制障碍函数 + 二次规划 |
| **PETS** | Chua et al. 2018 | 概率集成轨迹采样 |
| **Shielded_RL** | Alshiekh et al. 2018 | 防护强化学习 |
| **DeepReach** | Bansal et al. 2021 | 神经可达性分析 |
| **LiDAR_Stop** | 传统方法 | 激光雷达紧急停止 |

#### 场景设计 (100 组随机场景)

**第一组**: 3 基础场景 (empty/sparse/dense) × 5 故障 × 10 RTA × 10 次 = 1,500 次

**第二组**: 100 组随机障碍物配置 × 5 故障 × 10 RTA × 3 次 = 15,000 次

**总计**: 16,500 次试验

#### 评估指标

**主要指标** (证明 RTA 价值):
- 任务成功率 (有 RTA vs 无 RTA)
- 摔倒率 (ZMP 违反)
- 碰撞率
- 安全性提升倍数 = SR(Ours_Full) / SR(Pure_VLA)

**次要指标** (检测性能):
- 检测准确率/召回率
- 预警提前时间 (TTC)
- 虚警率
- 计算延迟

---

## 📋 执行清单

### 立即执行 (高优先级)
- [ ] **1.1** 实现简化视觉环境 (方案 C - 预渲染图像)
- [ ] **1.2** 集成 PyBullet 渲染 (方案 A - 完整实现)
- [ ] **2.1** 收集 OpenVLA 真实轨迹 (1000 条正常 + 500 条故障)
- [ ] **2.2** 基于真实轨迹重新生成 Region 2 训练数据
- [ ] **3.1** 收集 OpenVLA 正常激活参考数据
- [ ] **3.2** 学习掩码库 (关键特征组合)
- [ ] **3.3** 学习预警阈值 (ROC 曲线优化)
- [ ] **4.1** 设计消融试验脚本
- [ ] **4.2** 设计对比试验脚本

### 后续执行
- [ ] 运行完整试验 (16,500 次)
- [ ] 生成论文级图表
- [ ] 撰写实验报告

---

**生成时间**: 2026-04-01 06:28  
**下一步**: 等待 OpenVLA 集成测试完成，然后开始实施改进！
