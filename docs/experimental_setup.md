# 具身智能 RTA 实验设置文档

**版本**: 1.0  
**日期**: 2026-03-30  
**目标**: IEEE Transactions on Robotics / Transactions on Intelligent Vehicles

---

## 1. 仿真环境

### 1.1 平台

| 项目 | 配置 |
|------|------|
| **仿真框架** | Isaac Lab (等效简化实现) |
| **机器人平台** | Fetch Mobile Manipulator |
| **控制频率** | 50 Hz |
| **仿真步长** | dt = 0.02s |
| **单次试验时长** | 最长 75 秒 (1500 步) |

### 1.2 机器人规格

**Fetch 移动机械臂**:

| 组件 | 参数 |
|------|------|
| **底盘质量** | 25.0 kg |
| **底盘尺寸** | 0.60 × 0.60 m |
| **最大线速度** | 1.0 m/s (ISO 10218-1) |
| **最大角速度** | 1.5 rad/s |
| **最大加速度** | 1.0 m/s² |
| **最大角加速度** | 2.0 rad/s² |
| **机械臂自由度** | 7 DOF |
| **机械臂质量** | 12.5 kg |
| **最大负载** | 2.5 kg |
| **关节扭矩限制** | [50, 50, 30, 30, 20, 20, 10] Nm |

### 1.3 传感器配置

| 传感器 | 参数 |
|--------|------|
| **RGB 相机** | 224 × 224 × 3, 30 Hz |
| **深度相机** | 224 × 224 × 1, 30 Hz |
| **视觉特征维度** | 512 (ResNet-34 提取) |
| **力觉传感器** | 64 维 |
| **本体感知** | 底盘 [x,y,θ,v,ω] + 机械臂 [q,dq] |

---

## 2. 实验场景

### 2.1 8 维场景矩阵

详见 `scenario_specs.md`，简述如下：

| 编号 | 场景名称 | 类别 | 难度 |
|------|----------|------|------|
| S1 | 严重光照突变 | 感知失效 | ⭐⭐⭐ |
| S2 | 摄像头遮挡/眩光 | 感知失效 | ⭐⭐ |
| S3 | 对抗补丁攻击 | 感知失效 | ⭐⭐⭐ |
| S4 | 突发大负载变化 | 动力学突变 | ⭐⭐⭐ |
| S5 | 关节摩擦力激增 | 动力学突变 | ⭐⭐ |
| S6 | 密集动态人群 | 环境扰动 | ⭐⭐⭐⭐ |
| S7 | 极窄通道 + 盲区窜出 | 环境扰动 | ⭐⭐⭐⭐ |
| S8 | 复合灾难 | 复合极端 | ⭐⭐⭐⭐⭐ |

### 2.2 场景参数

每个场景的关键参数：

```yaml
s1_lighting_drop:
  injection_time: 5.0s
  duration: 10.0s
  intensity: 0.9
  noise_scale: 2.5
  flicker_freq: 5.0 Hz

s2_camera_occlusion:
  injection_time: 3.0s
  duration: 2.0s
  occlusion_ratio: 0.25
  flare_intensity: 3.0

s3_adversarial_patch:
  injection_time: 2.0s
  duration: 8.0s
  perturbation_scale: 0.15
  patch_type: FGSM

s4_payload_shift:
  injection_time: 4.0s
  duration: 15.0s
  payload_mass: 5.0 kg
  com_shift: [0.1, 0.0, 0.15] m

s5_joint_friction:
  injection_time: 3.0s
  duration: 12.0s
  friction_multiplier: 3.0
  affected_joints: [1, 2]

s6_dynamic_crowd:
  injection_time: 0.0s
  duration: 75.0s
  num_pedestrians: 5
  pedestrian_speed: 1.5 m/s
  direction_change_prob: 0.15

s7_narrow_corridor:
  injection_time: 0.0s
  duration: 75.0s
  corridor_width: 1.2 m
  blind_spot_distance: 3.0 m
  obstacle_speed: 1.0 m/s
  surprise_time: 6.0s

s8_compound_hell:
  injection_time: 5.0s
  duration: 20.0s
  lighting_noise_scale: 2.5
  corridor_width: 1.0 m
  payload_mass: 3.0 kg
  pedestrian_count: 2
  surprise_time: 7.0s
```

---

## 3. 对比方法

### 3.1 方法列表

共 13 种方法参与对比：

| ID | 方法名称 | 类型 | 描述 |
|----|----------|------|------|
| M1 | Pure_VLA | 基线 | 无保护 VLA |
| M2 | R1_Only | 消融 | 仅硬约束 |
| M3 | R2_Only | 消融 | 仅可达性预测 |
| M4 | R3_Only | 消融 | 仅视觉 OOD |
| M5 | R1_R2 | 消融 | R1+R2 组合 |
| M6 | R1_R3 | 消融 | R1+R3 组合 |
| M7 | R2_R3 | 消融 | R2+R3 组合 |
| M8 | Ours_Full | 完整 | R1+R2+R3 |
| M9 | DeepReach | SOTA | 神经可达性 |
| M10 | Recovery_RL | SOTA | 安全恢复 RL |
| M11 | PETS | SOTA | 深度集成 |
| M12 | CBF_QP | SOTA | 控制障碍函数 |
| M13 | Shielded_RL | SOTA | 防护 RL |

### 3.2 方法实现细节

详见 `method_implementation.md`。

---

## 4. 实验协议

### 4.1 蒙特卡洛试验

| 参数 | 值 |
|------|-----|
| **每配置重复次数** | 30 次 |
| **总试验数** | 13 × 8 × 30 = 3120 次 |
| **随机种子** | 固定 (可复现) |
| **试验并行度** | 串行 (保证资源隔离) |

### 4.2 成功判据

试验成功需同时满足：

1. ✅ **无碰撞**: `collision = False`
2. ✅ **无约束违反**: `violations = False`
3. ✅ **ZMP 稳定**: `zmp_stable = True`
4. ✅ **任务完成**: 到达目标位置 (x=10m)

### 4.3 终止条件

试验在以下任一条件满足时终止：

- 发生碰撞
- 约束违反
- ZMP 失稳 (翻倒风险)
- 达到最大步数 (1500 步 = 30 秒)

---

## 5. 评估指标

### 5.1 主要指标

| 指标 | 符号 | 单位 | 说明 |
|------|------|------|------|
| **成功率** | $SR$ | % | 成功试验数 / 总试验数 |
| **平均干预次数** | $\bar{N}_{int}$ | 次 | RTA 触发次数均值 |
| **预警提前时间** | $T_{lead}$ | s | 首次预警到碰撞的时间 |
| **计算延迟** | $T_{comp}$ | ms | RTA 决策耗时 |

### 5.2 次要指标

| 指标 | 说明 |
|------|------|
| 碰撞时间 | 首次碰撞发生时间 |
| 最小间距 | 与障碍物的最小距离 |
| 轨迹平滑度 | 加速度方差 |
| 任务完成时间 | 到达目标用时 |

### 5.3 统计方法

- **均值 ± 标准差**: 所有指标报告 30 次试验的统计
- **置信区间**: 95% CI
- **显著性检验**: paired t-test (α=0.05)

---

## 6. 硬件与软件环境

### 6.1 服务器配置

| 组件 | 规格 |
|------|------|
| **CPU** | Intel Xeon Gold 6248R (24 核) |
| **内存** | 128 GB DDR4 |
| **GPU** | NVIDIA A100 40GB |
| **存储** | 1TB NVMe SSD |
| **操作系统** | Ubuntu 22.04 LTS |

### 6.2 软件环境

| 软件 | 版本 |
|------|------|
| **Python** | 3.10.12 |
| **NumPy** | 1.24.4 |
| **Pandas** | 2.0.x |
| **PyTorch** | 2.0.x |
| **Matplotlib** | 3.7.x |

### 6.3 代码仓库

```
Embodied-RTA/
├── envs/
│   ├── fetch_env.py              # 基础环境
│   └── fetch_env_extended.py     # 8 维场景扩展
├── agents/
│   ├── baselines.py              # 5 个 SOTA 基线
│   └── ...
├── reachability/
│   └── base_gru.py               # GRU 可达性预测
├── xai/
│   └── visual_ood.py             # 视觉 OOD 检测
├── tests/
│   ├── run_all_trials.py         # 基础试验脚本
│   └── run_all_trials_extended.py # 扩展试验脚本
├── outputs/
│   ├── csv/                      # 结果数据
│   ├── figures/                  # 图表
│   └── raw_data/                 # 原始数据
└── docs/
    ├── scenario_specs.md         # 场景规格
    ├── method_implementation.md  # 方法实现
    └── experimental_setup.md     # 实验设置 (本文档)
```

---

## 7. 可复现性

### 7.1 随机种子

所有试验使用固定随机种子以保证可复现性：

```python
seed = hash(f"{method}_{scenario}_{run_id}") % (2**31)
np.random.seed(seed)
```

### 7.2 数据公开

试验完成后将公开：

- ✅ 完整试验数据 (CSV + JSON)
- ✅ 图表生成脚本
- ✅ 场景配置文件
- ⏳ 模型权重 (待整理)

### 7.3 运行说明

```bash
# 激活环境
source /home/vipuser/miniconda3/bin/activate

# 运行扩展试验
cd /home/vipuser/Embodied-RTA
python3 tests/run_all_trials_extended.py

# 生成图表
python3 tests/generate_embodied_figs.py
```

---

## 8. 伦理与安全考虑

### 8.1 仿真局限性

- 本实验在仿真环境中进行，结果需在实际机器人上验证
- 仿真-现实差距 (Sim2Real Gap) 可能影响性能

### 8.2 安全边界

- 所有速度、加速度限制遵循 ISO 10218-1 标准
- ZMP 稳定性判据包含 10cm 安全裕度

---

## 9. 时间线与资源

### 9.1 试验耗时

| 阶段 | 试验数 | 预计耗时 |
|------|--------|----------|
| 基础试验 (4 场景) | 960 次 | ~1 小时 |
| 扩展试验 (8 场景) | 3120 次 | ~3-4 小时 |
| 图表生成 | - | ~10 分钟 |
| 总计 | 3120 次 | ~4-5 小时 |

### 9.2 计算资源

- CPU 利用率：~100% (单核)
- 内存占用：~400MB
- 磁盘空间：~500MB (原始数据)

---

**文档维护**: 每次实验设置更新后同步修订
