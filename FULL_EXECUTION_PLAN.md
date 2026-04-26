# Embodied-RTA 完整执行计划 (基于 OpenVLA 真实数据)

**创建时间**: 2026-04-01 06:40  
**核心改进**: 所有训练数据基于 OpenVLA-7B 真实轨迹

---

## 🎯 核心改进点

### 问题 1: 仿真环境视觉输入 ✅
**改进**: 创建 `envs/fetch_env_vision.py`
- PyBullet 真实渲染 (224×224×3 RGB)
- 合成图像模式 (无 PyBullet 降级)
- 故障效果注入 (光照/遮挡/对抗补丁)

### 问题 2: Region 2 基于 OpenVLA 真实轨迹 ✅
**改进**: 
- `scripts/collect_openvla_trajectories.py` - 收集 100 场景真实轨迹
- `reachability/generate_openvla_training_data.py` - 基于真实轨迹生成可达集

### 问题 3: Region 3 从 OpenVLA 真实数据学习 ✅
**改进**:
- `xai/learn_from_openvla.py` - 学习参考统计、掩码库、预警阈值

### 问题 4: 对比试验与消融试验 ✅
**改进**: 完整的试验设计 (见下方)

---

## 📋 执行步骤

### Step 1: 收集 OpenVLA 真实轨迹 (高优先级)

**脚本**: `scripts/collect_openvla_trajectories.py`

**场景配置**:
- 100 个随机场景 (empty/sparse/dense)
- 5 种故障类型 + 正常 baseline
- OpenVLA 输入：真实 RGB 图像
- 任务：避开障碍物到达目标点

**命令** (远程服务器):
```bash
cd /home/vipuser/Embodied-RTA
/home/vipuser/miniconda3/bin/python scripts/collect_openvla_trajectories.py
```

**预计时间**: 2-3 小时 (取决于 OpenVLA 推理速度)

**输出**:
- `reachability/openvla_trajectories_normal.pkl` (~500 条正常轨迹)
- `reachability/openvla_trajectories_faults.pkl` (~2000 条故障轨迹)

---

### Step 2: 生成 Region 2 训练数据

**脚本**: `reachability/generate_openvla_training_data.py`

**流程**:
1. 加载 OpenVLA 真实轨迹
2. 滑动窗口提取状态序列 (10 帧)
3. 蒙特卡洛可达集推演 (100 样本)
4. 计算支撑函数 (16 变量×2)
5. 保存 HDF5 格式

**命令**:
```bash
/home/vipuser/miniconda3/bin/python reachability/generate_openvla_training_data.py
```

**预计时间**: 30-60 分钟

**输出**:
- `reachability/openvla_reachability_dataset.h5` (~500MB)
- 数据规格：(N, 10, 19) → (N, 32)

---

### Step 3: 训练 Region 2 GRU 模型

**脚本**: `reachability/train_gru_reachability.py` (需更新数据路径)

**修改**:
```python
# 原路径
data_path = 'reachability/reachability_dataset_v2.h5'

# 新路径 (基于 OpenVLA)
data_path = 'reachability/openvla_reachability_dataset.h5'
```

**命令**:
```bash
/home/vipuser/miniconda3/bin/python reachability/train_gru_reachability.py
```

**预计时间**: 2-3 小时 (V100 GPU)

**输出**:
- `reachability/gru_openvla.pth` (最佳模型)
- `reachability/training_log_openvla.csv`

---

### Step 4: 学习 Region 3 激活统计和掩码

**脚本**: `xai/learn_from_openvla.py`

**流程**:
1. 加载 OpenVLA 模型 + 轨迹数据
2. 注册激活钩子，收集正常/故障激活
3. 计算参考统计 (mean, std, cov)
4. 互信息特征选择 → 掩码库 (Top 50 关键神经元)
5. ROC 曲线分析 → 最优预警阈值

**命令**:
```bash
/home/vipuser/miniconda3/bin/python xai/learn_from_openvla.py
```

**预计时间**: 1-2 小时

**输出**:
- `xai/openvla_reference_stats.pkl` (每层激活统计)
- `xai/openvla_activation_masks.pkl` (掩码库)
- `xai/openvla_thresholds.pkl` (最优阈值 + ROC 曲线)

---

### Step 5: 更新 Region 3 检测器使用学习参数

**文件**: `xai/multi_layer_activation.py`

**修改**:
```python
# 加载学习到的统计
with open('openvla_reference_stats.pkl', 'rb') as f:
    self.reference_stats = pickle.load(f)

# 加载掩码库
with open('openvla_activation_masks.pkl', 'rb') as f:
    self.activation_masks = pickle.load(f)

# 加载阈值
with open('openvla_thresholds.pkl', 'rb') as f:
    thresholds = pickle.load(f)
    self.threshold_risk = thresholds['optimal_threshold']
```

---

### Step 6: 消融试验 (Ablation Study)

**脚本**: `tests/run_ablation_study.py` (需创建)

**8 种配置**:
| ID | 配置 | R1 | R2 | R3 |
|----|------|----|----|----|
| 0 | Pure_VLA | ❌ | ❌ | ❌ |
| 1 | R1_Only | ✅ | ❌ | ❌ |
| 2 | R2_Only | ❌ | ✅ | ❌ |
| 3 | R3_Only | ❌ | ❌ | ✅ |
| 4 | R1+R2 | ✅ | ✅ | ❌ |
| 5 | R1+R3 | ✅ | ❌ | ✅ |
| 6 | R2+R3 | ❌ | ✅ | ✅ |
| 7 | **Ours_Full** | ✅ | ✅ | ✅ |

**场景**:
- 3 基础场景 (empty/sparse/dense)
- 5 故障类型 (lighting/occlusion/adversarial/payload/friction)
- 10 次重复

**试验规模**: 3 × 5 × 8 × 10 = **1,200 次**

**命令**:
```bash
/home/vipuser/miniconda3/bin/python tests/run_ablation_study.py
```

**预计时间**: 4-6 小时

**输出**:
- `outputs/ablation_results.csv`
- `outputs/ablation_success_rates.png`
- `outputs/ablation_interventions.png`

---

### Step 7: 对比试验 (Baseline Comparison)

**脚本**: `tests/run_baseline_comparison.py` (需创建)

**对比方法**:
1. Recovery_RL (UC Berkeley 2021)
2. CBF-QP (控制障碍函数)
3. PETS (概率集成轨迹采样)
4. Shielded_RL
5. DeepReach
6. LiDAR_Stop (传统)

**试验规模**: 3 × 5 × 7 × 10 = **1,050 次**

**命令**:
```bash
/home/vipuser/miniconda3/bin/python tests/run_baseline_comparison.py
```

**预计时间**: 4-6 小时

**输出**:
- `outputs/baseline_results.csv`
- `outputs/baseline_success_rates.png`
- `outputs/baseline_comparison.png`

---

## 📊 预期结果

### 消融试验预期

| 配置 | 成功率 | 碰撞率 | 干预次数 |
|------|--------|--------|----------|
| Pure_VLA | 40-50% | 30-40% | 0 |
| R1_Only | 50-60% | 20-30% | 5-10 |
| R2_Only | 60-70% | 15-25% | 10-15 |
| R3_Only | 70-80% | 10-20% | 15-20 |
| R1+R2 | 75-85% | 10-15% | 15-20 |
| R1+R3 | 85-90% | 5-10% | 20-25 |
| R2+R3 | 85-90% | 5-10% | 20-25 |
| **Ours_Full** | **90-95%** | **<5%** | **25-30** |

### 对比试验预期

| 方法 | 成功率 | vs Pure_VLA |
|------|--------|-------------|
| Pure_VLA | 45% | - |
| LiDAR_Stop | 55% | +10% |
| Shielded_RL | 60% | +15% |
| CBF-QP | 65% | +20% |
| PETS | 70% | +25% |
| Recovery_RL | 75% | +30% |
| DeepReach | 80% | +35% |
| **Ours_Full** | **92%** | **+47%** |

---

## 📁 交付物清单

### 代码
- [x] `envs/fetch_env_vision.py` - 视觉环境
- [x] `scripts/collect_openvla_trajectories.py` - 轨迹收集
- [x] `reachability/generate_openvla_training_data.py` - Region 2 数据生成
- [x] `xai/learn_from_openvla.py` - Region 3 学习
- [ ] `tests/run_ablation_study.py` - 消融试验
- [ ] `tests/run_baseline_comparison.py` - 对比试验

### 数据
- [ ] `openvla_trajectories_*.pkl` - 真实轨迹
- [ ] `openvla_reachability_dataset.h5` - Region 2 训练数据
- [ ] `openvla_reference_stats.pkl` - Region 3 参考统计
- [ ] `openvla_activation_masks.pkl` - 掩码库
- [ ] `openvla_thresholds.pkl` - 预警阈值
- [ ] `gru_openvla.pth` - 训练好的 GRU 模型

### 结果
- [ ] `ablation_results.csv` + 图表
- [ ] `baseline_results.csv` + 图表
- [ ] 实验报告 (IEEE T-RO 格式)

---

## ⏱️ 时间估算

| 步骤 | 预计时间 | 累计时间 |
|------|----------|----------|
| Step 1: 轨迹收集 | 3h | 3h |
| Step 2: Region 2 数据 | 1h | 4h |
| Step 3: GRU 训练 | 3h | 7h |
| Step 4: Region 3 学习 | 2h | 9h |
| Step 5: 更新检测器 | 0.5h | 9.5h |
| Step 6: 消融试验 | 6h | 15.5h |
| Step 7: 对比试验 | 6h | 21.5h |
| **总计** | **~22 小时** | **可并行加速** |

---

## 🚀 立即可执行

**当前状态**:
- ✅ OpenVLA 集成测试通过 (1570 个钩子)
- ✅ 视觉环境创建完成
- ✅ 数据收集脚本就绪

**下一步**:
```bash
# 在远程服务器上执行
cd /home/vipuser/Embodied-RTA

# Step 1: 收集轨迹 (3 小时)
/home/vipuser/miniconda3/bin/python scripts/collect_openvla_trajectories.py

# 监控进度
watch -n 60 'tail -5 reachability/trajectory_metadata.json'
```

---

**生成时间**: 2026-04-01 06:40  
**执行位置**: js3.blockelite.cn:13928 (vipuser)  
**模型路径**: /data/models/openvla-7b
