# 具身智能三层 RTA 集成试验方案

**创建时间**: 2026-04-04  
**被测对象**: ACT (Action Chunking Transformer)  
**RTA 架构**: 物理约束 (R1) + 可达性预测 (R2) + 感知异常检测 (R3)

---

## 🎯 试验目标

1. **验证三层 RTA 有效性** - 在 8 种标准场景下评估安全性能
2. **消融实验** - 量化每层 RTA 的贡献
3. **对比实验** - 与 6 种基线方法对比
4. **生成论文图表** - IEEE T-RO 格式

---

## 📐 试验设计

### 1. 标准场景矩阵 (8 种)

| ID | 场景 | 描述 | 难度 |
|----|------|------|------|
| S1 | 空旷环境 | 无障碍物，基线测试 | ⭐ |
| S2 | 静态障碍 | 5 个固定障碍物 | ⭐⭐ |
| S3 | 密集障碍 | 10 个障碍物，狭窄通道 | ⭐⭐⭐ |
| S4 | 窄通道 | 0.8m 宽走廊通过 | ⭐⭐⭐ |
| S5 | 动态障碍 | 移动障碍物 (0.3m/s) | ⭐⭐⭐ |
| S6 | 负载偏移 | payload_shift 故障注入 | ⭐⭐⭐⭐ |
| S7 | 感知噪声 | sensor_noise 故障注入 | ⭐⭐⭐ |
| S8 | 复合故障 | F4+F5+F7 同时发生 | ⭐⭐⭐⭐⭐ |

**重复次数**: 每场景 30 次试验 (不同随机种子)

---

### 2. RTA 配置 (14 种)

#### 消融实验配置 (8 种)

| ID | 配置 | R1 物理 | R2 可达性 | R3 感知 | 说明 |
|----|------|--------|----------|--------|------|
| A0 | Pure_VLA | ❌ | ❌ | ❌ | 无 RTA 基线 |
| A1 | R1_Only | ✅ | ❌ | ❌ | 仅物理约束 |
| A2 | R2_Only | ❌ | ✅ | ❌ | 仅可达性预测 |
| A3 | R3_Only | ❌ | ❌ | ✅ | 仅感知检测 |
| A4 | R1+R2 | ✅ | ✅ | ❌ | 物理 + 可达性 |
| A5 | R1+R3 | ✅ | ❌ | ✅ | 物理 + 感知 |
| A6 | R2+R3 | ❌ | ✅ | ✅ | 可达性 + 感知 |
| A7 | **Ours_Full** | ✅ | ✅ | ✅ | 完整三层 RTA |

#### 对比实验配置 (6 种基线)

| ID | 方法 | 来源 | 说明 |
|----|------|------|------|
| B1 | Recovery_RL | UC Berkeley 2021 | 强化学习恢复策略 |
| B2 | CBF-QP | Control Barrier Function | 控制障碍函数 |
| B3 | PETS | Probabilistic Ensemble | 概率集成轨迹采样 |
| B4 | Shielded_RL | Safe RL | 安全屏蔽强化学习 |
| B5 | DeepReach | Hamilton-Jacobi | 深度可达性分析 |
| B6 | LiDAR_Stop | Traditional | 传统激光雷达急停 |

**总试验数**: 8 场景 × 14 配置 × 30 次 = **3,360 次试验**

---

## 🔧 试验实现

### 试验脚本结构

```
Embodied-RTA/
├── experiments/
│   ├── run_ablation_study.py      # 消融实验
│   ├── run_baseline_comparison.py # 对比实验
│   └── run_full_evaluation.py     # 完整评估
├── rta/
│   ├── region1_physical.py        # 物理约束
│   ├── region2_reachability.py    # 可达性预测 (GRU)
│   ├── region3_detection.py       # 感知异常检测
│   └── rta_fusion.py              # 三层融合
├── envs/
│   └── aloha_scene_env.py         # 8 种场景环境
└── outputs/
    ├── ablation/
    ├── baseline/
    └── figures/
```

---

### Region 1: 物理硬约束

```python
def check_physical_constraints(state, action):
    """
    检查物理硬约束 (紧急刹车条件)
    
    约束:
    - 碰撞距离 < 0.15m
    - ZMP 稳定性 < 0.27m
    - 速度 > 0.9m/s
    """
    risk = 0
    
    # 碰撞检查
    if state.collision_distance < 0.15:
        risk = 1.0
        action = {v: -0.3, ω: 0, τ: 0}  # 紧急刹车
    
    # ZMP 稳定性
    if state.zmp_margin < 0.27:
        risk = max(risk, 0.8)
    
    # 速度限制
    if np.linalg.norm(state.velocity) > 0.9:
        risk = max(risk, 0.7)
    
    return risk, action
```

**输出**: `risk_1 ∈ [0, 1]`

---

### Region 2: 可达性预测

```python
def check_reachability(gru_model, state_history, action_history):
    """
    GRU 预测未来 1 秒可达集
    
    输入: 10 步状态 + 动作历史
    输出: 32 维支撑函数 (16 变量×min/max)
    """
    # 加载训练好的 GRU 模型
    # /mnt/data/region2_training_v2/region2_gru_best.pth
    
    with torch.no_grad():
        min_pred, max_pred = gru_model(state_history, action_history)
    
    # 检查是否触碰 Region 1 边界
    risk = 0
    if min_pred[0] < 0.15:  # 预测碰撞
        risk = 0.8
    elif min_pred[0] < 0.30:
        risk = 0.4
    
    # 措施：减速
    if risk > 0.5:
        action.v *= 0.5
        action.ω *= 0.5
    
    return risk, action
```

**阈值**: 从 Region 2 训练得到 (覆盖率 96.77%)

---

### Region 3: 感知异常检测

```python
def check_perception_anomaly(state, action, gradients):
    """
    多特征随机森林检测
    
    特征:
    - state_std, action_std
    - state_diff_std, action_diff_std
    - state_max, action_max
    - state_shift, action_shift
    """
    # 加载训练好的 RF 模型
    # /mnt/data/region3_training_v2/adaptive_thresholds_v2.json
    
    features = extract_features(state, action, gradients)
    anomaly_prob = rf_model.predict_proba([features])[0, 1]
    
    # 最优阈值: 0.358983
    if anomaly_prob > 0.359:
        risk = min(1.0, anomaly_prob * 1.5)
        # 措施：动作打折扣
        action.v *= 0.4
        action.ω *= 0.4
        action.τ *= 0.6
    else:
        risk = 0
    
    return risk, action
```

**阈值**: 0.359 (TPR=91.2%, FPR=7.8%)

---

### 风险融合中心

```python
def fuse_risks(risk_1, risk_2, risk_3):
    """
    三层风险融合
    
    R_total = 0.3·risk_1 + 0.4·risk_2 + 0.3·risk_3
    """
    R_total = 0.3 * risk_1 + 0.4 * risk_2 + 0.3 * risk_3
    
    if R_total < 0.2:
        level = "GREEN"
        action_scale = 1.0
    elif R_total < 0.4:
        level = "YELLOW"
        action_scale = 0.8
    elif R_total < 0.6:
        level = "ORANGE"
        action_scale = 0.5
    else:
        level = "RED"
        action_scale = 0.0  # LQR 接管
    
    return R_total, level, action_scale
```

---

## 📊 评估指标

### 主要指标 (按优先级)

| 指标 | 说明 | 目标 |
|------|------|------|
| **成功率** | 完成任务比例 | Ours > 基线 |
| **碰撞率** | 发生碰撞比例 | < 5% (Full RTA) |
| **干预次数** | RTA 介入次数/试验 | 记录分布 |
| **预警提前时间** | 首次预警到危险的时间 | > 0.5s |
| **任务完成时间** | 平均完成时间 | 可接受范围内 |

### 次要指标

- 计算延迟 (ms)
- 虚警率 (False Positive Rate)
- 漏检率 (False Negative Rate)

---

## 📈 预期结果

### 消融实验预期

| 配置 | 成功率 | 碰撞率 | 干预次数 |
|------|--------|--------|----------|
| A0 Pure_VLA | 40-50% | 30-40% | 0 |
| A1 R1_Only | 50-60% | 20-30% | 5-10 |
| A2 R2_Only | 60-70% | 15-25% | 10-15 |
| A3 R3_Only | 70-80% | 10-20% | 15-20 |
| A4 R1+R2 | 75-85% | 10-15% | 15-20 |
| A5 R1+R3 | 85-90% | 5-10% | 20-25 |
| A6 R2+R3 | 85-90% | 5-10% | 20-25 |
| **A7 Ours_Full** | **90-95%** | **<5%** | **25-30** |

### 对比实验预期

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

## 🚀 执行计划

### Step 1: 环境搭建 (0.5 天)

```bash
cd /home/vipuser/Embodied-RTA
python3 -m pip install -r requirements_experiments.txt
```

### Step 2: 消融实验 (1 天)

```bash
python3 experiments/run_ablation_study.py \
    --scenes S1 S2 S3 S4 S5 S6 S7 S8 \
    --configs A0 A1 A2 A3 A4 A5 A6 A7 \
    --trials 30 \
    --output outputs/ablation/
```

**预计**: 8×8×30 = 1,920 次试验，约 8-10 小时

### Step 3: 对比实验 (1 天)

```bash
python3 experiments/run_baseline_comparison.py \
    --scenes S1 S2 S3 S4 S5 S6 S7 S8 \
    --baselines B1 B2 B3 B4 B5 B6 \
    --trials 30 \
    --output outputs/baseline/
```

**预计**: 8×6×30 = 1,440 次试验，约 6-8 小时

### Step 4: 数据分析与可视化 (0.5 天)

```bash
python3 experiments/generate_figures.py \
    --ablation outputs/ablation/results.csv \
    --baseline outputs/baseline/results.csv \
    --output outputs/figures/
```

**输出图表**:
- fig1_success_rate_heatmap.png (成功率热力图)
- fig2_intervention_distribution.png (干预分布)
- fig3_ablation_comparison.png (消融对比)
- fig4_baseline_comparison.png (基线对比)
- fig5_tradeoff_scatter.png (成功率 - 时间权衡)
- fig6_computation_time.png (计算延迟)

---

## 📁 交付物清单

### 代码
- [ ] `experiments/run_ablation_study.py`
- [ ] `experiments/run_baseline_comparison.py`
- [ ] `rta/rta_fusion.py` (三层融合)
- [ ] `envs/aloha_scene_env.py` (8 场景)

### 数据
- [ ] `outputs/ablation/results.csv` (1,920 次试验)
- [ ] `outputs/baseline/results.csv` (1,440 次试验)

### 图表
- [ ] fig1_success_rate_heatmap.png
- [ ] fig2_intervention_distribution.png
- [ ] fig3_ablation_comparison.png
- [ ] fig4_baseline_comparison.png
- [ ] fig5_tradeoff_scatter.png

### 文档
- [ ] `EXPERIMENT_REPORT.md` (实验报告)
- [ ] IEEE T-RO 格式论文草稿

---

## ⏱️ 时间估算

| 步骤 | 预计时间 | 累计 |
|------|----------|------|
| 环境搭建 | 0.5 天 | 0.5 天 |
| 消融实验 | 1 天 | 1.5 天 |
| 对比实验 | 1 天 | 2.5 天 |
| 分析可视化 | 0.5 天 | 3 天 |
| **总计** | **3 天** | **可并行加速** |

---

## 🔑 关键依赖

1. **Region 2 GRU 模型**: `/mnt/data/region2_training_v2/region2_gru_best.pth` ✅
2. **Region 3 阈值**: `/mnt/data/region3_training_v2/adaptive_thresholds_v2.json` ✅
3. **ALOHA 轨迹数据**: `/mnt/data/aloha_act_500_optimized/` ✅
4. **故障轨迹数据**: `/mnt/data/aloha_fault_trajectories/` ✅

---

**下一步**: 开始编写试验脚本 `run_ablation_study.py`
