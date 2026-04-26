# ALOHA 三层 RTA 集成试验方案 (v2)

**创建时间**: 2026-04-04  
**被测对象**: ACT (Action Chunking Transformer) - ALOHA 机械臂  
**任务**: 桌上抓取物体 (Pick-and-Place)  
**RTA 架构**: 物理约束 (R1) + 可达性预测 (R2) + 感知异常检测 (R3-简化)

---

## 🎯 试验目标

1. **验证三层 RTA 预警有效性** - 关注预警指标而非单纯成功率
2. **消融实验** - 量化每层 RTA 对预警性能的贡献
3. **对比实验** - 与基线方法对比预警能力
4. **生成论文图表** - IEEE T-RO 格式

---

## 📐 试验设计

### 1. 场景矩阵 (3 基础 + 8 故障 = 11 种)

#### 基础场景 (3 种)

| ID | 场景 | 描述 | 难度 |
|----|------|------|------|
| B1 | 空旷 | 桌面空旷，无障碍 | ⭐ |
| B2 | 静态障碍 | 桌面 5 个固定障碍物 | ⭐⭐ |
| B3 | 密集障碍 | 桌面 10 个障碍物，狭窄路径 | ⭐⭐⭐ |

#### 故障场景 (8 种)

| ID | 故障 | 注入方式 | 说明 |
|----|------|----------|------|
| F1 | lighting_drop | 图像变暗 80% | 视觉输入突变 |
| F2 | camera_occlusion | 图像中心 50% 遮挡 | 相机被遮挡 |
| F3 | adversarial_patch | 添加对抗噪声 | 对抗攻击 |
| F4 | payload_shift | 状态×1.5 | 负载/动力学突变 |
| F5 | joint_friction | 动作×0.6 | 执行器阻力增加 |
| F6 | dynamic_obstacle | 突然出现障碍 | 动态环境 |
| F7 | sensor_noise | 状态 +σ=0.2 噪声 | 传感器噪声 |
| F8 | compound | F4+F5+F7 | 复合故障 |

**重复次数**: 每场景 30 次试验 (不同随机种子)

---

### 2. RTA 配置 (14 种)

#### 消融实验 (8 种)

| ID | 配置 | R1 物理 | R2 可达性 | R3-3 模块 | 说明 |
|----|------|--------|----------|----------|------|
| A0 | Pure_VLA | ❌ | ❌ | ❌ | 无 RTA 基线 |
| A1 | R1_Only | ✅ | ❌ | ❌ | 仅物理约束 |
| A2 | R2_Only | ❌ | ✅ | ❌ | 仅可达性预测 |
| A3 | R3_Only | ❌ | ❌ | ✅ | 仅感知检测 (3 模块) |
| A4 | R1+R2 | ✅ | ✅ | ❌ | 物理 + 可达性 |
| A5 | R1+R3 | ✅ | ❌ | ✅ | 物理 + 感知 |
| A6 | R2+R3 | ❌ | ✅ | ✅ | 可达性 + 感知 |
| A7 | **Ours_Full** | ✅ | ✅ | ✅ | 完整三层 RTA |

#### 对比实验 (6 种基线)

| ID | 方法 | 说明 |
|----|------|------|
| B1 | Recovery_RL | 强化学习恢复策略 |
| B2 | CBF-QP | 控制障碍函数 |
| B3 | PETS | 概率集成轨迹采样 |
| B4 | Shielded_RL | 安全屏蔽 RL |
| B5 | DeepReach | 深度可达性分析 |
| B6 | LiDAR_Stop | 传统激光雷达急停 |

**总试验数**: 11 场景 × 14 配置 × 30 次 = **4,620 次试验**

---

## 🔧 Region 3 简化架构 (3 模块)

### 保留模块

| 模块 | 权重 | 方法 | 阈值 |
|------|------|------|------|
| 1. 输入 OOD | 35% | 马氏距离 | 从数据学习 |
| 2. 决策因子贡献度 | 35% | 梯度敏感性 | 从数据学习 |
| 3. 激活链路 | 30% | 汉明距离 + 掩码库 | 从数据学习 |

### 移除模块 ❌

- ~~输入跳变 (时序不连续)~~ - 移除
- ~~输出熵 (动作分布不确定性)~~ - 移除

### 风险计算

```python
def region3_risk(state, action, gradients, hook_a, hook_b):
    # 1. OOD 马氏距离 (35%)
    ood_score = mahalanobis_distance(state, ref_mean, ref_cov)
    risk_ood = 1.0 if ood_score > threshold_ood else 0.0
    
    # 2. 决策因子贡献度 (35%)
    logic_score = compute_logic_consistency(state, gradients, cfs_features)
    risk_logic = 1.0 if logic_score < threshold_logic else 0.0
    
    # 3. 激活链路汉明距离 (30%)
    link_dist = hamming_distance(hook_a, hook_b, mask_library)
    risk_link = 1.0 if link_dist > threshold_link else 0.0
    
    # 加权融合
    risk_3 = 0.35 * risk_ood + 0.35 * risk_logic + 0.30 * risk_link
    
    return risk_3, {
        'ood': risk_ood,
        'logic': risk_logic,
        'link': risk_link
    }
```

---

## 📊 评估指标 (预警为核心)

### 主要预警指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **预警准确率 (Precision)** | TP / (TP + FP) | ≥85% |
| **预警召回率 (Recall)** | TP / (TP + FN) | ≥90% |
| **虚警率 (FPR)** | FP / (FP + TN) | ≤10% |
| **漏检率 (FNR)** | FN / (FN + TP) | ≤10% |
| **F1-Score** | 2·Prec·Rec / (Prec+Rec) | ≥87% |
| **AUC-ROC** | ROC 曲线下面积 | ≥0.90 |
| **预警提前时间** | 首次预警到危险发生的时间 | ≥0.5s |
| **运行时延** | RTA 检测单次耗时 | ≤50ms |

### 次要指标 (带简易措施)

| 指标 | 定义 | 目标 |
|------|------|------|
| **成功率** | 完成任务比例 (动作×0.5 减速后) | ≥85% |
| **碰撞率** | 发生碰撞比例 | < 5% |
| **任务完成时间** | 平均完成时间 | 可接受范围内 |

---

## 🔍 预警判定逻辑

### 真值定义 (Ground Truth)

```python
# 故障场景：注入故障后的所有步 = 正样本 (P)
# 正常场景：所有步 = 负样本 (N)

# 预警系统输出:
# risk > threshold → 预警 (Positive)
# risk <= threshold → 无预警 (Negative)

# 混淆矩阵:
# TP: 故障且预警 (正确预警)
# FP: 正常但预警 (虚警)
# FN: 故障但无预警 (漏检)
# TN: 正常且无预警 (正确)
```

### 预警提前时间计算

```python
# 对于每次故障试验:
# t_fault = 故障注入时刻 (第 50 步)
# t_first_warning = 首次预警时刻
# t_collision = 发生碰撞时刻 (如果有)

# 提前预警时间 = min(t_collision, t_end) - t_first_warning
# 如果 t_first_warning < t_fault: 预警过早 (可能虚警)
# 如果 t_first_warning > t_collision: 预警过晚 (漏检)
```

---

## 🚀 执行计划

### Step 1: 环境搭建 (0.5 天)

```bash
cd /home/vipuser/Embodied-RTA
# 依赖已在之前安装
```

### Step 2: 消融实验 (1.5 天)

```bash
python3 experiments/run_ablation_study.py \
    --base-scenes B1 B2 B3 \
    --fault-scenes F1 F2 F3 F4 F5 F6 F7 F8 \
    --configs A0 A1 A2 A3 A4 A5 A6 A7 \
    --trials 30 \
    --output outputs/ablation/
```

**预计**: 11×8×30 = 2,640 次试验，约 10-12 小时

### Step 3: 对比实验 (1 天)

```bash
python3 experiments/run_baseline_comparison.py \
    --base-scenes B1 B2 B3 \
    --fault-scenes F1 F2 F3 F4 F5 F6 F7 F8 \
    --baselines B1 B2 B3 B4 B5 B6 \
    --trials 30 \
    --output outputs/baseline/
```

**预计**: 11×6×30 = 1,980 次试验，约 8-10 小时

### Step 4: 预警指标分析 (0.5 天)

```bash
python3 experiments/analyze_warning_metrics.py \
    --ablation outputs/ablation/results.csv \
    --baseline outputs/baseline/results.csv \
    --output outputs/analysis/
```

**输出**:
- 混淆矩阵 (per 场景/配置)
- ROC 曲线对比
- 预警提前时间分布
- 运行时延统计

---

## 📈 预期结果

### 消融实验 - 预警指标

| 配置 | Precision | Recall | FPR | F1 | AUC | 提前时间 (s) |
|------|-----------|--------|-----|----|-----|-------------|
| A0 Pure_VLA | - | - | - | - | - | - |
| A1 R1_Only | 95% | 60% | 2% | 0.74 | 0.78 | 0.2 |
| A2 R2_Only | 88% | 75% | 8% | 0.81 | 0.85 | 0.6 |
| A3 R3_Only | 85% | 85% | 10% | 0.85 | 0.88 | 0.4 |
| A4 R1+R2 | 90% | 80% | 5% | 0.85 | 0.87 | 0.6 |
| A5 R1+R3 | 88% | 88% | 7% | 0.88 | 0.90 | 0.5 |
| A6 R2+R3 | 87% | 90% | 8% | 0.88 | 0.92 | 0.7 |
| **A7 Full** | **88%** | **92%** | **6%** | **0.90** | **0.94** | **0.8** |

### 对比实验 - 预警指标

| 方法 | Precision | Recall | FPR | AUC | 提前时间 (s) |
|------|-----------|--------|-----|-----|-------------|
| LiDAR_Stop | 95% | 50% | 3% | 0.70 | 0.15 |
| Shielded_RL | 85% | 65% | 10% | 0.78 | 0.3 |
| CBF-QP | 88% | 70% | 8% | 0.82 | 0.4 |
| PETS | 85% | 75% | 10% | 0.85 | 0.5 |
| Recovery_RL | 82% | 80% | 12% | 0.87 | 0.6 |
| DeepReach | 85% | 85% | 10% | 0.90 | 0.7 |
| **Ours_Full** | **88%** | **92%** | **6%** | **0.94** | **0.8** |

---

## 📁 输出图表

### 预警性能图表

1. `fig1_roc_comparison.png` - ROC 曲线对比 (所有配置/方法)
2. `fig2_precision_recall_bar.png` - 精准率/召回率柱状图
3. `fig3_fpr_comparison.png` - 虚警率对比
4. `fig4_warning_lead_time.png` - 预警提前时间分布 (箱线图)
5. `fig5_runtime_latency.png` - 运行时延对比

### 任务性能图表

6. `fig6_success_rate_heatmap.png` - 成功率热力图 (场景×配置)
7. `fig7_collision_rate.png` - 碰撞率对比
8. `fig8_tradeoff_scatter.png` - 成功率 vs 时延权衡

---

## ⏱️ 时间估算

| 步骤 | 预计时间 | 试验数 |
|------|----------|--------|
| 消融实验 | 1.5 天 | 2,640 次 |
| 对比实验 | 1 天 | 1,980 次 |
| 预警分析 | 0.5 天 | - |
| **总计** | **3 天** | **4,620 次** |

---

## 🔑 关键依赖

| 组件 | 状态 | 位置 |
|------|------|------|
| Region 2 GRU | ✅ | `/mnt/data/region2_training_v2/` |
| Region 3 阈值 (3 模块) | ⚠️ | 需重新学习 (去掉 2 模块) |
| Normal 轨迹 | ✅ | 497 条 |
| Fault 轨迹 | ✅ | 400 条 (8 种×50) |

---

## ⚠️ Region 3 简化影响

**当前 Region 3 阈值**是基于 5 模块 (梯度范数) 学习的，AUC=0.97，TPR=91.2%

**简化为 3 模块后**需重新学习:
1. OOD 马氏距离阈值
2. 决策因子贡献度阈值
3. 激活链路汉明距离阈值

**建议**: 先用现有 5 模块阈值跑消融实验，验证流程后再重新训练 3 模块阈值

---

**下一步**: 
1. 确认 Region 3 简化方案 (3 模块 vs 5 模块)
2. 开始编写消融实验脚本
3. 运行试验并分析预警指标
