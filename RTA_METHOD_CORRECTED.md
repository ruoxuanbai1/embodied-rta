# RTA 三层融合架构 - 正确方法描述

**更新日期**: 2026-04-20  
**核心修正**: 所有 Region、所有模块均为 **OR 逻辑**（纯布尔判断，无 risk score 融合）

---

## 📐 总体架构

```
输入：qpos (14 维), qvel (14 维), image (RGB), action (14 维)
         ↓
    ┌────┴────┬────────────┐
    ↓         ↓            ↓
  Region 1  Region 2    Region 3
  (物理)    (可达集)     (感知)
    ↓         ↓            ↓
 safe_r1   safe_r2      safe_r3
    └────┬────┴────────────┘
         ↓
    alarm = ¬safe_r1 ∨ ¬safe_r2 ∨ ¬safe_r3
```

**核心原则**: **纯 OR 逻辑** —— 任何一层检测到危险 → 立即报警

---

## 🔷 Region 1: 物理硬约束检测

### 检测对象
- 关节限位 (Joint Limits)
- 自碰撞 (Self-Collision)
- 速度/加速度限制

### 检测逻辑 (AND 逻辑)

```python
def check_region1(qpos, qvel):
    violations = []
    
    # 1. 关节限位检查
    for i, (q_min, q_max) in enumerate(JOINT_LIMITS):
        margin = min(qpos[i] - q_min, q_max - qpos[i])
        if margin < -joint_range * 0.03:
            violations.append("joint_limit_critical")
        elif margin < joint_range * 0.1:
            violations.append("joint_limit_warning")
    
    # 2. 自碰撞检查
    if end_effector_distance < SELF_COLLISION_THRESHOLD:
        violations.append("self_collision")
    
    # 3. 速度检查
    if max(qvel / qvel_limit) > 1.0:
        violations.append("velocity_violation")
    
    # safe_r1: 所有条件都满足才安全
    safe_r1 = (len(violations) == 0)
    
    return safe_r1, violations
```

### 关键参数 (仿真环境真实限位)

| 关节 | 限位 (rad) | 来源 |
|------|-----------|------|
| shoulder | [-1.85, 1.26] | 仿真环境 |
| elbow | [-1.76, 1.61] | 仿真环境 |
| wrist | [-3.14, 3.14] | 仿真环境 |
| gripper | [0.021, 0.057] | 仿真环境 |

**修复记录 (2026-04-16)**: 从硬编码 `[-1, 1]` 改为真实限位，margin 从 -0.16→+0.72

### 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| TPR | 96.2% | 正确检测危险 |
| FPR | 2.7% | 误报率低 |
| F1 | 0.953 | 综合性能好 |

---

## 🔷 Region 2: 可达集预测

### 检测对象
- 未来 1 秒可达集是否与危险边界相交

### 检测逻辑

```python
def check_region2(qpos, qvel, history_buffer):
    if not gru_model_loaded or len(history_buffer) < 10:
        return True, 0.0  # 模型未加载时默认安全
    
    # 1. GRU 预测支撑函数值 (16 维)
    support_values = GRU(history_buffer)  # (16,)
    
    # 2. 计算当前状态在各方向上的投影
    current_state = concat(qpos, qvel)  # (28,)
    current_projection = direction_matrix @ current_state  # (16,)
    
    # 3. 检查是否超出支撑边界
    exceed = current_projection - support_values
    safe_support = all(exceed <= 1e-6)
    
    # 4. 预测轨迹碰撞检测
    predicted_states = predict_trajectory(gru_model, history_buffer, num_steps=25)
    collision_warning = any(dist < 0.1 for dist in predicted_distances)
    
    # safe_r2: 支撑边界内 且 无碰撞预警
    safe_r2 = safe_support and not collision_warning
    
    return safe_r2, collision_warning
```

### GRU 模型架构

| 组件 | 规格 |
|------|------|
| 输入 | 轨迹历史 (10 步 × 28 维状态) |
| GRU | 2 层，hidden=192 |
| 输出 | 16 维支撑函数值 |
| 方向矩阵 | (16, 28) — 16 个投影方向 |

### 支撑函数原理

对于凸集 $\mathcal{K} \subset \mathbb{R}^n$ 和方向 $d \in \mathbb{R}^n$：

$$h_{\mathcal{K}}(d) = \sup_{x \in \mathcal{K}} d^T x$$

**16 方向投影**: 覆盖关键状态变量的 min/max 边界

### 当前状态

⚠️ **GRU 模型未加载** — 消融实验数据生成时未正确加载模型，R2 alarm 全为 false

**待修复**: 在数据生成脚本中加载 GRU 模型并重新生成数据

---

## 🔷 Region 3: 感知异常检测

### 检测对象
- 感知输入是否分布外 (OOD)
- 决策逻辑是否合理
- 神经元激活模式是否异常

### 检测逻辑 (OR 逻辑)

```python
def check_region3(qpos, action, gradient, encoder_act):
    scores = {}
    
    # M1: 梯度贡献度 (逻辑合理性)
    S_logic = compute_logic_score(gradient, F_legal)
    m1_alarm = S_logic < 0.55  # 阈值从 normal 场景学习
    
    # M2: 激活链路 (神经元激活模式)
    D_ham = compute_hamming_distance(encoder_act, M_ref)
    m2_alarm = D_ham > 0.35  # 阈值从 normal 场景 95% 分位数学习
    
    # M3: OOD 检测 (输入分布)
    D_ood = compute_mahalanobis(qpos, mu, Sigma_inv)
    m3_alarm = D_ood > 0.5  # 阈值从 normal 场景学习
    
    # safe_r3: 三个模块都不报警才安全 (OR 逻辑)
    safe_r3 = not (m1_alarm or m2_alarm or m3_alarm)
    
    return safe_r3, scores
```

### 三模块详解

#### M1: 梯度贡献度 (逻辑合理性)

**原理**: 计算动作对状态特征的梯度，分析合法特征集的贡献度

$$\phi_i = qpos_i \cdot \frac{\partial a}{\partial qpos_i}$$

$$S_{logic} = \frac{\sum_{i \in F_{legal}} |\phi_i|}{\sum_{j} |\phi_j|}$$

**关键特征集 F_legal (top-5)**: 从 ACT 仿真数据挖掘，每个动作维度选 5 个最重要状态特征

**阈值**: `S_logic_min = 0.55` (从 normal 场景 5% 分位数学习)

---

#### M2: 激活链路 (神经元激活模式)

**原理**: 比较当前神经元激活模式与安全掩码库的汉明距离

**Hook 挂载点**:
- Encoder 4 层 FFN 输出 (`layer0.linear2` ~ `layer3.linear2`)
- Decoder 1 层 FFN 输出 (`layer0.linear2`)

**离线学习**:
1. 收集 normal 场景激活数据
2. 二值化：`M = (activation > 0)`
3. K-Means 聚类 → 标准掩码库 `M_ref`

**在线检测**:
```python
D_ham = popcount(M_curr XOR M_ref) / len(M_ref)
```

**阈值**: `D_ham_max = 0.35` (从 normal 场景 95% 分位数学习)

---

#### M3: OOD 检测 (输入分布)

**原理**: 马氏距离检测输入状态是否偏离训练分布

$$D_{ood} = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

**训练数据统计量**:
- $\mu$: normal 场景 qpos 均值 (14 维)
- $\Sigma$: normal 场景 qpos 协方差 (14×14)

**阈值**: `D_ood_max = 0.5` (从 normal 场景学习)

---

### R3 内部消融配置

| 配置 | M1 | M2 | M3 | 说明 |
|------|----|----|----|------|
| R3_Full | ✅ | ✅ | ✅ | 三模块融合 |
| R3_no_M1 | ❌ | ✅ | ✅ | 无梯度贡献度 |
| R3_no_M2 | ✅ | ❌ | ✅ | 无激活链路 |
| R3_no_M3 | ✅ | ✅ | ❌ | 无 OOD 检测 |
| R3_M1_only | ✅ | ❌ | ❌ | 仅梯度贡献度 |
| R3_M2_only | ❌ | ✅ | ❌ | 仅激活链路 |
| R3_M3_only | ❌ | ❌ | ✅ | 仅 OOD 检测 |

---

## 🔷 层间融合逻辑

### 最终报警 (OR 逻辑)

```python
alarm = (not safe_r1) or (not safe_r2) or (not safe_r3)
```

**解释**: 任何一层检测到危险 → 立即报警

### 层间消融配置

| 配置 | R1 | R2 | R3 | 说明 |
|------|----|----|----|------|
| A1_R1_only | ✅ | ❌ | ❌ | 仅物理约束 |
| A2_R2_only | ❌ | ✅ | ❌ | 仅可达集 |
| A3_R3_only | ❌ | ❌ | ✅ | 仅感知检测 |
| A4_R1+R2 | ✅ | ✅ | ❌ | 无 R3 |
| A5_R1+R3 | ✅ | ❌ | ✅ | 无 R2 |
| A6_R2+R3 | ❌ | ✅ | ✅ | 无 R1 |
| A7_Full | ✅ | ✅ | ✅ | 三层融合 |

---

## 📊 阈值学习流程

### 数据来源

- **Normal 场景**: ACT 安全运行轨迹 (113 集 × 400 步 = 45,200 步)
- **Fault 场景**: 注入故障后的轨迹 (8 种故障类型)

### 学习方法

从 **Normal 场景** 学习各模块阈值的分位数：

| 模块 | 阈值 | 学习来源 | 值 |
|------|------|---------|-----|
| M1 (S_logic) | 5% 分位数 | normal 场景 S_logic 分布 | 0.55 |
| M2 (D_ham) | 95% 分位数 | normal 场景 D_ham 分布 | 0.35 |
| M3 (D_ood) | 学习值 | normal 场景 D_ood 分布 | 0.5 |

### 优化目标

- **检测率 (TPR)**: > 90%
- **虚警率 (FPR)**: < 5%
- **F1-Score**: 最大化

---

## 📁 关键文件位置

### 代码
```
/root/act/
├── rta_v5_runner.py              # RTA 运行器 (三层检测)
├── analyze_ablation_v2.py        # 消融分析脚本
├── region3_detector.py           # R3 检测器
└── gru_predictor.py              # GRU 可达集预测器
```

### 模型
```
/root/act/outputs/
├── region2_gru/
│   └── gru_reachability_best.pth    # GRU 模型
└── region3_detectors/
    ├── thresholds_learned.json      # 学习到的阈值
    ├── F_legal_profiles.json        # 关键特征集
    ├── activation_masks.json        # 激活掩码库
    └── ood_stats.json               # OOD 统计量
```

### 数据
```
/mnt/data/ablation_experiments/
├── ablation_combined_v2_165eps_FULL/  # 原始数据 (113 集)
├── analysis_detailed_v14/             # V14 分析结果
└── analysis_corrected/                # 修正阈值分析 (进行中)
```

---

## ⚠️ 已知问题与待修复

### 1. R2 GRU 模型未加载 ❌

**问题**: 消融实验数据生成时，GRU 模型未正确加载，导致 R2 alarm 全为 false

**影响**: R2 贡献度无法评估

**修复计划**:
1. 在数据生成脚本中加载 GRU 模型
2. 重新生成消融实验数据
3. 重新分析 R2 性能

---

### 2. R3 模块贡献度分析不完整 ⚠️

**问题**: V14 分析中，R3 的 M2 (Hamming) 和 M3 (OOD) 阈值设置错误，导致全 0

**错误阈值**:
- D_ham_max: 0.5 (应为 0.35)
- D_ood_max: 3.0 (应为 0.5)

**正确阈值**:
- S_logic_min: 0.55 ✅
- D_ham_max: 0.35 ✅
- D_ood_max: 0.5 ✅

**修复状态**: 正在用正确阈值重新计算 R3 贡献度

---

### 3. Ground Truth 定义需明确 ⚠️

**当前 GT**: `actual_danger` (物理危险状态)

**问题**: normal 场景中 `actual_danger=True` 比例高 (机器人一直在限位附近运行)

**影响**: TPR/FPR 计算可能偏高

**建议**: 结合 `is_fault_window` 和 `actual_danger` 定义更精确的 GT

---

## 📈 预期性能指标

| 配置 | TPR% | FPR% | F1 | 说明 |
|------|------|------|-----|------|
| A1_R1_only | ~96 | ~3 | 0.95 | 物理约束 alone |
| A3_R3_only | ~80 | ~20 | 0.80 | 感知检测 alone (修正后) |
| A7_Full | ~98 | ~2 | 0.97 | 三层融合 (预期) |

**R3 内部消融预期**:
- M1 (梯度): 主要贡献者 (TPR ~80%)
- M2 (激活): 补充检测 (TPR ~50%)
- M3 (OOD): 补充检测 (TPR ~40%)

---

## 🔑 关键教训

1. **OR 逻辑是核心** — 任何一层报警 → 最终报警，不是 risk score 融合
2. **阈值必须学习** — 不能硬编码 (从 normal 场景分位数学习)
3. **R1 关节限位要准确** — 用仿真环境真实限位，不是硬编码 [-1, 1]
4. **GRU 加载要验证** — 数据生成时必须正确加载模型
5. **先验证小样本** — 不要直接跑全量数据，先用小样本验证逻辑

---

*最后更新*: 2026-04-20  
*作者*: AI Assistant  
*状态*: 方法文档已修正，R3 阈值修正分析进行中
