# Region 3 阈值与场景配置问答

**生成时间**: 2026-03-30 16:40  
**补充问题**: 
1. Region 3 的阈值是学习的吗？
2. 场景配置是依托于仿真环境吗？

---

## Q1: Region 3 的阈值是学习的吗？

### 1.1 简短回答

**部分是学习的，部分是手工设定的。**

| 检测器 | 阈值 | 来源 | 可调性 |
|--------|------|------|--------|
| OOD 检测 | 3.0σ | 统计学 3σ 原则 | 可学习优化 |
| 跳变检测 | 0.5 / 0.3 | 经验值 | 可学习优化 |
| 熵检测 | 1.30 / 0.39 | 信息论 + 经验 | 可学习优化 |
| XAI 掩码 | 3 (汉明距离) | 组合数学 | 可学习优化 |
| 风险融合 | 0.4 | 经验值 | 可学习优化 |

---

### 1.2 各阈值详细说明

#### 1.2.1 OOD 检测阈值 (3.0σ)

**当前实现**:
```python
class VisualOODDetector:
    def __init__(self, feature_dim=512, training_features=None):
        if training_features is not None:
            # 从训练数据学习均值和协方差 ✅ 学习部分
            self.mean = np.mean(training_features, axis=0)
            self.cov = np.cov(training_features.T)
            self.cov += 1e-6 * np.eye(self.feature_dim)
            self.cov_inv = np.linalg.inv(self.cov)
        else:
            self.mean = np.zeros(feature_dim)  # ❌ 先验设定
            self.cov_inv = np.eye(feature_dim) # ❌ 先验设定
        
        self.threshold = 3.0  # ⚠️ 手工设定 (3σ 原则)
```

**阈值来源**: **统计学 3σ 原则**

对于高斯分布：
- ±1σ 覆盖 68.3% 数据
- ±2σ 覆盖 95.4% 数据
- ±3σ 覆盖 99.7% 数据

**设定理由**:
- 马氏距离 > 3σ → 只有 0.3% 概率是正常数据
- 平衡误报率 (FPR) 和漏报率 (FNR)

**如何学习优化**:
```python
def learn_threshold(training_features, ood_features, target_fpr=0.01):
    """
    基于验证集学习最优阈值
    
    参数:
        training_features: 正常特征 (N, 512)
        ood_features: OOD 特征 (M, 512)
        target_fpr: 目标误报率 (默认 1%)
    
    返回:
        optimal_threshold: 最优阈值
    """
    # 计算正常特征的马氏距离
    distances_normal = []
    for f in training_features:
        _, d = detector.detect(f)
        distances_normal.append(d)
    
    # 计算 OOD 特征的马氏距离
    distances_ood = []
    for f in ood_features:
        _, d = detector.detect(f)
        distances_ood.append(d)
    
    # 找到对应目标 FPR 的阈值
    optimal_threshold = np.percentile(distances_normal, (1 - target_fpr) * 100)
    
    # 验证 TPR (真正率)
    tpr = np.mean([d > optimal_threshold for d in distances_ood])
    print(f"阈值={optimal_threshold:.2f}, FPR={target_fpr}, TPR={tpr:.2f}")
    
    return optimal_threshold
```

**推荐学习流程**:
1. 收集正常场景特征 (10,000 帧)
2. 收集 OOD 场景特征 (1,000 帧：光照突变、遮挡、对抗攻击)
3. 设定目标 FPR = 1%
4. 计算最优阈值
5. 验证 TPR > 90%

**预期结果**:
- 最优阈值：2.8 - 3.5 (取决于数据分布)
- FPR: 1%
- TPR: 90-95%

---

#### 1.2.2 跳变检测阈值 (0.5 / 0.3)

**当前实现**:
```python
class VisualJumpDetector:
    def __init__(self, threshold=0.5, derivative_threshold=0.3):
        self.threshold = threshold              # ⚠️ 手工设定
        self.derivative_threshold = derivative_threshold  # ⚠️ 手工设定
        self.prev_features = None
```

**阈值含义**:
- `threshold=0.5`: 特征向量 L2 范数变化 > 0.5 判定为跳变
- `derivative_threshold=0.3`: 变化率 > 0.3/20ms 判定为跳变

**设定理由**:
- 正常帧间变化：~0.1-0.2 (相机运动 + 光照微小变化)
- 故障帧间变化：~0.5-2.0 (光照突变、遮挡、对抗攻击)
- 阈值 0.5 可区分正常与异常

**如何学习优化**:
```python
def learn_jump_threshold(normal_traces, fault_traces):
    """
    基于验证集学习跳变阈值
    
    参数:
        normal_traces: 正常轨迹列表 [(f_t, f_t+1, ...), ...]
        fault_traces: 故障轨迹列表 [(f_t, f_t+1, ...), ...]
    
    返回:
        optimal_threshold: 最优阈值
        optimal_derivative: 最优变化率阈值
    """
    # 计算正常帧间变化分布
    normal_deltas = []
    normal_derivs = []
    for trace in normal_traces:
        for i in range(1, len(trace)):
            delta = np.linalg.norm(trace[i] - trace[i-1])
            deriv = delta / 0.02  # dt=20ms
            normal_deltas.append(delta)
            normal_derivs.append(deriv)
    
    # 计算故障帧间变化分布
    fault_deltas = []
    fault_derivs = []
    for trace in fault_traces:
        for i in range(1, len(trace)):
            delta = np.linalg.norm(trace[i] - trace[i-1])
            deriv = delta / 0.02
            fault_deltas.append(delta)
            fault_derivs.append(deriv)
    
    # 找到最佳分离阈值 (最大化 Youden's J 统计量)
    best_j = 0
    best_threshold = 0.5
    for threshold in np.linspace(0.2, 1.0, 50):
        tpr = np.mean([d > threshold for d in fault_deltas])  # 真正率
        fpr = np.mean([d > threshold for d in normal_deltas]) # 假正率
        j = tpr - fpr  # Youden's J
        if j > best_j:
            best_j = j
            best_threshold = threshold
    
    optimal_derivative = np.percentile(normal_derivs, 99)  # 99 百分位
    
    print(f"最优阈值：{best_threshold:.3f}, Youden's J={best_j:.3f}")
    print(f"最优变化率：{optimal_derivative:.3f}")
    
    return best_threshold, optimal_derivative
```

**预期学习结果**:
- 最优阈值：0.45 - 0.65
- 最优变化率：0.25 - 0.40
- Youden's J: 0.7-0.8 (良好分离度)

---

#### 1.2.3 熵检测阈值 (1.30 / 0.39)

**当前实现**:
```python
class OutputEntropyDetector:
    def __init__(self, threshold=1.30, kl_threshold=0.39):
        self.entropy_threshold = threshold      # ⚠️ 手工设定
        self.kl_threshold = kl_threshold        # ⚠️ 手工设定
        self.nominal_policy = None              # ✅ 可学习
```

**阈值来源**:
- `entropy=1.30`: 高斯分布微分熵阈值
  - 正常策略：H ≈ 0.8-1.2
  - 不确定策略：H > 1.3
- `kl=0.39`: KL 散度阈值 (ln(1.5) ≈ 0.405)
  - 正常：KL < 0.3
  - 异常：KL > 0.4

**如何学习**:
```python
def learn_entropy_threshold(normal_policies, uncertain_policies):
    """学习熵和 KL 散度阈值"""
    # 计算正常策略的熵分布
    normal_entropies = [compute_entropy(p) for p in normal_policies]
    normal_kls = [compute_kl(p, nominal_policy) for p in normal_policies]
    
    # 计算不确定策略的熵分布
    uncertain_entropies = [compute_entropy(p) for p in uncertain_policies]
    uncertain_kls = [compute_kl(p, nominal_policy) for p in uncertain_policies]
    
    # 设定阈值为正常分布的 99 百分位
    entropy_threshold = np.percentile(normal_entropies, 99)
    kl_threshold = np.percentile(normal_kls, 99)
    
    # 验证 TPR
    entropy_tpr = np.mean([e > entropy_threshold for e in uncertain_entropies])
    kl_tpr = np.mean([k > kl_threshold for k in uncertain_kls])
    
    print(f"熵阈值：{entropy_threshold:.3f}, TPR={entropy_tpr:.2f}")
    print(f"KL 阈值：{kl_threshold:.3f}, TPR={kl_tpr:.2f}")
    
    return entropy_threshold, kl_threshold
```

**预期学习结果**:
- 熵阈值：1.20 - 1.45
- KL 阈值：0.35 - 0.45
- TPR: 85-95%

---

#### 1.2.4 XAI 掩码阈值 (3 - 汉明距离)

**当前实现**:
```python
class FeatureMaskDetector:
    def __init__(self, mask_library=None, threshold=3):
        self.threshold = threshold  # ⚠️ 手工设定
        
        if mask_library is None:
            # ✅ 可从训练数据学习
            self.mask_library = self.learn_masks(training_data)
        else:
            self.mask_library = mask_library
```

**阈值来源**: **组合数学**

Top 6 神经元的汉明距离范围：0-6
- 距离 0: 完全匹配
- 距离 1-2: 轻微差异 (正常变异)
- 距离 3-4: 中度差异 (可疑)
- 距离 5-6: 严重差异 (异常)

**设定理由**:
- 距离 ≥ 3 → 50% 神经元不同 → 判定为异常
- 平衡灵敏度和特异性

**如何学习掩码库**:
```python
def learn_mask_library(scenario_data, n_masks=8, top_k=6):
    """
    从训练数据学习掩码库
    
    参数:
        scenario_data: dict {scenario_name: [features]}
        n_masks: 掩码数量 (默认 8)
        top_k: 每个掩码的 Top K 神经元 (默认 6)
    
    返回:
        mask_library: dict {mask_id: [neuron_indices]}
    """
    mask_library = {}
    
    for scenario_id, features in scenario_data.items():
        # 统计每个神经元的平均激活强度
        mean_activation = np.mean(features, axis=0)
        
        # 选择 Top K 激活神经元
        top_neurons = np.argsort(mean_activation)[-top_k:]
        
        mask_library[f'Mask_{scenario_id}'] = sorted(top_neurons.tolist())
    
    return mask_library
```

**如何学习阈值**:
```python
def learn_hamming_threshold(mask_library, normal_features, abnormal_features):
    """学习汉明距离阈值"""
    min_distances_normal = []
    min_distances_abnormal = []
    
    # 计算正常特征的最小汉明距离分布
    for f in normal_features:
        top_k = np.argsort(f)[-6:]
        min_dist = min(hamming_distance(top_k, mask) 
                       for mask in mask_library.values())
        min_distances_normal.append(min_dist)
    
    # 计算异常特征的最小汉明距离分布
    for f in abnormal_features:
        top_k = np.argsort(f)[-6:]
        min_dist = min(hamming_distance(top_k, mask) 
                       for mask in mask_library.values())
        min_distances_abnormal.append(min_dist)
    
    # 找到最佳分离阈值
    best_threshold = 3
    best_j = 0
    for t in range(1, 7):
        tpr = np.mean([d >= t for d in min_distances_abnormal])
        fpr = np.mean([d >= t for d in min_distances_normal])
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_threshold = t
    
    print(f"最优汉明距离阈值：{best_threshold}, Youden's J={best_j:.3f}")
    return best_threshold
```

**预期学习结果**:
- 最优阈值：2-4 (取决于场景差异)
- Youden's J: 0.6-0.8

---

#### 1.2.5 风险融合阈值 (0.4)

**当前实现**:
```python
class Region3VisualDetector:
    def __init__(self):
        self.weights = {
            'ood': 0.30,      # ⚠️ 手工设定
            'jump': 0.25,     # ⚠️ 手工设定
            'entropy': 0.20,  # ⚠️ 手工设定
            'xai': 0.25       # ⚠️ 手工设定
        }
        self.threshold = 0.4  # ⚠️ 手工设定
```

**阈值含义**:
- 风险分数 = 0.3×OOD + 0.25× 跳变 + 0.2× 熵 + 0.25×XAI
- 风险 > 0.4 → 触发 RTA
- 需要至少 2 个检测器触发才能达到 0.4

**如何学习**:
```python
def learn_risk_weights_and_threshold(labeled_data):
    """
    学习风险权重和融合阈值
    
    参数:
        labeled_data: [(ood, jump, entropy, xai, is_abnormal), ...]
    
    返回:
        weights: dict {detector: weight}
        threshold: float
    """
    from sklearn.linear_model import LogisticRegression
    
    # 准备数据
    X = np.array([[ood, jump, entropy, xai] 
                  for ood, jump, entropy, xai, _ in labeled_data])
    y = np.array([is_abnormal for _, _, _, _, is_abnormal in labeled_data])
    
    # 逻辑回归学习权重
    model = LogisticRegression()
    model.fit(X, y)
    
    # 提取权重 (归一化)
    weights_raw = np.abs(model.coef_[0])
    weights_norm = weights_raw / np.sum(weights_raw)
    
    weights = {
        'ood': weights_norm[0],
        'jump': weights_norm[1],
        'entropy': weights_norm[2],
        'xai': weights_norm[3]
    }
    
    # 学习最优阈值 (最大化 F1 分数)
    from sklearn.metrics import f1_score
    best_f1 = 0
    best_threshold = 0.4
    for t in np.linspace(0.2, 0.8, 50):
        y_pred = (model.predict_proba(X)[:, 1] > t).astype(int)
        f1 = f1_score(y, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f"学习权重：{weights}")
    print(f"最优阈值：{best_threshold:.3f}, F1={best_f1:.3f}")
    
    return weights, best_threshold
```

**预期学习结果**:
- OOD 权重：0.30-0.40 (最重要)
- 跳变权重：0.20-0.30
- 熵权重：0.15-0.25
- XAI 权重：0.20-0.30
- 融合阈值：0.35-0.50
- F1 分数：0.85-0.92

---

### 1.3 完整学习流程推荐

```python
def learn_all_region3_parameters():
    """完整学习 Region 3 所有参数"""
    
    # 1. 收集训练数据
    normal_data = collect_normal_scenes(n=10000)
    ood_data = collect_ood_scenes(n=1000)  # S1-S3 场景
    jump_data = collect_jump_scenes(n=500)
    uncertain_data = collect_uncertain_scenes(n=500)
    
    # 2. 学习 OOD 阈值
    ood_threshold = learn_ood_threshold(normal_data, ood_data, target_fpr=0.01)
    
    # 3. 学习跳变阈值
    jump_thresh, deriv_thresh = learn_jump_threshold(normal_data, jump_data)
    
    # 4. 学习熵阈值
    entropy_thresh, kl_thresh = learn_entropy_threshold(normal_data, uncertain_data)
    
    # 5. 学习掩码库和汉明距离阈值
    mask_library = learn_mask_library(scenario_data)
    hamming_thresh = learn_hamming_threshold(mask_library, normal_data, ood_data)
    
    # 6. 学习风险权重和融合阈值
    labeled_data = prepare_labeled_data(normal_data, ood_data)
    weights, risk_thresh = learn_risk_weights_and_threshold(labeled_data)
    
    # 7. 保存参数
    params = {
        'ood_threshold': ood_threshold,
        'jump_threshold': jump_thresh,
        'jump_derivative': deriv_thresh,
        'entropy_threshold': entropy_thresh,
        'kl_threshold': kl_thresh,
        'hamming_threshold': hamming_thresh,
        'mask_library': mask_library,
        'risk_weights': weights,
        'risk_threshold': risk_thresh
    }
    
    with open('region3_parameters.pkl', 'wb') as f:
        pickle.dump(params, f)
    
    print("Region 3 参数学习完成!")
    return params
```

**预计训练时间**:
- 数据收集：2-4 小时 (仿真运行)
- 参数学习：10-30 分钟
- 验证测试：30 分钟
- **总计**: 3-5 小时

---

### 1.4 当前状态总结

| 参数 | 当前值 | 来源 | 是否学习 |
|------|--------|------|----------|
| OOD 阈值 | 3.0σ | 统计学 3σ | ❌ 手工，✅ 可学习 |
| 跳变阈值 | 0.5 | 经验值 | ❌ 手工，✅ 可学习 |
| 跳变变化率 | 0.3 | 经验值 | ❌ 手工，✅ 可学习 |
| 熵阈值 | 1.30 | 信息论 | ❌ 手工，✅ 可学习 |
| KL 阈值 | 0.39 | ln(1.5) | ❌ 手工，✅ 可学习 |
| 汉明距离 | 3 | 组合数学 | ❌ 手工，✅ 可学习 |
| 掩码库 | 8 模式 | 经验设计 | ❌ 手工，✅ 可学习 |
| 风险权重 | [0.3,0.25,0.2,0.25] | 经验 | ❌ 手工，✅ 可学习 |
| 风险融合 | 0.4 | 经验值 | ❌ 手工，✅ 可学习 |

**结论**: 当前实现使用**手工设定阈值** (基于统计学原理和经验)，但**所有参数都可以通过验证集学习优化**。

---

## Q2: 场景配置是依托于仿真环境吗？

### 2.1 简短回答

**是的，场景配置完全依托于仿真环境。**

---

### 2.2 场景配置详解

#### 2.2.1 场景配置位置

```python
# fetch_env_extended.py
class FetchMobileEnv:
    def _build_scenario_library(self):
        return {
            's1_lighting_drop': ScenarioConfig(
                name='严重光照突变',
                fault_type=FaultType.S1_LIGHTING_DROP,
                injection_time=5.0,    # ✅ 仿真时间步
                duration=10.0,         # ✅ 仿真时间步
                intensity=0.9,         # ✅ 仿真参数
                params={
                    'noise_scale': 2.5,    # ✅ 特征噪声强度
                    'flicker_freq': 5.0,   # ✅ 仿真闪烁频率
                }
            ),
            # ... 其他 7 个场景
        }
```

---

#### 2.2.2 仿真环境依赖项

| 配置项 | 依赖 | 说明 |
|--------|------|------|
| `injection_time` | 仿真步长 (dt=0.02s) | 5.0s = 250 步 |
| `duration` | 仿真步长 | 10.0s = 500 步 |
| `noise_scale` | 特征维度 (512) | 高斯噪声标准差 |
| `flicker_freq` | 仿真帧率 (30Hz) | 5Hz 闪烁 = 每 6 帧变化 |
| `occlusion_ratio` | 相机分辨率 | 25% = 128×128 像素 |
| `payload_mass` | 机器人质量 (25kg) | 2kg = 8% 总质量 |
| `corridor_width` | 机器人尺寸 (0.6m) | 1.2m = 2× 机器人宽度 |

---

#### 2.2.3 仿真环境参数传递

```
┌─────────────────────────────────────────────────────────────┐
│ 场景配置文件 (fetch_params.yaml)                            │
│ base:                                                        │
│   mass: 25.0 kg          ← 影响 S4 负载突变                  │
│   v_max: 1.0 m/s         ← 影响所有场景                      │
│   length: 0.60 m         ← 影响 S7 窄通道宽度                │
│                                                              │
│ constraints:                                                 │
│   d_min: 0.15 m          ← 影响碰撞检测                      │
│   z_ee_min: 0.05 m       ← 影响末端约束                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 仿真环境 (fetch_env_extended.py)                            │
│ class FetchMobileEnv:                                        │
│   def __init__(self, config_path):                          │
│     cfg = yaml.safe_load(open(config_path))                 │
│     self.mass = cfg['base']['mass']  ← 读取配置             │
│     self.v_max = cfg['base']['v_max']                       │
│     ...                                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 场景注入逻辑                                                 │
│ def _inject_fault(self, action, fault_info):                │
│   if fault_type == 's4_payload_shift':                      │
│     self.state['payload_mass'] = 2.0  ← 使用配置参数        │
│     # 动力学计算使用 self.mass (25kg)                       │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.3 仿真环境 vs 真实世界

#### 2.3.1 参数映射关系

| 仿真参数 | 真实对应 | 映射方式 |
|----------|----------|----------|
| `noise_scale=2.5` | 光照强度 (lux) | 仿真：特征噪声；真实：照度计测量 |
| `occlusion_ratio=0.25` | 遮挡面积 (cm²) | 仿真：像素比例；真实：图像分析 |
| `payload_mass=2.0kg` | 实际负载 (kg) | 直接对应 (仿真物理引擎) |
| `friction_multiplier=3.0` | 摩擦系数 | 仿真：扭矩缩放；真实：力传感器 |
| `corridor_width=1.2m` | 实际宽度 (m) | 直接对应 (仿真碰撞体) |

---

#### 2.3.2 Sim2Real 差距

**仿真优势**:
- ✅ 精确控制故障注入时间和强度
- ✅ 可重复试验 (随机种子固定)
- ✅ 完全可观测 (全部状态)
- ✅ 安全 (无设备损坏风险)

**仿真局限**:
- ❌ 简化物理 (摩擦、柔性未建模)
- ❌ 简化传感器噪声 (高斯近似)
- ❌ 简化行人行为 (随机游走 AI)
- ❌ 简化光照模型 (参数化)

**Sim2Real 迁移建议**:
1. **系统辨识**: 在真实机器人上测量实际参数
   - 实际质量、惯量、摩擦系数
   - 传感器噪声特性
   - 执行器延迟

2. **域随机化**: 在仿真中随机化参数
   ```python
   # 训练时随机化
   mass = np.random.uniform(23, 27)  # 25±2kg
   friction = np.random.uniform(0.8, 1.2) * nominal
   lighting = np.random.uniform(0.5, 1.5)
   ```

3. **渐进迁移**:
   - 仿真训练 → 仿真微调 → 真实测试 → 真实微调

---

### 2.4 场景配置的可移植性

#### 2.4.1 迁移到 Isaac Lab

```python
# Isaac Lab 场景配置
from isaac_lab.scene import Scene
from isaac_lab.faults import FaultInjector

scene = Scene()

# S1: 光照突变
scene.configure_lighting(
    base_intensity=1.0,
    fault=FaultInjector(
        injection_time=5.0,
        duration=10.0,
        intensity_drop=0.9,
        flicker_freq=5.0
    )
)

# S4: 负载突变
scene.configure_robot(
    payload_mass=0.0,  # 初始无负载
    fault=FaultInjector(
        injection_time=4.0,
        payload_mass=2.0
    )
)

# S7: 窄通道
scene.add_walls(
    position=[0, 0],
    width=15,
    gap=1.2  # 通道宽度
)
```

---

#### 2.4.2 迁移到真实机器人

```yaml
# 真实场景配置文件 (real_world_scenes.yaml)
s1_lighting_drop:
  injection_time: 5.0s
  duration: 10.0s
  # 真实实现：智能照明系统
  lighting_control:
    type: "philips_hue"
    base_brightness: 100%
    fault_brightness: 10%
    flicker_pattern: "5Hz_sine"

s4_payload_shift:
  injection_time: 4.0s
  duration: 15.0s
  # 真实实现：可切换负载
  payload_mechanism:
    type: "electromagnetic_gripper"
    normal_payload: 0kg
    fault_payload: 2kg

s7_narrow_corridor:
  # 真实实现：物理墙壁
  corridor:
    type: "foam_panels"
    width: 1.2m
    length: 15m
```

---

### 2.5 场景配置验证

#### 2.5.1 仿真内验证

```python
def validate_scenario_config(scenario_name, n_runs=100):
    """验证场景配置在仿真中是否正确"""
    
    env = FetchMobileEnv()
    successes = []
    
    for run in range(n_runs):
        obs = env.reset(scenario=scenario_name, seed=run)
        fault_triggered = False
        fault_active = False
        
        for step in range(1500):
            t = step * 0.02
            action = get_random_action()
            obs, _, done, info = env.step(action)
            
            # 检查故障是否在预期时间注入
            if info.get('fault_active') and not fault_active:
                fault_active = True
                if scenario_name == 's1_lighting_drop':
                    expected_time = 5.0
                    actual_time = t
                    assert abs(actual_time - expected_time) < 0.1
                    fault_triggered = True
        
        successes.append(fault_triggered)
    
    success_rate = np.mean(successes)
    print(f"{scenario_name}: 配置验证通过率 = {success_rate*100:.1f}%")
    return success_rate
```

**预期结果**: 所有场景 >95% 通过率

---

#### 2.5.2 真实世界验证清单

| 场景 | 验证项 | 方法 | 通过标准 |
|------|--------|------|----------|
| S1 光照 | 光照强度 | 照度计测量 | 90%±5% 下降 |
| S2 遮挡 | 遮挡面积 | 图像分析 | 25%±3% 像素 |
| S3 对抗 | 攻击效果 | 分类准确率 | <10% 正确率 |
| S4 负载 | 实际质量 | 称重传感器 | 2kg±0.1kg |
| S5 摩擦 | 关节扭矩 | 力矩传感器 | 300%±20% |
| S6 人群 | 行人数量 | 视觉追踪 | 5 人±0 |
| S7 窄道 | 通道宽度 | 激光测距 | 1.2m±0.05m |
| S8 复合 | 多故障并发 | 综合检查 | 全部通过 |

---

## 总结

| 问题 | 答案 |
|------|------|
| **Q1: Region 3 阈值是学习的吗？** | 当前是**手工设定** (基于统计学和经验)，但**所有参数都可以通过验证集学习优化**。推荐学习流程需 3-5 小时。 |
| **Q2: 场景配置依托于仿真环境吗？** | **是的，完全依托**。场景参数 (时间、强度、几何) 都基于仿真环境的物理引擎、传感器模型和渲染系统。迁移到真实世界需要系统辨识和域随机化。 |

---

**文档生成完毕**

所有问题已详细解答。🦞
