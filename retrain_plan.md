# 🔄 GRU + OOD 重训练方案

**问题根因**: 当前模型只用正常轨迹训练，没见过故障数据，导致危险/安全状态区分度差 (Youden J=0.005)

---

## 📊 当前模型问题

### GRU 可达集预测
- **训练数据**: 只有 ACT 正常运行的安全轨迹
- **问题**: 预测的可达集边界在故障前也"看似安全"
- **结果**: R2 FPR=99.6%, Recall=94.6% — 几乎每步都报警

### OOD 马氏距离检测
- **统计量来源**: 只有正常状态的 mu 和 sigma
- **问题**: 故障状态的马氏距离并不显著更大
- **结果**: R3 FPR=61%, 区分度低

---

## 🎯 重训练目标

| 模块 | 当前 Youden J | 目标 Youden J | 预期 TPR | 预期 FPR |
|------|-------------|-------------|---------|---------|
| **GRU** | 0.005 | **>0.4** | 70-80% | 10-20% |
| **OOD** | 0.005 | **>0.3** | 60-70% | 15-25% |

---

## 📁 数据准备

### 1. 收集故障轨迹数据

```bash
# 数据位置
DATA_DIR="/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"

# 故障场景分类
# F1-F4: 感知故障 (视觉噪声、遮挡、状态漂移)
# F5-F8: 动力学故障 (质量变化、惯性、执行器延迟、摩擦力)
# F9: 突发障碍
# F10-F13: 复合故障

# 正常场景
# normal, B1_empty (无障碍)
```

### 2. 危险标签生成

```python
# 使用 GT 逻辑标记每一步
def is_danger(qpos, qvel, threshold=0.03):
    # 关节越界 + 速度超限
    ...
    
# 危险前 N 步 = 正样本 (预警窗口)
# 安全步 = 负样本
```

### 3. 数据集划分

| 用途 | 场景 | 集数 | 步数 |
|------|------|------|------|
| **训练集** | F1-F8 + normal (70%) | ~80 集 | ~32,000 步 |
| **验证集** | F9-F13 (15%) | ~17 集 | ~6,800 步 |
| **测试集** | 保留 10% | ~16 集 | ~6,400 步 |

---

## 🧠 GRU 重训练方案

### 方案 A: 加权损失 (推荐)

```python
# 危险样本权重 ×5
class WeightedLoss(nn.Module):
    def __init__(self, danger_weight=5.0):
        self.danger_weight = danger_weight
    
    def forward(self, pred, target, is_danger):
        base_loss = F.mse_loss(pred, target, reduction='none')
        weights = torch.where(is_danger, self.danger_weight, 1.0)
        return (base_loss * weights).mean()
```

### 方案 B: 对比学习

```python
# 危险前 1 秒 vs 安全轨迹 对比
# 目标：危险前 exceed << 0, 安全时 exceed >= 0

class ContrastiveLoss(nn.Module):
    def forward(self, exceed_safe, exceed_danger, margin=1.0):
        # 安全样本 exceed 应该 >= 0
        safe_loss = F.relu(-exceed_safe).mean()
        # 危险样本 exceed 应该 < -margin
        danger_loss = F.relu(exceed_danger + margin).mean()
        return safe_loss + danger_loss
```

### 方案 C: 二分类 + 回归 多任务

```python
# 头 1: 危险概率 (分类)
# 头 2: exceed 值 (回归)
# 共享 GRU 编码器
```

### 模型架构调整

```python
# 当前：GRU(4 层) → Linear → 16 维支撑函数
# 建议：
class ImprovedGRU(nn.Module):
    def __init__(self):
        # 编码器
        self.gru = nn.GRU(28, 256, 3 层，dropout=0.3)
        
        # 共享表示
        self.shared = nn.Linear(256, 128)
        
        # 头 1: 可达集预测
        self.reach_head = nn.Linear(128, 16)
        
        # 头 2: 危险概率
        self.danger_head = nn.Linear(128, 1)
```

---

## 🔍 OOD 重训练方案

### 方案 A: 故障数据重新统计

```python
# 用故障前 N 步的数据重新计算 mu 和 sigma
# 这样故障状态的马氏距离会显著更大

danger_states = []  # 危险前 1 秒的状态
safe_states = []    # 安全状态

for episode in fault_episodes:
    for step in episode:
        if is_danger(step):
            # 危险前 50 步 = 预警窗口
            for t in range(step-50, step):
                danger_states.append(state[t])
        else:
            safe_states.append(state[t])

# 分别计算统计量
mu_danger = np.mean(danger_states, axis=0)
mu_safe = np.mean(safe_states, axis=0)

# 或使用混合高斯
from sklearn.mixture import GaussianMixture
gmm = GMM(n_components=2)
gmm.fit(all_states)
```

### 方案 B: 集成 OOD 检测器

```python
# 多个检测器投票
class EnsembleOOD:
    def __init__(self):
        self.detectors = [
            MahalanobisOOD(),      # 马氏距离
            ReconstructionOOD(),   # 自编码器重构误差
            DensityOOD(),          # 核密度估计
            TemporalOOD(),         # 时序一致性
        ]
    
    def score(self, state):
        scores = [d.score(state) for d in self.detectors]
        return np.mean(scores)  # 或加权平均
```

### 方案 C: 时序 OOD

```python
# 连续 N 步异常才报警 (减少误报)
class TemporalOOD:
    def __init__(self, window=5, threshold=0.6):
        self.window = window
        self.threshold = threshold
    
    def detect(self, scores):
        # scores: 最近 N 步的 OOD 分数
        anomaly_count = sum(s > thresh for s in scores[-self.window:])
        return anomaly_count >= self.threshold * self.window
```

---

## 📝 训练脚本框架

### GRU 训练脚本

```python
# train_gru_weighted.py
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader

class RTADataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        # 加载轨迹数据
        # 生成 (history, current_state, exceed_label, is_danger)
        ...
    
    def __getitem__(self, idx):
        return history, state, exceed, danger_flag
    
    def __len__(self):
        return n_samples

# 训练循环
model = ImprovedGRU()
criterion = WeightedLoss(danger_weight=5.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(100):
    for history, state, exceed, danger in train_loader:
        pred = model(history)
        loss = criterion(pred, exceed, danger)
        loss.backward()
        optimizer.step()
    
    # 验证
    val_j = compute_youden(val_loader)
    if val_j > best_j:
        save_model(model, f'gru_best_j{val_j:.3f}.pth')
```

### OOD 统计量计算脚本

```python
# compute_ood_stats.py
import numpy as np

# 加载故障数据
danger_states = load_danger_states()  # 危险前 1 秒
safe_states = load_safe_states()

# 方案 1: 混合统计量
all_states = np.concatenate([danger_states, safe_states])
mu = np.mean(all_states, axis=0)
sigma = np.cov(all_states.T)
sigma_inv = np.linalg.pinv(sigma)

# 方案 2: 分别统计
mu_danger = np.mean(danger_states, axis=0)
mu_safe = np.mean(safe_states, axis=0)

# 保存
stats = {
    'mu': mu.tolist(),
    'sigma_inv': sigma_inv.tolist(),
    'mu_danger': mu_danger.tolist(),
    'mu_safe': mu_safe.tolist(),
}
with open('ood_stats_retrained.json', 'w') as f:
    json.dump(stats, f)
```

---

## ⏰ 执行计划

### 第 1 天：数据准备
- [ ] 提取所有故障轨迹的危险/安全标签
- [ ] 划分训练/验证/测试集
- [ ] 生成 GRU 训练数据 (history, exceed)
- [ ] 生成 OOD 统计量计算数据

### 第 2 天：GRU 训练
- [ ] 实现加权损失训练脚本
- [ ] 训练 GRU (约 2-4 小时)
- [ ] 验证集调参 (danger_weight, lr, 层数)
- [ ] 保存最佳模型

### 第 3 天：OOD 重计算 + 评估
- [ ] 用故障数据重新计算 OOD 统计量
- [ ] 集成新 GRU + 新 OOD 到评估脚本
- [ ] 运行完整 113 集评估
- [ ] 对比新旧模型的 P/R/F1/FPR

---

## 📊 评估指标

训练完成后，用以下指标判断是否成功：

| 指标 | 当前 | 目标 | 验收标准 |
|------|------|------|---------|
| **GRU Youden J** | 0.005 | >0.4 | J>0.3 可接受 |
| **OOD Youden J** | 0.005 | >0.3 | J>0.2 可接受 |
| **R2 TPR** | 94.6% | 70-80% | >60% 可接受 |
| **R2 FPR** | 99.6% | 10-20% | <30% 可接受 |
| **Full F1** | 0.54 | >0.65 | >0.6 可接受 |
| **Full FPR** | 99.8% | 10-20% | <30% 可接受 |

---

## 🚀 立即开始

```bash
# 1. SSH 登录
ssh root@js4.blockelite.cn -p 10320

# 2. 创建数据准备脚本
cd /root/act
mkdir -p retrain_data

# 3. 提取危险/安全标签
python3 prepare_retrain_data.py

# 4. 训练 GRU
python3 train_gru_weighted.py

# 5. 重计算 OOD
python3 compute_ood_stats_retrained.py

# 6. 评估新模型
python3 analyze_r2_retrained.py
```

---

*创建时间*: 2026-04-21 13:45  
*目标*: 将 RTA 系统从"几乎每步都报警"优化到"精准预警"
