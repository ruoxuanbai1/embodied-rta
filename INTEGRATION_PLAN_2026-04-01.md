# Embodied-RTA 集成与训练推进计划

**更新时间**: 2026-04-01 06:05  
**目标**: 完成 OpenVLA 集成 + Region 2/3 训练

---

## 📊 当前状态

| 模块 | 状态 | 进度 | 备注 |
|------|------|------|------|
| **Region 1** | ✅ 完成 | 100% | 物理硬包线已验证 |
| **Region 2 - 数据生成** | 🔄 进行中 | ~7% (500/7000) | 后台运行中，预计还需 2-3 小时 |
| **Region 2 - GRU 训练** | ⏳ 待开始 | 0% | 等待数据生成完成 |
| **Region 3 - 代码** | ✅ 完成 | 100% | `multi_layer_activation.py` 就绪 |
| **Region 3 - OpenVLA 集成** | ⏸️ 阻塞 | 0% | 需要下载 OpenVLA-7B (14GB) |
| **Region 3 - 参考数据收集** | ⏳ 待开始 | 0% | 需要 OpenVLA + 正常轨迹 |

---

## 🚀 推进步骤

### Step 1: 下载 OpenVLA-7B 模型 (阻塞性任务)

**命令**:
```bash
cd /home/admin/.openclaw/workspace/Embodied-RTA
python3 scripts/download_openvla.py
```

**预计时间**: 30-60 分钟 (14GB)  
**依赖**: HuggingFace token (可能需要配置 `HF_TOKEN` 环境变量)

**检查点**:
```bash
ls -lh /data/models/openvla-7b/
# 应包含：config.json, pytorch_model.bin, tokenizer.json 等
```

---

### Step 2: 等待 Region 2 数据生成完成

**当前进度**: 500/7000 轨迹 (~7%)  
**预计完成**: 还需 2-3 小时

**监控命令**:
```bash
tail -f /home/admin/.openclaw/workspace/Embodied-RTA/reachability/data_generation.log
```

**输出文件**:
- `reachability/reachability_dataset_v2.h5` (约 2-3GB)

---

### Step 3: 训练 Region 2 GRU 模型

**命令**:
```bash
cd /home/admin/.openclaw/workspace/Embodied-RTA
python3 reachability/train_gru_reachability.py
```

**预计时间**: 2-3 小时  
**输出**:
- `reachability/gru_reachability.pth` (最佳模型)
- `reachability/training_log.csv` (训练日志)

**训练配置**:
- 输入：(70000, 10, 19) - 10 帧状态序列
- 输出：(70000, 32) - 16 变量的 min+max 支撑函数
- 架构：GRU(128)×2 → FC(64) → ReLU → FC(32)
- 损失：非对称 Huber (欠预测×2 权重)
- Epochs: 200 (early stopping @ 150)

---

### Step 4: 集成 OpenVLA 到 Region 3

**文件**: `xai/multi_layer_activation.py` 已包含 OpenVLA 层钩子注册逻辑

**需要修改**:
1. 在 `tests/run_openvla_rta.py` 中导入 Region3Detector
2. 注册 OpenVLA 层钩子 (Vision→LLM→Action)
3. 收集正常激活参考数据

**集成代码示例**:
```python
from xai.region3_activation_link import Region3Detector

# 初始化检测器
detector = Region3Detector(model=vla.model)

# 注册钩子 (OpenVLA 结构)
layer_names = [
    'vision_model.encoder.layers.0',
    'vision_model.encoder.layers.6',
    'llm.model.layers.3',
    'llm.model.layers.12',
    'action_head'
]
detector.register_hooks(layer_names)

# 收集参考数据 (正常场景 100 次推理)
detector.collect_reference(normal_dataloader, n_samples=100)
```

---

### Step 5: 收集 Region 3 参考数据

**命令**:
```bash
python3 xai/collect_reference_activations.py
```

**预计时间**: 30 分钟  
**输出**:
- `xai/reference_activations.npy` (正常激活统计)
- `xai/layer_correlations.pkl` (层间相关性矩阵)

---

### Step 6: 完整试验 (可选)

**命令**:
```bash
python3 tests/run_openvla_rta.py
```

**试验规模**:
- 4 基础场景 × 13 故障类型 × 15 RTA 配置 × 30 种子 = **23,400 次**

**预计时间**: 8-12 小时

---

## ⏱️ 时间线预估

| 时间 (从 06:05 开始) | 事件 | 累计耗时 |
|---------------------|------|----------|
| **06:05** | 开始下载 OpenVLA | 0h |
| **07:05** | OpenVLA 下载完成 | +1h |
| **08:30** | Region 2 数据生成完成 | +2.5h |
| **09:00** | 开始 Region 2 训练 | +3h |
| **11:30** | Region 2 训练完成 | +5.5h |
| **12:00** | 开始 Region 3 集成 | +6h |
| **12:30** | Region 3 参考数据收集完成 | +6.5h |
| **13:00** | 准备就绪，可运行完整试验 | +7h |

---

## 🎯 立即可执行的任务

### 任务 A: 下载 OpenVLA (阻塞性，优先级最高)

```bash
# 检查 HF token
echo $HF_TOKEN

# 如果为空，需要设置:
# export HF_TOKEN="your_huggingface_token"

# 下载模型
cd /home/admin/.openclaw/workspace/Embodied-RTA
python3 scripts/download_openvla.py
```

### 任务 B: 监控 Region 2 数据生成

```bash
# 监控进度
watch -n 60 'tail -5 /home/admin/.openclaw/workspace/Embodied-RTA/reachability/data_generation.log'
```

### 任务 C: 准备 Region 2 训练脚本

```bash
# 检查训练脚本
cat /home/admin/.openclaw/workspace/Embodied-RTA/reachability/train_gru_reachability.py | head -50
```

---

## 📋 检查清单

- [ ] OpenVLA-7B 下载完成 (`/data/models/openvla-7b/` 存在)
- [ ] Region 2 数据生成完成 (`reachability_dataset_v2.h5` 生成)
- [ ] Region 2 GRU 训练完成 (`gru_reachability.pth` 生成)
- [ ] Region 3 钩子注册成功 (无报错)
- [ ] Region 3 参考数据收集完成 (`reference_activations.npy` 生成)
- [ ] 完整试验可运行 (可选)

---

**下一步**: 等待用户确认后，立即开始 Step 1 (OpenVLA 下载)！
