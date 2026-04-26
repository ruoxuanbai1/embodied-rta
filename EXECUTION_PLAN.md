# OpenVLA-7B 完整实验执行计划

**开始时间**: 2026-03-30 18:55  
**目标**: 尽快完成基于真实 OpenVLA-7B 的 3120 次试验

---

## 📋 并行执行计划

### 阶段 1: 环境准备 (正在进行，预计 1 小时)

```bash
# 1.1 升级 PyTorch (完成)
pip install torch==2.1.0 torchvision torchaudio --upgrade

# 1.2 安装 HF 依赖 (完成)
pip install huggingface_hub transformers accelerate einops

# 1.3 下载 OpenVLA-7B (需要 HF token)
# 位置：/data/models/openvla-7b (133GB 可用)
```

**状态**: ⏳ 等待 HuggingFace token

---

### 阶段 2: Region 1 实现 (已完成，0.5 小时)

**文件**: `envs/region1_constraints.py`

**实现内容**:
- ✅ 位置约束 (末端高度、防撞距离、工作空间)
- ✅ 速度约束 (底盘、关节)
- ✅ 加速度约束 (底盘、关节)
- ✅ 扭矩约束 (7 关节)
- ✅ ZMP 稳定性判据

**支撑变量**: 15 变量 × 2 (min+max) = **30 维输出**

---

### 阶段 3: Region 2 GRU 训练数据生成 (进行中，预计 3 小时)

**文件**: `reachability/generate_training_data.py`

**步骤**:
1. 生成/收集轨迹数据 (5000 正常 + 2000 故障)
2. 添加扰动 (控制 + 状态 + 参数)
3. 动力学方程推演可达集
4. 计算支撑函数 (上下界)
5. 保存训练数据 (70,000 样本)

---

### 阶段 4: Region 2 GRU 训练 (预计 2 小时)

**文件**: `reachability/train_gru_reachability.py`

**架构**:
```
输入：(10, 19) → GRU(128)×2 → FC(64) → ReLU → FC(30) → 输出
                  │
                  └─ 10 帧状态序列 [base_x, base_y, base_v, base_ω, 
                                     arm_q[7], arm_dq[7], com[3]]
```

**输出**: 30 维支撑函数 (15 变量的 min+max)

---

### 阶段 5: Region 3 多层激活链路 (预计 3 小时)

**文件**: `xai/multi_layer_activation.py`

**实现**:
- 注册 OpenVLA 各层钩子 (Vision + LLM + Action)
- 收集正常/OOD 激活模式
- 计算层间相关性
- 训练异常检测器

---

### 阶段 6: 完整试验 (预计 4-6 小时)

**文件**: `tests/run_openvla_trials.py`

**规模**: 13 方法 × 8 场景 × 30 次 = 3120 次

**OpenVLA 推理**: ~70ms/次 → 3120 次 ≈ 4 小时

---

### 阶段 7: 图表 + 报告 (预计 2 小时)

- 生成 15 张论文级图表
- 更新实验报告
- 补充技术文档

---

## ⚠️ 关键依赖

### HuggingFace Token (必需)

**获取方式**:
1. 访问 https://huggingface.co/settings/tokens
2. 创建新 token (read 权限)
3. 在服务器上运行：
   ```bash
   huggingface-cli login
   # 粘贴 token
   ```

**或者** 直接设置环境变量：
```bash
export HF_TOKEN=your_token_here
```

---

## 📊 预计完成时间

| 阶段 | 开始时间 | 结束时间 | 状态 |
|------|----------|----------|------|
| 1. 环境准备 | 18:55 | 20:00 | ⏳ 进行中 |
| 2. Region 1 | 18:55 | 19:25 | ✅ 完成 |
| 3. Region 2 数据 | 19:25 | 22:25 | ⏳ 待开始 |
| 4. Region 2 训练 | 22:25 | 00:25 | ⏳ 待开始 |
| 5. Region 3 | 19:25 | 22:25 | ⏳ 待开始 |
| 6. 完整试验 | 00:25 | 06:25 | ⏳ 待开始 |
| 7. 图表报告 | 06:25 | 08:25 | ⏳ 待开始 |

**预计完成**: 2026-03-31 08:25 (约 13 小时)

---

## 🚀 立即行动项

1. **提供 HuggingFace token** (阻塞中)
2. 确认 Region 1 实现是否符合预期
3. 开始 Region 2 训练数据生成脚本

---

**最后更新**: 2026-03-30 19:00
