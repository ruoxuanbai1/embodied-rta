# ALOHA 三层 RTA 真实闭环试验 - 工作方案

**创建时间**: 2026-04-04 13:45  
**执行人**: AI Agent  
**目标**: 基于 ALOHA 仿真环境的真实闭环 RTA 试验

---

## 📋 问题识别

### 现有数据缺陷

| 问题 | 描述 | 影响 |
|------|------|------|
| 状态生成 | `torch.randn(14)` 随机生成 | ❌ 不真实 |
| 仿真环境 | 无 ALOHA/Isaac Lab 仿真 | ❌ 无物理约束 |
| 训练数据 | 基于随机状态 + ACT 动作 | ❌ 不可靠 |
| RTA 阈值 | 基于缺陷数据训练 | ❌ 性能不可信 |

### 正确方案

**必须使用**:
- ✅ ALOHA 仿真环境 (MuJoCo/dm_control)
- ✅ 真实仿真状态 + ACT 推理动作
- ✅ 故障注入基于仿真物理
- ✅ 独立训练集和测试集

---

## 🎯 工作目标

### 阶段 1: 环境配置 (1-2 小时)

- [ ] 安装 MuJoCo 3.1.3
- [ ] 安装 dm_control 1.0.14
- [ ] 安装 gymnasium 0.29.0
- [ ] 创建 ALOHA 仿真环境类
- [ ] 验证环境可运行

### 阶段 2: 训练数据收集 (4-6 小时)

- [ ] Normal 轨迹：500 条 (ALOHA 仿真 + ACT 推理)
- [ ] Fault 轨迹：400 条 (8 种故障×50 条)
- [ ] 数据验证：检查状态/动作合理性
- [ ] 数据分割：训练集 80% / 测试集 20%

### 阶段 3: Region 2 重新训练 (2-3 小时)

- [ ] 生成可达集标签 (蒙特卡洛扰动)
- [ ] 训练 GRU 模型 (512×4 层)
- [ ] 评估覆盖率 (目标≥95%)
- [ ] 保存最佳模型

### 阶段 4: Region 3 重新训练 (1-2 小时)

- [ ] 提取 OOD 马氏距离特征
- [ ] 提取决策因子贡献度特征
- [ ] 提取激活链路掩码特征
- [ ] 学习三模块独立阈值
- [ ] 评估预警性能 (目标 TPR≥90%, FPR≤10%)

### 阶段 5: 真实闭环试验 (6-22 小时)

- [ ] 11 场景 (3 基础 + 8 故障)
- [ ] 14 配置 (8 消融 + 6 基线)
- [ ] 每场景 30 次重复
- [ ] 总计 4,620 次真实仿真试验
- [ ] 记录预警指标 + 任务性能

### 阶段 6: 数据分析 (2-3 小时)

- [ ] 计算 Precision/Recall/FPR/AUC
- [ ] 生成论文图表 (6-8 张)
- [ ] 撰写实验报告
- [ ] 统计显著性检验

---

## ⏱️ 时间估算

| 阶段 | 预计时间 | 累计时间 |
|------|----------|----------|
| 1. 环境配置 | 1-2h | 1-2h |
| 2. 数据收集 | 4-6h | 5-8h |
| 3. Region 2 训练 | 2-3h | 7-11h |
| 4. Region 3 训练 | 1-2h | 8-13h |
| 5. 闭环试验 | 6-22h | 14-35h |
| 6. 数据分析 | 2-3h | 16-38h |
| **总计** | **16-38 小时** | **约 2-5 天** |

---

## 📁 交付物

### 代码
- `envs/aloha_simulation_env.py` - ALOHA 仿真环境
- `data/collect_training_data.py` - 数据收集脚本
- `reachability/train_gru_v3.py` - Region 2 训练
- `xai/train_region3_v3.py` - Region 3 训练
- `experiments/run_real_closed_loop.py` - 闭环试验

### 数据
- `data/aloha_normal_500/` - 500 条 normal 轨迹
- `data/aloha_fault_400/` - 400 条 fault 轨迹
- `outputs/closed_loop/results.csv` - 试验结果

### 模型
- `models/region2_gru_v3_best.pth` - Region 2 模型
- `models/region3_thresholds_v3.json` - Region 3 阈值

### 图表
- `outputs/figures/roc_comparison.png`
- `outputs/figures/success_rate.png`
- `outputs/figures/collision_rate.png`
- `outputs/figures/warning_metrics.png`

### 文档
- `EXPERIMENT_REPORT_v3.md` - 完整实验报告
- `WORKSPACE_LOG.md` - 执行日志 (本文件)

---

## 🔑 关键质量要求

### 数据质量

- ✅ 状态来自 ALOHA 仿真 (非随机)
- ✅ 动作来自 ACT 推理 (非预设)
- ✅ 故障注入基于物理 (非简单缩放)
- ✅ 训练/测试集独立

### 模型性能

| 模块 | 指标 | 目标 |
|------|------|------|
| Region 2 | 覆盖率 | ≥95% |
| Region 3 | TPR | ≥90% |
| Region 3 | FPR | ≤10% |
| Region 3 | AUC | ≥0.90 |

### 试验严谨性

- ✅ 11 场景覆盖全面
- ✅ 14 配置对比完整
- ✅ 30 次重复统计显著
- ✅ 独立测试集验证

---

## 📊 执行日志

### 2026-04-04 13:45 - 方案确认

- ✅ 确认问题：现有数据基于随机状态，不可靠
- ✅ 确认方案：使用 ALOHA 仿真环境重新收集数据
- ✅ 确认时间：预计 16-38 小时
- ✅ 创建工作方案文档

### 下一步

1. 运行 `setup_aloha_env.sh` 安装环境
2. 验证 ALOHA 仿真环境
3. 开始收集训练数据

---

**记录人**: AI Agent  
**下次更新**: 环境安装完成后
