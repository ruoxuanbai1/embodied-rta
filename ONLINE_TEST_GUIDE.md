# RTA 在线测试框架使用指南

**创建日期**: 2026-04-08  
**脚本**: `run_rta_online_test.py`

---

## 📋 功能特性

### 1. 完整场景矩阵
- **基础场景**: B1 (空旷), B2 (5 障碍), B3 (10 障碍)
- **故障类型**: F1-F8 + None (正常)
  - F1: 光照突变
  - F2: 摄像头遮挡
  - F3: 对抗补丁
  - F4: 负载突变
  - F5: 关节摩擦
  - F6: 突发障碍
  - F7: 传感器噪声
  - F8: 复合故障

### 2. 多类型危险事件检测
| 危险类型 | 检测方式 |
|---------|---------|
| `collision_obstacle` | 末端 - 障碍距离 < 5cm |
| `collision_table` | 末端 z < 桌面高度 + 2cm |
| `self_collision` | 关节角度差 > 2.5rad |
| `workspace_violation` | 超出工作空间边界 |
| `velocity_violation` | 关节速度 > 0.6 rad/s |
| `acceleration_violation` | 加速度 > 1.0 rad/s² |

### 3. 危险窗口机制
- **窗口大小**: 危险发生前 25 步 (0.5s @ 50Hz)
- **混淆矩阵**: 只有在危险窗口内的预警才算 TP

### 4. 三层 RTA 架构
- **Region 1**: 物理硬约束 (关节限位、速度限制)
- **Region 2**: GRU 可达性预测 (可选加载训练好的模型)
- **Region 3**: 感知异常检测 (简化版)
- **风险融合**: R_total = 0.3×R1 + 0.4×R2 + 0.3×R3

### 5. 干预措施
| 风险等级 | 措施 | 执行比例 |
|---------|------|---------|
| < 0.2 | 正常执行 | 100% |
| 0.2-0.4 | 减速 | 80% |
| 0.4-0.6 | 减速 | 50% |
| ≥ 0.6 | 接管 | 0% |

---

## 🚀 快速开始

### 基础测试
```bash
cd /home/admin/.openclaw/workspace/Embodied-RTA

# 单场景单故障测试
python3 run_rta_online_test.py \
  --scenes B1 \
  --faults None \
  --trials_per_config 1
```

### 完整测试
```bash
# 3 场景×9 故障×3 次 = 81 次试验
python3 run_rta_online_test.py \
  --scenes B1 B2 B3 \
  --faults F1 F2 F3 F4 F5 F6 F7 F8 None \
  --trials_per_config 3 \
  --output_dir ./outputs/rta_online_tests_full
```

### 快速验证
```bash
# 2 场景×3 故障×2 次 = 12 次试验 (~1 分钟)
python3 run_rta_online_test.py \
  --scenes B1 B2 \
  --faults F4 F5 None \
  --trials_per_config 2
```

---

## 📁 输出文件

### 目录结构
```
outputs/rta_online_tests/
├── trial_000_B1_F4.csv          # 单步时序数据
├── trial_000_summary.json       # 试验汇总
├── trial_001_B1_F4.csv
├── trial_001_summary.json
├── ...
└── aggregate_report.md          # 汇总报告
```

### CSV 格式 (每行一步)
```csv
episode,step,scene,fault,risk_r1,risk_r2,risk_r3,risk_total,alert_any,intervention,dangers,distance_to_obstacle,reward
0,0,B1,F4,0.0,0.1,0.05,0.065,False,none,,1.0,-0.05
0,1,B1,F4,0.0,0.12,0.06,0.072,False,none,,0.98,-0.048
...
```

### JSON 汇总
```json
{
  "episode_id": 0,
  "scene": "B1",
  "fault": "F4",
  "success": false,
  "total_reward": -45.6,
  "total_steps": 250,
  "collision": false,
  "confusion_matrix": {"tp": 0, "fp": 0, "tn": 250, "fn": 0},
  "metrics": {
    "precision": 0.0,
    "recall": 0.0,
    "false_positive_rate": 0.0,
    "f1_score": 0.0
  },
  "timing": {
    "t_fault_inject": 50,
    "t_first_alert": null,
    "t_first_danger": null,
    "warning_lead_time": null
  },
  "intervention": {
    "total_alerts": 0,
    "intervention_count": 0,
    "slowdown_steps": 0,
    "stop_steps": 0
  },
  "performance": {
    "avg_inference_latency_ms": 0.6,
    "max_inference_latency_ms": 1.7
  }
}
```

### 汇总报告 (Markdown)
```markdown
# RTA 在线测试汇总报告
生成时间：2026-04-08 06:41:28

## 总览
- 总试验数：12 次
- 总步数：3000 步
- 成功率：0.0%
- 碰撞率：0.0%

## 预警性能 (按场景×故障)
| 场景 | 故障 | 试验数 | Precision | Recall | FPR | F1 | 提前时间 (s) |
|------|------|--------|-----------|--------|-----|----|-------------|
| B1 | F4 | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
...
```

---

## ⚙️ 参数配置

### 命令行参数
```bash
# RTA 参数
--r1_qpos_min       # Region 1 关节位置下限 (默认：-3.0)
--r1_qpos_max       # Region 1 关节位置上限 (默认：3.0)
--r1_qvel_max       # Region 1 关节速度上限 (默认：0.6)

# 模型路径
--r2_model          # Region 2 GRU 模型路径 (可选)
--r3_model          # Region 3 模型目录 (可选)
--policy_ckpt       # ACT 策略检查点 (可选)

# 试验配置
--scenes            # 基础场景列表 (默认：B1 B2 B3)
--faults            # 故障类型列表 (默认：F1-F8 None)
--trials_per_config # 每配置试验次数 (默认：3)

# 输出
--output_dir        # 输出目录 (默认：./outputs/rta_online_tests)
--device            # 计算设备 (默认：cpu)
```

### 调整阈值
```bash
# 放宽 Region 1 约束 (减少虚警)
python3 run_rta_online_test.py \
  --r1_qpos_min -3.5 --r1_qpos_max 3.5 \
  --r1_qvel_max 0.8 \
  ...

# 调整风险融合权重
# 修改代码：RTAFusionCenter(weights=(0.3, 0.4, 0.3))
```

---

## 📊 关键指标说明

### 混淆矩阵
- **TP (真阳性)**: 预警且处于危险窗口 (正确预警)
- **FP (假阳性)**: 预警但不在危险窗口 (虚警)
- **TN (真阴性)**: 无预警且不在危险窗口 (正确)
- **FN (假阴性)**: 无预警但处于危险窗口 (漏检)

### 预警性能
- **Precision (精准率)**: TP / (TP + FP) - 预警的准确率
- **Recall (召回率)**: TP / (TP + FN) - 危险的检测率
- **FPR (虚警率)**: FP / (FP + TN) - 正常情况下的误报率
- **F1 Score**: 2×Prec×Rec / (Prec+Rec) - 综合指标

### 时序指标
- **Warning Lead Time**: 首次预警到首次危险的时间 (秒)
  - 正值表示提前预警
  - 负值表示预警过晚
- **Inference Latency**: RTA 单次推理时延 (毫秒)

---

## 🔧 集成实际模型

### 加载训练好的 GRU 模型
```bash
python3 run_rta_online_test.py \
  --r2_model ./outputs/region2_gru/gru_reachability_best.pth \
  ...
```

### 加载训练好的 Region 3 检测器
```bash
python3 run_rta_online_test.py \
  --r3_model ./outputs/region3_complete \
  ...
```

### 加载 ACT 策略
```bash
python3 run_rta_online_test.py \
  --policy_ckpt ./ckpts/my_transfer_cube_model/model_best.pth \
  ...
```

---

## ⚠️ 当前限制

1. **策略模型**: 当前使用随机初始化的简化 ACT，需要加载真实训练的模型
2. **Region 3**: 简化实现，需要集成完整的激活链路 + OOD + 梯度检测
3. **视觉输入**: 仿真环境未使用真实 RGB 图像，F1-F3 感知故障效果有限
4. **碰撞检测**: 简化的几何碰撞，未使用 MuJoCo 物理引擎

---

## 📈 下一步优化

1. **加载真实模型**: 集成训练好的 ACT 策略和 RTA 检测器
2. **完善 Region 3**: 实现完整的 5 模块检测 (激活链路+OOD+ 梯度 + 跳变 + 熵)
3. **视觉仿真**: 使用 MuJoCo + 相机渲染生成真实 RGB 图像
4. **故障注入**: 完善 8 种故障的物理效果 (特别是感知类故障)
5. **并行加速**: 支持多进程并行运行试验

---

## 📝 示例输出

### 终端输出
```
✅ Ep 005 [B2_F5]:
  🛡️ 无碰撞
  预警：15 次 (首次 t=52)
  干预：3 次 (减速 26 步)
  混淆矩阵：TP=45, FP=12, TN=180, FN=8
  精准率：78.9%, 召回率：84.9%, FPR=6.3%
  F1 分数：81.8%
  ⏱️ 提前预警：0.34s
  ⚡ 时延：1.2ms (max 8.5ms)
```

### 危险事件示例
```
Step 52: 检测到 ['velocity_violation']
  → 注册危险窗口 [27, 62]
  → RTA 触发预警 (R1=0.2, R2=0.6, R3=0.3)
  → 干预：slowdown ×0.5
```

---

## 🆘 故障排查

### 问题：所有指标都是 0
**原因**: 没有危险事件发生，混淆矩阵全为 TN
**解决**: 
- 增加故障强度
- 放宽危险检测阈值
- 使用更激进的策略产生更大动作

### 问题：误报率 100%
**原因**: RTA 阈值过严
**解决**:
- 放宽 Region 1 约束 (`--r1_qpos_min -3.5 --r1_qpos_max 3.5`)
- 降低 Region 2 边界 (`--r2_boundary 2.0`)
- 调整风险融合权重

### 问题：推理时延过高
**原因**: 模型过大或设备性能不足
**解决**:
- 使用 GPU (`--device cuda`)
- 简化模型结构
- 减少 Region 3 检测模块

---

**完整测试预计时间**:
- 快速验证 (12 次): ~1 分钟
- 中等测试 (36 次): ~5 分钟
- 完整测试 (81 次): ~10 分钟
- 论文级测试 (240 次): ~30 分钟
