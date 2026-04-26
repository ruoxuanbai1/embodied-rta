# 🔍 实时监控状态

**更新时间**: 2026-04-21 19:15 GMT+8

---

## 📊 当前进度

### 数据生成 (进行中)
- **进度**: 20/113 集 (17.7%)
- **样本数**: 6,800
- **危险比例**: 86.5%
- **预计完成**: 21:30-22:00 (约 2-2.5 小时后)
- **Checkpoint**: ✓ 已保存 (17MB)

### 进程状态
```
✓ generate_gru_training_data.py - 运行中
✓ server_monitor.sh - 运行中 (自动启动训练)
```

---

## 📁 已生成文件

### 服务器端 (/root/act/)
```
retrain_data/
├── support_directions.npy        # 支撑方向 (28×28, 6.3KB)
├── data_generation_checkpoint.pkl # 数据生成 checkpoint (17MB)
└── [生成中] gru_train_*.npy       # 训练数据

WORKLOG_2026-04-21.md              # 工作日志
server_monitor.sh                  # 自动监控脚本
server_monitor.log                 # 监控日志
data_generation.log                # 数据生成日志
```

### 本地 (/workspace/Embodied-RTA/)
```
generate_support_directions_fast.py   # 支撑方向生成
generate_gru_training_data.py         # 数据生成 (蒙特卡洛)
train_gru_weighted.py                 # GRU 训练 (310K 参数)
server_monitor.sh                     # 服务器监控
WORKLOG_2026-04-21.md                 # 工作日志
MONITORING_STATUS.md                  # 本监控状态
```

---

## ⏰ 时间线

| 时间 | 事件 | 状态 |
|------|------|------|
| 18:25 | 支撑方向生成 | ✓ 完成 |
| 18:44 | 数据生成启动 | ✓ 运行中 (17.7%) |
| ~21:30 | 数据生成完成 | ⏳ 等待中 |
| ~21:35 | GRU 训练自动启动 | ⏳ 等待中 |
| ~23:30-01:30 | 训练完成 | ⏳ 等待中 |

---

## 📋 监控命令

### 实时查看进度
```bash
# 数据生成日志
ssh root@js4.blockelite.cn -p 10320 "tail -f /root/act/data_generation.log"

# 监控日志
ssh root@js4.blockelite.cn -p 10320 "tail -f /root/act/server_monitor.log"

# 进程状态
ssh root@js4.blockelite.cn -p 10320 "ps aux | grep python | grep -v grep"
```

### 检查文件
```bash
# 数据文件
ssh root@js4.blockelite.cn -p 10320 "ls -lh /root/act/retrain_data/"

# 训练输出
ssh root@js4.blockelite.cn -p 10320 "ls -lh /root/act/outputs/region2_gru_retrained/"
```

---

## 🌙 夜间运行说明

### 自动流程
1. **数据生成完成** → 自动检测
2. **自动启动训练** → server_monitor.sh 处理
3. **训练完成** → 保存最佳模型

### 明日早晨检查 (6:00 AM)
```bash
# 1. 检查训练是否完成
ssh root@js4.blockelite.cn -p 10320 "tail -50 /root/act/gru_training.log"

# 2. 查看最佳 Youden J
ssh root@js4.blockelite.cn -p 10320 "cat /root/act/gru_training.log | grep '最佳 Youden'"

# 3. 检查模型文件
ssh root@js4.blockelite.cn -p 10320 "ls -lh /root/act/outputs/region2_gru_retrained/"
```

---

## 📞 紧急联系

如果进程意外终止：

### 恢复数据生成
```bash
ssh root@js4.blockelite.cn -p 10320
cd /root/act
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate aloha_headless
python3 generate_gru_training_data.py  # 会自动从 checkpoint 恢复
```

### 恢复训练
```bash
ssh root@js4.blockelite.cn -p 10320
cd /root/act
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate aloha_headless
python3 train_gru_weighted.py  # 会自动从 checkpoint 恢复
```

---

## ✅ 一切正常

- ✅ 数据生成进度正常 (17.7%, 20/113 集)
- ✅ Checkpoint 定期保存 (每 10 集)
- ✅ 监控脚本运行中
- ✅ 工作日志已记录
- ✅ 明日交接文档已准备

---

*最后更新*: 2026-04-21 19:15  
*下次检查*: 21:30 (数据生成完成时)
