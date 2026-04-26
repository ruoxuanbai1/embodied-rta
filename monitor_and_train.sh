#!/bin/bash
# 自动监控数据生成并启动训练

set -e

echo "=============================================="
echo "【自动监控 + 训练启动脚本】"
echo "=============================================="
echo ""

SSH_CMD="sshpass -p 'ohm4ecah' ssh -p 10320 root@js4.blockelite.cn"
LOG_FILE="/root/act/data_generation.log"
TRAIN_LOG="/root/act/gru_training.log"
CHECKPOINT="/root/act/retrain_data/data_generation_checkpoint.pkl"

# 上传工作日志
echo "1. 上传工作日志..."
cat /home/admin/.openclaw/workspace/Embodied-RTA/WORKLOG_2026-04-21.md | $SSH_CMD "cat > /root/act/WORKLOG_2026-04-21.md"
echo "   ✓ WORKLOG_2026-04-21.md 已上传"

# 监控循环
echo ""
echo "2. 开始监控数据生成进度..."
echo "   日志文件：$LOG_FILE"
echo "   检查间隔：60 秒"
echo ""

LAST_LINE=""
while true; do
    # 检查进程是否还在运行
    RUNNING=$($SSH_CMD "ps aux | grep -v grep | grep 'generate_gru_training' | wc -l")
    
    # 获取最新日志
    CURRENT_LINE=$($SSH_CMD "tail -1 $LOG_FILE 2>/dev/null || echo '等待中...'")
    
    # 显示进度
    if [ "$CURRENT_LINE" != "$LAST_LINE" ]; then
        echo "  [$CURRENT_LINE]"
        LAST_LINE="$CURRENT_LINE"
    fi
    
    if [ "$RUNNING" -eq 0 ]; then
        # 进程已结束，检查是否成功
        echo ""
        echo "✓ 数据生成进程已结束"
        
        # 检查输出文件
        FILES_EXIST=$($SSH_CMD "ls /root/act/retrain_data/gru_train_*.npy 2>/dev/null | wc -l")
        
        if [ "$FILES_EXIST" -ge 3 ]; then
            echo "✓ 数据文件已生成，准备启动训练"
            break
        else
            echo "⚠ 数据文件可能缺失，检查日志..."
            $SSH_CMD "tail -50 $LOG_FILE"
            exit 1
        fi
    fi
    
    # 检查是否卡住 (60 秒无新日志)
    SLEEP_COUNT=0
    while [ "$CURRENT_LINE" = "$LAST_LINE" ] && [ "$SLEEP_COUNT" -lt 10 ]; do
        sleep 10
        SLEEP_COUNT=$((SLEEP_COUNT + 1))
        CURRENT_LINE=$($SSH_CMD "tail -1 $LOG_FILE 2>/dev/null || echo '等待中...'")
    done
    
    if [ "$SLEEP_COUNT" -ge 10 ]; then
        echo "  ⚠ 警告：60 秒无新日志，但进程仍在运行"
        $SSH_CMD "ps aux | grep 'generate_gru' | grep -v grep"
    fi
done

# 显示数据生成统计
echo ""
echo "3. 数据生成统计..."
$SSH_CMD "cd /root/act && python3 -c \"
import numpy as np
h = np.load('retrain_data/gru_train_history.npy')
t = np.load('retrain_data/gru_train_support_target.npy')
l = np.load('retrain_data/gru_train_danger_label.npy')
print(f'  样本数：{len(h):,}')
print(f'  历史形状：{h.shape}')
print(f'  目标形状：{t.shape}')
print(f'  危险比例：{np.mean(l)*100:.1f}%')
\""

# 启动训练
echo ""
echo "4. 启动 GRU 模型训练..."
echo "   预计时间：2-4 小时"
echo "   日志：$TRAIN_LOG"

$SSH_CMD "cd /root/act && source /home/vipuser/miniconda3/etc/profile.d/conda.sh && conda activate aloha_headless && \
    nohup python3 -u train_gru_weighted.py > $TRAIN_LOG 2>&1 & \
    echo \$! > gru_training.pid && \
    echo '✓ 训练进程已启动 (PID: '\$(cat gru_training.pid)')'"

# 等待 10 秒确认训练启动
sleep 10
$SSH_CMD "tail -30 $TRAIN_LOG"

echo ""
echo "=============================================="
echo "✓ 训练已启动！"
echo "=============================================="
echo ""
echo "监控命令:"
echo "  查看训练日志：$SSH_CMD 'tail -f $TRAIN_LOG'"
echo "  查看进程状态：$SSH_CMD 'ps aux | grep python' | grep -v grep"
echo ""
echo "模型输出位置:"
echo "  /root/act/outputs/region2_gru_retrained/gru_retrained_best.pth"
echo "  /root/act/outputs/region2_gru_retrained/gru_retrained_best_j.pth"
echo ""
echo "明日继续:"
echo "  1. 检查训练是否完成"
echo "  2. 阈值校准"
echo "  3. 完整评估 (113 集)"
echo ""
