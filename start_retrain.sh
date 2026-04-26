#!/bin/bash
# GRU 重训练启动脚本 - 带进程保护

set -e

echo "=============================================="
echo "【GRU 重训练启动脚本】"
echo "=============================================="
echo ""

# SSH 登录信息
SSH_CMD="sshpass -p 'ohm4ecah' ssh -p 10320 root@js4.blockelite.cn"

# 检查连接
echo "1. 检查 SSH 连接..."
$SSH_CMD "echo '✓ SSH 连接正常'" || {
    echo "✗ SSH 连接失败"
    exit 1
}

# 激活环境
echo ""
echo "2. 激活 conda 环境..."
$SSH_CMD "source /home/vipuser/miniconda3/etc/profile.d/conda.sh && conda activate aloha_headless && echo '✓ 环境激活'"

# 创建输出目录
echo ""
echo "3. 创建输出目录..."
$SSH_CMD "mkdir -p /root/act/retrain_data && mkdir -p /root/act/outputs/region2_gru_retrained && echo '✓ 目录就绪'"

# 上传脚本
echo ""
echo "4. 上传脚本..."
scp -P 10320 \
    /home/admin/.openclaw/workspace/Embodied-RTA/generate_support_directions.py \
    /home/admin/.openclaw/workspace/Embodied-RTA/generate_gru_training_data.py \
    /home/admin/.openclaw/workspace/Embodied-RTA/train_gru_weighted.py \
    root@js4.blockelite.cn:/root/act/
echo "✓ 脚本上传完成"

# 启动支撑方向生成
echo ""
echo "5. 生成支撑方向矩阵..."
$SSH_CMD "cd /root/act && source /home/vipuser/miniconda3/etc/profile.d/conda.sh && conda activate aloha_headless && python3 generate_support_directions.py"

# 启动训练数据生成 (后台，带日志)
echo ""
echo "6. 启动训练数据生成 (蒙特卡洛)..."
echo "   预计时间：2-4 小时"
echo "   日志：/root/act/data_generation.log"

$SSH_CMD "cd /root/act && source /home/vipuser/miniconda3/etc/profile.d/conda.sh && conda activate aloha_headless && \
    nohup python3 -u generate_gru_training_data.py > data_generation.log 2>&1 & \
    echo \$! > data_generation.pid && \
    echo '✓ 数据生成进程已启动 (PID: '\$(cat data_generation.pid)')'"

# 等待数据生成完成
echo ""
echo "7. 等待数据生成完成..."
echo "   使用以下命令监控进度:"
echo "   $SSH_CMD 'tail -f /root/act/data_generation.log'"
echo ""
echo "   或者等待自动继续 (按 Ctrl+C 可中断等待，进程会继续运行)..."
echo ""

# 监控循环
while true; do
    sleep 60
    
    # 检查进程是否还在运行
    RUNNING=$($SSH_CMD "ps aux | grep -v grep | grep 'generate_gru_training' | wc -l")
    
    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "✓ 数据生成完成！"
        break
    fi
    
    # 显示最新日志
    LAST_LINE=$($SSH_CMD "tail -1 /root/act/data_generation.log 2>/dev/null || echo '等待中...'")
    echo "  [$LAST_LINE]"
done

# 检查数据生成结果
echo ""
echo "8. 检查数据生成结果..."
$SSH_CMD "cd /root/act && ls -lh retrain_data/*.npy 2>/dev/null && echo '✓ 数据文件就绪' || echo '✗ 数据文件缺失'"

# 启动模型训练 (后台，带日志)
echo ""
echo "9. 启动 GRU 模型训练..."
echo "   预计时间：2-4 小时"
echo "   日志：/root/act/gru_training.log"

$SSH_CMD "cd /root/act && source /home/vipuser/miniconda3/etc/profile.d/conda.sh && conda activate aloha_headless && \
    nohup python3 -u train_gru_weighted.py > gru_training.log 2>&1 & \
    echo \$! > gru_training.pid && \
    echo '✓ 训练进程已启动 (PID: '\$(cat gru_training.pid)')'"

echo ""
echo "=============================================="
echo "✓ 所有进程已启动！"
echo "=============================================="
echo ""
echo "监控命令:"
echo "  查看数据生成日志：$SSH_CMD 'tail -f /root/act/data_generation.log'"
echo "  查看训练日志：$SSH_CMD 'tail -f /root/act/gru_training.log'"
echo "  查看进程状态：$SSH_CMD 'ps aux | grep python'"
echo ""
echo "模型输出:"
echo "  /root/act/outputs/region2_gru_retrained/gru_retrained_best.pth"
echo "  /root/act/outputs/region2_gru_retrained/gru_retrained_best_j.pth"
echo ""
