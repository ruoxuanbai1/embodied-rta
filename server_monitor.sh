#!/bin/bash
# 服务器端监控脚本 (数据生成 → GRU 训练 → OOD 优化 → 完整评估)

set -e

echo "=============================================="
echo "【完整重训练流程 - 自动执行】"
echo "=============================================="

LOG_FILE="/root/act/data_generation.log"
TRAIN_LOG="/root/act/gru_training.log"
OOD_LOG="/root/act/ood_retrain.log"
EVAL_LOG="/root/act/final_evaluation.log"

# ========== 阶段 1: 数据生成 ==========
echo ""
echo "【阶段 1/4】数据生成监控..."
echo "=============================================="

while true; do
    RUNNING=$(ps aux | grep -v grep | grep 'generate_gru_training' | wc -l)
    
    if [ "$RUNNING" -eq 0 ]; then
        echo "✓ 数据生成完成"
        
        if [ -f "/root/act/retrain_data/gru_train_history.npy" ]; then
            echo "✓ 数据文件就绪"
            break
        else
            echo "⚠ 数据文件缺失"
            exit 1
        fi
    fi
    
    tail -1 $LOG_FILE
    sleep 60
done

# 数据生成统计
echo ""
echo "数据生成统计:"
python3 -c "
import numpy as np
h = np.load('retrain_data/gru_train_history.npy')
l = np.load('retrain_data/gru_train_danger_label.npy')
print(f'  样本数：{len(h):,}')
print(f'  危险比例：{np.mean(l)*100:.1f}%')
"

# ========== 阶段 2: GRU 训练 ==========
echo ""
echo "【阶段 2/4】启动 GRU 训练..."
echo "=============================================="

source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate aloha_headless
cd /root/act

nohup python3 -u train_gru_weighted.py > $TRAIN_LOG 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > gru_training.pid
echo "✓ 训练已启动 (PID: $TRAIN_PID)"

sleep 10
echo "训练日志 (前 20 行):"
tail -20 $TRAIN_LOG

# 监控训练
echo ""
echo "监控 GRU 训练进度..."
while true; do
    TRAIN_RUNNING=$(ps aux | grep -v grep | grep 'train_gru_weighted' | wc -l)
    
    if [ "$TRAIN_RUNNING" -eq 0 ]; then
        echo "✓ GRU 训练完成"
        
        if [ -f "/root/act/outputs/region2_gru_retrained/gru_retrained_best_j.pth" ]; then
            echo "✓ 模型文件就绪"
            break
        else
            echo "⚠ 模型文件缺失"
            exit 1
        fi
    fi
    
    # 每 5 分钟显示一次进度
    tail -5 $TRAIN_LOG | grep "Epoch"
    sleep 300
done

# 显示训练结果
echo ""
echo "训练结果:"
grep "最佳 Youden" $TRAIN_LOG | tail -1

# ========== 阶段 3: OOD 优化 ==========
echo ""
echo "【阶段 3/4】OOD 统计量重计算..."
echo "=============================================="

python3 compute_ood_stats_retrained.py > $OOD_LOG 2>&1
echo "✓ OOD 重计算完成"

# 显示 OOD 区分度
echo ""
echo "OOD 区分度:"
grep "区分度" $OOD_LOG | tail -3

# ========== 阶段 4: 完整评估 ==========
echo ""
echo "【阶段 4/4】完整评估 (113 集)..."
echo "=============================================="

# 替换模型和 OOD 统计量
cp /root/act/outputs/region2_gru_retrained/gru_retrained_best_j.pth /root/act/outputs/region2_gru/gru_reachability_best.pth
cp /root/act/outputs/region3_detectors/ood_stats_retrained.json /root/act/outputs/region3_detectors/ood_stats.json

echo "✓ 模型和 OOD 统计量已更新"
echo ""
echo "准备运行完整评估..."
echo "注意：完整评估需要 1-2 小时，将在后台运行"
echo "日志：$EVAL_LOG"

# 提示用户手动运行评估 (因为需要选择正确的评估脚本)
echo ""
echo "=============================================="
echo "✓ 自动流程完成！"
echo "=============================================="
echo ""
echo "下一步 (手动执行):"
echo "  1. 运行完整评估:"
echo "     python3 analyze_r2_retrained.py"
echo ""
echo "  2. 查看结果:"
echo "     cat /mnt/data/ablation_experiments/analysis_retrained/summary.json"
echo ""
echo "  3. 对比新旧模型性能"
echo ""
echo "所有文件位置:"
echo "  GRU 模型：/root/act/outputs/region2_gru_retrained/"
echo "  OOD 统计：/root/act/outputs/region3_detectors/ood_stats_retrained.json"
echo "  工作日志：/root/act/WORKLOG_2026-04-21.md"
echo ""
