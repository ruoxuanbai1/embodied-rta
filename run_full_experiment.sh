#!/bin/bash
# RT-1 完整试验快速启动脚本

set -e

cd /root/Embodied-RTA
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rt1-isaac

echo "=========================================="
echo "RT-1 完整 RTA 试验"
echo "开始时间：$(date)"
echo "=========================================="

# 步骤 1: 采集轨迹数据
echo ""
echo "[1/4] 采集 ACT 轨迹数据..."
python3 collect_trajectories.py \
  --n-normal 50 \
  --n-fault 30 \
  --output-dir /root/rt1_trajectory_data

# 步骤 2: 训练 GRU 模型
echo ""
echo "[2/4] 训练 GRU 可达集模型..."
python3 train_gru_reachability.py \
  --trajectory-dir /root/rt1_trajectory_data \
  --output /root/Embodied-RTA/gru_reachability.pth \
  --epochs 50

# 步骤 3: 学习 Region 3 参数
echo ""
echo "[3/4] 学习 Region 3 参数..."
python3 learn_region3_params.py \
  --trajectory-dir /root/rt1_trajectory_data \
  --output /root/Embodied-RTA/region3_learned_params.json

# 步骤 4: 运行完整试验
echo ""
echo "[4/4] 运行完整试验..."
python3 full_experiment_runner.py \
  --trials 30 \
  --output /root/rt1_full_experiment_results \
  --resume

echo ""
echo "=========================================="
echo "试验完成!"
echo "结束时间：$(date)"
echo "结果位置：/root/rt1_full_experiment_results"
echo "=========================================="
