#!/bin/bash
# 具身智能三层 RTA - 自动运行脚本
# 使用 nohup 后台运行，确保不中断

cd /home/vipuser/Embodied-RTA

echo "========================================"
echo "具身智能三层 RTA 试验 - 启动时间: $(date)"
echo "========================================"

# 1. 运行试验 (32 配置)
echo "开始运行试验..."
source /home/vipuser/miniconda3/bin/activate
python3 tests/run_all_trials.py > /tmp/embodied_trials.log 2>&1

# 2. 生成图表
echo "试验完成，生成图表..."
python3 tests/generate_all_figs.py >> /tmp/embodied_trials.log 2>&1

# 3. 生成报告
echo "生成报告..."
python3 tests/generate_report.py >> /tmp/embodied_trials.log 2>&1

# 4. 自动下载到本地
echo "下载到本地..."
cd /home/vipuser/Embodied-RTA
tar -czf /tmp/embodied_rta_results.tar.gz outputs/ *.md

echo "========================================"
echo "全部完成 - 结束时间：$(date)"
echo "结果位置：/tmp/embodied_rta_results.tar.gz"
echo "日志位置：/tmp/embodied_trials.log"
echo "========================================"
