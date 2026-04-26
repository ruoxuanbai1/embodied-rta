#!/bin/bash
# OpenVLA-7B 下载脚本
# 存储位置：/data/models/openvla-7b (133GB 可用空间)

set -e

echo "=========================================="
echo "OpenVLA-7B 下载脚本"
echo "=========================================="

# 创建存储目录
MODEL_DIR="/data/models/openvla-7b"
mkdir -p $MODEL_DIR

echo "模型将下载到：$MODEL_DIR"

# 检查 HuggingFace 登录
if [ ! -f ~/.cache/huggingface/token ]; then
    echo "⚠️  未检测到 HuggingFace token"
    echo "请先运行：huggingface-cli login"
    echo "或手动设置：export HF_TOKEN=your_token"
    exit 1
fi

# 使用 Git LFS 下载
echo "开始下载 OpenVLA-7B (约 14GB, 30-60 分钟)..."
cd /data/models
git lfs install
git clone https://huggingface.co/openvla/openvla-7b

echo "=========================================="
echo "下载完成!"
echo "模型位置：$MODEL_DIR"
echo "=========================================="

# 验证下载
ls -lh $MODEL_DIR
