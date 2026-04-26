#!/bin/bash
# OpenVLA-7B 下载脚本 (使用国内镜像)
# 镜像源：https://hf-mirror.com

set -e

echo "=========================================="
echo "OpenVLA-7B 下载 (国内镜像加速)"
echo "=========================================="

MODEL_DIR="/data/models/openvla-7b"
# Set HF_TOKEN before running: export HF_TOKEN="your_hf_token"
if [ -z "$HF_TOKEN" ]; then echo "Error: HF_TOKEN not set"; exit 1; fi

echo "模型位置：$MODEL_DIR"

# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN

cd /data/models

# 检查是否已存在
if [ -d "openvla-7b" ]; then
    echo "检测到已有下载，尝试断点续传..."
    rm -rf openvla-7b/.git/lfs/tmp 2>/dev/null || true
fi

# 使用 git clone + LFS 下载
echo "开始下载 (使用 hf-mirror.com 镜像)..."
git lfs install
git clone https://hf-mirror.com/openvla/openvla-7b.git --depth 1 2>&1 || {
    echo "Git 方式失败，尝试 huggingface_hub..."
    source /home/vipuser/miniconda3/bin/activate
    python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openvla/openvla-7b',
    local_dir='/data/models/openvla-7b',
    resume_download=True,
    max_workers=8
)
"
}

echo "=========================================="
echo "下载完成!"
echo "模型位置：$MODEL_DIR"
echo "=========================================="

ls -lh $MODEL_DIR
