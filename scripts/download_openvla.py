#!/usr/bin/env python3
"""OpenVLA-7B 下载脚本"""

from huggingface_hub import snapshot_download
import os

print("="*60)
print("OpenVLA-7B 下载")
print("="*60)

output_dir = "/data/models/openvla-7b"
os.makedirs(output_dir, exist_ok=True)

print(f"下载位置：{output_dir}")
print("开始下载 (约 14GB, 30-60 分钟)...")

try:
    snapshot_download(
        repo_id="openvla/openvla-7b",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("\n✅ 下载完成!")
except Exception as e:
    print(f"\n❌ 下载失败：{e}")
    raise
