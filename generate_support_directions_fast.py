#!/usr/bin/env python3
"""
快速生成支撑方向矩阵 (不收集危险数据做 PCA)

直接用:
- 关节限位方向 (28 个)
- 速度 - 位置耦合方向 (28 个)
- 随机方向 (32 个)

合并后 SVD 压缩到 32 个
"""

import os
import numpy as np

OUTPUT_DIR = "/root/act/retrain_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORT_DIM = 28  # 不能超过原始特征数 (28 维状态)

print("="*60)
print("【快速生成支撑方向】")
print("="*60)

# 1. 关节限位方向 (28 个)
print("1. 关节限位方向 (28 个)...")
joint_limit_dirs = []
for i in range(14):
    d_min = np.zeros(28)
    d_min[i] = -1.0
    joint_limit_dirs.append(d_min)
    
    d_max = np.zeros(28)
    d_max[i] = 1.0
    joint_limit_dirs.append(d_max)
joint_limit_dirs = np.array(joint_limit_dirs)

# 2. 速度 - 位置耦合方向 (28 个)
print("2. 速度 - 位置耦合方向 (28 个)...")
coupling_dirs = []
for i in range(14):
    d1 = np.zeros(28)
    d1[i] = 1.0
    d1[14 + i] = 1.0
    coupling_dirs.append(d1)
    
    d2 = np.zeros(28)
    d2[i] = -1.0
    d2[14 + i] = -1.0
    coupling_dirs.append(d2)
coupling_dirs = np.array(coupling_dirs)

# 3. 随机方向 (32 个)
print("3. 随机方向 (32 个)...")
np.random.seed(42)
random_dirs = np.random.randn(32, 28)
random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)

# 4. 合并
print("4. 合并所有方向...")
all_dirs = np.vstack([
    joint_limit_dirs,   # (28, 28)
    coupling_dirs,      # (28, 28)
    random_dirs,        # (32, 28)
])
print(f"   合并后：{all_dirs.shape}")

# 5. SVD 压缩
print(f"5. SVD 压缩到 {SUPPORT_DIM} 个方向 (最大 28)...")
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=SUPPORT_DIM)
svd.fit(all_dirs)
support_directions = svd.components_

# 归一化
support_directions = support_directions / np.linalg.norm(support_directions, axis=1, keepdims=True)

# 6. 保存
output_path = os.path.join(OUTPUT_DIR, 'support_directions.npy')
np.save(output_path, support_directions)

print("\n" + "="*60)
print("✓ 支撑方向生成完成！")
print("="*60)
print(f"输出：{output_path}")
print(f"形状：{support_directions.shape}")
print(f"参数量：{support_directions.nbytes / 1024:.1f} KB")
