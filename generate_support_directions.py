#!/usr/bin/env python3
"""
生成 GRU 可达集预测模型的支撑方向矩阵

混合策略：
- 关节限位方向 (28 个)
- 速度 - 位置耦合方向 (28 个)
- PCA 学习方向 (16 个，从危险数据)
- 随机覆盖方向 (16 个)

最终 SVD 压缩到 32 个方向
"""

import os, sys, json
import numpy as np
from datetime import datetime

# ========== 配置 ==========
DATA_DIR = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
OUTPUT_DIR = "/root/act/retrain_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORT_DIM = 32  # 最终压缩到 32 个方向
N_PCA = 16
N_RANDOM = 16

JOINT_SAFE_RANGE = [
    (-1.85, 1.26), (-2.12, -0.96), (0.11, 1.57), (-0.16, 1.25),
    (-0.30, 3.51), (-2.43, 0.00), (0.08, 2.73), (-0.16, 1.26),
    (-2.26, -0.96), (0.11, 1.57), (-0.16, 1.25), (-1.85, 0.00),
    (0.00, 0.17), (0.08, 0.18),
]

# ========== 危险判定 ==========
def is_danger(qpos, qvel, threshold=0.03):
    for i in range(min(len(qpos), len(JOINT_SAFE_RANGE))):
        q_min, q_max = JOINT_SAFE_RANGE[i]
        margin = min(qpos[i] - q_min, q_max - qpos[i])
        joint_range = q_max - q_min
        if margin < -joint_range * threshold:
            return True
    
    if np.any(np.abs(qvel) > 0.6):
        return True
    
    return False

# ========== 生成支撑方向 ==========
def generate_support_directions():
    print("="*60)
    print("【生成支撑方向矩阵】")
    print("="*60)
    print(f"目标维度：{SUPPORT_DIM}")
    print(f"PCA 方向：{N_PCA}, 随机方向：{N_RANDOM}\n")
    
    # 1. 关节限位方向 (28 个)
    print("1. 生成关节限位方向 (28 个)...")
    joint_limit_dirs = []
    for i in range(14):
        d_min = np.zeros(28)
        d_min[i] = -1.0  # 指向下限
        joint_limit_dirs.append(d_min)
        
        d_max = np.zeros(28)
        d_max[i] = 1.0  # 指向上限
        joint_limit_dirs.append(d_max)
    joint_limit_dirs = np.array(joint_limit_dirs)  # (28, 28)
    print(f"   ✓ {joint_limit_dirs.shape}")
    
    # 2. 速度 - 位置耦合方向 (28 个)
    print("2. 生成速度 - 位置耦合方向 (28 个)...")
    coupling_dirs = []
    for i in range(14):
        # 位置 + 速度 同向 (推向限位)
        d1 = np.zeros(28)
        d1[i] = 1.0
        d1[14 + i] = 1.0
        coupling_dirs.append(d1)
        
        d2 = np.zeros(28)
        d2[i] = -1.0
        d2[14 + i] = -1.0
        coupling_dirs.append(d2)
    coupling_dirs = np.array(coupling_dirs)  # (28, 28)
    print(f"   ✓ {coupling_dirs.shape}")
    
    # 3. PCA 学习方向 (从危险数据)
    print(f"3. 收集危险状态用于 PCA ({N_PCA}个方向)...")
    danger_states = []
    
    # 扫描部分数据收集危险状态
    scenes_scanned = 0
    for scene in sorted(os.listdir(DATA_DIR)):
        scene_path = os.path.join(DATA_DIR, scene)
        if not os.path.isdir(scene_path): continue
        
        for fault in sorted(os.listdir(scene_path)):
            fault_path = os.path.join(scene_path, fault)
            if not os.path.isdir(fault_path): continue
            
            for ep_file in sorted(os.listdir(fault_path))[:3]:  # 每场景前 3 集
                if not ep_file.endswith('.jsonl'): continue
                
                try:
                    with open(os.path.join(fault_path, ep_file), 'r') as f:
                        for line in f:
                            step = json.loads(line)
                            if not step or "state" not in step: continue
                            
                            qpos = np.array(step["state"]["qpos"])
                            qvel = np.array(step["state"]["qvel"])
                            
                            if is_danger(qpos, qvel):
                                danger_states.append(np.concatenate([qpos, qvel]))
                except Exception as e:
                    continue
        
        scenes_scanned += 1
        if scenes_scanned >= 3:
            break
    
    print(f"   收集到 {len(danger_states)} 个危险状态")
    
    if len(danger_states) > 100:
        danger_states = np.array(danger_states)
        
        # 标准化后做 PCA
        danger_mean = np.mean(danger_states, axis=0)
        danger_std = np.std(danger_states, axis=0) + 1e-8
        danger_norm = (danger_states - danger_mean) / danger_std
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=N_PCA)
        pca.fit(danger_norm)
        pca_dirs = pca.components_  # (N_PCA, 28)
        
        # 反标准化
        pca_dirs = pca_dirs * danger_std[np.newaxis, :]
        pca_dirs = pca_dirs / np.linalg.norm(pca_dirs, axis=1, keepdims=True)
        
        print(f"   ✓ PCA 方向：{pca_dirs.shape}")
    else:
        print(f"   ⚠ 危险状态太少，用随机方向替代")
        np.random.seed(42)
        pca_dirs = np.random.randn(N_PCA, 28)
        pca_dirs = pca_dirs / np.linalg.norm(pca_dirs, axis=1, keepdims=True)
    
    # 4. 随机方向 (覆盖未建模方向)
    print(f"4. 生成随机方向 ({N_RANDOM}个)...")
    np.random.seed(42)
    random_dirs = np.random.randn(N_RANDOM, 28)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    print(f"   ✓ {random_dirs.shape}")
    
    # 5. 合并所有方向
    print("\n5. 合并所有方向...")
    all_dirs = np.vstack([
        joint_limit_dirs,   # (28, 28)
        coupling_dirs,      # (28, 28)
        pca_dirs,           # (N_PCA, 28)
        random_dirs,        # (N_RANDOM, 28)
    ])
    print(f"   合并后：{all_dirs.shape}")
    
    # 6. SVD 压缩到 K 个方向
    if all_dirs.shape[0] > SUPPORT_DIM:
        print(f"6. SVD 压缩到 {SUPPORT_DIM} 个方向...")
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=SUPPORT_DIM)
        svd.fit(all_dirs)
        support_directions = svd.components_  # (SUPPORT_DIM, 28)
        
        # 归一化
        support_directions = support_directions / np.linalg.norm(support_directions, axis=1, keepdims=True)
        print(f"   ✓ 压缩后：{support_directions.shape}")
    else:
        support_directions = all_dirs
    
    # 7. 保存
    output_path = os.path.join(OUTPUT_DIR, 'support_directions.npy')
    np.save(output_path, support_directions)
    
    print("\n" + "="*60)
    print("✓ 支撑方向生成完成！")
    print("="*60)
    print(f"输出：{output_path}")
    print(f"形状：{support_directions.shape}")
    print(f"方向归一化：✓")
    
    # 8. 可视化前 5 个方向
    print("\n前 5 个方向 (部分维度):")
    for i in range(5):
        d = support_directions[i]
        max_idx = np.argmax(np.abs(d))
        print(f"  方向{i}: 最大分量 [{max_idx}] = {d[max_idx]:.3f}")
    
    return support_directions

if __name__ == '__main__':
    generate_support_directions()
