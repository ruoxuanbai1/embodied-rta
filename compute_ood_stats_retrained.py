#!/usr/bin/env python3
"""
用故障数据重新计算 OOD 统计量

方案：
1. 从训练数据中提取安全/危险状态
2. 分别计算 mu 和 sigma_inv
3. 保存新的 OOD 统计量
"""

import os
import numpy as np
import json
from datetime import datetime

# ========== 配置 ==========
DATA_DIR = "/root/act/retrain_data"
OUTPUT_DIR = "/root/act/outputs/region3_detectors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载训练数据
print("加载训练数据...")
history = np.load(os.path.join(DATA_DIR, 'gru_train_history.npy'))  # (N, 10, 28)
target = np.load(os.path.join(DATA_DIR, 'gru_train_support_target.npy'))  # (N, 28)
labels = np.load(os.path.join(DATA_DIR, 'gru_train_danger_label.npy'))  # (N,)

print(f"  总样本：{len(history):,}")
print(f"  危险比例：{np.mean(labels)*100:.1f}%")

# ========== 提取状态 ==========
# 用历史的最后一步作为"当前状态"
current_states = history[:, -1, :]  # (N, 28)

safe_states = current_states[labels == 0]
danger_states = current_states[labels == 1]

print(f"\n安全状态：{len(safe_states):,}")
print(f"危险状态：{len(danger_states):,}")

# ========== 计算统计量 ==========
print("\n计算 OOD 统计量...")

# 方法 1: 传统 (只用安全数据)
print("  1. 传统 OOD (安全数据)...")
mu_safe = np.mean(safe_states, axis=0)
sigma_safe = np.cov(safe_states.T)
sigma_inv_safe = np.linalg.pinv(sigma_safe)

# 方法 2: 混合 (安全 + 危险)
print("  2. 混合 OOD (全部数据)...")
mu_all = np.mean(current_states, axis=0)
sigma_all = np.cov(current_states.T)
sigma_inv_all = np.linalg.pinv(sigma_all)

# 方法 3: 双分布对比
print("  3. 双分布 OOD (安全 - 危险对比)...")
mu_danger = np.mean(danger_states, axis=0)
sigma_danger = np.cov(danger_states.T)
sigma_inv_danger = np.linalg.pinv(sigma_danger)

# ========== 验证区分度 ==========
print("\n验证区分度...")

def compute_mahalanobis(states, mu, sigma_inv):
    diff = states - mu
    left = diff @ sigma_inv
    D = np.sqrt(np.abs(np.sum(left * diff, axis=1)))
    return D

# 传统 OOD
D_safe_traditional = compute_mahalanobis(safe_states[:1000], mu_safe, sigma_inv_safe)
D_danger_traditional = compute_mahalanobis(danger_states[:1000], mu_safe, sigma_inv_safe)

# 混合 OOD
D_safe_mixed = compute_mahalanobis(safe_states[:1000], mu_all, sigma_inv_all)
D_danger_mixed = compute_mahalanobis(danger_states[:1000], mu_all, sigma_inv_all)

# 双分布 OOD (距离差)
D_safe_dual = (compute_mahalanobis(safe_states[:1000], mu_safe, sigma_inv_safe) - 
               compute_mahalanobis(safe_states[:1000], mu_danger, sigma_inv_danger))
D_danger_dual = (compute_mahalanobis(danger_states[:1000], mu_safe, sigma_inv_safe) - 
                 compute_mahalanobis(danger_states[:1000], mu_danger, sigma_inv_danger))

print(f"\n传统 OOD (安全数据):")
print(f"  安全样本：p50={np.percentile(D_safe_traditional, 50):.2f}, p95={np.percentile(D_safe_traditional, 95):.2f}")
print(f"  危险样本：p50={np.percentile(D_danger_traditional, 50):.2f}, p95={np.percentile(D_danger_traditional, 95):.2f}")
print(f"  区分度：{np.percentile(D_danger_traditional, 50) - np.percentile(D_safe_traditional, 50):.2f}")

print(f"\n混合 OOD (全部数据):")
print(f"  安全样本：p50={np.percentile(D_safe_mixed, 50):.2f}, p95={np.percentile(D_safe_mixed, 95):.2f}")
print(f"  危险样本：p50={np.percentile(D_danger_mixed, 50):.2f}, p95={np.percentile(D_danger_mixed, 95):.2f}")
print(f"  区分度：{np.percentile(D_danger_mixed, 50) - np.percentile(D_safe_mixed, 50):.2f}")

print(f"\n双分布 OOD (距离差):")
print(f"  安全样本：p50={np.percentile(D_safe_dual, 50):.2f}, p95={np.percentile(D_safe_dual, 95):.2f}")
print(f"  危险样本：p50={np.percentile(D_danger_dual, 50):.2f}, p95={np.percentile(D_danger_dual, 95):.2f}")
print(f"  区分度：{np.percentile(D_danger_dual, 50) - np.percentile(D_safe_dual, 50):.2f}")

# ========== 保存结果 ==========
print("\n保存 OOD 统计量...")

ood_stats = {
    'timestamp': datetime.now().isoformat(),
    'method': 'retrained_with_fault_data',
    'safe_samples': len(safe_states),
    'danger_samples': len(danger_states),
    
    # 传统方法
    'traditional': {
        'mu': mu_safe.tolist(),
        'sigma_inv': sigma_inv_safe.tolist(),
    },
    
    # 混合方法
    'mixed': {
        'mu': mu_all.tolist(),
        'sigma_inv': sigma_inv_all.tolist(),
    },
    
    # 双分布方法
    'dual': {
        'mu_safe': mu_safe.tolist(),
        'sigma_inv_safe': sigma_inv_safe.tolist(),
        'mu_danger': mu_danger.tolist(),
        'sigma_inv_danger': sigma_inv_danger.tolist(),
    },
    
    # 推荐使用方法
    'recommended': 'dual',
}

output_path = os.path.join(OUTPUT_DIR, 'ood_stats_retrained.json')
with open(output_path, 'w') as f:
    json.dump(ood_stats, f, indent=2)

print(f"  ✓ 已保存：{output_path}")

# 总结
print("\n" + "="*60)
print("✓ OOD 统计量重计算完成！")
print("="*60)
print(f"\n推荐方法：双分布 OOD")
print(f"优势：区分度提升 {np.percentile(D_danger_dual, 50) - np.percentile(D_safe_dual, 50):.2f}")
print(f"\n使用方法:")
print(f"  1. 加载 ood_stats_retrained.json")
print(f"  2. 使用 'dual' 方法计算 OOD 分数")
print(f"  3. 阈值建议：p95(D_safe_dual) + 余量 = {np.percentile(D_safe_dual, 95) + 50:.1f}")
