#!/usr/bin/env python3
"""
准备 GRU 和 OOD 重训练数据
从故障轨迹中提取危险/安全样本
"""

import os, sys, json
import numpy as np
from datetime import datetime

# ========== 配置 ==========
DATA_DIR = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
OUTPUT_DIR = "/root/act/retrain_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JOINT_SAFE_RANGE = [
    (-1.85, 1.26), (-2.12, -0.96), (0.11, 1.57), (-0.16, 1.25),
    (-0.30, 3.51), (-2.43, 0.00), (0.08, 2.73), (-0.16, 1.26),
    (-2.26, -0.96), (0.11, 1.57), (-0.16, 1.25), (-1.85, 0.00),
    (0.00, 0.17), (0.08, 0.18),
]

# ========== GT 计算 ==========
def compute_ground_truth(qpos, qvel, threshold=0.03):
    violations = []
    for i in range(min(len(qpos), len(JOINT_SAFE_RANGE))):
        q_min, q_max = JOINT_SAFE_RANGE[i]
        margin = min(qpos[i] - q_min, q_max - qpos[i])
        joint_range = q_max - q_min
        if margin < -joint_range * threshold:
            violations.append(f"joint{i}_critical")
    if np.any(np.abs(qvel) > 0.6):
        violations.append("velocity_violation")
    return len(violations) > 0, violations

# ========== 数据收集 ==========
print("="*60)
print("【准备重训练数据】")
print("="*60)
print(f"数据目录：{DATA_DIR}")
print(f"输出目录：{OUTPUT_DIR}\n")

# 数据结构
gru_train_data = []  # (history_10, current_state, exceed_label, is_danger)
ood_safe_states = []  # 安全状态
ood_danger_states = []  # 危险前 1 秒状态 (预警窗口)

ep_count = 0
total_steps = 0
danger_steps = 0
warning_window = 50  # 危险前 50 步 = 1 秒 @ 50Hz

# 故障场景列表 (用于训练集)
fault_scenes_for_train = [
    'F1_visual_noise', 'F2_visual_occlusion', 'F3_state_drift',
    'F5_dynamics', 'F6_friction', 'F7_actuator_delay', 'F8_sensor_noise',
    'normal'
]

print("处理轨迹数据...")
for scene in sorted(os.listdir(DATA_DIR)):
    scene_path = os.path.join(DATA_DIR, scene)
    if not os.path.isdir(scene_path): continue
    
    for fault in sorted(os.listdir(scene_path)):
        fault_path = os.path.join(scene_path, fault)
        if not os.path.isdir(fault_path): continue
        
        # 只处理训练集场景
        if fault not in fault_scenes_for_train:
            continue
        
        for ep_file in sorted(os.listdir(fault_path)):
            if not ep_file.endswith('.jsonl'): continue
            ep_count += 1
            ep_path = os.path.join(fault_path, ep_file)
            
            # 读取整集数据
            episode_steps = []
            with open(ep_path, 'r') as f:
                for line in f:
                    step = json.loads(line)
                    if not step or "state" not in step: continue
                    episode_steps.append(step)
            
            if len(episode_steps) < 100:
                continue
            
            # 标记危险步
            danger_indices = []
            for i, step in enumerate(episode_steps):
                qpos = np.array(step["state"]["qpos"])
                qvel = np.array(step["state"]["qvel"])
                danger, _ = compute_ground_truth(qpos, qvel)
                if danger:
                    danger_indices.append(i)
            
            # 预警窗口：危险前 50 步
            warning_indices = set()
            for danger_idx in danger_indices:
                for t in range(max(0, danger_idx - warning_window), danger_idx):
                    warning_indices.add(t)
            
            # 提取数据
            trajectory_buffer = []
            for i, step in enumerate(episode_steps):
                total_steps += 1
                qpos = np.array(step["state"]["qpos"])
                qvel = np.array(step["state"]["qvel"])
                state = np.concatenate([qpos, qvel])  # 28 维
                
                danger, _ = compute_ground_truth(qpos, qvel)
                if danger:
                    danger_steps += 1
                
                # OOD 数据
                if i in warning_indices:
                    ood_danger_states.append(state)
                elif i not in danger_indices:
                    ood_safe_states.append(state)
                
                # GRU 数据 (需要 10 步历史)
                trajectory_buffer.append(state)
                if len(trajectory_buffer) >= 10:
                    history = np.array(trajectory_buffer[-10:])
                    # exceed 标签：危险=1, 安全=0
                    is_danger = 1 if (i in warning_indices or i in danger_indices) else 0
                    gru_train_data.append((history, state, is_danger))
                
                if len(trajectory_buffer) > 10:
                    trajectory_buffer.pop(0)
            
            if ep_count % 10 == 0:
                print(f"  已处理 {ep_count} 集...")

print(f"\n✓ 数据处理完成")
print(f"  总集数：{ep_count}")
print(f"  总步数：{total_steps}")
print(f"  危险步数：{danger_steps} ({danger_steps/total_steps*100:.1f}%)")
print(f"  GRU 训练样本：{len(gru_train_data)}")
print(f"  OOD 安全样本：{len(ood_safe_states)}")
print(f"  OOD 危险样本：{len(ood_danger_states)}")

# ========== 保存 GRU 训练数据 ==========
print("\n保存 GRU 训练数据...")
gru_histories = np.array([x[0] for x in gru_train_data])  # (N, 10, 28)
gru_states = np.array([x[1] for x in gru_train_data])      # (N, 28)
gru_labels = np.array([x[2] for x in gru_train_data])      # (N,)

np.save(os.path.join(OUTPUT_DIR, 'gru_train_history.npy'), gru_histories)
np.save(os.path.join(OUTPUT_DIR, 'gru_train_state.npy'), gru_states)
np.save(os.path.join(OUTPUT_DIR, 'gru_train_labels.npy'), gru_labels)
print(f"  ✓ gru_train_history.npy: {gru_histories.shape}")
print(f"  ✓ gru_train_state.npy: {gru_states.shape}")
print(f"  ✓ gru_train_labels.npy: {gru_labels.shape}")
print(f"  危险样本比例：{np.mean(gru_labels)*100:.1f}%")

# ========== 保存 OOD 统计量 ==========
print("\n计算 OOD 统计量...")
safe_arr = np.array(ood_safe_states)
danger_arr = np.array(ood_danger_states)

# 安全数据统计
mu_safe = np.mean(safe_arr, axis=0)
sigma_safe = np.cov(safe_arr.T)
sigma_inv_safe = np.linalg.pinv(sigma_safe)

# 危险数据统计
mu_danger = np.mean(danger_arr, axis=0)
sigma_danger = np.cov(danger_arr.T)
sigma_inv_danger = np.linalg.pinv(sigma_danger)

# 混合统计 (传统方法)
all_states = np.concatenate([safe_arr, danger_arr])
mu_all = np.mean(all_states, axis=0)
sigma_all = np.cov(all_states.T)
sigma_inv_all = np.linalg.pinv(sigma_all)

ood_stats = {
    'timestamp': datetime.now().isoformat(),
    'safe_samples': len(safe_arr),
    'danger_samples': len(danger_arr),
    'mu_safe': mu_safe.tolist(),
    'sigma_inv_safe': sigma_inv_safe.tolist(),
    'mu_danger': mu_danger.tolist(),
    'sigma_inv_danger': sigma_inv_danger.tolist(),
    'mu_all': mu_all.tolist(),
    'sigma_inv_all': sigma_inv_all.tolist(),
}

with open(os.path.join(OUTPUT_DIR, 'ood_stats_retrained.json'), 'w') as f:
    json.dump(ood_stats, f, indent=2)
print(f"  ✓ ood_stats_retrained.json")
print(f"  安全样本：{len(safe_arr)}, 危险样本：{len(danger_arr)}")

# ========== 验证新 OOD 统计量 ==========
print("\n验证新 OOD 统计量区分度...")
safe_scores = []
danger_scores = []

for state in safe_arr[:1000]:  # 采样 1000 个
    diff = state - mu_safe
    score = np.sqrt(np.abs(diff @ sigma_inv_safe @ diff))
    safe_scores.append(score)

for state in danger_arr[:1000]:
    diff = state - mu_safe  # 用安全分布计算
    score = np.sqrt(np.abs(diff @ sigma_inv_safe @ diff))
    danger_scores.append(score)

safe_p50 = np.percentile(safe_scores, 50)
safe_p95 = np.percentile(safe_scores, 95)
danger_p50 = np.percentile(danger_scores, 50)
danger_p95 = np.percentile(danger_scores, 95)

print(f"  安全样本 OOD 分数：p50={safe_p50:.1f}, p95={safe_p95:.1f}")
print(f"  危险样本 OOD 分数：p50={danger_p50:.1f}, p95={danger_p95:.1f}")
print(f"  区分度：{danger_p50 - safe_p50:.1f} (越大越好)")

# ========== 数据集划分 ==========
print("\n划分训练/验证/测试集...")
n_samples = len(gru_train_data)
indices = np.random.permutation(n_samples)
train_end = int(0.7 * n_samples)
val_end = int(0.85 * n_samples)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

np.save(os.path.join(OUTPUT_DIR, 'train_idx.npy'), train_idx)
np.save(os.path.join(OUTPUT_DIR, 'val_idx.npy'), val_idx)
np.save(os.path.join(OUTPUT_DIR, 'test_idx.npy'), test_idx)
print(f"  ✓ 训练集：{len(train_idx)} 样本 ({len(train_idx)/n_samples*100:.0f}%)")
print(f"  ✓ 验证集：{len(val_idx)} 样本 ({len(val_idx)/n_samples*100:.0f}%)")
print(f"  ✓ 测试集：{len(test_idx)} 样本 ({len(test_idx)/n_samples*100:.0f}%)")

print("\n" + "="*60)
print("✓ 数据准备完成！")
print("="*60)
print(f"\n下一步:")
print(f"  1. 训练 GRU: python3 train_gru_weighted.py")
print(f"  2. 使用新 OOD 统计量：{OUTPUT_DIR}/ood_stats_retrained.json")
print(f"  3. 评估：python3 analyze_r2_retrained.py")
