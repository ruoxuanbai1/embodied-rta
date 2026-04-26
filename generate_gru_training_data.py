#!/usr/bin/env python3
"""
生成 GRU 可达集预测模型的训练数据

使用蒙特卡洛方法生成可达集包络：
- 对每个样本，进行 30 次蒙特卡洛扰动模拟
- 支撑值 = 所有扰动轨迹的最大投影 (包络上界)

输出:
- gru_train_history.npy: (N, 10, 28)
- gru_train_support_target.npy: (N, 32)
- gru_train_danger_label.npy: (N,)
"""

import os, sys, json
import numpy as np
from datetime import datetime
import pickle

# ========== 配置 ==========
DATA_DIR = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
OUTPUT_DIR = "/root/act/retrain_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAST_HORIZON = 10     # 过去 10 步 (0.2 秒)
FUTURE_HORIZON = 50   # 未来 50 步 (1 秒)
N_MONTE_CARLO = 30    # 蒙特卡洛次数
CHECKPOINT_EVERY = 10 # 每 10 集保存一次 checkpoint

# 加载支撑方向
SUPPORT_PATH = os.path.join(OUTPUT_DIR, 'support_directions.npy')
if os.path.exists(SUPPORT_PATH):
    support_directions = np.load(SUPPORT_PATH)
    print(f"✓ 加载支撑方向：{support_directions.shape}")
else:
    raise FileNotFoundError("请先运行 generate_support_directions.py")

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

# ========== 蒙特卡洛可达集生成 ==========
def compute_support_monte_carlo(current_state, history, n_simulations=N_MONTE_CARLO):
    """
    用蒙特卡洛模拟生成可达集支撑值
    
    参数:
        current_state: (28,) 当前状态
        history: (10, 28) 历史状态 (用于估计噪声)
        n_simulations: 蒙特卡洛次数
    
    返回:
        support_targets: (K,) K 个支撑方向的支撑值
    """
    # 从历史估计状态噪声
    state_std = np.std(history, axis=0)  # (28,)
    state_std = np.clip(state_std, 0.01, 0.5)  # 限制范围
    
    all_trajectories = []
    
    for i in range(n_simulations):
        np.random.seed(i)
        
        # 扰动 1: 初始状态扰动
        x0 = current_state + np.random.randn(28) * state_std * 0.5
        
        # 扰动 2: 速度扰动
        vel_perturb = np.random.randn(14) * state_std[14:] * 0.3
        vel_perturb = np.clip(vel_perturb, -0.3, 0.3)
        
        # 简单动力学外推
        trajectory = np.zeros((FUTURE_HORIZON, 28))
        for t in range(FUTURE_HORIZON):
            dt = 0.02  # 50Hz
            # 位置 = 当前位置 + 速度 × 时间
            trajectory[t, :14] = x0[:14] + (x0[14:] + vel_perturb) * (t * dt)
            # 速度保持不变
            trajectory[t, 14:] = x0[14:] + vel_perturb
        
        # 限制在物理范围内
        for j in range(14):
            q_min, q_max = JOINT_SAFE_RANGE[j]
            trajectory[:, j] = np.clip(trajectory[:, j], q_min - 0.5, q_max + 0.5)
            trajectory[:, 14+j] = np.clip(trajectory[:, 14+j], -1.0, 1.0)
        
        all_trajectories.append(trajectory)
    
    # 计算支撑值 (所有轨迹的包络)
    all_projections = []
    for traj in all_trajectories:
        # (K, 28) @ (28, 50) = (K, 50)
        proj = support_directions @ traj.T
        all_projections.append(proj)
    
    # (n_simulations, K, 50)
    all_projections = np.stack(all_projections)
    
    # 支撑值 = 所有模拟、所有时间步的最大值
    support_targets = np.max(all_projections, axis=(0, 2))  # (K,)
    
    return support_targets

# ========== 主数据生成流程 ==========
def generate_training_data():
    print("="*60)
    print("【生成 GRU 训练数据 - 蒙特卡洛版】")
    print("="*60)
    print(f"过去窗口：{PAST_HORIZON} 步")
    print(f"未来窗口：{FUTURE_HORIZON} 步")
    print(f"蒙特卡洛：{N_MONTE_CARLO} 次")
    print(f"支撑维度：{support_directions.shape[0]}")
    print()
    
    # 检查 checkpoint
    checkpoint_path = os.path.join(OUTPUT_DIR, 'data_generation_checkpoint.pkl')
    start_ep = 0
    train_history = []
    train_support_target = []
    train_danger_label = []
    
    if os.path.exists(checkpoint_path):
        print(f"发现 checkpoint，是否恢复？(y/n)")
        # 自动恢复
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        start_ep = checkpoint['ep_count']
        train_history = checkpoint['train_history']
        train_support_target = checkpoint['train_support_target']
        train_danger_label = checkpoint['train_danger_label']
        print(f"✓ 恢复到第 {start_ep} 集，已有 {len(train_history)} 样本\n")
    
    # 遍历所有轨迹
    print("处理轨迹数据...")
    ep_count = 0
    sample_count = len(train_history)
    
    all_episodes = []
    for scene in sorted(os.listdir(DATA_DIR)):
        scene_path = os.path.join(DATA_DIR, scene)
        if not os.path.isdir(scene_path): continue
        
        for fault in sorted(os.listdir(scene_path)):
            fault_path = os.path.join(scene_path, fault)
            if not os.path.isdir(fault_path): continue
            
            for ep_file in sorted(os.listdir(fault_path)):
                if not ep_file.endswith('.jsonl'): continue
                all_episodes.append(os.path.join(fault_path, ep_file))
    
    print(f"总集数：{len(all_episodes)}")
    print(f"从第 {start_ep + 1} 集开始\n")
    
    for ep_idx, ep_path in enumerate(all_episodes):
        if ep_idx < start_ep:
            continue
        
        ep_count = ep_idx + 1
        
        try:
            # 读取整集轨迹
            episode_states = []
            with open(ep_path, 'r') as f:
                for line in f:
                    step = json.loads(line)
                    if not step or "state" not in step: continue
                    
                    qpos = np.array(step["state"]["qpos"])
                    qvel = np.array(step["state"]["qvel"])
                    state = np.concatenate([qpos, qvel])
                    episode_states.append(state)
            
            episode_states = np.array(episode_states)
            
            if len(episode_states) < PAST_HORIZON + FUTURE_HORIZON:
                continue
            
            # 滑动窗口生成样本
            for t in range(PAST_HORIZON, len(episode_states) - FUTURE_HORIZON):
                history = episode_states[t-PAST_HORIZON:t]  # (10, 28)
                current_state = episode_states[t]            # (28,)
                future = episode_states[t:t+FUTURE_HORIZON]  # (50, 28)
                
                # 蒙特卡洛生成支撑值
                support_target = compute_support_monte_carlo(current_state, history)
                
                # 危险标签
                danger_label = 0.0
                for step_state in future:
                    qpos = step_state[:14]
                    qvel = step_state[14:]
                    if is_danger(qpos, qvel):
                        danger_label = 1.0
                        break
                
                train_history.append(history)
                train_support_target.append(support_target)
                train_danger_label.append(danger_label)
                
                sample_count += 1
            
            # 定期保存 checkpoint
            if ep_count % CHECKPOINT_EVERY == 0:
                checkpoint = {
                    'ep_count': ep_count,
                    'train_history': train_history,
                    'train_support_target': train_support_target,
                    'train_danger_label': train_danger_label,
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                
                danger_ratio = np.mean(train_danger_label) * 100
                print(f"  [{ep_count}/{len(all_episodes)}] 样本：{sample_count}, 危险比例：{danger_ratio:.1f}%")
        
        except Exception as e:
            print(f"  ⚠ 处理 {ep_path} 时出错：{e}")
            continue
    
    print(f"\n✓ 数据处理完成")
    print(f"  总集数：{ep_count}")
    print(f"  总样本：{sample_count}")
    print(f"  危险样本比例：{np.mean(train_danger_label)*100:.1f}%")
    
    # 转换为 numpy 数组
    print("\n转换为 numpy 数组...")
    train_history = np.array(train_history, dtype=np.float32)
    train_support_target = np.array(train_support_target, dtype=np.float32)
    train_danger_label = np.array(train_danger_label, dtype=np.float32)
    
    print(f"  history: {train_history.shape}")
    print(f"  support_target: {train_support_target.shape}")
    print(f"  danger_label: {train_danger_label.shape}")
    
    # 保存
    print("\n保存数据...")
    np.save(os.path.join(OUTPUT_DIR, 'gru_train_history.npy'), train_history)
    np.save(os.path.join(OUTPUT_DIR, 'gru_train_support_target.npy'), train_support_target)
    np.save(os.path.join(OUTPUT_DIR, 'gru_train_danger_label.npy'), train_danger_label)
    
    # 划分数据集
    print("\n划分训练/验证/测试集...")
    n_samples = len(train_history)
    indices = np.random.permutation(n_samples)
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    np.save(os.path.join(OUTPUT_DIR, 'train_idx.npy'), train_idx)
    np.save(os.path.join(OUTPUT_DIR, 'val_idx.npy'), val_idx)
    np.save(os.path.join(OUTPUT_DIR, 'test_idx.npy'), test_idx)
    
    print(f"  训练集：{len(train_idx)} 样本 ({len(train_idx)/n_samples*100:.0f}%)")
    print(f"  验证集：{len(val_idx)} 样本 ({len(val_idx)/n_samples*100:.0f}%)")
    print(f"  测试集：{len(test_idx)} 样本 ({len(test_idx)/n_samples*100:.0f}%)")
    
    # 清理 checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("\n" + "="*60)
    print("✓ 训练数据生成完成！")
    print("="*60)
    print(f"\n输出文件:")
    print(f"  {OUTPUT_DIR}/gru_train_history.npy")
    print(f"  {OUTPUT_DIR}/gru_train_support_target.npy")
    print(f"  {OUTPUT_DIR}/gru_train_danger_label.npy")
    print(f"  {OUTPUT_DIR}/train_idx.npy")
    print(f"  {OUTPUT_DIR}/val_idx.npy")
    print(f"  {OUTPUT_DIR}/test_idx.npy")

if __name__ == '__main__':
    generate_training_data()
