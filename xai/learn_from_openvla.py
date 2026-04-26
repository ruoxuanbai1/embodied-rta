#!/usr/bin/env python3
"""
Region 3: 从 OpenVLA 真实数据学习激活掩码和预警阈值

输入:
- OpenVLA 正常轨迹 (images + states)
- OpenVLA 故障轨迹 (images + states)

输出:
- xai/openvla_reference_stats.pkl (正常激活统计)
- xai/openvla_activation_masks.pkl (掩码库 - 关键特征组合)
- xai/openvla_thresholds.pkl (预警阈值 - ROC 曲线优化)
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

# 添加路径
PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'agents'))
sys.path.insert(0, str(PROJECT_ROOT / 'xai'))

print("="*80)
print("Region 3: 从 OpenVLA 真实数据学习激活掩码和阈值")
print("="*80)

# ============== 加载模型和环境 ==============

print("\n[1/6] 加载 OpenVLA-7B 模型...")
from openvla_agent import OpenVLAAgent
from multi_layer_activation import MultiLayerActivationHook

vla = OpenVLAAgent(model_path='/data/models/openvla-7b', device='cuda')
print("✅ OpenVLA-7B 加载成功")

print("\n[2/6] 注册激活钩子...")
hook_manager = MultiLayerActivationHook(vla.model)
print(f"✅ 钩子注册成功 ({len(hook_manager.hooks)} 个)")

print("\n[3/6] 加载轨迹数据...")
with open(PROJECT_ROOT / 'reachability' / 'openvla_trajectories_normal.pkl', 'rb') as f:
    normal_trajs = pickle.load(f)

with open(PROJECT_ROOT / 'reachability' / 'openvla_trajectories_faults.pkl', 'rb') as f:
    fault_trajs = pickle.load(f)

print(f"  正常轨迹：{len(normal_trajs)} 条")
print(f"  故障轨迹：{len(fault_trajs)} 条")

# ============== 收集激活数据 ==============

print("\n[4/6] 收集激活数据...")

# 正常激活
normal_activations = defaultdict(list)
normal_count = 0

# 故障激活
fault_activations = defaultdict(list)
fault_count = 0

# 只选择成功的轨迹 (避免异常数据污染参考统计)
successful_normal = [t for t in normal_trajs if t['success']]
print(f"  使用成功轨迹：{len(successful_normal)} 条正常")

for traj_idx, traj in enumerate(successful_normal):
    for step_idx in range(0, len(traj['images']), 10):  # 每 10 帧采样一次
        image = traj['images'][step_idx]
        
        # 清空并推理
        hook_manager.clear_activations()
        _ = vla.get_action(image, "navigate to the goal")
        
        # 保存激活
        for name, act in hook_manager.get_all_activations().items():
            normal_activations[name].append(act.cpu().numpy())
        
        normal_count += 1
        
        if (traj_idx + 1) % 20 == 0:
            print(f"    正常样本：{normal_count}")

# 故障轨迹激活
for traj_idx, traj in enumerate(fault_trajs):
    for step_idx in range(0, len(traj['images']), 5):  # 更密集采样
        image = traj['images'][step_idx]
        
        hook_manager.clear_activations()
        _ = vla.get_action(image, "navigate to the goal")
        
        for name, act in hook_manager.get_all_activations().items():
            fault_activations[name].append(act.cpu().numpy())
        
        fault_count += 1

print(f"  ✅ 收集完成：{normal_count} 正常，{fault_count} 故障")

# ============== 计算参考统计 ==============

print("\n[5/6] 计算参考统计...")

reference_stats = {}

for layer_name, acts in normal_activations.items():
    # 展平所有激活
    acts_flat = np.concatenate([a.flatten() for a in acts], axis=0)
    
    # 计算统计量
    mean = np.mean(acts_flat, axis=0)
    std = np.std(acts_flat, axis=0) + 1e-8  # 避免除零
    
    # 百分位数 (用于异常检测)
    p05 = np.percentile(acts_flat, 5, axis=0)
    p95 = np.percentile(acts_flat, 95, axis=0)
    
    # 协方差矩阵 (用于马氏距离，只取前 1000 维避免内存爆炸)
    if acts_flat.shape[1] > 1000:
        cov = np.cov(acts_flat[:, :1000].T)
    else:
        cov = np.cov(acts_flat.T)
    
    reference_stats[layer_name] = {
        'mean': mean,
        'std': std,
        'p05': p05,
        'p95': p95,
        'cov': cov,
        'n_samples': len(acts),
    }

print(f"  ✅ 计算完成：{len(reference_stats)} 层")

# ============== 学习掩码库 (关键特征组合) ==============

print("\n[6/6] 学习掩码库...")

# 使用互信息特征选择找出关键神经元
try:
    from sklearn.feature_selection import mutual_info_classif
    HAS_SKLEARN = True
except ImportError:
    print("  ⚠️ scikit-learn 未安装，使用简化方法 (方差选择)")
    HAS_SKLEARN = False

activation_masks = {}

for layer_name in reference_stats.keys():
    normal_acts = normal_activations[layer_name]
    fault_acts = fault_activations.get(layer_name, [])
    
    if len(fault_acts) == 0:
        continue
    
    # 展平
    normal_flat = np.concatenate([a.flatten() for a in normal_acts], axis=0)
    fault_flat = np.concatenate([a.flatten() for a in fault_acts], axis=0)
    
    # 限制维度 (避免内存问题)
    max_dim = min(normal_flat.shape[1], 5000)
    normal_flat = normal_flat[:, :max_dim]
    fault_flat = fault_flat[:, :max_dim]
    
    # 合并数据
    X = np.concatenate([normal_flat, fault_flat], axis=0)
    y = np.concatenate([np.zeros(len(normal_flat)), np.ones(len(fault_flat))])
    
    if HAS_SKLEARN:
        # 互信息特征选择
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        # 简化：使用方差 + 均值差异
        mean_diff = np.abs(normal_flat.mean(0) - fault_flat.mean(0))
        std_normal = normal_flat.std(0) + 1e-8
        mi_scores = mean_diff / std_normal  # 效应量 (Cohen's d 简化)
    
    # 选择 Top-K 关键神经元
    K = min(50, max_dim)  # Top 50
    top_k_indices = np.argsort(mi_scores)[-K:]
    
    # 保存掩码
    activation_masks[layer_name] = {
        'key_neurons': top_k_indices.tolist(),
        'scores': mi_scores[top_k_indices].tolist(),
        'normal_mean': normal_flat[:, top_k_indices].mean(0).tolist(),
        'normal_std': normal_flat[:, top_k_indices].std(0).tolist(),
        'fault_mean': fault_flat[:, top_k_indices].mean(0).tolist(),
        'n_normal': len(normal_flat),
        'n_fault': len(fault_flat),
    }

print(f"  ✅ 掩码库学习完成：{len(activation_masks)} 层")

# ============== 学习预警阈值 (ROC 曲线) ==============

print("\n计算预警阈值...")

# 计算正常和故障的风险分数
def compute_risk_score(acts_dict, ref_stats, masks):
    """计算单样本风险分数"""
    scores = []
    
    for layer_name, act in acts_dict.items():
        if layer_name not in ref_stats:
            continue
        
        ref = ref_stats[layer_name]
        act_flat = act.flatten()[:len(ref['mean'])]
        
        # Z-score 异常
        z_scores = np.abs((act_flat - ref['mean']) / ref['std'])
        layer_score = np.mean(z_scores > 3.0)  # 超过 3σ 的比例
        
        scores.append(layer_score)
    
    return np.mean(scores) if scores else 0.0

# 正常风险分数分布
normal_risks = []
for traj in successful_normal[:50]:  # 子集加速
    for step_idx in range(0, len(traj['images']), 20):
        image = traj['images'][step_idx]
        hook_manager.clear_activations()
        _ = vla.get_action(image, "navigate")
        
        risk = compute_risk_score(hook_manager.get_all_activations(), reference_stats, activation_masks)
        normal_risks.append(risk)

# 故障风险分数分布
fault_risks = []
for traj in fault_trajs[:50]:
    for step_idx in range(0, len(traj['images']), 10):
        image = traj['images'][step_idx]
        hook_manager.clear_activations()
        _ = vla.get_action(image, "navigate")
        
        risk = compute_risk_score(hook_manager.get_all_activations(), reference_stats, activation_masks)
        fault_risks.append(risk)

print(f"  正常风险分数：{len(normal_risks)} 样本, mean={np.mean(normal_risks):.3f}, std={np.std(normal_risks):.3f}")
print(f"  故障风险分数：{len(fault_risks)} 样本, mean={np.mean(fault_risks):.3f}, std={np.std(fault_risks):.3f}")

# ROC 曲线分析
try:
    from sklearn.metrics import roc_curve, auc
    y_true = np.concatenate([np.zeros(len(normal_risks)), np.ones(len(fault_risks))])
    y_scores = np.concatenate([normal_risks, fault_risks])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 最优阈值 (Youden's J)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n  ROC AUC: {roc_auc:.3f}")
    print(f"  最优阈值：{optimal_threshold:.3f}")
    print(f"  对应 TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f}")
    
    threshold_info = {
        'optimal_threshold': float(optimal_threshold),
        'tpr': float(tpr[optimal_idx]),
        'fpr': float(fpr[optimal_idx]),
        'auc': float(roc_auc),
        'thresholds': thresholds.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
    }
except ImportError:
    # 简化：使用 3σ 原则
    optimal_threshold = np.mean(normal_risks) + 3 * np.std(normal_risks)
    print(f"\n  简化阈值 (3σ): {optimal_threshold:.3f}")
    
    threshold_info = {
        'optimal_threshold': float(optimal_threshold),
        'method': '3_sigma',
    }

# ============== 保存结果 ==============

print("\n保存结果...")

# 参考统计
output_ref = PROJECT_ROOT / 'xai' / 'openvla_reference_stats.pkl'
with open(output_ref, 'wb') as f:
    pickle.dump(reference_stats, f)
print(f"✅ 参考统计：{output_ref} ({len(reference_stats)} 层)")

# 掩码库
output_masks = PROJECT_ROOT / 'xai' / 'openvla_activation_masks.pkl'
with open(output_masks, 'wb') as f:
    pickle.dump(activation_masks, f)
print(f"✅ 掩码库：{output_masks} ({len(activation_masks)} 层)")

# 阈值
output_thresholds = PROJECT_ROOT / 'xai' / 'openvla_thresholds.pkl'
with open(output_thresholds, 'wb') as f:
    pickle.dump(threshold_info, f)
print(f"✅ 阈值：{output_thresholds}")

# ============== 总结 ==============

print("\n" + "="*80)
print("Region 3 学习完成!")
print("="*80)
print(f"""
数据来源:
  - 正常激活：{normal_count} 样本
  - 故障激活：{fault_count} 样本

输出文件:
  1. openvla_reference_stats.pkl
     - 每层激活的 mean, std, cov, 百分位数
     - 用于 OOD 检测 (马氏距离)
  
  2. openvla_activation_masks.pkl
     - 每层 Top 50 关键神经元
     - 用于掩码匹配检测
  
  3. openvla_thresholds.pkl
     - 最优预警阈值：{optimal_threshold:.3f}
     - ROC AUC: {threshold_info.get('auc', 'N/A')}

下一步:
  - 更新 xai/multi_layer_activation.py 使用学习到的统计和掩码
  - 运行完整试验验证检测效果
""")
print("="*80)
