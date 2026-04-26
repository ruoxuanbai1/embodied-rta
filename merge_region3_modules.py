#!/usr/bin/env python3
"""
merge_region3_modules.py - 合并 Region 3 三个模块

输入:
- outputs/region3_activation/ (激活链路)
- outputs/region3_gradient_ood/ (梯度贡献 + OOD)

输出:
- outputs/region3_complete/ (完整 Region 3 检测器)
"""

import json
import os
import shutil


def merge_region3():
    print("=" * 60)
    print("合并 Region 3 三个模块")
    print("=" * 60)
    
    activation_dir = './outputs/region3_activation'
    gradient_ood_dir = './outputs/region3_gradient_ood'
    output_dir = './outputs/region3_complete'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制激活模块文件
    print("\n📦 复制激活链路模块...")
    for f in ['kmeans_activation.pkl', 'activation_masks.json', 'activation_summary.json']:
        src = os.path.join(activation_dir, f)
        dst = os.path.join(output_dir, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  ✓ {f}")
        else:
            print(f"  ⚠️ {f} 不存在")
    
    # 复制梯度 + OOD 模块文件
    print("\n📦 复制梯度 + OOD 模块...")
    for f in ['kmeans_gradient_ood.pkl', 'F_legal_profiles.json', 'ood_stats.json', 'gradient_ood_summary.json']:
        src = os.path.join(gradient_ood_dir, f)
        dst = os.path.join(output_dir, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  ✓ {f}")
        else:
            print(f"  ⚠️ {f} 不存在")
    
    # 创建合并摘要
    print("\n📝 创建合并摘要...")
    
    # 读取各模块摘要
    with open(os.path.join(activation_dir, 'activation_summary.json')) as f:
        act_summary = json.load(f)
    
    with open(os.path.join(gradient_ood_dir, 'gradient_ood_summary.json')) as f:
        grad_summary = json.load(f)
    
    merged_summary = {
        'name': 'Region 3 Complete Detector',
        'modules': {
            'activation_link': {
                'K': act_summary['K'],
                'num_episodes': act_summary.get('num_normal', act_summary.get('num_episodes', 50)),
                'layers': act_summary['layers'],
                'dim_per_layer': act_summary['dim_per_layer'],
            },
            'gradient_contribution': {
                'K': grad_summary['K'],
                'num_episodes': grad_summary.get('num_episodes', 50),
            },
            'ood_detection': {
                'threshold': grad_summary['ood_threshold'],
            }
        },
        'weights': {
            'activation': 0.35,
            'gradient': 0.35,
            'ood': 0.30,
        },
        'risk_thresholds': {
            'GREEN': 0.2,
            'YELLOW': 0.4,
            'ORANGE': 0.6,
            'RED': 1.0,
        }
    }
    
    with open(os.path.join(output_dir, 'region3_complete_summary.json'), 'w') as f:
        json.dump(merged_summary, f, indent=2)
    print(f"  ✓ region3_complete_summary.json")
    
    print("\n" + "=" * 60)
    print("✅ Region 3 完整检测器合并完成!")
    print(f"输出目录：{output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    merge_region3()
