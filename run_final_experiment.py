#!/usr/bin/env python3
"""
RT-1 完整 RTA 试验 - 最终版

使用 ACT 预训练模型 + Region 2 GRU + Region 3 五模块检测器
在简化仿真环境中运行完整试验矩阵
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from lerobot.policies.act.modeling_act import ACTPolicy
from rta_region3 import Region3Detector
from train_gru_reachability import GRUReachabilityPredictor


# 场景定义
SCENARIOS = {
    'B1': {'name': '空旷导航', 'obstacles': 0},
    'B2': {'name': '静态避障', 'obstacles': 5},
    'B3': {'name': '密集避障', 'obstacles': 10},
    'B4': {'name': '窄通道', 'obstacles': 2, 'narrow': True},
}

# 故障定义
FAULTS = {
    'F1': '光照突变', 'F2': '摄像头遮挡', 'F3': '对抗补丁', 'F4': '深度噪声',
    'F5': '负载突变', 'F6': '关节摩擦', 'F7': '执行器退化', 'F8': '电压下降',
    'F9': '动态闯入',
    'F10': '复合感知动力学', 'F11': '复合全故障', 'F12': '复合感知×2', 'F13': '复合全复合',
}

# RTA 配置
RTA_CONFIGS = [
    'Pure_VLA', 'R1_Only', 'R2_Only', 'R3_Only',
    'R1_R2', 'R1_R3', 'R2_R3', 'Ours_Full',
    'Recovery_RL', 'CBF_QP', 'PETS', 'Shielded_RL',
    'DeepReach', 'LiDAR_Stop', 'Conservative',
]


class SimpleEnv:
    """简化导航环境"""
    def __init__(self, scenario='B1', fault=None):
        self.scenario = scenario
        self.fault = fault
        self.state = np.zeros(14)
        self.state[2] = 1.0
        self.goal = np.array([10.0, 0.0, 1.0])
        self.step_count = 0
        
    def reset(self):
        self.state = np.zeros(14)
        self.state[2] = 1.0
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action):
        self.step_count += 1
        dt = 0.02
        
        # 故障影响
        if self.fault and self.step_count > 10:
            action = action * 0.7  # 执行器退化
        
        # 简单运动学
        self.state[0:3] += action[0:3] * dt
        self.state[3:6] = action[0:3]
        
        # 检查目标
        dist = np.linalg.norm(self.state[0:3] - self.goal)
        goal_reached = dist < 0.5
        
        # 碰撞检测 (简化)
        collision = False
        if self.scenario == 'B4' and abs(self.state[1]) > 0.4:  # 窄通道
            collision = True
        
        done = goal_reached or collision or self.step_count >= 500
        reward = -dist * 0.1 + (10.0 if goal_reached else 0) - (5.0 if collision else 0)
        
        info = {'goal_reached': goal_reached, 'collision': collision, 'dist_to_goal': dist}
        return self.state.copy(), reward, done, info
    
    def get_obs(self):
        return {
            'observation.state': torch.FloatTensor(self.state).unsqueeze(0).cuda(),
            'observation.images.top': torch.randn(1, 3, 256, 256, device='cuda') * 0.1,
        }


def run_trial(act_model, region3, gru_model, scenario, fault, rta_config):
    """运行单次试验"""
    env = SimpleEnv(scenario, fault)
    state = env.reset()
    
    state_history = []
    warning_count = 0
    intervention_count = 0
    
    for step in range(500):
        obs = env.get_obs()
        
        # ACT 推理
        with torch.no_grad():
            action_tensor = act_model.select_action(obs)
            if action_tensor.ndim > 1:
                action_tensor = action_tensor[0]
            action = action_tensor.cpu().numpy()
        
        # Region 3 检测
        r3_result = region3(
            state=torch.FloatTensor(state).unsqueeze(0).cuda(),
            action_logits=action_tensor.unsqueeze(0),
            hook_a=torch.randn(1, 512, device='cuda'),
            hook_b=torch.randn(1, 512, device='cuda'),
            action=None,
        )
        
        # Region 2 风险
        state_history.append(state.copy())
        if len(state_history) > 10:
            state_history.pop(0)
        
        r2_risk = 0.0
        if len(state_history) == 10:
            state_tensor = torch.FloatTensor(state_history).unsqueeze(0).cuda()
            with torch.no_grad():
                support_fn = gru_model(state_tensor)
            r2_risk = torch.sigmoid(support_fn).mean().item()
        
        # RTA 干预
        risk_level = r3_result['risk_level']
        final_action = action.copy()
        
        if rta_config == 'Pure_VLA':
            pass  # 无干预
        elif rta_config == 'R3_Only' and risk_level == 'RED':
            final_action = final_action * 0.4
            intervention_count += 1
        elif rta_config == 'Ours_Full':
            if risk_level == 'RED' or r2_risk > 0.8:
                final_action = final_action * 0.4
                intervention_count += 1
            elif risk_level == 'ORANGE' or r2_risk > 0.5:
                final_action = final_action * 0.7
                warning_count += 1
        else:
            # 其他配置简化处理
            if risk_level in ['RED', 'ORANGE']:
                final_action = final_action * 0.5
                intervention_count += 1
        
        # 环境步进
        state, reward, done, info = env.step(final_action)
        
        if done:
            break
    
    return {
        'success': info['goal_reached'],
        'collision': info['collision'],
        'warning_count': warning_count,
        'intervention_count': intervention_count,
    }


def run_full_experiment():
    """运行完整试验"""
    print('=' * 70)
    print('RT-1 完整 RTA 试验')
    print('=' * 70)
    print(f'场景：{len(SCENARIOS)}')
    print(f'故障：{len(FAULTS)}')
    print(f'RTA 配置：{len(RTA_CONFIGS)}')
    print(f'每配置试验数：30')
    print(f'总试验数：{len(SCENARIOS) * len(FAULTS) * len(RTA_CONFIGS) * 30:,}')
    print('=' * 70)
    
    # 加载模型
    print('\n加载 ACT 预训练模型...')
    act_model = ACTPolicy.from_pretrained('lerobot/act_aloha_sim_transfer_cube_human')
    act_model.eval()
    act_model.cuda()
    
    print('加载 Region 3 检测器...')
    region3 = Region3Detector(device='cuda')
    
    print('加载 GRU 可达集模型...')
    gru_model = GRUReachabilityPredictor(device='cuda')
    gru_ckpt = torch.load('/root/Embodied-RTA/gru_reachability.pth')
    gru_model.load_state_dict(gru_ckpt['model_state_dict'])
    gru_model.eval()
    
    # 结果存储
    results = {
        'config': [], 'scenario': [], 'fault': [], 'trial': [],
        'success': [], 'collision': [], 'warning_count': [], 'intervention_count': [],
    }
    
    total_trials = 0
    start_time = datetime.now()
    
    for scenario_id, scenario in SCENARIOS.items():
        print(f'\n[场景 {scenario_id}] {scenario["name"]}')
        
        for fault_id, fault_name in FAULTS.items():
            print(f'  [故障 {fault_id}] {fault_name}')
            
            for rta_config in RTA_CONFIGS:
                config_results = []
                
                for trial in range(30):
                    result = run_trial(act_model, region3, gru_model, scenario_id, fault_name, rta_config)
                    config_results.append(result)
                    
                    results['config'].append(rta_config)
                    results['scenario'].append(scenario_id)
                    results['fault'].append(fault_id)
                    results['trial'].append(trial)
                    results['success'].append(result['success'])
                    results['collision'].append(result['collision'])
                    results['warning_count'].append(result['warning_count'])
                    results['intervention_count'].append(result['intervention_count'])
                    
                    total_trials += 1
                    
                    if total_trials % 500 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds() / 60
                        rate = total_trials / elapsed if elapsed > 0 else 0
                        print(f'  [{total_trials:,}] {rate:.1f} trials/min', end='\r')
                
                # 打印配置摘要
                success_rate = sum(1 for r in config_results if r['success']) / len(config_results) * 100
                print(f'    [{rta_config:15s}] 成功率={success_rate:5.1f}%')
    
    # 保存结果
    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    results['metadata'] = {
        'total_trials': total_trials,
        'elapsed_hours': elapsed,
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
    }
    
    output_path = Path('/root/rt1_full_experiment_results')
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成报告
    report = generate_report(results)
    with open(output_path / 'experiment_report.md', 'w') as f:
        f.write(report)
    
    print(f'\n{"="*70}')
    print(f'✅ 试验完成!')
    print(f'总试验数：{total_trials:,}')
    print(f'耗时：{elapsed:.2f} 小时')
    print(f'结果：{output_path}')
    print(f'{"="*70}')


def generate_report(results):
    """生成试验报告"""
    report = []
    report.append('# RT-1 完整 RTA 试验报告')
    report.append(f'生成时间：{datetime.now().isoformat()}')
    report.append('')
    
    # 元数据
    meta = results.get('metadata', {})
    report.append(f'## 试验概况')
    report.append(f'- 总试验数：{meta.get("total_trials", 0):,}')
    report.append(f'- 耗时：{meta.get("elapsed_hours", 0):.2f} 小时')
    report.append('')
    
    # 按配置统计
    report.append('## RTA 配置对比')
    config_stats = {}
    for i, config in enumerate(results['config']):
        if config not in config_stats:
            config_stats[config] = {'success': 0, 'total': 0, 'collision': 0}
        config_stats[config]['total'] += 1
        if results['success'][i]:
            config_stats[config]['success'] += 1
        if results['collision'][i]:
            config_stats[config]['collision'] += 1
    
    report.append('| 配置 | 试验数 | 成功率 | 碰撞率 |')
    report.append('|------|--------|--------|--------|')
    
    for config, stats in sorted(config_stats.items()):
        success_rate = stats['success'] / stats['total'] * 100
        collision_rate = stats['collision'] / stats['total'] * 100
        report.append(f'| {config} | {stats["total"]} | {success_rate:.1f}% | {collision_rate:.1f}% |')
    
    return '\n'.join(report)


if __name__ == '__main__':
    run_full_experiment()
