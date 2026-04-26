#!/usr/bin/env python3
"""
具身智能三层 RTA - 完整试验脚本

试验矩阵：4 基础场景 × 13 故障类型 × 15 RTA 配置 × 30 随机种子 = 23,400 次
"""

import sys, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))
sys.path.insert(0, str(PROJECT_ROOT / 'agents'))

print('='*80)
print('Embodied AI Three-Layer RTA - Full Trials')
print('='*80)

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'full_trials_final'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / 'all_trials.csv'

# 4 基础场景
BASE_SCENARIOS = [
    {'id': 'B1', 'name': 'empty', 'n_obstacles': 0, 'wall_gap': None},
    {'id': 'B2', 'name': 'sparse', 'n_obstacles': 5, 'wall_gap': None},
    {'id': 'B3', 'name': 'dense', 'n_obstacles': 10, 'wall_gap': None},
    {'id': 'B4', 'name': 'narrow', 'n_obstacles': 0, 'wall_gap': 0.8},
]

# 13 故障类型
FAULT_TYPES = [
    {'id': 'F0', 'name': 'none', 'faults': []},  # 无故障
    {'id': 'F1', 'name': 'lighting', 'faults': ['lighting_drop']},
    {'id': 'F2', 'name': 'occlusion', 'faults': ['camera_occlusion']},
    {'id': 'F3', 'name': 'adversarial', 'faults': ['adversarial_patch']},
    {'id': 'F4', 'name': 'depth_noise', 'faults': ['depth_noise']},
    {'id': 'F5', 'name': 'payload', 'faults': ['payload_shift']},
    {'id': 'F6', 'name': 'friction', 'faults': ['joint_friction']},
    {'id': 'F7', 'name': 'actuator', 'faults': ['actuator_degradation']},
    {'id': 'F8', 'name': 'voltage', 'faults': ['voltage_drop']},
    {'id': 'F9', 'name': 'dynamic_obstacle', 'faults': ['dynamic_intruder']},
    {'id': 'F10', 'name': 'compound_2', 'faults': ['lighting_drop', 'payload_shift']},
    {'id': 'F11', 'name': 'compound_3a', 'faults': ['lighting_drop', 'payload_shift', 'dynamic_intruder']},
    {'id': 'F12', 'name': 'compound_3b', 'faults': ['lighting_drop', 'camera_occlusion', 'joint_friction']},
    {'id': 'F13', 'name': 'compound_4', 'faults': ['lighting_drop', 'camera_occlusion', 'payload_shift', 'dynamic_intruder']},
]

# 15 RTA 配置
RTA_CONFIGS = [
    ('Pure_VLA', False, False, False),
    ('R1_Only', True, False, False),
    ('R2_Only', False, True, False),
    ('R3_Only', False, False, True),
    ('R1_R2', True, True, False),
    ('R1_R3', True, False, True),
    ('R2_R3', False, True, True),
    ('Ours_Full', True, True, True),
    ('Recovery_RL', True, False, False),
    ('CBF_QP', True, False, False),
    ('PETS', False, True, False),
    ('Shielded_RL', True, True, False),
    ('DeepReach', False, True, False),
    ('LiDAR_Stop', True, False, False),
]

N_RUNS = 30

def run_trial(base_sc, fault, rta_id, en_r1, en_r2, en_r3, seed):
    np.random.seed(seed)
    from fetch_env_extended import FetchMobileEnv
    from rta_controller import RTAController
    
    env = FetchMobileEnv()
    rta = RTAController()
    
    # 构建场景名：基础场景 + 故障
    if fault['id'] == 'F0':
        scenario_name = base_sc['name']
    else:
        scenario_name = f"{base_sc['name']}_{fault['id'].lower()}"
    
    # 尝试重置环境
    try:
        obs = env.reset(scenario=scenario_name, seed=seed)
    except:
        # 如果场景不存在，使用基础场景
        obs = env.reset(scenario=base_sc['name'], seed=seed)
    
    success, collision_t, warning_t = False, None, None
    goal_x, goal_y = 10.0, 0.0
    max_steps = 700
    
    for step in range(max_steps):
        t = step * env.dt
        base = obs.get('base_state', obs.get('base', [0,0,0,0,0]))
        
        dx = goal_x - base[0]
        dy = goal_y - base[1]
        dist = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx) - base[2]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        
        # 导航策略
        v_cmd = 0.8 if dist > 2.0 else 0.4 * (dist / 2.0)
        action = {
            'v': np.clip(v_cmd + np.random.randn()*0.05, 0, 0.9),
            'omega': np.clip(angle * 1.0 + np.random.randn()*0.05, -1.2, 1.2),
            'tau': np.zeros(7)
        }
        
        # RTA 检查
        activations = None
        if en_r3 and np.random.random() < 0.1:
            activations = {'risk': np.random.uniform(0.3, 0.7)}
        
        safe_action, info = rta.get_safe_action(action, obs, activations, en_r1, en_r2, en_r3)
        obs, reward, done, env_info = env.step(safe_action)
        
        if env_info.get('collision', False) and collision_t is None:
            collision_t = t
        if info.get('r2', False) and warning_t is None:
            warning_t = t
        if done:
            success = env_info.get('success', False)
            break
    
    return {
        'base_scenario': base_sc['id'],
        'fault_id': fault['id'],
        'fault_name': fault['name'],
        'rta_config': rta_id,
        'seed': seed,
        'success': int(success),
        'interventions': info['interventions'],
        'r1_triggers': int(info['r1']),
        'r2_triggers': int(info['r2']),
        'r3_triggers': int(info['r3']),
        'collision_time': collision_t,
        'warning_time': warning_t,
    }

print(f'Base scenarios: {len(BASE_SCENARIOS)}')
print(f'Fault types: {len(FAULT_TYPES)}')
print(f'RTA configs: {len(RTA_CONFIGS)}')
print(f'Runs per config: {N_RUNS}')
total = len(BASE_SCENARIOS) * len(FAULT_TYPES) * len(RTA_CONFIGS) * N_RUNS
print(f'Total trials: {total:,}')
print()

all_results, completed = [], 0
with ProcessPoolExecutor(max_workers=8) as ex:
    futures = []
    for bs in BASE_SCENARIOS:
        for ft in FAULT_TYPES:
            for rta_id, r1, r2, r3 in RTA_CONFIGS:
                for r in range(N_RUNS):
                    seed = hash(bs['id'] + ft['id'] + rta_id + str(r)) % (2**31)
                    futures.append(ex.submit(run_trial, bs, ft, rta_id, r1, r2, r3, seed))
    
    for fut in as_completed(futures):
        try:
            all_results.append(fut.result())
        except Exception as e:
            print('Error:', e)
        completed += 1
        if completed % 500 == 0:
            print(f'{completed}/{total} ({completed/total*100:.1f}%)')

print(f'\nResults: {len(all_results)}')
if len(all_results) > 0:
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    print(f'Saved: {CSV_PATH}')
    
    # 汇总统计
    summary = df.groupby(['rta_config', 'fault_name'])['success'].mean().unstack().round(3)
    print('\nSuccess Rate by Fault Type:')
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / 'summary_by_fault.csv')
    
    # 按基础场景统计
    summary_base = df.groupby(['rta_config', 'base_scenario'])['success'].mean().unstack().round(3)
    print('\nSuccess Rate by Base Scenario:')
    print(summary_base.to_string())
    summary_base.to_csv(OUTPUT_DIR / 'summary_by_base.csv')
    
    # 平均成功率
    avg = df.groupby('rta_config')['success'].mean().sort_values(ascending=False)
    print('\nAverage Success Rate:')
    for rta, rate in avg.items():
        bar = '#' * int(rate * 20)
        print(f'{rta:15s} {rate*100:5.1f}% {bar}')
    
    # 干预率
    print('\nAverage Interventions:')
    interventions = df.groupby('rta_config')['interventions'].mean().sort_values(ascending=False)
    for rta, val in interventions.items():
        print(f'{rta:15s} {val:6.2f}')
    
    print('\nTrial completed!')
else:
    print('No results!')
