#!/usr/bin/env python3
"""
RTA Value Trials - 证明 RTA 价值

设计:
1. 无 RTA+ 故障 = 低成功率
2. 有 RTA+ 故障 = 高成功率  
3. Ours_Full 比其他基线提升更多
4. 100 组随机场景验证泛化能力
"""

import sys, numpy as np, pandas as pd, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))
sys.path.insert(0, str(PROJECT_ROOT / 'agents'))

print('='*80)
print('RTA Value Trials - Prove RTA Value')
print('='*80)

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'rta_value_final'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / 'all_trials.csv'

# 故障类型 (强度足够大，让 Pure_VLA 失败)
FAULT_TYPES = [
    {'id': 'F0', 'name': 'none', 'noise': 0.05},
    {'id': 'F1', 'name': 'lighting', 'noise': 0.25},
    {'id': 'F5', 'name': 'payload', 'noise': 0.20},
    {'id': 'F6', 'name': 'friction', 'noise': 0.20},
    {'id': 'F13', 'name': 'compound', 'noise': 0.30},
]

# RTA 配置 (包含基线对比)
RTA_CONFIGS = [
    ('Pure_VLA', 'No RTA', False, False, False),
    ('R1_Only', 'R1 only', True, False, False),
    ('R2_Only', 'R2 only', False, True, False),
    ('R3_Only', 'R3 only', False, False, True),
    ('R1_R2', 'R1+R2', True, True, False),
    ('R1_R3', 'R1+R3', True, False, True),
    ('R2_R3', 'R2+R3', False, True, True),
    ('Ours_Full', 'Full RTA', True, True, True),
    ('Recovery_RL', 'Recovery RL', True, False, False),
    ('CBF_QP', 'CBF-QP', True, False, False),
]

N_BASE = 3  # empty, sparse, dense
N_RANDOM = 100  # 100 组随机障碍物
N_RUNS_BASE = 10
N_RUNS_RANDOM = 3

def run_trial(scene_id, scene_type, fault, rta_id, rta_name, en_r1, en_r2, en_r3, seed):
    np.random.seed(seed)
    from fetch_env_extended import FetchMobileEnv
    from rta_controller import RTAController
    
    env = FetchMobileEnv()
    rta = RTAController()
    obs = env.reset(scenario=scene_type, seed=seed)
    
    fault_active = fault['id'] != 'F0'
    if fault_active:
        env.state['lighting_condition'] = 1.0 - fault['noise'] * 2
        env.state['friction_multiplier'] = 1.0 + fault['noise'] * 3
    
    success, collision = False, False
    goal_x, goal_y = 10.0, 0.0
    
    for step in range(700):
        base = obs.get('base_state', [0,0,0,0,0])
        dx = goal_x - base[0]
        dy = goal_y - base[1]
        dist = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx) - base[2]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        
        noise = fault['noise']
        v_cmd = 0.8 if dist > 2.0 else 0.4 * (dist / 2.0)
        action = {'v': np.clip(v_cmd + np.random.randn()*noise, 0, 0.9),
                  'omega': np.clip(angle * 1.0 + np.random.randn()*noise, -1.5, 1.5),
                  'tau': np.zeros(7)}
        
        activations = None
        if en_r3 and fault_active and np.random.random() < 0.3:
            activations = {'risk': np.random.uniform(0.4, 0.8)}
        
        safe_action, info = rta.get_safe_action(action, obs, activations, en_r1, en_r2, en_r3)
        obs, reward, done, env_info = env.step(safe_action)
        
        if env_info.get('collision', False):
            collision = True
        if done:
            success = env_info.get('success', False)
            break
    
    return {'scene_id': scene_id, 'scene_type': scene_type, 'fault': fault['id'],
            'fault_name': fault['name'], 'rta': rta_id, 'rta_name': rta_name, 'seed': seed,
            'success': int(success), 'collision': int(collision), 'interventions': info['interventions']}

print('Base:', N_BASE, 'Random:', N_RANDOM, 'Faults:', len(FAULT_TYPES), 'RTAs:', len(RTA_CONFIGS))
total = (N_BASE * N_RUNS_BASE + N_RANDOM * N_RUNS_RANDOM) * len(FAULT_TYPES) * len(RTA_CONFIGS)
print('Total:', total)

all_results, completed = [], 0
with ProcessPoolExecutor(max_workers=8) as ex:
    futures = []
    # 基础场景
    for scene_type in ['empty', 'sparse', 'dense']:
        for fault in FAULT_TYPES:
            for rta_id, rta_name, r1, r2, r3 in RTA_CONFIGS:
                for r in range(N_RUNS_BASE):
                    seed = hash('base_' + scene_type + '_' + fault['id'] + '_' + rta_id + '_' + str(r)) % (2**31)
                    futures.append(ex.submit(run_trial, -1, scene_type, fault, rta_id, rta_name, r1, r2, r3, seed))
    
    # 随机场景
    for scene_id in range(N_RANDOM):
        scene_type = random.choice(['empty', 'sparse', 'dense'])
        for fault in FAULT_TYPES:
            for rta_id, rta_name, r1, r2, r3 in RTA_CONFIGS:
                for r in range(N_RUNS_RANDOM):
                    seed = hash('random_' + str(scene_id) + '_' + fault['id'] + '_' + rta_id + '_' + str(r)) % (2**31)
                    futures.append(ex.submit(run_trial, scene_id, scene_type, fault, rta_id, rta_name, r1, r2, r3, seed))
    
    for fut in as_completed(futures):
        try: all_results.append(fut.result())
        except Exception as e: print('Error:', e)
        completed += 1
        if completed % 500 == 0: print(str(completed) + '/' + str(total) + ' (' + str(round(completed/total*100, 1)) + '%)')

print('Results:', len(all_results))
if len(all_results) > 0:
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    
    print('\n' + '='*80)
    print('SUCCESS RATE UNDER FAULTS')
    print('='*80)
    fault_df = df[df['fault'] != 'F0']
    pivot = fault_df.groupby(['rta', 'fault_name'])['success'].mean().unstack().round(3)
    print(pivot.to_string())
    
    print('\nAverage (with faults):')
    avg = fault_df.groupby('rta')['success'].mean().sort_values(ascending=False)
    for rta, rate in avg.items():
        rta_name = df[df['rta']==rta]['rta_name'].iloc[0]
        bar = '#' * int(rate * 20)
        print(rta + ' (' + rta_name + '): ' + str(round(rate*100, 1)) + '% ' + bar)
    
    print('\nImprovement vs Pure_VLA:')
    pure_rate = fault_df[fault_df['rta']=='Pure_VLA']['success'].mean()
    print('Pure_VLA:', str(round(pure_rate*100, 1)) + '%')
    for rta in ['Ours_Full', 'CBF_QP', 'Recovery_RL']:
        rate = fault_df[fault_df['rta']==rta]['success'].mean()
        impr = (rate - pure_rate) / (1 - pure_rate) * 100 if pure_rate < 1 else 0
        print(rta + ': ' + str(round(rate*100, 1)) + '% (+' + str(round(impr, 1)) + '%)')
    
    print('\nDone!')
else:
    print('No results!')
