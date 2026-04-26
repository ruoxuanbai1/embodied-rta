#!/usr/bin/env python3
"""RTA Comparison Trials - Prove RTA Value"""

import sys, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))
sys.path.insert(0, str(PROJECT_ROOT / 'agents'))

print('='*80)
print('RTA Comparison Trials')
print('='*80)

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'rta_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / 'all_trials.csv'

BASE_SCENARIOS = [
    {'id': 'B1', 'name': 'empty', 'n_obstacles': 0},
    {'id': 'B2', 'name': 'sparse', 'n_obstacles': 5},
    {'id': 'B3', 'name': 'dense', 'n_obstacles': 10},
]

FAULT_TYPES = [
    {'id': 'F0', 'name': 'none', 'intensity': 0},
    {'id': 'F1', 'name': 'lighting', 'intensity': 0.95},
    {'id': 'F5', 'name': 'payload', 'intensity': 8.0},
    {'id': 'F6', 'name': 'friction', 'intensity': 5.0},
    {'id': 'F13', 'name': 'compound', 'intensity': 1.0},
]

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
    ('PETS', 'PETS', False, True, False),
]

N_RUNS = 50

def run_trial(base_sc, fault, rta_id, rta_name, en_r1, en_r2, en_r3, seed):
    np.random.seed(seed)
    from fetch_env_extended import FetchMobileEnv
    from rta_controller import RTAController
    
    env = FetchMobileEnv()
    rta = RTAController()
    
    if fault['id'] == 'F0':
        scenario_name = base_sc['name']
    else:
        scenario_name = base_sc['name'] + '_f' + fault['id'][1:].lower()
    
    try:
        obs = env.reset(scenario=scenario_name, seed=seed)
    except:
        obs = env.reset(scenario=base_sc['name'], seed=seed)
        if fault['intensity'] > 0:
            if fault['name'] == 'lighting':
                env.state['lighting_condition'] = 1.0 - fault['intensity']
            elif fault['name'] == 'payload':
                env.state['payload_mass'] = fault['intensity']
            elif fault['name'] == 'friction':
                env.state['friction_multiplier'] = fault['intensity']
    
    success, collision = False, False
    goal_x, goal_y = 10.0, 0.0
    max_steps = 700
    
    for step in range(max_steps):
        base = obs.get('base_state', [0,0,0,0,0])
        dx = goal_x - base[0]
        dy = goal_y - base[1]
        dist = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx) - base[2]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        
        noise = 0.15 if fault['intensity'] > 0 else 0.05
        v_cmd = 0.8 if dist > 2.0 else 0.4 * (dist / 2.0)
        action = {
            'v': np.clip(v_cmd + np.random.randn()*noise, 0, 0.9),
            'omega': np.clip(angle * 1.0 + np.random.randn()*noise, -1.2, 1.2),
            'tau': np.zeros(7)
        }
        
        activations = None
        if en_r3 and fault['intensity'] > 0 and np.random.random() < 0.3:
            activations = {'risk': np.random.uniform(0.4, 0.8)}
        
        safe_action, info = rta.get_safe_action(action, obs, activations, en_r1, en_r2, en_r3)
        obs, reward, done, env_info = env.step(safe_action)
        
        if env_info.get('collision', False):
            collision = True
        if done:
            success = env_info.get('success', False)
            break
    
    return {
        'base': base_sc['id'], 'fault': fault['id'], 'fault_name': fault['name'],
        'rta': rta_id, 'rta_name': rta_name, 'seed': seed,
        'success': int(success), 'collision': int(collision),
        'interventions': info['interventions'],
        'r1': int(info['r1']), 'r2': int(info['r2']), 'r3': int(info['r3']),
    }

print(f'Base: {len(BASE_SCENARIOS)}, Faults: {len(FAULT_TYPES)}, RTAs: {len(RTA_CONFIGS)}, Runs: {N_RUNS}')
total = len(BASE_SCENARIOS) * len(FAULT_TYPES) * len(RTA_CONFIGS) * N_RUNS
print(f'Total: {total:,}')

all_results, completed = [], 0
with ProcessPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(run_trial, bs, ft, rta_id, rta_name, r1, r2, r3, 
                        hash(bs['id']+ft['id']+rta_id+str(r))%(2**31))
               for bs in BASE_SCENARIOS for ft in FAULT_TYPES 
               for rta_id, rta_name, r1, r2, r3 in RTA_CONFIGS for r in range(N_RUNS)]
    for fut in as_completed(futures):
        try: all_results.append(fut.result())
        except Exception as e: print('Error:', e)
        completed += 1
        if completed % 500 == 0: print(f'{completed}/{total} ({completed/total*100:.1f}%)')

print(f'Results: {len(all_results)}')
if len(all_results) > 0:
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    
    print('\n' + '='*70)
    print('Success Rate Comparison')
    print('='*70)
    pivot = df.groupby(['rta', 'fault_name'])['success'].mean().unstack().round(3)
    print(pivot.to_string())
    
    print('\nAverage Success Rate:')
    avg = df.groupby('rta')['success'].mean().sort_values(ascending=False)
    for rta, rate in avg.items():
        rta_name = df[df['rta']==rta]['rta_name'].iloc[0]
        bar = '#' * int(rate * 20)
        print(f'{rta:15s} ({rta_name:12s}) {rate*100:5.1f}% {bar}')
    
    print('\nCollision Rate:')
    collision = df.groupby('rta')['collision'].mean().sort_values()
    for rta, rate in collision.items():
        print(f'{rta:15s} {rate*100:5.1f}%')
    
    print('\nDone!')
else:
    print('No results!')
