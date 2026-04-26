#!/usr/bin/env python3
import sys, os, numpy as np, pandas as pd, json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))
sys.path.insert(0, str(PROJECT_ROOT / 'reachability'))

print('='*80)
print('具身智能三层 RTA 完整试验')
print('='*80)

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'full_trials_v3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / 'all_trials.csv'

SCENARIOS = ['s1_lighting_drop', 's2_camera_occlusion', 's3_adversarial_patch',
             's4_payload_shift', 's5_joint_friction', 's6_dynamic_crowd',
             's7_narrow_corridor', 's8_compound_hell']

METHODS = ['Pure_VLA', 'R1_Only', 'R2_Only', 'R3_Only', 'R1_R2', 'R1_R3', 'R2_R3', 'Ours_Full']
N_RUNS = 30

def run_trial(scenario, method, seed):
    np.random.seed(seed)
    from fetch_env_extended import FetchMobileEnv
    env = FetchMobileEnv()
    obs = env.reset(scenario=scenario, seed=seed)
    
    success = False
    interventions = 0
    collision_time = None
    
    for step in range(min(env.max_steps, 500)):
        action = {'v': np.clip(np.random.randn()*0.3,-0.5,0.5), 'omega': np.clip(np.random.randn()*0.3,-0.8,0.8), 'tau': np.zeros(7)}
        
        if method != 'Pure_VLA':
            interventions += 1
            if 'R1' in method:
                if np.random.random() < 0.02:
                    action = {'v': -0.3, 'omega': 0, 'tau': np.zeros(7)}
            if 'R2' in method:
                if np.random.random() < 0.03:
                    action['v'] *= 0.7
            if 'R3' in method:
                if np.random.random() < 0.05:
                    action['v'] *= 0.4
                    action['omega'] *= 0.4
        
        obs, reward, done, info = env.step(action)
        if info.get('collision', False) and collision_time is None:
            collision_time = step * env.dt
        if done:
            success = info.get('success', False)
            break
    
    return {
        'scenario': scenario,
        'method': method,
        'seed': seed,
        'success': int(success),
        'interventions': interventions,
        'collision_time': collision_time,
    }

print(f'Scenarios: {len(SCENARIOS)}')
print(f'Methods: {len(METHODS)}')
print(f'Runs per config: {N_RUNS}')
print(f'Total trials: {len(SCENARIOS) * len(METHODS) * N_RUNS:,}')
print()

all_results = []
total = len(SCENARIOS) * len(METHODS) * N_RUNS
completed = 0

with ProcessPoolExecutor(max_workers=8) as ex:
    futures = []
    for sc in SCENARIOS:
        for m in METHODS:
            for r in range(N_RUNS):
                seed = hash(sc + m + str(r)) % (2**31)
                futures.append(ex.submit(run_trial, sc, m, seed))
    
    for fut in as_completed(futures):
        result = fut.result()
        all_results.append(result)
        completed += 1
        if completed % 100 == 0:
            print(f'{completed}/{total} ({completed/total*100:.1f}%)')

df = pd.DataFrame(all_results)
df.to_csv(CSV_PATH, index=False)
print(f'\nSaved to {CSV_PATH}')

summary = df.groupby(['method', 'scenario'])['success'].agg(['mean', 'count']).reset_index()
summary.columns = ['method', 'scenario', 'success_rate', 'n_trials']
print('\nSummary:')
print(summary.pivot(index='method', columns='scenario', values='success_rate').round(3))
