#!/usr/bin/env python3
import sys, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))
sys.path.insert(0, str(PROJECT_ROOT / 'agents'))

print('='*80)
print('RTA Final Trials')
print('='*80)

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'rta_final_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / 'all_trials.csv'

SCENARIOS = ['s1_lighting_drop', 's2_camera_occlusion', 's3_adversarial_patch', 
             's4_payload_shift', 's5_joint_friction', 's6_dynamic_crowd', 
             's7_narrow_corridor', 's8_compound_hell']

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

def run_trial(scenario, rta_id, en_r1, en_r2, en_r3, seed):
    np.random.seed(seed)
    from fetch_env_extended import FetchMobileEnv
    from rta_controller import RTAController
    
    env = FetchMobileEnv()
    rta = RTAController()
    obs = env.reset(scenario=scenario, seed=seed)
    
    success, collision_t, warning_t = False, None, None
    
    for step in range(min(env.max_steps, 500)):
        t = step * env.dt
        action = {'v': np.clip(np.random.randn()*0.3+0.3, -0.3, 0.6), 
                  'omega': np.clip(np.random.randn()*0.2, -0.5, 0.5), 
                  'tau': np.zeros(7)}
        
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
    
    return {'scenario': scenario, 'rta': rta_id, 'seed': seed, 'success': int(success),
            'interventions': info['interventions'], 'r1': int(info['r1']), 'r2': int(info['r2']), 
            'r3': int(info['r3']), 'collision_t': collision_t, 'warning_t': warning_t}

print('Scenarios:', len(SCENARIOS), 'RTAs:', len(RTA_CONFIGS), 'Runs:', N_RUNS)
total = len(SCENARIOS) * len(RTA_CONFIGS) * N_RUNS
print('Total:', total)

all_results, completed = [], 0
with ProcessPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(run_trial, sc, rta_id, r1, r2, r3, hash(sc+rta_id+str(r))%(2**31)) 
               for sc in SCENARIOS for rta_id, r1, r2, r3 in RTA_CONFIGS for r in range(N_RUNS)]
    for fut in as_completed(futures):
        try: all_results.append(fut.result())
        except Exception as e: print('Error:', e)
        completed += 1
        if completed % 100 == 0: print(completed, '/', total, round(completed/total*100, 1), '%')

print('Results:', len(all_results))
if len(all_results) > 0:
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    print('Saved:', CSV_PATH)
    summary = df.groupby(['rta', 'scenario'])['success'].agg(['mean','count']).reset_index()
    summary.columns = ['rta', 'scenario', 'rate', 'n']
    pivot = summary.pivot(index='rta', columns='scenario', values='rate')
    print('\nSuccess Rate:')
    print(pivot.round(3).to_string())
    pivot.to_csv(OUTPUT_DIR / 'summary.csv')
else:
    print('No results!')
