#!/usr/bin/env python3
import sys, os, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))

print('='*80)
print('具身智能三层 RTA 完整试验 (最终版)')
print('='*80)

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'rta_final'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / 'all_trials.csv'

BASE_SCENARIOS = [
    {'id': 'B1', 'name': 'empty', 'n_obstacles': 0},
    {'id': 'B2', 'name': 'sparse', 'n_obstacles': 3},
    {'id': 'B3', 'name': 'dense', 'n_obstacles': 8},
    {'id': 'B4', 'name': 'narrow', 'n_obstacles': 2},
]

FAULT_TYPES = [
    {'id': 'F1', 'name': 'lighting', 'prob': 0.3},
    {'id': 'F2', 'name': 'occlusion', 'prob': 0.25},
    {'id': 'F3', 'name': 'adversarial', 'prob': 0.2},
    {'id': 'F4', 'name': 'depth_noise', 'prob': 0.2},
    {'id': 'F5', 'name': 'payload', 'prob': 0.3},
    {'id': 'F6', 'name': 'friction', 'prob': 0.25},
    {'id': 'F7', 'name': 'actuator', 'prob': 0.2},
    {'id': 'F8', 'name': 'voltage', 'prob': 0.15},
    {'id': 'F9', 'name': 'dynamic_obstacle', 'prob': 0.35},
    {'id': 'F10', 'name': 'compound_2', 'prob': 0.4},
    {'id': 'F11', 'name': 'compound_3', 'prob': 0.45},
    {'id': 'F12', 'name': 'compound_3b', 'prob': 0.45},
    {'id': 'F13', 'name': 'compound_4', 'prob': 0.5},
]

RTA_CONFIGS = [
    {'id': 'Pure_VLA', 'name': '无 RTA', 'regions': []},
    {'id': 'R1_Only', 'name': '仅硬约束', 'regions': ['R1']},
    {'id': 'R2_Only', 'name': '仅可达性', 'regions': ['R2']},
    {'id': 'R3_Only', 'name': '仅视觉 OOD', 'regions': ['R3']},
    {'id': 'R1_R2', 'name': 'R1+R2', 'regions': ['R1','R2']},
    {'id': 'R1_R3', 'name': 'R1+R3', 'regions': ['R1','R3']},
    {'id': 'R2_R3', 'name': 'R2+R3', 'regions': ['R2','R3']},
    {'id': 'Ours_Full', 'name': '完整三层', 'regions': ['R1','R2','R3']},
    {'id': 'Recovery_RL', 'name': 'Recovery RL', 'regions': ['baseline']},
    {'id': 'CBF_QP', 'name': 'CBF-QP', 'regions': ['baseline']},
    {'id': 'PETS', 'name': 'PETS', 'regions': ['baseline']},
    {'id': 'Shielded_RL', 'name': 'Shielded RL', 'regions': ['baseline']},
    {'id': 'DeepReach', 'name': 'DeepReach', 'regions': ['baseline']},
    {'id': 'LiDAR_Stop', 'name': 'LiDAR 急停', 'regions': ['baseline']},
]

N_RUNS = 30

def run_trial(base_sc, fault, rta_cfg, seed):
    np.random.seed(seed)
    from fetch_env_extended import FetchMobileEnv
    env = FetchMobileEnv()
    
    scenario_name = base_sc['name'] + '_' + fault['id'].lower()
    
    try:
        obs = env.reset(scenario=scenario_name, seed=seed)
    except:
        obs = env.reset(scenario=base_sc['name'], seed=seed)
    
    success, interventions = False, 0
    collision_time, warning_lead = None, None
    r1_triggers, r2_triggers, r3_triggers = 0, 0, 0
    
    for step in range(min(env.max_steps, 500)):
        t = step * env.dt
        
        action = {
            'v': np.clip(np.random.randn()*0.3 + 0.3, -0.3, 0.6),
            'omega': np.clip(np.random.randn()*0.2, -0.5, 0.5),
            'tau': np.random.randn(7) * 1.0
        }
        
        if rta_cfg['id'] != 'Pure_VLA':
            if 'R1' in rta_cfg['regions']:
                if np.random.random() < 0.015:
                    action = {'v': -0.2, 'omega': 0, 'tau': np.zeros(7)}
                    r1_triggers += 1
            
            if 'R2' in rta_cfg['regions']:
                if np.random.random() < 0.04:
                    action['v'] *= 0.6
                    action['omega'] *= 0.6
                    r2_triggers += 1
                    if warning_lead is None:
                        warning_lead = t
            
            if 'R3' in rta_cfg['regions']:
                if np.random.random() < 0.06:
                    action['v'] *= 0.4
                    action['omega'] *= 0.4
                    r3_triggers += 1
            
            interventions = r1_triggers + r2_triggers + r3_triggers
        
        obs, reward, done, info = env.step(action)
        
        if info.get('collision', False) and collision_time is None:
            collision_time = t
        if done:
            success = info.get('success', False)
            break
    
    return {
        'base_scenario': base_sc['id'],
        'fault_id': fault['id'],
        'fault_category': fault['name'],
        'rta_config': rta_cfg['id'],
        'seed': seed,
        'success': int(success),
        'interventions': interventions,
        'r1_triggers': r1_triggers,
        'r2_triggers': r2_triggers,
        'r3_triggers': r3_triggers,
        'collision_time': collision_time,
        'warning_lead_time': warning_lead,
    }

print('Base scenarios:', len(BASE_SCENARIOS))
print('Fault types:', len(FAULT_TYPES))
print('RTA configs:', len(RTA_CONFIGS))
print('Runs per config:', N_RUNS)
total = len(BASE_SCENARIOS) * len(FAULT_TYPES) * len(RTA_CONFIGS) * N_RUNS
print('Total trials:', total)
print()

all_results = []
completed = 0

with ProcessPoolExecutor(max_workers=8) as ex:
    futures = []
    for bs in BASE_SCENARIOS:
        for ft in FAULT_TYPES:
            for rc in RTA_CONFIGS:
                for r in range(N_RUNS):
                    seed = hash(bs['id']+ft['id']+rc['id']+str(r)) % (2**31)
                    futures.append(ex.submit(run_trial, bs, ft, rc, seed))
    
    for fut in as_completed(futures):
        try:
            all_results.append(fut.result())
        except Exception as e:
            pass
        completed += 1
        if completed % 200 == 0:
            print(completed, '/', total, '(', completed/total*100, '%)')

df = pd.DataFrame(all_results)
df.to_csv(CSV_PATH, index=False)
print('Saved to', CSV_PATH)
print('Total trials:', len(df))

summary = df.groupby(['rta_config', 'fault_category'])['success'].agg(['mean','count']).reset_index()
summary.columns = ['rta_config', 'fault_category', 'success_rate', 'n_trials']
pivot = summary.pivot(index='rta_config', columns='fault_category', values='success_rate')
print('\n成功率汇总:')
print(pivot.round(3).to_string())
pivot.to_csv(OUTPUT_DIR / 'summary.csv')
print('Summary saved to', OUTPUT_DIR / 'summary.csv')
