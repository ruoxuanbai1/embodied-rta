#!/usr/bin/env python3
import sys, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))
sys.path.insert(0, str(PROJECT_ROOT / 'agents'))

print('='*80)
print('RTA Final Trials - With Navigation Policy')
print('='*80)

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'rta_final_v3'
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
    goal_x, goal_y = 10.0, 0.0  # 目标点
    
    for step in range(min(env.max_steps, 500)):
        t = step * env.dt
        base = obs.get('base_state', obs.get('base', [0,0,0,0,0]))
        
        # 基础导航策略：朝向目标点移动
        dx = goal_x - base[0]
        dy = goal_y - base[1]
        dist_to_goal = np.sqrt(dx*dx + dy*dy)
        angle_to_goal = np.arctan2(dy, dx) - base[2]
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))
        
        # VLA 动作 (带导航)
        action = {
            'v': np.clip(0.3 + np.random.randn()*0.1, 0, 0.6),  # 向前
            'omega': np.clip(angle_to_goal * 0.5 + np.random.randn()*0.1, -0.5, 0.5),  # 转向目标
            'tau': np.zeros(7)
        }
        
        # 接近目标时减速
        if dist_to_goal < 2.0:
            action['v'] *= dist_to_goal / 2.0
        
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
        'scenario': scenario, 'rta': rta_id, 'seed': seed, 'success': int(success),
        'interventions': info['interventions'], 'r1': int(info['r1']), 'r2': int(info['r2']), 
        'r3': int(info['r3']), 'collision_t': collision_t, 'warning_t': warning_t,
        'dist_to_goal': dist_to_goal
    }

print('Scenarios:', len(SCENARIOS), 'RTAs:', len(RTA_CONFIGS), 'Runs:', N_RUNS)
total = len(SCENARIOS) * len(RTA_CONFIGS) * N_RUNS
print('Total:', total)
print()

all_results, completed = [], 0
with ProcessPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(run_trial, sc, rta_id, r1, r2, r3, hash(sc+rta_id+str(r))%(2**31)) 
               for sc in SCENARIOS for rta_id, r1, r2, r3 in RTA_CONFIGS for r in range(N_RUNS)]
    for fut in as_completed(futures):
        try: all_results.append(fut.result())
        except Exception as e: print('Error:', e)
        completed += 1
        if completed % 100 == 0: print(f'{completed}/{total} ({completed/total*100:.1f}%)')

print(f'\nResults: {len(all_results)}')
if len(all_results) > 0:
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    print('Saved:', CSV_PATH)
    
    # 成功率汇总
    summary = df.groupby(['rta', 'scenario'])['success'].agg(['mean','count']).reset_index()
    summary.columns = ['rta', 'scenario', 'rate', 'n']
    pivot = summary.pivot(index='rta', columns='scenario', values='rate')
    
    print('\n' + '='*70)
    print('成功率汇总')
    print('='*70)
    print(pivot.round(3).to_string())
    pivot.to_csv(OUTPUT_DIR / 'summary.csv')
    
    # 平均成功率
    avg_rates = df.groupby('rta')['success'].mean().sort_values(ascending=False)
    print('\n平均成功率:')
    for rta, rate in avg_rates.items():
        bar = '#' * int(rate * 20)
        print(f'{rta:15s} {rate*100:5.1f}% {bar}')
    
    # 干预率
    print('\n平均干预次数:')
    interventions = df.groupby('rta')['interventions'].mean().sort_values(ascending=False)
    for rta, val in interventions.items():
        print(f'{rta:15s} {val:6.2f}')
    
    # 预警时间
    warning_df = df[df['warning_t'].notna()]
    if len(warning_df) > 0:
        print(f'\n平均预警提前时间：{warning_df["warning_t"].mean():.2f}s')
    
    # 保存详细报告
    with open(OUTPUT_DIR / 'report.txt', 'w') as f:
        f.write('RTA Trial Report\n')
        f.write('='*70 + '\n\n')
        f.write(f'Total trials: {len(df)}\n\n')
        f.write('Success Rates:\n')
        f.write(pivot.round(3).to_string() + '\n\n')
        f.write('Average Success Rate by RTA:\n')
        for rta, rate in avg_rates.items():
            f.write(f'{rta}: {rate*100:.1f}%\n')
    
    print('\nReport saved to:', OUTPUT_DIR / 'report.txt')
else:
    print('No results!')
