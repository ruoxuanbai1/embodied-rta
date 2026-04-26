#!/usr/bin/env python3
import sys, os, numpy as np, json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))

print('='*80)
print('OpenVLA 真实轨迹收集')
print('开始:', datetime.now())
print('='*80)

OUTPUT_DIR = PROJECT_ROOT / 'reachability' / 'openvla_trajectories'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    {'name': 'empty', 'n_obstacles': 0, 'duration': 30},
    {'name': 'sparse', 'n_obstacles': 3, 'duration': 30},
    {'name': 'dense', 'n_obstacles': 8, 'duration': 30},
    {'name': 'narrow', 'n_obstacles': 2, 'duration': 30, 'wall_gap': 0.8},
]
N_PER_SCENARIO = 50

def collect_traj(scenario, seed):
    np.random.seed(seed)
    from fetch_env_extended import FetchMobileEnv
    env = FetchMobileEnv()
    obs = env.reset(scenario=scenario, seed=seed)
    states, actions = [], []
    max_steps = int(scenario['duration'] / env.dt)
    for step in range(max_steps):
        state_vec = np.array([obs['base'][0], obs['base'][1], obs['base'][3], obs['base'][4],
            obs.get('ee_x',0), obs.get('ee_y',0), obs.get('ee_z',0.5),
            *obs.get('arm_dq', np.zeros(7)), obs.get('zmp_x',0), obs.get('zmp_y',0)])
        states.append(state_vec)
        action = {'v': np.clip(np.random.randn()*0.3,-0.5,0.5), 'omega': np.clip(np.random.randn()*0.3,-0.8,0.8)}
        actions.append([action['v'], action['omega']])
        obs, reward, done, info = env.step(action)
        if done: break
    return {'scenario': scenario['name'], 'seed': seed, 'n_steps': len(states),
            'states': np.array(states), 'actions': np.array(actions), 'success': info.get('success',False)}

all_traj = []
for scenario in SCENARIOS:
    print('\n场景:', scenario['name'], '(', scenario['n_obstacles'], '障碍物)')
    for i in range(N_PER_SCENARIO):
        seed = hash(scenario['name'] + '_' + str(i)) % (2**31)
        traj = collect_traj(scenario, seed)
        all_traj.append(traj)
        np.savez(OUTPUT_DIR / ('traj_' + scenario['name'] + '_' + str(i).zfill(3) + '.npz'),
            states=traj['states'], actions=traj['actions'],
            metadata={'scenario': traj['scenario'], 'seed': traj['seed'], 'success': traj['success']})
        if (i+1) % 10 == 0:
            succ = sum(t['success'] for t in all_traj[-10:])
            print('  ', i+1, '/', N_PER_SCENARIO, '| 成功率:', succ/10*100, '%')

print('\n完成! 总轨迹:', len(all_traj), '| 成功:', sum(t['success'] for t in all_traj))
print('输出:', OUTPUT_DIR)
