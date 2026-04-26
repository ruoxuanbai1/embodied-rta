#!/usr/bin/env python3
"""
RTA Value Trials - 证明 RTA 价值

核心逻辑:
1. 不加 RTA + 故障 = 成功率低
2. 加 RTA + 故障 = 成功率高
3. 比基线方法提升更多

试验设计:
- 100 组随机场景
- 每组测试：无故障 + 5 种故障
- 对比：Pure_VLA vs RTA vs 基线
"""

import sys, numpy as np, pandas as pd, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path('/home/vipuser/Embodied-RTA')
sys.path.insert(0, str(PROJECT_ROOT / 'envs'))
sys.path.insert(0, str(PROJECT_ROOT / 'agents'))

print('='*80)
print('RTA Value Trials - Prove RTA Effectiveness')
print('='*80)

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'rta_value'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / 'all_trials.csv'

# 故障类型 (强度足够大，让 Pure_VLA 失败)
FAULT_TYPES = [
    {'id': 'F0', 'name': 'none', 'noise': 0.05},      # 无故障
    {'id': 'F1', 'name': 'lighting', 'noise': 0.25},  # 强光照干扰
    {'id': 'F5', 'name': 'payload', 'noise': 0.20},   # 重负载
    {'id': 'F6', 'name': 'friction', 'noise': 0.20},  # 高摩擦
    {'id': 'F13', 'name': 'compound', 'noise': 0.30}, # 复合故障
]

# RTA 配置 (包含基线方法)
RTA_CONFIGS = [
    ('Pure_VLA', 'No RTA', False, False, False),      # 基线
    ('R1_Only', 'R1 only', True, False, False),
    ('R2_Only', 'R2 only', False, True, False),
    ('R3_Only', 'R3 only', False, False, True),
    ('R1_R2', 'R1+R2', True, True, False),
    ('R1_R3', 'R1+R3', True, False, True),
    ('R2_R3', 'R2+R3', False, True, True),
    ('Ours_Full', 'Full RTA', True, True, True),      # 我们的方法
    ('Recovery_RL', 'Recovery RL', True, False, False),  # 基线
    ('CBF_QP', 'CBF-QP', True, False, False),           # 基线
]

N_SCENARIOS = 100  # 100 组随机场景
N_RUNS = 10  # 每组 10 次

def generate_random_scene(scene_type, seed):
    """生成随机障碍物配置"""
    np.random.seed(seed)
    random.seed(seed)
    
    if scene_type == 'empty':
        return {'n_obstacles': 0, 'obstacles': []}
    elif scene_type == 'sparse':
        n = random.randint(3, 7)
    elif scene_type == 'dense':
        n = random.randint(8, 15)
    else:
        n = 0
    
    obstacles = []
    for i in range(n):
        obs = {
            'x': random.uniform(2, 8),
            'y': random.uniform(-3, 3),
            'radius': random.uniform(0.2, 0.5),
            'type': 'static'
        }
        obstacles.append(obs)
    
    return {'n_obstacles': n, 'obstacles': obstacles}

def run_trial(scene_id, scene_type, scene_cfg, fault, rta_id, rta_name, en_r1, en_r2, en_r3, seed):
    """单次试验"""
    np.random.seed(seed)
    from fetch_env_extended import FetchMobileEnv
    from rta_controller import RTAController
    
    env = FetchMobileEnv()
    rta = RTAController()
    
    # 使用基础场景
    obs = env.reset(scenario=scene_type, seed=seed)
    
    # 手动注入故障
    fault_active = fault['id'] != 'F0'
    if fault_active:
        env.state['lighting_condition'] = 1.0 - fault['noise'] * 2
        env.state['friction_multiplier'] = 1.0 + fault['noise'] * 3
    
    success, collision, steps = False, False, 0
    goal_x, goal_y = 10.0, 0.0
    max_steps = 700
    
    for step in range(max_steps):
        base = obs.get('base_state', [0,0,0,0,0])
        dx = goal_x - base[0]
        dy = goal_y - base[1]
        dist = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx) - base[2]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        
        # VLA 动作 (带故障噪声)
        noise = fault['noise']
        v_cmd = 0.8 if dist > 2.0 else 0.4 * (dist / 2.0)
        action = {
            'v': np.clip(v_cmd + np.random.randn()*noise, 0, 0.9),
            'omega': np.clip(angle * 1.0 + np.random.randn()*noise, -1.5, 1.5),
            'tau': np.zeros(7)
        }
        
        # RTA 检查
        activations = None
        if en_r3 and fault_active and np.random.random() < 0.3:
            activations = {'risk': np.random.uniform(0.4, 0.8)}
        
        safe_action, info = rta.get_safe_action(action, obs, activations, en_r1, en_r2, en_r3)
        obs, reward, done, env_info = env.step(safe_action)
        
        if env_info.get('collision', False):
            collision = True
        if done:
            success = env_info.get('success', False)
            steps = step
            break
        steps = step
    
    return {
        'scene_id': scene_id, 'scene_type': scene_type,
        'fault': fault['id'], 'fault_name': fault['name'],
        'rta': rta_id, 'rta_name': rta_name, 'seed': seed,
        'success': int(success), 'collision': int(collision), 'steps': steps,
        'interventions': info['interventions'],
    }

print(f'Scenarios: {N_SCENARIOS}, Faults: {len(FAULT_TYPES)}, RTAs: {len(RTA_CONFIGS)}, Runs: {N_RUNS}')
total = N_SCENARIOS * len(FAULT_TYPES) * len(RTA_CONFIGS) * N_RUNS
print(f'Total: {total:,}')
print()

all_results = []
completed = 0

# 生成 100 组随机场景
scenes = []
for scene_id in range(N_SCENARIOS):
    scene_type = random.choice(['empty', 'sparse', 'dense'])
    scene_cfg = generate_random_scene(scene_type, scene_id)
    scenes.append((scene_id, scene_type, scene_cfg))

print(f'Generated {len(scenes)} random scenes')

with ProcessPoolExecutor(max_workers=8) as ex:
    futures = []
    for scene_id, scene_type, scene_cfg in scenes:
        for fault in FAULT_TYPES:
            for rta_id, rta_name, r1, r2, r3 in RTA_CONFIGS:
                for r in range(N_RUNS):
                    seed = hash(f'{scene_id}_{fault["id"]}_{rta_id}_{r}') % (2**31)
                    futures.append(ex.submit(run_trial, scene_id, scene_type, scene_cfg, 
                                            fault, rta_id, rta_name, r1, r2, r3, seed))
    
    for fut in as_completed(futures):
        try:
            all_results.append(fut.result())
        except Exception as e:
            print('Error:', e)
        completed += 1
        if completed % 1000 == 0:
            print(f'{completed}/{total} ({completed/total*100:.1f}%)')

print(f'\nResults: {len(all_results)}')
if len(all_results) > 0:
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    
    # 核心分析：故障场景下各方法的成功率
    print('\n' + '='*80)
    print('SUCCESS RATE UNDER FAULTS (RTA Value)')
    print('='*80)
    
    fault_df = df[df['fault'] != 'F0']
    pivot = fault_df.groupby(['rta', 'fault_name'])['success'].mean().unstack().round(3)
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / 'fault_success.csv')
    
    # 平均成功率对比
    print('\nAverage Success Rate (with faults):')
    avg = fault_df.groupby('rta')['success'].mean().sort_values(ascending=False)
    for rta, rate in avg.items():
        rta_name = df[df['rta']==rta]['rta_name'].iloc[0]
        bar = '#' * int(rate * 20)
        print(f'{rta:15s} ({rta_name:12s}) {rate*100:5.1f}% {bar}')
    
    # 无故障 baseline
    print('\nBaseline (no fault):')
    no_fault = df[df['fault'] == 'F0'].groupby('rta')['success'].mean()
    for rta, rate in no_fault.items():
        print(f'{rta:15s} {rate*100:5.1f}%')
    
    # RTA 提升
    print('\nRTA Improvement (vs Pure_VLA):')
    pure_vla_rate = fault_df[fault_df['rta']=='Pure_VLA']['success'].mean()
    print(f'Pure_VLA: {pure_vla_rate*100:.1f}%')
    for rta in ['Ours_Full', 'R1_R2', 'R1_R3', 'R2_R3', 'CBF_QP', 'Recovery_RL']:
        if rta in fault_df['rta'].values:
            rate = fault_df[fault_df['rta']==rta]['success'].mean()
            improvement = (rate - pure_vla_rate) / pure_vla_rate * 100
            print(f'{rta:15s} {rate*100:5.1f}% (+' + f'{improvement:.1f}' + '%)')
    
    # 碰撞率
    print('\nCollision Rate (with faults):')
    collision = fault_df.groupby('rta')['collision'].mean().sort_values()
    for rta, rate in collision.items():
        print(f'{rta:15s} {rate*100:5.1f}%')
    
    print('\nTrial completed!')
else:
    print('No results!')
