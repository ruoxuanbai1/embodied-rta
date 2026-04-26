#!/usr/bin/env python3
"""收集 ALOHA 真实仿真训练数据"""
import sys
sys.path.insert(0, '/home/vipuser/Embodied-RTA')
from envs.aloha_sim import ALOHASimulationEnv
import numpy as np, time, glob
from pathlib import Path
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

print("="*60)
print("ALOHA 真实仿真数据收集")
print("="*60)

# 配置
NORMAL_DIR = Path("/mnt/data/aloha_real_normal")
FAULT_DIR = Path("/mnt/data/aloha_real_fault")
NORMAL_DIR.mkdir(exist_ok=True)
FAULT_DIR.mkdir(exist_ok=True)

N_NORMAL = 500
N_FAULT_PER_TYPE = 50
MAX_STEPS = 250

FAULT_TYPES = ['F1_lighting', 'F2_occlusion', 'F3_adversarial', 'F4_payload', 
               'F5_friction', 'F6_dynamic', 'F7_sensor', 'F8_compound']

# 加载 ACT 模型
print("\n加载 ACT 模型...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
act_model = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human").eval().to(DEVICE)

# Hook 用于提取激活
hooks = {'hook_a': None, 'hook_b': None}
def save_a(m, i, o): hooks['hook_a'] = o.detach().cpu().numpy()
def save_b(m, i, o): hooks['hook_b'] = o.detach().cpu().numpy()
act_model.model.encoder.layers[-1].linear2.register_forward_hook(save_a)
act_model.model.decoder.layers[-1].linear2.register_forward_hook(save_b)

print(f"设备：{DEVICE}")
print(f"Normal 轨迹：{N_NORMAL} 条")
print(f"Fault 轨迹：{len(FAULT_TYPES)} × {N_FAULT_PER_TYPE} = {len(FAULT_TYPES)*N_FAULT_PER_TYPE} 条")

# 梯度计算
def compute_gradients(state, model, device):
    """计算状态梯度"""
    state = torch.tensor(state, dtype=torch.float32).to(device)
    state.requires_grad = True
    inp = {"observation.state": state.unsqueeze(0), "observation.images.top": torch.randn(1,3,480,640).float().to(device)}
    action = model.select_action(inp)[0]
    action.sum().backward()
    grad = state.grad.cpu().numpy()
    return grad

# 收集函数
def collect_trajectory(scene, fault_type, seed, output_dir):
    """收集单条轨迹"""
    env = ALOHASimulationEnv(scene=scene, fault_type=fault_type, seed=seed)
    state = env.reset()
    
    states, actions, grads, hook_as, hook_bs = [], [], [], [], []
    
    for step in range(MAX_STEPS):
        # 计算梯度 (每 5 步)
        if step % 5 == 0:
            grad = compute_gradients(state, act_model, DEVICE)
        else:
            grad = grads[-1] if grads else np.zeros(14)
        grads.append(grad)
        
        # ACT 推理
        with torch.no_grad():
            inp = {"observation.state": torch.tensor(state).unsqueeze(0).float().to(DEVICE),
                   "observation.images.top": torch.randn(1,3,480,640).float().to(DEVICE)}
            action = act_model.select_action(inp)[0].cpu().numpy()
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        states.append(state.copy())
        actions.append(action.copy())
        if hooks['hook_a'] is not None: hook_as.append(hooks['hook_a'].copy())
        if hooks['hook_b'] is not None: hook_bs.append(hooks['hook_b'].copy())
        
        state = next_state
        if done: break
    
    # 保存
    traj_id = len(glob.glob(str(output_dir / "*.npz")))
    data = {
        'fault_type': fault_type or 'normal',
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'gradients': np.array(grads, dtype=np.float32),
    }
    if hook_as: data['hook_a'] = np.array(hook_as, dtype=np.float16)
    if hook_bs: data['hook_b'] = np.array(hook_bs, dtype=np.float16)
    
    output_file = output_dir / f"traj_{traj_id:04d}_{fault_type or 'normal'}.npz"
    np.savez_compressed(str(output_file), **data)
    
    return len(states), info.get('collision', False)

# 收集 Normal 轨迹
print("\n" + "="*60)
print("收集 Normal 轨迹...")
print("="*60)
start = time.time()
for i in range(N_NORMAL):
    steps, collision = collect_trajectory('empty', None, seed=42+i, output_dir=NORMAL_DIR)
    if (i+1) % 50 == 0:
        elapsed = (time.time()-start)/60
        print(f"[{i+1}/{N_NORMAL}] 已收集 {i+1} 条，{elapsed:.1f}分钟")

print(f"Normal 完成：{N_NORMAL} 条，耗时 {(time.time()-start)/3600:.2f}小时")

# 收集 Fault 轨迹
print("\n" + "="*60)
print("收集 Fault 轨迹...")
print("="*60)
traj_id = 0
start = time.time()
for fault_type in FAULT_TYPES:
    print(f"\n故障类型：{fault_type}")
    for i in range(N_FAULT_PER_TYPE):
        steps, collision = collect_trajectory('empty', fault_type, seed=1000+traj_id, output_dir=FAULT_DIR)
        traj_id += 1
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{N_FAULT_PER_TYPE}]")
    
    elapsed = (time.time()-start)/3600
    print(f"  {fault_type} 完成，累计 {elapsed:.2f}小时")

print(f"\nFault 完成：{len(FAULT_TYPES)*N_FAULT_PER_TYPE} 条，耗时 {(time.time()-start)/3600:.2f}小时")
print("\n" + "="*60)
print("数据收集完成!")
print(f"Normal: {NORMAL_DIR} ({N_NORMAL}条)")
print(f"Fault: {FAULT_DIR} ({len(FAULT_TYPES)*N_FAULT_PER_TYPE}条)")
print("="*60)
