#!/usr/bin/env python3
"""
ALOHA 真实数据收集 - 基于 dm_control 仿真环境

使用 dm_control 的 ALOHA 环境生成真实状态，ACT 模型推理动作
"""

import torch, numpy as np, time, multiprocessing as mp
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
from dm_control import suite

def collect_single(args):
    """收集单条轨迹"""
    traj_id, fault_type, max_steps, seed = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载 ACT 模型
    model = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human").eval().to(device)
    
    # Hook 提取激活
    hooks = {"hook_a": None, "hook_b": None}
    def save_a(m, i, o): hooks["hook_a"] = o.detach().cpu().numpy()
    def save_b(m, i, o): hooks["hook_b"] = o.detach().cpu().numpy()
    model.model.encoder.layers[-1].linear2.register_forward_hook(save_a)
    model.model.decoder.layers[-1].linear2.register_forward_hook(save_b)
    
    # 加载 ALOHA 仿真环境 (dm_control)
    env = suite.load(domain_name="aloha", task_name="reach", 
                     task_kwargs={"random": seed})
    
    # 梯度计算
    def grad(state, eps=1e-4):
        base = {"observation.state": torch.tensor(state).unsqueeze(0).float().to(device),
                "observation.images.top": torch.randn(1,3,480,640).float().to(device)}
        base_a = model.select_action(base)[0]
        g = np.zeros(14, dtype=np.float32)
        for i in range(14):
            p = state.copy(); p[i] += eps
            inp = {"observation.state": torch.tensor(p).unsqueeze(0).float().to(device),
                   "observation.images.top": base["observation.images.top"]}
            g[i] = (model.select_action(inp)[0] - base_a).sum().item() / eps
        return g
    
    # 重置环境
    timestep = env.reset()
    states, actions, hook_as, hook_bs, grads = [], [], [], [], []
    
    for step in range(max_steps):
        # 获取真实仿真状态
        obs = timestep.observation
        state = np.concatenate([
            obs["qpos"][:7],  # 左臂
            obs["qpos"][7:14],  # 右臂  
            obs["qpos"][14:15],  # 左 gripper
            obs["qpos"][15:16],  # 右 gripper
        ]).astype(np.float32)
        
        # 注入故障
        if fault_type and step >= 50:
            if fault_type == "F4_payload":
                state = state * 1.3
            elif fault_type == "F5_friction":
                state = state * 0.7
            elif fault_type == "F7_sensor":
                state = state + np.random.randn(14) * 0.2
        
        # 计算梯度
        if step % 5 == 0:
            grads.append(grad(state))
        else:
            grads.append(grads[-1] if grads else np.zeros(14, dtype=np.float32))
        
        # ACT 推理
        inp = {"observation.state": torch.tensor(state).unsqueeze(0).float().to(device),
               "observation.images.top": torch.randn(1,3,480,640).float().to(device)}
        with torch.no_grad():
            act = model.select_action(inp)[0].cpu().numpy()
        
        # 执行动作
        action_env = np.zeros(14)
        action_env[:14] = act[:14]
        timestep = env.step(action_env)
        
        states.append(state.copy())
        actions.append(act.copy())
        if hooks["hook_a"] is not None: hook_as.append(hooks["hook_a"].copy())
        if hooks["hook_b"] is not None: hook_bs.append(hooks["hook_b"].copy())
    
    # 保存数据
    data = {
        "fault_type": fault_type or "normal",
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "gradients": np.array(grads, dtype=np.float32),
    }
    if hook_as: data["hook_a"] = np.array(hook_as, dtype=np.float16)
    if hook_bs: data["hook_b"] = np.array(hook_bs, dtype=np.float16)
    
    return (traj_id, data)

def main():
    out = Path("/mnt/data/aloha_real_dmcontrol")
    out.mkdir(parents=True, exist_ok=True)
    
    # 500 normal + 400 fault
    tasks = []
    for i in range(500):
        tasks.append((f"traj_{i:04d}_normal", "normal", 250, 42+i))
    
    faults = ["F1_lighting", "F2_occlusion", "F3_adversarial", "F4_payload", 
              "F5_friction", "F6_dynamic", "F7_sensor", "F8_compound"]
    for fi, f in enumerate(faults):
        for i in range(50):
            idx = 500 + fi * 50 + i
            tasks.append((f"traj_{idx:04d}_{f}", f, 250, 1000+idx))
    
    print(f"Total: {len(tasks)} trajectories")
    print(f"Using {mp.cpu_count()} cores")
    
    start = time.time()
    
    with mp.Pool(mp.cpu_count()) as pool:
        for i, (tid, data) in enumerate(pool.imap_unordered(collect_single, tasks)):
            np.savez_compressed(out / (tid + ".npz"), **data)
            if (i+1) % 50 == 0:
                elapsed = (time.time()-start)/60
                print(f"{i+1}/{len(tasks)} ({elapsed:.1f} min)")
    
    print(f"Done in {(time.time()-start)/3600:.2f} hours!")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
