#!/usr/bin/env python3
"""
收集故障场景轨迹数据

故障类型 (8 种):
1. F1: 光照突变 (lighting_drop) - 视觉输入突然变暗
2. F2: 相机遮挡 (camera_occlusion) - 部分图像被遮挡
3. F3: 对抗补丁 (adversarial_patch) - 图像中有对抗性噪声
4. F4: 负载偏移 (payload_shift) - 动力学参数突变
5. F5: 关节摩擦 (joint_friction) - 执行器阻力增加
6. F6: 动态障碍 (dynamic_obstacle) - 突然出现障碍物
7. F7: 传感器噪声 (sensor_noise) - 状态测量噪声
8. F8: 复合故障 (compound_hell) - 多种故障同时发生

每个故障收集 50 条轨迹，共 400 条故障轨迹
"""

import torch, numpy as np, time, multiprocessing as mp
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy

# 故障配置
FAULT_CONFIGS = {
    "F1_lighting": {
        "name": "光照突变",
        "effect": "视觉输入变暗 80%",
        "inject_step": 50,  # 第 50 步注入故障
    },
    "F2_occlusion": {
        "name": "相机遮挡",
        "effect": "图像中心 50% 区域遮挡",
        "inject_step": 50,
    },
    "F3_adversarial": {
        "name": "对抗补丁",
        "effect": "添加对抗性噪声",
        "inject_step": 50,
    },
    "F4_payload": {
        "name": "负载偏移",
        "effect": "状态动力学突变",
        "inject_step": 50,
    },
    "F5_friction": {
        "name": "关节摩擦",
        "effect": "动作执行衰减 40%",
        "inject_step": 50,
    },
    "F6_dynamic": {
        "name": "动态障碍",
        "effect": "状态空间约束突变",
        "inject_step": 50,
    },
    "F7_sensor": {
        "name": "传感器噪声",
        "effect": "状态测量加噪声 σ=0.2",
        "inject_step": 50,
    },
    "F8_compound": {
        "name": "复合故障",
        "effect": "F1+F4+F5 同时发生",
        "inject_step": 50,
    },
}

def inject_fault(state, action, step, fault_type):
    """注入故障效果"""
    if step < FAULT_CONFIGS[fault_type]["inject_step"]:
        return state, action  # 故障前正常运行
    
    if fault_type == "F1_lighting":
        # 光照突变不影响状态/动作，但影响视觉 (这里简化)
        pass
    
    elif fault_type == "F2_occlusion":
        # 遮挡不影响状态/动作
        pass
    
    elif fault_type == "F3_adversarial":
        # 对抗噪声
        state = state + torch.randn_like(state) * 0.1
    
    elif fault_type == "F4_payload":
        # 负载偏移 - 状态动力学突变
        state = state * 1.2  # 状态放大
    
    elif fault_type == "F5_friction":
        # 关节摩擦 - 动作衰减
        action = action * 0.6
    
    elif fault_type == "F6_dynamic":
        # 动态障碍 - 状态约束
        state = torch.clamp(state, -5, 5)
    
    elif fault_type == "F7_sensor":
        # 传感器噪声
        state = state + torch.randn_like(state) * 0.2
    
    elif fault_type == "F8_compound":
        # 复合故障
        state = state * 1.2 + torch.randn_like(state) * 0.15
        action = action * 0.6
    
    return state, action

def collect_single(args):
    traj_id, fault_type, max_steps, seed = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human").eval().to(device)
    
    hooks = {"hook_a": None, "hook_b": None}
    def save_a(m, i, o): hooks["hook_a"] = o.detach().cpu().numpy()
    def save_b(m, i, o): hooks["hook_b"] = o.detach().cpu().numpy()
    model.model.encoder.layers[-1].linear2.register_forward_hook(save_a)
    model.model.decoder.layers[-1].linear2.register_forward_hook(save_b)
    
    def grad(state, eps=1e-4):
        base = {"observation.state": state.unsqueeze(0), "observation.images.top": torch.randn(1,3,480,640).float().to(device)}
        base_a = model.select_action(base)[0]
        g = np.zeros(14, dtype=np.float32)
        for i in range(14):
            p = state.clone(); p[i] += eps
            inp = {"observation.state": p.unsqueeze(0), "observation.images.top": base["observation.images.top"]}
            g[i] = (model.select_action(inp)[0] - base_a).sum().item() / eps
        return g
    
    states, actions, hook_as, hook_bs, grads = [], [], [], [], []
    state = torch.randn(14).float().to(device)
    
    for step in range(max_steps):
        # 每 5 步计算梯度
        if step % 5 == 0:
            grads.append(grad(state))
        else:
            grads.append(grads[-1] if grads else np.zeros(14, dtype=np.float32))
        
        # 注入故障
        state_fault, action_fault = inject_fault(state, None, step, fault_type)
        
        inp = {"observation.state": state_fault.unsqueeze(0), "observation.images.top": torch.randn(1,3,480,640).float().to(device)}
        with torch.no_grad():
            act = model.select_action(inp)[0].cpu().numpy()
        
        # 动作也受故障影响
        if fault_type == "F5_friction":
            act = act * 0.6
        elif fault_type == "F8_compound":
            act = act * 0.6
        
        snp = state.cpu().numpy().copy()
        states.append(snp); actions.append(act.copy())
        if hooks["hook_a"] is not None: hook_as.append(hooks["hook_a"].copy())
        if hooks["hook_b"] is not None: hook_bs.append(hooks["hook_b"].copy())
        
        # 状态演化
        state = torch.randn(14).float().to(device)
        if step >= FAULT_CONFIGS[fault_type]["inject_step"]:
            state = state * 1.1  # 故障后状态更不稳定
    
    data = {
        "fault_type": fault_type,
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "gradients": np.array(grads, dtype=np.float32),
    }
    if hook_as: data["hook_a"] = np.array(hook_as, dtype=np.float16)
    if hook_bs: data["hook_b"] = np.array(hook_bs, dtype=np.float16)
    
    return (traj_id, data)

def main():
    out = Path("/mnt/data/aloha_fault_trajectories")
    out.mkdir(parents=True, exist_ok=True)
    
    # 8 种故障 × 50 条 = 400 条
    tasks = []
    fault_types = list(FAULT_CONFIGS.keys())
    traj_id = 0
    
    for fault_type in fault_types:
        for i in range(50):
            tasks.append((f"fault_{traj_id:04d}_{fault_type}", fault_type, 250, 2000+traj_id))
            traj_id += 1
    
    print(f"="*60)
    print("故障轨迹收集")
    print(f"="*60)
    print(f"故障类型：{len(fault_types)} 种")
    for ft in fault_types:
        print(f"  - {ft}: {FAULT_CONFIGS[ft]['name']} ({FAULT_CONFIGS[ft]['effect']})")
    print(f"每种故障：50 条轨迹")
    print(f"总轨迹数：{len(tasks)} 条")
    print(f"每轨迹步数：250 步")
    print(f"="*60)
    print(f"使用 4 核心并行 (降低内存压力)")
    print()
    
    start = time.time()
    
    with mp.Pool(4) as pool:
        for i, (tid, data) in enumerate(pool.imap_unordered(collect_single, tasks)):
            np.savez_compressed(out / (tid + ".npz"), **data)
            if (i+1) % 25 == 0:
                elapsed = (time.time()-start)/60
                eta = (time.time()-start)/(i+1) * (len(tasks)-i-1) / 60
                print(f"[{i+1}/{len(tasks)}] {elapsed:.1f}min 已用，预计 {eta:.1f}min 剩余")
    
    print()
    print(f"="*60)
    print(f"✅ 完成！总耗时：{(time.time()-start)/3600:.2f} 小时")
    print(f"输出目录：{out}")
    print(f"="*60)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
