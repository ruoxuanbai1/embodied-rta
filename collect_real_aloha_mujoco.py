#!/usr/bin/env python3
"""
ALOHA 真实数据收集 - 基于 MuJoCo 仿真

使用 MuJoCo 加载 ALOHA 双臂机械臂模型，生成真实状态
"""

import torch, numpy as np, time, multiprocessing as mp
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
import mujoco

# ALOHA 双臂 MJCF 模型
ALPHA_MJCF = """
<mujoco model="aloha">
  <compiler angle="radian"/>
  <option timestep="0.02"/>
  
  <worldbody>
    <light name="top" pos="0 0 2"/>
    <body name="base" pos="0 0 0.5">
      <!-- 左臂 -->
      <body name="left_base" pos="-0.3 0 0">
        <joint name="left_joint_1" type="hinge" axis="0 0 1" damping="0.1"/>
        <body name="left_link_1" pos="0 0 0.15">
          <joint name="left_joint_2" type="hinge" axis="0 1 0" damping="0.1"/>
          <body name="left_link_2" pos="0 0 0.2">
            <joint name="left_joint_3" type="hinge" axis="0 0 1" damping="0.1"/>
            <body name="left_link_3" pos="0 0 0.15">
              <joint name="left_joint_4" type="hinge" axis="0 1 0" damping="0.1"/>
              <body name="left_link_4" pos="0 0 0.15">
                <joint name="left_joint_5" type="hinge" axis="0 0 1" damping="0.1"/>
                <body name="left_link_5" pos="0 0 0.1">
                  <joint name="left_joint_6" type="hinge" axis="0 1 0" damping="0.1"/>
                  <body name="left_gripper" pos="0 0 0.1">
                    <joint name="left_gripper_open" type="slide" axis="1 0 0" range="0 0.05" damping="0.5"/>
                    <geom type="box" size="0.02 0.01 0.01" rgba="0.3 0.3 0.8 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
      
      <!-- 右臂 -->
      <body name="right_base" pos="0.3 0 0">
        <joint name="right_joint_1" type="hinge" axis="0 0 1" damping="0.1"/>
        <body name="right_link_1" pos="0 0 0.15">
          <joint name="right_joint_2" type="hinge" axis="0 1 0" damping="0.1"/>
          <body name="right_link_2" pos="0 0 0.2">
            <joint name="right_joint_3" type="hinge" axis="0 0 1" damping="0.1"/>
            <body name="right_link_3" pos="0 0 0.15">
              <joint name="right_joint_4" type="hinge" axis="0 1 0" damping="0.1"/>
              <body name="right_link_4" pos="0 0 0.15">
                <joint name="right_joint_5" type="hinge" axis="0 0 1" damping="0.1"/>
                <body name="right_link_5" pos="0 0 0.1">
                  <joint name="right_joint_6" type="hinge" axis="0 1 0" damping="0.1"/>
                  <body name="right_gripper" pos="0 0 0.1">
                    <joint name="right_gripper_open" type="slide" axis="1 0 0" range="0 0.05" damping="0.5"/>
                    <geom type="box" size="0.02 0.01 0.01" rgba="0.8 0.3 0.3 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
      
      <!-- 桌面 -->
      <geom type="box" size="0.5 0.5 0.02" pos="0 0 -0.02" rgba="0.5 0.3 0.1 1"/>
      
      <!-- 目标物体 -->
      <body name="target" pos="0 0.3 0.02">
        <geom type="box" size="0.03 0.03 0.03" rgba="0.2 0.8 0.2 1"/>
        <joint type="free"/>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="left_1" joint="left_joint_1" ctrlrange="-1 1"/>
    <motor name="left_2" joint="left_joint_2" ctrlrange="-1 1"/>
    <motor name="left_3" joint="left_joint_3" ctrlrange="-1 1"/>
    <motor name="left_4" joint="left_joint_4" ctrlrange="-1 1"/>
    <motor name="left_5" joint="left_joint_5" ctrlrange="-1 1"/>
    <motor name="left_6" joint="left_joint_6" ctrlrange="-1 1"/>
    <motor name="left_gripper" joint="left_gripper_open" ctrlrange="-0.5 0.5"/>
    
    <motor name="right_1" joint="right_joint_1" ctrlrange="-1 1"/>
    <motor name="right_2" joint="right_joint_2" ctrlrange="-1 1"/>
    <motor name="right_3" joint="right_joint_3" ctrlrange="-1 1"/>
    <motor name="right_4" joint="right_joint_4" ctrlrange="-1 1"/>
    <motor name="right_5" joint="right_joint_5" ctrlrange="-1 1"/>
    <motor name="right_6" joint="right_joint_6" ctrlrange="-1 1"/>
    <motor name="right_gripper" joint="right_gripper_open" ctrlrange="-0.5 0.5"/>
  </actuator>
</mujoco>
"""

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
    
    # 加载 MuJoCo 模型
    model_mj = mujoco.MjModel.from_xml_string(ALPHA_MJCF)
    data = mujoco.MjData(model_mj)
    
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
    
    # 重置
    mujoco.mj_resetData(model_mj, data)
    states, actions, hook_as, hook_bs, grads = [], [], [], [], []
    
    for step in range(max_steps):
        # 获取真实仿真状态 (14 维：6+6+1+1)
        state = data.qpos[:14].copy().astype(np.float32)
        
        # 注入故障
        if fault_type and step >= 50:
            if fault_type == "F4_payload":
                data.qpos[:] *= 1.3
            elif fault_type == "F5_friction":
                data.ctrl[:] *= 0.6
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
        
        # 执行动作 (MuJoCo 仿真)
        data.ctrl[:14] = act[:14]
        mujoco.mj_step(model_mj, data)
        
        states.append(state.copy())
        actions.append(act.copy())
        if hooks["hook_a"] is not None: hook_as.append(hooks["hook_a"].copy())
        if hooks["hook_b"] is not None: hook_bs.append(hooks["hook_b"].copy())
    
    # 保存
    data_dict = {
        "fault_type": fault_type or "normal",
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "gradients": np.array(grads, dtype=np.float32),
    }
    if hook_as: data_dict["hook_a"] = np.array(hook_as, dtype=np.float16)
    if hook_bs: data_dict["hook_b"] = np.array(hook_bs, dtype=np.float16)
    
    return (traj_id, data_dict)

def main():
    out = Path("/mnt/data/aloha_real_mujoco")
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
    
    print(f"Total: {len(tasks)} trajectories (MuJoCo ALOHA)")
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
