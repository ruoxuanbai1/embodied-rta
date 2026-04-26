#!/usr/bin/env python3
"""
具身智能 8 维场景示例图生成
为每个场景生成可视化示意图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

output_dir = '/home/admin/.openclaw/workspace/Embodied-RTA/outputs/figures/scenario_examples'
os.makedirs(output_dir, exist_ok=True)

# ============ 场景配置 ============
SCENARIOS = {
    'S1': {'name': '严重光照突变', 'file': 's1_lighting_drop.png'},
    'S2': {'name': '摄像头遮挡/眩光', 'file': 's2_camera_occlusion.png'},
    'S3': {'name': '对抗补丁攻击', 'file': 's3_adversarial_patch.png'},
    'S4': {'name': '突发大负载变化', 'file': 's4_payload_shift.png'},
    'S5': {'name': '关节摩擦力激增', 'file': 's5_joint_friction.png'},
    'S6': {'name': '密集动态人群', 'file': 's6_dynamic_crowd.png'},
    'S7': {'name': '极窄通道 + 盲区窜出', 'file': 's7_narrow_corridor.png'},
    'S8': {'name': '复合灾难', 'file': 's8_compound_hell.png'},
}

# ============ S1: 光照突变 ============
def plot_s1_lighting():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 正常光照
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # 机器人
    robot = patches.Circle((5, 5), 0.5, color='blue', label='Robot')
    ax.add_patch(robot)
    
    # 目标
    ax.plot(9, 5, 'g*', markersize=20, label='Goal')
    
    # 障碍物
    for i in range(3):
        obs = patches.Circle((7, 3+i*2), 0.4, color='red', alpha=0.5)
        ax.add_patch(obs)
    
    # 光源 (正常)
    ax.plot(1, 9, 'y*', markersize=30, label='Light Source')
    ax.set_title('Normal Lighting', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 光照突变
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a1a')  # 暗背景
    
    # 机器人
    robot = patches.Circle((5, 5), 0.5, color='blue')
    ax.add_patch(robot)
    
    # 目标
    ax.plot(9, 5, 'g*', markersize=20)
    
    # 障碍物
    for i in range(3):
        obs = patches.Circle((7, 3+i*2), 0.4, color='red', alpha=0.8)
        ax.add_patch(obs)
    
    # 闪烁光源
    ax.plot(1, 9, 'y*', markersize=15, alpha=0.3)
    
    # 局部射灯效果
    cone = patches.Wedge((5, 5), 4, -30, 30, alpha=0.2, color='yellow')
    ax.add_patch(cone)
    
    ax.set_title('Severe Illumination Drop + Flickering', fontsize=14)
    ax.grid(True, alpha=0.1)
    
    plt.suptitle('S1: 严重光照突变 (Severe Illumination Drop)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/s1_lighting_drop.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============ S2: 摄像头遮挡 ============
def plot_s2_occlusion():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 正常视野
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # 相机视野锥
    cone = patches.Polygon([[5, 5], [9, 2], [9, 8]], closed=True, alpha=0.2, color='cyan')
    ax.add_patch(cone)
    
    # 机器人
    ax.plot(5, 5, 'bo', markersize=15, label='Robot')
    
    # 障碍物
    ax.plot(7, 5, 'rs', markersize=15, label='Obstacle')
    
    ax.set_title('Normal Vision', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 遮挡
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # 相机视野锥
    cone = patches.Polygon([[5, 5], [9, 2], [9, 8]], closed=True, alpha=0.1, color='cyan')
    ax.add_patch(cone)
    
    # 机器人
    ax.plot(5, 5, 'bo', markersize=15)
    
    # 障碍物
    ax.plot(7, 5, 'rs', markersize=15)
    
    # 遮挡区域 (左上角)
    rect = patches.Rectangle((0, 5), 5, 5, color='black', alpha=0.7, label='Occlusion')
    ax.add_patch(rect)
    
    # 眩光效果
    glare = patches.Circle((8, 3), 1.0, color='white', alpha=0.5, label='Lens Flare')
    ax.add_patch(glare)
    
    ax.set_title('Camera Occlusion + Lens Flare', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('S2: 摄像头遮挡/眩光 (Camera Occlusion / Lens Flare)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/s2_camera_occlusion.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============ S3: 对抗补丁 ============
def plot_s3_adversarial():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 正常桌面
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # 桌面
    table = patches.Rectangle((3, 3), 4, 4, color='brown', alpha=0.5, label='Table')
    ax.add_patch(table)
    
    # 机器人
    ax.plot(1, 5, 'bo', markersize=15, label='Robot')
    
    # 目标物体
    ax.plot(5, 5, 'g*', markersize=20, label='Target')
    
    # 机械臂轨迹
    ax.plot([1, 5], [5, 5], 'b--', linewidth=2, label='Planned Path')
    
    ax.set_title('Normal Table', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 对抗补丁
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # 桌面
    table = patches.Rectangle((3, 3), 4, 4, color='brown', alpha=0.5)
    ax.add_patch(table)
    
    # 机器人
    ax.plot(1, 5, 'bo', markersize=15)
    
    # 对抗补丁 (棋盘格图案)
    for i in range(4):
        for j in range(4):
            color = 'black' if (i+j) % 2 == 0 else 'white'
            square = patches.Rectangle((3.5+i*0.8, 3.5+j*0.8), 0.8, 0.8, color=color)
            ax.add_patch(square)
    
    # VLA 错误理解 (认为桌子是洞)
    ax.plot(5, 5, 'rx', markersize=30, label='VLA Perception: HOLE!')
    
    # 错误轨迹
    ax.plot([1, 5], [5, 2], 'r--', linewidth=2, label='Erratic Path')
    
    ax.set_title('Adversarial Patch Attack', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('S3: 隐蔽对抗补丁贴纸 (Adversarial Patch Attack)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/s3_adversarial_patch.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============ S4: 负载突变 ============
def plot_s4_payload():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 正常负载
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # 机器人底盘
    base = patches.Rectangle((4, 4), 2, 2, color='blue', alpha=0.5, label='Base')
    ax.add_patch(base)
    
    # 机械臂
    ax.plot([5, 5, 7], [5, 7, 7], 'b-', linewidth=3, label='Arm')
    
    # 末端执行器
    ax.plot(7, 7, 'go', markersize=10, label='End-effector')
    
    # 小负载
    ax.plot(7, 7, 'yo', markersize=8, label='Payload (0.5kg)')
    
    # 重心
    ax.plot(5, 5.5, 'r+', markersize=15, label='CoM')
    
    # ZMP 区域
    zmp = patches.Rectangle((4.2, 4.2), 1.6, 1.6, fill=False, edgecolor='green', linestyle='--', label='ZMP Safe')
    ax.add_patch(zmp)
    
    ax.set_title('Normal Payload (0.5kg)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 负载突变
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # 机器人底盘
    base = patches.Rectangle((4, 4), 2, 2, color='blue', alpha=0.5)
    ax.add_patch(base)
    
    # 机械臂
    ax.plot([5, 5, 7], [5, 7, 7], 'b-', linewidth=3)
    
    # 末端执行器
    ax.plot(7, 7, 'go', markersize=10)
    
    # 大负载 (5kg)
    ax.plot(7, 7, 'ro', markersize=20, label='Payload (5kg)')
    
    # 重心偏移
    ax.plot(5.3, 5.8, 'r+', markersize=20, label='CoM (shifted)')
    
    # ZMP 超出区域
    zmp = patches.Rectangle((4.2, 4.2), 1.6, 1.6, fill=False, edgecolor='red', linestyle='--', label='ZMP Unstable!')
    ax.add_patch(zmp)
    ax.plot(6.5, 5, 'rx', markersize=20)  # ZMP 超出
    
    ax.set_title('Sudden Payload Shift (5kg)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('S4: 突发大负载变化 (Sudden Payload Shift)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/s4_payload_shift.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============ S5: 关节摩擦 ============
def plot_s5_friction():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 正常关节
    ax = axes[0]
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    for i, angle in enumerate(angles):
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot(x, y, 'bo', markersize=15)
        ax.text(x*1.3, y*1.3, f'J{i+1}', ha='center', va='center')
    
    # 期望轨迹
    circle = patches.Circle((0, 0), 1, fill=False, color='green', linestyle='--', label='Expected Trajectory')
    ax.add_patch(circle)
    
    # 实际轨迹 (正常)
    ax.plot(np.cos(angles), np.sin(angles), 'g-', linewidth=2, label='Actual (Normal)')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Normal Joint Friction', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 摩擦激增
    ax = axes[1]
    
    for i, angle in enumerate(angles):
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot(x, y, 'bo', markersize=15)
        ax.text(x*1.3, y*1.3, f'J{i+1}', ha='center', va='center')
    
    # 期望轨迹
    circle = patches.Circle((0, 0), 1, fill=False, color='green', linestyle='--', label='Expected')
    ax.add_patch(circle)
    
    # 实际轨迹 (滞后)
    lagged_angles = angles - 0.3  # 滞后
    ax.plot(np.cos(lagged_angles), np.sin(lagged_angles), 'r-', linewidth=2, label='Actual (High Friction)')
    
    # 高亮受影响的关节
    ax.plot(np.cos(angles[1]), np.sin(angles[1]), 'ro', markersize=25, fillstyle='none', linewidth=3, label='J2, J3 Affected')
    ax.plot(np.cos(angles[2]), np.sin(angles[2]), 'ro', markersize=25, fillstyle='none', linewidth=3)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Joint Friction ×300%', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('S5: 关节摩擦力激增/电机老化 (Joint Friction / Actuator Degradation)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/s5_joint_friction.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============ S6: 密集人群 ============
def plot_s6_crowd():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 初始状态
    ax = axes[0]
    ax.set_xlim(0, 15)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    
    # 机器人
    ax.plot(1, 0, 'bo', markersize=15, label='Robot')
    
    # 目标
    ax.plot(14, 0, 'g*', markersize=20, label='Goal')
    
    # 行人 (初始位置)
    np.random.seed(42)
    pedestrians = []
    for i in range(5):
        x = np.random.uniform(5, 10)
        y = np.random.uniform(-3, 3)
        pedestrians.append((x, y))
        ax.plot(x, y, 'ro', markersize=12, alpha=0.5)
        ax.plot([x, x+0.5], [y, y+0.3], 'r-', linewidth=1, alpha=0.5)  # 速度向量
    
    ax.set_title('Initial State (5 Pedestrians)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 动态过程
    ax = axes[1]
    ax.set_xlim(0, 15)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    
    # 机器人轨迹
    ax.plot([1, 3, 5, 7, 9], [0, 0.5, -0.3, 0.8, -0.5], 'b-', linewidth=2, label='Robot Trajectory')
    ax.plot(9, -0.5, 'bo', markersize=15)
    
    # 目标
    ax.plot(14, 0, 'g*', markersize=20)
    
    # 行人轨迹
    for i, (x, y) in enumerate(pedestrians):
        # 随机游走轨迹
        traj_x = [x]
        traj_y = [y]
        cx, cy = x, y
        for _ in range(5):
            cx += np.random.uniform(-0.5, 0.5)
            cy += np.random.uniform(-0.5, 0.5)
            traj_x.append(cx)
            traj_y.append(cy)
        ax.plot(traj_x, traj_y, 'r-', linewidth=1, alpha=0.3)
        ax.plot(cx, cy, 'ro', markersize=12, alpha=0.5)
    
    ax.set_title('Dynamic Movement (Random Direction Changes)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('S6: 密集动态人群穿行 (Dense Dynamic Crowd)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/s6_dynamic_crowd.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============ S7: 窄通道 ============
def plot_s7_corridor():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 初始状态
    ax = axes[0]
    ax.set_xlim(0, 15)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    
    # 走廊墙壁
    ax.axhline(y=0.6, color='gray', linewidth=3, label='Wall')
    ax.axhline(y=-0.6, color='gray', linewidth=3)
    
    # 机器人
    ax.plot(2, 0, 'bo', markersize=15, label='Robot')
    
    # 目标
    ax.plot(14, 0, 'g*', markersize=20, label='Goal')
    
    # 盲区 (虚线)
    ax.axvline(x=8, color='orange', linestyle='--', linewidth=2, label='Blind Spot')
    ax.fill_between([8, 15], -0.6, 0.6, alpha=0.1, color='orange')
    
    ax.set_title('Approaching Blind Spot', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 盲区窜出
    ax = axes[1]
    ax.set_xlim(0, 15)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    
    # 走廊墙壁
    ax.axhline(y=0.6, color='gray', linewidth=3)
    ax.axhline(y=-0.6, color='gray', linewidth=3)
    
    # 机器人 (刹车)
    ax.plot(7, 0, 'bo', markersize=15)
    
    # 刹车轨迹
    ax.plot([7, 6.5], [0, 0], 'b--', linewidth=3, label='Braking')
    
    # 目标
    ax.plot(14, 0, 'g*', markersize=20)
    
    # 盲区障碍物 (推车)
    cart = patches.Rectangle((8.5, -0.3), 0.8, 0.6, color='red', alpha=0.7, label='Surprise Obstacle')
    ax.add_patch(cart)
    ax.plot(9, 0, 'r>', markersize=15)  # 运动方向
    
    # 碰撞警告区域
    ax.fill_between([7, 9.5], -0.6, 0.6, alpha=0.2, color='red')
    
    ax.set_title('Obstacle Emerges from Blind Spot', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('S7: 极窄通道与盲区窜出 (Narrow Corridor & Blind Spot Dash)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/s7_narrow_corridor.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============ S8: 复合灾难 ============
def plot_s8_compound():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 正常状态
    ax = axes[0]
    ax.set_xlim(0, 15)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    
    # 走廊
    ax.axhline(y=0.5, color='gray', linewidth=2)
    ax.axhline(y=-0.5, color='gray', linewidth=2)
    
    # 机器人 (带负载)
    ax.plot(3, 0, 'bo', markersize=15, label='Robot + Payload')
    ax.plot(3, 0, 'yo', markersize=10, alpha=0.5)  # 负载
    
    # 目标
    ax.plot(14, 0, 'g*', markersize=20, label='Goal')
    
    # 行人
    ax.plot(10, 0.3, 'ro', markersize=12, label='Pedestrian')
    
    ax.set_title('Before Compound Fault', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 复合故障状态
    ax = axes[1]
    ax.set_xlim(0, 15)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a1a')  # 暗背景 (光照丧失)
    
    # 走廊
    ax.axhline(y=0.5, color='gray', linewidth=2, alpha=0.5)
    ax.axhline(y=-0.5, color='gray', linewidth=2, alpha=0.5)
    
    # 机器人 (刹车)
    ax.plot(5, 0, 'bo', markersize=15)
    ax.plot(5, 0, 'yo', markersize=10, alpha=0.5)
    
    # 刹车轨迹
    ax.plot([5, 4.5], [0, 0], 'b--', linewidth=3, label='Emergency Brake')
    
    # 目标 (暗淡)
    ax.plot(14, 0, 'g*', markersize=20, alpha=0.3)
    
    # 行人 1
    ax.plot(8, 0.2, 'ro', markersize=12, alpha=0.8)
    
    # 行人 2 (盲区窜出)
    ax.plot(7, -0.3, 'ro', markersize=12, alpha=0.8, label='Surprise Pedestrian')
    
    # RTA 触发指示
    ax.text(5, 1.2, 'R3: OOD Detected!', ha='center', color='orange', fontsize=12, fontweight='bold')
    ax.text(5, 0.8, 'R2: Reachability Warning', ha='center', color='orange', fontsize=10)
    ax.text(5, 0.4, 'R1: Emergency Brake', ha='center', color='red', fontsize=10, fontweight='bold')
    
    ax.set_title('Compound Fault: Lighting + Obstacles + Payload', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.1)
    
    plt.suptitle('S8: 复合灾难 (Compound Extreme - S1+S4+S7)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/s8_compound_hell.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============ 生成所有场景图 ============
if __name__ == '__main__':
    print("生成 8 维场景示例图...")
    
    plot_s1_lighting()
    print("  ✓ S1: 严重光照突变")
    
    plot_s2_occlusion()
    print("  ✓ S2: 摄像头遮挡/眩光")
    
    plot_s3_adversarial()
    print("  ✓ S3: 对抗补丁攻击")
    
    plot_s4_payload()
    print("  ✓ S4: 突发大负载变化")
    
    plot_s5_friction()
    print("  ✓ S5: 关节摩擦力激增")
    
    plot_s6_crowd()
    print("  ✓ S6: 密集动态人群")
    
    plot_s7_corridor()
    print("  ✓ S7: 极窄通道 + 盲区窜出")
    
    plot_s8_compound()
    print("  ✓ S8: 复合灾难")
    
    print(f"\n✅ 所有场景图已保存到：{output_dir}")
