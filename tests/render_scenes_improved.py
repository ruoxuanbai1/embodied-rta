#!/usr/bin/env python3
"""
具身智能场景渲染脚本 (改进版)
添加纹理、家具、灯光效果
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

output_dir = '/home/vipuser/Embodied-RTA/outputs/figures/realistic_renders_improved'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['figure.facecolor'] = '#f0f0f0'
plt.rcParams['axes.facecolor'] = '#f0f0f0'

print("="*60)
print("具身智能场景渲染 (改进版)")
print("="*60)

def add_box(ax, center, size, color, alpha=1.0, edgecolor='black', linewidth=0.5):
    """添加 3D 盒子"""
    x, y, z = center
    dx, dy, dz = size
    
    vertices = [
        [x-dx/2, y-dy/2, z-dz/2], [x+dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y+dy/2, z-dz/2], [x-dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y-dy/2, z+dz/2], [x+dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y+dy/2, z+dz/2], [x-dx/2, y+dy/2, z+dz/2],
    ]
    
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5],
    ]
    
    face_verts = [[vertices[i] for i in face] for face in faces]
    collection = Poly3DCollection(face_verts, facecolors=color, alpha=alpha, 
                                   linewidths=linewidth, edgecolors=edgecolor)
    ax.add_collection3d(collection)

def add_cylinder(ax, center, radius, height, color, alpha=1.0, n_points=20):
    """添加 3D 圆柱体"""
    x, y, z = center
    theta = np.linspace(0, 2*np.pi, n_points)
    
    for i in range(n_points-1):
        verts = [
            [x + radius*np.cos(theta[i]), y + radius*np.sin(theta[i]), z],
            [x + radius*np.cos(theta[i+1]), y + radius*np.sin(theta[i+1]), z],
            [x + radius*np.cos(theta[i+1]), y + radius*np.sin(theta[i+1]), z+height],
            [x + radius*np.cos(theta[i]), y + radius*np.sin(theta[i]), z+height],
        ]
        collection = Poly3DCollection([verts], facecolors=color, alpha=alpha, linewidths=0)
        ax.add_collection3d(collection)

def add_sphere(ax, center, radius, color, alpha=1.0, n_points=15):
    """添加 3D 球体"""
    x, y, z = center
    phi = np.linspace(0, np.pi, n_points)
    theta = np.linspace(0, 2*np.pi, n_points)
    
    for i in range(n_points-2):
        for j in range(n_points-1):
            verts = [
                [x + radius*np.sin(phi[i])*np.cos(theta[j]), y + radius*np.sin(phi[i])*np.sin(theta[j]), z + radius*np.cos(phi[i])],
                [x + radius*np.sin(phi[i+1])*np.cos(theta[j]), y + radius*np.sin(phi[i+1])*np.sin(theta[j]), z + radius*np.cos(phi[i+1])],
                [x + radius*np.sin(phi[i+1])*np.cos(theta[j+1]), y + radius*np.sin(phi[i+1])*np.sin(theta[j+1]), z + radius*np.cos(phi[i+1])],
            ]
            collection = Poly3DCollection([verts], facecolors=color, alpha=alpha, linewidths=0)
            ax.add_collection3d(collection)
            
            verts = [
                [x + radius*np.sin(phi[i])*np.cos(theta[j]), y + radius*np.sin(phi[i])*np.sin(theta[j]), z + radius*np.cos(phi[i])],
                [x + radius*np.sin(phi[i+1])*np.cos(theta[j+1]), y + radius*np.sin(phi[i+1])*np.sin(theta[j+1]), z + radius*np.cos(phi[i+1])],
                [x + radius*np.sin(phi[i])*np.cos(theta[j+1]), y + radius*np.sin(phi[i])*np.sin(theta[j+1]), z + radius*np.cos(phi[i])],
            ]
            collection = Poly3DCollection([verts], facecolors=color, alpha=alpha, linewidths=0)
            ax.add_collection3d(collection)

def add_person(ax, position, color=(0.8, 0.6, 0.4), alpha=1.0):
    """添加行人"""
    if len(position) == 2:
        position = [position[0], position[1], 0]
    x, y, z = position
    add_cylinder(ax, [x, y, z], 0.2, 1.6, color, alpha)
    add_sphere(ax, [x, y, z+1.75], 0.15, color, alpha)

def add_table(ax, position, size=(1.0, 0.6, 0.75)):
    """添加桌子"""
    if len(position) == 2:
        position = [position[0], position[1], 0]
    x, y, z = position
    # 桌面
    add_box(ax, [x, y, z+size[2]], [size[0], size[1], 0.05], (0.6, 0.4, 0.2))
    # 桌腿
    leg_positions = [
        [x-size[0]/2+0.1, y-size[1]/2+0.1],
        [x+size[0]/2-0.1, y-size[1]/2+0.1],
        [x+size[0]/2-0.1, y+size[1]/2-0.1],
        [x-size[0]/2+0.1, y+size[1]/2-0.1],
    ]
    for lx, ly in leg_positions:
        add_cylinder(ax, [lx, ly, z], 0.03, size[2], (0.3, 0.3, 0.3))

def add_chair(ax, position):
    """添加椅子"""
    if len(position) == 2:
        position = [position[0], position[1], 0]
    x, y, z = position
    # 座位
    add_box(ax, [x, y, z+0.45], [0.4, 0.4, 0.05], (0.5, 0.35, 0.2))
    # 椅腿
    for dx, dy in [(-0.15, -0.15), (0.15, -0.15), (0.15, 0.15), (-0.15, 0.15)]:
        add_cylinder(ax, [x+dx, y+dy, z], 0.02, 0.45, (0.3, 0.3, 0.3))
    # 靠背
    add_box(ax, [x, y+0.2, z+0.7], [0.4, 0.05, 0.5], (0.5, 0.35, 0.2))

def add_plant(ax, position):
    """添加植物"""
    if len(position) == 2:
        position = [position[0], position[1], 0]
    x, y, z = position
    # 花盆
    add_cylinder(ax, [x, y, z+0.1], 0.15, 0.2, (0.6, 0.3, 0.1))
    # 植物 (绿色球体簇)
    for _ in range(5):
        dx, dy = np.random.uniform(-0.1, 0.1, 2)
        add_sphere(ax, [x+dx, y+dy, z+0.3], 0.1, (0.2, 0.6, 0.2))

def add_grid_floor(ax, size=(20, 10), grid_size=1.0):
    """添加网格地板"""
    sx, sy = size
    
    # 地板底色
    add_box(ax, [sx/2, 0, -0.05], [sx, sy, 0.1], (0.7, 0.7, 0.7), alpha=0.3)
    
    # 网格线 (简化为细盒子)
    for x in np.arange(0, sx, grid_size):
        add_box(ax, [x, 0, -0.04], [0.02, sy, 0.08], (0.3, 0.3, 0.3), alpha=0.5, linewidth=0)
    for y in np.arange(-sy/2, sy/2, grid_size):
        add_box(ax, [sx/2, y, -0.04], [sx, 0.02, 0.08], (0.3, 0.3, 0.3), alpha=0.5, linewidth=0)

def add_robot(ax, position, with_payload=False):
    """添加机器人"""
    if len(position) == 2:
        position = [position[0], position[1], 0]
    x, y, z = position
    # 底盘 (蓝色)
    add_box(ax, [x, y, z+0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    # 机械臂 (灰色)
    add_cylinder(ax, [x, y, z+0.45], 0.1, 0.5, (0.5, 0.5, 0.5))
    # 末端 (银色)
    add_sphere(ax, [x, y, z+0.95], 0.08, (0.7, 0.7, 0.7))
    # 负载 (红色)
    if with_payload:
        add_sphere(ax, [x+0.3, y, z+0.9], 0.15, (1, 0, 0))

def render_scene(scene_name, setup_func, camera_angle=(30, 45), save_name=None,
                 xlim=(-2, 15), ylim=(-5, 5), zlim=(0, 5), bg_color='#f0f0f0'):
    """渲染场景"""
    
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_facecolor(bg_color)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    setup_func(ax)
    
    ax.view_init(elev=camera_angle[0], azim=camera_angle[1])
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    ax.set_axis_off()
    
    if save_name is None:
        save_name = scene_name
    
    save_path = f'{output_dir}/{save_name}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=bg_color)
    plt.close()
    
    print(f"  ✓ {scene_name} → {save_path}")
    return save_path


# ============ 场景定义 ============
print("\n生成改进场景...")

# 场景 0: 正常场景 (带家具)
def scene_normal(ax):
    add_grid_floor(ax, (20, 10), 1.0)
    add_robot(ax, [0, 0], with_payload=False)
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.8, 0))  # 目标
    # 家具
    add_table(ax, [3, 3])
    add_chair(ax, [3, 2])
    add_chair(ax, [3, 4])
    add_plant(ax, [12, 3])
    add_plant(ax, [12, -3])

render_scene("正常场景", scene_normal, camera_angle=(30, 45), save_name="00_normal_view_improved")

# 场景 1: S1 光照突变 (暗背景 + 聚光灯)
def scene_s1(ax):
    add_grid_floor(ax, (20, 10), 1.0)
    add_robot(ax, [0, 0], with_payload=False)
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.4, 0))  # 暗淡目标
    # 家具 (暗淡)
    add_table(ax, [3, 3])
    add_plant(ax, [12, 3])

render_scene("S1 光照突变", scene_s1, camera_angle=(30, 45), save_name="01_s1_lighting_drop_improved", bg_color='#1a1a1a')

# 场景 2: S2 摄像头遮挡 (第一人称 + 家具)
def scene_s2(ax):
    add_grid_floor(ax, (20, 10), 1.0)
    add_robot(ax, [0, 0], with_payload=False)
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.8, 0))
    # 遮挡物 (黑色半透明)
    add_box(ax, [-1, -2, 1.5], [0.1, 2, 1.5], (0, 0, 0), alpha=0.9)
    # 家具
    add_table(ax, [5, 3])
    add_chair(ax, [5, 2])

render_scene("S2 摄像头遮挡", scene_s2, camera_angle=(20, 0), xlim=(-3, 15), ylim=(-5, 5), save_name="02_s2_camera_occlusion_improved")

# 场景 3: S3 对抗补丁 (桌子特写)
def scene_s3(ax):
    add_grid_floor(ax, (20, 10), 1.0)
    add_robot(ax, [0, 0], with_payload=False)
    # 桌子
    add_table(ax, [5, 0])
    # 对抗补丁 (棋盘格简化)
    add_box(ax, [5, 0, 0.78], [0.4, 0.4, 0.02], (0, 0, 0))
    # 装饰
    add_plant(ax, [8, 3])

render_scene("S3 对抗补丁", scene_s3, camera_angle=(30, 45), save_name="03_s3_adversarial_patch_improved")

# 场景 4: S4 负载突变 (带红色负载)
def scene_s4(ax):
    add_grid_floor(ax, (20, 10), 1.0)
    add_robot(ax, [0, 0], with_payload=True)  # 带负载
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.8, 0))
    # 家具
    add_table(ax, [3, 3])
    add_chair(ax, [3, 2])

render_scene("S4 负载突变", scene_s4, camera_angle=(30, 45), save_name="04_s4_payload_shift_improved")

# 场景 5: S5 关节摩擦 (机械臂偏移)
def scene_s5(ax):
    add_grid_floor(ax, (20, 10), 1.0)
    # 机器人 (机械臂偏移)
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    add_cylinder(ax, [0.15, 0, 0.45], 0.1, 0.5, (0.5, 0.5, 0.5))  # 偏移
    add_sphere(ax, [0.15, 0, 0.95], 0.08, (0.7, 0.7, 0.7))
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.8, 0))
    # 家具
    add_table(ax, [5, 3])

render_scene("S5 关节摩擦", scene_s5, camera_angle=(30, 45), save_name="05_s5_joint_friction_improved")

# 场景 6: S6 密集人群 (多行人 + 家具)
def scene_s6(ax):
    add_grid_floor(ax, (20, 10), 1.0)
    add_robot(ax, [0, 0], with_payload=False)
    # 5 个行人
    pedestrian_positions = [[5, 2, 0], [6, -2, 0], [7, 1.5, 0], [8, -1.5, 0], [9, 0, 0]]
    for pos in pedestrian_positions:
        add_person(ax, pos)
    add_sphere(ax, [14, 0, 0.5], 0.3, (0, 0.8, 0))
    # 家具
    add_table(ax, [3, 3])
    add_chair(ax, [3, 2])
    add_plant(ax, [12, 3])

render_scene("S6 密集人群", scene_s6, camera_angle=(35, 35), xlim=(-2, 15), ylim=(-4, 4), save_name="06_s6_dynamic_crowd_improved")

# 场景 7: S7 极窄通道 (带纹理墙壁)
def scene_s7(ax):
    add_grid_floor(ax, (20, 10), 1.0)
    add_robot(ax, [0, 0], with_payload=False)
    # 墙壁 (带边框)
    add_box(ax, [7.5, 0.65, 1.5], [15, 0.1, 3], (0.6, 0.6, 0.6), edgecolor='black', linewidth=1)
    add_box(ax, [7.5, -0.65, 1.5], [15, 0.1, 3], (0.6, 0.6, 0.6), edgecolor='black', linewidth=1)
    # 盲区障碍物
    add_box(ax, [10, 0.8, 0.4], [0.5, 0.5, 0.8], (1, 0, 0), edgecolor='black', linewidth=1)
    add_sphere(ax, [14, 0, 0.5], 0.3, (0, 0.8, 0))

render_scene("S7 极窄通道", scene_s7, camera_angle=(25, 0), xlim=(-2, 15), ylim=(-2, 2), save_name="07_s7_narrow_corridor_improved")

# 场景 8: S8 复合灾难 (暗背景 + 所有元素)
def scene_s8(ax):
    add_grid_floor(ax, (20, 10), 1.0)
    add_robot(ax, [0, 0], with_payload=True)  # 带负载
    # 窄通道
    add_box(ax, [7.5, 0.55, 1.5], [15, 0.1, 3], (0.6, 0.6, 0.6), edgecolor='black', linewidth=1)
    add_box(ax, [7.5, -0.55, 1.5], [15, 0.1, 3], (0.6, 0.6, 0.6), edgecolor='black', linewidth=1)
    # 行人
    add_person(ax, [8, 0.3, 0])
    add_person(ax, [9, -0.3, 0])
    # 暗淡目标
    add_sphere(ax, [14, 0, 0.5], 0.3, (0, 0.4, 0))
    # 家具
    add_table(ax, [3, 3])

render_scene("S8 复合灾难", scene_s8, camera_angle=(30, 35), xlim=(-2, 15), ylim=(-2, 2), zlim=(0, 4), save_name="08_s8_compound_hell_improved", bg_color='#1a1a1a')

# ============ 完成 ============
print("\n" + "="*60)
print("改进版渲染完成!")
print(f"输出目录：{output_dir}")
print("="*60)

import subprocess
result = subprocess.run(['ls', '-lh', output_dir], capture_output=True, text=True)
print(result.stdout)
