#!/usr/bin/env python3
"""
具身智能场景渲染脚本 (Matplotlib 3D 版本)
无需 GPU/X11，纯 CPU 渲染
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

output_dir = '/home/vipuser/Embodied-RTA/outputs/figures/realistic_renders'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

print("="*60)
print("具身智能场景渲染 (Matplotlib 3D)")
print("="*60)

def add_box(ax, center, size, color, alpha=1.0):
    """添加 3D 盒子"""
    x, y, z = center
    dx, dy, dz = size
    
    # 8 个顶点
    vertices = [
        [x-dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y+dy/2, z+dz/2],
        [x-dx/2, y+dy/2, z+dz/2],
    ]
    
    # 6 个面
    faces = [
        [0, 1, 2, 3],  # 底面
        [4, 5, 6, 7],  # 顶面
        [0, 1, 5, 4],  # 前面
        [2, 3, 7, 6],  # 后面
        [0, 3, 7, 4],  # 左面
        [1, 2, 6, 5],  # 右面
    ]
    
    face_verts = [[vertices[i] for i in face] for face in faces]
    collection = Poly3DCollection(face_verts, facecolors=color, alpha=alpha, linewidths=0.5, edgecolors='black')
    ax.add_collection3d(collection)

def add_cylinder(ax, center, radius, height, color, alpha=1.0, n_points=20):
    """添加 3D 圆柱体"""
    x, y, z = center
    theta = np.linspace(0, 2*np.pi, n_points)
    
    # 侧面
    for i in range(n_points-1):
        verts = [
            [x + radius*np.cos(theta[i]), y + radius*np.sin(theta[i]), z],
            [x + radius*np.cos(theta[i+1]), y + radius*np.sin(theta[i+1]), z],
            [x + radius*np.cos(theta[i+1]), y + radius*np.sin(theta[i+1]), z+height],
            [x + radius*np.cos(theta[i]), y + radius*np.sin(theta[i]), z+height],
        ]
        collection = Poly3DCollection([verts], facecolors=color, alpha=alpha, linewidths=0)
        ax.add_collection3d(collection)
    
    # 顶面和底面
    circle_top = np.column_stack([x + radius*np.cos(theta), y + radius*np.sin(theta), [z+height]*n_points])
    circle_bottom = np.column_stack([x + radius*np.cos(theta), y + radius*np.sin(theta), [z]*n_points])
    
    # 简化：用多边形近似
    for i in range(n_points-2):
        # 顶面
        verts = [circle_top[0], circle_top[i+1], circle_top[i+2]]
        collection = Poly3DCollection([verts], facecolors=color, alpha=alpha, linewidths=0)
        ax.add_collection3d(collection)
        # 底面
        verts = [circle_bottom[0], circle_bottom[i+2], circle_bottom[i+1]]
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
    x, y, z = position
    # 身体
    add_cylinder(ax, [x, y, z], 0.2, 1.6, color, alpha)
    # 头部
    add_sphere(ax, [x, y, z+1.75], 0.15, color, alpha)

def render_scene(scene_name, setup_func, camera_angle=(30, 45), save_name=None, 
                 xlim=(-5, 15), ylim=(-5, 5), zlim=(0, 5), bg_color='white'):
    """渲染场景"""
    
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置背景
    ax.set_facecolor(bg_color)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 调用场景设置函数
    setup_func(ax)
    
    # 设置视角
    ax.view_init(elev=camera_angle[0], azim=camera_angle[1])
    
    # 设置范围
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    # 隐藏坐标轴
    ax.set_axis_off()
    
    # 保存
    if save_name is None:
        save_name = scene_name
    
    save_path = f'{output_dir}/{save_name}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=bg_color)
    plt.close()
    
    print(f"  ✓ {scene_name} → {save_path}")
    return save_path


# ============ 场景定义 ============
print("\n生成场景...")

# 场景 0: 正常场景
def scene_normal(ax):
    # 地面
    add_box(ax, [5, 0, -0.05], [20, 10, 0.1], (0.4, 0.4, 0.4))
    # 机器人 (蓝色底盘 + 灰色机械臂)
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    add_cylinder(ax, [0, 0, 0.45], 0.1, 0.5, (0.5, 0.5, 0.5))
    add_sphere(ax, [0, 0, 0.95], 0.08, (0.7, 0.7, 0.7))
    # 目标
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.8, 0))

render_scene("正常场景", scene_normal, camera_angle=(30, 45), save_name="00_normal_view")

# 场景 1: S1 光照突变
def scene_s1(ax):
    # 地面 (暗)
    add_box(ax, [5, 0, -0.05], [20, 10, 0.1], (0.15, 0.15, 0.15))
    # 机器人
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    add_cylinder(ax, [0, 0, 0.45], 0.1, 0.5, (0.5, 0.5, 0.5))
    add_sphere(ax, [0, 0, 0.95], 0.08, (0.7, 0.7, 0.7))
    # 目标 (暗淡)
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.4, 0))

render_scene("S1 光照突变", scene_s1, camera_angle=(30, 45), save_name="01_s1_lighting_drop", bg_color='#1a1a1a')

# 场景 2: S2 摄像头遮挡 (第一人称视角)
def scene_s2(ax):
    # 地面
    add_box(ax, [5, 0, -0.05], [20, 10, 0.1], (0.4, 0.4, 0.4))
    # 机器人 (近景)
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    add_cylinder(ax, [0, 0, 0.45], 0.1, 0.5, (0.5, 0.5, 0.5))
    # 遮挡 (黑色矩形)
    add_box(ax, [-1, -2, 1.5], [0.1, 2, 1.5], (0, 0, 0), alpha=0.9)
    # 目标
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.8, 0))

render_scene("S2 摄像头遮挡", scene_s2, camera_angle=(20, 0), xlim=(-3, 15), ylim=(-5, 5), save_name="02_s2_camera_occlusion")

# 场景 3: S3 对抗补丁
def scene_s3(ax):
    # 地面
    add_box(ax, [5, 0, -0.05], [20, 10, 0.1], (0.4, 0.4, 0.4))
    # 桌子 (棕色)
    add_box(ax, [5, 0, 0.35], [2, 2, 0.1], (0.6, 0.4, 0.2))
    # 对抗补丁 (棋盘格简化为黑色方块)
    add_box(ax, [5, 0, 0.42], [0.4, 0.4, 0.02], (0, 0, 0))
    # 机器人
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    add_cylinder(ax, [0, 0, 0.45], 0.1, 0.5, (0.5, 0.5, 0.5))

render_scene("S3 对抗补丁", scene_s3, camera_angle=(30, 45), save_name="03_s3_adversarial_patch")

# 场景 4: S4 负载突变
def scene_s4(ax):
    # 地面
    add_box(ax, [5, 0, -0.05], [20, 10, 0.1], (0.4, 0.4, 0.4))
    # 机器人
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    add_cylinder(ax, [0, 0, 0.45], 0.1, 0.5, (0.5, 0.5, 0.5))
    add_sphere(ax, [0, 0, 0.95], 0.08, (0.7, 0.7, 0.7))
    # 负载 (红色球体)
    add_sphere(ax, [0.3, 0, 0.9], 0.15, (1, 0, 0))
    # 目标
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.8, 0))

render_scene("S4 负载突变", scene_s4, camera_angle=(30, 45), save_name="04_s4_payload_shift")

# 场景 5: S5 关节摩擦
def scene_s5(ax):
    # 地面
    add_box(ax, [5, 0, -0.05], [20, 10, 0.1], (0.4, 0.4, 0.4))
    # 机器人 (机械臂异常角度)
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    add_cylinder(ax, [0.1, 0, 0.5], 0.1, 0.5, (0.5, 0.5, 0.5))  # 偏移表示滞后
    add_sphere(ax, [0.1, 0, 0.95], 0.08, (0.7, 0.7, 0.7))
    # 目标
    add_sphere(ax, [10, 0, 0.5], 0.3, (0, 0.8, 0))

render_scene("S5 关节摩擦", scene_s5, camera_angle=(30, 45), save_name="05_s5_joint_friction")

# 场景 6: S6 密集人群
def scene_s6(ax):
    # 地面
    add_box(ax, [7, 0, -0.05], [20, 10, 0.1], (0.4, 0.4, 0.4))
    # 机器人
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    # 5 个行人
    pedestrian_positions = [[5, 2, 0], [6, -2, 0], [7, 1.5, 0], [8, -1.5, 0], [9, 0, 0]]
    for pos in pedestrian_positions:
        add_person(ax, pos)
    # 目标
    add_sphere(ax, [14, 0, 0.5], 0.3, (0, 0.8, 0))

render_scene("S6 密集人群", scene_s6, camera_angle=(35, 35), xlim=(-2, 15), ylim=(-4, 4), save_name="06_s6_dynamic_crowd")

# 场景 7: S7 极窄通道
def scene_s7(ax):
    # 地面
    add_box(ax, [7, 0, -0.05], [20, 10, 0.1], (0.4, 0.4, 0.4))
    # 机器人
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    # 墙壁 (灰色)
    add_box(ax, [7.5, 0.65, 1.5], [15, 0.1, 3], (0.6, 0.6, 0.6))
    add_box(ax, [7.5, -0.65, 1.5], [15, 0.1, 3], (0.6, 0.6, 0.6))
    # 盲区障碍物 (红色)
    add_box(ax, [10, 0.8, 0.4], [0.5, 0.5, 0.8], (1, 0, 0))
    # 目标
    add_sphere(ax, [14, 0, 0.5], 0.3, (0, 0.8, 0))

render_scene("S7 极窄通道", scene_s7, camera_angle=(25, 0), xlim=(-2, 15), ylim=(-2, 2), save_name="07_s7_narrow_corridor")

# 场景 8: S8 复合灾难
def scene_s8(ax):
    # 地面 (暗)
    add_box(ax, [7, 0, -0.05], [20, 10, 0.1], (0.15, 0.15, 0.15))
    # 机器人带负载
    add_box(ax, [0, 0, 0.15], [0.6, 0.6, 0.3], (0.2, 0.4, 0.8))
    add_cylinder(ax, [0, 0, 0.45], 0.1, 0.5, (0.5, 0.5, 0.5))
    add_sphere(ax, [0.3, 0, 0.9], 0.15, (1, 0, 0))  # 负载
    # 窄通道墙壁
    add_box(ax, [7.5, 0.55, 1.5], [15, 0.1, 3], (0.6, 0.6, 0.6))
    add_box(ax, [7.5, -0.55, 1.5], [15, 0.1, 3], (0.6, 0.6, 0.6))
    # 行人
    add_person(ax, [8, 0.3, 0])
    add_person(ax, [9, -0.3, 0])
    # 目标 (暗淡)
    add_sphere(ax, [14, 0, 0.5], 0.3, (0, 0.4, 0))

render_scene("S8 复合灾难", scene_s8, camera_angle=(30, 35), xlim=(-2, 15), ylim=(-2, 2), zlim=(0, 4), save_name="08_s8_compound_hell", bg_color='#1a1a1a')

# ============ 完成 ============
print("\n" + "="*60)
print("渲染完成!")
print(f"输出目录：{output_dir}")
print("="*60)

# 列出所有生成的文件
import subprocess
result = subprocess.run(['ls', '-lh', output_dir], capture_output=True, text=True)
print(result.stdout)
