#!/usr/bin/env python3
"""
具身智能场景真实渲染脚本
使用 pyrender + trimesh 生成照片级场景截图
"""

import numpy as np
import trimesh
import pyrender
import os
from PIL import Image

output_dir = '/home/vipuser/Embodied-RTA/outputs/figures/realistic_renders'
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("具身智能场景真实渲染")
print("="*60)

# ============ 基础场景设置 ============

def create_fetch_robot():
    """创建简化的 Fetch 机器人模型"""
    robot = trimesh.Trimesh()
    
    # 底盘 (蓝色盒子)
    base = trimesh.primitives.Box(extents=[0.6, 0.6, 0.3])
    base.apply_translation([0, 0, 0.15])  # 抬高
    base.visual.vertex_colors = [50, 100, 200, 255]
    robot = robot + base
    
    # 机械臂 (灰色圆柱体)
    arm = trimesh.primitives.Cylinder(radius=0.1, height=0.5)
    arm.apply_translation([0, 0, 0.65])
    arm.visual.vertex_colors = [80, 80, 80, 255]
    robot = robot + arm
    
    # 机械臂末端 (银色球体)
    end = trimesh.primitives.Sphere(radius=0.08)
    end.apply_translation([0, 0, 0.95])
    end.visual.vertex_colors = [150, 150, 150, 255]
    robot = robot + end
    
    return robot

def create_pedestrian(position):
    """创建行人模型"""
    person = trimesh.Trimesh()
    
    # 身体 (圆柱体)
    body = trimesh.primitives.Cylinder(radius=0.2, height=1.6)
    body.apply_translation([position[0], position[1], position[2] + 0.8])
    body.visual.vertex_colors = [200, 150, 100, 255]  # 肤色
    person = person + body
    
    # 头部 (球体)
    head = trimesh.primitives.Sphere(radius=0.15)
    head.apply_translation([position[0], position[1], position[2] + 1.75])
    head.visual.vertex_colors = [200, 150, 100, 255]
    person = person + head
    
    return person

def create_wall(position, size):
    """创建墙壁"""
    wall = trimesh.primitives.Box(extents=size)
    wall.apply_translation(position)
    wall.visual.vertex_colors = [180, 180, 180, 255]  # 灰色
    return wall

def create_floor():
    """创建地面"""
    floor = trimesh.primitives.Box(extents=[20, 20, 0.1])
    floor.apply_translation([0, 0, -0.05])
    floor.visual.vertex_colors = [100, 100, 100, 255]  # 深灰色
    return floor

def create_table(position):
    """创建桌子"""
    table = trimesh.primitives.Box(extents=[2, 2, 0.1])
    table.apply_translation(position)
    table.visual.vertex_colors = [150, 100, 50, 255]  # 棕色
    return table

def create_payload(position):
    """创建负载 (红色球体)"""
    payload = trimesh.primitives.Sphere(radius=0.15)
    payload.apply_translation(position)
    payload.visual.vertex_colors = [255, 0, 0, 255]  # 红色
    return payload

def create_obstacle(position):
    """创建障碍物 (红色盒子)"""
    obs = trimesh.primitives.Box(extents=[0.5, 0.5, 0.8])
    obs.apply_translation(position)
    obs.visual.vertex_colors = [255, 0, 0, 255]  # 红色
    return obs

def render_scene(scene_name, objects, camera_position, camera_lookat, 
                 lighting_intensity=1.0, save_name=None):
    """渲染场景"""
    
    # 创建场景
    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2],
                           bg_color=[0.95, 0.95, 0.95, 1.0])
    
    # 添加物体 (转换 trimesh 为 pyrender)
    for obj in objects:
        if isinstance(obj, trimesh.Trimesh):
            mesh = pyrender.Mesh.from_trimesh(obj, smooth=False)
            scene.add(mesh)
        else:
            scene.add(obj)
    
    # 添加相机
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_position
    
    # 相机看向目标点
    direction = np.array(camera_lookat) - np.array(camera_position)
    direction = direction / np.linalg.norm(direction)
    
    # 设置相机方向
    forward = -direction
    right = np.cross(forward, [0, 0, 1])
    right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else [1, 0, 0]
    up = np.cross(right, forward)
    
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = forward
    
    scene.add(camera, pose=camera_pose)
    
    # 添加主光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=lighting_intensity)
    light_pose = np.eye(4)
    light_pose[:3, 3] = camera_position
    scene.add(light, pose=light_pose)
    
    # 添加点光源 (模拟环境光)
    point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.3 * lighting_intensity)
    scene.add(point_light, pose=np.eye(4))
    
    # 渲染
    renderer = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    
    # 保存
    if save_name is None:
        save_name = scene_name
    
    image = Image.fromarray(color)
    save_path = f'{output_dir}/{save_name}.png'
    image.save(save_path)
    
    print(f"  ✓ {scene_name} → {save_path}")
    return save_path


# ============ 场景渲染 ============
print("\n生成场景...")

# 场景 0: 正常场景
objects = [create_floor(), create_fetch_robot()]
render_scene("正常场景", objects, 
             camera_position=[5, -8, 3],
             camera_lookat=[5, 0, 1],
             save_name="00_normal_view")

# 场景 1: S1 光照突变
objects = [create_floor(), create_fetch_robot()]
render_scene("S1 光照突变", objects,
             camera_position=[5, -8, 3],
             camera_lookat=[5, 0, 1],
             lighting_intensity=0.1,  # 90% 光照丧失
             save_name="01_s1_lighting_drop")

# 场景 2: S2 摄像头遮挡 (用第一人称视角表示)
objects = [create_floor(), create_fetch_robot()]
render_scene("S2 摄像头遮挡", objects,
             camera_position=[0, 0, 1.5],  # 机器人视角
             camera_lookat=[5, 0, 1],
             save_name="02_s2_camera_occlusion")

# 场景 3: S3 对抗补丁
objects = [create_floor(), create_table([5, 0, 0.35]), 
           create_payload([5, 0, 0.45]), create_fetch_robot()]
render_scene("S3 对抗补丁", objects,
             camera_position=[5, -8, 3],
             camera_lookat=[5, 0, 1],
             save_name="03_s3_adversarial_patch")

# 场景 4: S4 负载突变
robot = create_fetch_robot()
robot = robot + create_payload([0.3, 0, 0.85])  # 红色负载
objects = [create_floor(), robot]
render_scene("S4 负载突变", objects,
             camera_position=[5, -8, 3],
             camera_lookat=[5, 0, 1],
             save_name="04_s4_payload_shift")

# 场景 5: S5 关节摩擦
robot = create_fetch_robot()
objects = [create_floor(), robot]
render_scene("S5 关节摩擦", objects,
             camera_position=[5, -8, 3],
             camera_lookat=[5, 0, 1],
             save_name="05_s5_joint_friction")

# 场景 6: S6 密集人群
objects = [create_floor(), create_fetch_robot()]
pedestrian_positions = [
    [5, 2, 0], [6, -2, 0], [7, 1.5, 0], [8, -1.5, 0], [9, 0, 0]
]
for pos in pedestrian_positions:
    objects.append(create_pedestrian(pos))
render_scene("S6 密集人群", objects,
             camera_position=[0, -10, 5],
             camera_lookat=[7, 0, 1],
             save_name="06_s6_dynamic_crowd")

# 场景 7: S7 极窄通道
objects = [create_floor(), create_fetch_robot()]
objects.append(create_wall([7.5, 0.65, 1.5], [15, 0.1, 3]))  # 上墙
objects.append(create_wall([7.5, -0.65, 1.5], [15, 0.1, 3]))  # 下墙
objects.append(create_obstacle([10, 0.8, 0.4]))  # 盲区障碍物
render_scene("S7 极窄通道", objects,
             camera_position=[0, -5, 3],
             camera_lookat=[7, 0, 1],
             save_name="07_s7_narrow_corridor")

# 场景 8: S8 复合灾难
objects = [create_floor()]
robot = create_fetch_robot()
robot = robot + create_payload([0.3, 0, 0.85])
objects.append(robot)
objects.append(create_wall([7.5, 0.55, 1.5], [15, 0.1, 3]))
objects.append(create_wall([7.5, -0.55, 1.5], [15, 0.1, 3]))
objects.append(create_pedestrian([8, 0.3, 0]))
objects.append(create_pedestrian([9, -0.3, 0]))
render_scene("S8 复合灾难", objects,
             camera_position=[0, -8, 4],
             camera_lookat=[7, 0, 1],
             lighting_intensity=0.2,  # 低光照
             save_name="08_s8_compound_hell")

# ============ 完成 ============
print("\n" + "="*60)
print("渲染完成!")
print(f"输出目录：{output_dir}")
print("="*60)

# 列出所有生成的文件
import subprocess
result = subprocess.run(['ls', '-lh', output_dir], capture_output=True, text=True)
print(result.stdout)
