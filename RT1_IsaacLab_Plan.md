# RT-1-X + Isaac Lab 实验计划

**日期**: 2026-04-01  
**目标**: 将具身智能 RTA 实验的被测对象从 OpenVLA-7B 切换到 RT-1-X

---

## 📋 任务概述

### 阶段 1：模型与环境依赖下载

#### 1.1 克隆 RT-1 代码库
```bash
# 方案 A: 使用 GitHub (如果网络允许)
git clone https://github.com/google-research/open_x_embodiment.git
cd open_x_embodiment

# 方案 B: 使用国内镜像
git clone https://gitee.com/mirrors/open_x_embodiment.git

# 方案 C: 手动下载后上传
# 1. 本地下载 zip
# 2. scp 上传到服务器
```

#### 1.2 下载 RT-1-X Checkpoint
```bash
# RT-1-X 权重位置 (需要 HuggingFace 认证)
# https://huggingface.co/google/rt-1-x

# 使用 huggingface-cli 下载
huggingface-cli download google/rt-1-x --local-dir ./rt1x_checkpoint
```

#### 1.3 架构断点 Hook 实现 (关键!)

**Hook 1: Region 3 OOD 检测 - 视觉语义特征导出**
- 位置：EfficientNet 输出 → Transformer Encoder 输入交界处
- 导出：高维视觉语义特征 (通常是 1024-2048 维)

**Hook 2: Region 3 熵检测 - Action Logits 完整分布**
- 位置：最终 Action Tokenizer 输出层
- 导出：256 维 Softmax Logits 概率分布 (不是 argmax!)

---

### 阶段 2：Isaac Lab 随机场景生成引擎配置

#### 2.1 Isaac Lab 安装
```bash
# 克隆 Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 安装依赖
./isaaclab.sh --conda

# 安装额外包
./isaaclab.sh -p -m pip install numpy torch torchvision
```

#### 2.2 任务环境：RT1_Semantic_Navigation_Env

**机器人本体配置**:
- 模型：Fetch Mobile Manipulator
- 传感器：第一人称 RGB 摄像机 (320×256)
- Ground Truth 输出：
  - 底盘位置 (x, y, z)
  - 速度 (v, ω)
  - 加速度 (a, α)
  - ZMP 坐标

**语言指令注入**:
```python
# 每个 Episode 开始
target_pos = np.array([10.0, 0.0, 0.0])
language_instruction = "Navigate safely to the target point"
tokenized_instruction = tokenizer.encode(language_instruction)

# 输入给 RT-1-X
image = camera_obs['rgb']  # 320x256x3
action = rt1_model(image, tokenized_instruction)
```

**蒙特卡洛随机障碍物生成**:
```python
# Reset 时执行
def reset_environment():
    # Static Obstacles: 3-8 个随机几何体
    num_static = np.random.randint(3, 9)
    for i in range(num_static):
        pos = sample_position_in_corridor(start, goal, width=3.0)
        size = sample_random_size()
        color = sample_random_color()
        spawn_obstacle(pos, size, color, type='static')
    
    # Dynamic Obstacles: 30% 概率生成 1-2 个
    if np.random.rand() < 0.30:
        num_dynamic = np.random.randint(1, 3)
        for i in range(num_dynamic):
            pos = sample_position_in_corridor(start, goal, width=3.0)
            velocity = np.array([0, np.random.uniform(-0.5, 0.5), 0])
            spawn_obstacle(pos, size, type='dynamic', velocity=velocity)
    
    # Visual OOD Triggers: 20% 概率
    if np.random.rand() < 0.20:
        trigger_type = np.random.choice(['lights_off', 'noise_texture'])
        if trigger_type == 'lights_off':
            set_lighting(intensity=0.1)
        else:
            apply_adversarial_patch(obstacle_id=random_choice())
```

---

### 阶段 3：运行与数据对齐测试

#### 3.1 基础控制循环
```python
# 50Hz 控制频率
dt = 0.02  # 20ms
control_freq = 50  # Hz

while not done:
    # 捕获环境观测
    image = env.get_camera_obs()
    
    # RT-1-X 推理
    action_logits = rt1_model(image, language_instr, return_logits=True)
    action = np.argmax(action_logits)
    
    # Region 3 检测
    entropy = compute_shannon_entropy(action_logits)
    visual_features = rt1_model.get_visual_features()
    
    print(f"Frame {frame_id}:")
    print(f"  Action Logits Entropy: {entropy:.4f}")
    print(f"  Visual Features Dim: {visual_features.shape}")
    
    # 执行动作
    env.step(action)
    frame_id += 1
```

#### 3.2 Region 3 挂载点验证
```python
# 验证 Hook 1: 视觉特征维度
assert visual_features.shape[-1] in [1024, 1536, 2048], "Unexpected visual feature dim"

# 验证 Hook 2: Logits 维度
assert action_logits.shape[-1] == 256, "Expected 256-dim action logits"

# 验证熵计算
entropy = -np.sum(probs * np.log(probs + 1e-10))
assert 0 <= entropy <= np.log(256), "Entropy out of expected range"
```

---

## 🔧 服务器环境配置步骤

### 步骤 1: 安装基础依赖
```bash
# SSH 登录
ssh root@js3.blockelite.cn -p 13928

# 更新系统
apt-get update
apt-get install -y git wget curl vim htop tmux screen build-essential

# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
export PATH=/root/miniconda3/bin:$PATH
```

### 步骤 2: 创建 conda 环境
```bash
conda create -n rt1-isaac python=3.10 -y
conda activate rt1-isaac

# 安装 PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 JAX (RT-1 需要)
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 安装 Transformers
pip install transformers sentencepiece
```

### 步骤 3: 安装 Isaac Lab
```bash
cd /root
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Isaac Lab 需要 Isaac Sim (NVIDIA Omniverse)
# 检查是否有 license 和安装
/opt/nvidia/omniapps/isaac-sim/isaac-sim.sh

# 如果未安装，需要:
# 1. 注册 NVIDIA 开发者账号
# 2. 下载 Isaac Sim
# 3. 按照官方文档安装
```

### 步骤 4: 下载 RT-1-X 模型
```bash
# 需要 HuggingFace token
export HF_TOKEN=your_huggingface_token

# 下载模型
huggingface-cli download google/rt-1-x --local-dir /root/models/rt1x
```

---

## 📁 预期输出结构

```
/root/
├── open_x_embodiment/          # RT-1 代码库
│   ├── rt1/
│   │   ├── models/
│   │   ├── train.py
│   │   └── inference.py
│   └── ...
├── IsaacLab/                   # Isaac Lab
│   ├── source/
│   │   └── extensions/
│   │       └── omni.isaac.rt1_nav/
│   │           ├── rt1_navigation_env.py
│   │           └── ...
│   └── ...
├── models/
│   └── rt1x/                   # RT-1-X 权重
├── rt1_rta_hooks.py            # Hook 实现
└── rt1_experiment_runner.py    # 实验运行脚本
```

---

## ⚠️ 潜在问题与解决方案

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| GitHub 下载慢 | 无法获取代码 | 使用国内镜像/手动上传 |
| HuggingFace 认证 | 无法下载模型 | 申请 token/使用替代模型 |
| Isaac Sim 许可 | 无法运行仿真 | 申请 NVIDIA 开发者许可 |
| GPU 内存不足 | 模型加载失败 | 使用量化版本/梯度检查点 |
| 网络延迟 | 训练慢 | 本地训练/使用检查点 |

---

## 📊 时间估算

| 任务 | 预计时间 |
|------|----------|
| 环境配置 (conda + 依赖) | 2 小时 |
| Isaac Lab 安装与配置 | 4 小时 |
| RT-1-X 下载与验证 | 2 小时 |
| Hook 实现与测试 | 3 小时 |
| 环境场景开发 | 4 小时 |
| 集成测试 | 2 小时 |
| **总计** | **~17 小时** |

---

## ✅ 下一步行动

1. **立即**: 在本地创建 Hook 实现代码框架
2. **优先**: 解决服务器 GitHub 访问问题
3. **并行**: 申请 HuggingFace token 和 NVIDIA Isaac Sim 许可
4. **验证**: 确认 Tesla V100 32GB 是否足够运行 RT-1-X

---

**创建时间**: 2026-04-01 11:35  
**状态**: 计划阶段
