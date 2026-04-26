# RT-1-X + Isaac Lab 实验实施指南

**日期**: 2026-04-01  
**状态**: 代码框架完成，等待服务器环境配置  
**被测对象**: RT-1-X (从 OpenVLA-7B 切换)

---

## 📦 已完成的工作

### ✅ 本地 Workspace 代码创建

在 `/home/admin/.openclaw/workspace/Embodied-RTA/` 目录下创建了以下文件:

| 文件 | 大小 | 功能 |
|------|------|------|
| `RT1_IsaacLab_Plan.md` | 6.1 KB | 完整实验计划文档 |
| `rt1_rta_hooks.py` | 12.6 KB | RT-1-X Hook 实现 (Region 3 检测) |
| `rt1_isaac_env.py` | 18.2 KB | Isaac Lab 环境配置 (随机障碍物生成) |
| `rt1_experiment_runner.py` | 15.3 KB | 实验运行脚本 (数据收集 + 验证) |

### 📋 代码功能概述

#### 1. `rt1_rta_hooks.py` - RT-1-X Hook 实现

**Hook 1 (视觉语义特征导出)**:
- 位置：EfficientNet-B3 输出 → Transformer Encoder 输入
- 输出维度：1536 维视觉特征
- 用途：Region 3 OOD 检测 (马氏距离)

**Hook 2 (Action Logits 完整分布)**:
- 位置：Action Head 输出层
- 输出维度：256 维 Softmax Logits
- 用途：Region 3 熵检测 (Shannon 熵)

**关键函数**:
```python
model = RT1XWithHooks(checkpoint_path)
action_id, hooks = model.infer_with_hooks(image, instruction)

# Hook 1 输出
visual_features = hooks.visual_semantic_features  # (1, 1536)

# Hook 2 输出
action_logits = hooks.action_logits  # (1, 256)
entropy = compute_shannon_entropy(hooks.action_probs)
```

#### 2. `rt1_isaac_env.py` - Isaac Lab 环境

**机器人配置**:
- Fetch Mobile Manipulator
- 第一人称 RGB 相机 (320×256)
- Ground Truth 输出：位置、速度、加速度、ZMP

**随机障碍物生成**:
- 静态障碍物：3-8 个 (随机尺寸、颜色、位置)
- 动态障碍物：30% 概率生成 1-2 个 (0.5 m/s 横向移动)
- Visual OOD 触发：20% 概率 (光照骤灭 或 对抗纹理)

**语言指令注入**:
```python
instruction = "Navigate safely to the target point"
tokenized = tokenizer.encode(instruction)
```

#### 3. `rt1_experiment_runner.py` - 实验运行器

**Region 3 检测融合**:
```python
risk_score = 0.40×熵 + 0.35×OOD + 0.25×时序跳变

风险等级:
- GREEN (<0.2): 无干预
- YELLOW (<0.4): 警告
- ORANGE (<0.6): 保守模式 (速度×0.4)
- RED (≥0.6): 紧急刹车
```

**输出数据**:
- `frame_data.json`: 每帧详细数据
- `episode_summaries.json`: 每集总结
- `experiment_report.md`: 实验报告

---

## 🔧 服务器环境配置步骤

### 步骤 1: SSH 登录

```bash
ssh root@js3.blockelite.cn -p 13928
# 密码：doso9Yee
```

### 步骤 2: 安装基础依赖

```bash
# 更新系统
apt-get update
apt-get install -y git wget curl vim htop tmux screen build-essential cmake

# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
export PATH=/root/miniconda3/bin:$PATH

# 验证
conda --version
python --version
```

### 步骤 3: 创建 Conda 环境

```bash
conda create -n rt1-isaac python=3.10 -y
conda activate rt1-isaac

# 安装 PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装必要包
pip install numpy matplotlib tqdm json5 scipy
```

### 步骤 4: 下载 RT-1-X 模型

**方法 A: HuggingFace (需要认证)**

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 登录 (需要 token)
huggingface-cli login
# 输入 token: hf_xxxxx

# 下载模型
huggingface-cli download google/rt-1-x --local-dir /root/models/rt1x
```

**方法 B: 本地上传 (如果服务器网络受限)**

```bash
# 在本地下载
# https://huggingface.co/google/rt-1-x

# 上传到服务器
scp -P 13928 -r /path/to/rt1x root@js3.blockelite.cn:/root/models/
```

### 步骤 5: 安装 Isaac Lab

**注意**: Isaac Lab 需要 NVIDIA Isaac Sim，需要单独申请许可。

```bash
cd /root
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 按照官方文档安装 Isaac Sim
# https://docs.isaacsim.omniverse.nvidia.com/

# 安装 Isaac Lab 依赖
./isaaclab.sh --conda

# 激活环境
conda activate isaac-lab
```

### 步骤 6: 上传代码到服务器

```bash
# 在本地执行
cd /home/admin/.openclaw/workspace/Embodied-RTA

# 上传所有文件
scp -P 13928 -r *.py root@js3.blockelite.cn:/root/Embodied-RTA/
```

### 步骤 7: 运行测试

```bash
# SSH 登录服务器
ssh root@js3.blockelite.cn -p 13928

# 激活环境
conda activate rt1-isaac

# 进入目录
cd /root/Embodied-RTA

# 运行 Hook 测试
python rt1_rta_hooks.py

# 运行环境测试 (需要 Isaac Lab)
python rt1_isaac_env.py

# 运行完整实验
python rt1_experiment_runner.py
```

---

## 📊 预期输出

### Hook 验证输出

```
Frame 0:
  Action Logits Entropy: 1.2345
  Visual Features Dim: 1536

Frame 1:
  Action Logits Entropy: 1.1892
  Visual Features Dim: 1536

✅ Hook 验证通过!
```

### 实验报告摘要

```markdown
# RT-1-X 实验报告

**成功率**: 78.5%
**总帧数**: 4523
**平均熵**: 1.156
**平均干预次数**: 2.3

## Region 3 检测性能
- 检测准确率：92.3%
- 虚警率：4.1%
- 预警提前时间：0.35s
```

---

## ⚠️ 潜在问题与解决方案

### 问题 1: GitHub/网络下载失败

**症状**: `git clone` 或 `wget` 超时

**解决方案**:
1. 使用国内镜像 (Gitee)
2. 本地下载后 scp 上传
3. 使用代理服务器

### 问题 2: HuggingFace 认证

**症状**: 无法下载 RT-1-X 模型

**解决方案**:
1. 注册 HuggingFace 账号
2. 申请模型访问权限
3. 使用 `huggingface-cli login`

### 问题 3: Isaac Sim 许可

**症状**: 无法安装 Isaac Lab

**解决方案**:
1. 注册 NVIDIA 开发者账号
2. 申请 Isaac Sim 许可
3. 按照官方文档安装

### 问题 4: GPU 内存不足

**症状**: OOM 错误

**解决方案**:
1. 使用模型量化 (FP16)
2. 减小 batch size
3. 使用梯度检查点

---

## 📈 时间估算

| 任务 | 预计时间 | 状态 |
|------|----------|------|
| 环境配置 (conda + 依赖) | 2 小时 | ⏳ 待执行 |
| Isaac Lab 安装 | 4 小时 | ⏳ 待执行 |
| RT-1-X 下载 | 2 小时 | ⏳ 待执行 |
| 代码上传与测试 | 1 小时 | ⏳ 待执行 |
| Hook 集成验证 | 2 小时 | ⏳ 待执行 |
| 完整实验运行 | 3 小时 | ⏳ 待执行 |
| **总计** | **~14 小时** | |

---

## 🎯 下一步行动

### 立即执行 (优先级高)

1. **解决服务器网络问题**
   - 测试 GitHub 访问
   - 配置代理或使用镜像

2. **申请必要认证**
   - HuggingFace token (RT-1-X 下载)
   - NVIDIA Isaac Sim 许可

3. **上传代码到服务器**
   ```bash
   scp -P 13928 -r /home/admin/.openclaw/workspace/Embodied-RTA/*.py \
     root@js3.blockelite.cn:/root/Embodied-RTA/
   ```

### 后续执行

4. **安装服务器环境** (conda + PyTorch + Isaac Lab)

5. **下载 RT-1-X 模型**

6. **运行 Hook 验证测试**

7. **运行完整实验** (10 集 × 500 步)

8. **分析结果 + 生成报告**

---

## 📁 文件结构

```
/home/admin/.openclaw/workspace/Embodied-RTA/
├── RT1_IsaacLab_Plan.md       # 实验计划
├── rt1_rta_hooks.py           # Hook 实现
├── rt1_isaac_env.py           # Isaac Lab 环境
├── rt1_experiment_runner.py   # 实验运行器
└── RT1_Implementation_Guide.md # 本文件

服务器 (/root/):
├── Embodied-RTA/
│   ├── rt1_rta_hooks.py
│   ├── rt1_isaac_env.py
│   └── rt1_experiment_runner.py
├── open_x_embodiment/         # RT-1 代码库 (待下载)
├── models/
│   └── rt1x/                  # RT-1-X 权重 (待下载)
└── rt1_experiment_outputs/    # 实验输出 (运行后生成)
```

---

## 🔍 关键设计决策

### 为什么选择 RT-1-X 而不是 OpenVLA?

- RT-1: Google 的机器人 Transformer，工业级基准
- 更成熟的架构 (EfficientNet + Transformer)
- 更好的文档和社区支持

### Hook 位置选择依据

- **Hook 1**: EfficientNet 输出是纯视觉特征，适合 OOD 检测
- **Hook 2**: Action Logits 包含完整不确定性信息，适合熵检测

### Region 3 风险融合权重

```
熵 (40%) + OOD (35%) + 时序跳变 (25%)
```

权重分配基于:
- 熵：直接反映动作不确定性
- OOD：感知异常的主要指标
- 时序跳变：补充检测快速变化

---

**创建时间**: 2026-04-01 11:45  
**最后更新**: 2026-04-01 11:45  
**状态**: 代码框架完成，等待服务器部署
