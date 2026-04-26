# ALOHA 三层 RTA 真实闭环试验计划

**创建时间**: 2026-04-04  
**目标**: 在 Isaac Lab 仿真环境中运行真实闭环试验

---

## 📋 当前状态

### ✅ 已就绪组件

| 组件 | 状态 | 位置 |
|------|------|------|
| Isaac Lab | ✅ 已安装 | `/home/vipuser/IsaacLab/` |
| ACT 预训练模型 | ✅ 可用 | `lerobot/act_aloha_sim_transfer_cube_human` |
| Region 2 GRU | ✅ 训练完成 | `/mnt/data/region2_training_v2/` |
| Region 3 阈值 | ✅ 学习完成 | `/mnt/data/region3_training_v2/` |
| Python 3.10 | ✅ 已安装 | `/usr/bin/python3` |

### ⚠️ 需要配置

| 项目 | 当前状态 | 需要操作 |
|------|----------|----------|
| Isaac Lab Python 环境 | ❌ 未配置 | 创建 conda 环境 |
| ALOHA 仿真场景 | ❌ 未创建 | 编写场景配置 |
| RTA 集成脚本 | ❌ 未编写 | 集成三层 RTA |
| 试验管理脚本 | ❌ 未编写 | 批量试验管理 |

---

## 🚀 执行步骤

### Step 1: 配置 Isaac Lab 环境 (预计 2-3 小时)

```bash
cd /home/vipuser/IsaacLab

# 创建 conda 环境
conda env create -f environment.yml --name isaaclab

# 激活环境
conda activate isaaclab

# 安装额外依赖
pip install lerobot torch torchvision

# 测试 Isaac Lab 导入
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

---

### Step 2: 创建 ALOHA 仿真场景 (预计 3-4 小时)

创建文件：`envs/aloha_pick_place_env.py`

**场景要素**:
- ALOHA 双臂机械臂模型
- 桌面环境
- 目标物体 (cube)
- 障碍物 (可选)
- 相机传感器 (RGB 图像)
- 状态空间 (14 维)
- 动作空间 (14 维)

**场景配置**:
- B1: 空旷桌面
- B2: 5 个静态障碍
- B3: 10 个密集障碍
- F1-F8: 故障注入配置

---

### Step 3: 集成三层 RTA (预计 2-3 小时)

创建文件：`rta/rtf_fusion_integrated.py`

```python
class ThreeLayerRTA:
    def __init__(self):
        # 加载 Region 2 GRU
        self.r2_model = load_gru_model('/mnt/data/region2_training_v2/region2_gru_best.pth')
        
        # 加载 Region 3 阈值
        with open('/mnt/data/region3_training_v2/adaptive_thresholds_v2.json') as f:
            self.r3_config = json.load(f)
        
        # Region 1 物理参数
        self.r1_thresholds = {
            'collision_distance': 0.15,
            'zmp_margin': 0.27,
            'max_velocity': 0.9,
        }
    
    def check_r1(self, state):
        """物理硬约束检查"""
        risk = 0.0
        if state.collision_distance < self.r1_thresholds['collision_distance']:
            risk = 1.0
        return risk
    
    def check_r2(self, state_history, action_history):
        """可达性预测"""
        # GRU 推理
        with torch.no_grad():
            min_pred, max_pred = self.r2_model(state_history, action_history)
        # 检查边界
        risk = compute_risk(min_pred, max_pred)
        return risk
    
    def check_r3(self, state, gradients):
        """感知异常检测"""
        # 提取特征
        features = extract_features(state, gradients)
        # RF 分类
        anomaly_prob = self.rf_model.predict_proba([features])[0, 1]
        return 1.0 if anomaly_prob > self.r3_threshold else 0.0
    
    def fuse_risks(self, r1, r2, r3):
        """风险融合"""
        R = 0.3*r1 + 0.4*r2 + 0.3*r3
        return R
    
    def intervene(self, action, risk_level):
        """RTA 干预"""
        if risk_level > 0.6:
            return action * 0.0  # 紧急停止
        elif risk_level > 0.4:
            return action * 0.5  # 减速
        elif risk_level > 0.2:
            return action * 0.8  # 轻微减速
        return action  # 正常执行
```

---

### Step 4: 编写试验脚本 (预计 2-3 小时)

创建文件：`experiments/run_real_ablation.py`

```python
#!/usr/bin/env python3
"""真实闭环消融试验"""

from envs.aloha_pick_place_env import ALOHAPickPlaceEnv
from rta.rtf_fusion_integrated import ThreeLayerRTA
from lerobot.policies.act import ACTPolicy

# 配置
SCENES = ["B1_empty", "B2_static", "B3_dense", 
          "F1_lighting", "F2_occlusion", "F3_adversarial",
          "F4_payload", "F5_friction", "F6_dynamic", 
          "F7_sensor", "F8_compound"]

RTA_CONFIGS = {
    "A0_Pure_VLA": {"r1": False, "r2": False, "r3": False},
    "A1_R1_Only": {"r1": True, "r2": False, "r3": False},
    ...
    "A7_Full": {"r1": True, "r2": True, "r3": True},
}

N_TRIALS = 30

# 主循环
for scene in SCENES:
    for config_name, config in RTA_CONFIGS.items():
        for trial in range(N_TRIALS):
            # 初始化环境
            env = ALOHAPickPlaceEnv(scene=scene)
            
            # 初始化 RTA
            rta = ThreeLayerRTA()
            
            # 加载 ACT 模型
            act = ACTPolicy.from_pretrained(...)
            
            # 运行试验
            result = run_episode(env, act, rta, config)
            
            # 记录结果
            log_result(result)
```

---

### Step 5: 运行试验 (预计 2-3 天)

**试验规模**: 11 场景 × 8 配置 × 30 次 = 2,640 次

**预计时间**:
- 单次试验：~30 秒 (250 步 × 0.02s/步)
- 总时间：2,640 × 30s = 79,200s ≈ 22 小时
- 并行加速 (4 进程): ~6 小时

**命令**:
```bash
cd /home/vipuser/Embodied-RTA

# 单机 4 进程并行
python3 experiments/run_real_ablation.py --parallel 4 --output outputs/real_ablation/

# 监控进度
tail -f outputs/real_ablation/progress.log
```

---

### Step 6: 数据分析 (预计 2-3 小时)

```bash
python3 experiments/analyze_real_results.py \
    --input outputs/real_ablation/results.csv \
    --output outputs/real_ablation/figures/
```

---

## 📁 交付物

### 代码
- `envs/aloha_pick_place_env.py` - ALOHA 仿真环境
- `rta/rtf_fusion_integrated.py` - 三层 RTA 集成
- `experiments/run_real_ablation.py` - 试验脚本
- `experiments/analyze_real_results.py` - 分析脚本

### 数据
- `outputs/real_ablation/results.csv` - 原始试验数据
- `outputs/real_ablation/trajectories/` - 轨迹数据

### 图表
- `outputs/real_ablation/figures/fig_*.png` - 论文图表

### 文档
- `outputs/real_ablation/EXPERIMENT_REPORT.md` - 实验报告

---

## ⏱️ 时间估算

| 步骤 | 预计时间 |
|------|----------|
| Step 1: 环境配置 | 2-3 小时 |
| Step 2: 场景创建 | 3-4 小时 |
| Step 3: RTA 集成 | 2-3 小时 |
| Step 4: 试验脚本 | 2-3 小时 |
| Step 5: 运行试验 | 6-22 小时 (取决于并行) |
| Step 6: 数据分析 | 2-3 小时 |
| **总计** | **17-38 小时** |

---

## ⚠️ 风险与应对

| 风险 | 可能性 | 应对措施 |
|------|--------|----------|
| Isaac Lab 环境配置失败 | 中 | 使用 Docker 容器 |
| ACT 模型推理速度慢 | 中 | 使用 GPU 加速 |
| 仿真环境不稳定 | 低 | 增加错误重试 |
| 试验时间过长 | 中 | 增加并行度 |

---

## 🚀 立即可执行

**下一步**: 开始 Step 1 - 配置 Isaac Lab 环境

```bash
cd /home/vipuser/IsaacLab
conda env create -f environment.yml --name isaaclab
```

需要我立即开始配置环境吗？
