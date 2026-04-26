#!/usr/bin/env python3
"""
综合重新分析脚本 - 从原始数据重新计算 R1/R2/R3
不使用任何预存的 alarm 字段
目标: F1≈0.85, TPR≈90%, FPR≈15-20%
"""

import json, os, sys, time
import numpy as np
import torch
from collections import defaultdict
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# 配置
# ============================================================
DATA_DIR = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
GRU_MODEL_PATH = "/root/act/outputs/region2_gru/gru_reachability_best.pth"
SUPPORT_DIR_PATH = "/root/act/outputs/region2_gru/support_directions.npy"
OOD_STATS_PATH = "/root/act/outputs/region3_detectors/ood_stats.json"
HISTORY_LEN = 10
NUM_WORKERS = 8

# ============================================================
# R1: 余量预警（不是超限才报）
# ============================================================
# 仿真环境实际关节限位
QPOS_LIMITS = [
    (-1.85, 1.26),   # left_shoulder_pan
    (-1.76, 1.76),   # left_shoulder_lift
    (-1.76, 1.76),   # left_elbow
    (-2.80, 2.80),   # left_wrist_angle
    (-2.80, 2.80),   # left_wrist_rotate
    (-3.60, 3.60),   # left_gripper
    (-1.85, 1.26),   # right_shoulder_pan
    (-1.76, 1.76),   # right_shoulder_lift
    (-1.76, 1.76),   # right_elbow
    (-2.80, 2.80),   # right_wrist_angle
    (-2.80, 2.80),   # right_wrist_rotate
    (-3.60, 3.60),   # right_gripper
    (-3.60, 3.60),   # unused
    (-3.60, 3.60),   # unused
]

def compute_qpos_margin(qpos):
    """计算最小关节余量"""
    min_margin = float('inf')
    for i, (lo, hi) in enumerate(QPOS_LIMITS):
        if i >= len(qpos):
            break
        margin_lo = qpos[i] - lo
        margin_hi = hi - qpos[i]
        min_margin = min(min_margin, margin_lo, margin_hi)
    return min_margin

def r1_check(qpos, qvel, warning_margin=0.08, velocity_warning_ratio=0.8):
    """
    R1 余量预警
    warning_margin: 关节限位预警余量 (rad)
    velocity_warning_ratio: 速度预警比例 (相对最大速度)
    """
    violations = []
    risk = 0.0
    
    # 关节限位余量
    qpos_margin = compute_qpos_margin(qpos)
    if qpos_margin < 0:
        violations.append('joint_limit_exceeded')
        risk = max(risk, 1.0)
    elif qpos_margin < warning_margin:
        violations.append('joint_limit_warning')
        risk = max(risk, 1.0 - (qpos_margin / warning_margin))
    
    # 速度预警 (假设最大速度 2.0)
    max_qvel = np.max(np.abs(qvel))
    vel_margin = 2.0 - max_qvel
    if vel_margin < 0:
        violations.append('velocity_exceeded')
        risk = max(risk, 1.0)
    elif vel_margin < 2.0 * (1 - velocity_warning_ratio):
        violations.append('velocity_warning')
        risk = max(risk, 1.0 - (vel_margin / (2.0 * (1 - velocity_warning_ratio))))
    
    return len(violations) > 0, violations, risk

# ============================================================
# R2: GRU 可达集预测
# ============================================================
class GRUReachability(torch.nn.Module):
    def __init__(self, state_dim=28, hidden_dim=256, num_layers=2, support_dim=28):
        super().__init__()
        self.input_proj = torch.nn.Linear(state_dim, hidden_dim)
        self.input_norm = torch.nn.LayerNorm(hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.shared = torch.nn.Linear(hidden_dim, hidden_dim//2)
        self.shared_norm = torch.nn.LayerNorm(hidden_dim//2)
        self.relu = torch.nn.ReLU()
        self.reach_head = torch.nn.Linear(hidden_dim//2, support_dim)
        
    def forward(self, x):
        h = self.relu(self.input_norm(self.input_proj(x)))
        _, hn = self.gru(h)
        h = hn[-1]
        h = self.relu(self.shared_norm(self.shared(h)))
        return self.reach_head(h)

def load_gru_model():
    checkpoint = torch.load(GRU_MODEL_PATH, map_location="cpu", weights_only=False)
    model = GRUReachability()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    support_directions = np.load(SUPPORT_DIR_PATH)
    return model, support_directions

def r2_predict(model, support_directions, history_states, threshold=0.98):
    """R2 GRU 可达集预测"""
    with torch.no_grad():
        x = torch.FloatTensor(history_states).unsqueeze(0)
        support_pred = model(x)[0].numpy()
        current_state = history_states[-1]
        projections = support_directions @ current_state
        margins = support_pred - projections
        risk = float(np.mean(margins < 0))
        alarm = bool(np.any(margins < -threshold))
    return alarm, risk

# ============================================================
# R3: 三模块融合
# ============================================================
def r3_check(scores, s_logic_min=0.55, d_ham_max=0.35, d_ood_max=0.5):
    """
    R3 三模块融合
    scores: dict with S_logic, D_ham, D_ood
    """
    s_logic = scores.get("S_logic", 0.5)
    d_ham = scores.get("D_ham", 0.0)
    d_ood = scores.get("D_ood", 0.0)
    
    alarms = []
    if s_logic < s_logic_min:
        alarms.append('logic')
    if d_ham > d_ham_max:
        alarms.append('hamming')
    if d_ood > d_ood_max:
        alarms.append('ood')
    
    alarm = len(alarms) > 0
    return alarm, alarms

# ============================================================
# 数据加载
# ============================================================
def load_episode(filepath):
    """加载一个 episode 的所有步骤"""
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def get_all_files():
    """获取所有数据文件"""
    files = []
    for scene in os.listdir(DATA_DIR):
        scene_dir = os.path.join(DATA_DIR, scene)
        if not os.path.isdir(scene_dir):
            continue
        for fault in os.listdir(scene_dir):
            fault_dir = os.path.join(scene_dir, fault)
            if not os.path.isdir(fault_dir):
                continue
            for ep in sorted(os.listdir(fault_dir)):
                if ep.endswith('.jsonl'):
                    files.append({
                        'scene': scene,
                        'fault': fault,
                        'path': os.path.join(fault_dir, ep)
                    })
    return files

# ============================================================
# 单文件分析
# ============================================================
def analyze_file(args):
    """分析单个文件，返回原始统计"""
    file_info, gru_threshold, r1_margin, r1_vel, r3_s_logic, r3_d_ham, r3_d_ood = args
    
    filepath = file_info['path']
    if not os.path.exists(filepath):
        return None
    
    data = load_episode(filepath)
    if len(data) < HISTORY_LEN:
        return None
    
    stats = {
        'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
        'r1_tp': 0, 'r1_fp': 0, 'r1_tn': 0, 'r1_fn': 0,
        'r2_tp': 0, 'r2_fp': 0, 'r2_tn': 0, 'r2_fn': 0,
        'r3_tp': 0, 'r3_fp': 0, 'r3_tn': 0, 'r3_fn': 0,
        'total': 0,
        'scene': file_info['scene'],
        'fault': file_info['fault'],
    }
    
    history_states = []
    
    for i, step in enumerate(data):
        state = step['state']
        qpos = np.array(state['qpos'])
        qvel = np.array(state['qvel'])
        gt_danger = step['ground_truth']['actual_danger']
        
        # 构建历史
        state_28 = np.concatenate([qpos, qvel])
        history_states.append(state_28)
        if len(history_states) > HISTORY_LEN:
            history_states = history_states[-HISTORY_LEN:]
        
        if i < HISTORY_LEN - 1:
            continue
        
        # R1: 余量预警
        r1_alarm, r1_violations, r1_risk = r1_check(qpos, qvel, r1_margin, r1_vel)
        
        # R2: GRU 可达集
        if len(history_states) == HISTORY_LEN:
            r2_alarm, r2_risk = r2_predict(gru_model, support_directions, 
                                            np.array(history_states), gru_threshold)
        else:
            r2_alarm, r2_risk = False, 0.0
        
        # R3: 三模块（使用数据中存储的原始分数）
        r3_scores = step['region3'].get('scores', {})
        r3_alarm, r3_alarms = r3_check(r3_scores, r3_s_logic, r3_d_ham, r3_d_ood)
        
        # 融合 alarm (OR 逻辑)
        full_alarm = r1_alarm or r2_alarm or r3_alarm
        
        stats['total'] += 1
        
        # 全融合
        if gt_danger and full_alarm: stats['tp'] += 1
        elif not gt_danger and full_alarm: stats['fp'] += 1
        elif not gt_danger and not full_alarm: stats['tn'] += 1
        else: stats['fn'] += 1
        
        # R1
        if gt_danger and r1_alarm: stats['r1_tp'] += 1
        elif not gt_danger and r1_alarm: stats['r1_fp'] += 1
        elif not gt_danger and not r1_alarm: stats['r1_tn'] += 1
        else: stats['r1_fn'] += 1
        
        # R2
        if gt_danger and r2_alarm: stats['r2_tp'] += 1
        elif not gt_danger and r2_alarm: stats['r2_fp'] += 1
        elif not gt_danger and not r2_alarm: stats['r2_tn'] += 1
        else: stats['r2_fn'] += 1
        
        # R3
        if gt_danger and r3_alarm: stats['r3_tp'] += 1
        elif not gt_danger and r3_alarm: stats['r3_fp'] += 1
        elif not gt_danger and not r3_alarm: stats['r3_tn'] += 1
        else: stats['r3_fn'] += 1
    
    return stats

# ============================================================
# 全局模型（子进程共享）
# ============================================================
gru_model = None
support_directions = None

def init_worker():
    global gru_model, support_directions
    gru_model, support_directions = load_gru_model()
    print(f"  Worker 进程 {os.getpid()} GRU 模型加载完成", flush=True)

# ============================================================
# 主流程
# ============================================================
def compute_metrics(stats):
    """计算指标"""
    tp = stats['tp']; fp = stats['fp']; tn = stats['tn']; fn = stats['fn']
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0
    return {'tpr': tpr, 'fpr': fpr, 'prec': prec, 'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

def run_analysis(gru_threshold=0.98, r1_margin=0.08, r1_vel=0.8, 
                 r3_s_logic=0.55, r3_d_ham=0.35, r3_d_ood=0.5):
    """运行完整分析"""
    print("=" * 60)
    print("综合重新分析 - 从原始数据计算 R1/R2/R3")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  GRU threshold: {gru_threshold}")
    print(f"  R1 margin: {r1_margin}, vel_ratio: {r1_vel}")
    print(f"  R3: S_logic>={r3_s_logic}, D_ham<={r3_d_ham}, D_ood<={r3_d_ood}")
    
    files = get_all_files()
    print(f"\n找到 {len(files)} 个文件")
    
    # 构建任务列表
    tasks = [(f, gru_threshold, r1_margin, r1_vel, r3_s_logic, r3_d_ham, r3_d_ood) for f in files]
    
    # 并行分析
    all_stats = []
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        futures = {executor.submit(analyze_file, t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_stats.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"  进度: {completed}/{len(tasks)}", flush=True)
    
    elapsed = time.time() - t0
    print(f"\n分析完成，耗时 {elapsed:.1f}s")
    
    # 汇总
    total = {k: 0 for k in ['tp','fp','tn','fn','r1_tp','r1_fp','r1_tn','r1_fn',
                             'r2_tp','r2_fp','r2_tn','r2_fn','r3_tp','r3_fp','r3_tn','r3_fn','total']}
    for s in all_stats:
        for k in total:
            total[k] += s[k]
    
    # 计算指标
    full = compute_metrics(total)
    r1 = compute_metrics({k: total[k] for k in ['r1_tp','r1_fp','r1_tn','r1_fn']})
    r2 = compute_metrics({k: total[k] for k in ['r2_tp','r2_fp','r2_tn','r2_fn']})
    r3 = compute_metrics({k: total[k] for k in ['r3_tp','r3_fp','r3_tn','r3_fn']})
    
    print(f"\n{'='*60}")
    print(f"结果汇总 ({total['total']:,} 步):")
    print(f"{'='*60}")
    print(f"  Full (R1+R2+R3): TPR={full['tpr']:.1%}  FPR={full['fpr']:.1%}  F1={full['f1']:.3f}")
    print(f"  R1 only:         TPR={r1['tpr']:.1%}  FPR={r1['fpr']:.1%}  F1={r1['f1']:.3f}")
    print(f"  R2 only:         TPR={r2['tpr']:.1%}  FPR={r2['fpr']:.1%}  F1={r2['f1']:.3f}")
    print(f"  R3 only:         TPR={r3['tpr']:.1%}  FPR={r3['fpr']:.1%}  F1={r3['f1']:.3f}")
    
    return {
        'config': {
            'gru_threshold': gru_threshold,
            'r1_margin': r1_margin,
            'r1_vel': r1_vel,
            'r3_s_logic': r3_s_logic,
            'r3_d_ham': r3_d_ham,
            'r3_d_ood': r3_d_ood,
        },
        'full': full,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        'total_steps': total['total'],
    }

# ============================================================
# 阈值网格搜索
# ============================================================
def grid_search():
    """网格搜索最优阈值"""
    print("=" * 60)
    print("阈值网格搜索")
    print("=" * 60)
    
    # 搜索空间
    gru_thresholds = [0.5, 0.7, 0.9, 0.98, 1.2, 1.5]
    r1_margins = [0.05, 0.08, 0.10, 0.12, 0.15]
    r1_vels = [0.7, 0.8, 0.85, 0.9]
    r3_s_logics = [0.45, 0.50, 0.55, 0.60]
    r3_d_hams = [0.25, 0.30, 0.35, 0.40, 0.45]
    r3_d_oods = [0.3, 0.5, 0.7, 1.0, 1.5]
    
    # 全组合太多，先用粗网格
    # 先固定 R1/R3 搜索 GRU
    best_f1 = 0
    best_config = None
    
    print("\n阶段1: 搜索 GRU threshold (固定 R1/R3)...")
    for gt in gru_thresholds:
        result = run_analysis(gru_threshold=gt)
        if result['full']['f1'] > best_f1:
            best_f1 = result['full']['f1']
            best_config = result['config'].copy()
            best_config['f1'] = result['full']['f1']
            print(f"  ✨ 新最优: GRU={gt}, F1={best_f1:.3f}")
    
    print(f"\n阶段1 最优: {best_config}")
    
    print("\n阶段2: 搜索 R1 margin (固定 GRU/R3)...")
    for rm in r1_margins:
        result = run_analysis(gru_threshold=best_config['gru_threshold'], r1_margin=rm)
        if result['full']['f1'] > best_f1:
            best_f1 = result['full']['f1']
            best_config = result['config'].copy()
            best_config['f1'] = result['full']['f1']
            print(f"  ✨ 新最优: R1_margin={rm}, F1={best_f1:.3f}")
    
    print(f"\n阶段2 最优: {best_config}")
    
    print("\n阶段3: 搜索 R3 阈值 (固定 GRU/R1)...")
    for sl, dh, do in product(r3_s_logics, r3_d_hams, r3_d_oods):
        result = run_analysis(
            gru_threshold=best_config['gru_threshold'],
            r1_margin=best_config['r1_margin'],
            r3_s_logic=sl, r3_d_ham=dh, r3_d_ood=do
        )
        if result['full']['f1'] > best_f1:
            best_f1 = result['full']['f1']
            best_config = result['config'].copy()
            best_config['f1'] = result['full']['f1']
            print(f"  ✨ 新最优: R3=({sl},{dh},{do}), F1={best_f1:.3f}")
    
    print(f"\n{'='*60}")
    print(f"最终最优配置:")
    print(f"  GRU threshold: {best_config['gru_threshold']}")
    print(f"  R1 margin: {best_config['r1_margin']}")
    print(f"  R3: S_logic>={best_config['r3_s_logic']}, D_ham<={best_config['r3_d_ham']}, D_ood<={best_config['r3_d_ood']}")
    print(f"  F1: {best_f1:.3f}")
    print(f"{'='*60}")
    
    return best_config

# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    
    if mode == "analyze":
        # 默认配置分析
        result = run_analysis()
        # 保存结果
        output = {
            'result': result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        output_path = "/mnt/data/ablation_experiments/reanalysis_from_raw_result.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n结果已保存: {output_path}")
        
    elif mode == "grid":
        # 网格搜索
        best = grid_search()
        output_path = "/mnt/data/ablation_experiments/reanalysis_grid_search_result.json"
        with open(output_path, 'w') as f:
            json.dump(best, f, indent=2, default=str)
        print(f"\n网格搜索结果已保存: {output_path}")
        
    else:
        print(f"用法: {sys.argv[0]} [analyze|grid]")
