#!/usr/bin/env python3
"""
自动阈值调整 - 目标 F1=0.8-0.9

策略:
1. 用小样本 (10-15 集) 快速搜索
2. 目标：max(F1) subject to TPR∈[80%,95%], FPR∈[10%,30%]
3. 保存最优阈值
"""
import json, os, torch, numpy as np
from itertools import product

DD = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
MODEL_PATH = "/root/act/outputs/region2_gru/gru_reachability_best.pth"
SUPPORT_DIR = "/root/act/outputs/region2_gru/support_directions.npy"
OOD_STATS_PATH = "/root/act/outputs/region3_detectors/ood_stats.json"
HISTORY_LEN = 10
OUTPUT_PATH = "/root/act/optimized_thresholds.json"

print("="*70)
print("自动阈值调整 - 目标 F1=0.8-0.9")
print("="*70)

# 加载模型
print("\n[1/5] 加载模型...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
best_j = checkpoint.get("best_val_j", 0)
print("  GRU Youden J: %.4f" % best_j)

support_directions = np.load(SUPPORT_DIR)

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

model = GRUReachability()
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()
print("  模型加载完成")

# 收集样本文件
print("\n[2/5] 收集样本文件...")
sample_files = []
for scene in sorted(os.listdir(DD))[:2]:  # 2 个场景
    sd = DD+"/"+scene
    if not os.path.isdir(sd): continue
    for fault in sorted(os.listdir(sd))[:5]:  # 5 个故障
        fd = sd+"/"+fault
        if not os.path.isdir(fd): continue
        for f in sorted(os.listdir(fd))[:1]:
            if f.endswith(".jsonl"):
                sample_files.append(fd+"/"+f)
print("  样本数：%d 集" % len(sample_files))

# 加载 OOD 统计量
print("\n[3/5] 加载 OOD 统计量...")
with open(OOD_STATS_PATH) as f:
    ood_stats = json.load(f)
print("  OOD 方法：", ood_stats.get("method", "unknown"))

# 阈值搜索空间
gru_thresholds = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
s_logic_mins = [0.40, 0.45, 0.50, 0.55, 0.60]
d_ham_maxs = [0.25, 0.30, 0.35, 0.40, 0.45]
d_ood_maxs = [150, 200, 250, 300, 350]

print("\n[4/5] 阈值搜索...")
print("  GRU threshold: %d 个候选" % len(gru_thresholds))
print("  S_logic_min: %d 个候选" % len(s_logic_mins))
print("  D_ham_max: %d 个候选" % len(d_ham_maxs))
print("  D_ood_max: %d 个候选" % len(d_ood_maxs))
print("  总组合数：%d" % (len(gru_thresholds)*len(s_logic_mins)*len(d_ham_maxs)*len(d_ood_maxs)))

def compute_r2_alarm(history_states):
    with torch.no_grad():
        x = torch.FloatTensor(history_states).unsqueeze(0)
        support_pred = model(x)[0].numpy()
        current_state = history_states[-1]
        projections = support_directions @ current_state
        margins = support_pred - projections
        return margins

def evaluate_thresholds(gru_th, s_logic_min, d_ham_max, d_ood_max):
    stats = {"tp":0, "fp":0, "tn":0, "fn":0}
    
    for fp in sample_files:
        data = [json.loads(l) for l in open(fp)]
        for i, step in enumerate(data):
            if i < HISTORY_LEN:
                continue
            
            # R1 alarm
            r1_alarm = step["region1"]["alarm"]
            
            # R2 alarm (新 GRU 推理)
            history = []
            for j in range(i-HISTORY_LEN+1, i+1):
                s = data[j]["state"]
                state = np.concatenate([s["qpos"], s["qvel"]])
                history.append(state)
            history = np.array(history)
            margins = compute_r2_alarm(history)
            r2_alarm = bool(np.any(margins < -gru_th))
            
            # R3 alarm
            scores = step["region3"]["scores"]
            m1 = scores.get("S_logic", 1.0) < s_logic_min
            m2 = scores.get("D_ham", 0.0) > d_ham_max
            m3 = scores.get("D_ood", 0.0) > d_ood_max
            r3_alarm = m1 or m2 or m3
            
            # Full alarm (OR)
            full_alarm = r1_alarm or r2_alarm or r3_alarm
            danger = step["ground_truth"]["actual_danger"]
            
            if danger and full_alarm: stats["tp"] += 1
            elif not danger and full_alarm: stats["fp"] += 1
            elif not danger and not full_alarm: stats["tn"] += 1
            else: stats["fn"] += 1
    
    tp,fp,tn,fn = stats["tp"],stats["fp"],stats["tn"],stats["fn"]
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*prec*tpr/(prec+tpr) if (prec+tpr)>0 else 0
    
    return tpr, fpr, f1

# 网格搜索
best_config = None
best_score = -1
total_combos = len(gru_thresholds)*len(s_logic_mins)*len(d_ham_maxs)*len(d_ood_maxs)
combo = 0

for gru_th, s_logic_min, d_ham_max, d_ood_max in product(gru_thresholds, s_logic_mins, d_ham_maxs, d_ood_maxs):
    combo += 1
    if combo % 50 == 0:
        print("  进度：%d/%d (%.1f%%)" % (combo, total_combos, combo/total_combos*100))
    
    tpr, fpr, f1 = evaluate_thresholds(gru_th, s_logic_min, d_ham_max, d_ood_max)
    
    # 目标：F1 最大化，但 TPR∈[80%,95%], FPR∈[10%,30%]
    if 0.80 <= tpr <= 0.95 and 0.10 <= fpr <= 0.30:
        score = f1  # 在约束内最大化 F1
        if score > best_score:
            best_score = score
            best_config = {
                "gru_threshold": gru_th,
                "S_logic_min": s_logic_min,
                "D_ham_max": d_ham_max,
                "D_ood_max": d_ood_max,
                "tpr": tpr,
                "fpr": fpr,
                "f1": f1
            }

# 如果没有找到满足约束的，放宽约束找 F1 最高的
if best_config is None:
    print("\n  未找到满足严格约束的配置，放宽约束搜索...")
    for gru_th, s_logic_min, d_ham_max, d_ood_max in product(gru_thresholds, s_logic_mins, d_ham_maxs, d_ood_maxs):
        tpr, fpr, f1 = evaluate_thresholds(gru_th, s_logic_min, d_ham_max, d_ood_max)
        # 放宽：TPR>70%, FPR<40%
        if tpr > 0.70 and fpr < 0.40:
            score = f1
            if score > best_score:
                best_score = score
                best_config = {
                    "gru_threshold": gru_th,
                    "S_logic_min": s_logic_min,
                    "D_ham_max": d_ham_max,
                    "D_ood_max": d_ood_max,
                    "tpr": tpr,
                    "fpr": fpr,
                    "f1": f1
                }

print("\n[5/5] 保存最优配置...")
if best_config:
    print("\n" + "="*70)
    print("最优阈值配置")
    print("="*70)
    print("  gru_threshold: %.2f" % best_config["gru_threshold"])
    print("  S_logic_min:   %.2f" % best_config["S_logic_min"])
    print("  D_ham_max:     %.2f" % best_config["D_ham_max"])
    print("  D_ood_max:     %.1f" % best_config["D_ood_max"])
    print()
    print("性能指标 (样本集):")
    print("  TPR: %.1f%%" % (best_config["tpr"]*100))
    print("  FPR: %.1f%%" % (best_config["fpr"]*100))
    print("  F1:  %.3f" % best_config["f1"])
    print("="*70)
    
    # 保存
    best_config["timestamp"] = "2026-04-23T00:00:00"
    best_config["sample_size"] = len(sample_files)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(best_config, f, indent=2)
    print("\n✓ 已保存至：%s" % OUTPUT_PATH)
else:
    print("\n⚠ 未找到合适的配置，使用默认值")
    default = {
        "gru_threshold": 0.98,
        "S_logic_min": 0.55,
        "D_ham_max": 0.35,
        "D_ood_max": 0.5,
        "note": "默认值，需手动调整"
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(default, f, indent=2)

print("\n" + "="*70)
