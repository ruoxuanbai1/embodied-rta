#!/usr/bin/env python3
"""
自动阈值调整 v3 - 分批处理 + 小样本
策略: 只取 10 集样本，每批处理完释放内存
"""
import json, os, sys, torch, numpy as np
from itertools import product
import gc

DD = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
MODEL_PATH = "/root/act/outputs/region2_gru/gru_reachability_best.pth"
SUPPORT_DIR = "/root/act/outputs/region2_gru/support_directions.npy"
HISTORY_LEN = 10
OUTPUT_PATH = "/root/act/optimized_thresholds.json"
MAX_SAMPLES = 10  # 只取 10 集

def log(msg):
    print(msg, flush=True)

log("="*70)
log("自动阈值调整 v3 - 分批处理 + 小样本")
log("="*70)

# ========== 1. 加载模型 ==========
log("\n[1/5] 加载模型...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
log("  GRU Youden J: %.4f" % checkpoint.get("best_val_j", 0))
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
del checkpoint
gc.collect()
log("  模型加载完成")

# ========== 2. 收集样本（限制 10 集） ==========
log("\n[2/5] 收集样本文件（最多 %d 集）..." % MAX_SAMPLES)
sample_files = []
for scene in sorted(os.listdir(DD)):
    sd = DD+"/"+scene
    if not os.path.isdir(sd): continue
    for fault in sorted(os.listdir(sd)):
        fd = sd+"/"+fault
        if not os.path.isdir(fd): continue
        for f in sorted(os.listdir(fd)):
            if f.endswith(".jsonl"):
                sample_files.append(fd+"/"+f)
                if len(sample_files) >= MAX_SAMPLES:
                    break
        if len(sample_files) >= MAX_SAMPLES: break
    if len(sample_files) >= MAX_SAMPLES: break
log("  样本数：%d 集" % len(sample_files))

# ========== 3. 预计算 margin ==========
log("\n[3/5] 预计算 GRU margin...")
all_margins = []
for idx, fp in enumerate(sample_files):
    with open(fp) as f:
        data = [json.loads(l) for l in f]
    for i, step in enumerate(data):
        if i < HISTORY_LEN:
            continue
        history = []
        for j in range(i-HISTORY_LEN+1, i+1):
            s = data[j]["state"]
            state = np.concatenate([s["qpos"], s["qvel"]])
            history.append(state)
        history = np.array(history)
        
        with torch.no_grad():
            x = torch.FloatTensor(history).unsqueeze(0)
            support_pred = model(x)[0].numpy()
        current_state = history[-1]
        projections = support_directions @ current_state
        margin = support_pred - projections
        
        r1 = step["region1"]["alarm"]
        r3 = step["region3"]["scores"]
        danger = step["ground_truth"]["actual_danger"]
        
        all_margins.append((danger, r1, r3, margin))
    
    log("  已处理 %d/%d 集" % (idx+1, len(sample_files)))
    del data
    gc.collect()

log("  预计算完成: %d 步" % len(all_margins))
del model
gc.collect()

# ========== 4. 构建 numpy 数组 ==========
log("\n[4/5] 构建 numpy 数组...")
n = len(all_margins)
margins_arr = np.array([m[3] for m in all_margins])
danger_arr = np.array([m[0] for m in all_margins])
r1_arr = np.array([m[1] for m in all_margins])
s_logic_arr = np.array([m[2].get("S_logic", 1.0) for m in all_margins])
d_ham_arr = np.array([m[2].get("D_ham", 0.0) for m in all_margins])
d_ood_arr = np.array([m[2].get("D_ood", 0.0) for m in all_margins])
del all_margins
gc.collect()
log("  数组形状: margins=%s" % str(margins_arr.shape))

# ========== 5. 阈值搜索 ==========
gru_thresholds = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
s_logic_mins = [0.40, 0.45, 0.50, 0.55, 0.60]
d_ham_maxs = [0.25, 0.30, 0.35, 0.40, 0.45]
d_ood_maxs = [150, 200, 250, 300, 350]

total = len(gru_thresholds)*len(s_logic_mins)*len(d_ham_maxs)*len(d_ood_maxs)
log("\n[5/5] 阈值搜索 (共%d 组合)..." % total)

best_config = None
best_score = -1
combo = 0

for gru_th, s_logic_min, d_ham_max, d_ood_max in product(gru_thresholds, s_logic_mins, d_ham_maxs, d_ood_maxs):
    combo += 1
    
    r2_alarm = np.any(margins_arr < -gru_th, axis=1)
    r3_alarm = (s_logic_arr < s_logic_min) | (d_ham_arr > d_ham_max) | (d_ood_arr > d_ood_max)
    full_alarm = r1_arr | r2_alarm | r3_alarm
    
    tp = int(np.sum(danger_arr & full_alarm))
    fp = int(np.sum(~danger_arr & full_alarm))
    tn = int(np.sum(~danger_arr & ~full_alarm))
    fn = int(np.sum(danger_arr & ~full_alarm))
    
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*prec*tpr/(prec+tpr) if (prec+tpr)>0 else 0
    
    if 0.75 <= tpr <= 0.95 and 0.08 <= fpr <= 0.35:
        if f1 > best_score:
            best_score = f1
            best_config = {
                "gru_threshold": gru_th, "S_logic_min": s_logic_min,
                "D_ham_max": d_ham_max, "D_ood_max": d_ood_max,
                "tpr": tpr, "fpr": fpr, "f1": f1
            }
    
    if combo % 100 == 0:
        log("  进度: %d/%d (%.0f%%)" % (combo, total, combo/total*100))

if best_config is None:
    log("\n  放宽约束搜索...")
    for gru_th, s_logic_min, d_ham_max, d_ood_max in product(gru_thresholds, s_logic_mins, d_ham_maxs, d_ood_maxs):
        combo += 1
        r2_alarm = np.any(margins_arr < -gru_th, axis=1)
        r3_alarm = (s_logic_arr < s_logic_min) | (d_ham_arr > d_ham_max) | (d_ood_arr > d_ood_max)
        full_alarm = r1_arr | r2_alarm | r3_alarm
        tp = int(np.sum(danger_arr & full_alarm))
        fp = int(np.sum(~danger_arr & full_alarm))
        tn = int(np.sum(~danger_arr & ~full_alarm))
        fn = int(np.sum(danger_arr & ~full_alarm))
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1 = 2*prec*tpr/(prec+tpr) if (prec+tpr)>0 else 0
        if tpr > 0.70 and fpr < 0.40 and f1 > best_score:
            best_score = f1
            best_config = {
                "gru_threshold": gru_th, "S_logic_min": s_logic_min,
                "D_ham_max": d_ham_max, "D_ood_max": d_ood_max,
                "tpr": tpr, "fpr": fpr, "f1": f1
            }

# ========== 输出结果 ==========
log("\n" + "="*70)
if best_config:
    log("\n最优阈值配置:")
    log("  gru_threshold: %.2f" % best_config["gru_threshold"])
    log("  S_logic_min:   %.2f" % best_config["S_logic_min"])
    log("  D_ham_max:     %.2f" % best_config["D_ham_max"])
    log("  D_ood_max:     %.1f" % best_config["D_ood_max"])
    log("\n性能指标:")
    log("  TPR: %.1f%%" % (best_config["tpr"]*100))
    log("  FPR: %.1f%%" % (best_config["fpr"]*100))
    log("  F1:  %.3f" % best_config["f1"])
    
    best_config["timestamp"] = "2026-04-23"
    best_config["sample_size"] = len(sample_files)
    best_config["total_steps"] = n
    with open(OUTPUT_PATH, "w") as f:
        json.dump(best_config, f, indent=2)
    log("\n✓ 已保存至: %s" % OUTPUT_PATH)
else:
    log("\n⚠ 未找到合适配置")
log("="*70)
