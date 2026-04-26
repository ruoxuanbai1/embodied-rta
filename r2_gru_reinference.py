#!/usr/bin/env python3
"""用新 GRU 模型对离线数据重新推理 Region 2 alarm"""
import json, os, torch, numpy as np

DD = "/mnt/data/ablation_experiments/ablation_combined_v2_165eps_FULL"
MODEL_PATH = "/root/act/outputs/region2_gru/gru_reachability_best.pth"
SUPPORT_DIR = "/root/act/outputs/region2_gru/support_directions.npy"
HISTORY_LEN = 10

print("="*60)
print("Region 2 新 GRU 模型重推理")
print("="*60)

# 加载模型
print("\n加载模型...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
best_j = checkpoint.get('best_val_j', 0)
threshold = checkpoint.get('best_threshold', 0.98)
print(f"  Youden J: {best_j:.4f}")
print(f"  阈值：{threshold}")

# 加载支撑方向
support_directions = np.load(SUPPORT_DIR)
print(f"  支撑方向形状：{support_directions.shape}")

# 定义 GRU 模型
class GRUReachability(torch.nn.Module):
    def __init__(self, state_dim=28, hidden_dim=192, num_layers=2, support_dim=28):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim//2, hidden_dim//4),
            torch.nn.LayerNorm(hidden_dim//4),
            torch.nn.ReLU()
        )
        self.reach_head = torch.nn.Linear(hidden_dim//4, support_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        _, hn = self.gru(h)
        h = hn[-1]
        h = self.fc(h)
        return self.reach_head(h)

model = GRUReachability()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  模型加载完成")

# 推理函数
def compute_r2_alarm(history_states, support_directions, model, threshold):
    """
    history_states: (H, 28) - H 步历史状态
    返回：alarm (bool), risk (float)
    """
    with torch.no_grad():
        # 输入形状：(1, H, 28)
        x = torch.FloatTensor(history_states).unsqueeze(0)
        # 预测支撑值：(1, 28)
        support_pred = model(x)[0]
        
        # 计算当前状态的支撑值投影
        current_state = history_states[-1]  # (28,)
        projections = support_directions @ current_state  # (28,)
        
        # 支撑值 - 投影 = 余量 (margin)
        # margin < 0 表示超出可达集边界 -> 危险
        margins = support_pred.numpy() - projections
        risk = float(np.mean(margins < 0))
        alarm = bool(torch.any(margins < -threshold))
        
    return alarm, risk, margins

# 抽样评估
print("\n抽样评估...")
sample_files = [
    DD+"/B1_empty/normal/ep000_full_data.jsonl",
    DD+"/B1_empty/F1_visual_noise/ep001_full_data.jsonl",
    DD+"/B2_static/F13_full/ep000_full_data.jsonl",
]

for fp in sample_files:
    if not os.path.exists(fp):
        print(f"  SKIP: {fp}")
        continue
    
    print(f"\n  文件：{fp.split('/')[-3]}/{fp.split('/')[-2]}")
    data = [json.loads(l) for l in open(fp)]
    
    stats = {"tp":0, "fp":0, "tn":0, "fn":0}
    alarms = []
    
    for i, step in enumerate(data):
        # 构建历史 (需要至少 HISTORY_LEN 步)
        if i < HISTORY_LEN:
            continue
        
        # 从历史步骤构建状态序列
        history = []
        for j in range(i-HISTORY_LEN+1, i+1):
            s = data[j]["state"]
            state = np.concatenate([s["qpos"], s["qvel"]])  # 28 维
            history.append(state)
        history = np.array(history)  # (10, 28)
        
        # GRU 推理
        alarm, risk, margins = compute_r2_alarm(history, support_directions, model, threshold)
        alarms.append(alarm)
        
        # 对比 ground truth
        danger = step["ground_truth"]["actual_danger"]
        if danger and alarm: stats["tp"] += 1
        elif not danger and alarm: stats["fp"] += 1
        elif not danger and not alarm: stats["tn"] += 1
        else: stats["fn"] += 1
    
    tp,fp,tn,fn = stats["tp"],stats["fp"],stats["tn"],stats["fn"]
    tpr = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    fpr = fp/(fp+tn)*100 if (fp+tn)>0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
    f1 = 2*prec*tpr/(prec+tpr) if (prec+tpr)>0 else 0
    
    print(f"    步数：{tp+fp+tn+fn}")
    print(f"    TPR={tpr:.1f}% FPR={fpr:.1f}% Prec={prec:.1f}% F1={f1:.2f}")
    print(f"    R2 alarm 比例：{sum(alarms)/len(alarms)*100:.1f}%")

print("\n"+"="*60)
