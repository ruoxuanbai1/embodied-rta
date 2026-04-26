#!/usr/bin/env python3
"""Region 2 GRU 模型验证"""
import torch, torch.nn as nn, numpy as np, json, time
from pathlib import Path

print("="*80)
print("Region 2 GRU 模型验证")
print("="*80)

MODEL_PATH = Path("/mnt/data/region2_training_v2/region2_gru_best.pth")
DATA_PATH = Path("/mnt/data/region2_training_v2/region2_training_data.npz")
OUTPUT_DIR = Path("/mnt/data/region2_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备：{DEVICE}")

class ReachabilityGRU(nn.Module):
    def __init__(self, state_dim=14, action_dim=14, hidden_dim=512, num_layers=4, n_variables=14):
        super().__init__()
        input_dim = state_dim + action_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.gru_norm = nn.LayerNorm(hidden_dim)
        self.fc_min = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim//2, n_variables))
        self.fc_max = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim//2, n_variables))
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = torch.relu(x)
        _, h = self.gru(x)
        h = h[-1]
        h = self.gru_norm(h)
        h = torch.relu(h)
        return self.fc_min(h), self.fc_max(h)

print("\n[1/4] 加载模型...")
model = ReachabilityGRU(hidden_dim=512, num_layers=4).to(DEVICE)
checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"✅ 模型加载成功")

print("\n[2/4] 加载数据...")
data = np.load(str(DATA_PATH))
X_states, X_actions, Y_min, Y_max = data["X_states"], data["X_actions"], data["Y_min"], data["Y_max"]
print(f"总样本数：{len(X_states)}")

np.random.seed(42)
test_idx = np.random.permutation(len(X_states))[:len(X_states)//10]
print(f"测试集：{len(test_idx)} 样本")

X_states_t = torch.from_numpy(X_states[test_idx]).float().to(DEVICE)
X_actions_t = torch.from_numpy(X_actions[test_idx]).float().to(DEVICE)
Y_min_t = torch.from_numpy(Y_min[test_idx]).float().to(DEVICE)
Y_max_t = torch.from_numpy(Y_max[test_idx]).float().to(DEVICE)

print("\n[3/4] 运行预测...")
start = time.time()
with torch.no_grad():
    min_p, max_p = model(X_states_t, X_actions_t)
print(f"推理时间：{time.time()-start:.3f}s ({len(test_idx)/(time.time()-start):.1f} samples/s)")

print("\n[4/4] 性能评估...")
min_np, max_np = min_p.cpu().numpy(), max_p.cpu().numpy()
Y_min_np, Y_max_np = Y_min_t.cpu().numpy(), Y_max_t.cpu().numpy()

mae_min = np.mean(np.abs(min_np - Y_min_np))
mae_max = np.mean(np.abs(max_np - Y_max_np))
mae_total = (mae_min + mae_max) / 2

covered = ((Y_min_np >= min_np) & (Y_max_np <= max_np)).all(axis=1)
coverage = covered.mean() * 100

conservatism = np.mean((max_np - min_np) / (Y_max_np - Y_min_np + 1e-8))

var_names = ["base_x","base_y","base_v","base_w","ee_x","ee_y","ee_z","ee_vx","ee_vy","ee_vz","ee_qx","ee_qy","ee_qz","ee_qw"]
var_cov = ((Y_min_np >= min_np) & (Y_max_np <= max_np)).mean(axis=0) * 100

print("\n" + "="*50)
print("验证结果")
print("="*50)
print(f"覆盖率：{coverage:.2f}% (目标≥95%)")
print(f"MAE: {mae_total:.4f}")
print(f"保守度：{conservatism:.2f}x")

print("\n各变量覆盖率:")
for n, c in zip(var_names, var_cov):
    print(f"  {'✅' if c>=95 else '⚠️'} {n:8s}: {c:6.2f}%")

not_covered = ~covered
if not_covered.sum() > 0:
    var_not_covered = ~((Y_min_np >= min_np) & (Y_max_np <= max_np))
    var_fail = var_not_covered.sum(axis=0)
    worst_idx = np.argmax(var_fail)
    print(f"\n最难预测变量：{var_names[worst_idx]} (失败{var_fail[worst_idx]}次)")
    
    errors = np.abs(min_np - Y_min_np) + np.abs(max_np - Y_max_np)
    print(f"\n误差分布：P50={np.percentile(errors,50):.4f}, P95={np.percentile(errors,95):.4f}, P99={np.percentile(errors,99):.4f}")

report = {
    "coverage": float(coverage),
    "mae_total": float(mae_total),
    "conservatism": float(conservatism),
    "var_coverage": dict(zip(var_names, [float(c) for c in var_cov])),
    "test_samples": len(test_idx),
    "inference_speed": float(len(test_idx)/(time.time()-start))
}

with open(OUTPUT_DIR/"verification.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*50)
if coverage >= 95:
    print("✅ 模型验证通过！")
else:
    print("❌ 模型验证未通过！")
print(f"报告：{OUTPUT_DIR/'verification.json'}")
