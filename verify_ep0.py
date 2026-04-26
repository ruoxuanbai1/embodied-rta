#!/usr/bin/env python3
import pickle
import numpy as np

print("="*70)
print("验证 v3.1 激活数据 (第一集)")
print("="*70)

with open('/root/act/data/rta_training_v3/ep000.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"\n成功加载第 1 集")
print(f"键：{list(data.keys())}")
print(f"qpos shape: {data['qpos'].shape}")
print(f"action shape: {data['action'].shape}")
print(f"success: {data['success']}")

print("\n" + "="*70)
print("激活数据验证")
print("="*70)

layer_acts = data['layer_activations']
all_good = True
for layer_key in ['layer0_ffn', 'layer1_ffn', 'layer2_ffn', 'layer3_ffn']:
    arr = layer_acts.get(layer_key)
    if arr is None:
        print(f"❌ {layer_key}: None")
        all_good = False
    else:
        print(f"✅ {layer_key}: shape={arr.shape}")
        frame0 = arr[0]
        nonzero = (frame0!=0).sum()/frame0.size*100
        print(f"   第 1 帧：非零={nonzero:.1f}%, mean={frame0.mean():.6f}, std={frame0.std():.6f}")
        if (frame0==0).all():
            print(f"   ❌ 全零！")
            all_good = False

print("\n" + "="*70)
if all_good:
    print("🎉 所有层激活数据正常！")
else:
    print("⚠️ 部分数据异常")
print("="*70)
