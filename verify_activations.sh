#!/bin/bash
# 监控收集进度，完成后验证数据

echo "等待收集完成..."
while ! grep -q "COMPLETE" /root/act/collect_v3_xvfb.log 2>/dev/null; do
    sleep 60
    tail -3 /root/act/collect_v3_xvfb.log | grep "Collecting"
done

echo ""
echo "=========================================="
echo "收集完成！验证激活数据..."
echo "=========================================="

/home/vipuser/miniconda3/envs/aloha_headless/bin/python3 << 'PYEOF'
import pickle
import numpy as np

print("加载数据...")
with open('/root/act/data/rta_training_v3/activations_v3.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"\n✅ 成功加载 {len(data)} 集")
print(f"第一集类型：{type(data[0])}")
print(f"键：{list(data[0].keys())}")

print("\n" + "="*60)
print("激活数据验证")
print("="*60)

ep0 = data[0]
for k, v in ep0.items():
    if k == 'layer_activations':
        print(f"\n{k}:")
        for layer_key, layer_data in v.items():
            if layer_data is not None and len(layer_data) > 0:
                arr = layer_data[0]
                print(f"  {layer_key}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}, nonzero={(arr!=0).sum()}/{arr.size} ({(arr!=0).sum()/arr.size*100:.1f}%)")
            else:
                print(f"  {layer_key}: None 或空")
    elif v is not None:
        if hasattr(v, 'shape'):
            print(f"{k}: shape={v.shape}, mean={v.mean():.6f}")
        else:
            print(f"{k}: {v}")
    else:
        print(f"{k}: None")

print("\n" + "="*60)
print("验证结论")
print("="*60)

# 检查激活数据是否非空
all_good = True
for k, v in ep0['layer_activations'].items():
    if v is None or len(v) == 0:
        print(f"❌ {k}: 数据为空")
        all_good = False
    else:
        arr = v[0]
        if (arr == 0).all():
            print(f"❌ {k}: 全零数据")
            all_good = False
        else:
            nonzero_pct = (arr!=0).sum()/arr.size*100
            print(f"✅ {k}: 正常 (非零元素 {nonzero_pct:.1f}%)")

if ep0.get('decoder_act') is None or len(ep0['decoder_act']) == 0:
    print(f"❌ decoder_act: 数据为空")
    all_good = False
else:
    arr = ep0['decoder_act'][0]
    if (arr == 0).all():
        print(f"❌ decoder_act: 全零数据")
        all_good = False
    else:
        nonzero_pct = (arr!=0).sum()/arr.size*100
        print(f"✅ decoder_act: 正常 (非零元素 {nonzero_pct:.1f}%)")

print("\n" + "="*60)
if all_good:
    print("🎉 所有激活数据正常！可以继续训练 Region 3")
else:
    print("⚠️ 部分数据异常，请检查")
print("="*60)
PYEOF
