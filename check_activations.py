#!/usr/bin/env python3
import pickle
import numpy as np
import os

print("="*70)
print("验证现有激活数据")
print("="*70)

# 检查 v2 数据
print("\n【1】检查 v2 数据：/root/act/data/rta_training/activations.pkl")
print("-"*70)
try:
    with open('/root/act/data/rta_training/activations.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"集数：{len(data)}")
    print(f"第一集键：{list(data[0].keys())}")
    
    ep0 = data[0]
    
    enc = ep0.get('encoder_act')
    if enc is None:
        print(f"encoder_act: ❌ None (空)")
    elif isinstance(enc, np.ndarray):
        nonzero_pct = (enc!=0).sum()/enc.size*100
        print(f"encoder_act: shape={enc.shape}, 非零={nonzero_pct:.2f}%, mean={enc.mean():.6f}")
        if (enc==0).all():
            print(f"  ❌ 全零数据！")
        else:
            print(f"  ✅ 正常数据")
    
    dec = ep0.get('decoder_act')
    if dec is None:
        print(f"decoder_act: ❌ None (空)")
    elif isinstance(dec, np.ndarray):
        nonzero_pct = (dec!=0).sum()/dec.size*100
        print(f"decoder_act: shape={dec.shape}, 非零={nonzero_pct:.2f}%, mean={dec.mean():.6f}")
        if (dec==0).all():
            print(f"  ❌ 全零数据！")
        else:
            print(f"  ✅ 正常数据")

except Exception as e:
    print(f"读取失败：{e}")

# 检查 v3 数据
print("\n【2】检查 v3 数据：/root/act/data/rta_training_v3/activations_v3.pkl")
print("-"*70)
if os.path.exists('/root/act/data/rta_training_v3/activations_v3.pkl'):
    try:
        with open('/root/act/data/rta_training_v3/activations_v3.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"集数：{len(data)}")
        
        ep0 = data[0]
        if 'layer_activations' in ep0:
            print("\nlayer_activations:")
            for layer_key, layer_data in ep0['layer_activations'].items():
                if layer_data is not None and len(layer_data) > 0:
                    arr = layer_data[0]
                    nonzero = (arr!=0).sum()/arr.size*100
                    print(f"  {layer_key}: shape={arr.shape}, 非零={nonzero:.1f}%, mean={arr.mean():.6f}")
                else:
                    print(f"  {layer_key}: ❌ None 或空")
            
            dec = ep0.get('decoder_act')
            if dec is not None and len(dec) > 0:
                arr = dec[0]
                nonzero = (arr!=0).sum()/arr.size*100
                print(f"  decoder_act: shape={arr.shape}, 非零={nonzero:.1f}%, mean={arr.mean():.6f}")
            else:
                print(f"  decoder_act: ❌ None 或空")
    except Exception as e:
        print(f"读取失败：{e}")
else:
    print("⏳ v3 数据尚未生成（收集中...）")

print("\n" + "="*70)
print("结论")
print("="*70)
