#!/usr/bin/env python3
"""
OpenVLA-7B Agent - 被测对象 (真实模型，float32 稳定版)
"""

import torch
import numpy as np
from PIL import Image
import gc

if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

from typing import Dict, Optional, Union

class OpenVLAAgent:
    def __init__(self, model_path="/data/models/openvla-7b", device="cuda"):
        print(f"Loading OpenVLA-7B from {model_path}...")
        self.device = device
        self.model = None
        self.processor = None
        self._load_model_float32(model_path)
        print("✅ OpenVLA-7B loaded")
        
    def _load_model_float32(self, model_path: str):
        """Float32 加载 (稳定但耗内存)"""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        print("  Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        print("  Loading model (float32, low_cpu_mem_usage)...")
        # 使用 low_cpu_mem_usage 减少 CPU 内存
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # 明确移动到 GPU
        self.model.to(self.device)
        self.model.eval()
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"  ✅ Model loaded (GPU: {torch.cuda.get_device_name(0)})")
        if torch.cuda.is_available():
            print(f"     GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
            
    def get_action(self, image: Union[np.ndarray, Image.Image], 
                   instruction: str,
                   proprio: Optional[Dict] = None) -> Dict:
        if self.model is None:
            raise RuntimeError("OpenVLA model not loaded!")
        
        # 转换为 PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        # Processor 处理
        inputs = self.processor(text=instruction, images=[image], return_tensors="pt")
        
        # 转移设备
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # OpenVLA 推理 (使用 generate 直接调用)
        with torch.no_grad():
            # 直接调用 generate
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=256,
                do_sample=False,
            )
            
            # 提取动作 tokens (去掉输入部分)
            action_tokens = generated_ids[0, input_ids.shape[1]:]
            
            # OpenVLA action tokenization: tokens 需要减去 32000 偏移量
            # 参考：https://github.com/openvla/openvla/blob/main/prismatic/models/vla.py
            action_bins = action_tokens.cpu().numpy().astype(np.int32) - 32000
            
            # 归一化到 [-1, 1] 范围 (OpenVLA 使用 256 bins)
            action_normalized = (action_bins.astype(np.float32) / 128.0) - 1.0
        
        # 获取动作 (OpenVLA 输出 7 维：[vx, vy, vz, wx, wy, wz, gripper])
        # 对于移动底盘：v = vx, omega = wz
        v = float(np.clip(action_normalized[0], -1, 1)) * 0.5 if len(action_normalized) > 0 else 0.3
        omega = float(np.clip(action_normalized[5], -1, 1)) * 0.8 if len(action_normalized) > 5 else 0.0
        
        return {"v": v, "omega": omega, "tau": np.zeros(7)}
    
    def reset(self):
        pass
