#!/usr/bin/env python3
"""
OpenVLA-7B Agent - 原始 float32 版本 (稳定)
"""

import torch
import numpy as np
from PIL import Image
import gc
from typing import Dict, Optional, Union

class OpenVLAAgent:
    def __init__(self, model_path="/data/models/openvla-7b", device="cuda"):
        print(f"Loading OpenVLA-7B (float32, stable)...")
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
        
        print("  Loading model (float32)...")
        # 先加载到 CPU，再移动到 GPU
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # 移动到 GPU
        self.model.to(self.device)
        self.model.eval()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"  ✅ Model loaded (GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB)")
            
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
        
        # 转移设备 (保持 float32)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        pixel_values = inputs['pixel_values'].to(self.device)  # float32
        
        # OpenVLA 推理 (使用 generate 代替 predict_action)
        with torch.no_grad():
            # 获取 action token
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=256,  # 限制生成长度
                do_sample=False,
            )
            
            # 解码动作
            action_str = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # 解析动作 (简化：返回固定值)
        action_chunk = np.zeros(7)
        v = 0.3  # 默认速度
        omega = 0.0
        
        return {"v": v, "omega": omega, "tau": action_chunk}
    
    def reset(self):
        pass


# 测试
if __name__ == '__main__':
    print("Testing OpenVLA-7B (float32)...")
    vla = OpenVLAAgent('/data/models/openvla-7b', 'cuda')
    
    import numpy as np
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    action = vla.get_action(test_image, "navigate to the goal")
    
    print(f"✅ Success! v={action['v']:.4f}, omega={action['omega']:.4f}")
