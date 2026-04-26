#!/usr/bin/env python3
"""
OpenVLA-7B Agent - 8bit 量化版本 (节省显存)
"""

import torch
import numpy as np
from PIL import Image
import gc
from typing import Dict, Optional, Union

class OpenVLAAgent:
    def __init__(self, model_path="/data/models/openvla-7b", device="cuda"):
        print(f"Loading OpenVLA-7B (8bit quantization)...")
        self.device = device
        self.model = None
        self.processor = None
        self._load_model_8bit(model_path)
        print("✅ OpenVLA-7B loaded")
        
    def _load_model_8bit(self, model_path: str):
        """8bit 量化加载 (节省显存)"""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        print("  Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        print("  Loading model (8bit)...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # FP16
            trust_remote_code=True,
            load_in_8bit=True,  # 8bit 量化
            device_map="auto",  # 自动分配到 GPU
        )
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
        
        # 转移设备 (保持 float16)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        pixel_values = inputs['pixel_values'].to(self.device).half()  # 转为 float16
        
        # OpenVLA 推理
        with torch.no_grad():
            outputs = self.model.predict_action(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                unnorm_key="bridge_orig"
            )
        
        action_chunk = outputs[0].cpu().numpy()
        v = float(action_chunk[0]) * 0.5
        omega = float(action_chunk[1]) * 0.8
        
        return {"v": v, "omega": omega, "tau": np.zeros(7)}
    
    def reset(self):
        pass


# 测试
if __name__ == '__main__':
    print("Testing OpenVLA-7B (8bit)...")
    vla = OpenVLAAgent('/data/models/openvla-7b', 'cuda')
    
    import numpy as np
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    action = vla.get_action(test_image, "navigate to the goal")
    
    print(f"✅ Success! v={action['v']:.4f}, omega={action['omega']:.4f}")
