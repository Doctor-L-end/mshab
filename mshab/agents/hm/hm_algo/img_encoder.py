import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoImageProcessor
from peft import get_peft_model, LoraConfig
from typing import Tuple, Dict

class DINOv2Encoder(nn.Module):
    """
    DINOv2 backbone with optional LoRA fine-tuning.
    """
    def __init__(
        self, 
        name: str = "dinov2-base", 
        out_dim: int = 512,
        finetune: str = "lora", 
        dtype = torch.float32,
        lora_rank: int = 16, 
        lora_dropout: float = 0.1,
        extract_layers: Tuple[int] = (4, 8, 12), 
        output_keys: Tuple[str] = ('shallow', 'mid', 'deep') 
    ):
        super().__init__()
        assert finetune in ["full", "lora", "none"], "finetune parameter should be one of [full, lora, none]."
        assert len(extract_layers) == len(output_keys), "extract_layers 和 output_keys 必须长度一致"
        
        self.extract_layers = extract_layers
        self.output_keys = output_keys
        
        model_path = os.path.join("/raid/ljh/DSPv2/weights/", name)
        if not os.path.exists(model_path):
             print(f"Warning: Local path {model_path} not found. Trying 'facebook/{name}'.")
             model_path = f"facebook/{name}"
             
        dino = AutoModel.from_pretrained(model_path, torch_dtype = dtype)
        self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)

        if finetune == "lora":
            dino.requires_grad_(False)
            config = LoraConfig(
                r              = lora_rank,
                lora_alpha     = lora_rank,
                target_modules = ['projection', 'query', 'key', 'value', 'dense', 'fc1', 'fc2'],
                lora_dropout   = lora_dropout,
                bias           = 'none',
                use_rslora     = True,
            )
            dino = get_peft_model(dino, config)
            for name, param in dino.named_parameters():
                if "lora_" in name:
                    param.data = param.data.float()
        elif finetune == "none":
            dino.requires_grad_(False)
        
        self.model = dino

        self.patch_size = dino.config.patch_size
        hidden_size = dino.config.hidden_size
        if hidden_size != out_dim:
            # 投影层
            self.proj = nn.Linear(hidden_size, out_dim)
        else:
            self.proj = nn.Identity()
        self.num_channels = out_dim

        self.output_ln = nn.LayerNorm(out_dim)

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回一个多尺度特征字典。
        """
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = inputs['pixel_values']
        
        # 1. 请求模型输出所有隐藏层
        outputs = self.model(inputs, output_hidden_states=True)
        
        # .hidden_states 是一个元组: (embeds, L1_out, ..., L12_out)
        all_hidden_states = outputs.hidden_states 

        # 2. 循环提取、处理和投影我们想要的层
        output_dict = {}
        for key, layer_idx in zip(self.output_keys, self.extract_layers):
            # (B, N_patches + 1, D_hidden)
            # layer_idx 对应 all_hidden_states 中的索引 (e.g., L4 = 索引 4)
            layer_feats = all_hidden_states[layer_idx]
            
            # (B, N_patches, D_hidden) -> 丢弃 [CLS] token
            patch_feats = layer_feats[:, 1:]
            
            # (B, N_patches, D_out) -> 应用投影
            projected_feats = self.proj(patch_feats)
            projected_feats = self.output_ln(projected_feats)
            output_dict[key] = projected_feats
            
        output_dict["global"] = self.proj(all_hidden_states[-1][:,0,:]).unsqueeze(1) # (B,1,D_out)
        # 3. 返回字典{'shallow': (B,N,D), 'mid': (B,N,D), 'deep': (B,N,D), 'global': (B,1,D)}
        return output_dict