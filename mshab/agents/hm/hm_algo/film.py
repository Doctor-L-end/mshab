import torch 
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple

class FiLMGenerator(nn.Module):
    def __init__(self, 
                 history_hidden_dim: int, 
                 feature_dim: int,
                 context_dim: int, 
                 mlp_hidden_dim: int = 256
                ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(history_hidden_dim + context_dim, mlp_hidden_dim), 
            nn.GELU(),
            nn.LayerNorm(mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, feature_dim * 2)
        )
    
    def forward(self, 
                h_t: torch.Tensor,       # (B, history_hidden_dim)
                context: torch.Tensor    # (B, context_dim)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 在输入端融合 "意图" 和 "上下文"
        combined_input = torch.cat([h_t, context], dim=-1) # (B, D_hist + D_context)
        
        gamma_beta = self.mlp(combined_input)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        gamma = (torch.tanh(gamma) + 1.0) 
        
        return gamma, beta # gamma:[0,2] beta:无限制