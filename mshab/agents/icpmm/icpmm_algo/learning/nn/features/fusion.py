import torch
import torch.nn as nn
from einops import rearrange

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_head=8, is_causal=False):
        super().__init__()
        self.num_head = num_head
        self.is_causal = is_causal
        self.scale = (dim // num_head) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)  
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, state, pointcloud):
        # state: (B, N, D), pointcloud: (B, N, D)
        q = self.to_q(state) # (B, N, D)
        k, v = self.to_kv(pointcloud).chunk(2, dim=-1)  # 各(B, N, D)
        
        # 多头拆分
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_head)  # (B, H, N, D/H)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_head)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_head)
        
        # 注意力计算
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, N, N)
        # 添加因果掩码
        if self.is_causal:
            N = attn.shape[-1]
            mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
            attn = attn.masked_fill(mask.to(attn.device), -torch.inf)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, H, N, D/H)
        
        # 合并多头
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # (B, N, D)
    
# 分层交叉注意力
class EnhancedFusion(nn.Module):
    def __init__(self, dim, num_head=8, is_causal=False):
        super().__init__()
        self.cross_attn1 = CrossModalAttention(dim, num_head=num_head, is_causal=is_causal)  # 状态→点云
        self.cross_attn2 = CrossModalAttention(dim, num_head=num_head, is_causal=is_causal)  # 点云→状态
        self.mlp = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, state, pointcloud):
        # 双向交叉注意力
        fused_state = self.cross_attn1(state, pointcloud)  # 用状态查询点云
        fused_pc = self.cross_attn2(pointcloud, state)    # 用点云查询状态
        
        # 合并双向结果
        out = torch.cat([fused_state, fused_pc], dim=-1)
        return self.mlp(out)  # (B,N,D)