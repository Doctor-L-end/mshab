import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DARM(nn.Module):
    """
    DARM (Dual-Affinity Refinement Module)
    """
    def __init__(self, 
                 hidden_dim: int, 
                 nheads: int, 
                 dim_feedforward: int, 
                 num_views: int,
                 img_h: int = 16, 
                 img_w: int = 16,
                 dropout: float = 0.1):
        """
        初始化 DARM 模块。
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.head_dim = hidden_dim // nheads
        assert self.head_dim * nheads == self.hidden_dim, "hidden_dim 必须是 nheads 的整数倍"
        self.num_views = num_views
        self.num_patches_per_view = img_h * img_w

        # --- 1. 投影层 (Projections) ---
        self.q_pos_proj = nn.Linear(hidden_dim, hidden_dim) 
        self.q_geo_proj = nn.Linear(hidden_dim, hidden_dim) 
        self.k_sem_proj = nn.Linear(hidden_dim, hidden_dim) 
        self.k_pos_proj = nn.Linear(hidden_dim, hidden_dim) 
        self.v_img_proj = nn.Linear(hidden_dim, hidden_dim)

        # 创建固定的 2D Sinusoidal 位置嵌入
        fixed_pos_emb = self.build_fixed_2d_sinusoidal_pos_emb(
            img_h, img_w, hidden_dim
        )
        self.register_buffer("fixed_img_pos_emb", fixed_pos_emb)

        # 可学习的 "视角" 嵌入
        self.view_pos_emb = nn.Parameter(
            torch.randn(1, num_views, 1, hidden_dim)
        )
        
        # --- 2. 亲和度精炼器 (Affinity Refiner) ---
        self.affinity_refiner = nn.Sequential(
            nn.Conv2d(in_channels=2 * nheads, 
                      out_channels=8 * nheads, 
                      kernel_size=1,  
                      padding=0,      
                      groups=nheads),
            nn.GELU(),
            nn.Conv2d(in_channels=8 * nheads, 
                      out_channels=1 * nheads, 
                      kernel_size=1, 
                      padding=0, 
                      groups=nheads)
        )
        
        # --- 3. 最终融合 FFN ---
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def build_fixed_2d_sinusoidal_pos_emb(self, 
                                          h: int, 
                                          w: int, 
                                          embed_dim: int) -> torch.Tensor:
        """
        构建一个固定的 (1, h*w, embed_dim) 的 2D 位置嵌入
        """
        assert embed_dim % 2 == 0, "嵌入维度 embed_dim 必须是偶数"
        
        dim_t = torch.arange(embed_dim // 2, dtype=torch.float32) 
        dim_t = 10000.0 ** (-2.0 * dim_t / embed_dim)
        
        pos_y, pos_x = torch.meshgrid(torch.arange(h, dtype=torch.float32), 
                                      torch.arange(w, dtype=torch.float32), 
                                      indexing='ij')
        
        embed_x = pos_x.unsqueeze(-1) * dim_t
        embed_y = pos_y.unsqueeze(-1) * dim_t
        
        pos_embed = torch.zeros(h, w, embed_dim)
        pos_embed[:, :, 0::2] = torch.sin(embed_x)
        pos_embed[:, :, 1::2] = torch.cos(embed_x)
        pos_embed[:, :, 0::2] += torch.sin(embed_y)
        pos_embed[:, :, 1::2] += torch.cos(embed_y)
        
        return pos_embed.view(1, h * w, embed_dim) # (1, M, D)
    
    def sinkhorn_algorithm(self, cost_matrix, epsilon=0.05, iterations=5):
        """
        Sinkhorn-Knopp 算法：求解熵正则化的最优传输计划
        
        Args:
            cost_matrix: (B, N, M) 代价矩阵。值越大表示传输代价越高。
            epsilon: 正则化系数。越小越接近真实 OT，但计算越不稳定；越大越平滑。
            iterations: 迭代次数。通常 5-10 次即可收敛到足够好的近似解。
            
        Returns:
            transport_plan: (B, N, M) 最优传输矩阵，行和约为 1/N，列和约为 1/M。
        """
        B, N, M = cost_matrix.shape
        
        # 1. 定义边缘分布 (Marginals)
        # 假设源域 (N) 和目标域 (M) 都是均匀分布
        # r (row) sums to 1, c (col) sums to 1
        # 注意：如果不归一化到 1，Sinkhorn 也能工作，但解释性稍差
        r = torch.ones(B, N, device=cost_matrix.device) / N
        c = torch.ones(B, M, device=cost_matrix.device) / M
        
        # 2. 计算 Gibbs Kernel
        # K = exp(-C / epsilon)
        # epsilon 控制了匹配的"模糊程度"。
        K = torch.exp(-cost_matrix / epsilon)
        
        # 3. Sinkhorn 迭代 (Iterative Scaling)
        # 目标是找到 scaling vectors u 和 v，使得 P = diag(u) * K * diag(v)
        u = torch.ones(B, N, device=cost_matrix.device) / N
        # v = torch.ones(B, M, device=cost_matrix.device) / M (循环中第一步会覆盖它，无需初始化)
        
        for _ in range(iterations):
            # 更新 v: 使得列和逼近 c
            # v = c / (K^T @ u)
            # (B, M, N) @ (B, N, 1) -> (B, M, 1)
            v = c / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-9)
            
            # 更新 u: 使得行和逼近 r
            # u = r / (K @ v)
            # (B, N, M) @ (B, M, 1) -> (B, N, 1)
            u = r / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-9)
            
        # 4. 构建传输计划
        # P = u * K * v
        # 利用广播机制: (B, N, 1) * (B, N, M) * (B, 1, M)
        transport_plan = u.unsqueeze(2) * K * v.unsqueeze(1)
        
        return transport_plan

    def compute_g_scot_loss(self, aff_spatial, aff_semantic, mode='direct'):
        """
        G-SCOT Loss: 几何-语义协同最优传输损失
        
        Args:
            aff_spatial: 空间亲和度 (B, H, N, M) - "Teacher / Cost Provider"
            aff_semantic: 语义亲和度 (B, H, N, M) - "Student / Transporter"
            mode: 'direct' (最小化代价) 或 'sinkhorn' (匹配最优计划)
        """
        # 1. 准备分布
        # 空间分数越高 -> 距离越近 -> 代价(Cost)越小
        # 使用 detach()，因为空间是"物理真理"，我们不反向优化它，只用它来约束语义
        spatial_score = F.softmax(aff_spatial.detach(), dim=-1)
        
        # 语义分布 (我们希望优化的对象)
        semantic_score = F.softmax(aff_semantic, dim=-1)
        
        # 2. 定义代价矩阵 (Cost Matrix)
        # Cost = 1 - Spatial_Score (范围 0~1)
        # 空间上完全不重合(0) -> 代价极大(1)
        # 空间上完全重合(1) -> 代价极小(0)
        cost_matrix = 1.0 - spatial_score 
        
        if mode == 'direct':
            # --- 模式 A: 直接运输代价最小化 (Efficient) ---
            # 目标：让语义分布尽可能集中在"低代价"(即空间近)的区域
            # Expected Cost = Sum( P_sem[i,j] * C[i,j] )
            # 如果 Semantic 在 Spatial=0 (Cost=1) 的地方很高，Loss 就会很大
            
            loss = (semantic_score * cost_matrix).sum(dim=-1).mean()
            
        elif mode == 'sinkhorn':
            # --- 模式 B: Sinkhorn 蒸馏 (High-Precision) ---
            # 目标：计算几何本身的最优传输计划 T*，然后让语义分布 P_sem 去逼近 T*
            
            # 1. 基于几何代价，计算"理想的"传输计划
            # (假设均匀边缘分布，求解最优匹配)
            target_plan = self.sinkhorn_algorithm(
                cost_matrix.view(-1, cost_matrix.shape[2], cost_matrix.shape[3]), 
                epsilon=0.05, 
                iterations=5
            ).view_as(cost_matrix)
            
            # 2. KL 散度：让语义分布逼近理想计划
            # Loss = KL(Target_Plan || Semantic_Prob)
            # 注意：Target_Plan 是硬约束，Semantic 是软预测
            loss = F.kl_div(
                F.log_softmax(aff_semantic, dim=-1), 
                target_plan.detach(), # Teacher 也就是 Target
                reduction='batchmean'
            )
            
        return loss
    
    def forward(self, 
                pc_tokens: torch.Tensor, 
                pc_pos: torch.Tensor, 
                img_patches_multiview: torch.Tensor,
                pc_padding_mask: Optional[torch.Tensor] = None,
                is_training: bool = False,
                ):
       
        B, N, _ = pc_tokens.shape
        _V, M = img_patches_multiview.shape[1:3]
        M_total = _V * M
        img_patches = img_patches_multiview.view(B, M_total, self.hidden_dim)
        
        assert _V == self.num_views, "输入图像的视角数与 DARM 初始化的视角数不匹配"
        assert M == self.num_patches_per_view, "输入图像的 patch 数与 DARM 初始化的 patch 数不匹配"

        # --- 1. 计算 Q, K, V ---
        Q_pos = self.q_pos_proj(pc_pos).view(B, N, self.nheads, self.head_dim).transpose(1, 2)
        Q_geo = self.q_geo_proj(pc_tokens).view(B, N, self.nheads, self.head_dim).transpose(1, 2)
        
        K_sem = self.k_sem_proj(img_patches).view(B, M_total, self.nheads, self.head_dim).transpose(1, 2)
        V_img = self.v_img_proj(img_patches).view(B, M_total, self.nheads, self.head_dim).transpose(1, 2)

        base_pos_emb = self.fixed_img_pos_emb.unsqueeze(1)
        view_pos_emb = self.view_pos_emb
        K_pos_features = (base_pos_emb + view_pos_emb).view(1, M_total, self.hidden_dim)
        K_pos = self.k_pos_proj(K_pos_features.expand(B, -1, -1))
        K_pos = K_pos.view(B, M_total, self.nheads, self.head_dim).transpose(1, 2)

        # --- 2. 显式计算双亲和度矩阵 ---
        affinity_spatial = (Q_pos @ K_pos.transpose(-1, -2))
        affinity_semantic = (Q_geo @ K_sem.transpose(-1, -2))
        
        # --- 3. 先归一化 ---
        scale = self.head_dim ** 0.5
        affinity_spatial_norm = affinity_spatial / scale
        affinity_semantic_norm = affinity_semantic / scale

        # --- 4. 再精炼 (使用 1x1 卷积) ---
        raw_affinity_norm = torch.cat([affinity_spatial_norm, affinity_semantic_norm], dim=1)
        refined_affinity = self.affinity_refiner(raw_affinity_norm).view(B, self.nheads, N, M_total)

        # --- 5. 处理 Padding Mask ---
        attention_matrix = F.softmax(refined_affinity, dim=-1)

        if pc_padding_mask is not None:
            # mask shape: (B, N) -> (B, 1, N, 1)
            mask = pc_padding_mask.view(B, 1, N, 1).expand_as(attention_matrix)
            attention_matrix = attention_matrix.masked_fill(mask, 0.0)

        # --- 6. 获取图像上下文 ---
        img_context_h = (attention_matrix @ V_img)
        img_context = img_context_h.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        
        # --- 7. 最终融合 (残差 + FFN) ---
        fused = self.norm1(pc_tokens + self.dropout(img_context))
        fused = self.norm2(fused + self.dropout(self.ffn(fused)))

        # --- 8. 计算gscot_loss
        loss = None
        if is_training:
            loss = self.compute_g_scot_loss(affinity_spatial_norm, affinity_semantic_norm, "sinkhorn")
        
        return fused, loss