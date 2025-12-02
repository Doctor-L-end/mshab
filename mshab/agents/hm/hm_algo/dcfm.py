import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoderLayer 
from typing import Optional

class DCFM_ActionDecoder(nn.Module):
    """
    DCFM (Decoupled Coordinated Flow Matching) 动作解码器
    架构流程:
    1. Action Space: 采样噪声 x0, 获取目标 x1, 插值得到 xt, 计算目标流 yt。
    2. Encoder: 将 xt 投影到 Latent Space。
    3. DCFM Stack: 在 Latent Space 进行解耦协调 (Arm <-> Base)。
    4. Decoder: 将 Latent Space 的输出投影回 Action Space, 得到 pred_yt。
    5. Action Space: 计算 Loss(pred_yt, yt)。
    """
    def __init__(
        self,
        obs_feature_dim: int,    # 观测融合输出的维度
        state_dim: int,          # 本体感觉状态的维度
        base_action_dim: int,    # 底盘动作维度
        arm_action_dim: int,     # 手臂动作维度
        Tpred: int,              # 预测的轨迹长度
        
        # --- 超参数 ---
        hidden_dim: int = 512,     # 内部 FlowNet 的维度
        nheads: int = 8,           # DiT 和 协调器 的头数
        num_flownet_layers: int = 4, # 两个 FlowNet 的层数
        dropout: float = 0.1
    ):
        super().__init__()
        self.Tpred = Tpred
        self.arm_action_dim = arm_action_dim
        self.base_action_dim = base_action_dim
        # 计算总动作维度，用于共享噪声的采样
        self.total_action_dim = arm_action_dim + base_action_dim
        
        self.hidden_dim = hidden_dim
        self.num_flownet_layers = num_flownet_layers
        self.dropout = dropout

        # 1. 条件投影层
        self.fused_proj = nn.Linear(obs_feature_dim, hidden_dim) # obs_cond
        self.state_proj = nn.Linear(state_dim, hidden_dim)       # state_cond
        
        # 时间嵌入 MLP
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 2, hidden_dim), 
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 编码器 (Encoder): Action Space -> Latent Space
        self.flow_proj_arm = nn.Linear(self.arm_action_dim, hidden_dim) 
        self.flow_proj_base = nn.Linear(self.base_action_dim, hidden_dim)
        
        # 轨迹位置嵌入 (Tpred + 3 个条件 token + 1 个意图 token)
        self.num_cond_tokens = 3 # (obs, state, t)
        self.num_intent_tokens = 1 # 专用的 "意图" token
        self.traj_pos_embed = nn.Parameter(
            torch.randn(1, Tpred + self.num_cond_tokens + self.num_intent_tokens, hidden_dim)
        )

        # 2. 解耦的 FlowNet Stack (DiT + DAC 协调器)
        self.sa_arm_layers = nn.ModuleList()   # Self-Attention
        self.sa_base_layers = nn.ModuleList()
        
        self.coord_arm_layers = nn.ModuleList() # Cross-Attention (Arm queries Base)
        self.coord_base_layers = nn.ModuleList() # Cross-Attention (Base queries Arm)

        # 专用的 "意图" token
        self.arm_intent_token = nn.Parameter(torch.randn(1, self.num_intent_tokens, hidden_dim))
        self.base_intent_token = nn.Parameter(torch.randn(1, self.num_intent_tokens, hidden_dim))

        # 后处理 FFN 和 Norms
        self.norm1_arm = nn.ModuleList()
        self.norm1_base = nn.ModuleList()
        self.norm2_arm = nn.ModuleList()
        self.norm2_base = nn.ModuleList()
        self.norm3_arm = nn.ModuleList()
        self.norm3_base = nn.ModuleList()
        
        self.ffn_arm = nn.ModuleList()
        self.ffn_base = nn.ModuleList()

        for _ in range(self.num_flownet_layers):
            # Self-Attention (DiT)
            self.sa_arm_layers.append(nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True))
            self.sa_base_layers.append(nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True))
            
            # Cross-Attention (Coordinator)
            self.coord_arm_layers.append(nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True))
            self.coord_base_layers.append(nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True))
            
            # Norms
            self.norm1_arm.append(nn.LayerNorm(hidden_dim))
            self.norm1_base.append(nn.LayerNorm(hidden_dim))
            self.norm2_arm.append(nn.LayerNorm(hidden_dim))
            self.norm2_base.append(nn.LayerNorm(hidden_dim))
            self.norm3_arm.append(nn.LayerNorm(hidden_dim))
            self.norm3_base.append(nn.LayerNorm(hidden_dim))
            
            # FFNs
            self.ffn_arm.append(self._build_ffn(hidden_dim, hidden_dim * 4, dropout))
            self.ffn_base.append(self._build_ffn(hidden_dim, hidden_dim * 4, dropout))

        # 3. 解码器 (Decoder): Latent Space -> Action Space
        self.arm_flow_head = nn.Linear(hidden_dim, self.arm_action_dim)
        self.base_flow_head = nn.Linear(hidden_dim, self.base_action_dim)
        
        # 4. 损失函数
        self.flow_loss_fn = nn.MSELoss()

    def _build_ffn(self, hidden_dim, dim_feedforward, dropout):
        return nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )

    def _build_time_emb(self, t: torch.Tensor) -> torch.Tensor:
        """构建正弦时间嵌入"""
        B = t.shape[0]
        t_emb = t.view(B, 1) 
        half_dim = self.hidden_dim // 2
        dim_t = torch.arange(half_dim, dtype=torch.float32, device=t.device)
        dim_t = 10000.0 ** (-2.0 * dim_t / half_dim)
        sin_t = torch.sin(t_emb * dim_t)
        cos_t = torch.cos(t_emb * dim_t)
        time_emb = torch.cat([sin_t, cos_t], dim=-1)
        if self.hidden_dim % 2 == 1:
            time_emb = F.pad(time_emb, (0, 1))
        return self.time_proj(time_emb)

    def _run_dcfm_stack(self, 
                       xt_arm_latent: torch.Tensor, # (B, T, D_hidden)
                       xt_base_latent: torch.Tensor, # (B, T, D_hidden)
                       cond_tokens: torch.Tensor     # (B, 3, D_hidden)
                       ):
        """
        运行 DCFM 堆栈 (在高维隐空间中进行解耦协调)
        """
        B = xt_arm_latent.shape[0] 

        # (B, 3+1+T, D) - 插入专用的 intent token
        xt_arm_in = torch.cat([
            cond_tokens, 
            self.arm_intent_token.expand(B, -1, -1), 
            xt_arm_latent
        ], dim=1) + self.traj_pos_embed
        
        xt_base_in = torch.cat([
            cond_tokens, 
            self.base_intent_token.expand(B, -1, -1), 
            xt_base_latent
        ], dim=1) + self.traj_pos_embed

        for i in range(self.num_flownet_layers):
            
            # --- 1. 自注意力 (Indepedent Thinking) ---
            norm_arm_in = self.norm1_arm[i](xt_arm_in)
            norm_base_in = self.norm1_base[i](xt_base_in)
            
            xt_arm_sa_out, _ = self.sa_arm_layers[i](norm_arm_in, norm_arm_in, norm_arm_in)
            xt_base_sa_out, _ = self.sa_base_layers[i](norm_base_in, norm_base_in, norm_base_in)
            
            xt_arm_sa = xt_arm_in + xt_arm_sa_out
            xt_base_sa = xt_base_in + xt_base_sa_out

            # --- 2. 提取意图 & Detach (Gradient Decoupling) ---
            # 提取 intent token 对应的输出
            idx_start = self.num_cond_tokens
            idx_end = self.num_cond_tokens + self.num_intent_tokens
            
            arm_intent_vec = xt_arm_sa[:, idx_start:idx_end, :]
            base_intent_vec = xt_base_sa[:, idx_start:idx_end, :]
            
            # 关键步骤：阻断梯度传播
            arm_intent = arm_intent_vec.detach()
            base_intent = base_intent_vec.detach()
            
            # --- 3. 解耦协调 (Coordinated Communication) ---
            norm_arm_sa = self.norm2_arm[i](xt_arm_sa)
            norm_base_sa = self.norm2_base[i](xt_base_sa)
            
            # Arm 查询 Base 的意图
            context_for_arm, _ = self.coord_arm_layers[i](norm_arm_sa, base_intent, base_intent)
            # Base 查询 Arm 的意图
            context_for_base, _ = self.coord_base_layers[i](norm_base_sa, arm_intent, arm_intent)
            
            xt_arm_coord = xt_arm_sa + context_for_arm
            xt_base_coord = xt_base_sa + context_for_base

            # --- 4. FFN ---
            xt_arm_ffn_out = self.ffn_arm[i](self.norm3_arm[i](xt_arm_coord))
            xt_base_ffn_out = self.ffn_base[i](self.norm3_base[i](xt_base_coord))

            xt_arm_in = xt_arm_coord + xt_arm_ffn_out
            xt_base_in = xt_base_coord + xt_base_ffn_out
            
        # 移除条件 token 和 意图 token，只返回预测的向量场 Latent
        output_slice = slice(self.num_cond_tokens + self.num_intent_tokens, None)
        pred_latent_arm = xt_arm_in[:, output_slice, :] 
        pred_latent_base = xt_base_in[:, output_slice, :]
        
        return pred_latent_arm, pred_latent_base

    def forward(
        self,
        fused_vector: torch.Tensor, # (B, D_obs)
        current_state: torch.Tensor,  # (B, D_state)
        # --- 仅在训练时提供 ---
        gt_future_actions: Optional[torch.Tensor] = None # (B, Tpred, D_action)
    ):
        """
        前向传播函数
        """
        B = fused_vector.shape[0]
        device = fused_vector.device

        # --- 准备共享条件 Token ---
        obs_cond = self.fused_proj(fused_vector).unsqueeze(1)
        state_cond = self.state_proj(current_state).unsqueeze(1)
        
        # --- 训练模式 ---
        if gt_future_actions is not None:
            
            # 1. 准备时间 (t)
            t_rand = torch.rand(B, 1, 1, device=device) # (B, 1, 1)
            time_emb = self._build_time_emb(t_rand) # (B, D_hidden)
            t_cond = time_emb.unsqueeze(1) # (B, 1, D_hidden)
            
            cond_tokens = torch.cat([obs_cond, state_cond, t_cond], dim=1) # (B, 3, D_hidden)

            # 2. 在 Action-Space 准备流匹配目标
            
            # 2a. 准备 x1 (Target)
            x1_base_action = gt_future_actions[..., :self.base_action_dim]
            x1_arm_action = gt_future_actions[..., self.base_action_dim:]

            # 2b. 准备 x0 (Noise) - 共享噪声源
            # 即使 Arm 和 Base 维度不同，也应该从同一组 RNG 种子生成，或者共享部分结构
            # 这里最简单的做法是：生成一个完整的随机动作向量，然后切分
            # 这样确保了在 t=0 时，Arm 和 Base 的噪声输入虽然值不同，但统计分布是同步的
            x0_noise_action_full = torch.randn_like(gt_future_actions)
            x0_base_action = x0_noise_action_full[..., :self.base_action_dim]
            x0_arm_action = x0_noise_action_full[..., self.base_action_dim:]

            # 2c. 插值 xt (Input)
            xt_base_action = (1.0 - t_rand) * x0_base_action + t_rand * x1_base_action
            xt_arm_action = (1.0 - t_rand) * x0_arm_action + t_rand * x1_arm_action
            
            # 2d. 计算 yt (Target Flow)
            yt_base_action_target = x1_base_action - x0_base_action
            yt_arm_action_target = x1_arm_action - x0_arm_action
            
            # 3. 编码: Action Space -> Latent Space
            xt_base_latent = self.flow_proj_base(xt_base_action)
            xt_arm_latent = self.flow_proj_arm(xt_arm_action)

            # 4. 运行 DCFM 堆栈 (High-Dim Processing)
            pred_latent_arm, pred_latent_base = self._run_dcfm_stack(
                xt_arm_latent, xt_base_latent, cond_tokens
            )
            
            # 5. 解码: Latent Space -> Action Space
            pred_yt_arm_action = self.arm_flow_head(pred_latent_arm)
            pred_yt_base_action = self.base_flow_head(pred_latent_base)
            
            # 6. 计算损失 (Action Space)
            loss_arm = self.flow_loss_fn(pred_yt_arm_action, yt_arm_action_target)
            loss_base = self.flow_loss_fn(pred_yt_base_action, yt_base_action_target)
            total_loss = loss_arm + loss_base
            
            return total_loss

        # --- 推理模式 ---
        else:
            # 1. 准备时间 (t=0)
            t_zero = torch.zeros(B, 1, 1, device=device)
            time_emb = self._build_time_emb(t_zero)
            t_cond = time_emb.unsqueeze(1) # (B, 1, D_hidden)
            cond_tokens = torch.cat([obs_cond, state_cond, t_cond], dim=1)

            # 2. 准备 x0 (Action Space) - 共享噪声
            x0_noise_action_full = torch.randn(B, self.Tpred, self.total_action_dim, device=device)
            x0_base_action = x0_noise_action_full[..., :self.base_action_dim]
            x0_arm_action = x0_noise_action_full[..., self.base_action_dim:]

            # 3. 编码
            x0_base_latent = self.flow_proj_base(x0_base_action)
            x0_arm_latent = self.flow_proj_arm(x0_arm_action)

            # 4. 运行 DCFM
            pred_latent_arm, pred_latent_base = self._run_dcfm_stack(
                x0_arm_latent, x0_base_latent, cond_tokens
            )
            
            # 5. 解码
            pred_yt_arm_action = self.arm_flow_head(pred_latent_arm)
            pred_yt_base_action = self.base_flow_head(pred_latent_base)
            
            # 6. 单步流匹配更新
            # x1 = x0 + yt
            x1_arm_action = x0_arm_action + pred_yt_arm_action
            x1_base_action = x0_base_action + pred_yt_base_action
            
            # 7. 组合输出
            pred_actions = torch.cat([x1_base_action, x1_arm_action], dim=-1)

            return pred_actions