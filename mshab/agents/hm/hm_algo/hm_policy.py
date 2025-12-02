import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Dict

from .pc_encoder import Sparse3DEncoderPMP
from .img_encoder import DINOv2Encoder
from .film import FiLMGenerator
from .darm import DARM
from .dcfm import DCFM_ActionDecoder

import MinkowskiEngine as ME

class HMPolicy(nn.Module):
    """
    1.  历史意图 (h_t):
        - 使用一个 Transformer (History Transformer) 来融合 "历史本体感觉" (s_hist)
          和 "当前视觉全局特征" (v_t_global)。
    2.  分层融合 (Hierarchical Fusion):
            - shallow_pc + shallow_img
            - mid_pc + mid_img
            - deep_pc + deep_img
    3.  感知调制 (PMP):
        - 使用h_t通过 FiLM 动态调制每个层次融合后的特征。
    4.  解码:
        - 压缩所有调制后的融合特征，送入动作解码器。
    """
    def __init__(
        self,
        Tpred = 8,
        Tact = 2,
        T_hist = 16,
        pc_feats_input_dim = 3,
        obs_feature_dim = 512,
        action_dim = 11,
        mobility_action_dim = 3,
        state_dim = 42,
        num_views = 2,
        history_hidden_dim: int = 256,
        history_nhead: int = 4,
        history_nlayers: int = 2,
        fusion_nhead: int = 8,
    ):
        super().__init__()
        self.Tact = Tact
        self.T_hist = T_hist
        self.num_views = num_views
        self.obs_feature_dim = obs_feature_dim
        self.history_hidden_dim = history_hidden_dim
        self.scale_keys = ['shallow', 'mid', 'deep'] # 2D/3D 编码器统一使用
        self.num_scales = len(self.scale_keys)

        # --- 1. 感知编码器 (2D & 3D) ---
        self.pc_encoder = Sparse3DEncoderPMP(
            input_dim=pc_feats_input_dim,
            output_dim=obs_feature_dim,
        )

        self.img_encoder = DINOv2Encoder(
            out_dim=obs_feature_dim, finetune="lora", dtype=torch.float32,
            output_keys=self.scale_keys
        )
        self.patch_size = 16  # DINOv2 默认 patch size

        # --- 2. PMP 历史编码器---
        self.history_state_proj = nn.Linear(state_dim, history_hidden_dim)
        self.history_vision_proj = nn.Linear(num_views * obs_feature_dim, history_hidden_dim)
        self.history_cls_token = nn.Parameter(torch.randn(1, 1, history_hidden_dim))
        history_transformer_layer = nn.TransformerEncoderLayer(
            d_model=history_hidden_dim,
            nhead=history_nhead,
            dim_feedforward=history_hidden_dim * 4,
            batch_first=True,
            activation=F.gelu
        )
        self.history_transformer = nn.TransformerEncoder(
            history_transformer_layer,
            num_layers=history_nlayers
        )
        self.history_pos_emb = nn.Parameter(torch.randn(1, T_hist + 2, history_hidden_dim))

        # --- 3. DARM融合器 ---
        self.darms = nn.ModuleList([
            DARM(hidden_dim=obs_feature_dim, nheads=fusion_nhead, dim_feedforward=obs_feature_dim * 4,
                num_views=num_views,
                img_h=self.patch_size,
                img_w=self.patch_size)
            for _ in range(self.num_scales)
        ])

        # --- 4. PMP - FiLM 生成器 (用于融合后的特征) ---
        self.scale_context_embeddings = nn.Parameter(
            torch.randn(self.num_scales, self.history_hidden_dim) # (3, D_hist)
        )

        self.film_generator = FiLMGenerator(
            history_hidden_dim=self.history_hidden_dim,
            feature_dim=self.obs_feature_dim,
            context_dim=self.history_hidden_dim
        )

        # --- 5. 融合后压缩 ---
        # 用于压缩多尺度融合后的 token
        self.cls_token_s = nn.Parameter(torch.randn(1, 1, obs_feature_dim))
        self.cls_token_m = nn.Parameter(torch.randn(1, 1, obs_feature_dim))
        self.cls_token_d = nn.Parameter(torch.randn(1, 1, obs_feature_dim))

        summarizer_layer_s = nn.TransformerEncoderLayer(
            d_model=obs_feature_dim, nhead=fusion_nhead,
            dim_feedforward=obs_feature_dim * 4, batch_first=True, activation=F.gelu
        )

        summarizer_layer_m = nn.TransformerEncoderLayer(
            d_model=obs_feature_dim, nhead=fusion_nhead,
            dim_feedforward=obs_feature_dim * 4, batch_first=True, activation=F.gelu
        )

        summarizer_layer_d = nn.TransformerEncoderLayer(
            d_model=obs_feature_dim, nhead=fusion_nhead,
            dim_feedforward=obs_feature_dim * 4, batch_first=True, activation=F.gelu
        )

        self.summarizer_s = nn.TransformerEncoder(summarizer_layer_s, num_layers=2)
        self.summarizer_m = nn.TransformerEncoder(summarizer_layer_m, num_layers=2)
        self.summarizer_d = nn.TransformerEncoder(summarizer_layer_d, num_layers=2)

        self.global_cls_token = nn.Parameter(torch.randn(1, 1, obs_feature_dim))
        final_fusion_layer = nn.TransformerEncoderLayer(
            d_model=obs_feature_dim, nhead=fusion_nhead,
            dim_feedforward=obs_feature_dim * 4, batch_first=True, activation=F.gelu
        )

        self.final_summarizer = nn.TransformerEncoder(final_fusion_layer, num_layers=2)

        # --- 6. 动作解码器 ---
        self.action_decoder = DCFM_ActionDecoder(
            obs_feature_dim=obs_feature_dim,
            state_dim=state_dim,
            base_action_dim=mobility_action_dim,
            arm_action_dim=action_dim-mobility_action_dim,
            Tpred=Tpred
        )

    def _get_history_context(
        self,
        history_state: torch.Tensor,
        current_global_vision: torch.Tensor
    ) -> torch.Tensor:

        """
        历史意图编码器
        """
        B = history_state.shape[0]
        state_tokens = self.history_state_proj(history_state)
        vision_token = self.history_vision_proj(current_global_vision)
        cls_token = self.history_cls_token.expand(B, -1, -1)
        transformer_input = torch.cat([cls_token, vision_token, state_tokens], dim=1)
        transformer_input = transformer_input + self.history_pos_emb
        transformer_output = self.history_transformer(transformer_input)
        h_t = transformer_output[:, 0]
        return h_t

    def forward(self,
                pointcloud: ME.SparseTensor,
                imgs: torch.Tensor,
                states: torch.Tensor,
                actions = None
                ):
        """
        Args:
            pointcloud (ME.SparseTensor): 当前帧的稀疏体素。
            imgs (torch.Tensor): 当前帧的图像 (B, num_views, C, H, W)
            states (torch.Tensor): 历史帧的状态 (B, T_hist, state_dim)
            actions (torch.Tensor): (B, Tpred, action_dim) (用于训练)
        """
        B = imgs.shape[0] # 从 imgs 获取 B
        device = imgs.device

        # --- 1. 拆分历史和当前状态 ---
        current_state = states[:, -1]
        history_state = states[:, :self.T_hist]

        # --- 2. 编码当前视觉 (2D) ---
        current_imgs_flat = imgs.view(B * self.num_views, *imgs.shape[2:])
        img_feat_dict = self.img_encoder(current_imgs_flat)

        # --- 3. 编码“历史意图” ---
        global_feats_viewed = img_feat_dict['global'].view(B, self.num_views, self.obs_feature_dim)
        current_global_vision = global_feats_viewed.view(B, 1, self.num_views * self.obs_feature_dim)
        h_t = self._get_history_context(history_state, current_global_vision)

        # --- 4. 编码 3D Tokens ---
        pc_token_dict = self.pc_encoder(pointcloud, batch_size=B)
        fused_tokens_list = []
        fused_mask_list = []
        g_scot_loss_list = []
    
        # --- 5. 分层融合 ---
        for i, key in enumerate(self.scale_keys):
            # 1. 获取 3D token
            pc_tokens, pc_pos, pc_mask = pc_token_dict[key]
            # pc_tokens: (B, N_pc, D)
            # pc_pos: (B, N_pc, D)
            # pc_mask: (B, N_pc) - True 为 padding

            # 2. 准备 2D (K,V) 上下文
            # (B*V, N_patch, D) -> (B, V, N_patch, D)
            img_patches_kv = img_feat_dict[key].view(
                B,
                self.num_views,
                img_feat_dict[key].shape[1],
                self.obs_feature_dim
            )

            # 3. 单独融合各尺度特征
            if actions == None:
                is_training = False
            else:
                is_training = True
            darm_module = self.darms[i]
            fused_tokens, g_scot_loss = darm_module(
                pc_tokens=pc_tokens,
                pc_pos=pc_pos,
                img_patches_multiview=img_patches_kv,
                pc_padding_mask=pc_mask,
                is_training=is_training
            ) # 输出: (B, N_pc, D)

            # 4. 存储结果
            fused_tokens_list.append(fused_tokens)
            fused_mask_list.append(pc_mask)
            g_scot_loss_list.append(g_scot_loss)

        # --- 6. Fusion 压缩 and Film 调制 ---
        summary_vectors = []
        summarizers = [self.summarizer_s, self.summarizer_m, self.summarizer_d]
        cls_tokens = [self.cls_token_s, self.cls_token_m, self.cls_token_d]

        # 尺度内压缩
        for i in range(self.num_scales):
            tokens = fused_tokens_list[i]    # (B, N_i, D)
            mask = fused_mask_list[i]      # (B, N_i)
            # 准备 [CLS] Token
            cls_token_i = cls_tokens[i].expand(B, -1, -1) # (B, 1, D)
            cls_mask_i = torch.zeros(B, 1, dtype=torch.bool, device=device)
            # 准备输入
            summarizer_input_i = torch.cat([cls_token_i, tokens], dim=1) # (B, 1+N_i, D)
            summarizer_mask_i = torch.cat([cls_mask_i, mask], dim=1)   # (B, 1+N_i)
            # 运行压缩器
            summarizer_output_i = summarizers[i](
                summarizer_input_i,
                src_key_padding_mask=summarizer_mask_i
            )

            # FiLM 调制融合and压缩后的特征
            scale_context = self.scale_context_embeddings[i].expand(B, -1)
            gamma, beta = self.film_generator(h_t, scale_context)
            summary_vector = (gamma * summarizer_output_i[:, 0]) + beta

            # 存入列表
            summary_vectors.append(summary_vector.unsqueeze(1)) # (B, 1, D)

        # 尺度间压缩
        # summary_vectors 现在是 [ (B, 1, D)_s, (B, 1, D)_m, (B, 1, D)_d ]
        global_cls = self.global_cls_token.expand(B, -1, -1) # (B, 1, D)
        # (B, 4, D) = [Global_CLS, Summary_S, Summary_M, Summary_D]
        final_input = torch.cat([global_cls] + summary_vectors, dim=1)
        final_output = self.final_summarizer(final_input)
        # (B, D) -> 提取最终的 [Global_CLS] Token
        fused_vector = final_output[:, 0]

        # --- 7. 解码动作 ---
        if actions is not None:
            # 训练模式 需要计算损失
            loss = self.action_decoder(fused_vector, current_state, gt_future_actions=actions)
            for i in range(self.num_scales):
                loss = loss + 0.01 * g_scot_loss_list[i]

            return loss
        else:
            # 推理模式 只返回动作Tact长度的动作
            pred_actions = self.action_decoder(fused_vector, current_state)
            return pred_actions[:, :self.Tact, :]