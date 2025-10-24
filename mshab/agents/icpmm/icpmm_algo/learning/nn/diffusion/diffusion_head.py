from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import SchedulerMixin
from einops import rearrange
from .unet import ConditionalUnet1D

class NoMergeMyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 分别定义Q、K、V的投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim*num_heads)
        self.k_proj = nn.Linear(embed_dim, embed_dim*num_heads)
        self.v_proj = nn.Linear(embed_dim, embed_dim*num_heads)
        
    def forward(self, query, key, value):
        # 输入形状: (batch_size, seq_len, embed_dim)
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 投影并分割成多个头
        q = self.q_proj(query)  # (batch_size, seq_len, embed_dim*num_heads)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为 (batch_size, seq_len, num_heads, embed_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.embed_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.embed_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.embed_dim)
        
        # 转置以获取 (batch_size, num_heads, seq_len, embed_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, embed_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 保存每个头的Q、K、V
        self.q_heads = q  # (batch_size, num_heads, seq_len, embed_dim)
        self.k_heads = k
        self.v_heads = v
        
        # 计算注意力分数
        # (batch_size, num_heads, seq_len, embed_dim) @ (batch_size, num_heads, embed_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 应用注意力权重到V
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, embed_dim)
        # -> (batch_size, num_heads, seq_len, embed_dim)
        head_outputs = torch.matmul(attn_weights, v)
        
        return head_outputs

class WholeBodyUNetDiffusionHead(nn.Module):
    def __init__(
        self,
        *,
        # ====== whole body ======
        decoder_part_keys: list[str],
        action_dim_per_part: dict[str, int],
        # ====== model ======
        intent_dim: int,
        action_feature_dim: int,
        action_horizon: int,
        diffusion_step_embed_dim: int,
        unet_down_dims: List[int],
        unet_kernel_size: int,
        unet_n_groups: int,
        unet_cond_predict_scale: bool,
        # ====== noise scheduler ======
        noise_scheduler: SchedulerMixin,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        # ====== inference ======
        inference_denoise_steps: int,
    ):
        super().__init__()
        assert set(decoder_part_keys) == set(action_dim_per_part.keys())

        self.models = nn.ModuleDict()

        self.num_parts = len(decoder_part_keys)
        self.mh_attention = NoMergeMyMultiheadAttention(embed_dim=intent_dim, num_heads=self.num_parts)
        
        for i, part in enumerate(decoder_part_keys):    
            total_global_cond_dim = action_feature_dim + intent_dim
            
            model = ConditionalUnet1D(
                action_dim_per_part[part],
                local_cond_dim=None,
                global_cond_dim=total_global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=unet_down_dims,
                kernel_size=unet_kernel_size,
                n_groups=unet_n_groups,
                cond_predict_scale=unet_cond_predict_scale,
            )
            self.models[part] = model
            
        self.decoder_part_keys = decoder_part_keys
        self.action_dim_per_part = action_dim_per_part
        self.action_dim = sum(action_dim_per_part.values())
        self.action_horizon = action_horizon
        self.intent_dim = intent_dim
        self.action_feature_dim = action_feature_dim
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_step_kwargs = noise_scheduler_step_kwargs or {}
        self.inference_denoise_steps = inference_denoise_steps

    def forward(
        self,
        intent_feature: torch.Tensor,
        action_feature: torch.Tensor,
        *,
        dependent_action_input: dict[str, torch.Tensor],
        noisy_action: dict[str, torch.Tensor],
        diffusion_timestep: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Run one pass to predict noise.

        Args:
            intent_feature: Intent features of size (B, 1, D_intent).
            action_feature: Action features of size (B, 1, D_action).
            dependent_action_input: Dict of dependent action inputs, each of size (B, 1, T_act, A).
            noisy_action: Dict of noisy actions, each of size (B, 1, T_act, A).
            diffusion_timestep: (B, 1, 1), timestep for diffusion process. Can be the same for all parts,
             or different for each part (provide as dict).

        Return:
            Dict of predicted noise, each of size (B, 1, T_act, A), corresponding to each part.
        """
        assert (
            intent_feature.ndim == 3
        ), f"intent_feature should have 3 dimensions (B, 1, D_intent), got {intent_feature.ndim}."
        assert (
            action_feature.ndim == 3
        ), f"action_feature should have 3 dimensions (B, 1, D_action), got {action_feature.ndim}."
        B = intent_feature.shape[0]
        
        assert set(dependent_action_input.keys()) == set(
            self.decoder_part_keys[:-1]
        )
        for dependent_part in self.decoder_part_keys[:-1]:
            assert dependent_action_input[dependent_part].shape == (B, 1,
                self.action_horizon,
                self.action_dim_per_part[dependent_part],
            )
        assert set(noisy_action.keys()) == set(self.decoder_part_keys)
        for part in self.decoder_part_keys:
            assert noisy_action[part].shape == (B, 1,
                self.action_horizon,
                self.action_dim_per_part[part],
            )
        if not isinstance(diffusion_timestep, dict):
            diffusion_timestep = {
                part: diffusion_timestep for part in self.decoder_part_keys
            }

        implicit_hierarchical_features = self.mh_attention(intent_feature, intent_feature, intent_feature) # (batch_size, num_heads, seq_len, embed_dim)

        pred_eps_all_parts = {}
        for part_idx, part_name in enumerate(self.decoder_part_keys):
            global_cond = []
            global_cond.append(implicit_hierarchical_features[:, part_idx, ...])
            global_cond.append(action_feature)
            global_cond = torch.cat(global_cond, dim=-1)
                
            denoise_in = rearrange(
                noisy_action[part_name], "B 1 T_act A -> (B 1) T_act A"
            )
            global_cond = rearrange(global_cond, "B 1 D -> (B 1) D")
            
            pred_eps = self.models[part_name](
                sample=denoise_in,
                timestep=rearrange(
                    diffusion_timestep[part_name], "B 1 1 -> (B 1)"
                ),
                global_cond=global_cond,
            )  # (B * 1, T_act, A)
            pred_eps = rearrange(
                pred_eps, "(B 1) T_act A -> B 1 T_act A", B=B
            )
            pred_eps_all_parts[part_name] = pred_eps
            
        return pred_eps_all_parts

    def compute_loss(
        self,
        intent_feature: torch.Tensor,
        action_feature: torch.Tensor,
        *,
        gt_action: dict[str, torch.Tensor],
    ):
        """
        Run one pass to predict noise and compute loss.

        Args:
            intent_feature: Intent features of size (B, 1, D_intent).
            action_feature: Action features of size (B, 1, D_action).
            gt_action: dict of ground truth action of size (B, 1, T_act, A), where T_act = action prediction horizon.
        """
        assert set(gt_action.keys()) == set(self.decoder_part_keys)
        B = intent_feature.shape[0]

        noises, noisy_actions, diffusion_timesteps = {}, {}, {}
        for part in self.decoder_part_keys:
            # flatten first two dim of gt_action
            gt_action_this_part = rearrange(
                gt_action[part], "B 1 T_act A -> (B 1) T_act A"
            )
            # sample noise
            noise_this_part = torch.randn(
                gt_action_this_part.shape, device=gt_action_this_part.device
            )  # (B * 1, T_act, A)
            # sample diffusion timesteps
            timesteps_this_part = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B * 1,),
                device=gt_action_this_part.device,
            )
            noisy_trajs_this_part = self.noise_scheduler.add_noise(
                gt_action_this_part, noise_this_part, timesteps_this_part
            )  # (B * 1, T_act, A)
            noisy_trajs_this_part = rearrange(
                noisy_trajs_this_part, "(B 1) T_act A -> B 1 T_act A", B=B
            )
            noise_this_part = rearrange(
                noise_this_part, "(B 1) T_act A -> B 1 T_act A", B=B
            )
            timesteps_this_part = rearrange(
                timesteps_this_part, "(B 1) -> B 1", B=B
            ).unsqueeze(
                -1
            )  # (B, 1, 1)

            noises[part] = noise_this_part
            noisy_actions[part] = noisy_trajs_this_part
            diffusion_timesteps[part] = timesteps_this_part

        # for dependent action inputs, we use gt actions
        dependent_action_input = {
            part: gt_action[part] for part in self.decoder_part_keys[:-1]
        }

        pred_eps = self.forward(
            intent_feature=intent_feature,
            action_feature=action_feature,
            dependent_action_input=dependent_action_input,
            noisy_action=noisy_actions,
            diffusion_timestep=diffusion_timesteps,
        )  # dict of (B, 1, T_act, A)
        # concat all parts
        pred_eps = torch.cat(
            [pred_eps[part] for part in self.decoder_part_keys], dim=-1
        )
        noise = torch.cat(
            [noises[part] for part in self.decoder_part_keys], dim=-1
        )
        mse_loss = F.mse_loss(pred_eps, noise, reduction="none")  # (B, 1, T_act, A)
        # sum over action dim instead of avg
        mse_loss = mse_loss.sum(dim=-1)  # (B, 1, T_act)
        return mse_loss

    @torch.no_grad()
    def inference(
        self,
        intent_feature: torch.Tensor,
        action_feature: torch.Tensor,
    ):
        """
        Run inference to predict future actions.

        Args:
            intent_feature: Intent features of size (B, 1, D_intent).
            action_feature: Action features of size (B, 1, D_action).
        """
        B = intent_feature.shape[0]

        if self.noise_scheduler.num_inference_steps != self.inference_denoise_steps:
            self.noise_scheduler.set_timesteps(self.inference_denoise_steps)

        implicit_hierarchical_features = self.mh_attention(intent_feature, intent_feature, intent_feature) # (batch_size, num_heads, seq_len, embed_dim)

        pred_action_all_parts = {}
        for part_idx, part in enumerate(self.decoder_part_keys):
            noisy_traj = torch.randn(
                size=(B, 1, self.action_horizon, self.action_dim_per_part[part]),
                device=intent_feature.device,
                dtype=intent_feature.dtype,
            )
            
            global_cond = []
            global_cond.append(implicit_hierarchical_features[:, part_idx, ...])
            global_cond.append(action_feature)
            global_cond = torch.cat(global_cond, dim=-1)
            global_cond = rearrange(global_cond, "B 1 D -> (B 1) D")
            
            for t in self.noise_scheduler.timesteps:
                timesteps = (
                    torch.ones((B, 1, 1), device=intent_feature.device, dtype=intent_feature.dtype) * t
                )
                
                denoise_in = rearrange(
                    noisy_traj, "B 1 T_act A -> (B 1) T_act A"
                )
                
                
                pred = self.models[part](
                    sample=denoise_in,
                    timestep=rearrange(timesteps, "B 1 1 -> (B 1)"),
                    global_cond=global_cond,
                )  # (B * 1, T_act, A)
                noisy_traj = rearrange(
                    noisy_traj, "B 1 T_act A -> (B 1) T_act A"
                )
                noisy_traj = self.noise_scheduler.step(
                    pred, t, noisy_traj, **self.noise_scheduler_step_kwargs
                ).prev_sample  # (B * 1, T_act, A)
                noisy_traj = rearrange(
                    noisy_traj, "(B 1) T_act A -> B 1 T_act A", B=B
                )
            pred_action_all_parts[part] = noisy_traj
            
        # Return all parts as (B, T_act, A) for each part
        return {k: v[:, 0] for k, v in pred_action_all_parts.items()}