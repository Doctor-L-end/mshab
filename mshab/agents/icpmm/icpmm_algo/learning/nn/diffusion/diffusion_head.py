from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import SchedulerMixin
from einops import rearrange
from .unet import ConditionalUnet1D


class DiffusionHead(nn.Module):
    """
    Action head that generates actions through diffusion denosing.
    """

    """Backbone model of diffusion head, e.g., an MLP, a UNet, etc."""
    model: nn.Module
    """Action dimension."""
    action_dim: int
    """Action horizon."""
    action_horizon: int
    """Noise scheduler used in diffusion process."""
    noise_scheduler: SchedulerMixin
    """kwargs passed to noise scheduler's step method."""
    noise_scheduler_step_kwargs: Optional[dict] = None
    """Number of denoising steps during inference."""
    inference_denoise_steps: int

    def forward(
        self,
        intent_feature: torch.Tensor,
        action_feature: torch.Tensor,
        *,
        additional_input: Optional[torch.Tensor] = None,
        noisy_action: torch.Tensor,
        diffusion_timestep: torch.Tensor,
    ):
        """
        Run one pass to predict noise.

        Args:
            intent_feature: Intent features of size (B, 1, D_intent).
            action_feature: Action features of size (B, 1, D_action).
            additional_input: Additional input features of size (B, 1, D_additional).
            noisy_action: Noisy action of size (B, 1, T_act, A), where T_act = action prediction horizon.
            diffusion_timestep: (B, 1, 1), timestep for diffusion process.

        Return:
            Predicted noise of size (B, 1, T_act, A).
        """
        assert (
            intent_feature.ndim == 3
        ), f"intent_feature should have 3 dimensions (B, 1, D_intent), got {intent_feature.ndim}."
        assert (
            action_feature.ndim == 3
        ), f"action_feature should have 3 dimensions (B, 1, D_action), got {action_feature.ndim}."
        
        # Combine intent and action features
        cond_features = torch.cat([intent_feature, action_feature], dim=-1)
        
        if additional_input is not None:
            assert (
                additional_input.ndim == 3
            ), f"additional_input should have 3 dimensions (B, 1, D_additional), got {additional_input.ndim}."
            assert (
                additional_input.shape[:2] == intent_feature.shape[:2]
            ), f"additional_input and intent_feature should have the same batch size and time dimension."
            cond_features = torch.cat([cond_features, additional_input], dim=-1)
            
        assert (
            noisy_action.ndim == 4
        ), f"noisy_action should have 4 dimensions (B, 1, T_act, A), got {noisy_action.ndim}."
        assert (
            noisy_action.shape[:2] == intent_feature.shape[:2]
        ), f"noisy_action and intent_feature should have the same batch size and time dimension."
        
        flattened_noisy_action = rearrange(
            noisy_action, "B 1 T_act A -> B 1 (T_act A)"
        )
        denoise_in = torch.cat([cond_features, flattened_noisy_action], dim=-1)
        pred_eps = self.model(x=denoise_in, diffusion_t=diffusion_timestep)
        pred_eps = rearrange(
            pred_eps, "B 1 (T_act A) -> B 1 T_act A", T_act=self.action_horizon
        )
        return pred_eps

    def compute_loss(
        self,
        intent_feature: torch.Tensor,
        action_feature: torch.Tensor,
        *,
        additional_input: Optional[torch.Tensor] = None,
        gt_action: torch.Tensor,
    ):
        """
        Run one pass to predict noise and compute loss.

        Args:
            intent_feature: Intent features of size (B, 1, D_intent).
            action_feature: Action features of size (B, 1, D_action).
            additional_input: Additional input features of size (B, 1, D_additional).
            gt_action: Ground truth action of size (B, 1, T_act, A), where T_act = action prediction horizon.
        """
        B = intent_feature.shape[0]
        # flatten first two dim of gt_action
        gt_action = rearrange(gt_action, "B 1 T_act A -> (B 1) T_act A")
        # sample noise
        noise = torch.randn(
            gt_action.shape, device=gt_action.device
        )  # (B * 1, T_act, A)
        # sample diffusion timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B * 1,),
            device=gt_action.device,
        )
        noisy_trajs = self.noise_scheduler.add_noise(
            gt_action, noise, timesteps
        )  # (B * 1, T_act, A)
        noisy_trajs = rearrange(
            noisy_trajs, "(B 1) T_act A -> B 1 T_act A", B=B
        )
        noise = rearrange(noise, "(B 1) T_act A -> B 1 T_act A", B=B)
        timesteps = rearrange(timesteps, "(B 1) -> B 1", B=B).unsqueeze(
            -1
        )  # (B, 1, 1)
        pred_eps = self.forward(
            intent_feature,
            action_feature,
            additional_input=additional_input,
            noisy_action=noisy_trajs,
            diffusion_timestep=timesteps,
        )  # (B, 1, T_act, A)
        mse_loss = F.mse_loss(pred_eps, noise, reduction="none")  # (B, 1, T_act, A)
        # sum over action dim instead of avg
        mse_loss = mse_loss.sum(dim=-1)  # (B, 1, T_act)
        return mse_loss

    @torch.no_grad()
    def inference(
        self,
        intent_feature: torch.Tensor,
        action_feature: torch.Tensor,
        *,
        additional_input: Optional[torch.Tensor] = None,
    ):
        """
        Run inference to predict future actions.

        Args:
            intent_feature: Intent features of size (B, 1, D_intent).
            action_feature: Action features of size (B, 1, D_action).
            additional_input: Additional input features of size (B, 1, D_additional).
        """
        B = intent_feature.shape[0]
        noisy_traj = torch.randn(
            size=(B, 1, self.action_horizon, self.action_dim),
            device=intent_feature.device,
            dtype=intent_feature.dtype,
        )
        if self.noise_scheduler.num_inference_steps != self.inference_denoise_steps:
            self.noise_scheduler.set_timesteps(self.inference_denoise_steps)

        for t in self.noise_scheduler.timesteps:
            timesteps = (
                torch.ones((B, 1, 1), device=intent_feature.device, dtype=intent_feature.dtype) * t
            )
            pred = self.forward(
                intent_feature,
                action_feature,
                additional_input=additional_input,
                noisy_action=noisy_traj,
                diffusion_timestep=timesteps,
            )  # (B, 1, T_act, A)
            # denosing
            pred = rearrange(pred, "B 1 T_act A -> (B 1) T_act A")
            noisy_traj = rearrange(noisy_traj, "B 1 T_act A -> (B 1) T_act A")
            noisy_traj = self.noise_scheduler.step(
                pred, t, noisy_traj, **self.noise_scheduler_step_kwargs
            ).prev_sample  # (B * 1, T_act, A)
            noisy_traj = rearrange(
                noisy_traj, "(B 1) T_act A -> B 1 T_act A", B=B
            )
        return noisy_traj[:, 0]  # Return (B, T_act, A)


class UNetDiffusionHead(DiffusionHead):
    def __init__(
        self,
        *,
        # ====== model ======
        intent_dim: int,
        action_feature_dim: int,
        action_dim: int,
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
        # Calculate total condition dimension
        total_cond_dim = intent_dim + action_feature_dim
        
        self.model = ConditionalUnet1D(
            action_dim,
            local_cond_dim=None,
            global_cond_dim=total_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=unet_n_groups,
            cond_predict_scale=unet_cond_predict_scale,
        )
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_step_kwargs = noise_scheduler_step_kwargs or {}
        self.inference_denoise_steps = inference_denoise_steps

    def forward(
        self,
        intent_feature: torch.Tensor,
        action_feature: torch.Tensor,
        *,
        additional_input: Optional[torch.Tensor] = None,
        noisy_action: torch.Tensor,
        diffusion_timestep: torch.Tensor,
    ):
        B = intent_feature.shape[0]
        # Combine intent and action features
        cond_features = torch.cat([intent_feature, action_feature], dim=-1)
        
        if additional_input is not None:
            cond_features = torch.cat([cond_features, additional_input], dim=-1)
            
        cond_features = rearrange(cond_features, "B 1 D -> (B 1) D")
        noisy_action = rearrange(noisy_action, "B 1 T_act A -> (B 1) T_act A")
        diffusion_timestep = rearrange(diffusion_timestep, "B 1 1 -> (B 1)")
        pred = self.model(
            sample=noisy_action,
            timestep=diffusion_timestep,
            global_cond=cond_features,
        )  # (B * 1, T_act, A)
        pred = rearrange(pred, "(B 1) T_act A -> B 1 T_act A", B=B)
        return pred


class WholeBodyUNetDiffusionHead(nn.Module):
    def __init__(
        self,
        *,
        # ====== whole body ======
        whole_body_decoding_order: list[str],
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
        assert set(whole_body_decoding_order) == set(action_dim_per_part.keys())

        self.models = nn.ModuleDict()
        self.part_cond_types = {}  # 记录每个部分接受的条件类型
        
        for i, part in enumerate(whole_body_decoding_order):
            # 确定当前部分接受的条件类型
            if part in ['mobile_base', 'torso', 'head']:  # 底盘、腰部和头部接受意图和动作特征
                cond_types = ['intent', 'action_feature']
            else:  # 其他部分（如手臂）只接受动作特征
                cond_types = ['action_feature']
            self.part_cond_types[part] = cond_types
            
            # 计算基础条件维度
            base_cond_dim = 0
            if 'intent' in cond_types:
                base_cond_dim += intent_dim
            if 'action_feature' in cond_types:
                base_cond_dim += action_feature_dim
                
            # 添加之前解码部分的动作维度
            additional_input_dim = 0
            for j in range(i):
                dependent_part = whole_body_decoding_order[j]
                additional_input_dim += (
                    action_dim_per_part[dependent_part] * action_horizon
                )
                
            total_global_cond_dim = base_cond_dim + additional_input_dim
            
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
            
        self.whole_body_decoding_order = whole_body_decoding_order
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
            self.whole_body_decoding_order[:-1]
        )
        for dependent_part in self.whole_body_decoding_order[:-1]:
            assert dependent_action_input[dependent_part].shape == (B, 1,
                self.action_horizon,
                self.action_dim_per_part[dependent_part],
            )
        assert set(noisy_action.keys()) == set(self.whole_body_decoding_order)
        for part in self.whole_body_decoding_order:
            assert noisy_action[part].shape == (B, 1,
                self.action_horizon,
                self.action_dim_per_part[part],
            )
        if not isinstance(diffusion_timestep, dict):
            diffusion_timestep = {
                part: diffusion_timestep for part in self.whole_body_decoding_order
            }

        pred_eps_all_parts = {}
        for part_idx, part_name in enumerate(self.whole_body_decoding_order):
            # 构建基础条件（意图和/或动作特征）
            base_cond = []
            cond_types = self.part_cond_types[part_name]
            
            if 'intent' in cond_types:
                base_cond.append(intent_feature)
            if 'action_feature' in cond_types:
                base_cond.append(action_feature)
                
            if base_cond:
                base_cond = torch.cat(base_cond, dim=-1)  # (B, 1, D_base)
            else:
                base_cond = torch.zeros(B, 1, 0, device=intent_feature.device, dtype=intent_feature.dtype)
            
            # 添加依赖的动作输入
            all_dependent_action = None
            if part_idx > 0:
                all_dependent_action = []
                for j in range(part_idx):
                    dependent_action = dependent_action_input[
                        self.whole_body_decoding_order[j]
                    ]
                    dependent_action = rearrange(
                        dependent_action, "B 1 T_act A -> B 1 (T_act A)"
                    )
                    all_dependent_action.append(dependent_action)
                all_dependent_action = torch.cat(
                    all_dependent_action, dim=-1
                )  # (B, 1, D_dependent)
                
                # 合并基础条件和依赖动作
                if base_cond.shape[-1] > 0:
                    global_cond = torch.cat([base_cond, all_dependent_action], dim=-1)
                else:
                    global_cond = all_dependent_action
            else:
                global_cond = base_cond
                
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
        assert set(gt_action.keys()) == set(self.whole_body_decoding_order)
        B = intent_feature.shape[0]

        noises, noisy_actions, diffusion_timesteps = {}, {}, {}
        for part in self.whole_body_decoding_order:
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
            part: gt_action[part] for part in self.whole_body_decoding_order[:-1]
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
            [pred_eps[part] for part in self.whole_body_decoding_order], dim=-1
        )
        noise = torch.cat(
            [noises[part] for part in self.whole_body_decoding_order], dim=-1
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

        pred_action_all_parts = {}
        for part_idx, part in enumerate(self.whole_body_decoding_order):
            noisy_traj = torch.randn(
                size=(B, 1, self.action_horizon, self.action_dim_per_part[part]),
                device=intent_feature.device,
                dtype=intent_feature.dtype,
            )
            
            # 构建基础条件（意图和/或动作特征）
            base_cond = []
            cond_types = self.part_cond_types[part]
            
            if 'intent' in cond_types:
                base_cond.append(intent_feature)
            if 'action_feature' in cond_types:
                base_cond.append(action_feature)
                
            if base_cond:
                base_cond = torch.cat(base_cond, dim=-1)  # (B, 1, D_base)
            else:
                base_cond = torch.zeros(B, 1, 0, device=intent_feature.device, dtype=intent_feature.dtype)
            
            for t in self.noise_scheduler.timesteps:
                timesteps = (
                    torch.ones((B, 1, 1), device=intent_feature.device, dtype=intent_feature.dtype) * t
                )
                
                # 添加依赖的动作输入
                all_dependent_action = None
                if part_idx > 0:
                    all_dependent_action = []
                    for j in range(part_idx):
                        dependent_action = pred_action_all_parts[
                            self.whole_body_decoding_order[j]
                        ]
                        dependent_action = rearrange(
                            dependent_action, "B 1 T_act A -> B 1 (T_act A)"
                        )
                        all_dependent_action.append(dependent_action)
                    all_dependent_action = torch.cat(
                        all_dependent_action, dim=-1
                    )  # (B, 1, D_dependent)
                    
                    # 合并基础条件和依赖动作
                    if base_cond.shape[-1] > 0:
                        global_cond = torch.cat([base_cond, all_dependent_action], dim=-1)
                    else:
                        global_cond = all_dependent_action
                else:
                    global_cond = base_cond
                    
                denoise_in = rearrange(
                    noisy_traj, "B 1 T_act A -> (B 1) T_act A"
                )
                global_cond = rearrange(global_cond, "B 1 D -> (B 1) D")
                
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