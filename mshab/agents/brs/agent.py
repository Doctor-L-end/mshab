from mshab.agents.brs.brs_algo.learning.policy import WBVIMAPolicy
import torch
import torch.nn as nn
import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from mshab.vis import normalize_point_cloud

class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        # 验证观测和动作空间
        assert len(env.single_observation_space['state'].shape) == 2
        assert len(env.single_action_space.shape) == 1
        assert np.all(env.single_action_space.high == 1) and np.all(
            env.single_action_space.low == -1
        )

        # 初始化参数
        self.state_dim = env.single_observation_space['state'].shape[1] # 状态维度42(包含所有额外信息的话)
        self.act_dim = env.single_action_space.shape[0] # 动作维度13

        self.policy = WBVIMAPolicy(
            prop_dim=self.state_dim,
            prop_keys=[
                "state", # include 'qpos', 'qvel','tcp_pose_wrt_base', 'obj_pose_wrt_base', 'goal_pos_wrt_base', 'is_grasped'
            ],
            prop_mlp_hidden_depth=2,
            prop_mlp_hidden_dim=256,
            pointnet_n_coordinates=3,
            pointnet_n_color=3,
            pointnet_hidden_depth=2,
            pointnet_hidden_dim=256,
            action_keys=[
                "mobile_base",
                "torso",
                "head",
                "arm",
            ],
            action_key_dims={
                "mobile_base": 2,
                "torso": 1,
                "head": 2,  
                "arm": 8,
            },
            num_latest_obs=args.obs_horizon,
            use_modality_type_tokens=False,
            xf_n_embd=256,
            xf_n_layer=2,
            xf_n_head=8,
            xf_dropout_rate=0.1,
            xf_use_geglu=True,
            learnable_action_readout_token=False,
            action_dim=self.act_dim,
            action_prediction_horizon=args.pred_horizon,
            diffusion_step_embed_dim=128,
            unet_down_dims=[64, 128],
            unet_kernel_size=5,
            unet_n_groups=8,
            unet_cond_predict_scale=True,
            noise_scheduler=DDIMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            ),
            noise_scheduler_step_kwargs=None,
            num_denoise_steps_per_inference=16,
        )
        
    def compute_loss(self, obs, action_seq):
        obs['pointcloud'] = normalize_point_cloud(obs['pointcloud'])
        loss = self.policy.compute_loss(obs=obs, gt_action=action_seq)
        return loss

    def get_action(self, obs):
        raw_action = self.policy.act(obs)
        action = torch.cat([raw_action["arms"], raw_action["head"], raw_action["torso"], raw_action["mobile_base"]], dim=-1)
        return action
