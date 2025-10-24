from icpmm.icpmm_algo.learning.policy import ICPMMPolicy
import torch
import torch.nn as nn
import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from mshab.vis import normalize_point_cloud, robust_normalize_to_01
import torchvision.transforms as T

class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        # 验证观测和动作空间
        assert len(env.single_observation_space['state'].shape) == 2
        assert len(env.single_action_space.shape) == 1
        assert np.all(env.single_action_space.high == 1) and np.all(
            env.single_action_space.low == -1
        )

        self.normalize_rgbd = T.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])

        # 初始化参数
        self.state_dim = env.single_observation_space['state'].shape[1] # 状态维度42(包含所有额外信息的话)  # include 'qpos', 'qvel','tcp_pose_wrt_base', 'obj_pose_wrt_base', 'goal_pos_wrt_base', 'is_grasped'
        self.act_dim = env.single_action_space.shape[0] # 动作维度13

        self.policy = ICPMMPolicy(
            agent_state_dim=24, 
            agent_state_key="agent_state",
            prop_dim=self.state_dim,
            prop_key="state",
            prop_mlp_hidden_depth=2,
            prop_mlp_hidden_dim=256,
            pointnet_n_coordinates=3,
            pointnet_n_color=0,
            pointnet_hidden_depth=2,
            pointnet_hidden_dim=256,
            resnet_num_blocks=[2, 2, 2, 2],
            resnet_use_depth=True,
            resnet_cam_num=2,
            target_obj_pose_decoder_mlp_hidden_depth=2,
            target_obj_pose_decoder_mlp_hidden_dim=128,
            target_obj_pose_dim=7,
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
            learnable_intention_readout_token=True,
            learnable_target_obj_pose_readout_token=True,
            tf1_n_embd=256,
            tf1_n_layer=2,
            tf1_n_head=8,
            tf1_dropout_rate=0.1,
            tf1_use_geglu=True,
            tf2_n_embd=256,
            tf2_n_layer=2,
            tf2_n_head=8,
            tf2_dropout_rate=0.1,
            tf2_use_geglu=True,
            learnable_action_readout_token=True,
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

        self.obs_horizon = args.obs_horizon
        self.pred_horizon = args.pred_horizon
        
    def compute_loss(self, obs, action_seq):
        obs['pointcloud'] = normalize_point_cloud(obs['pointcloud'])
        img_seq = []
        #只要最新的一张图片
        obs['fetch_head_rgb'] = obs['fetch_head_rgb'][:, -1:, ...]
        obs['fetch_head_depth'] = obs['fetch_head_depth'][:, -1:, ...]
        obs['fetch_hand_rgb'] = obs['fetch_hand_rgb'][:, -1:, ...]
        obs['fetch_hand_depth'] = obs['fetch_hand_depth'][:, -1:, ...]
        img1 = torch.cat([robust_normalize_to_01(obs['fetch_head_rgb']), robust_normalize_to_01(obs['fetch_head_depth'])], dim=2)
        img1 = img1.view(-1, img1.shape[2], img1.shape[3], img1.shape[4])
        img1 = self.normalize_rgbd(img1)
        img1 = img1.view(-1, 1, img1.shape[1], img1.shape[2], img1.shape[3])
        img2 = torch.cat([robust_normalize_to_01(obs['fetch_hand_rgb']), robust_normalize_to_01(obs['fetch_hand_depth'])], dim=2)
        img2 = img2.view(-1, img2.shape[2], img2.shape[3], img2.shape[4])
        img2 = self.normalize_rgbd(img2)
        img2 = img2.view(-1, 1, img2.shape[1], img2.shape[2], img2.shape[3])       
        img_seq.append(img1)
        img_seq.append(img2)
        # 将所有图像数据沿通道维度拼接
        img_seq = torch.cat(
            img_seq, dim=2
        )  # 形状变为(B, 1, C1+C2, H, W)
        obs['images'] = img_seq
        obs.pop('fetch_head_rgb')
        obs.pop('fetch_head_depth')
        obs.pop('fetch_hand_rgb')
        obs.pop('fetch_hand_depth')
        # 处理状态
        obs["agent_state"] = obs["state"][..., :24]
        obs["state"] = obs["state"][:, -1:, :]
        loss = self.policy.compute_loss(obs=obs, gt_action=action_seq.unsqueeze(1))
        return loss

    def get_action(self, obs):
        obs['pointcloud'] = normalize_point_cloud(obs['pointcloud'])
        img_seq = []
        #只要最新的一张图片
        obs['fetch_head_rgb'] = obs['fetch_head_rgb'][:, -1:, ...]
        obs['fetch_head_depth'] = obs['fetch_head_depth'][:, -1:, ...]
        obs['fetch_hand_rgb'] = obs['fetch_hand_rgb'][:, -1:, ...]
        obs['fetch_hand_depth'] = obs['fetch_hand_depth'][:, -1:, ...]
        img1 = torch.cat([robust_normalize_to_01(obs['fetch_head_rgb']), robust_normalize_to_01(obs['fetch_head_depth'])], dim=2)
        img1 = img1.view(-1, img1.shape[2], img1.shape[3], img1.shape[4])
        img1 = self.normalize_rgbd(img1)
        img1 = img1.view(-1, 1, img1.shape[1], img1.shape[2], img1.shape[3])
        img2 = torch.cat([robust_normalize_to_01(obs['fetch_hand_rgb']), robust_normalize_to_01(obs['fetch_hand_depth'])], dim=2)
        img2 = img2.view(-1, img2.shape[2], img2.shape[3], img2.shape[4])
        img2 = self.normalize_rgbd(img2)
        img2 = img2.view(-1, 1, img2.shape[1], img2.shape[2], img2.shape[3])       
        img_seq.append(img1)
        img_seq.append(img2)
        # 将所有图像数据沿通道维度拼接
        img_seq = torch.cat(
            img_seq, dim=2
        )  # 形状变为(B, 1, C1+C2, H, W)
        obs['images'] = img_seq
        obs.pop('fetch_head_rgb')
        obs.pop('fetch_head_depth')
        obs.pop('fetch_hand_rgb')
        obs.pop('fetch_hand_depth')
        # 处理状态
        obs["agent_state"] = obs["state"][..., :24]
        obs["state"] = obs["state"][:, -1:, :]
        raw_action = self.policy.act(obs)
        action = torch.cat([raw_action["arm"], raw_action["head"], raw_action["torso"], raw_action["mobile_base"]], dim=-1)
        return action
