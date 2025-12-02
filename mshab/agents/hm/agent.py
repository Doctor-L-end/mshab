from mshab.agents.hm.hm_algo import HMPolicy
import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        # 初始化参数
        self.state_dim = env.single_observation_space['state'].shape[1] # 状态维度42(包含所有额外信息的话)  # include 'qpos', 'qvel','tcp_pose_wrt_base', 'obj_pose_wrt_base', 'goal_pos_wrt_base', 'is_grasped'
        self.act_dim = env.single_action_space.shape[0] # 动作维度13

        self.policy = HMPolicy(
            Tpred = args.pred_horizon,
            Tact = args.act_horizon,
            T_hist = args.obs_horizon,
            pc_feats_input_dim = 3,
            obs_feature_dim = 512,
            action_dim = self.act_dim - 2, # 去掉头部的2维
            mobility_action_dim = 3,
            state_dim = self.state_dim,
            num_views = 2,
        )
        
    def compute_loss(self, pointcloud, imgs, states, actions):
        mobile_base_action = actions[..., -2:]  # 移动底座动作
        torso_action = actions[..., -3:-2]  # 躯干动作
        head_action = actions[..., -5:-3]  # 头部动作
        arms_action = actions[..., :-5]  # 手臂动作
        actions = torch.cat([mobile_base_action, torso_action, arms_action], dim=-1)  # 合并动作
        loss = self.policy(pointcloud, imgs, states, actions)
        return loss

    def get_action(self, pointcloud, imgs, states):
        actions = self.policy(pointcloud, imgs, states)
        mobile_base_action = actions[..., :2]
        torso_action = actions[..., 2:3]
        arms_action = actions[..., 3:]
        head_dim = 2
        batch_shape = actions.shape[:-1]
        head_action = torch.zeros(
            (*batch_shape, head_dim), 
            device=actions.device, 
            dtype=actions.dtype
        )
        full_action = torch.cat(
            [arms_action, head_action, torso_action, mobile_base_action], 
            dim=-1
        )
        return full_action
