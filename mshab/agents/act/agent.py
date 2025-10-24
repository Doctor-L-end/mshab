import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from mshab.agents.act.detr.backbone import build_backbone  # DETR模型骨干网络
from mshab.agents.act.detr.transformer import build_transformer  # Transformer构建
from mshab.agents.act.detr.detr_vae import build_encoder, DETRVAE  # DETR-VAE模型

from mshab.vis import robust_normalize_to_01 # 用于图像归一化

class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        # assert args.include_depth == True, "当前仅支持包含深度图的观测空间"
        # assert args.obs_horizon == 1, "当前仅支持观测历史长度为1"
        
        # 从观测空间中提取所有图像类型的键（排除"state"）
        self.image_keys = [k for k in env.single_observation_space.keys() if k != "state"]
        # 从图像键中筛选出深度图相关的键（名称包含"depth"）
        self.depth_keys = [k for k in self.image_keys if "depth" in k]

        # 验证观测和动作空间
        assert len(env.single_observation_space['state'].shape) == 2
        assert len(env.single_action_space.shape) == 1
        assert np.all(env.single_action_space.high == 1) and np.all(
            env.single_action_space.low == -1
        )

        # 初始化参数
        self.state_dim = env.single_observation_space['state'].shape[1]
        self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.normalize_rgb = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize_rgbd = T.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        self.normalize_d = T.Normalize(mean=[0.5,0.5,0.5,0.5], std=[0.5,0.5,0.5,0.5])
        self.include_depth = args.include_depth
        self.include_rgb = args.include_rgb

        # 构建模型组件
        backbones = [build_backbone(args)]
        transformer = build_transformer(args)
        encoder = build_encoder(args)
        
        # 构建完整模型
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim*args.obs_horizon,
            action_dim=self.act_dim,
            # action_dims={
            #     "mobile_base": 2,
            #     "torso": 1,
            #     "head": 2,  
            #     "arm": 8,
            # },
            num_queries=args.num_queries,
        )

    def compute_loss(self, obs, action_seq):
        # print(obs["fetch_head_depth"]) # 出现了大于1000的值
        # print(obs["fetch_head_depth"].dtype) # torch.float32
        # print(obs["fetch_head_rgb"]) # 0-255
        # print(obs["fetch_head_rgb"].dtype) # torch.float32
       
        # # 处理深度图：使用tanh变换进行归一化
        # for dk in self.depth_keys:
        #     # 将深度图转换为浮点数并应用变换
        #     # obs[dk] = 1 - torch.tanh(obs[dk].float() / 1000)
        #     obs[dk] = obs[dk].float() / 1024.0
        
        img_seq = []
        # if self.include_rgb == False:
        #     img_seq.append(obs['fetch_head_depth'].repeat(1, 1, 4, 1, 1))
        #     img_seq.append(obs['fetch_hand_depth'].repeat(1, 1, 4, 1, 1))
        # elif self.include_depth:
        #     img_seq.append(torch.cat([self.normalize(obs['fetch_head_rgb'].float()/255.0), obs['fetch_head_depth']], dim=2))
        #     img_seq.append(torch.cat([self.normalize(obs['fetch_hand_rgb'].float()/255.0), obs['fetch_hand_depth']], dim=2))
        # else:
        #     img_seq.append(self.normalize(obs['fetch_head_rgb'].float()/255.0))
        #     img_seq.append(self.normalize(obs['fetch_hand_rgb'].float()/255.0))
        if self.include_rgb == False:
            img_seq.append(robust_normalize_to_01(obs['fetch_head_depth']).repeat(1, 1, 4, 1, 1))
            img_seq.append(robust_normalize_to_01(obs['fetch_hand_depth']).repeat(1, 1, 4, 1, 1))
        elif self.include_depth:
            img_seq.append(torch.cat([robust_normalize_to_01(obs['fetch_head_rgb']), robust_normalize_to_01(obs['fetch_head_depth'])], dim=2))
            img_seq.append(torch.cat([robust_normalize_to_01(obs['fetch_hand_rgb']), robust_normalize_to_01(obs['fetch_hand_depth'])], dim=2))
        else:
            img_seq.append(robust_normalize_to_01(obs['fetch_head_rgb']))
            img_seq.append(robust_normalize_to_01(obs['fetch_hand_rgb']))

        # 将所有图像数据拼接
        img_seq_ = torch.cat(
            img_seq, dim=1
        )  # 形状变为(B, obs_horizon * cam_num, C, H, W) obs_horizon一般指定为1

        # 标准化图像数据为[-1, 1]范围
        B, N, C, H, W = img_seq_.shape
        img_seq_reshaped = img_seq_.view(-1, C, H, W)
        if self.include_rgb == False:
            img_seq_normalized = self.normalize_d(img_seq_reshaped)
        elif self.include_depth:
            img_seq_normalized = self.normalize_rgbd(img_seq_reshaped)
        else:
            img_seq_normalized = self.normalize_rgb(img_seq_reshaped)
        img_seq_ = img_seq_normalized.view(B, N, C, H, W)

        obs_ = {}
        obs_["images"] = img_seq_
        # obs_['state'] = obs['state'].squeeze(1)  # (B, 1, state_dim) -> (B, state_dim)
        obs_['state'] = obs['state'].reshape(obs['state'].shape[0], -1)  # (B, T, state_dim) -> (B, T*state_dim)
        
        # # 分割真实动作到不同部件
        # mobile_base_action = action_seq[..., -2:]  # 移动底座动作
        # torso_action = action_seq[..., -3:-2]  # 躯干动作
        # head_action = action_seq[..., -5:-3]  # 头部动作
        # arm_action = action_seq[..., :-5]  # 手臂动作
        # action_seq = torch.cat([mobile_base_action, torso_action, head_action, arm_action], dim=-1)

        # 前向传播
        a_hat, (mu, logvar) = self.model(obs_, action_seq)
        # pred_action_seq = torch.cat([a_hat["mobile_base"], a_hat["torso"], a_hat["head"], a_hat["arm"]], dim=-1)

        # 计算KL散度
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        
        # print(action_seq.shape, a_hat.shape)  # (B, T, act_dim) (B, T, act_dim)
        # 计算L1损失
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        # all_l1 = F.l1_loss(action_seq, pred_action_seq, reduction='none')
        l1 = all_l1.mean()

        # 组合损失
        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        return loss_dict

    def get_action(self, obs):
        # # 处理深度图：使用tanh变换进行归一化
        # for dk in self.depth_keys:
        #     # 将深度图转换为浮点数并应用变换
        #     obs[dk] = 1 - torch.tanh(obs[dk].float() / 1000)
        
        img_seq = []
        # if self.include_rgb == False:
        #     img_seq.append(obs['fetch_head_depth'].repeat(1, 1, 4, 1, 1))
        #     img_seq.append(obs['fetch_hand_depth'].repeat(1, 1, 4, 1, 1))
        # elif self.include_depth:
        #     img_seq.append(torch.cat([self.normalize(obs['fetch_head_rgb'].float()/255.0), obs['fetch_head_depth']], dim=2))
        #     img_seq.append(torch.cat([self.normalize(obs['fetch_hand_rgb'].float()/255.0), obs['fetch_hand_depth']], dim=2))
        # else:
        #     img_seq.append(self.normalize(obs['fetch_head_rgb'].float()/255.0))
        #     img_seq.append(self.normalize(obs['fetch_hand_rgb'].float()/255.0))
        if self.include_rgb == False:
            img_seq.append(robust_normalize_to_01(obs['fetch_head_depth']).repeat(1, 1, 4, 1, 1))
            img_seq.append(robust_normalize_to_01(obs['fetch_hand_depth']).repeat(1, 1, 4, 1, 1))
        elif self.include_depth:
            img_seq.append(torch.cat([robust_normalize_to_01(obs['fetch_head_rgb']), robust_normalize_to_01(obs['fetch_head_depth'])], dim=2))
            img_seq.append(torch.cat([robust_normalize_to_01(obs['fetch_hand_rgb']), robust_normalize_to_01(obs['fetch_hand_depth'])], dim=2))
        else:
            img_seq.append(robust_normalize_to_01(obs['fetch_head_rgb']))
            img_seq.append(robust_normalize_to_01(obs['fetch_hand_rgb']))
            
        # 将所有图像数据沿通道维度拼接
        img_seq_ = torch.cat(
            img_seq, dim=1
        )  # 形状变为(B, obs_horizon * cam_num, C, H, W) obs_horizon一般指定为1

        # 标准化图像数据为[-1, 1]范围
        B, N, C, H, W = img_seq_.shape
        img_seq_reshaped = img_seq_.view(-1, C, H, W)
        if self.include_rgb == False:
            img_seq_normalized = self.normalize_d(img_seq_reshaped)
        elif self.include_depth:
            img_seq_normalized = self.normalize_rgbd(img_seq_reshaped)
        else:
            img_seq_normalized = self.normalize_rgb(img_seq_reshaped)
        img_seq_ = img_seq_normalized.view(B, N, C, H, W)
        
        obs_ = {}
        obs_["images"] = img_seq_
        # obs_['state'] = obs['state'].squeeze(1)  # (B, 1, state_dim) -> (B, state_dim)
        obs_['state'] = obs['state'].reshape(obs['state'].shape[0], -1)  # (B, T, state_dim) -> (B, T*state_dim)

        # 从先验采样动作
        a_hat, (_, _) = self.model(obs_)

        # # 分割真实动作到不同部件
        # mobile_base_action = a_hat["mobile_base"]  # 移动底座动作
        # torso_action = a_hat["torso"]  # 躯干动作
        # head_action = a_hat["head"]  # 头部动作
        # arm_action = a_hat["arm"]  # 手臂动作
        # a_hat = torch.cat([arm_action, head_action, torso_action, mobile_base_action], dim=-1)

        return a_hat

# KL散度计算函数
def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    # 计算KL散度
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld