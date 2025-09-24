# 导入类型提示模块中的List类型
from typing import List

# 导入Gymnasium库，用于强化学习环境
import gymnasium as gym

# 导入NumPy库，用于数值计算
import numpy as np
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch的函数模块（包含激活函数等）
import torch.nn.functional as F

# 从diffusers库导入DDPMScheduler，用于扩散模型的噪声调度
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# 从自定义模块导入条件U-Net 1D模型
from mshab.agents.dp.conditional_unet1d import ConditionalUnet1D
# 从自定义模块导入普通卷积网络
from mshab.agents.dp.plain_conv import PlainConv

from mshab.vis import robust_normalize_to_01 # 用于图像归一化
import torchvision.transforms as T

# 定义Agent类，继承自nn.Module
class Agent(nn.Module):
    def __init__(
        self,
        single_observation_space: gym.spaces.Dict,  # 单个观测空间（字典类型）
        single_action_space: gym.spaces.Box,         # 单个动作空间（Box类型）
        obs_horizon: int,       # 观测序列长度（历史帧数）
        act_horizon: int,       # 执行动作序列长度
        pred_horizon: int,      # 预测动作序列长度
        diffusion_step_embed_dim: int,  # 扩散步骤嵌入维度
        unet_dims: List[int],   # U-Net各层维度配置
        n_groups: int,          # 分组归一化中的组数
        device: torch.device,   # 计算设备（CPU/GPU）
    ):
        # 调用父类nn.Module的构造函数
        super().__init__()

        # 将设备信息存储到实例变量
        self.device = device

        # 存储时间序列相关参数
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon

        self.normalize_rgbd = T.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])

        # 从观测空间中提取所有图像类型的键（排除"state"）
        self.image_keys = [k for k in single_observation_space.keys() if k != "state"]
        # 从图像键中筛选出深度图相关的键（名称包含"depth"）
        self.depth_keys = [k for k in self.image_keys if "depth" in k]

        # 验证观测空间中的"state"形状是否符合要求（应为二维：[obs_horizon, obs_dim]）
        assert (
            len(single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        # 验证动作空间是否符合要求（应为连续动作空间且范围在[-1,1]）
        assert len(single_action_space.shape) == 1  # (act_dim, )
        assert np.all(single_action_space.high == 1) and np.all(
            single_action_space.low == -1
        )
        # 存储动作维度
        self.act_dim = single_action_space.shape[0]
        # 获取状态观测的维度
        obs_state_dim = single_observation_space["state"].shape[1]

        # 计算所有图像通道的总数
        in_c = 0
        for imk in self.image_keys:
            # 获取每个图像的空间维度（帧数，通道，高，宽）
            fs, d, h, w = single_observation_space[imk].shape
            in_c += d  # 累加通道数
        
        # 视觉特征维度
        visual_feature_dim = 256
        # 创建视觉编码器（普通卷积网络）
        self.visual_encoder = PlainConv(
            img_shape=(in_c, h, w),  # 输入图像形状
            out_dim=visual_feature_dim,  # 输出特征维度
            pool_feature_map=True  # 是否池化特征图
        )

        # 创建噪声预测网络（条件U-Net 1D）
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # 输入维度（动作维度）
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),  # 全局条件维度
            diffusion_step_embed_dim=diffusion_step_embed_dim,  # 扩散步骤嵌入维度
            down_dims=unet_dims,  # U-Net下采样维度配置
            n_groups=n_groups,  # 分组归一化的组数
        )
        
        # 设置扩散过程的总迭代步数
        self.num_diffusion_iters = 100
        # 创建DDPM噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,  # 训练时间步数
            beta_schedule="squaredcos_cap_v2",  # β调度策略（对性能影响大）
            clip_sample=True,  # 是否将输出裁剪到[-1,1]以提升稳定性
            prediction_type="epsilon",  # 预测类型（预测噪声而非去噪后的动作）
        )

    # 定义观测编码方法
    def encode_obs(self, obs_seq, eval_mode):
        # # 处理深度图：使用tanh变换进行归一化
        # for dk in self.depth_keys:
        #     # 将深度图转换为浮点数并应用变换
        #     obs_seq[dk] = 1 - torch.tanh(obs_seq[dk].float() / 1000)
        
        # # 将所有图像数据沿通道维度拼接
        # img_seq = torch.cat(
        #     [obs_seq[k] for k in self.image_keys], dim=2
        # )  # 形状变为(B, obs_horizon, C, H, W)

        img_seq = []
        img1 = torch.cat([robust_normalize_to_01(obs_seq['fetch_head_rgb']), robust_normalize_to_01(obs_seq['fetch_head_depth'])], dim=2)
        img1 = img1.view(-1, img1.shape[2], img1.shape[3], img1.shape[4])
        img1 = self.normalize_rgbd(img1)
        img1 = img1.view(-1, self.obs_horizon, img1.shape[1], img1.shape[2], img1.shape[3])
        img2 = torch.cat([robust_normalize_to_01(obs_seq['fetch_hand_rgb']), robust_normalize_to_01(obs_seq['fetch_hand_depth'])], dim=2)
        img2 = img2.view(-1, img2.shape[2], img2.shape[3], img2.shape[4])
        img2 = self.normalize_rgbd(img2)
        img2 = img2.view(-1, self.obs_horizon, img2.shape[1], img2.shape[2], img2.shape[3])       
        img_seq.append(img1)
        img_seq.append(img2)
        # 将所有图像数据沿通道维度拼接
        img_seq = torch.cat(
            img_seq, dim=2
        )  # 形状变为(B, obs_horizon, C*cam_num, H, W)
        
        # 获取batch大小
        B = img_seq.shape[0]
        # 将前两个维度（batch和序列）展平
        img_seq = img_seq.flatten(end_dim=1)  # 形状变为(B*obs_horizon, C, H, W)
        
        # 如果存在数据增强且非评估模式，则应用增强
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # 应用数据增强
        
        # 通过视觉编码器提取特征
        visual_feature = self.visual_encoder(img_seq)  # 输出形状(B*obs_horizon, D)
        # 将特征恢复为序列形式
        visual_feature = visual_feature.reshape(
            B, self.obs_horizon, visual_feature.shape[1]
        )  # 形状变为(B, obs_horizon, D)
        
        # 将视觉特征和状态特征拼接
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # 形状(B, obs_horizon, D+obs_state_dim)
        
        # 展平所有特征作为全局条件
        return feature.flatten(start_dim=1)  # 形状(B, obs_horizon*(D+obs_state_dim))

    # 定义损失计算方法
    def compute_loss(self, obs_seq, action_seq):
        # 获取batch大小
        B = obs_seq["state"].shape[0]

        # 编码观测作为条件
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # 形状(B, obs_horizon*obs_dim)

        # 生成随机噪声（与动作序列同形）
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

        # 为每个数据点随机采样扩散时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
        ).long()

        # 前向扩散过程：向干净动作添加噪声
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # 预测噪声残差
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        # 计算预测噪声与实际噪声的MSE损失
        return F.mse_loss(noise_pred, noise)

    # 定义获取动作的方法（推理过程）
    def get_action(self, obs_seq):
        # 获取batch大小
        B = obs_seq["state"].shape[0]
        # 禁用梯度计算
        with torch.no_grad():
            # 编码观测（评估模式）
            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True
            )  # 形状(B, obs_horizon*obs_dim)

            # 初始化高斯噪声动作序列
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            # 遍历所有扩散时间步进行去噪
            for k in self.noise_scheduler.timesteps:
                # 预测噪声
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # 逆向扩散步骤：去除噪声
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # 从预测序列中截取要执行的动作部分
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # 形状(B, act_horizon, act_dim)