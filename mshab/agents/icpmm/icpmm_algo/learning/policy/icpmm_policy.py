# 导入必要的库和模块
from typing import Optional, Union, List  # 用于类型注解

import torch
import torch.nn as nn  # PyTorch神经网络模块
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # 扩散模型DDPM调度器
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # 扩散模型DDIM调度器
from einops import rearrange  # 张量重排操作

# 从自定义模块导入所需组件
from icpmm_algo.learning.nn.common import MLP  # 多层感知机
from icpmm_algo.learning.nn.gpt.gpt import GPT  # GPT Transformer模型
from icpmm_algo.learning.nn.gpt.bigpt import BiGPT  # 支持可控双向注意力的Transformer模型
from icpmm_algo.learning.nn.features import PointNet, ResNet, ObsTokenizer  # PointNet、ResNet和观测令牌化器
from icpmm_algo.learning.nn.diffusion import WholeBodyUNetDiffusionHead  # 全身UNet扩散头

# 定义ICPMM策略类
class ICPMMPolicy(nn.Module):
    # 初始化函数，定义模型的所有组件和参数
    def __init__(
        self,
        *,
        # 本体感觉(proprioception)相关参数
        agent_state_dim: int, # 机器人状态的维度
        agent_state_key: str, # 机器人状态的键名
        prop_dim: int,  # 本体感觉数据的维度
        prop_key: str,  # 本体感觉数据的键名
        prop_mlp_hidden_depth: int,  # 本体感觉MLP隐藏层深度
        prop_mlp_hidden_dim: int,  # 本体感觉MLP隐藏层维度
        
        # PointNet点云处理相关参数
        pointnet_n_coordinates: int,  # 点云坐标数量(通常是3: x,y,z)
        pointnet_n_color: int,  # 点云颜色通道数量(通常是3: r,g,b)
        pointnet_hidden_depth: int,  # PointNet隐藏层深度
        pointnet_hidden_dim: int,  # PointNet隐藏层维度

        # ResNet图像处理相关参数
        resnet_num_blocks: List[int],  # ResNet每个阶段的块数列表
        resnet_use_depth: bool,  # 是否使用深度图像
        resnet_cam_num: int,  # 相机数量
        
        # 观测相关参数
        num_latest_obs: int,  # 使用的最近观测数量(时间步长)
        learnable_future_pointcloud_feature_readout_token: bool,  # 未来点云特征读出令牌是否可学习
        learnable_intention_readout_token: bool,  # 意图读出令牌是否可学习
        
        # ====== Transformer 参数 ======
        xf_n_embd: int,  # Transformer嵌入维度
        xf_n_layer: int,  # Transformer层数
        xf_n_head: int,  # Transformer头数
        xf_dropout_rate: float,  # Transformer dropout率
        xf_use_geglu: bool,  # 是否使用GEGLU激活函数
        
        # ====== 动作解码参数 ======
        learnable_action_readout_token: bool,  # 动作读出令牌是否可学习
        action_dim: int,  # 总动作维度
        action_prediction_horizon: int,  # 动作预测步长(时间步)
        diffusion_step_embed_dim: int,  # 扩散步骤嵌入维度
        unet_down_dims: List[int],  # UNet下采样维度列表
        unet_kernel_size: int,  # UNet卷积核大小
        unet_n_groups: int,  # UNet分组归一化的组数
        unet_cond_predict_scale: bool,  # UNet条件预测缩放
        action_keys: List[str],  # 动作键列表
        action_key_dims: dict[str, int],  # 各动作部分的维度字典
        
        # ====== 扩散过程参数 ======
        noise_scheduler: Union[DDPMScheduler, DDIMScheduler],  # 噪声调度器
        noise_scheduler_step_kwargs: Optional[dict] = None,  # 噪声调度器步进参数
        num_denoise_steps_per_inference: int,  # 每次推理的去噪步数
    ):
        # 调用父类初始化
        super().__init__()

        # 存储机器人状态键
        self.agent_state_key = agent_state_key
        # 初始化机器人状态的MLP
        self.agent_state_mlp = MLP(
            input_dim=agent_state_dim,  # 输入维度
            hidden_dim=prop_mlp_hidden_dim,  # 隐藏层维度
            output_dim=xf_n_embd,  # 输出维度
            hidden_depth=prop_mlp_hidden_depth,  # 隐藏层深度
            add_output_activation=True,  # 添加输出激活函数
        )

        # 存储本体感觉键
        self.prop_key = prop_key
        # 初始化本体感觉MLP
        self.prop_mlp = MLP(
            input_dim=prop_dim,  # 输入维度
            hidden_dim=prop_mlp_hidden_dim,  # 隐藏层维度
            output_dim=xf_n_embd,  # 输出维度
            hidden_depth=prop_mlp_hidden_depth,  # 隐藏层深度
            add_output_activation=True,  # 添加输出激活函数
        )
        
        # 点云键
        self.pointcloud_key = "pointcloud"
        # pointnet
        self.pointnet = PointNet(
            n_coordinates=pointnet_n_coordinates,  # 坐标数量
            n_color=pointnet_n_color,  # 颜色通道数
            output_dim=xf_n_embd,  # 输出维度
            hidden_dim=pointnet_hidden_dim,  # 隐藏层维度
            hidden_depth=pointnet_hidden_depth,  # 隐藏层深度
        )

        if learnable_future_pointcloud_feature_readout_token:
            # 初始化未来点云特征读出令牌
            self.future_pointcloud_feature_readout_token = nn.Parameter(
                torch.zeros(
                    xf_n_embd*2,  # 维度与Transformer嵌入一致
                )
            )
        else:
            self.future_pointcloud_feature_readout_token = torch.zeros(xf_n_embd*2)

        # 初始化未来点云特征MLP
        self.future_pointcloud_feature_mlp = MLP(
            input_dim=xf_n_embd*2,  # 输入维度
            hidden_dim=xf_n_embd,  # 隐藏层维度
            output_dim=xf_n_embd,  # 输出维度
            hidden_depth=1,  # 隐藏层深度
            add_output_activation=True,  # 添加输出激活函数
        )

        # 图像键
        self.image_key = "images"
        # resnet
        self.resnet = ResNet(
            output_dim=xf_n_embd, 
            num_blocks=resnet_num_blocks, # 每个阶段的块数
            use_depth=resnet_use_depth, # 是否使用深度图
            image_num=resnet_cam_num, # 这里有两个摄像头
            )
        
        if learnable_intention_readout_token:
            # 初始化意图读出令牌
            self.intention_readout_token = nn.Parameter(
                torch.zeros(
                    xf_n_embd*2,
                )
            )
        else:
            self.intention_readout_token = torch.zeros(xf_n_embd*2)
        
        # 初始化意图特征MLP
        self.intention_mlp = MLP(
            input_dim=xf_n_embd*2,  # 输入维度
            hidden_dim=xf_n_embd,  # 隐藏层维度
            output_dim=xf_n_embd,  # 输出维度
            hidden_depth=1,  # 隐藏层深度
            add_output_activation=True,  # 添加输出激活函数
        )
        
        # 初始化动作读出令牌
        if learnable_action_readout_token:
            # 可学习的参数
            self.action_readout_token = nn.Parameter(
                torch.zeros(
                    xf_n_embd,  # 维度与Transformer嵌入一致
                )
            )
        else:
            # 固定零张量
            self.action_readout_token = torch.zeros(xf_n_embd)
            
        # 初始化意图推理Transformer模型
        self.intention_transformer = GPT(
            n_embd=xf_n_embd*2,  # 嵌入维度
            n_layer=xf_n_layer,  # 层数
            n_head=xf_n_head,  # 头数
            dropout=xf_dropout_rate,  # dropout率
            use_geglu=xf_use_geglu,  # 是否使用GEGLU
        )
        # 初始化动作特征Transformer模型
        self.action_transformer = BiGPT(
            n_embd=xf_n_embd,  # 嵌入维度
            n_layer=xf_n_layer,  # 层数
            n_head=xf_n_head,  # 头数
            dropout=xf_dropout_rate,  # dropout率
            use_geglu=xf_use_geglu,  # 是否使用GEGLU
        )
        
        # 初始化全身UNet扩散头用于动作解码
        self.action_decoder = WholeBodyUNetDiffusionHead(
            whole_body_decoding_order=["mobile_base", "torso", "head", "arm"],  # 全身解码顺序
            action_dim_per_part={"mobile_base": 2, "torso": 1, "head": 2, "arm": 8},  # 各部件动作维度
            intent_dim=xf_n_embd,  # 意图特征维度
            action_feature_dim=xf_n_embd,  # 动作特征维度
            action_horizon=action_prediction_horizon,  # 动作预测步长
            diffusion_step_embed_dim=diffusion_step_embed_dim,  # 扩散步骤嵌入维度
            noise_scheduler=noise_scheduler,  # 噪声调度器
            noise_scheduler_step_kwargs=noise_scheduler_step_kwargs,  # 噪声调度器步进参数
            inference_denoise_steps=num_denoise_steps_per_inference,  # 推理去噪步数
            unet_down_dims=unet_down_dims,  # UNet下采样维度
            unet_kernel_size=unet_kernel_size,  # UNet卷积核大小
            unet_n_groups=unet_n_groups,  # UNet分组数
            unet_cond_predict_scale=unet_cond_predict_scale,  # UNet条件预测缩放
        )
        
        # 存储最近观测数量
        self.num_latest_obs = num_latest_obs
        # 存储动作相关参数
        self.action_dim = action_dim
        # 验证动作维度总和与总动作维度一致
        assert sum(action_key_dims.values()) == action_dim
        # 验证动作键一致性
        assert set(action_keys) == set(action_key_dims.keys())
        self._action_keys = action_keys
        self._action_key_dims = action_key_dims

    # 前向传播函数
    def forward(
        self,
        obs: dict[str, torch.Tensor],  # 观测字典，包含多种模态的数据
        is_training: bool = True,  # 是否处于训练模式
    ):
        """
        obs: 观测字典, pointcloud形状为 (B, L, ...)，其中 L = num_latest_obs (最新观测数量)
                        agent_state形状为 (B, L, state_dim)
                        images形状为 (B, 1, C, H, W)
                        state形状为 (B, 1, state_dim)
        """
        agent_state_feature = self.agent_state_mlp(
            obs[self.agent_state_key]
        )  # (B, L, E)

        # 把读出令牌转到正确的设备
        self.future_pointcloud_feature_readout_token = self.future_pointcloud_feature_readout_token.to(agent_state_feature.device)
        self.intention_readout_token = self.intention_readout_token.to(agent_state_feature.device)
        self.action_readout_token = self.action_readout_token.to(agent_state_feature.device)

        if is_training:
            # 同时处理历史L帧点云以及1帧未来点云
            pointcloud_feature_all = self.pointnet(
                obs[self.pointcloud_key]
            )  # (B, L+1, E)
            pointcloud_feature = pointcloud_feature_all[:, :-1, :]  # (B, L, E)
            future_pointcloud_feature_gt = pointcloud_feature_all[:, -1:, :]  # (B, 1, E)
        else:
            # 只处理历史L帧点云
            pointcloud_feature = self.pointnet(
                obs[self.pointcloud_key]
            )  # (B, L, E)
            future_pointcloud_feature_gt = None  # 推理时不需要真实未来点云特征
        pointcloud_feature_concated = torch.cat(
            [pointcloud_feature, agent_state_feature], dim=-1
        )  # (B, L, 2*E)

        intention_transformer_input = torch.cat(
            [pointcloud_feature_concated, 
             self.future_pointcloud_feature_readout_token.view(1, 1, -1).expand(pointcloud_feature_concated.shape[0], 1, -1), 
             self.intention_readout_token.view(1, 1, -1).expand(pointcloud_feature_concated.shape[0], 1, -1)], 
             dim=1
        )  # (B, L+2, 2*E)
        seq_len = intention_transformer_input.shape[1]  # L+2
        # 位置编码
        position_ids = torch.arange(
            seq_len,  # 序列长度L+1
            device=intention_transformer_input.device,
        ).unsqueeze(0).expand(intention_transformer_input.shape[0], -1)  # (B, L+2)
        # 前向传播
        intention_transformer_output = self.intention_transformer(
            intention_transformer_input.transpose(0, 1),
            position_ids=position_ids,  # 位置编码
            batch_first=False,  # 批次不在第一维
        ).transpose(0, 1)  # (B, L+2, 2*E)

        if is_training:
            future_pointcloud_token = intention_transformer_output[:, -2, :]  # (B, 2*E)
            future_pointcloud_feature = self.future_pointcloud_feature_mlp(future_pointcloud_token).unsqueeze(1)  # (B, 1, E)
            # 计算预测的未来点云特征与真实未来点云特征的余弦相似度损失
            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            future_pointcloud_feature_loss = 1 - cos(future_pointcloud_feature, future_pointcloud_feature_gt).mean()
        else:
            future_pointcloud_feature_loss = None  # 推理时不计算损失

        intention_token = intention_transformer_output[:, -1, :]  # (B, 2*E)
        intention_feature = self.intention_mlp(intention_token).unsqueeze(1)  # (B, 1, E)

        state_feature = self.prop_mlp(obs[self.prop_key])  # (B, 1, E)
        image_feature = self.resnet(obs[self.image_key])  # (B, 1, E)
  
        action_transformer_input = torch.cat(
            [intention_feature, 
             state_feature, 
             image_feature, 
             self.action_readout_token.view(1, 1, -1).expand(intention_feature.shape[0], 1, -1)], 
             dim=1
        )  # (B, 3+1, E)
        seq_len = action_transformer_input.shape[1]  # 4
        action_transformer_mask = torch.ones(
            action_transformer_input.shape[0],  # 批次大小B
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=action_transformer_input.device,
        )  # (B, 4, 4)
        # 设置动作查询令牌的注意力范围：可以关注所有索引 <= 自身的位置，而之前的Token无法关注动作查询令牌，只能相互关注
        for i in range(seq_len):
            action_transformer_mask[:, i, :i+1] = True
            action_transformer_mask[:, i, i+1:] = False
        action_transformer_mask[:, :3, :3] = True  # 前3个令牌之间完全双向
        # 位置编码
        position_ids = torch.arange(
            seq_len, 
            device=action_transformer_input.device,
        ).unsqueeze(0).expand(action_transformer_input.shape[0], -1)  # (B, 3+1)
        # 前向传播
        action_transformer_output = self.action_transformer(
            action_transformer_input.transpose(0, 1),
            custom_mask=action_transformer_mask,  # 自定义注意力掩码
            position_ids=position_ids,  # 位置编码
            batch_first=False,  # 批次不在第一维
        ).transpose(0, 1)  # (B, 3+1, E)
        action_readout_feature = action_transformer_output[:, -1, :].unsqueeze(1)  # (B, 1, E)
      
        # 验证输出形状
        assert action_readout_feature.shape == (obs[self.prop_key].shape[0], 1, self.action_readout_token.shape[0])
        assert intention_feature.shape == action_readout_feature.shape
      
        return intention_feature, action_readout_feature, future_pointcloud_feature_loss  # (B, 1, E), (B, 1, E), scalar

    # 计算损失函数
    def compute_loss(
        self,
        *,
        obs: dict[str, torch.Tensor] | None = None,  # 可选观测字典
        gt_action: torch.Tensor,  # 真实动作
        alpha: float = 0.5,  # 损失加权参数
    ):
        """
        参数:
            obs: 观测字典
            gt_action: 真实动作，形状为 (B, 1, T_act, A)，其中 T_act = 动作预测步长。
            模型预测T_act个未来动作。
        """
        # 验证输入参数
        assert not (
            obs is None
        ), "必须提供obst"
        
        intention_feature, action_readout_feature, future_pointcloud_feature_loss = self.forward(obs, is_training=True)
        
        # 分割真实动作到不同部件
        mobile_base_action = gt_action[..., -2:]  # 移动底座动作
        torso_action = gt_action[..., -3:-2]  # 躯干动作
        head_action = gt_action[..., -5:-3]  # 头部动作
        arm_action = gt_action[..., :-5]  # 手臂动作
        
        # 计算扩散损失
        loss = self.action_decoder.compute_loss(
            intent_feature=intention_feature,  # 意图特征
            action_feature=action_readout_feature,  # 动作特征
            gt_action={
                "mobile_base": mobile_base_action,  # 移动底座真实动作
                "torso": torso_action,  # 躯干真实动作
                "head": head_action,  # 头部真实动作
                "arm": arm_action,  # 手臂真实动作
            },
        )
        
        return (alpha * loss.mean() + (1-alpha)*future_pointcloud_feature_loss) if future_pointcloud_feature_loss is not None else loss.mean()  # 标量损失

    # 推理函数（无梯度计算）
    @torch.no_grad()
    def inference(
        self,
        *,
        obs: dict[str, torch.Tensor] | None = None,  # 可选观测字典
    ):
        """
        参数:
            obs: 观测字典
            gt_action: 真实动作，形状为 (B, T_act, A)，其中 T_act = 动作预测步长。
            模型预测T_act个未来动作。
        """
        # 验证输入参数
        assert not (
            obs is None
        ), "必须提供obs"
        
        intention_feature, action_readout_feature, _ = self.forward(obs, is_training=False)
        
        # 使用动作解码器进行推理
        pred = self.action_decoder.inference(
            intent_feature=intention_feature,  # 意图特征
            action_feature=action_readout_feature,  # 动作特征
        ) # (B, T_act, A)
        
        # 将预测结果分割到不同部件
        return {
            "mobile_base": pred["mobile_base"],  # 移动底座预测
            "torso": pred["torso"],  # 躯干预测
            "head": pred["head"],  # 头部预测
            "arm": pred["arm"]  # 臂预测
        }

    # 行动函数（无梯度计算）
    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],  # 观测字典
    ):
        # 调用推理函数
        return self.inference(
            obs=obs
        )
    