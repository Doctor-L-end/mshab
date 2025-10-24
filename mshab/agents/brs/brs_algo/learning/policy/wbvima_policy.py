# 导入必要的库和模块
from typing import Optional, Union, List  # 用于类型注解

import torch
import torch.nn as nn  # PyTorch神经网络模块
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # 扩散模型DDPM调度器
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # 扩散模型DDIM调度器
from einops import rearrange  # 张量重排操作

# 从自定义模块导入所需组件
from mshab.agents.brs.brs_algo.learning.nn.common import MLP  # 多层感知机
from mshab.agents.brs.brs_algo.learning.nn.gpt.gpt import GPT  # GPT Transformer模型
from mshab.agents.brs.brs_algo.learning.nn.features import PointNet, ObsTokenizer  # PointNet和观测令牌化器
from mshab.agents.brs.brs_algo.learning.policy.base import BaseDiffusionPolicy  # 基础扩散策略类
from mshab.agents.brs.brs_algo.learning.nn.diffusion import WholeBodyUNetDiffusionHead  # 全身UNet扩散头
from mshab.agents.brs.brs_algo.optim import default_optimizer_groups, check_optimizer_groups  # 优化器组相关函数
from mshab.agents.brs.brs_algo.lightning.lightning import rank_zero_info  # 日志记录函数

# 定义WB-VIMA策略类，继承自BaseDiffusionPolicy
class WBVIMAPolicy(BaseDiffusionPolicy):
    # 类属性，标记这是一个序列策略
    is_sequence_policy = True

    # 初始化函数，定义模型的所有组件和参数
    def __init__(
        self,
        *,
        # 本体感觉(proprioception)相关参数
        prop_dim: int,  # 本体感觉数据的维度
        prop_keys: List[str],  # 本体感觉数据的键名列表
        prop_mlp_hidden_depth: int,  # 本体感觉MLP隐藏层深度
        prop_mlp_hidden_dim: int,  # 本体感觉MLP隐藏层维度
        
        # PointNet点云处理相关参数
        pointnet_n_coordinates: int,  # 点云坐标数量(通常是3: x,y,z)
        pointnet_n_color: int,  # 点云颜色通道数量(通常是3: r,g,b)
        pointnet_hidden_depth: int,  # PointNet隐藏层深度
        pointnet_hidden_dim: int,  # PointNet隐藏层维度
        
        # 观测序列相关参数
        num_latest_obs: int,  # 使用的最近观测数量(时间步长)
        use_modality_type_tokens: bool,  # 是否使用模态类型令牌
        
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

        # 存储本体感觉键
        self._prop_keys = prop_keys
        
        # 初始化观测令牌化器，处理多模态输入
        self.obs_tokenizer = ObsTokenizer(
            {
                # 本体感觉模态使用MLP处理
                "proprioception": MLP(
                    prop_dim,  # 输入维度
                    hidden_dim=prop_mlp_hidden_dim,  # 隐藏层维度
                    output_dim=xf_n_embd,  # 输出维度(与Transformer嵌入维度一致)
                    hidden_depth=prop_mlp_hidden_depth,  # 隐藏层深度
                    add_output_activation=True,  # 添加输出激活函数
                ),
                # 点云模态使用PointNet处理
                "pointcloud": PointNet(
                    n_coordinates=pointnet_n_coordinates,  # 坐标数量
                    n_color=pointnet_n_color,  # 颜色通道数
                    output_dim=xf_n_embd,  # 输出维度
                    hidden_dim=pointnet_hidden_dim,  # 隐藏层维度
                    hidden_depth=pointnet_hidden_depth,  # 隐藏层深度
                ),
            },
            use_modality_type_tokens=use_modality_type_tokens,  # 是否使用模态类型令牌
            token_dim=xf_n_embd,  # 令牌维度
            token_concat_order=["proprioception", "pointcloud"],  # 令牌拼接顺序
            strict=True,  # 严格模式
        )
        
        # 存储最近观测数量
        self.num_latest_obs = num_latest_obs
        
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
            
        # 初始化Transformer模型
        self.transformer = GPT(
            n_embd=xf_n_embd,  # 嵌入维度
            n_layer=xf_n_layer,  # 层数
            n_head=xf_n_head,  # 头数
            dropout=xf_dropout_rate,  # dropout率
            use_geglu=xf_use_geglu,  # 是否使用GEGLU
        )
        
        # 初始化全身UNet扩散头用于动作解码
        self.action_decoder = WholeBodyUNetDiffusionHead(
            whole_body_decoding_order=["mobile_base", "torso", "head", "arms"],  # 全身解码顺序
            action_dim_per_part={"mobile_base": 2, "torso": 1, "head": 2, "arms": 8},  # 各部件动作维度
            # whole_body_decoding_order=["action"],  # 全身解码顺序
            # action_dim_per_part={"action": action_dim},  # 各部件动作维
            obs_dim=xf_n_embd,  # 观测维度
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
    ):
        """
        obs: 观测字典，形状为 (B, L, ...)，其中 L = num_latest_obs (最新观测数量)
        """
        # 构建本体感觉观测
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                # 处理分组键（如"proprioception/velocity"）
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                # 处理普通键
                prop_obs.append(obs[prop_key])
        # 沿最后一维拼接所有本体感觉数据
        prop_obs = torch.cat(prop_obs, dim=-1)  # (B, L, Prop_dim)

        # 使用观测令牌化器处理多模态观测
        obs_tokens = self.obs_tokenizer(
            {
                "proprioception": prop_obs,  # 本体感觉数据
                "pointcloud": obs["pointcloud"],  # 点云数据
            }
        )  # (B, L, E)，其中L是交错的多模态令牌
        
        # 获取批次大小、序列长度和嵌入维度
        B, _, E = obs_tokens.shape
        
        # 准备动作读出令牌
        action_readout_tokens = self.action_readout_token.view(1, 1, -1).expand(
            B, self.num_latest_obs, -1
        )

        # 计算每步的令牌数量（观测令牌 + 动作读出令牌）
        n_tokens_per_step = self.obs_tokenizer.num_tokens_per_step + 1
        # 计算总令牌数量
        n_total_tokens = self.num_latest_obs * n_tokens_per_step
        
        # 初始化输入令牌张量
        tokens_in = torch.zeros(
            (B, n_total_tokens, E),
            device=obs_tokens.device,
            dtype=obs_tokens.dtype,
        )
        
        # 插入观测令牌（交错插入不同模态的令牌）
        for j in range(self.obs_tokenizer.num_tokens_per_step):
            tokens_in[:, j::n_tokens_per_step] = obs_tokens[
                :, j :: self.obs_tokenizer.num_tokens_per_step
            ]
            
        # 插入动作读出令牌
        tokens_in[:, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step] = (
            action_readout_tokens
        )

        # 构建注意力掩码
        mask = torch.ones(B, n_total_tokens, dtype=torch.bool, device=obs_tokens.device)
        # 掩码动作读出令牌（使其只能关注之前的令牌）
        mask[:, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step] = False

        # 构建位置ID（从0开始）
        # 同一步中的所有观测令牌共享相同的位置ID
        position_ids = torch.zeros(
            (B, n_total_tokens), device=obs_tokens.device, dtype=torch.long
        )
        p_id = 0
        for t in range(self.num_latest_obs):
            # 计算当前步观测令牌的起始位置
            obs_st = t * n_tokens_per_step
            # 计算当前步观测令牌的结束位置
            obs_end = obs_st + self.obs_tokenizer.num_tokens_per_step
            # 计算动作读出令牌的位置
            action_readout_p = obs_st + self.obs_tokenizer.num_tokens_per_step
            # 为观测令牌分配位置ID
            position_ids[:, obs_st:obs_end] = p_id
            p_id += 1
            # 为动作读出令牌分配位置ID
            position_ids[:, action_readout_p] = p_id
            p_id += 1

        # 重排张量维度以适应Transformer输入 (序列长度, 批次大小, 嵌入维度)
        tokens_in = rearrange(tokens_in, "B T E -> T B E")
        # 扩展掩码维度
        mask = mask.unsqueeze(1)  # (B, 1, T)
        
        # Transformer前向传播
        tokens_out = self.transformer(
            tokens_in, 
            custom_mask=mask,  # 自定义注意力掩码
            batch_first=False,  # 批次不在第一维
            position_ids=position_ids  # 位置编码
        )
        
        # 验证输出形状
        assert tokens_out.shape == (n_total_tokens, B, E)
        # 重排回原始维度
        tokens_out = rearrange(tokens_out, "T B E -> B T E")
        return tokens_out

    # 计算损失函数
    def compute_loss(
        self,
        *,
        obs: dict[str, torch.Tensor] | None = None,  # 可选观测字典
        transformer_output: torch.Tensor | None = None,  # 可选Transformer输出
        gt_action: torch.Tensor,  # 真实动作
    ):
        """
        计算损失，必须提供obs或transformer_output之一

        参数:
            obs: 观测字典，形状为 (B, T, ...)，其中 T = num_latest_obs
            transformer_output: Transformer输出，形状为 (B, L, E)，其中 L = num_latest_obs * n_tokens_per_step
            gt_action: 真实动作，形状为 (B, T_obs, T_act, A)，其中 T_act = 动作预测步长。
                即对于每个观测，模型预测T_act个未来动作。
        """
        # 验证输入参数
        assert not (
            obs is None and transformer_output is None
        ), "必须提供obs或transformer_output"
        
        # 如果没有提供transformer_output，则通过前向传播计算
        if transformer_output is None:
            transformer_output = self.forward(obs)
            
        # 从Transformer输出中提取动作读出令牌
        action_readout_tokens = self._get_action_readout_tokens(transformer_output)
        
        # 分割真实动作到不同部件
        mobile_base_action = gt_action[..., -2:]  # 移动底座动作
        torso_action = gt_action[..., -3:-2]  # 躯干动作
        head_action = gt_action[..., -5:-3]  # 头部动作
        arms_action = gt_action[..., :-5]  # 手臂动作
        
        # 计算扩散损失
        loss = self.action_decoder.compute_loss(
            obs=action_readout_tokens,  # 观测（动作读出令牌）
            gt_action={
                "mobile_base": mobile_base_action,  # 移动底座真实动作
                "torso": torso_action,  # 躯干真实动作
                "head": head_action,  # 头部真实动作
                "arms": arms_action,  # 手臂真实动作
            },
            # gt_action={
            #     "action": gt_action,  # 全部真实动作
            # },
        )
        
        return loss.mean()

    # 推理函数（无梯度计算）
    @torch.no_grad()
    def inference(
        self,
        *,
        obs: dict[str, torch.Tensor] | None = None,  # 可选观测字典
        transformer_output: torch.Tensor | None = None,  # 可选Transformer输出
        return_last_timestep_only: bool,  # 是否只返回最后时间步
    ):
        """
        计算预测，必须提供obs或transformer_output之一

        参数:
            obs: 观测字典，形状为 (B, T, ...)，其中 T = num_latest_obs
            transformer_output: Transformer输出，形状为 (B, L, E)，其中 L = num_latest_obs * n_tokens_per_step
            return_last_timestep_only: 是否只返回最后时间步的动作块。
        """
        # 验证输入参数
        assert not (
            obs is None and transformer_output is None
        ), "必须提供obs或transformer_output"
        
        # 如果没有提供transformer_output，则通过前向传播计算
        if transformer_output is None:
            transformer_output = self.forward(obs)
            
        # 从Transformer输出中提取动作读出令牌
        action_readout_tokens = self._get_action_readout_tokens(transformer_output)
        
        # 使用动作解码器进行推理
        pred = self.action_decoder.inference(
            obs=action_readout_tokens,
            return_last_timestep_only=return_last_timestep_only,
        )  # (B, T_obs, T_act, A) 或 (B, T_act, A)
        
        # 将预测结果分割到不同部件
        return {
            "mobile_base": pred["mobile_base"],  # 移动底座预测
            "torso": pred["torso"],  # 躯干预测
            "head": pred["head"],  # 头部预测
            "arms": pred["arms"]  # 臂预测
            # "action": pred["action"]  # 全部预测
        }

    # 行动函数（无梯度计算）
    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],  # 观测字典
    ):
        # 调用推理函数，只返回最后时间步的动作
        return self.inference(
            obs=obs,
            return_last_timestep_only=True,
        )

    # 辅助函数：从Transformer输出中提取动作读出令牌
    def _get_action_readout_tokens(self, transformer_output: torch.Tensor):
        # 获取批次大小、序列长度和嵌入维度
        B, _, E = transformer_output.shape
        # 计算每步的令牌数量
        n_tokens_per_step = self.obs_tokenizer.num_tokens_per_step + 1
        
        # 提取动作读出令牌（每隔n_tokens_per_step取一个令牌）
        action_readout_tokens = transformer_output[
            :, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step
        ]  # (B, T_obs, E)
        
        # 验证形状
        assert action_readout_tokens.shape == (B, self.num_latest_obs, E)
        return action_readout_tokens

    # 获取优化器参数组
    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        # 获取观测令牌化器的优化器参数组
        (
            feature_encoder_pg,  # 参数组
            feature_encoder_pid,  # 参数ID
        ) = self.obs_tokenizer.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        
        # 获取Transformer的优化器参数组
        transformer_pg, transformer_pid = self.transformer.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        
        # 获取动作解码器的优化器参数组
        action_decoder_pg, action_decoder_pid = (
            self.action_decoder.get_optimizer_groups(
                weight_decay=weight_decay,
                lr_layer_decay=lr_layer_decay,
                lr_scale=lr_scale,
            )
        )
        
        # 获取其余参数的优化器参数组
        other_pg, _ = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "action_readout_token",  # 动作读出令牌不应用权重衰减
            ],
            # 排除已经在其他组中的参数
            exclude_filter=lambda name, p: id(p)
            in feature_encoder_pid + transformer_pid + action_decoder_pid,
        )
        
        # 合并所有参数组
        all_groups = feature_encoder_pg + transformer_pg + action_decoder_pg + other_pg
        
        # 检查优化器组并记录信息
        _, table_str = check_optimizer_groups(self, all_groups, verbose=True)
        rank_zero_info(table_str)  # 记录信息
        
        return all_groups