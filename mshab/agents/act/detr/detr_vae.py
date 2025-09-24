# 版权声明，代码版权归属于Facebook及其关联公司
"""
DETR模型和损失函数类。
"""
import torch
from torch import nn
from torch.autograd import Variable  # 用于自动微分
from mshab.agents.act.detr.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer  # 导入Transformer相关组件

import numpy as np

import IPython  # 用于交互式开发
e = IPython.embed  # 简写嵌入函数


# 重参数化函数（用于变分自编码器）
def reparametrize(mu, logvar):
    # 计算标准差：logvar除以2后取指数
    std = logvar.div(2).exp()
    # 生成与std相同形状的正态分布随机噪声
    eps = Variable(std.data.new(std.size()).normal_())
    # 返回重参数化后的潜在变量：均值 + 标准差 * 随机噪声
    return mu + std * eps


# 生成正弦位置编码表
def get_sinusoid_encoding_table(n_position, d_hid):
    # 内部函数：计算每个位置的角度向量
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # 创建位置编码表：形状为(n_position, d_hid)
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # 对偶数索引应用正弦函数
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    # 对奇数索引应用余弦函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # 转换为PyTorch张量并增加批次维度
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


# DETRVAE模型类（基于DETR的变分自编码器）
class DETRVAE(nn.Module):
    """ 这是执行目标检测的DETR模块 """
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim, num_queries):
        super().__init__()
        # 初始化查询数量（目标检测中的对象查询）
        self.num_queries = num_queries
        # Transformer解码器
        self.transformer = transformer
        # Transformer编码器（用于CVAE）
        self.encoder = encoder
        # 获取Transformer的隐藏维度
        hidden_dim = transformer.d_model
        # 动作预测头：将隐藏状态映射到动作空间
        self.action_head = nn.Linear(hidden_dim, action_dim)
        # 可学习的查询嵌入（用于Transformer解码器）
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 处理视觉主干网络
        if backbones is not None:
            # 1x1卷积将主干特征图通道数转换为隐藏维度
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            # 视觉主干网络列表（多相机）
            self.backbones = nn.ModuleList(backbones)
            # 机器人状态投影层（状态向量->隐藏维度）
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # 无视觉主干时仅使用状态投影
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.backbones = None

        # 编码器额外参数（CVAE相关）
        self.latent_dim = 32  # 潜在变量z的维度
        self.cls_embed = nn.Embedding(1, hidden_dim)  # 额外的CLS令牌嵌入
        self.encoder_state_proj = nn.Linear(state_dim, hidden_dim)  # 状态投影到嵌入
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)  # 动作投影到嵌入
        # 潜在变量投影层：隐藏状态->潜在变量的均值和方差
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)
        # 注册缓冲区：正弦位置编码表（用于[CLS], 状态, 动作序列）
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))

        # 解码器额外参数
        # 潜在变量投影层：潜在样本->嵌入空间
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        # 可学习的位置嵌入（用于状态和本体感知）
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, obs, actions=None):
        # 判断是否训练模式（actions不为None表示训练）
        is_training = actions is not None
        # 提取状态（若有视觉主干则从obs字典获取）
        state = obs['state'] if self.backbones is not None else obs
        bs = state.shape[0]  # 批次大小

        # CVAE编码器部分（仅在训练时使用）
        if is_training:
            # 处理CLS令牌嵌入：(1, hidden_dim) -> (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)
            # 投影状态向量：(bs, state_dim) -> (bs, 1, hidden_dim)
            state_embed = self.encoder_state_proj(state)
            state_embed = torch.unsqueeze(state_embed, axis=1)
            # 投影动作序列：(bs, seq, action_dim) -> (bs, seq, hidden_dim)
            action_embed = self.encoder_action_proj(actions)
            # 拼接输入序列：[CLS, 状态, 动作]
            encoder_input = torch.cat([cls_embed, state_embed, action_embed], axis=1)
            # 调整维度为(seq_len, bs, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)
            # 创建全False的填充掩码（无填充）
            is_pad = torch.full((bs, encoder_input.shape[0]), False).to(state.device)
            # 获取位置编码并调整维度
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)
            # 通过编码器处理
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            # 取CLS令牌的输出（索引0）
            encoder_output = encoder_output[0]
            # 投影到潜在空间：均值+对数方差
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]  # 均值
            logvar = latent_info[:, self.latent_dim:]  # 对数方差
            # 重参数化采样潜在变量
            latent_sample = reparametrize(mu, logvar)
            # 投影潜在样本到嵌入空间
            latent_input = self.latent_out_proj(latent_sample)
        else:
            # 测试模式：使用零向量作为潜在变量
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(state.device)
            latent_input = self.latent_out_proj(latent_sample)

        # CVAE解码器部分（包含视觉处理）
        if self.backbones is not None:
            # 获取视觉数据
            vis_data = obs['images']  # 形状(B, cams, C, H, W)
            # # 若有深度数据则拼接
            # if "depth" in obs:
            #     vis_data = torch.cat([vis_data, obs['depth']], dim=2)
            num_cams = vis_data.shape[1]  # 相机数量

            # 处理多相机数据
            all_cam_features = []  # 存储特征图
            all_cam_pos = []  # 存储位置编码
            for cam_id in range(num_cams):
                # 通过主干网络提取特征（固定使用第一个主干）
                features, pos = self.backbones[0](vis_data[:, cam_id])
                features = features[0]  # 取最后一层特征
                pos = pos[0]  # 对应位置编码
                # 投影特征图通道数
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            # 处理本体感知（机器人状态）
            proprio_input = self.input_proj_robot_state(state)
            # 沿宽度维度拼接多相机特征
            src = torch.cat(all_cam_features, axis=3)  # (batch, hidden_dim, H, W*cams)
            pos = torch.cat(all_cam_pos, axis=3)  # (batch, hidden_dim, H, W*cams)
            # 通过Transformer解码器
            hs = self.transformer(
                src, None, self.query_embed.weight, pos, 
                latent_input, proprio_input, self.additional_pos_embed.weight
            )[0]  # 取输出序列
        else:
            # 无视觉模式：仅处理状态
            state = self.input_proj_robot_state(state)
            hs = self.transformer(
                None, None, self.query_embed.weight, None, 
                latent_input, state, self.additional_pos_embed.weight
            )[0] # 0是最后一层

        # 通过动作头预测动作
        a_hat = self.action_head(hs)
        # 返回预测动作和潜在分布参数
        return a_hat, [mu, logvar]


# 构建编码器函数
def build_encoder(args):
    # 从参数获取配置
    d_model = args.hidden_dim  # 隐藏维度（默认256）
    dropout = args.dropout  # dropout率（默认0.1）
    nhead = args.nheads  # 多头注意力头数（默认8）
    dim_feedforward = args.dim_feedforward  # FFN维度（默认2048）
    num_encoder_layers = args.enc_layers  # 编码器层数（默认4）
    normalize_before = args.pre_norm  # 是否使用Pre-LN（默认False）
    activation = "relu"  # 激活函数

    # 创建编码器层
    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    # 归一化层（Pre-LN模式使用）
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    # 构建完整编码器
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder