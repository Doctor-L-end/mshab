# 导入数学库
import math
# 导入类型提示模块
from typing import Union

# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn

# 定义正弦位置嵌入模块
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 嵌入维度

    def forward(self, x):
        device = x.device  # 获取输入张量的设备
        half_dim = self.dim // 2  # 计算一半维度
        emb = math.log(10000) / (half_dim - 1)  # 计算位置编码的缩放因子
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 计算指数衰减
        emb = x[:, None] * emb[None, :]  # 应用缩放
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 组合正弦和余弦分量
        return emb  # 返回位置编码


# 定义一维下采样模块
class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 创建步长为2的卷积层（下采样）
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)  # 应用卷积下采样


# 定义一维上采样模块
class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 创建转置卷积上采样层
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)  # 应用转置卷积上采样


# 定义一维卷积块模块
class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    标准卷积块：卷积层 -> 组归一化 -> Mish激活函数
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        # 创建顺序模块：卷积+归一化+激活
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),  # 一维卷积，保持时间维度不变
            nn.GroupNorm(n_groups, out_channels),  # 组归一化
            nn.Mish(),  # Mish激活函数
        )

    def forward(self, x):
        return self.block(x)  # 应用卷积块


# 定义条件残差块模块
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        # 创建两个卷积块组成的序列
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM调制（特征线性调制）：https://arxiv.org/abs/1709.07871
        # 为每个通道预测缩放因子和偏置
        cond_channels = out_channels * 2  # 条件编码的输出维度（缩放和偏置各占一半）
        self.out_channels = out_channels  # 保存输出通道数
        # 创建条件编码器：Mish激活 -> 线性层 -> 解压为三维张量
        self.cond_encoder = nn.Sequential(
            nn.Mish(),  # Mish激活函数
            nn.Linear(cond_dim, cond_channels),  # 线性变换
            nn.Unflatten(-1, (-1, 1))  # 解压为(B, cond_channels, 1)
        )

        # 创建残差连接：如果输入输出通道数不同则使用1x1卷积调整维度
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)  # 1x1卷积调整维度
            if in_channels != out_channels
            else nn.Identity()  # 相同则使用恒等映射
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ] 输入张量
        cond : [ batch_size x cond_dim] 条件向量
        
        返回:
        out : [ batch_size x out_channels x horizon ] 输出张量
        """
        out = self.blocks[0](x)  # 通过第一个卷积块
        embed = self.cond_encoder(cond)  # 编码条件向量

        # 重塑条件编码：分离缩放和偏置分量
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]  # 提取缩放因子 (B, C, 1)
        bias = embed[:, 1, ...]   # 提取偏置项 (B, C, 1)
        out = scale * out + bias  # 应用FiLM调制：缩放和偏置

        out = self.blocks[1](out)  # 通过第二个卷积块
        out = out + self.residual_conv(x)  # 添加残差连接
        return out


# 定义条件U-Net 1D主网络
class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,  # 输入维度（动作维度）
        global_cond_dim,  # 全局条件维度（观测特征）
        diffusion_step_embed_dim=256,  # 扩散步骤嵌入维度
        down_dims=[256, 512, 1024],  # 下采样路径各层维度
        kernel_size=5,  # 卷积核大小
        n_groups=8,  # 组归一化的组数
    ):
        """
        input_dim: 动作维度
        global_cond_dim: 全局条件维度（通常是obs_horizon * obs_dim）
        diffusion_step_embed_dim: 扩散步骤嵌入的维度
        down_dims: UNet各层的通道大小，数组长度决定层级数
        kernel_size: 卷积核大小
        n_groups: 组归一化的分组数
        """
        super().__init__()
        # 构建所有维度列表（输入+下采样各层）
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]  # 起始维度（第一层输出）

        dsed = diffusion_step_embed_dim  # 缩写
        # 扩散步骤编码器：位置编码 -> 线性层 -> Mish -> 线性层
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),  # 正弦位置编码
            nn.Linear(dsed, dsed * 4),  # 线性变换扩展维度
            nn.Mish(),  # Mish激活
            nn.Linear(dsed * 4, dsed),  # 线性变换压缩维度
        )
        # 总条件维度 = 扩散步骤嵌入 + 全局条件
        cond_dim = dsed + global_cond_dim

        # 创建下采样路径的输入输出对
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]  # 最底层（瓶颈层）维度
        # 创建中间模块（两个条件残差块）
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        # 创建下采样模块
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)  # 是否为最后一层
            down_modules.append(
                nn.ModuleList(
                    [
                        # 第一个条件残差块
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        # 第二个条件残差块
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        # 下采样（非最后一层）或恒等映射（最后一层）
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 创建上采样模块
        up_modules = nn.ModuleList([])
        # 反转下采样路径（从底层到顶层）
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)  # 是否为最后一层
            up_modules.append(
                nn.ModuleList(
                    [
                        # 条件残差块（输入是跳跃连接+上采样的拼接）
                        ConditionalResidualBlock1D(
                            dim_out * 2,  # 输入是跳跃连接+上采样的拼接
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        # 第二个条件残差块
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        # 上采样（非最后一层）或恒等映射（最后一层）
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 最终输出层：卷积块 -> 1x1卷积
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),  # 卷积块
            nn.Conv1d(start_dim, input_dim, 1),  # 1x1卷积输出到输入维度
        )

        # 保存各组件
        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # 打印参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")

    def forward(
        self,
        sample: torch.Tensor,  # 输入样本（带噪声的动作序列）
        timestep: Union[torch.Tensor, float, int],  # 扩散时间步
        global_cond=None,  # 全局条件（观测特征）
    ):
        """
        x: (B,T,input_dim) 输入张量 [批次大小, 时间步, 输入维度]
        timestep: (B,) 或标量，扩散步骤
        global_cond: (B,global_cond_dim) 全局条件
        输出: (B,T,input_dim) 预测结果
        """
        # 调整维度：从(B,T,C)到(B,C,T)
        sample = sample.moveaxis(-1, -2)

        # 处理时间步输入
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # 将标量转换为张量（注意CPU-GPU同步问题）
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            # 处理0维张量（标量）
            timesteps = timesteps[None].to(sample.device)
        # 扩展时间步到批次大小
        timesteps = timesteps.expand(sample.shape[0])

        # 编码扩散时间步
        global_feature = self.diffusion_step_encoder(timesteps)

        # 拼接全局条件（如果存在）
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample  # 输入样本
        h = []  # 保存跳跃连接的列表

        # 下采样路径
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)  # 第一个残差块
            x = resnet2(x, global_feature)  # 第二个残差块
            h.append(x)  # 保存当前层输出（用于跳跃连接）
            x = downsample(x)  # 下采样

        # 中间层（瓶颈层）
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)  # 应用中间模块

        # 上采样路径
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)  # 拼接跳跃连接
            x = resnet(x, global_feature)  # 第一个残差块
            x = resnet2(x, global_feature)  # 第二个残差块
            x = upsample(x)  # 上采样

        # 最终输出层
        x = self.final_conv(x)

        # 调整维度：从(B,C,T)回到(B,T,C)
        x = x.moveaxis(-1, -2)
        return x