# 导入必要的库
from typing import Dict  # 用于类型注解，表示字典类型
from gymnasium import spaces  # 强化学习环境空间定义
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from torch.distributions.normal import Normal  # 正态分布，用于连续动作空间
from mshab.vis import robust_normalize_to_01  # 用于图像归一化

# 神经网络层初始化函数
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    使用正交初始化方法初始化神经网络层
    
    参数:
        layer: 需要初始化的神经网络层
        std: 初始化权重的标准差，默认为√2（适合ReLU激活函数）
        bias_const: 偏置项的常数初始化值，默认为0.0
    
    返回:
        初始化后的层
    """
    # 使用正交初始化方法初始化权重
    torch.nn.init.orthogonal_(layer.weight, std)
    # 使用常数初始化方法初始化偏置
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# 定义强化学习智能体神经网络
class Agent(nn.Module):
    def __init__(self, sample_obs, single_act_shape):
        """
        初始化Agent神经网络
        
        参数:
            sample_obs: 示例观测值，用于确定网络结构
            single_act_shape: 单个动作的形状（维度）
        """
        super().__init__()  # 调用父类nn.Module的初始化方法

        # 创建特征提取器字典（用于处理不同类型的输入）
        extractors = dict()
        extractor_out_features = 0  # 记录特征提取器输出的总特征数
        feature_size = 1024  # 每个特征提取器的输出特征维度

        # 从示例观测中分离像素观测和状态观测
        pixel_obs: Dict[str, torch.Tensor] = sample_obs["pixels"]  # 字典形式的多相机像素数据
        state_obs: torch.Tensor = sample_obs["state"]  # 状态向量数据
        
        pixel_obs_ = {}
        pixel_obs_["fetch_head_rgbd"]=torch.cat([robust_normalize_to_01(pixel_obs['fetch_head_rgb']), 
                                                robust_normalize_to_01(pixel_obs['fetch_head_depth'])], dim=-3)
        pixel_obs_["fetch_hand_rgbd"]=torch.cat([robust_normalize_to_01(pixel_obs['fetch_hand_rgb']), 
                                                robust_normalize_to_01(pixel_obs['fetch_hand_depth'])], dim=-3)
        # print("pixel_obs_['fetch_head_rgbd'] shape:", pixel_obs_["fetch_head_rgbd"].shape)

        # 为每个像素观测源（相机）创建特征提取器
        for k, pobs in pixel_obs_.items():
            # 处理5维张量（批处理大小×帧堆叠×深度×高度×宽度）
            if len(pobs.shape) == 5:
                b, fs, d, h, w = pobs.shape
                pobs = pobs.reshape(b, fs * d, h, w)  # 合并帧堆叠和深度维度
            
            # 获取合并后的通道数
            pobs_stack = pobs.size(1)
            
            # 定义CNN特征提取器（基于NVIDIA的端到端自动驾驶架构）
            cnn = nn.Sequential(
                # 第一卷积层：5×5卷积核，步长2，无填充
                nn.Conv2d(
                    in_channels=pobs_stack,
                    out_channels=24,
                    kernel_size=5,
                    stride=2,
                    padding="valid", # 表示不使用填充，只进行有效卷积，导致输出尺寸减小。
                ),
                nn.ReLU(),  # ReLU激活函数
                # 第二卷积层：5×5卷积核，步长2
                nn.Conv2d(
                    in_channels=24,
                    out_channels=36,
                    kernel_size=5,
                    stride=2,
                    padding="valid",
                ),
                nn.ReLU(),
                # 第三卷积层：5×5卷积核，步长2
                nn.Conv2d(
                    in_channels=36,
                    out_channels=48,
                    kernel_size=5,
                    stride=2,
                    padding="valid",
                ),
                nn.ReLU(),
                # 第四卷积层：3×3卷积核，步长1
                nn.Conv2d(
                    in_channels=48,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="valid",
                ),
                nn.ReLU(),
                # 第五卷积层：3×3卷积核，步长1
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Flatten(),  # 展平多维特征图为一维向量
            )
            
            # 动态计算CNN输出的展平尺寸
            with torch.no_grad():  # 禁用梯度计算
                n_flatten = cnn(pobs.float().cpu()).shape[1]  # 获取展平后的特征数量
            
            # 添加全连接层将CNN输出降维到统一特征大小
            fc = nn.Sequential(
                nn.Linear(n_flatten, feature_size),  # 线性层
                nn.ReLU()  # ReLU激活函数
            )
            
            # 将CNN和FC组合作为该像素源的特征提取器
            extractors[k] = nn.Sequential(cnn, fc)
            extractor_out_features += feature_size  # 累加特征维度

        # 为状态观测创建特征提取器（单层线性层）
        extractors["state"] = nn.Linear(state_obs.size(-1), feature_size)
        extractor_out_features += feature_size  # 累加特征维度

        # 将特征提取器字典转换为PyTorch模块字典
        self.extractors = nn.ModuleDict(extractors)

        # 定义多层感知机（MLP）处理融合特征
        self.mlp = nn.Sequential(
            # 第一全连接层（使用自定义初始化）
            layer_init(nn.Linear(extractor_out_features, 2048)),
            nn.ReLU(inplace=True),  # 原地ReLU节省内存
            # 第二全连接层
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(inplace=True),
            # 第三全连接层
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(inplace=True),
            # 输出层（使用更小的标准差初始化）
            layer_init(
                nn.Linear(512, np.prod(single_act_shape)),  # 输出维度=动作空间维度
                std=0.01 * np.sqrt(2),  # 减小标准差防止输出过大
            ),
        )

    def forward(self, observations) -> torch.Tensor:
        """
        前向传播处理观测数据
        
        参数:
            observations: 包含像素和状态观测的字典
        
        返回:
            动作张量
        """
        # 分离像素观测和状态观测
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]

        pixel_obs_ = {}
        pixel_obs_["fetch_head_rgbd"]=torch.cat([robust_normalize_to_01(pixels['fetch_head_rgb']), 
                                                robust_normalize_to_01(pixels['fetch_head_depth'])], dim=-3)
        pixel_obs_["fetch_hand_rgbd"]=torch.cat([robust_normalize_to_01(pixels['fetch_hand_rgb']), 
                                                robust_normalize_to_01(pixels['fetch_hand_depth'])], dim=-3)
        
        # 存储各特征提取器的输出
        encoded_tensor_list = []
        
        # 处理每个特征提取器
        for key, extractor in self.extractors.items():
            if key == "state":
                # 处理状态观测（直接通过线性层）
                encoded_tensor_list.append(extractor(state))
            else:
                # 处理像素观测（禁用自动求导以节省内存）
                with torch.no_grad():
                    pobs = pixel_obs_[key].float()
                    # # 像素值预处理（两种备选方案，当前使用双曲正切方案）
                    # # pobs = 1 / (1 + pixel_obs_[key].float() / 400)  # 备选预处理方案1
                    # pobs = 1 - torch.tanh(pixel_obs_[key].float() / 1000)  # 双曲正切预处理方案
                    pobs=robust_normalize_to_01(pobs)
                    # 处理5维张量（批处理大小×帧堆叠×深度×高度×宽度）
                    if len(pobs.shape) == 5:
                        b, fs, d, h, w = pobs.shape
                        pobs = pobs.reshape(b, fs * d, h, w)  # 合并帧堆叠和深度维度
                
                # 通过特征提取器并保存结果
                encoded_tensor_list.append(extractor(pobs))
        
        # 拼接所有特征并输入MLP
        return self.mlp(torch.cat(encoded_tensor_list, dim=1))