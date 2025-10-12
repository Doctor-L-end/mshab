import torch
import torch.nn as nn

import icpmm_algo.utils as U
from icpmm_algo.learning.nn.common import build_mlp

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation="relu"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # 激活函数
        if activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class ResNetCore(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 4,  # RGB + Depth
        output_dim: int = 512,
        block=BasicBlock,
        num_blocks: list = [2, 2, 2, 2],
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_planes = 64
        self.activation = activation

        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 激活函数
        if activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

        # 残差块层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, output_dim)
        self.output_dim = output_dim

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = U.any_to_torch_tensor(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int = 512,
        num_blocks: list = [2, 2, 2, 2],
        activation: str = "gelu",
        use_depth: bool = True,
        image_num: int = 1,
    ):
        super().__init__()
        in_channels = 3  # RGB only
        if use_depth:
            in_channels += 1  # Add depth channel
        in_channels *= image_num  # Total channels considering number of images
        self.resnet = ResNetCore(
            in_channels=in_channels,
            output_dim=output_dim,
            num_blocks=num_blocks,
            activation=activation,
        )
        self.use_depth = use_depth
        self.image_num = image_num
        self.output_dim = self.resnet.output_dim

    def forward(self, x):
        if self.use_depth:
            assert x.shape[-3] == 4*self.image_num, f"Expected input with 4N channels (RGBD), but got {x.shape[-3]}"
        else:
            assert x.shape[-3] == 3*self.image_num, f"Expected input with 3N channels (RGB), but got {x.shape[-3]}"        
            
        if x.dim() == 5:
            # 输入形状为 (B, T, C, H, W)，将时间维度和批次维度合并
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        x = self.resnet(x)
        if x.dim() == 2:
            # 输出形状为 (B*T, E)，将时间维度和批次维度分开
            x = x.view(B, T, -1)
        return x
