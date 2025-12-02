import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiNetwork

from .common import ConvType, NormType, get_norm, conv, sum_pool
from .resnet_block import BasicBlock, Bottleneck


class ResNetBase(MinkowskiNetwork):
  BLOCK = None
  LAYERS = ()
  INIT_DIM = 64
  PLANES = (64, 128, 256, 512)
  OUT_PIXEL_DIST = 32
  HAS_LAST_BLOCK = False
  CONV_TYPE = ConvType.HYPERCUBE

  def __init__(self, in_channels, out_channels, conv1_kernel_size, dilations, bn_momentum, D=3, **kwargs):
    assert self.BLOCK is not None
    assert self.OUT_PIXEL_DIST > 0

    super().__init__(D)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.network_initialization(in_channels, out_channels, conv1_kernel_size, dilations, bn_momentum, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, out_channels, conv1_kernel_size, dilations, bn_momentum, D):

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    self.inplanes = self.INIT_DIM
    self.conv1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(conv1_kernel_size, 1),
        stride=1,
        D=D)

    self.bn1 = get_norm(NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum)
    self.relu = ME.MinkowskiReLU(inplace=True)
    self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

    self.layer1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[0], 1))
    self.layer2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[1], 1))
    self.layer3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[2], 1))
    self.layer4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[3], 1))

    self.final = conv(
        self.PLANES[3] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True, D=D)

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

  def _make_layer(self,
                  block,
                  planes,
                  blocks,
                  stride=1,
                  dilation=1,
                  norm_type=NormType.BATCH_NORM,
                  bn_momentum=0.1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv(
              self.inplanes,
              planes * block.expansion,
              kernel_size=1,
              stride=stride,
              bias=False,
              D=self.D),
          get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
      )
    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=self.CONV_TYPE,
            D=self.D))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              stride=1,
              dilation=dilation,
              conv_type=self.CONV_TYPE,
              D=self.D))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.final(x)
    return x


class ResNet14(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 23, 3)


class ResNet18PMP(ResNet18):
    """
    PMP 骨干网 (基于 ResNet18)。
    返回一个包含多尺度稀疏特征的字典。
    """
    def __init__(self, in_channels, out_channels, conv1_kernel_size, dilations, bn_momentum, D=3, **kwargs):
        """
        Args:
            in_channels (int): 输入点云特征维度 
            out_channels (int): 目标输出维度
        """
        # 1. 调用 ResNet18 的 __init__
        # ResNet18.LAYERS = (2, 2, 2, 2)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels, # 原始 `final` 层现在用于 L4
            conv1_kernel_size=conv1_kernel_size,
            dilations=dilations,
            bn_momentum=bn_momentum,
            D=D,
            **kwargs 
        )
        
        # 2. 重命名原始的 `final` 层
        self.proj_l4 = self.final 
        
        # 3. 为 L2 和 L3 创建新的 1x1 卷积投影头
        # ResNet18 (BasicBlock) 中 L2 的输出平面
        # ResNetBase.PLANES[1] = 128
        l2_planes = self.PLANES[1] * self.BLOCK.expansion # 128
        self.proj_l2 = conv(
            l2_planes, out_channels, kernel_size=1, bias=True, D=D
        )
        
        # L3 的输出平面
        # ResNetBase.PLANES[2] = 256
        l3_planes = self.PLANES[2] * self.BLOCK.expansion # 256
        self.proj_l3 = conv(
            l3_planes, out_channels, kernel_size=1, bias=True, D=D
        )

        # 4. 删除不再需要的原始 self.final (已被重命名)
        del self.final

    def forward(self, x: ME.SparseTensor):
        """
        重写 forward，返回多尺度特征字典
        """
        intermediate_features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Layer 1
        x_l1 = self.layer1(x) 
        # Layer 2
        x_l2 = self.layer2(x_l1)
        intermediate_features['shallow'] = self.proj_l2(x_l2)
        # Layer 3
        x_l3 = self.layer3(x_l2)
        intermediate_features['mid'] = self.proj_l3(x_l3)
        # Layer 4
        x_l4 = self.layer4(x_l3)
        intermediate_features['deep'] = self.proj_l4(x_l4)
        
        return intermediate_features

 