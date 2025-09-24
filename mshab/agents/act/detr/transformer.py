# 版权声明
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
DETR Transformer 类。

基于 torch.nn.Transformer 复制修改：
    * 位置编码在多头注意力中传递
    * 移除了编码器末尾的额外 LayerNorm
    * 解码器返回所有解码层的激活值
"""
import copy  # 用于深度复制模块
from typing import Optional, List  # 类型提示

import torch
import torch.nn.functional as F  # 神经网络函数库
from torch import nn, Tensor  # 神经网络模块和张量

import IPython  # 用于调试的交互式Python环境
e = IPython.embed  # 调试用的嵌入函数

class Transformer(nn.Module):
    """DETR 的 Transformer 主类"""
    
    def __init__(self, 
                 d_model=512,          # 特征维度
                 nhead=8,              # 多头注意力头数
                 num_encoder_layers=6, # 编码器层数
                 num_decoder_layers=6, # 解码器层数
                 dim_feedforward=2048, # 前馈网络维度
                 dropout=0.1,           # Dropout概率
                 activation="relu",    # 激活函数类型
                 normalize_before=False, # 是否使用Pre-LN结构
                 return_intermediate_dec=False): # 是否返回中间解码结果
        super().__init__()  # 调用父类初始化

        # 创建编码器层
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        # 编码器归一化层（仅在Pre-LN模式下使用）
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # 构建编码器堆叠
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 创建解码器层
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        # 解码器归一化层
        decoder_norm = nn.LayerNorm(d_model)
        # 构建解码器堆叠
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # 初始化参数
        self._reset_parameters()

        # 保存配置参数
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Xavier均匀初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:  # 只初始化二维以上参数（权重矩阵）
                nn.init.xavier_uniform_(p)

    def forward(self, 
                src,                # 源序列 (图像特征)
                mask,               # 序列掩码
                query_embed,        # 查询嵌入 (目标查询)
                pos_embed,          # 位置编码
                latent_input=None,   # 潜在输入 (如状态向量)
                proprio_input=None, # 本体感知输入
                additional_pos_embed=None): # 额外位置编码
        
        # 处理无图像输入的情况 (仅使用状态输入)
        if src is None:
            bs = proprio_input.shape[0]  # 批量大小
            # 扩展查询嵌入 [num_queries, bs, dim]
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            # 扩展位置编码 [seq, bs, dim]
   
            # print(additional_pos_embed.shape) # torch.Size([2, 256])
    
            pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)
            # 组合输入 [2, bs, dim]
            src = torch.stack([latent_input, proprio_input], axis=0)
        
        # 处理4D图像输入 (标准DETR流程)
        elif len(src.shape) == 4:  # [bs, c, h, w] 格式
            bs, c, h, w = src.shape
            # 展平空间维度: [bs, c, h, w] -> [h*w, bs, c]
            src = src.flatten(2).permute(2, 0, 1)
            # 展平位置编码并扩展: [c, h, w] -> [h*w, bs, c]
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            # 准备目标查询: [num_queries, dim] -> [num_queries, bs, dim]
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            
            # 准备额外位置编码 (状态输入的位置编码)
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)

            # 拼接位置编码: [2 + h*w, bs, dim]
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            # 准备额外输入 (状态输入)
            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            # 拼接输入序列: [2 + h*w, bs, dim]
            src = torch.cat([addition_input, src], axis=0)

        # 初始化目标序列 (全零)
        tgt = torch.zeros_like(query_embed)
        # 编码器前向传播
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # 解码器前向传播
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # 调整输出维度: [layers, seq, bs, dim] -> [layers, bs, seq, dim]
        hs = hs.transpose(1, 2)
        return hs  # 返回解码器输出


class TransformerEncoder(nn.Module):
    """Transformer 编码器堆叠"""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 克隆多个编码器层
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm  # 最终归一化层

    def forward(self, src,
                mask: Optional[Tensor] = None,           # 序列掩码
                src_key_padding_mask: Optional[Tensor] = None, # 键填充掩码
                pos: Optional[Tensor] = None):           # 位置编码
        output = src  # 初始输入

        # 逐层处理
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        # 最终归一化
        if self.norm is not None:
            output = self.norm(output)

        return output  # 返回编码结果


class TransformerDecoder(nn.Module):
    """Transformer 解码器堆叠"""

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # 克隆多个解码器层
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm  # 归一化层
        self.return_intermediate = return_intermediate  # 是否返回中间结果

    def forward(self, tgt, memory,  # 目标序列和编码器输出
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,        # 编码器位置编码
                query_pos: Optional[Tensor] = None): # 目标位置编码
        output = tgt  # 初始目标序列
        intermediate = []  # 存储中间结果

        # 逐层处理
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            # 存储中间结果
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # 最终归一化处理
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()  # 移除未归一化的最后输出
                intermediate.append(output)  # 添加归一化后的输出

        # 返回所有层结果或最后一层结果
        if self.return_intermediate:
            return torch.stack(intermediate)  # [layers, seq_len, bs, dim]
        return output.unsqueeze(0)  # 增加层维度 [1, seq_len, bs, dim]


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = _get_activation_fn(activation)
        # Pre-LN或Post-LN结构标志
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """将位置编码添加到张量"""
        return tensor if pos is None else tensor + pos

    def forward_post(self,  # Post-LN结构实现
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 自注意力计算 (带位置编码)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # 残差连接 + Dropout
        src = src + self.dropout1(src2)
        # 层归一化
        src = self.norm1(src)
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 残差连接 + Dropout
        src = src + self.dropout2(src2)
        # 层归一化
        src = self.norm2(src)
        return src

    def forward_pre(self,  # Pre-LN结构实现
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # 先归一化
        src2 = self.norm1(src)
        # 自注意力计算 (带位置编码)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # 残差连接
        src = src + self.dropout1(src2)
        # 第二次归一化
        src2 = self.norm2(src)
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # 残差连接
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # 根据配置选择Pre-LN或Post-LN
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """Transformer 解码器层"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 交叉注意力机制
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 激活函数
        self.activation = _get_activation_fn(activation)
        # Pre-LN或Post-LN结构标志
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """将位置编码添加到张量"""
        return tensor if pos is None else tensor + pos

    def forward_post(self,  # Post-LN结构实现
                     tgt, memory,  # 目标序列和编码器输出
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,        # 编码器位置编码
                     query_pos: Optional[Tensor] = None): # 目标位置编码
        # 自注意力 (带目标位置编码)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 残差连接 + Dropout
        tgt = tgt + self.dropout1(tgt2)
        # 层归一化
        tgt = self.norm1(tgt)
        # 交叉注意力 (带位置编码)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, 
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        # 残差连接 + Dropout
        tgt = tgt + self.dropout2(tgt2)
        # 层归一化
        tgt = self.norm2(tgt)
        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 残差连接 + Dropout
        tgt = tgt + self.dropout3(tgt2)
        # 层归一化
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self,  # Pre-LN结构实现
                    tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # 第一次归一化
        tgt2 = self.norm1(tgt)
        # 自注意力 (带位置编码)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 残差连接
        tgt = tgt + self.dropout1(tgt2)
        # 第二次归一化
        tgt2 = self.norm2(tgt)
        # 交叉注意力
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, 
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        # 残差连接
        tgt = tgt + self.dropout2(tgt2)
        # 第三次归一化
        tgt2 = self.norm3(tgt)
        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        # 残差连接
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # 根据配置选择Pre-LN或Post-LN
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    """创建N个相同模块的深拷贝"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    """根据配置参数构建Transformer"""
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,  # DETR需要所有解码层输出
    )


def _get_activation_fn(activation):
    """获取激活函数"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")