import torch
from torch import nn
from .minkowski.resnet import ResNet18PMP
import MinkowskiEngine as ME
from typing import Dict, Tuple

def local_sample_farthest_points(points: torch.Tensor, K: int):
    """
    原生 PyTorch 实现的最远点采样 (FPS)。
    仅在无法导入 pytorch3d 时使用。
    """
    B, N, C = points.shape
    device = points.device
    indices = torch.zeros(B, K, dtype=torch.long).to(device)
    distances = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(K):
        indices[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, -1)[1]
        
    batch_indices_gather = batch_indices.view(B, 1).expand(B, K)
    sampled_points = points[batch_indices_gather, indices, :]
    
    return sampled_points, indices

# 尝试导入 PyTorch3D，如果失败则绑定本地实现
try:
    from pytorch3d.ops import sample_farthest_points
except ImportError:
    sample_farthest_points = local_sample_farthest_points


class Sparse3DEncoderPMP(torch.nn.Module):
    """
    PMP 专用的稀疏 3D 编码器。

    使用 PMP 骨干网 (ResNet18PMP) 并返回一个已Token化的多尺度特征字典。
    输出字典的每个值都是一个 (B, N_level, D) 的稠密张量元组。
    """
    def __init__(self, 
                 input_dim = 3, 
                 output_dim = 512, 
                 max_tokens_l2: int = 256,
                 max_tokens_l3: int = 128,
                 max_tokens_l4: int = 64
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 1. 使用 ResNet18 PMP 骨干网
        self.cloud_encoder = ResNet18PMP(
            in_channels=input_dim, 
            out_channels=output_dim, 
            conv1_kernel_size=3, 
            dilations=(1,1,1,1), 
            bn_momentum=0.02
        )
        
        # 2. 位置编码器
        self.position_embedding = SparsePositionalEncoding(output_dim)

        # 3. 为每个尺度定义 Token 数量
        self.max_tokens = {
            'shallow': max_tokens_l2,
            'mid': max_tokens_l3,
            'deep': max_tokens_l4,
        }

        self.output_ln = nn.LayerNorm(output_dim)

    def _tokenize_sparse_tensor(self, 
                                soutput: ME.SparseTensor, 
                                max_num_token: int, 
                                batch_size: int
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将稀疏张量转换为 (B, N, D) 的 token。
        """
        feats_batch, coords_batch = soutput.F, soutput.C
        feats_list = []
        coords_list = []
        for i in range(batch_size):
            mask = (coords_batch[:,0] == i)
            feats_list.append(feats_batch[mask])
            coords_list.append(coords_batch[mask])
        
        # 位置编码
        pos_list = self.position_embedding(coords_list)

        tokens = torch.zeros([batch_size, max_num_token, self.output_dim], dtype=feats_batch.dtype, device=feats_batch.device)
        pos_emb = torch.zeros([batch_size, max_num_token, self.output_dim], dtype=feats_batch.dtype, device=feats_batch.device)
        token_padding_mask = torch.ones([batch_size, max_num_token], dtype=torch.bool, device=feats_batch.device)
        
        for i, (feats, pos) in enumerate(zip(feats_list, pos_list)):
            
            num_points = len(feats)

            feats = self.output_ln(feats)
            
            if num_points == 0:
                continue
            
            if num_points > max_num_token:
                # 1. 点太多：使用 FPS 下采样
                
                # 获取当前点云的坐标 (N, 4) -> (N, 3)，去掉batch索引列
                # FPS需要 (B, N, 3) 格式的 float 类型输入
                current_coords_xyz = coords_list[i][:, 1:].float().unsqueeze(0) # (1, N, 3)
                
                _, sampled_indices = sample_farthest_points(current_coords_xyz, K=max_num_token)
                
                # 转换索引形状 (1, K) -> (K,)
                sampled_indices = sampled_indices.squeeze(0) 
                
                # 使用采样到的索引来选择特征和位置编码
                tokens[i, :max_num_token] = feats[sampled_indices]
                pos_emb[i, :max_num_token] = pos[sampled_indices]
                token_padding_mask[i, :max_num_token] = False
            
            else:
                # 2. 点不够：直接使用所有点，剩下的会是 padding
                num_token = num_points # 实际 token 数
                
                tokens[i,:num_token] = feats
                pos_emb[i,:num_token] = pos
                token_padding_mask[i,:num_token] = False

        return tokens, pos_emb, token_padding_mask

    def forward(self, 
                sinput: ME.SparseTensor, 
                batch_size: int,
                **kwargs # 不使用
                ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        PMP 编码器的前向传播。
        """
        # 1. 从骨干网获取多尺度稀疏特征
        multi_scale_sparse_features = self.cloud_encoder(sinput)
        
        # 2. 对每个尺度分别进行 Token 化
        output_dict = {}
        for level, sparse_feat in multi_scale_sparse_features.items():
            max_tokens_for_level = self.max_tokens[level]
          
            # 进行解批、位置编码和填充
            tokens, pos_emb, mask = self._tokenize_sparse_tensor(
                sparse_feat, 
                max_tokens_for_level, 
                batch_size
            )
            
            output_dict[level] = (tokens, pos_emb, mask)
            
        return output_dict
    
class SparsePositionalEncoding(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, max_pos=800):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.max_pos = max_pos
        self.origin_pos = max_pos // 2
        self._init_position_vector()

    def _init_position_vector(self):
        x_steps = y_steps = self.num_pos_feats // 3
        z_steps = self.num_pos_feats - x_steps - y_steps
        xyz_embed = torch.arange(self.max_pos, dtype=torch.float32)[:,None]
        x_dim_t = torch.arange(x_steps, dtype=torch.float32); y_dim_t = torch.arange(y_steps, dtype=torch.float32); z_dim_t = torch.arange(z_steps, dtype=torch.float32)
        x_dim_t = self.temperature ** (2 * (x_dim_t // 2) / x_steps); y_dim_t = self.temperature ** (2 * (y_dim_t // 2) / y_steps); z_dim_t = self.temperature ** (2 * (z_dim_t // 2) / z_steps)
        pos_x_vector = xyz_embed / x_dim_t; pos_y_vector = xyz_embed / y_dim_t; pos_z_vector = xyz_embed / z_dim_t
        self.pos_x_vector = torch.stack([pos_x_vector[:,0::2].sin(), pos_x_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_y_vector = torch.stack([pos_y_vector[:,0::2].sin(), pos_y_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_z_vector = torch.stack([pos_z_vector[:,0::2].sin(), pos_z_vector[:,1::2].cos()], dim=2).flatten(1)

    def forward(self, coords_list):
        pos_list = []
        for coords in coords_list:
            coords = (coords[:,1:4] + self.origin_pos).long()
            coords[:,0] = torch.clamp(coords[:,0], 0, self.max_pos-1); coords[:,1] = torch.clamp(coords[:,1], 0, self.max_pos-1); coords[:,2] = torch.clamp(coords[:,2], 0, self.max_pos-1)
            pos_x = self.pos_x_vector.to(coords.device)[coords[:,0]]; pos_y = self.pos_y_vector.to(coords.device)[coords[:,1]]; pos_z = self.pos_z_vector.to(coords.device)[coords[:,2]]
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)
            pos_list.append(pos)
        return pos_list