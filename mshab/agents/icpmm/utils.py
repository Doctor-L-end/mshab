from h5py import Dataset, File, Group

from gymnasium import spaces

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

from typing import Optional


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0
    

def pose_7d_to_4x4_matrix_torch(poses_7d: torch.Tensor) -> torch.Tensor:
    num_envs = poses_7d.shape[0]
    positions = poses_7d[:, :3]
    quats_wxyz = poses_7d[:, 3:]
    w, x, y, z = quats_wxyz.unbind(dim=1)
    rot_matrices = torch.zeros(num_envs, 3, 3, device=poses_7d.device)
    rot_matrices[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_matrices[:, 0, 1] = 2 * (x * y - w * z)
    rot_matrices[:, 0, 2] = 2 * (x * z + w * y)
    rot_matrices[:, 1, 0] = 2 * (x * y + w * z)
    rot_matrices[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_matrices[:, 1, 2] = 2 * (y * z - w * x)
    rot_matrices[:, 2, 0] = 2 * (x * z - w * y)
    rot_matrices[:, 2, 1] = 2 * (y * z + w * x)
    rot_matrices[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    transform_matrices = torch.zeros(num_envs, 4, 4, device=poses_7d.device)
    transform_matrices[:, :3, :3] = rot_matrices
    transform_matrices[:, :3, 3] = positions
    transform_matrices[:, 3, 3] = 1.0
    return transform_matrices

# ========================= depth -> point cloud =========================
def _depth_to_pointcloud_torch_batched(depth_imgs, rgb_imgs, intrinsics, depth_trunc):
    """depth (N,H,W) --> point cloud (N, H*W, 3)"""
    N, H, W = depth_imgs.shape
    device = depth_imgs.device

    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    v, u = v.expand(N, H, W), u.expand(N, H, W)
    
    # 单位转换
    z = depth_imgs * 0.001

    fx, fy = intrinsics[:, 0, 0, None, None], intrinsics[:, 1, 1, None, None]
    cx, cy = intrinsics[:, 0, 2, None, None], intrinsics[:, 1, 2, None, None]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = torch.stack([x, y, z], dim=-1).view(N, H * W, 3)
    colors = (rgb_imgs.float() / 255.0).view(N, H * W, 3)
    
    valid_mask = (depth_imgs > 10) & (depth_imgs < (depth_trunc * 1000.0)) # e.g., > 10mm and < 2000mm
    mask = valid_mask.view(N, H * W)
    
    return points, colors, mask

# ========================= 暂时用random，不用fps =========================
def _random_sample_points_torch_batched(points, colors, mask, num_samples):
    if num_samples <= 0:
        return None, None
    
    N, P, _ = points.shape
    device = points.device
    
    weights = mask.float()
    non_empty_mask = weights.sum(dim=1) > 0
    if not non_empty_mask.all():
        weights[~non_empty_mask, 0] = 1
    
    indices = torch.multinomial(weights, num_samples, replacement=True)
    
    idx_expanded_xyz = indices.unsqueeze(-1).expand(-1, -1, 3)
    sampled_points = torch.gather(points, 1, idx_expanded_xyz)
    
    idx_expanded_rgb = indices.unsqueeze(-1).expand(-1, -1, 3)
    sampled_colors = torch.gather(colors, 1, idx_expanded_rgb)
    
    num_valid_points = mask.sum(dim=1)
    valid_samples_mask = torch.arange(num_samples, device=device).expand(N, -1) < num_valid_points.unsqueeze(-1)
    
    sampled_points[~valid_samples_mask] = 0
    sampled_colors[~valid_samples_mask] = 0
    
    return sampled_points, sampled_colors

def _get_merged_pc_vectorized(sensor_data, sensor_params, num_envs):
    T_cv_to_gl = torch.tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=torch.float32)
    # 计算世界到Head相机的变换矩阵
    T_w_to_head = torch.inverse(sensor_params["fetch_head"]["cam2world_gl"] @ T_cv_to_gl)
    # 生成原始点云（在各自相机坐标系下）
    head_points, head_colors, head_mask = _depth_to_pointcloud_torch_batched(
        sensor_data["fetch_head"]["depth"], sensor_data["fetch_head"]["rgb"],
        sensor_params["fetch_head"]["intrinsic_cv"], depth_trunc=2.0
    )
    hand_points, hand_colors, hand_mask = _depth_to_pointcloud_torch_batched(
        sensor_data["fetch_hand"]["depth"], sensor_data["fetch_hand"]["rgb"],
        sensor_params["fetch_hand"]["intrinsic_cv"], depth_trunc=2.0
    )
    # Head点云变换：相机坐标系 → Head相机坐标系
    head_extrinsics = T_w_to_head @ (sensor_params["fetch_head"]["cam2world_gl"] @ T_cv_to_gl)
    points_homo = torch.cat([head_points, torch.ones(num_envs, head_points.shape[1], 1)], dim=2)
    transformed_head_points = torch.bmm(points_homo, head_extrinsics.transpose(1, 2))[:, :, :3]
    # Hand点云变换：相机坐标系 → Head相机坐标系
    hand_extrinsics = T_w_to_head @ (sensor_params["fetch_hand"]["cam2world_gl"] @ T_cv_to_gl)
    points_homo = torch.cat([hand_points, torch.ones(num_envs, hand_points.shape[1], 1)], dim=2)
    transformed_hand_points = torch.bmm(points_homo, hand_extrinsics.transpose(1, 2))[:, :, :3]
    # print(transformed_head_points.shape)  # 16384个点
    # print(transformed_hand_points.shape)
    # 在Head坐标系下的过滤条件
    final_head_mask = head_mask & \
                    (transformed_head_points[:, :, 2] >= 0.1) & \
                    (transformed_head_points[:, :, 2] < 1.7)
    
    final_hand_mask = hand_mask & \
                    (transformed_hand_points[:, :, 2] >= 0.1) & \
                    (transformed_hand_points[:, :, 2] < 1.7) & \
                    (transformed_hand_points[:, :, 0] >= -0.5) & \
                    (transformed_hand_points[:, :, 0] <= 0.5)
    num_total, hand_ratio = 1024, 0.0
    num_hand, num_head = int(num_total * hand_ratio), num_total - int(num_total * hand_ratio)
    # 采样和合并点云
    sampled_head_points, sampled_head_colors = _random_sample_points_torch_batched(
        transformed_head_points, head_colors, final_head_mask, num_head
    )
    sampled_hand_points, sampled_hand_colors = _random_sample_points_torch_batched(
        transformed_hand_points, hand_colors, final_hand_mask, num_hand
    )
    if sampled_head_points is None and sampled_hand_points is None:
        raise ValueError("No points sampled!")
    elif sampled_head_points is None:
        return torch.cat([sampled_hand_points, sampled_hand_colors], dim=2)
    elif sampled_hand_points is None:
        return torch.cat([sampled_head_points, sampled_head_colors], dim=2)
    else:
        final_points = torch.cat([sampled_head_points, sampled_hand_points], dim=1)
        final_colors = torch.cat([sampled_head_colors, sampled_hand_colors], dim=1)
    
    return torch.cat([final_points, final_colors], dim=2)
