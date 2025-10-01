from h5py import Dataset, File, Group

from gymnasium import spaces

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler


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


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "observations": "obs",
    "success": "success",
    "next_observations": "obs",
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    "actions": "actions",
}


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path, num_traj=None):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret


def load_demo_dataset(
    path, keys=["observations", "actions"], num_traj=None, concat=True
):
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data["traj_0"]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[:-1] for t in dataset[target_key]], axis=0
                )
            elif target_key in ["next_observations", "next_states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[1:] for t in dataset[target_key]], axis=0
                )
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset


def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor):
    img_dict = obs["sensor_data"]
    new_img_dict = {
        key: transpose_fn(
            concat_fn([v[key] for v in img_dict.values()])
        )  # (C, H, W) or (B, C, H, W)
        for key in ["rgb", "depth"]
    }
    # if isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
    #     new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

    # Unified version
    states_to_stack = state_obs_extractor(obs)
    for j in range(len(states_to_stack)):
        if states_to_stack[j].dtype == np.float64:
            states_to_stack[j] = states_to_stack[j].astype(np.float32)
    try:
        state = np.hstack(states_to_stack)
    except:  # dirty fix for concat trajectory of states
        state = np.column_stack(states_to_stack)
    if state.dtype == np.float64:
        for x in states_to_stack:
            print(x.shape, x.dtype)
        import pdb

        pdb.set_trace()

    out_dict = {
        "state": state,
        "rgb": new_img_dict["rgb"],
        "depth": new_img_dict["depth"],
    }
    return out_dict


def build_obs_space(env, depth_dtype, state_obs_extractor):
    # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
    obs_space = env.observation_space

    # Unified version
    state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

    single_img_space = next(iter(env.observation_space["image"].values()))
    h, w, _ = single_img_space["rgb"].shape
    n_images = len(env.observation_space["image"])

    return spaces.Dict(
        {
            "state": spaces.Box(
                -float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32
            ),
            "rgb": spaces.Box(0, 255, shape=(n_images * 3, h, w), dtype=np.uint8),
            "depth": spaces.Box(
                -float("inf"), float("inf"), shape=(n_images, h, w), dtype=depth_dtype
            ),
        }
    )


def build_state_obs_extractor(env_id):
    env_name = env_id.split("-")[0]
    if env_name in ["TurnFaucet", "StackCube"]:
        return lambda obs: list(obs["extra"].values())
    elif env_name == "PushChair" or env_name == "PickCube":
        return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())
    else:
        raise NotImplementedError(f"Please tune state obs by yourself")






from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from h5py import File, Group, Dataset
from typing import Optional


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
    num_total, hand_ratio = 4096, 0.6
    num_hand, num_head = int(num_total * hand_ratio), num_total - int(num_total * hand_ratio)
    # 采样和合并点云
    sampled_head_points, sampled_head_colors = _random_sample_points_torch_batched(
        transformed_head_points, head_colors, final_head_mask, num_head
    )
    sampled_hand_points, sampled_hand_colors = _random_sample_points_torch_batched(
        transformed_hand_points, hand_colors, final_hand_mask, num_hand
    )
    
    final_points = torch.cat([sampled_head_points, sampled_hand_points], dim=1)
    final_colors = torch.cat([sampled_head_colors, sampled_hand_colors], dim=1)
    
    return torch.cat([final_points, final_colors], dim=2)
