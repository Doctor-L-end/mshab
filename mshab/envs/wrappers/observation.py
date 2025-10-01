from collections import deque
from typing import Dict, List, Optional
import os
import glob
import json
import torch.nn.functional as F
import cv2
import math
import ipdb
from scipy.spatial.transform import Rotation

import gymnasium as gym

import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.common import flatten_state_dict

import torchvision.transforms as T

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

class FetchPointcloudFromDepthObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, cat_state=True, cat_pixels=False) -> None:
        super().__init__(env)

        self.cat_pixels = cat_pixels
        self.cat_state = cat_state
        self._stack_fn = torch.stack
        self._cat_fn = torch.cat

        self._base_env: BaseEnv = env.unwrapped
        init_raw_obs = common.to_tensor(self._base_env._init_raw_obs)
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])
        self._base_env.update_obs_space(common.to_numpy(self.observation(init_raw_obs)))
        self.device = self._base_env.device

    def _get_merged_pc_vectorized(self, sensor_data, sensor_params, num_envs):
        T_cv_to_gl = torch.tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], device=self.device, dtype=torch.float32)

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
        points_homo = torch.cat([head_points, torch.ones(num_envs, head_points.shape[1], 1, device=self.device)], dim=2)
        transformed_head_points = torch.bmm(points_homo, head_extrinsics.transpose(1, 2))[:, :, :3]

        # Hand点云变换：相机坐标系 → Head相机坐标系
        hand_extrinsics = T_w_to_head @ (sensor_params["fetch_hand"]["cam2world_gl"] @ T_cv_to_gl)
        points_homo = torch.cat([hand_points, torch.ones(num_envs, hand_points.shape[1], 1, device=self.device)], dim=2)
        transformed_hand_points = torch.bmm(points_homo, hand_extrinsics.transpose(1, 2))[:, :, :3]
        
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

    def observation(self, observation):
        # print(observation.keys()) # dict_keys(['agent', 'extra', 'sensor_data', 'sensor_param'])
        agent_obs = observation["agent"]
        extra_obs = observation["extra"]["tcp_pose_wrt_base"]
        # print(agent_obs.keys()) # dict_keys(['qpos', 'qvel'])
        # print(extra_obs.keys()) # dict_keys(['tcp_pose_wrt_base', 'obj_pose_wrt_base', 'goal_pos_wrt_base', 'is_grasped'])
        # for k, v in agent_obs.items():
        #     print(k, v.shape)
        # for k, v in extra_obs.items():
        #     print(k, v.shape)
        # qpos torch.Size([189, 12])
        # qvel torch.Size([189, 12])
        # tcp_pose_wrt_base torch.Size([189, 7])
        # obj_pose_wrt_base torch.Size([189, 7])
        # goal_pos_wrt_base torch.Size([189, 3])
        # is_grasped torch.Size([189])
        fetch_head_depth = self.transforms(observation["sensor_data"]["fetch_head"]["depth"].permute(
            0, 3, 1, 2
        ))
        fetch_hand_depth = self.transforms(observation["sensor_data"]["fetch_hand"]["depth"].permute(
            0, 3, 1, 2
        ))
        fetch_head_rgb = self.transforms(observation["sensor_data"]["fetch_head"]["rgb"].permute(
            0, 3, 1, 2
        ))
        fetch_hand_rgb = self.transforms(observation["sensor_data"]["fetch_hand"]["rgb"].permute(
            0, 3, 1, 2
        ))

        sensor_data = observation["sensor_data"]
        if sensor_data["fetch_head"]["depth"].ndim == 4:
            sensor_data["fetch_head"]["depth"] = sensor_data["fetch_head"]["depth"].squeeze(-1)
        if sensor_data["fetch_hand"]["depth"].ndim == 4:
            sensor_data["fetch_hand"]["depth"] = sensor_data["fetch_hand"]["depth"].squeeze(-1)

        point_clouds_tensor = self._get_merged_pc_vectorized(
            sensor_data, observation["sensor_param"], fetch_head_depth.shape[0]
        )

        image_pixels = (
            dict(
                all_image=self._stack_fn([fetch_head_rgb, fetch_head_depth, fetch_hand_rgb, fetch_hand_depth], axis=-3)
            )
            if self.cat_pixels
            else dict(
                fetch_head_rgb=fetch_head_rgb,
                fetch_hand_rgb=fetch_hand_rgb,
                fetch_head_depth=fetch_head_depth,
                fetch_hand_depth=fetch_hand_depth,
            )
        )
        return (
            dict(
                state=self._cat_fn(
                    [
                        flatten_state_dict(agent_obs, use_torch=True) if isinstance(agent_obs, dict) else agent_obs,
                        flatten_state_dict(extra_obs, use_torch=True) if isinstance(extra_obs, dict) else extra_obs,
                    ],
                    axis=1,
                ),
                pointcloud=point_clouds_tensor,
                **image_pixels,
            )
            if self.cat_state
            else dict(
                agent=agent_obs,
                extra=extra_obs,
                pointcloud=point_clouds_tensor,
                **image_pixels,
            )
        )

class FetchRGBDObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, cat_state=True, cat_pixels=False) -> None:
        super().__init__(env)

        self.cat_pixels = cat_pixels
        self.cat_state = cat_state
        self._stack_fn = torch.stack
        self._cat_fn = torch.cat

        self._base_env: BaseEnv = env.unwrapped
        init_raw_obs = common.to_tensor(self._base_env._init_raw_obs)
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])
        self._base_env.update_obs_space(common.to_numpy(self.observation(init_raw_obs)))

    def observation(self, observation):
        # print(observation.keys()) # dict_keys(['agent', 'extra', 'sensor_data', 'sensor_param'])
        agent_obs = observation["agent"]
        extra_obs = observation["extra"]["tcp_pose_wrt_base"]
        fetch_head_depth = self.transforms(observation["sensor_data"]["fetch_head"]["depth"].permute(
            0, 3, 1, 2
        ))
        fetch_hand_depth = self.transforms(observation["sensor_data"]["fetch_hand"]["depth"].permute(
            0, 3, 1, 2
        ))
        fetch_head_rgb = self.transforms(observation["sensor_data"]["fetch_head"]["rgb"].permute(
            0, 3, 1, 2
        ))
        fetch_hand_rgb = self.transforms(observation["sensor_data"]["fetch_hand"]["rgb"].permute(
            0, 3, 1, 2
        ))

        image_pixels = (
            dict(
                all_image=self._stack_fn([fetch_head_rgb, fetch_head_depth, fetch_hand_rgb, fetch_hand_depth], axis=-3)
            )
            if self.cat_pixels
            else dict(
                fetch_head_rgb=fetch_head_rgb,
                fetch_hand_rgb=fetch_hand_rgb,
                fetch_head_depth=fetch_head_depth,
                fetch_hand_depth=fetch_hand_depth,
            )
        )
        return (
            dict(
                state=self._cat_fn(
                    [
                        flatten_state_dict(agent_obs, use_torch=True) if isinstance(agent_obs, dict) else agent_obs,
                        flatten_state_dict(extra_obs, use_torch=True) if isinstance(extra_obs, dict) else extra_obs,
                    ],
                    axis=1,
                ),
                **image_pixels,
            )
            if self.cat_state
            else dict(
                agent=agent_obs,
                extra=extra_obs,
                **image_pixels,
            )
        )
    
class FetchPointcloudObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, cat_state=True, cat_pixels=False) -> None:
        super().__init__(env)

        self.cat_pixels = cat_pixels
        self.cat_state = cat_state
        self._stack_fn = torch.stack
        self._cat_fn = torch.cat

        self._base_env: BaseEnv = env.unwrapped
        init_raw_obs = common.to_tensor(self._base_env._init_raw_obs)
        self._base_env.update_obs_space(common.to_numpy(self.observation(init_raw_obs)))

    def observation(self, observation):
        # print(observation.keys()) # dict_keys(['agent', 'extra', 'sensor_data', 'sensor_param', 'pointcloud'])
        # print(observation["pointcloud"].keys()) # dict_keys(['xyzw', 'rgb', 'segmentation'])
        # print(observation["pointcloud"]["xyzw"].shape) # torch.Size([252, 32768, 4])
        # print(observation["pointcloud"]["rgb"].shape) # torch.Size([252, 32768, 3])
        # print(observation["pointcloud"]["segmentation"].shape) # torch.Size([252, 32768, 1])
      
        agent_obs = observation["agent"]
        extra_obs = observation["extra"]

        pointcloud = (
            dict(
                pointcloud = self._cat_fn(
                    [
                        observation["pointcloud"]["xyzw"][:, :, :3], # 只取xyz坐标
                        observation["pointcloud"]["rgb"],
                    ], 
                    dim=-1
                )
            )
        )

        return (
            dict(
                state=self._cat_fn(
                    [
                        flatten_state_dict(agent_obs, use_torch=True) if isinstance(agent_obs, dict) else agent_obs,
                        flatten_state_dict(extra_obs, use_torch=True) if isinstance(extra_obs, dict) else extra_obs,
                    ],
                    axis=1,
                ),
                **pointcloud,
            )
            if self.cat_state
            else dict(
                agent=agent_obs,
                extra=extra_obs,
                **pointcloud,
            )
        )

# TODO (arth): deprecate this in favor of StackedDictObservationWrapper + stacking_keys
#   will need to update rl and bc train scripts to matchs new output (i.e. no "pixels" key)
class FrameStack(gym.Wrapper):
    def __init__(
        self,
        env,
        num_stack: int,
        stacking_keys: List[str] = ["fetch_head_rgb", "fetch_head_depth", "fetch_hand_rgb", "fetch_hand_depth"],
    ) -> None:
        super().__init__(env)
        self._base_env = env.unwrapped
        self._num_stack = num_stack
        self._stacking_keys = stacking_keys

        assert all([k in env.observation_space.spaces for k in stacking_keys])

        self._frames: Dict[str, deque] = dict()

        init_raw_obs: dict = self._base_env._init_raw_obs
        pixel_init_raw_obs = dict()
        self._stack_dim = dict()
        for sk in self._stacking_keys:
            obs_space = self.observation_space.spaces[sk]
            init_raw_obs_sk_replace = init_raw_obs.pop(sk)[:, None, ...]
            stack_dim = -len(obs_space.shape[1:]) - 1

            pixel_init_raw_obs[sk] = np.repeat(
                init_raw_obs_sk_replace, num_stack, axis=stack_dim
            )
            self._frames[sk] = deque(maxlen=num_stack)
            self._stack_dim[sk] = stack_dim
        init_raw_obs["pixels"] = pixel_init_raw_obs
        self._base_env.update_obs_space(init_raw_obs)

        self._stack_fn = torch.stack

    def _get_stacked_frames(self):
        return dict(
            (sk, self._stack_fn(tuple(self._frames[sk]), axis=self._stack_dim[sk]))
            for sk in self._stacking_keys
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            frame = obs.pop(sk)
            for _ in range(self._num_stack):
                self._frames[sk].append(frame)
        obs["pixels"] = self._get_stacked_frames()
        return obs, info

    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            self._frames[sk].append(obs.pop(sk))
        obs["pixels"] = self._get_stacked_frames()
        return obs, rew, term, trunc, info


class StackedDictObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        num_stack: int,
        stacking_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(env)
        self._base_env: BaseEnv = env.unwrapped
        self._num_stack = num_stack

        if stacking_keys is None:
            assert isinstance(env.single_observation_space, gym.spaces.Dict)
            self._stacking_keys = env.single_observation_space.keys()
        else:
            assert all([k in env.observation_space.spaces for k in stacking_keys])
            self._stacking_keys = stacking_keys

        self._running_stacks: Dict[str, deque] = dict()

        init_raw_obs: dict = self._base_env._init_raw_obs
        stacked_init_raw_obs = dict()
        self._stack_dim = dict()
        for sk in self._stacking_keys:
            obs_space = self.observation_space.spaces[sk]
            init_raw_obs_sk_replace = init_raw_obs.pop(sk)[:, None, ...]
            stack_dim = -len(obs_space.shape[1:]) - 1

            stacked_init_raw_obs[sk] = np.repeat(
                init_raw_obs_sk_replace, num_stack, axis=stack_dim
            )
            self._running_stacks[sk] = deque(maxlen=num_stack)
            self._stack_dim[sk] = stack_dim
        init_raw_obs.update(**stacked_init_raw_obs)
        self._base_env.update_obs_space(init_raw_obs)

        self._stack_fn = torch.stack

    def _get_stacked_obs(self):
        return dict(
            (
                sk,
                self._stack_fn(
                    tuple(self._running_stacks[sk]), axis=self._stack_dim[sk]
                ),
            )
            for sk in self._stacking_keys
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            frame = obs.pop(sk)
            for _ in range(self._num_stack):
                self._running_stacks[sk].append(frame)
        return self._get_stacked_obs(), info

    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            self._running_stacks[sk].append(obs.pop(sk))
        return self._get_stacked_obs(), rew, term, trunc, info
