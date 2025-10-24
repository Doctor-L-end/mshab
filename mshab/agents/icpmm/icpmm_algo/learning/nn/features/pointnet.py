import torch
import torch.nn as nn

import icpmm_algo.utils as U
from icpmm_algo.learning.nn.common import build_mlp


class PointNetCore(nn.Module):
    def __init__(
        self,
        *,
        point_channels: int = 3,
        output_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self._mlp = build_mlp(
            input_dim=point_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            activation=activation,
        )
        self.output_dim = output_dim

    def forward(self, x):
        """
        x: (..., points, point_channels)
        """
        x = U.any_to_torch_tensor(x)
        x = self._mlp(x)  # (..., points, output_dim)
        x = torch.max(x, dim=-2)[0]  # (..., output_dim)
        return x


class PointNet(nn.Module):
    def __init__(
        self,
        *,
        n_coordinates: int = 3,
        n_color: int = 3,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_depth: int = 2,
        activation: str = "gelu",
        subtract_mean: bool = False,
    ):
        super().__init__()
        assert n_coordinates == 3
        assert n_color == 0 or n_color == 3
        self.n_coordinates = n_coordinates
        self.n_color = n_color
        pn_in_channels = n_coordinates + n_color
        if subtract_mean:
            pn_in_channels += n_coordinates
        self.pointnet = PointNetCore(
            point_channels=pn_in_channels,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
        )
        self.subtract_mean = subtract_mean
        self.output_dim = self.pointnet.output_dim

    def forward(self, x):
        # xyz = x["xyz"]
        # rgb = x["rgb"]
        if self.n_coordinates > 0 and self.n_color > 0:
            xyz = x[..., :3]
            rgb = x[..., 3:6]
            point = U.any_to_torch_tensor(xyz)
            if self.subtract_mean:
                mean = torch.mean(point, dim=-2, keepdim=True)  # (..., 1, coordinates)
                mean = torch.broadcast_to(mean, point.shape)  # (..., points, coordinates)
                point = point - mean
                point = torch.cat([point, mean], dim=-1)  # (..., points, 2 * coordinates)
            rgb = U.any_to_torch_tensor(rgb)
            x = torch.cat([point, rgb], dim=-1)
        elif self.n_coordinates > 0 and self.n_color == 0:
            xyz = x[..., :3]
            point = U.any_to_torch_tensor(xyz)
            if self.subtract_mean:
                mean = torch.mean(point, dim=-2, keepdim=True)  # (..., 1, coordinates)
                mean = torch.broadcast_to(mean, point.shape)  # (..., points, coordinates)
                point = point - mean
                point = torch.cat([point, mean], dim=-1)  # (..., points, 2 * coordinates)
            x = point
        else:
            raise ValueError("n_coordinates和n_color不符合规定")
        return self.pointnet(x)

