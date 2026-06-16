from typing import Dict
import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class PointNetEncoderXYZ(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 512,
            use_layernorm: bool = False,
    ):
        super().__init__()
        assert in_channels == 3

        def norm(dim):
            return nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            norm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            norm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            norm(256),
            nn.ReLU(),
        )
        self.final_projection = nn.Linear(256, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]
        x = self.final_projection(x)
        return x
    
class PointCloudObsEncoder(ModuleAttrMixin):
    def __init__(
            self,
            shape_meta: dict,
            pointnet_out_dim: int = 512,
            use_layernorm: bool = False,
    ):
        super().__init__()

        obs_shape_meta = shape_meta["obs"]

        assert "point_cloud" in obs_shape_meta
        assert "agent_pos" in obs_shape_meta

        self.point_cloud_shape = tuple(obs_shape_meta["point_cloud"]["shape"])
        self.agent_pos_shape = tuple(obs_shape_meta["agent_pos"]["shape"])

        assert len(self.point_cloud_shape) == 2
        assert self.point_cloud_shape[-1] == 3
        assert len(self.agent_pos_shape) == 1

        self.pointnet_out_dim = pointnet_out_dim
        self.agent_pos_dim = self.agent_pos_shape[0]

        self.pointnet = PointNetEncoderXYZ(
            in_channels=3,
            out_channels=pointnet_out_dim,
            use_layernorm=use_layernorm,
        )

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        point_cloud = obs_dict["point_cloud"]
        agent_pos = obs_dict["agent_pos"]

        assert point_cloud.ndim == 3, point_cloud.shape

        point_feat = self.pointnet(point_cloud)
        feat = torch.cat([point_feat, agent_pos], dim=-1)
        return feat
    
    @torch.no_grad()
    def output_shape(self):
        return (self.pointnet_out_dim + self.agent_pos_dim,)


class MultiViewPointCloudObsEncoder(ModuleAttrMixin):
    def __init__(
            self,
            shape_meta: dict,
            pointnet_out_dim: int = 512,
            fusion_hidden_dim: int = 512,
            use_layernorm: bool = False,
            view_keys: tuple[str, ...] = (
                "head_point_cloud",
                "left_wrist_point_cloud",
                "right_wrist_point_cloud",
            ),
    ):
        super().__init__()

        obs_shape_meta = shape_meta["obs"]

        assert "agent_pos" in obs_shape_meta
        for key in view_keys:
            assert key in obs_shape_meta, f"Missing multiview point-cloud key `{key}`."

        self.view_keys = tuple(str(key) for key in view_keys)
        self.point_cloud_shapes = {
            key: tuple(obs_shape_meta[key]["shape"])
            for key in self.view_keys
        }
        self.agent_pos_shape = tuple(obs_shape_meta["agent_pos"]["shape"])

        for key, shape in self.point_cloud_shapes.items():
            assert len(shape) == 2, (key, shape)
            assert shape[-1] == 3, (key, shape)
        assert len(self.agent_pos_shape) == 1

        self.pointnet_out_dim = pointnet_out_dim
        self.agent_pos_dim = self.agent_pos_shape[0]

        self.pointnet = PointNetEncoderXYZ(
            in_channels=3,
            out_channels=pointnet_out_dim,
            use_layernorm=use_layernorm,
        )

        def norm(dim):
            return nn.LayerNorm(dim) if use_layernorm else nn.Identity()

        self.fusion = nn.Sequential(
            nn.Linear(pointnet_out_dim * len(self.view_keys), fusion_hidden_dim),
            norm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, pointnet_out_dim),
        )

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        agent_pos = obs_dict["agent_pos"]
        view_feats = []
        for key in self.view_keys:
            point_cloud = obs_dict[key]
            assert point_cloud.ndim == 3, (key, point_cloud.shape)
            view_feats.append(self.pointnet(point_cloud))

        point_feat = self.fusion(torch.cat(view_feats, dim=-1))
        feat = torch.cat([point_feat, agent_pos], dim=-1)
        return feat

    @torch.no_grad()
    def output_shape(self):
        return (self.pointnet_out_dim + self.agent_pos_dim,)
