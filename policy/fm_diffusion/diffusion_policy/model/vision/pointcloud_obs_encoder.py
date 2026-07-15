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

    def encode_points(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def project_points(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_projection(self.encode_points(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_points(x)
        x = torch.max(x, dim=1)[0]
        return self.final_projection(x)


class PointNetSpatialTokenEncoderXYZ(nn.Module):
    """Pool per-point features into a small set of learned spatial tokens."""

    def __init__(
            self,
            num_views: int,
            tokens_per_view: int = 8,
            out_channels: int = 512,
            num_heads: int = 8,
            use_layernorm: bool = False,
    ):
        super().__init__()
        if num_views <= 0 or tokens_per_view <= 0:
            raise ValueError("num_views and tokens_per_view must be positive.")
        if out_channels <= 0 or num_heads <= 0:
            raise ValueError("out_channels and num_heads must be positive.")
        if out_channels % num_heads != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by num_heads={num_heads}."
            )

        self.num_views = num_views
        self.tokens_per_view = tokens_per_view
        self.pointnet = PointNetEncoderXYZ(
            in_channels=3,
            out_channels=out_channels,
            use_layernorm=use_layernorm,
        )
        self.query_tokens = nn.Parameter(
            torch.empty(num_views, tokens_per_view, out_channels)
        )
        self.query_norm = nn.LayerNorm(out_channels)
        self.point_norm = nn.LayerNorm(out_channels)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 4 * out_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * out_channels, out_channels),
        )
        nn.init.normal_(self.query_tokens, std=0.02)

    def forward(self, x: torch.Tensor, view_index: int) -> torch.Tensor:
        if not 0 <= view_index < self.num_views:
            raise ValueError(
                f"view_index must be in [0, {self.num_views}), got {view_index}."
            )
        point_tokens = self.point_norm(self.pointnet.project_points(x))
        queries = self.query_tokens[view_index][None].expand(x.shape[0], -1, -1)
        pooled = self.cross_attention(
            self.query_norm(queries),
            point_tokens,
            point_tokens,
            need_weights=False,
        )[0]
        queries = queries + pooled
        return queries + self.mlp(self.mlp_norm(queries))
    
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


class MultiViewPointCloudSpatialTokenObsEncoder(ModuleAttrMixin):
    """Produce multiple learned spatial tokens for every point-cloud view."""

    def __init__(
            self,
            shape_meta: dict,
            pointnet_out_dim: int = 512,
            tokens_per_view: int = 8,
            token_num_heads: int = 8,
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
        self.tokens_per_view = tokens_per_view
        self.pointnet_out_dim = pointnet_out_dim
        self.agent_pos_shape = tuple(obs_shape_meta["agent_pos"]["shape"])
        self.agent_pos_dim = self.agent_pos_shape[0]

        for key in self.view_keys:
            shape = tuple(obs_shape_meta[key]["shape"])
            assert len(shape) == 2, (key, shape)
            assert shape[-1] == 3, (key, shape)
        assert len(self.agent_pos_shape) == 1

        self.token_encoder = PointNetSpatialTokenEncoderXYZ(
            num_views=len(self.view_keys),
            tokens_per_view=tokens_per_view,
            out_channels=pointnet_out_dim,
            num_heads=token_num_heads,
            use_layernorm=use_layernorm,
        )

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        view_tokens = []
        for view_index, key in enumerate(self.view_keys):
            point_cloud = obs_dict[key]
            assert point_cloud.ndim == 3, (key, point_cloud.shape)
            view_tokens.append(self.token_encoder(point_cloud, view_index))

        batch_size = obs_dict["agent_pos"].shape[0]
        flattened_tokens = torch.cat(view_tokens, dim=1).reshape(batch_size, -1)
        return torch.cat([flattened_tokens, obs_dict["agent_pos"]], dim=-1)

    @torch.no_grad()
    def output_shape(self):
        token_dim = (
            self.pointnet_out_dim * self.tokens_per_view * len(self.view_keys)
        )
        return (token_dim + self.agent_pos_dim,)
