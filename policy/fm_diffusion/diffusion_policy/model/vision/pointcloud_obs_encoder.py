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


class PointNetPatchTokenEncoderXYZ(nn.Module):
    """Encode FPS-centered local point patches into spatially anchored tokens."""

    def __init__(
            self,
            out_channels: int = 512,
            point_feature_dim: int = 256,
            neighbors_per_patch: int = 32,
            ignore_zero_points: bool = False,
            use_layernorm: bool = False,
    ):
        super().__init__()
        if out_channels <= 0 or point_feature_dim <= 0:
            raise ValueError("Point feature and output dimensions must be positive.")
        if neighbors_per_patch <= 0:
            raise ValueError("neighbors_per_patch must be positive.")

        def norm(dim):
            return nn.LayerNorm(dim) if use_layernorm else nn.Identity()

        self.neighbors_per_patch = neighbors_per_patch
        self.ignore_zero_points = ignore_zero_points
        self.local_mlp = nn.Sequential(
            nn.Linear(3, 64),
            norm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            norm(128),
            nn.ReLU(),
            nn.Linear(128, point_feature_dim),
            norm(point_feature_dim),
            nn.ReLU(),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(3, point_feature_dim),
            norm(point_feature_dim),
            nn.ReLU(),
            nn.Linear(point_feature_dim, point_feature_dim),
        )
        self.output_norm = norm(point_feature_dim)
        self.output_projection = nn.Linear(point_feature_dim, out_channels)

    @staticmethod
    def _farthest_point_indices(
            points: torch.Tensor,
            num_samples: int,
            valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_points, _ = points.shape
        if num_samples > num_points:
            raise ValueError(
                f"Cannot sample {num_samples} patch centers from {num_points} points."
            )

        # Sampling indices do not need gradients. Float32 distance calculations
        # keep FPS stable when the learned layers run under BF16 autocast.
        selection_points = points.detach().float()
        valid_weights = valid_mask.to(selection_points.dtype)
        centroid = torch.sum(selection_points * valid_weights[..., None], dim=1, keepdim=True)
        centroid = centroid / valid_weights.sum(dim=1, keepdim=True).clamp_min(1)[..., None]
        first_distances = torch.sum((selection_points - centroid) ** 2, dim=-1)
        farthest = first_distances.masked_fill(~valid_mask, -1).argmax(dim=1)
        min_distances = torch.full(
            (batch_size, num_points),
            torch.finfo(selection_points.dtype).max,
            device=points.device,
            dtype=selection_points.dtype,
        )
        batch_indices = torch.arange(batch_size, device=points.device)
        sampled_indices = []

        for _ in range(num_samples):
            sampled_indices.append(farthest)
            center = selection_points[batch_indices, farthest][:, None, :]
            distances = torch.sum((selection_points - center) ** 2, dim=-1)
            min_distances = torch.minimum(min_distances, distances)
            farthest = min_distances.masked_fill(~valid_mask, -1).argmax(dim=1)

        return torch.stack(sampled_indices, dim=1)

    @staticmethod
    def _gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_indices = indices[..., None].expand(*indices.shape, points.shape[-1])
        return torch.gather(points, dim=1, index=gather_indices)

    def forward(self, points: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected point cloud [B, N, 3], got {tuple(points.shape)}.")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        if num_tokens > points.shape[1]:
            raise ValueError(
                f"Cannot sample {num_tokens} patch centers from {points.shape[1]} points."
            )
        if self.neighbors_per_patch > points.shape[1]:
            raise ValueError(
                f"neighbors_per_patch={self.neighbors_per_patch} exceeds "
                f"point count {points.shape[1]}."
            )

        if self.ignore_zero_points:
            valid_mask = torch.any(points.detach() != 0, dim=-1)
            has_valid_points = valid_mask.any(dim=1, keepdim=True)
            valid_mask = torch.where(
                has_valid_points,
                valid_mask,
                torch.ones_like(valid_mask),
            )
        else:
            valid_mask = torch.ones(
                points.shape[:2],
                device=points.device,
                dtype=torch.bool,
            )

        center_indices = self._farthest_point_indices(points, num_tokens, valid_mask)
        centers = self._gather_points(points, center_indices)

        selection_points = points.detach().float()
        selection_centers = self._gather_points(selection_points, center_indices)
        point_sq = torch.sum(selection_points ** 2, dim=-1)[:, None, :]
        center_sq = torch.sum(selection_centers ** 2, dim=-1)[:, :, None]
        squared_distances = center_sq + point_sq - 2 * torch.bmm(
            selection_centers,
            selection_points.transpose(1, 2),
        )
        squared_distances = squared_distances.clamp_min(0)
        squared_distances = squared_distances.masked_fill(~valid_mask[:, None, :], float("inf"))
        neighbor_indices = torch.topk(
            squared_distances,
            k=self.neighbors_per_patch,
            dim=-1,
            largest=False,
            sorted=False,
        ).indices
        neighbor_valid_mask = torch.gather(
            valid_mask[:, None, :].expand(-1, num_tokens, -1),
            dim=2,
            index=neighbor_indices,
        )
        neighbor_indices = torch.where(
            neighbor_valid_mask,
            neighbor_indices,
            center_indices[:, :, None],
        )

        expanded_points = points[:, None].expand(-1, num_tokens, -1, -1)
        neighbor_gather_indices = neighbor_indices[..., None].expand(-1, -1, -1, 3)
        neighbors = torch.gather(
            expanded_points,
            dim=2,
            index=neighbor_gather_indices,
        )
        relative_points = neighbors - centers[:, :, None, :]

        local_features = self.local_mlp(relative_points).amax(dim=2)
        position_features = self.position_mlp(centers)
        return self.output_projection(self.output_norm(local_features + position_features))
    
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


class MultiViewPointCloudPatchTokenObsEncoder(ModuleAttrMixin):
    """Produce spatially anchored local patch tokens for each point-cloud view."""

    def __init__(
            self,
            shape_meta: dict,
            pointnet_out_dim: int = 512,
            point_feature_dim: int = 256,
            view_token_counts: tuple[int, ...] = (12, 6, 6),
            neighbors_per_patch: int = 32,
            ignore_zero_points: bool = False,
            use_layernorm: bool = False,
            view_keys: tuple[str, ...] = (
                "head_point_cloud",
                "left_wrist_point_cloud",
                "right_wrist_point_cloud",
            ),
    ):
        super().__init__()
        obs_shape_meta = shape_meta["obs"]
        if len(view_keys) != len(view_token_counts):
            raise ValueError(
                "view_keys and view_token_counts must contain the same number of views."
            )
        if any(int(count) <= 0 for count in view_token_counts):
            raise ValueError("Every view token count must be positive.")
        assert "agent_pos" in obs_shape_meta
        for key in view_keys:
            assert key in obs_shape_meta, f"Missing multiview point-cloud key `{key}`."

        self.view_keys = tuple(str(key) for key in view_keys)
        self.view_token_counts = tuple(int(count) for count in view_token_counts)
        self.pointnet_out_dim = pointnet_out_dim
        self.agent_pos_shape = tuple(obs_shape_meta["agent_pos"]["shape"])
        self.agent_pos_dim = self.agent_pos_shape[0]

        for key, token_count in zip(self.view_keys, self.view_token_counts):
            shape = tuple(obs_shape_meta[key]["shape"])
            assert len(shape) == 2, (key, shape)
            assert shape[-1] == 3, (key, shape)
            if token_count > shape[0]:
                raise ValueError(
                    f"View `{key}` requests {token_count} tokens from only {shape[0]} points."
                )
            if neighbors_per_patch > shape[0]:
                raise ValueError(
                    f"View `{key}` has {shape[0]} points, fewer than "
                    f"neighbors_per_patch={neighbors_per_patch}."
                )
        assert len(self.agent_pos_shape) == 1

        self.token_encoder = PointNetPatchTokenEncoderXYZ(
            out_channels=pointnet_out_dim,
            point_feature_dim=point_feature_dim,
            neighbors_per_patch=neighbors_per_patch,
            ignore_zero_points=ignore_zero_points,
            use_layernorm=use_layernorm,
        )

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        view_tokens = []
        for key, token_count in zip(self.view_keys, self.view_token_counts):
            point_cloud = obs_dict[key]
            assert point_cloud.ndim == 3, (key, point_cloud.shape)
            view_tokens.append(self.token_encoder(point_cloud, token_count))

        batch_size = obs_dict["agent_pos"].shape[0]
        flattened_tokens = torch.cat(view_tokens, dim=1).reshape(batch_size, -1)
        return torch.cat([flattened_tokens, obs_dict["agent_pos"]], dim=-1)

    @torch.no_grad()
    def output_shape(self):
        token_dim = self.pointnet_out_dim * sum(self.view_token_counts)
        return (token_dim + self.agent_pos_dim,)
