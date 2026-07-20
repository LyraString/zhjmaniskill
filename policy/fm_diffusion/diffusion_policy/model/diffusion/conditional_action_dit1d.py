from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class ActionDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)

        self.self_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim),
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.time_modulation(time_embedding).chunk(9, dim=-1)

        self_input = _modulate(self.self_norm(action_tokens), shift_attn, scale_attn)
        self_output = self.self_attn(
            self_input,
            self_input,
            self_input,
            need_weights=False,
        )[0]
        action_tokens = action_tokens + gate_attn[:, None, :] * self_output

        cross_input = _modulate(
            self.cross_norm(action_tokens),
            shift_cross,
            scale_cross,
        )
        cross_output = self.cross_attn(
            cross_input,
            condition_tokens,
            condition_tokens,
            need_weights=False,
        )[0]
        action_tokens = action_tokens + gate_cross[:, None, :] * cross_output

        mlp_input = _modulate(self.mlp_norm(action_tokens), shift_mlp, scale_mlp)
        action_tokens = action_tokens + gate_mlp[:, None, :] * self.mlp(mlp_input)
        return action_tokens


class ConditionTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.attn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(condition_tokens)
        condition_tokens = condition_tokens + self.attention(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
        )[0]
        return condition_tokens + self.mlp(self.mlp_norm(condition_tokens))


class ConditionalBimanualActionDiT1D(nn.Module):
    """Action DiT for two-arm trajectories with structured observation tokens.

    Actions are ordered per step as ``[left_arm, right_arm]``. Each observation
    frame is flattened as ``[view_0, ..., view_n, left_proprio, right_proprio]``;
    this module restores those fixed boundaries before applying attention.
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        n_obs_steps: int,
        obs_feature_dim: int,
        view_feature_dim: int = 512,
        num_views: int = 3,
        view_tokens_per_view: int = 1,
        view_token_counts: tuple[int, ...] | None = None,
        use_spatial_token_embedding: bool = True,
        num_arms: int = 2,
        proprio_dim_per_arm: int = 7,
        condition_depth: int = 0,
        condition_num_heads: int | None = None,
        hidden_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            num_views <= 0
            or view_tokens_per_view <= 0
            or num_arms <= 0
            or proprio_dim_per_arm <= 0
        ):
            raise ValueError("View, token, arm, and proprioception dimensions must be positive.")
        if hidden_dim <= 0 or num_heads <= 0:
            raise ValueError("Hidden dimension and attention heads must be positive.")
        if input_dim % num_arms != 0:
            raise ValueError(
                f"Action dimension {input_dim} is not divisible by {num_arms} arms."
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )
        if depth <= 0:
            raise ValueError("Action DiT depth must be positive.")
        if condition_depth < 0:
            raise ValueError("Condition Transformer depth cannot be negative.")
        if horizon <= 0 or n_obs_steps <= 0:
            raise ValueError("Action horizon and observation steps must be positive.")

        if condition_num_heads is None:
            condition_num_heads = num_heads
        if condition_num_heads <= 0 or hidden_dim % condition_num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by the positive condition_num_heads."
            )

        if view_token_counts is None:
            resolved_view_token_counts = (view_tokens_per_view,) * num_views
        else:
            resolved_view_token_counts = tuple(int(count) for count in view_token_counts)
            if len(resolved_view_token_counts) != num_views:
                raise ValueError(
                    f"Expected {num_views} view token counts, got "
                    f"{len(resolved_view_token_counts)}."
                )
            if any(count <= 0 for count in resolved_view_token_counts):
                raise ValueError("Every view token count must be positive.")

        action_dim_per_arm = input_dim // num_arms
        expected_obs_feature_dim = (
            sum(resolved_view_token_counts) * view_feature_dim
            + num_arms * proprio_dim_per_arm
        )
        if obs_feature_dim != expected_obs_feature_dim:
            raise ValueError(
                "Token observation layout mismatch: expected per-frame feature dimension "
                f"{expected_obs_feature_dim}, got {obs_feature_dim}."
            )

        self.input_dim = input_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.view_feature_dim = view_feature_dim
        self.num_views = num_views
        self.view_tokens_per_view = view_tokens_per_view
        self.view_token_counts = resolved_view_token_counts
        self.num_arms = num_arms
        self.action_dim_per_arm = action_dim_per_arm
        self.proprio_dim_per_arm = proprio_dim_per_arm
        self.hidden_dim = hidden_dim

        self.action_projection = nn.Linear(action_dim_per_arm, hidden_dim)
        self.view_projection = nn.Linear(view_feature_dim, hidden_dim)
        self.proprio_projection = nn.Linear(proprio_dim_per_arm, hidden_dim)

        self.action_time_embedding = nn.Embedding(horizon, hidden_dim)
        self.obs_time_embedding = nn.Embedding(n_obs_steps, hidden_dim)
        self.view_embedding = nn.Embedding(num_views, hidden_dim)
        self.spatial_token_embedding = (
            nn.Embedding(max(resolved_view_token_counts), hidden_dim)
            if use_spatial_token_embedding
            else None
        )
        self.arm_embedding = nn.Embedding(num_arms, hidden_dim)
        self.token_type_embedding = nn.Embedding(3, hidden_dim)

        view_ids = []
        spatial_token_ids = []
        for view_index, token_count in enumerate(resolved_view_token_counts):
            view_ids.extend([view_index] * token_count)
            spatial_token_ids.extend(range(token_count))
        self.register_buffer(
            "_view_ids",
            torch.tensor(view_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_spatial_token_ids",
            torch.tensor(spatial_token_ids, dtype=torch.long),
            persistent=False,
        )

        self.flow_time_embedding = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.condition_blocks = nn.ModuleList(
            [
                ConditionTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=condition_num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(condition_depth)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                ActionDiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.output_projection = nn.Linear(hidden_dim, action_dim_per_arm)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(init_module)
        for block in self.blocks:
            nn.init.zeros_(block.time_modulation[-1].weight)
            nn.init.zeros_(block.time_modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _build_condition_tokens(self, global_cond: torch.Tensor) -> torch.Tensor:
        expected_dim = self.n_obs_steps * self.obs_feature_dim
        if global_cond.ndim != 2:
            raise ValueError(
                "Action DiT requires flattened structured observations with shape "
                f"[B, {expected_dim}], got {tuple(global_cond.shape)}."
            )
        batch_size = global_cond.shape[0]
        if global_cond.shape[1] != expected_dim:
            raise ValueError(
                "Action DiT requires flattened structured observations with shape "
                f"[B, {expected_dim}], got {tuple(global_cond.shape)}."
            )

        frame_features = global_cond.reshape(
            batch_size,
            self.n_obs_steps,
            self.obs_feature_dim,
        )
        total_view_tokens = sum(self.view_token_counts)
        view_feature_size = total_view_tokens * self.view_feature_dim
        view_values = frame_features[..., :view_feature_size]
        view_values = view_values.reshape(
            batch_size,
            self.n_obs_steps,
            total_view_tokens,
            self.view_feature_dim,
        )
        proprio_values = frame_features[..., view_feature_size:]
        proprio_values = proprio_values.reshape(
            batch_size,
            self.n_obs_steps,
            self.num_arms,
            self.proprio_dim_per_arm,
        )

        obs_time_ids = torch.arange(self.n_obs_steps, device=global_cond.device)
        arm_ids = torch.arange(self.num_arms, device=global_cond.device)

        view_tokens = self.view_projection(view_values)
        view_tokens = view_tokens + self.obs_time_embedding(obs_time_ids)[None, :, None, :]
        view_tokens = view_tokens + self.view_embedding(self._view_ids)[None, None, :, :]
        if self.spatial_token_embedding is not None:
            view_tokens = view_tokens + self.spatial_token_embedding(
                self._spatial_token_ids
            )[None, None, :, :]
        view_tokens = view_tokens + self.token_type_embedding.weight[0][None, None, None, :]

        proprio_tokens = self.proprio_projection(proprio_values)
        proprio_tokens = proprio_tokens + self.obs_time_embedding(obs_time_ids)[None, :, None, :]
        proprio_tokens = proprio_tokens + self.arm_embedding(arm_ids)[None, None, :, :]
        proprio_tokens = proprio_tokens + self.token_type_embedding.weight[1][None, None, None, :]

        condition_tokens = torch.cat([view_tokens, proprio_tokens], dim=2)
        condition_tokens = condition_tokens.reshape(batch_size, -1, self.hidden_dim)
        for block in self.condition_blocks:
            condition_tokens = block(condition_tokens)
        return self.condition_norm(condition_tokens)

    def encode_condition_tokens(self, global_cond: torch.Tensor) -> torch.Tensor:
        """Encode observation tokens once for reuse across Flow ODE steps."""
        return self._build_condition_tokens(global_cond)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond: torch.Tensor | None = None,
        condition_tokens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if local_cond is not None:
            raise ValueError("ConditionalBimanualActionDiT1D does not use local_cond.")
        if global_cond is None:
            raise ValueError("ConditionalBimanualActionDiT1D requires global_cond.")
        if sample.ndim != 3 or sample.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected action sample [B, T, {self.input_dim}], got {tuple(sample.shape)}."
            )
        if sample.shape[1] != self.horizon:
            raise ValueError(
                f"Expected action horizon {self.horizon}, got {sample.shape[1]}."
            )
        if global_cond.ndim != 2:
            raise ValueError(
                "Action DiT requires 2D flattened structured observations, got "
                f"{tuple(global_cond.shape)}."
            )
        if global_cond.shape[0] != sample.shape[0]:
            raise ValueError(
                "Action and observation batch sizes must match, got "
                f"{sample.shape[0]} and {global_cond.shape[0]}."
            )
        if global_cond.device != sample.device:
            raise ValueError(
                "Action and observation tensors must share a device, got "
                f"{sample.device} and {global_cond.device}."
            )

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        else:
            timesteps = timesteps.to(sample.device)
            if timesteps.ndim == 0:
                timesteps = timesteps[None]
        if timesteps.ndim != 1:
            raise ValueError(
                f"Expected scalar or 1D timesteps, got {tuple(timesteps.shape)}."
            )
        if timesteps.shape[0] == 1:
            timesteps = timesteps.expand(sample.shape[0])
        elif timesteps.shape[0] != sample.shape[0]:
            raise ValueError(
                "Timestep batch size must be 1 or match the action batch, got "
                f"{timesteps.shape[0]} and {sample.shape[0]}."
            )
        time_embedding = self.flow_time_embedding(timesteps)

        batch_size = sample.shape[0]
        action_values = sample.reshape(
            batch_size,
            self.horizon,
            self.num_arms,
            self.action_dim_per_arm,
        )
        action_tokens = self.action_projection(action_values)

        action_time_ids = torch.arange(self.horizon, device=sample.device)
        arm_ids = torch.arange(self.num_arms, device=sample.device)
        action_tokens = action_tokens + self.action_time_embedding(action_time_ids)[None, :, None, :]
        action_tokens = action_tokens + self.arm_embedding(arm_ids)[None, None, :, :]
        action_tokens = action_tokens + self.token_type_embedding.weight[2][None, None, None, :]
        action_tokens = action_tokens.reshape(batch_size, -1, self.hidden_dim)

        if condition_tokens is None:
            condition_tokens = self.encode_condition_tokens(global_cond)
        else:
            expected_tokens = self.n_obs_steps * (
                sum(self.view_token_counts) + self.num_arms
            )
            expected_shape = (batch_size, expected_tokens, self.hidden_dim)
            if tuple(condition_tokens.shape) != expected_shape:
                raise ValueError(
                    "Precomputed Action DiT condition tokens must have shape "
                    f"{expected_shape}, got {tuple(condition_tokens.shape)}."
                )
            if condition_tokens.device != sample.device:
                raise ValueError(
                    "Precomputed condition tokens and action samples must share a device, "
                    f"got {condition_tokens.device} and {sample.device}."
                )
        for block in self.blocks:
            action_tokens = block(action_tokens, condition_tokens, time_embedding)

        shift, scale = self.final_modulation(time_embedding).chunk(2, dim=-1)
        action_tokens = _modulate(self.final_norm(action_tokens), shift, scale)
        output = self.output_projection(action_tokens)
        return output.reshape(batch_size, self.horizon, self.input_dim)
