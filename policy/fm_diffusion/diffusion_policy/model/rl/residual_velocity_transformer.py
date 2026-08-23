from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


class ResidualVelocityTransformer(nn.Module):
    """Zero-initialized correction to a frozen bimanual flow velocity field."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        condition_dim: int,
        num_arms: int = 2,
        hidden_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        residual_scale: float | Sequence[float] = 0.1,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or horizon <= 0 or condition_dim <= 0:
            raise ValueError("action_dim, horizon, and condition_dim must be positive.")
        if num_arms <= 0 or action_dim % num_arms != 0:
            raise ValueError("action_dim must be divisible by the positive num_arms.")
        if hidden_dim <= 0 or depth <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim, depth, and num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")

        action_dim_per_arm = action_dim // num_arms
        self.action_dim = int(action_dim)
        self.action_dim_per_arm = int(action_dim_per_arm)
        self.horizon = int(horizon)
        self.condition_dim = int(condition_dim)
        self.num_arms = int(num_arms)
        self.hidden_dim = int(hidden_dim)

        # Each token sees both the noisy action and the frozen base velocity.
        self.action_projection = nn.Linear(2 * action_dim_per_arm, hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.action_time_embedding = nn.Embedding(horizon, hidden_dim)
        self.arm_embedding = nn.Embedding(num_arms, hidden_dim)
        self.flow_time_embedding = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.output_projection = nn.Linear(hidden_dim, action_dim_per_arm)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        scale = torch.as_tensor(residual_scale, dtype=torch.float32)
        if scale.ndim == 0:
            scale = scale.expand(action_dim).clone()
        if tuple(scale.shape) != (action_dim,):
            raise ValueError(
                f"residual_scale must be scalar or shape ({action_dim},), "
                f"got {tuple(scale.shape)}."
            )
        if not torch.isfinite(scale).all() or torch.any(scale < 0):
            raise ValueError("residual_scale must contain finite non-negative values.")
        self.register_buffer("residual_scale", scale)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        *,
        base_velocity: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if sample.ndim != 3 or tuple(sample.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"Expected sample [B, {self.horizon}, {self.action_dim}], "
                f"got {tuple(sample.shape)}."
            )
        if base_velocity.shape != sample.shape:
            raise ValueError(
                "base_velocity must match sample shape, got "
                f"{tuple(base_velocity.shape)} and {tuple(sample.shape)}."
            )
        if condition_tokens.ndim != 3:
            raise ValueError(
                "condition_tokens must have shape [B, tokens, condition_dim], got "
                f"{tuple(condition_tokens.shape)}."
            )
        if condition_tokens.shape[0] != sample.shape[0]:
            raise ValueError("Action and condition token batch sizes must match.")
        if condition_tokens.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dimension {self.condition_dim}, got "
                f"{condition_tokens.shape[-1]}."
            )

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        else:
            timesteps = timesteps.to(sample.device)
            if timesteps.ndim == 0:
                timesteps = timesteps[None]
        if timesteps.ndim != 1:
            raise ValueError(f"Expected scalar or 1D timesteps, got {tuple(timesteps.shape)}.")
        if timesteps.shape[0] == 1:
            timesteps = timesteps.expand(sample.shape[0])
        elif timesteps.shape[0] != sample.shape[0]:
            raise ValueError("Timestep batch size must be 1 or match the action batch.")

        batch_size = sample.shape[0]
        action_values = sample.reshape(
            batch_size,
            self.horizon,
            self.num_arms,
            self.action_dim_per_arm,
        )
        base_values = base_velocity.reshape_as(action_values)
        action_tokens = self.action_projection(
            torch.cat((action_values, base_values), dim=-1)
        )

        action_time_ids = torch.arange(self.horizon, device=sample.device)
        arm_ids = torch.arange(self.num_arms, device=sample.device)
        action_tokens = action_tokens + self.action_time_embedding(action_time_ids)[
            None, :, None, :
        ]
        action_tokens = action_tokens + self.arm_embedding(arm_ids)[None, None, :, :]
        action_tokens = action_tokens + self.flow_time_embedding(timesteps)[
            :, None, None, :
        ]
        action_tokens = action_tokens.reshape(batch_size, -1, self.hidden_dim)

        memory = self.condition_projection(condition_tokens)
        residual = self.output_projection(self.decoder(action_tokens, memory))
        residual = residual.reshape(batch_size, self.horizon, self.action_dim)
        scale = self.residual_scale.to(device=residual.device, dtype=residual.dtype)
        return torch.tanh(residual) * scale
