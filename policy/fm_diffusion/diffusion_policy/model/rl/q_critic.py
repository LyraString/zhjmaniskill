from collections.abc import Iterable

import torch
import torch.nn as nn

from diffusion_policy.model.rl.value_mlp import CriticMLP


class TwinQCritic(nn.Module):
    """Twin action-value networks for clipped double-Q targets."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int] = (512, 512, 256),
        activation: str = "silu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or action_dim <= 0:
            raise ValueError("feature_dim and action_dim must be positive.")
        input_dim = int(feature_dim) + int(action_dim)
        critic_kwargs = {
            "input_dim": input_dim,
            "hidden_dims": tuple(hidden_dims),
            "activation": activation,
            "use_layer_norm": use_layer_norm,
        }
        self.q1 = CriticMLP(**critic_kwargs)
        self.q2 = CriticMLP(**critic_kwargs)

    def forward(
        self,
        obs_feature: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs_feature.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "obs_feature and action batch shapes must match, got "
                f"{tuple(obs_feature.shape)} and {tuple(action.shape)}."
            )
        critic_input = torch.cat([obs_feature, action], dim=-1)
        return self.q1(critic_input), self.q2(critic_input)

    def q1_forward(
        self,
        obs_feature: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if obs_feature.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "obs_feature and action batch shapes must match, got "
                f"{tuple(obs_feature.shape)} and {tuple(action.shape)}."
            )
        return self.q1(torch.cat([obs_feature, action], dim=-1))
