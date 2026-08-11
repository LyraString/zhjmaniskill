from collections.abc import Iterable

import torch
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles


_ARM_ACTION_DIM = 7
_BIMANUAL_ACTION_DIM = 14


def compose_bimanual_delta_pose_action(
    base_action: torch.Tensor,
    residual_action: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    """Compose a residual with a world/root-aligned bimanual delta-pose action."""
    if base_action.shape != residual_action.shape:
        raise ValueError(
            "base_action and residual_action must have identical shapes, got "
            f"{tuple(base_action.shape)} and {tuple(residual_action.shape)}."
        )
    if base_action.shape[-1] != _BIMANUAL_ACTION_DIM:
        raise ValueError(
            f"Expected { _BIMANUAL_ACTION_DIM } action values, got "
            f"{base_action.shape[-1]}."
        )
    low = torch.as_tensor(action_low, device=base_action.device, dtype=base_action.dtype)
    high = torch.as_tensor(action_high, device=base_action.device, dtype=base_action.dtype)
    if low.shape != (_BIMANUAL_ACTION_DIM,) or high.shape != (_BIMANUAL_ACTION_DIM,):
        raise ValueError("action_low and action_high must both have shape (14,).")

    composed = base_action.clone()
    for arm_offset in (0, _ARM_ACTION_DIM):
        pos_slice = slice(arm_offset, arm_offset + 3)
        rot_slice = slice(arm_offset + 3, arm_offset + 6)
        gripper_index = arm_offset + 6

        composed[..., pos_slice] = (
            base_action[..., pos_slice] + residual_action[..., pos_slice]
        )

        base_rotation = base_action[..., rot_slice]
        residual_rotation = residual_action[..., rot_slice]
        rotation_matrix = (
            euler_angles_to_matrix(residual_rotation.float(), convention="XYZ")
            @ euler_angles_to_matrix(base_rotation.float(), convention="XYZ")
        )
        composed_rotation = matrix_to_euler_angles(
            rotation_matrix,
            convention="XYZ",
        ).to(base_action.dtype)
        zero_residual = torch.all(residual_rotation == 0, dim=-1, keepdim=True)
        exact_identity = composed_rotation + (base_rotation - composed_rotation).detach()
        composed[..., rot_slice] = torch.where(
            zero_residual,
            exact_identity,
            composed_rotation,
        )

        composed[..., gripper_index] = (
            base_action[..., gripper_index] + residual_action[..., gripper_index]
        )

    return torch.maximum(torch.minimum(composed, high), low)


class ResidualActor(nn.Module):
    """Small zero-initialized actor that predicts bounded action corrections."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int = _BIMANUAL_ACTION_DIM,
        hidden_dims: Iterable[int] = (256, 256),
        residual_scale: Iterable[float] | torch.Tensor | None = None,
        activation: str = "silu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if action_dim != _BIMANUAL_ACTION_DIM:
            raise ValueError("ResFiT v0 expects the 14D bimanual delta-pose action.")

        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes.")

        activation_cls = self._activation_class(activation)
        layers: list[nn.Module] = []
        input_dim = int(feature_dim) + int(action_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_cls())
            input_dim = hidden_dim
        output_layer = nn.Linear(input_dim, action_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        if residual_scale is None:
            raise ValueError(
                "residual_scale must be set explicitly for the robot action semantics."
            )
        scale = torch.as_tensor(residual_scale, dtype=torch.float32)
        if scale.shape != (action_dim,):
            raise ValueError(f"residual_scale must have shape ({action_dim},).")
        if not torch.isfinite(scale).all() or torch.any(scale < 0):
            raise ValueError("residual_scale must contain finite non-negative values.")

        self.network = nn.Sequential(*layers)
        self.register_buffer("residual_scale", scale)
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)

    def forward(
        self,
        obs_feature: torch.Tensor,
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        if obs_feature.shape[:-1] != base_action.shape[:-1]:
            raise ValueError(
                "obs_feature and base_action batch shapes must match, got "
                f"{tuple(obs_feature.shape)} and {tuple(base_action.shape)}."
            )
        if base_action.shape[-1] != self.residual_scale.numel():
            raise ValueError(
                f"Expected base action dimension {self.residual_scale.numel()}, got "
                f"{base_action.shape[-1]}."
            )
        actor_input = torch.cat([obs_feature, base_action], dim=-1).float()
        return torch.tanh(self.network(actor_input)) * self.residual_scale

    @staticmethod
    def _activation_class(name: str) -> type[nn.Module]:
        activation_classes = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        try:
            return activation_classes[name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported activation: {name}") from exc
