from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from diffusion_policy.model.rl.residual_actor import (
    ResidualActor,
    compose_bimanual_delta_pose_action,
)


class ResFiTPolicy(nn.Module):
    """Frozen Flow Matching policy with a trainable per-step residual actor."""

    def __init__(
        self,
        base_policy: nn.Module,
        residual_actor: ResidualActor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        super().__init__()
        if not callable(getattr(base_policy, "predict_action_with_feature", None)):
            raise TypeError(
                "base_policy must implement predict_action_with_feature(obs_dict)."
            )
        action_low = torch.as_tensor(action_low, dtype=torch.float32).flatten()
        action_high = torch.as_tensor(action_high, dtype=torch.float32).flatten()
        if action_low.shape != (14,) or action_high.shape != (14,):
            raise ValueError("ResFiTPolicy expects 14D action bounds.")
        if not torch.all(action_low < action_high):
            raise ValueError("Every action lower bound must be smaller than its upper bound.")

        self.base_policy = base_policy
        self.residual_actor = residual_actor
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self._freeze_base_policy()

    @property
    def device(self) -> torch.device:
        return self.action_low.device

    @property
    def dtype(self) -> torch.dtype:
        return self.action_low.dtype

    def _freeze_base_policy(self) -> None:
        self.base_policy.eval()
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_policy.eval()
        return self

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        *,
        generator=None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            base_result = self.base_policy.predict_action_with_feature(
                obs_dict,
                generator=generator,
            )
        base_action_sequence = base_result["action"]
        if base_action_sequence.ndim != 3 or base_action_sequence.shape[1:] != (1, 14):
            raise ValueError(
                "ResFiT v0 requires base actions with shape [B, 1, 14], got "
                f"{tuple(base_action_sequence.shape)}."
            )

        base_action = base_action_sequence[:, 0]
        obs_feature = base_result["obs_feature"].detach()
        residual_action = self.residual_actor(obs_feature, base_action)
        action = compose_bimanual_delta_pose_action(
            base_action,
            residual_action,
            self.action_low,
            self.action_high,
        )
        return {
            "action": action[:, None],
            "base_action": base_action,
            "residual_action": residual_action,
            "obs_feature": obs_feature,
        }
