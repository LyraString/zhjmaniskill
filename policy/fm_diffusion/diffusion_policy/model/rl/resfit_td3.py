from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusion_policy.model.rl.residual_actor import (
    compose_bimanual_delta_pose_action,
)


def add_bounded_residual_noise(
    residual_action: torch.Tensor,
    residual_scale: torch.Tensor,
    *,
    noise_fraction: float,
    noise_clip_fraction: float,
) -> torch.Tensor:
    """Add dimension-aware noise while preserving residual action bounds."""
    if noise_fraction < 0 or noise_clip_fraction < 0:
        raise ValueError("Noise fractions must be non-negative.")
    scale = torch.as_tensor(
        residual_scale,
        device=residual_action.device,
        dtype=residual_action.dtype,
    )
    if scale.shape != (residual_action.shape[-1],):
        raise ValueError(
            "residual_scale must match the action dimension, got "
            f"{tuple(scale.shape)} for {tuple(residual_action.shape)}."
        )
    if noise_fraction == 0 or noise_clip_fraction == 0:
        return residual_action

    noise = torch.randn(
        residual_action.shape,
        device=residual_action.device,
        dtype=residual_action.dtype,
    )
    noise = noise * scale * float(noise_fraction)
    noise_limit = scale * float(noise_clip_fraction)
    noise = torch.maximum(torch.minimum(noise, noise_limit), -noise_limit)
    noisy_residual = residual_action + noise
    return torch.maximum(torch.minimum(noisy_residual, scale), -scale)


def compute_resfit_td_target(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute a TD target where time-limit truncation remains bootstrappable."""
    if reward.shape != terminated.shape or reward.shape != next_q.shape:
        raise ValueError("reward, terminated, and next_q must have identical shapes.")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1].")
    return reward + float(gamma) * (1.0 - terminated) * next_q


@torch.no_grad()
def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    """Apply target <- (1 - tau) * target + tau * source."""
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1].")
    source_parameters = tuple(source.parameters())
    target_parameters = tuple(target.parameters())
    if len(source_parameters) != len(target_parameters):
        raise ValueError("Source and target modules must have matching parameters.")
    for source_parameter, target_parameter in zip(
        source_parameters,
        target_parameters,
        strict=True,
    ):
        target_parameter.lerp_(source_parameter, float(tau))


def resfit_td3_update(
    *,
    residual_actor: torch.nn.Module,
    critic: torch.nn.Module,
    target_residual_actor: torch.nn.Module,
    target_critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    gamma: float,
    tau: float,
    critic_update_step: int,
    policy_delay: int,
    target_noise_fraction: float,
    target_noise_clip_fraction: float,
    residual_actor_forward: torch.nn.Module | None = None,
    critic_forward: torch.nn.Module | None = None,
) -> dict[str, float | bool]:
    """Run one critic update and an optional delayed residual-actor update."""
    if critic_update_step <= 0:
        raise ValueError("critic_update_step must be positive.")
    if policy_delay <= 0:
        raise ValueError("policy_delay must be positive.")
    actor_forward = (
        residual_actor if residual_actor_forward is None else residual_actor_forward
    )
    train_critic_forward = critic if critic_forward is None else critic_forward

    with torch.no_grad():
        next_residual = target_residual_actor(
            batch["next_obs_feature"],
            batch["next_base_action"],
        )
        next_residual = add_bounded_residual_noise(
            next_residual,
            target_residual_actor.residual_scale,
            noise_fraction=target_noise_fraction,
            noise_clip_fraction=target_noise_clip_fraction,
        )
        next_action = compose_bimanual_delta_pose_action(
            batch["next_base_action"],
            next_residual,
            action_low,
            action_high,
        )
        target_q1, target_q2 = target_critic(
            batch["next_obs_feature"],
            next_action,
        )
        target_q = compute_resfit_td_target(
            batch["reward"],
            batch["terminated"],
            torch.minimum(target_q1, target_q2),
            gamma,
        )

    q1, q2 = train_critic_forward(batch["obs_feature"], batch["action"])
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    if not torch.isfinite(critic_loss):
        raise RuntimeError("ResFiT critic loss is NaN or Inf.")
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()
    soft_update(critic, target_critic, tau)

    actor_updated = critic_update_step % policy_delay == 0
    actor_loss_value = float("nan")
    actor_residual_abs_max = 0.0
    if actor_updated:
        critic_parameters = tuple(critic.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            actor_residual = actor_forward(
                batch["obs_feature"],
                batch["base_action"],
            )
            actor_action = compose_bimanual_delta_pose_action(
                batch["base_action"],
                actor_residual,
                action_low,
                action_high,
            )
            actor_loss = -critic.q1_forward(
                batch["obs_feature"],
                actor_action,
            ).mean()
            if not torch.isfinite(actor_loss):
                raise RuntimeError("ResFiT actor loss is NaN or Inf.")
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)
        soft_update(residual_actor, target_residual_actor, tau)
        actor_loss_value = float(actor_loss.detach().item())
        actor_residual_abs_max = float(actor_residual.detach().abs().amax().item())

    return {
        "critic_loss": float(critic_loss.detach().item()),
        "q1_mean": float(q1.detach().mean().item()),
        "q2_mean": float(q2.detach().mean().item()),
        "target_q_mean": float(target_q.detach().mean().item()),
        "actor_updated": actor_updated,
        "actor_loss": actor_loss_value,
        "actor_residual_abs_max": actor_residual_abs_max,
        "target_action_abs_max": float(next_action.detach().abs().amax().item()),
    }
