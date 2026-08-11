from __future__ import annotations

import copy
import pathlib
import random

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.online_replay_buffer import OnlineReplayBuffer
from diffusion_policy.env_runner.maniskill_cuda_fpo_runner import ManiSkillCudaFPORunner
from diffusion_policy.model.rl.resfit_td3 import (
    add_bounded_residual_noise,
    resfit_td3_update,
)
from diffusion_policy.model.rl.residual_actor import (
    compose_bimanual_delta_pose_action,
)
from diffusion_policy.policy.resfit_policy import ResFiTPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class ResFiTWorkspace(BaseWorkspace):
    """ResFiT workspace with identity and one-batch learner smoke phases."""

    include_keys = ("global_step", "epoch", "critic_update_step")

    def __init__(self, cfg: OmegaConf, output_dir=None) -> None:
        super().__init__(cfg, output_dir=output_dir)
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Keep these names aligned with Flow Matching checkpoint payload keys.
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model)
        self.residual_actor = hydra.utils.instantiate(cfg.residual_actor)
        self.critic = hydra.utils.instantiate(cfg.critic)
        self.target_residual_actor = copy.deepcopy(self.residual_actor)
        self.target_critic = copy.deepcopy(self.critic)
        for target_module in (self.target_residual_actor, self.target_critic):
            for parameter in target_module.parameters():
                parameter.requires_grad_(False)
        self.actor_optimizer = hydra.utils.instantiate(
            cfg.actor_optimizer,
            params=self.residual_actor.parameters(),
        )
        self.critic_optimizer = hydra.utils.instantiate(
            cfg.critic_optimizer,
            params=self.critic.parameters(),
        )
        self.global_step = 0
        self.epoch = 0
        self.critic_update_step = 0

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        device = torch.device(str(cfg.training.device))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The ManiSkill ResFiT identity rollout requires CUDA.")
        torch.cuda.set_device(device)

        checkpoint_path = pathlib.Path(str(cfg.training.base_checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Base Flow Matching checkpoint not found: {checkpoint_path}")
        payload = self.load_checkpoint(
            path=checkpoint_path,
            exclude_keys=("optimizer",),
            include_keys=(),
        )
        self._validate_base_checkpoint(payload, cfg)

        source = str(cfg.training.base_policy_source).lower()
        if source == "ema_model":
            if "ema_model" not in payload.get("state_dicts", {}):
                raise RuntimeError("Requested EMA base policy, but checkpoint has no ema_model.")
            base_policy = self.ema_model
        elif source == "model":
            base_policy = self.model
        else:
            raise ValueError("training.base_policy_source must be `model` or `ema_model`.")
        base_policy.to(device).eval()
        self.residual_actor.to(device)
        self.critic.to(device)
        self.target_residual_actor.to(device).eval()
        self.target_critic.to(device).eval()

        task_cfg = cfg.task.maniskill
        runner = ManiSkillCudaFPORunner(
            task_name=str(cfg.task.name),
            shape_meta=OmegaConf.to_container(cfg.shape_meta, resolve=True),
            task_profile=str(task_cfg.profile),
            env_id=str(task_cfg.env_id),
            num_envs=int(cfg.training.num_envs),
            base_seed=int(cfg.training.seed),
            cuda_device_index=device.index,
            action_type=str(task_cfg.action_type),
            obs_mode=str(task_cfg.obs_mode),
            control_mode=str(task_cfg.control_mode),
            render_mode=task_cfg.render_mode,
            reward_mode=str(task_cfg.reward_mode),
            robot_uids=task_cfg.robot_uids,
            sim_backend=cfg.training.sim_backend,
            render_backend=cfg.training.render_backend,
            sim_freq=int(task_cfg.sim_freq),
            control_freq=int(task_cfg.control_freq),
            point_cloud_sampling=str(cfg.training.point_cloud_sampling),
        )
        if runner.control_mode != "bimanual_pd_ee_delta_pose":
            runner.close()
            raise RuntimeError(
                "ResFiT v0 requires control_mode=bimanual_pd_ee_delta_pose, got "
                f"{runner.control_mode}."
            )

        try:
            policy = ResFiTPolicy(
                base_policy=base_policy,
                residual_actor=self.residual_actor,
                action_low=runner.action_low,
                action_high=runner.action_high,
            ).to(device).eval()
            identity_summary = self._run_identity_rollout(policy, runner, cfg)
            learner_summary = self._run_learner_smoke(policy, runner, cfg, device)
        finally:
            runner.close()

        print(
            "[ResFiT identity] "
            f"checkpoint={checkpoint_path} source={source} "
            f"steps={identity_summary['steps']} envs={identity_summary['num_envs']} "
            f"feature_shape={identity_summary['feature_shape']} "
            f"residual_abs_max={identity_summary['residual_abs_max']:.3e} "
            f"identity_error_max={identity_summary['identity_error_max']:.3e} "
            f"action_abs_max={identity_summary['action_abs_max']:.6f}"
        )
        print(
            "[ResFiT learner smoke] "
            f"transitions={learner_summary['transitions']} "
            f"updates={learner_summary['updates']} "
            f"reward_mean={learner_summary['reward_mean']:.6f} "
            f"critic_loss={learner_summary['critic_loss']:.6f} "
            f"actor_loss={learner_summary['actor_loss']:.6f} "
            f"actor_change_max={learner_summary['actor_parameter_change_max']:.3e} "
            f"terminated={learner_summary['terminated_count']} "
            f"truncated={learner_summary['truncated_count']}"
        )

    @staticmethod
    def _validate_base_checkpoint(payload: dict, cfg: OmegaConf) -> None:
        checkpoint_cfg = payload.get("cfg")
        if checkpoint_cfg is None:
            raise RuntimeError("Base checkpoint has no saved Hydra configuration.")

        expected = {
            "n_obs_steps": int(cfg.n_obs_steps),
            "horizon": int(cfg.horizon),
            "action_dim": int(cfg.shape_meta.action.shape[0]),
            "control_mode": "bimanual_pd_ee_delta_pose",
            "action_semantics": "delta_pose_actual",
            "rotation_representation": "euler_xyz",
            "translation_frame": "root_world_aligned",
            "gripper_semantics": "absolute",
            "control_freq": 20,
        }
        actual = {
            "n_obs_steps": int(checkpoint_cfg.n_obs_steps),
            "horizon": int(checkpoint_cfg.horizon),
            "action_dim": int(checkpoint_cfg.shape_meta.action.shape[0]),
            "control_mode": str(checkpoint_cfg.task.maniskill.control_mode),
            "action_semantics": str(checkpoint_cfg.task.maniskill.action_semantics),
            "rotation_representation": str(
                checkpoint_cfg.task.maniskill.rotation_representation
            ),
            "translation_frame": str(checkpoint_cfg.task.maniskill.translation_frame),
            "gripper_semantics": str(checkpoint_cfg.task.maniskill.gripper_semantics),
            "control_freq": int(checkpoint_cfg.task.maniskill.control_freq),
        }
        mismatches = {
            key: (expected_value, actual[key])
            for key, expected_value in expected.items()
            if actual[key] != expected_value
        }
        if str(checkpoint_cfg.policy.denoiser_type) != "action_dit":
            mismatches["denoiser_type"] = (
                "action_dit",
                str(checkpoint_cfg.policy.denoiser_type),
            )
        checkpoint_shape_meta = OmegaConf.to_container(
            checkpoint_cfg.shape_meta,
            resolve=True,
        )
        expected_shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
        if checkpoint_shape_meta != expected_shape_meta:
            mismatches["shape_meta"] = (expected_shape_meta, checkpoint_shape_meta)
        if mismatches:
            raise RuntimeError(f"Base checkpoint semantics mismatch: {mismatches}")

        source_state = payload.get("state_dicts", {}).get(
            str(cfg.training.base_policy_source),
            {},
        )
        if not any(str(key).startswith("normalizer.") for key in source_state):
            raise RuntimeError("Selected base policy checkpoint has no saved normalizer.")

    def _run_identity_rollout(self, policy, runner, cfg) -> dict:
        steps = int(cfg.training.identity_rollout_steps)
        if steps <= 0:
            raise ValueError("training.identity_rollout_steps must be positive.")

        obs, _ = runner.reset()
        obs_history = self._init_obs_history(obs, int(policy.base_policy.n_obs_steps))
        residual_abs_max = 0.0
        identity_error_max = 0.0
        action_abs_max = 0.0
        feature_shape = None

        for _ in range(steps):
            with torch.inference_mode():
                result = policy.predict_action(obs_history)
            base_action = result["base_action"]
            residual_action = result["residual_action"]
            action = result["action"][:, 0]
            expected_action = runner.clip_actions(base_action)

            if not all(
                torch.isfinite(value).all()
                for value in (base_action, residual_action, action, result["obs_feature"])
            ):
                raise RuntimeError("ResFiT identity rollout produced NaN or Inf values.")

            residual_max = float(residual_action.abs().amax().item())
            identity_error = float((action - expected_action).abs().amax().item())
            if residual_max != 0.0:
                raise RuntimeError(
                    f"Zero-initialized residual actor produced non-zero output: {residual_max}."
                )
            if identity_error != 0.0:
                raise RuntimeError(
                    "Zero-residual ResFiT action differs from the clipped base action: "
                    f"max error={identity_error}."
                )

            residual_abs_max = max(residual_abs_max, residual_max)
            identity_error_max = max(identity_error_max, identity_error)
            action_abs_max = max(action_abs_max, float(action.abs().amax().item()))
            feature_shape = tuple(result["obs_feature"].shape)

            next_obs, _, terminated, truncated, _ = runner.step(action)
            obs_history = self._append_obs_history(obs_history, next_obs)
            done_ids = torch.logical_or(terminated, truncated).nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                reset_obs, _ = runner.reset_at(done_ids.detach().cpu().tolist())
                self._reset_obs_history(obs_history, reset_obs, done_ids)

        expected_feature_shape = (
            int(cfg.training.num_envs),
            int(cfg.residual_actor.feature_dim),
        )
        if feature_shape != expected_feature_shape:
            raise RuntimeError(
                f"Expected pooled feature shape {expected_feature_shape}, got {feature_shape}."
            )
        return {
            "steps": steps,
            "num_envs": int(cfg.training.num_envs),
            "feature_shape": feature_shape,
            "residual_abs_max": residual_abs_max,
            "identity_error_max": identity_error_max,
            "action_abs_max": action_abs_max,
        }

    def _run_learner_smoke(self, policy, runner, cfg, device: torch.device) -> dict:
        replay = OnlineReplayBuffer(
            capacity=int(cfg.training.replay_capacity),
            feature_dim=int(cfg.residual_actor.feature_dim),
            action_dim=int(cfg.residual_actor.action_dim),
            device=str(cfg.training.replay_device),
        )
        rollout_summary = self._collect_online_transitions(
            policy,
            runner,
            replay,
            cfg,
        )

        batch_size = int(cfg.training.batch_size)
        if len(replay) < batch_size:
            raise RuntimeError(
                f"Learner smoke collected {len(replay)} transitions, below "
                f"batch_size={batch_size}."
            )
        updates = int(cfg.training.learner_updates)
        if updates <= 0:
            raise ValueError("training.learner_updates must be positive.")

        actor_before = tuple(
            parameter.detach().clone()
            for parameter in self.residual_actor.parameters()
        )
        policy.train()
        self.critic.train()
        last_update = None
        actor_updated = False
        for _ in range(updates):
            self.critic_update_step += 1
            batch = replay.sample(batch_size, device=device)
            last_update = resfit_td3_update(
                residual_actor=self.residual_actor,
                critic=self.critic,
                target_residual_actor=self.target_residual_actor,
                target_critic=self.target_critic,
                actor_optimizer=self.actor_optimizer,
                critic_optimizer=self.critic_optimizer,
                batch=batch,
                action_low=policy.action_low,
                action_high=policy.action_high,
                gamma=float(cfg.training.gamma),
                tau=float(cfg.training.tau),
                critic_update_step=self.critic_update_step,
                policy_delay=int(cfg.training.policy_delay),
                target_noise_fraction=float(cfg.training.target_noise_fraction),
                target_noise_clip_fraction=float(
                    cfg.training.target_noise_clip_fraction
                ),
            )
            actor_updated = actor_updated or bool(last_update["actor_updated"])

        if last_update is None:
            raise RuntimeError("ResFiT learner smoke did not execute an update.")
        if not actor_updated:
            raise RuntimeError(
                "Learner smoke did not reach a delayed actor update; increase "
                "training.learner_updates or reduce training.policy_delay."
            )
        actor_change_max = max(
            float((after.detach() - before).abs().amax().item())
            for before, after in zip(
                actor_before,
                self.residual_actor.parameters(),
                strict=True,
            )
        )
        if actor_change_max <= 0:
            raise RuntimeError("Residual actor parameters did not change after its update.")
        if any(parameter.grad is not None for parameter in policy.base_policy.parameters()):
            raise RuntimeError("Frozen base policy received gradients during ResFiT update.")

        return {
            **rollout_summary,
            **last_update,
            "updates": updates,
            "actor_parameter_change_max": actor_change_max,
        }

    def _collect_online_transitions(self, policy, runner, replay, cfg) -> dict:
        rollout_steps = int(cfg.training.learner_rollout_steps)
        if rollout_steps <= 0:
            raise ValueError("training.learner_rollout_steps must be positive.")

        obs, _ = runner.reset()
        obs_history = self._init_obs_history(obs, int(policy.base_policy.n_obs_steps))
        policy.eval()
        with torch.inference_mode():
            current_result = policy.predict_action(obs_history)

        reward_sum = 0.0
        terminated_count = 0
        truncated_count = 0
        residual_abs_max = 0.0
        action_abs_max = 0.0
        for _ in range(rollout_steps):
            with torch.inference_mode():
                residual_action = add_bounded_residual_noise(
                    current_result["residual_action"],
                    policy.residual_actor.residual_scale,
                    noise_fraction=float(cfg.training.exploration_noise_fraction),
                    noise_clip_fraction=float(
                        cfg.training.exploration_noise_clip_fraction
                    ),
                )
                action = compose_bimanual_delta_pose_action(
                    current_result["base_action"],
                    residual_action,
                    policy.action_low,
                    policy.action_high,
                )
            if not torch.isfinite(action).all():
                raise RuntimeError("ResFiT rollout action contains NaN or Inf values.")
            if torch.any(action < policy.action_low) or torch.any(action > policy.action_high):
                raise RuntimeError("ResFiT rollout action exceeded environment bounds.")

            next_obs, reward, terminated, truncated, _ = runner.step(action)
            next_obs_history = self._append_obs_history(obs_history, next_obs)
            with torch.inference_mode():
                next_result = policy.predict_action(next_obs_history)

            replay.add_batch(
                obs_feature=current_result["obs_feature"].detach(),
                base_action=current_result["base_action"].detach(),
                action=action.detach(),
                reward=reward.detach().to(dtype=torch.float32),
                next_obs_feature=next_result["obs_feature"].detach(),
                next_base_action=next_result["base_action"].detach(),
                terminated=terminated.detach().to(dtype=torch.float32),
            )

            reward_sum += float(reward.detach().sum().item())
            terminated_count += int(terminated.detach().sum().item())
            truncated_count += int(truncated.detach().sum().item())
            residual_abs_max = max(
                residual_abs_max,
                float(residual_action.detach().abs().amax().item()),
            )
            action_abs_max = max(
                action_abs_max,
                float(action.detach().abs().amax().item()),
            )

            done_ids = torch.logical_or(terminated, truncated).nonzero(
                as_tuple=False
            ).flatten()
            if done_ids.numel() > 0:
                reset_obs, _ = runner.reset_at(done_ids.detach().cpu().tolist())
                self._reset_obs_history(next_obs_history, reset_obs, done_ids)
                with torch.inference_mode():
                    current_result = policy.predict_action(next_obs_history)
            else:
                current_result = next_result
            obs_history = next_obs_history

        transitions = rollout_steps * int(cfg.training.num_envs)
        return {
            "transitions": transitions,
            "reward_mean": reward_sum / transitions,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "rollout_residual_abs_max": residual_abs_max,
            "rollout_action_abs_max": action_abs_max,
        }

    @staticmethod
    def _init_obs_history(obs: dict[str, torch.Tensor], n_steps: int) -> dict[str, torch.Tensor]:
        return {
            key: value.unsqueeze(1).expand(value.shape[0], n_steps, *value.shape[1:]).clone()
            for key, value in obs.items()
        }

    @staticmethod
    def _append_obs_history(
        obs_history: dict[str, torch.Tensor],
        obs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: torch.cat((value[:, 1:], obs[key].unsqueeze(1)), dim=1)
            for key, value in obs_history.items()
        }

    @staticmethod
    def _reset_obs_history(
        obs_history: dict[str, torch.Tensor],
        obs: dict[str, torch.Tensor],
        env_ids: torch.Tensor,
    ) -> None:
        for key, value in obs_history.items():
            reset_value = obs[key][env_ids].unsqueeze(1).expand(
                -1,
                value.shape[1],
                *value.shape[2:],
            )
            value[env_ids] = reset_value
