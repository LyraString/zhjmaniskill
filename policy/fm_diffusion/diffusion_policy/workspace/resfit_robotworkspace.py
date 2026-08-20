from __future__ import annotations

import contextlib
import copy
import os
import pathlib
import random
import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.online_replay_buffer import OnlineReplayBuffer
from diffusion_policy.common.pytorch_util import optimizer_to
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


def _distributed_context() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class ResFiTWorkspace(BaseWorkspace):
    """Online residual fine-tuning workspace for a frozen Flow Matching policy."""

    supports_distributed = True
    include_keys = ("global_step", "epoch", "critic_update_step")
    exclude_keys = ("model", "ema_model")

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
        rank, world_size = _distributed_context()
        distributed = world_size > 1
        is_main_process = rank == 0
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

        resume = bool(cfg.training.resume)
        if resume:
            resume_path = cfg.training.resume_checkpoint
            if resume_path in (None, ""):
                raise ValueError(
                    "training.resume_checkpoint is required when training.resume=True."
                )
            resume_path = pathlib.Path(str(resume_path)).expanduser()
            if not resume_path.is_file():
                raise FileNotFoundError(f"ResFiT checkpoint not found: {resume_path}")
            resume_payload = self.load_checkpoint(
                path=resume_path,
                exclude_keys=self.exclude_keys,
                include_keys=self.include_keys,
            )
            self._validate_resume_checkpoint(resume_payload, cfg)
            optimizer_to(self.actor_optimizer, device)
            optimizer_to(self.critic_optimizer, device)
            self.target_residual_actor.eval()
            self.target_critic.eval()

        actor_forward = self.residual_actor
        critic_forward = self.critic
        if distributed:
            if device.index is None:
                raise RuntimeError(
                    f"Distributed ResFiT requires an explicit CUDA device, got {device}."
                )
            actor_forward = DistributedDataParallel(
                self.residual_actor,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            critic_forward = DistributedDataParallel(
                self.critic,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            self._broadcast_module(self.target_residual_actor)
            self._broadcast_module(self.target_critic)

        process_seed = int(cfg.training.seed) + rank
        torch.manual_seed(process_seed)
        torch.cuda.manual_seed(process_seed)
        np.random.seed(process_seed)
        random.seed(process_seed)

        task_cfg = cfg.task.maniskill
        runner = ManiSkillCudaFPORunner(
            task_name=str(cfg.task.name),
            shape_meta=OmegaConf.to_container(cfg.shape_meta, resolve=True),
            task_profile=str(task_cfg.profile),
            env_id=str(task_cfg.env_id),
            num_envs=int(cfg.training.num_envs),
            base_seed=int(cfg.training.seed),
            seed_rank=rank,
            seed_world_size=world_size,
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
            policy_action_low, policy_action_high = self._configure_policy_action_bounds(
                base_policy,
                runner,
            )
            policy = ResFiTPolicy(
                base_policy=base_policy,
                residual_actor=self.residual_actor,
                action_low=policy_action_low,
                action_high=policy_action_high,
            ).to(device).eval()
            if is_main_process and distributed:
                print(
                    "[ResFiT distributed] "
                    f"world_size={world_size} "
                    f"envs_per_rank={int(cfg.training.num_envs)} "
                    f"global_envs={int(cfg.training.num_envs) * world_size} "
                    f"batch_size_per_rank={int(cfg.training.batch_size)} "
                    f"effective_batch_size={int(cfg.training.batch_size) * world_size}"
                )
            identity_summary = None
            if not resume and int(cfg.training.identity_rollout_steps) > 0:
                identity_summary = self._run_identity_rollout(policy, runner, cfg)
                if is_main_process:
                    print(
                        "[ResFiT identity] "
                        f"checkpoint={checkpoint_path} source={source} "
                        f"steps={identity_summary['steps']} "
                        f"envs={identity_summary['num_envs']} "
                        f"feature_shape={identity_summary['feature_shape']} "
                        f"residual_abs_max={identity_summary['residual_abs_max']:.3e} "
                        f"identity_error_max={identity_summary['identity_error_max']:.3e} "
                        f"action_abs_max={identity_summary['action_abs_max']:.6f}"
                    )
            training_summary = self._run_online_training(
                policy,
                runner,
                cfg,
                device,
                actor_forward=actor_forward,
                critic_forward=critic_forward,
                is_main_process=is_main_process,
            )
        finally:
            runner.close()

        if is_main_process:
            print(
                "[ResFiT complete] "
                f"iterations={training_summary['iteration']} "
                f"global_step={training_summary['global_step']} "
                f"replay_size={training_summary['replay_size']} "
                f"checkpoint={training_summary['checkpoint_path']}"
            )

    @staticmethod
    def _validate_base_checkpoint(payload: dict, cfg: OmegaConf) -> None:
        checkpoint_cfg = payload.get("cfg")
        if checkpoint_cfg is None:
            raise RuntimeError("Base checkpoint has no saved Hydra configuration.")

        expected = {
            "n_obs_steps": int(cfg.n_obs_steps),
            "n_action_steps": int(cfg.n_action_steps),
            "horizon": int(cfg.horizon),
            "action_dim": int(cfg.shape_meta.action.shape[0]),
            "control_mode": "bimanual_pd_ee_delta_pose",
            "action_semantics": "chunk_anchor_relative_target",
            "rotation_representation": "euler_xyz",
            "translation_frame": "root_aligned",
            "gripper_semantics": "absolute",
            "control_freq": 20,
        }
        actual = {
            "n_obs_steps": int(checkpoint_cfg.n_obs_steps),
            "n_action_steps": int(checkpoint_cfg.n_action_steps),
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

    @staticmethod
    def _validate_resume_checkpoint(payload: dict, cfg: OmegaConf) -> None:
        checkpoint_cfg = payload.get("cfg")
        if checkpoint_cfg is None:
            raise RuntimeError("ResFiT resume checkpoint has no saved Hydra configuration.")
        expected = {
            "n_obs_steps": int(cfg.n_obs_steps),
            "n_action_steps": int(cfg.n_action_steps),
            "horizon": int(cfg.horizon),
            "action_dim": int(cfg.shape_meta.action.shape[0]),
            "control_mode": "bimanual_pd_ee_delta_pose",
            "action_semantics": "chunk_anchor_relative_target",
            "rotation_representation": "euler_xyz",
            "translation_frame": "root_aligned",
            "gripper_semantics": "absolute",
        }
        try:
            actual = {
                "n_obs_steps": int(checkpoint_cfg.n_obs_steps),
                "n_action_steps": int(checkpoint_cfg.n_action_steps),
                "horizon": int(checkpoint_cfg.horizon),
                "action_dim": int(checkpoint_cfg.shape_meta.action.shape[0]),
                "control_mode": str(checkpoint_cfg.task.maniskill.control_mode),
                "action_semantics": str(
                    checkpoint_cfg.task.maniskill.action_semantics
                ),
                "rotation_representation": str(
                    checkpoint_cfg.task.maniskill.rotation_representation
                ),
                "translation_frame": str(
                    checkpoint_cfg.task.maniskill.translation_frame
                ),
                "gripper_semantics": str(
                    checkpoint_cfg.task.maniskill.gripper_semantics
                ),
            }
        except (AttributeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "ResFiT resume checkpoint is missing action execution semantics."
            ) from exc
        mismatches = {
            key: (expected_value, actual[key])
            for key, expected_value in expected.items()
            if actual[key] != expected_value
        }
        if mismatches:
            raise RuntimeError(
                f"ResFiT resume checkpoint semantics mismatch: {mismatches}"
            )

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
            absolute_action_chunk = self._prepare_absolute_action_chunk(
                runner,
                result["action"],
            )
            absolute_action = absolute_action_chunk[:, 0]
            expected_action = runner.clip_actions(base_action)

            if not all(
                torch.isfinite(value).all()
                for value in (base_action, residual_action, action, result["obs_feature"])
            ):
                raise RuntimeError("ResFiT identity rollout produced NaN or Inf values.")

            residual_max = float(residual_action.abs().amax().item())
            identity_error = float(
                (result["action"] - expected_action).abs().amax().item()
            )
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

            next_obs, _, terminated, truncated, _ = runner.step(absolute_action)
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

    @staticmethod
    def _configure_policy_action_bounds(base_policy, runner):
        """Install ResFiT algorithm bounds from the base action normalizer."""

        try:
            action_stats = base_policy.normalizer["action"].get_input_stats()
            action_low = torch.as_tensor(action_stats["min"], dtype=torch.float32).flatten()
            action_high = torch.as_tensor(action_stats["max"], dtype=torch.float32).flatten()
        except (AttributeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "Base policy action normalizer must expose finite `min`/`max` input stats."
            ) from exc
        if action_low.shape != (14,) or action_high.shape != (14,):
            raise RuntimeError(
                "Base policy action normalizer bounds must both have shape (14,), got "
                f"{tuple(action_low.shape)} and {tuple(action_high.shape)}."
            )
        if not torch.isfinite(action_low).all() or not torch.isfinite(action_high).all():
            raise RuntimeError("Base policy action normalizer bounds contain NaN or Inf values.")
        if not torch.all(action_low < action_high):
            raise RuntimeError(
                "Every base policy action normalizer minimum must be smaller than its maximum."
            )

        runner.set_policy_action_bounds(action_low, action_high)
        configured_low = torch.as_tensor(
            runner.action_low,
            dtype=torch.float32,
        ).flatten()
        configured_high = torch.as_tensor(
            runner.action_high,
            dtype=torch.float32,
        ).flatten()
        if not torch.equal(configured_low.cpu(), action_low.cpu()) or not torch.equal(
            configured_high.cpu(),
            action_high.cpu(),
        ):
            raise RuntimeError(
                "Runner policy action bounds differ from the base normalizer input stats."
            )
        return configured_low, configured_high

    def _run_online_training(
        self,
        policy,
        runner,
        cfg,
        device: torch.device,
        *,
        actor_forward: torch.nn.Module,
        critic_forward: torch.nn.Module,
        is_main_process: bool,
    ) -> dict:
        _, world_size = _distributed_context()
        replay = OnlineReplayBuffer(
            capacity=int(cfg.training.replay_capacity),
            feature_dim=int(cfg.residual_actor.feature_dim),
            action_dim=int(cfg.residual_actor.action_dim),
            device=str(cfg.training.replay_device),
        )
        batch_size = int(cfg.training.batch_size)
        learning_starts = int(cfg.training.learning_starts)
        if learning_starts < batch_size:
            raise ValueError("training.learning_starts must be at least batch_size.")
        if learning_starts > replay.capacity:
            raise ValueError("training.learning_starts must not exceed replay_capacity.")

        rollout_state = self._initialize_rollout_state(policy, runner, cfg)
        while len(replay) < learning_starts:
            missing = learning_starts - len(replay)
            rollout_steps = max(
                1,
                (missing + int(cfg.training.num_envs) - 1)
                // int(cfg.training.num_envs),
            )
            warmup_summary = self._collect_online_transitions(
                policy,
                runner,
                replay,
                cfg,
                rollout_state,
                rollout_steps,
            )
            self.global_step += int(warmup_summary["transitions"]) * world_size
        if is_main_process:
            print(
                "[ResFiT warmup] "
                f"replay_size_per_rank={len(replay)} "
                f"global_replay_size={len(replay) * world_size} "
                f"global_step={self.global_step}"
            )

        online_iters = int(cfg.training.online_iters)
        start_iteration = int(self.epoch)
        if online_iters <= start_iteration:
            raise ValueError(
                f"training.online_iters={online_iters} must exceed resumed "
                f"iteration={start_iteration}."
            )

        output_dir = pathlib.Path(self.output_dir)
        if is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        tb_cfg = cfg.logging.tensorboard
        tb_writer = (
            SummaryWriter(
                log_dir=output_dir / "tensorboard",
                flush_secs=int(tb_cfg.flush_secs),
            )
            if bool(tb_cfg.enabled) and is_main_process
            else None
        )
        checkpoint_every = int(cfg.training.checkpoint_every)
        last_checkpoint_epoch = -1
        checkpoint_path = ""
        last_log = None
        json_logger_context = (
            JsonLogger(str(output_dir / "logs.json.txt"))
            if is_main_process
            else contextlib.nullcontext(None)
        )
        try:
            with json_logger_context as json_logger:
                with tqdm.tqdm(
                    range(start_iteration, online_iters),
                    desc="ResFiT Training",
                    dynamic_ncols=True,
                    mininterval=float(cfg.training.tqdm_interval_sec),
                    leave=True,
                    disable=not is_main_process,
                ) as progress:
                    for iteration in progress:
                        started_at = time.perf_counter()
                        local_rollout_summary = self._collect_online_transitions(
                            policy,
                            runner,
                            replay,
                            cfg,
                            rollout_state,
                            int(cfg.training.rollout_steps_per_iteration),
                        )
                        rollout_summary = self._reduce_rollout_summary(
                            local_rollout_summary,
                            device,
                        )
                        self.global_step += int(rollout_summary["transitions"])
                        local_update_summary = self._run_updates(
                            policy,
                            replay,
                            cfg,
                            device,
                            actor_forward=actor_forward,
                            critic_forward=critic_forward,
                        )
                        update_summary = self._reduce_update_summary(
                            local_update_summary,
                            device,
                        )
                        self.epoch = iteration + 1
                        iteration_seconds = time.perf_counter() - started_at
                        iteration_seconds = self._distributed_scalar(
                            iteration_seconds,
                            device,
                            op=dist.ReduceOp.MAX,
                        )
                        last_log = {
                            "iteration": self.epoch,
                            "global_step": self.global_step,
                            "replay_size": len(replay) * world_size,
                            "replay_size_per_rank": len(replay),
                            "iteration_seconds": iteration_seconds,
                            "iteration_transitions_per_second": (
                                rollout_summary["transitions"] / iteration_seconds
                            ),
                            **rollout_summary,
                            **update_summary,
                        }
                        if is_main_process:
                            json_logger.log(last_log)
                            self._write_tensorboard(tb_writer, last_log)
                            progress.set_postfix(
                                reward=f"{last_log['reward_mean']:.3f}",
                                stage=f"{last_log['reward_stage_mean']:.2f}",
                                critic=f"{last_log['critic_loss']:.3f}",
                                actor=f"{last_log['actor_loss']:.3f}",
                            )

                        if (
                            is_main_process
                            and checkpoint_every > 0
                            and self.epoch % checkpoint_every == 0
                        ):
                            checkpoint_path = self.save_checkpoint(
                                path=output_dir / "checkpoints" / f"{self.epoch}.ckpt",
                                use_thread=False,
                            )
                            last_checkpoint_epoch = self.epoch
        finally:
            if tb_writer is not None:
                tb_writer.close()

        if last_log is None:
            raise RuntimeError("ResFiT online loop completed without an iteration.")
        if (
            is_main_process
            and bool(cfg.training.save_final_checkpoint)
            and last_checkpoint_epoch != self.epoch
        ):
            checkpoint_path = self.save_checkpoint(
                path=output_dir / "checkpoints" / f"{self.epoch}.ckpt",
                use_thread=False,
            )
        if world_size > 1:
            shared_checkpoint_path = [checkpoint_path if is_main_process else None]
            dist.broadcast_object_list(shared_checkpoint_path, src=0)
            checkpoint_path = shared_checkpoint_path[0] or ""
        last_log["checkpoint_path"] = checkpoint_path
        return last_log

    def _run_updates(
        self,
        policy,
        replay,
        cfg,
        device: torch.device,
        *,
        actor_forward: torch.nn.Module,
        critic_forward: torch.nn.Module,
    ) -> dict:
        updates = int(cfg.training.updates_per_iteration)
        if updates <= 0:
            raise ValueError("training.updates_per_iteration must be positive.")
        if len(replay) < int(cfg.training.batch_size):
            raise RuntimeError("Replay buffer has fewer transitions than batch_size.")

        policy.train()
        self.critic.train()
        actor_forward.train()
        critic_forward.train()
        update_logs = []
        for _ in range(updates):
            self.critic_update_step += 1
            batch = replay.sample(int(cfg.training.batch_size), device=device)
            update_logs.append(
                resfit_td3_update(
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
                    target_noise_fraction=float(
                        cfg.training.target_noise_fraction
                    ),
                    target_noise_clip_fraction=float(
                        cfg.training.target_noise_clip_fraction
                    ),
                    residual_actor_forward=actor_forward,
                    critic_forward=critic_forward,
                )
            )
        if any(parameter.grad is not None for parameter in policy.base_policy.parameters()):
            raise RuntimeError("Frozen base policy received gradients during ResFiT update.")

        actor_logs = [log for log in update_logs if bool(log["actor_updated"])]
        return {
            "updates": updates,
            "actor_updates": len(actor_logs),
            "critic_loss": float(np.mean([log["critic_loss"] for log in update_logs])),
            "actor_loss": (
                float(np.mean([log["actor_loss"] for log in actor_logs]))
                if actor_logs
                else 0.0
            ),
            "q1_mean": float(np.mean([log["q1_mean"] for log in update_logs])),
            "q2_mean": float(np.mean([log["q2_mean"] for log in update_logs])),
            "target_q_mean": float(
                np.mean([log["target_q_mean"] for log in update_logs])
            ),
            "actor_residual_abs_max": max(
                log["actor_residual_abs_max"] for log in update_logs
            ),
            "target_action_abs_max": max(
                log["target_action_abs_max"] for log in update_logs
            ),
        }

    @staticmethod
    def _broadcast_module(module: torch.nn.Module) -> None:
        _, world_size = _distributed_context()
        if world_size == 1:
            return
        with torch.no_grad():
            for tensor in tuple(module.parameters()) + tuple(module.buffers()):
                dist.broadcast(tensor, src=0)

    @staticmethod
    def _distributed_scalar(
        value: float,
        device: torch.device,
        *,
        op,
    ) -> float:
        _, world_size = _distributed_context()
        if world_size == 1:
            return float(value)
        scalar = torch.tensor(float(value), device=device, dtype=torch.float64)
        dist.all_reduce(scalar, op=op)
        return float(scalar.item())

    @staticmethod
    def _reduce_rollout_summary(summary: dict, device: torch.device) -> dict:
        _, world_size = _distributed_context()
        if world_size == 1:
            return summary

        transitions = float(summary["transitions"])
        completed = float(summary["completed_episodes"])
        sums = torch.tensor(
            [
                transitions,
                float(summary["reward_mean"]) * transitions,
                float(summary["terminated_count"]),
                float(summary["truncated_count"]),
                float(summary["reward_stage_mean"]) * transitions,
                completed,
                float(summary["episode_return_mean"]) * completed,
                float(summary["episode_length_mean"]) * completed,
                float(summary["episode_success_rate"]) * completed,
            ],
            device=device,
            dtype=torch.float64,
        )
        maxima = torch.tensor(
            [
                float(summary["rollout_residual_abs_max"]),
                float(summary["rollout_action_abs_max"]),
                float(summary["reward_stage_max"]),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)

        global_transitions = max(1.0, float(sums[0].item()))
        global_completed = float(sums[5].item())
        completed_denominator = max(1.0, global_completed)
        return {
            "transitions": int(sums[0].item()),
            "reward_mean": float(sums[1].item()) / global_transitions,
            "terminated_count": int(sums[2].item()),
            "truncated_count": int(sums[3].item()),
            "rollout_residual_abs_max": float(maxima[0].item()),
            "rollout_action_abs_max": float(maxima[1].item()),
            "reward_stage_mean": float(sums[4].item()) / global_transitions,
            "reward_stage_max": float(maxima[2].item()),
            "completed_episodes": int(global_completed),
            "episode_return_mean": float(sums[6].item()) / completed_denominator,
            "episode_length_mean": float(sums[7].item()) / completed_denominator,
            "episode_success_rate": float(sums[8].item()) / completed_denominator,
        }

    @staticmethod
    def _reduce_update_summary(summary: dict, device: torch.device) -> dict:
        _, world_size = _distributed_context()
        if world_size == 1:
            return summary

        means = torch.tensor(
            [
                float(summary["critic_loss"]),
                float(summary["actor_loss"]),
                float(summary["q1_mean"]),
                float(summary["q2_mean"]),
                float(summary["target_q_mean"]),
            ],
            device=device,
            dtype=torch.float64,
        )
        maxima = torch.tensor(
            [
                float(summary["actor_residual_abs_max"]),
                float(summary["target_action_abs_max"]),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(means, op=dist.ReduceOp.SUM)
        means /= world_size
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        return {
            "updates": int(summary["updates"]),
            "actor_updates": int(summary["actor_updates"]),
            "critic_loss": float(means[0].item()),
            "actor_loss": float(means[1].item()),
            "q1_mean": float(means[2].item()),
            "q2_mean": float(means[3].item()),
            "target_q_mean": float(means[4].item()),
            "actor_residual_abs_max": float(maxima[0].item()),
            "target_action_abs_max": float(maxima[1].item()),
        }

    def _initialize_rollout_state(self, policy, runner, cfg) -> dict:
        obs, _ = runner.reset()
        obs_history = self._init_obs_history(obs, int(policy.base_policy.n_obs_steps))
        policy.eval()
        with torch.inference_mode():
            base_result = policy.predict_base_action_chunk(obs_history)
            current_result = self._prepare_rollout_chunk(
                policy,
                base_result,
                cfg,
                runner,
            )
        return {
            "obs_history": obs_history,
            "current_result": current_result,
            "chunk_index": 0,
            "episode_return": torch.zeros(
                runner.num_envs,
                device=runner.device,
                dtype=torch.float32,
            ),
            "episode_length": torch.zeros(
                runner.num_envs,
                device=runner.device,
                dtype=torch.long,
            ),
        }

    def _collect_online_transitions(
        self,
        policy,
        runner,
        replay,
        cfg,
        rollout_state,
        rollout_steps: int,
    ) -> dict:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive.")

        obs_history = rollout_state["obs_history"]
        current_result = rollout_state["current_result"]
        chunk_index = int(rollout_state["chunk_index"])
        policy.eval()

        reward_sum = 0.0
        terminated_count = 0
        truncated_count = 0
        residual_abs_max = 0.0
        action_abs_max = 0.0
        stage_sum = 0.0
        stage_count = 0
        stage_max = 0.0
        completed_returns = []
        completed_lengths = []
        completed_successes = []
        for _ in range(rollout_steps):
            base_action_sequence = current_result["base_action"]
            if not 0 <= chunk_index < base_action_sequence.shape[1]:
                raise RuntimeError(
                    f"Invalid ResFiT action chunk index {chunk_index} for "
                    f"shape {tuple(base_action_sequence.shape)}."
                )
            obs_feature = current_result["obs_feature"]
            base_action = base_action_sequence[:, chunk_index]
            residual_action = current_result["residual_action"][:, chunk_index]
            action = current_result["action"][:, chunk_index]
            absolute_action = current_result["absolute_action"][:, chunk_index]
            if not torch.isfinite(action).all():
                raise RuntimeError("ResFiT rollout action contains NaN or Inf values.")
            if torch.any(action < policy.action_low) or torch.any(action > policy.action_high):
                raise RuntimeError("ResFiT rollout action exceeded policy normalizer bounds.")

            next_obs, reward, terminated, truncated, info = runner.step(absolute_action)
            next_obs_history = self._append_obs_history(obs_history, next_obs)
            done_ids = torch.logical_or(terminated, truncated).nonzero(
                as_tuple=False
            ).flatten()
            next_chunk_index = chunk_index + 1
            refresh_chunk = (
                next_chunk_index >= base_action_sequence.shape[1]
                or done_ids.numel() > 0
            )
            if refresh_chunk:
                with torch.inference_mode():
                    transition_next_result = policy.predict_base_action_chunk(
                        next_obs_history
                    )
                transition_next_index = 0
            else:
                transition_next_result = current_result
                transition_next_index = next_chunk_index
            next_obs_feature = transition_next_result["obs_feature"]
            next_base_action = transition_next_result["base_action"][
                :, transition_next_index
            ]

            replay.add_batch(
                obs_feature=obs_feature.detach(),
                base_action=base_action.detach(),
                action=action.detach(),
                reward=reward.detach().to(dtype=torch.float32),
                next_obs_feature=next_obs_feature.detach(),
                next_base_action=next_base_action.detach(),
                terminated=terminated.detach().to(dtype=torch.float32),
            )

            reward_sum += float(reward.detach().sum().item())
            terminated_count += int(terminated.detach().sum().item())
            truncated_count += int(truncated.detach().sum().item())
            rollout_state["episode_return"] += reward.detach().to(dtype=torch.float32)
            rollout_state["episode_length"] += 1
            reward_stage = self._info_tensor(info, "reward_stage", runner)
            if reward_stage is not None:
                stage_sum += float(reward_stage.sum().item())
                stage_count += int(reward_stage.numel())
                stage_max = max(stage_max, float(reward_stage.max().item()))
            success = self._info_tensor(info, "success", runner, dtype=torch.bool)
            residual_abs_max = max(
                residual_abs_max,
                float(residual_action.detach().abs().amax().item()),
            )
            action_abs_max = max(
                action_abs_max,
                float(action.detach().abs().amax().item()),
            )

            if done_ids.numel() > 0:
                completed_returns.extend(
                    rollout_state["episode_return"][done_ids].detach().cpu().tolist()
                )
                completed_lengths.extend(
                    rollout_state["episode_length"][done_ids].detach().cpu().tolist()
                )
                if success is None:
                    completed_successes.extend([0.0] * int(done_ids.numel()))
                else:
                    completed_successes.extend(
                        success[done_ids].to(dtype=torch.float32).detach().cpu().tolist()
                    )
                reset_obs, _ = runner.reset_at(done_ids.detach().cpu().tolist())
                self._reset_obs_history(next_obs_history, reset_obs, done_ids)
                rollout_state["episode_return"][done_ids] = 0.0
                rollout_state["episode_length"][done_ids] = 0
                reset_obs_history = {
                    key: value.index_select(0, done_ids)
                    for key, value in next_obs_history.items()
                }
                with torch.inference_mode():
                    reset_result = policy.predict_base_action_chunk(
                        reset_obs_history
                    )
                current_result = {
                    key: value.clone()
                    for key, value in transition_next_result.items()
                    if key in ("base_action", "obs_feature")
                }
                for key, value in current_result.items():
                    value.index_copy_(0, done_ids, reset_result[key])
                current_result = self._prepare_rollout_chunk(
                    policy,
                    current_result,
                    cfg,
                    runner,
                )
                chunk_index = 0
            else:
                if refresh_chunk:
                    current_result = self._prepare_rollout_chunk(
                        policy,
                        transition_next_result,
                        cfg,
                        runner,
                    )
                else:
                    current_result = transition_next_result
                chunk_index = transition_next_index
            obs_history = next_obs_history

        rollout_state["obs_history"] = obs_history
        rollout_state["current_result"] = current_result
        rollout_state["chunk_index"] = chunk_index
        transitions = rollout_steps * int(cfg.training.num_envs)
        return {
            "transitions": transitions,
            "reward_mean": reward_sum / transitions,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "rollout_residual_abs_max": residual_abs_max,
            "rollout_action_abs_max": action_abs_max,
            "reward_stage_mean": stage_sum / max(1, stage_count),
            "reward_stage_max": stage_max,
            "completed_episodes": len(completed_returns),
            "episode_return_mean": (
                float(np.mean(completed_returns)) if completed_returns else 0.0
            ),
            "episode_length_mean": (
                float(np.mean(completed_lengths)) if completed_lengths else 0.0
            ),
            "episode_success_rate": (
                float(np.mean(completed_successes)) if completed_successes else 0.0
            ),
        }

    @staticmethod
    def _prepare_absolute_action_chunk(runner, action_chunk) -> torch.Tensor:
        """Resolve a policy-domain chunk once against the current EEF anchors."""

        absolute_action_chunk = runner.prepare_action_chunk(action_chunk)
        absolute_action_chunk = torch.as_tensor(
            absolute_action_chunk,
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )
        if absolute_action_chunk.shape != action_chunk.shape:
            raise RuntimeError(
                "Prepared ResFiT action chunk shape mismatch: "
                f"policy={tuple(action_chunk.shape)}, "
                f"absolute={tuple(absolute_action_chunk.shape)}."
            )
        if not torch.isfinite(absolute_action_chunk).all():
            raise RuntimeError("Prepared ResFiT action chunk contains NaN or Inf values.")
        return absolute_action_chunk

    @staticmethod
    def _prepare_rollout_chunk(policy, base_result, cfg, runner) -> dict[str, torch.Tensor]:
        base_action = base_result["base_action"]
        obs_feature = base_result["obs_feature"]
        feature_sequence = obs_feature[:, None, :].expand(
            -1,
            base_action.shape[1],
            -1,
        )
        with torch.inference_mode():
            current_residual = policy.residual_actor(
                feature_sequence,
                base_action,
            )
            residual_action = add_bounded_residual_noise(
                current_residual,
                policy.residual_actor.residual_scale,
                noise_fraction=float(cfg.training.exploration_noise_fraction),
                noise_clip_fraction=float(
                    cfg.training.exploration_noise_clip_fraction
                ),
            )
            action = compose_bimanual_delta_pose_action(
                base_action,
                residual_action,
                policy.action_low,
                policy.action_high,
            )
            absolute_action = ResFiTWorkspace._prepare_absolute_action_chunk(
                runner,
                action,
            )
        return {
            "base_action": base_action,
            "obs_feature": obs_feature,
            "residual_action": residual_action,
            "action": action,
            "absolute_action": absolute_action,
        }

    @staticmethod
    def _info_tensor(info, key, runner, dtype=torch.float32):
        if not isinstance(info, dict) or key not in info or info[key] is None:
            return None
        value = torch.as_tensor(info[key], device=runner.device, dtype=dtype).reshape(-1)
        if value.numel() != runner.num_envs:
            return None
        return value

    @staticmethod
    def _write_tensorboard(writer, step_log) -> None:
        if writer is None:
            return
        step = int(step_log["global_step"])
        for key, value in step_log.items():
            if isinstance(value, (bool, int, float)) and np.isfinite(value):
                writer.add_scalar(f"online/{key}", value, step)

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
