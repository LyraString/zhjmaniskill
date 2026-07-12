import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import (
    ManiSkillTaskFactory,
    ManiSkillTrajectoryRecorder,
    apply_task_profile,
    resolve_runtime_control_mode,
    resolve_runtime_task_target,
    resolve_task_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect ManiSkill episodes into `.npz` files.")
    parser.add_argument("task_name", type=str, help="ManiSkill env id or the env id associated with --task-profile")
    parser.add_argument("task_config", type=str, help="Dataset subdirectory name under data/<task_name>/")
    parser.add_argument("episode_num", type=int, help="Number of episodes to collect")
    parser.add_argument("--task-profile", type=str, default=None, help="Optional registered ManiSkill task profile")
    parser.add_argument("--seed", type=int, default=42, help="Base reset seed")
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--action-type", type=str, default=None)
    parser.add_argument("--obs-mode", type=str, default=None)
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--sim-backend", type=str, default=None)
    parser.add_argument("--render-backend", type=str, default=None)
    parser.add_argument("--reward-mode", type=str, default=None)
    parser.add_argument("--robot-uids", type=str, default=None)
    parser.add_argument("--sim-freq", type=int, default=None)
    parser.add_argument("--control-freq", type=int, default=None)
    parser.add_argument("--save-video-dir", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--collector",
        choices=(
            "random",
            "legacy_replay",
            "scripted_expert",
            "scripted_delta_pose_expert",
            "policy_intervention",
            "policy_manual_intervention",
        ),
        default="random",
    )
    parser.add_argument("--teacher-zarr-path", type=str, default=None)
    parser.add_argument("--teacher-seed-path", type=str, default=None)
    parser.add_argument("--policy-kind", choices=("flowmatching", "fpo"), default="flowmatching")
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--save-only-success", action="store_true")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Optional cap for seed attempts before stopping an attempt-based collector.",
    )
    parser.add_argument("--debug-scripted-expert", action="store_true")
    parser.add_argument("--debug-policy-intervention", action="store_true")
    parser.add_argument("--debug-policy-manual-intervention", action="store_true")
    return parser


def _resolve_repo_path(path_str: str | None) -> Path | None:
    if path_str in (None, ""):
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _to_bool_scalar(value) -> bool:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _raw_recorder(recorder):
    return getattr(recorder, "_recorder", recorder)


def _recorder_actions(recorder) -> np.ndarray:
    actions = getattr(_raw_recorder(recorder), "_actions", None)
    if actions is None:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(actions, dtype=np.float32)


def _delta_pose_actions_valid(actions: np.ndarray) -> bool:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.size == 0 or actions.ndim != 2 or actions.shape[1] != 14:
        return False
    if not np.all(np.isfinite(actions)):
        return False
    eps = 1e-6
    for offset in (0, 7):
        translation_norms = np.linalg.norm(actions[:, offset : offset + 3], axis=1)
        rotation_norms = np.linalg.norm(actions[:, offset + 3 : offset + 6], axis=1)
        if np.any(translation_norms > 0.015 + eps) or np.any(rotation_norms > 0.10 + eps):
            return False
    return True


def _validate_action_replay(
    *,
    task_name: str,
    resolved_runtime: dict,
    seed: int,
    actions: np.ndarray,
    max_episode_steps: int,
) -> tuple[bool, dict]:
    replay_runtime = dict(resolved_runtime)
    replay_runtime["render_mode"] = None
    replay_runtime["obs_mode"] = "state_dict"
    replay_env = ManiSkillTaskFactory().create_env(
        task_name=task_name,
        **replay_runtime,
    )
    try:
        replay_env.reset(seed=int(seed))
        steps = 0
        terminated = False
        truncated = False
        success = False
        for action in np.asarray(actions, dtype=np.float32)[: int(max_episode_steps)]:
            result = replay_env.step(
                ActionSpec.from_value(action, action_type=replay_runtime.get("action_type"))
            )
            steps += 1
            success = _to_bool_scalar(result.info.get("success", False))
            if success:
                break
            terminated = bool(result.terminated)
            truncated = bool(result.truncated)
            if terminated or truncated:
                break

        eval_info = {}
        evaluator = getattr(replay_env.env.unwrapped, "evaluate", None)
        if callable(evaluator):
            raw_eval = evaluator()
            eval_info = {
                key: _to_bool_scalar(value)
                for key, value in raw_eval.items()
                if key
                in {
                    "success",
                    "spatial_success",
                    "right_gripper_open",
                    "right_ee_clear",
                    "box_static",
                    "xy_aligned",
                    "z_aligned",
                }
            }
            success = bool(eval_info.get("success", success))

        debug: dict[str, object] = {
            "seed": int(seed),
            "steps": int(steps),
            "success": int(success),
            "terminated": int(terminated),
            "truncated": int(truncated),
            **eval_info,
        }

        base_env = replay_env.env.unwrapped
        if hasattr(base_env, "box") and hasattr(base_env, "target_box"):
            box = _to_numpy(base_env.box.pose.p).reshape(-1, 3)[0]
            target = _to_numpy(base_env.target_box.pose.p).reshape(-1, 3)[0]
            bottom_minus_top = (box - np.asarray([0.0, 0.0, 0.1])) - (
                target + np.asarray([0.0, 0.0, 0.005])
            )
            debug["bottom_minus_target_top"] = [
                round(float(value), 4) for value in bottom_minus_top
            ]
        return bool(success), debug
    finally:
        replay_env.close()


def _write_episode_metadata(out_path: Path, metadata: dict) -> None:
    metadata_path = out_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _maybe_capture_frame(env_backend, frames: list[np.ndarray]) -> None:
    frame = env_backend.render()
    if frame is None:
        return
    if hasattr(frame, "detach") and hasattr(frame, "cpu"):
        frame = frame.detach().cpu()
    frame = np.asarray(frame)
    # ManiSkill rgb_array render returns (num_envs, H, W, C) for vectorized envs,
    # even when num_envs == 1.
    if frame.ndim == 4 and frame.shape[0] >= 1:
        frame = frame[0]
    if frame.ndim != 3:
        return
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    frames.append(frame)


class VideoTrajectoryRecorder:
    def __init__(
        self,
        recorder: ManiSkillTrajectoryRecorder,
        *,
        env_backend,
        video_dir: Path | None,
        video_fps: int,
    ) -> None:
        self._recorder = recorder
        self._env_backend = env_backend
        self._video_dir = video_dir
        self._video_fps = int(video_fps)
        self._frames: list[np.ndarray] = []
        if self._video_dir is not None:
            self._video_dir.mkdir(parents=True, exist_ok=True)

    def start_episode(self, initial_obs: dict) -> None:
        self._recorder.start_episode(initial_obs)
        self._frames = []
        if self._video_dir is not None:
            _maybe_capture_frame(self._env_backend, self._frames)

    def append_step(self, action, next_obs: dict) -> None:
        self._recorder.append_step(action, next_obs)
        if self._video_dir is not None:
            _maybe_capture_frame(self._env_backend, self._frames)

    def save_episode(self, episode_idx: int) -> Path:
        out_path = self._recorder.save_episode(episode_idx)
        if self._video_dir is not None and len(self._frames) > 0:
            imageio.mimsave(
                self._video_dir / f"episode{int(episode_idx)}.mp4",
                self._frames,
                fps=self._video_fps,
            )
        return out_path


def _load_teacher_actions(zarr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open(str(zarr_path), mode="r")
    data_group = root["data"]
    if "action" in data_group:
        actions = np.asarray(data_group["action"], dtype=np.float32)
    elif "actions" in data_group:
        actions = np.asarray(data_group["actions"], dtype=np.float32)
    else:
        raise KeyError(f"No `action` or `actions` array found in teacher zarr: {zarr_path}")
    episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
    return actions, episode_ends


def _load_teacher_seeds(seed_path: Path | None) -> list[int] | None:
    if seed_path is None or not seed_path.is_file():
        return None
    raw = seed_path.read_text(encoding="utf-8").split()
    if len(raw) == 0:
        return None
    return [int(value) for value in raw]


def _resolve_teacher_assets(args, resolved_profile_name: str | None) -> tuple[Path, Path | None]:
    teacher_zarr_path = _resolve_repo_path(args.teacher_zarr_path)
    teacher_seed_path = _resolve_repo_path(args.teacher_seed_path)
    if teacher_zarr_path is not None:
        return teacher_zarr_path, teacher_seed_path

    if resolved_profile_name is None:
        raise ValueError(
            "--collector legacy_replay requires either --teacher-zarr-path "
            "or a task profile with teacher_dataset defaults."
        )

    profile = resolve_task_profile(resolved_profile_name)
    teacher_cfg = (profile or {}).get("teacher_dataset", {})
    teacher_zarr_path = _resolve_repo_path(teacher_cfg.get("zarr_path"))
    teacher_seed_path = _resolve_repo_path(teacher_cfg.get("seed_path"))
    if teacher_zarr_path is None:
        raise ValueError(
            f"Task profile `{resolved_profile_name}` does not define a teacher_dataset.zarr_path "
            "and --teacher-zarr-path was not provided."
        )
    return teacher_zarr_path, teacher_seed_path


def run_collection(args) -> None:
    runtime_env_id = args.task_name
    if args.task_profile is not None and args.task_name == args.task_profile:
        runtime_env_id = None
    runtime_task_name, resolved_profile_name = resolve_runtime_task_target(
        task_name=args.task_name,
        task_profile=args.task_profile,
        env_id=runtime_env_id,
    )
    runtime_kwargs = {
        "action_type": args.action_type,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "render_mode": args.render_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "reward_mode": args.reward_mode,
        "robot_uids": args.robot_uids,
        "sim_freq": args.sim_freq,
        "control_freq": args.control_freq,
        "max_episode_steps": args.max_episode_steps,
    }
    if args.collector == "scripted_delta_pose_expert":
        delta_control_mode = "bimanual_pd_ee_delta_pose"
        if runtime_kwargs["control_mode"] not in (None, "", delta_control_mode):
            raise ValueError(
                "scripted_delta_pose_expert requires "
                f"--control-mode {delta_control_mode}; got {runtime_kwargs['control_mode']}"
            )
        runtime_kwargs["control_mode"] = delta_control_mode
    _resolved_action_type, resolved_control_mode = resolve_runtime_control_mode(
        action_type=runtime_kwargs["action_type"],
        control_mode=runtime_kwargs["control_mode"],
    )
    runtime_kwargs["control_mode"] = resolved_control_mode
    if args.save_video_dir is not None and runtime_kwargs["render_mode"] in (None, ""):
        runtime_kwargs["render_mode"] = "rgb_array"
    profile_runtime_key = resolved_profile_name or runtime_task_name
    _, resolved_runtime, _profile_name = apply_task_profile(profile_runtime_key, runtime_kwargs)
    save_dir = REPO_ROOT / "data" / runtime_task_name / args.task_config
    video_dir = _resolve_repo_path(args.save_video_dir)

    def create_runtime():
        runtime_env = ManiSkillTaskFactory().create_env(
            task_name=runtime_task_name,
            **resolved_runtime,
        )
        runtime_recorder = VideoTrajectoryRecorder(
            ManiSkillTrajectoryRecorder(save_dir),
            env_backend=runtime_env,
            video_dir=video_dir,
            video_fps=args.video_fps,
        )
        return runtime_env, runtime_recorder

    env, recorder = create_runtime()
    # Reusing the real-layout SAPIEN/MPlib state makes later attempts depend on
    # the preceding placement rollout, so isolate each scripted attempt.
    recreate_real_layout_scripted_runtime = (
        resolved_profile_name == "handover_block_ur5_v2_real_layout_probe"
    )

    def recreate_runtime() -> None:
        nonlocal env, recorder
        env.close()
        env, recorder = create_runtime()

    def maybe_render() -> None:
        if getattr(env, "render_mode", None) == "human":
            env.render()

    try:
        if args.collector == "random":
            for episode_idx in range(args.episode_num):
                obs = env.reset(seed=args.seed + episode_idx)
                recorder.start_episode(dict(obs.data))
                maybe_render()

                for _ in range(args.max_episode_steps):
                    action = env.env.action_space.sample()
                    result = env.step(
                        ActionSpec.from_value(action, action_type=resolved_runtime["action_type"])
                    )
                    recorder.append_step(action, dict(result.observation.data))
                    maybe_render()
                    if result.terminated or result.truncated:
                        break

                out_path = recorder.save_episode(episode_idx)
                print(f"saved {out_path}")
            return

        if args.collector == "scripted_expert":
            from sim.maniskill.expert.handover_block_solver import HandoverBlockSolver

            solver = HandoverBlockSolver(env)
            saved_episode_idx = 0
            attempted_episode_idx = 0
            while saved_episode_idx < args.episode_num and (
                args.max_attempts is None or attempted_episode_idx < int(args.max_attempts)
            ):
                if recreate_real_layout_scripted_runtime and attempted_episode_idx > 0:
                    recreate_runtime()
                    solver = HandoverBlockSolver(env)
                reset_seed = args.seed + attempted_episode_idx
                obs = env.reset(seed=reset_seed)
                recorder.start_episode(dict(obs.data))
                maybe_render()
                success, episode_length, status = solver.solve_episode(
                    recorder=recorder,
                    max_episode_steps=args.max_episode_steps,
                )

                replay_debug = None
                replay_success = None
                if episode_length > 0 and success:
                    try:
                        replay_success, replay_debug = _validate_action_replay(
                            task_name=runtime_task_name,
                            resolved_runtime=resolved_runtime,
                            seed=reset_seed,
                            actions=_recorder_actions(recorder),
                            max_episode_steps=args.max_episode_steps,
                        )
                    except Exception as exc:
                        replay_success = False
                        replay_debug = {
                            "seed": int(reset_seed),
                            "success": 0,
                            "error": repr(exc),
                        }
                    if not replay_success:
                        status = f"{status}|action_replay_failed"

                should_save = (success and (replay_success is not False)) or not args.save_only_success
                if should_save:
                    if episode_length > 0:
                        out_path = recorder.save_episode(saved_episode_idx)
                        metadata = {
                            "collector": "scripted_expert",
                            "seed": int(reset_seed),
                            "length": int(episode_length),
                            "success": bool(success),
                            "replay_success": None if replay_success is None else bool(replay_success),
                            "status": str(status),
                            "action_replay": replay_debug,
                            "first_pick_debug": getattr(solver, "first_pick_debug", None),
                            "debug": getattr(solver, "last_debug_info", None),
                        }
                        _write_episode_metadata(out_path, metadata)
                        print(
                            f"saved {out_path} "
                            f"(seed={reset_seed}, length={episode_length}, "
                            f"success={int(success)}, "
                            f"replay_success={int(replay_success) if replay_success is not None else 'n/a'}, "
                            f"status={status})"
                        )
                        saved_episode_idx += 1
                    else:
                        print(
                            f"skipped scripted_expert attempt={attempted_episode_idx} "
                            f"(seed={reset_seed}, length=0, success={int(success)}, status={status})"
                        )
                    if args.debug_scripted_expert and getattr(solver, "first_pick_debug", None):
                        print("scripted_expert_pick_debug=" + json.dumps(solver.first_pick_debug, ensure_ascii=False))
                    if args.debug_scripted_expert and getattr(solver, "last_debug_info", None):
                        print("scripted_expert_debug=" + json.dumps(solver.last_debug_info, ensure_ascii=False))
                else:
                    print(
                        f"skipped scripted_expert attempt={attempted_episode_idx} "
                        f"(seed={reset_seed}, length={episode_length}, success={int(success)}, "
                        f"replay_success={int(replay_success) if replay_success is not None else 'n/a'}, "
                        f"status={status})"
                    )
                    if replay_debug is not None:
                        print(
                            "scripted_expert_replay_debug="
                            + json.dumps(replay_debug, ensure_ascii=False)
                        )
                    if args.debug_scripted_expert and getattr(solver, "first_pick_debug", None):
                        print("scripted_expert_pick_debug=" + json.dumps(solver.first_pick_debug, ensure_ascii=False))
                    if args.debug_scripted_expert and getattr(solver, "last_debug_info", None):
                        print("scripted_expert_debug=" + json.dumps(solver.last_debug_info, ensure_ascii=False))
                attempted_episode_idx += 1
            if saved_episode_idx < args.episode_num and args.max_attempts is not None:
                print(
                    f"warning: requested {args.episode_num} episodes but only saved {saved_episode_idx} "
                    f"after {attempted_episode_idx} scripted_expert attempts"
                )
            return

        if args.collector == "scripted_delta_pose_expert":
            from sim.maniskill.expert.handover_block_solver import HandoverBlockSolver
            from sim.maniskill.expert.primitives import DeltaPoseMotionPrimitives

            solver = HandoverBlockSolver(env, ctrl_cls=DeltaPoseMotionPrimitives)
            saved_episode_idx = 0
            attempted_episode_idx = 0
            while saved_episode_idx < args.episode_num and (
                args.max_attempts is None or attempted_episode_idx < int(args.max_attempts)
            ):
                if recreate_real_layout_scripted_runtime and attempted_episode_idx > 0:
                    recreate_runtime()
                    solver = HandoverBlockSolver(env, ctrl_cls=DeltaPoseMotionPrimitives)
                reset_seed = args.seed + attempted_episode_idx
                obs = env.reset(seed=reset_seed)
                recorder.start_episode(dict(obs.data))
                maybe_render()
                success, episode_length, status = solver.solve_episode(
                    recorder=recorder,
                    max_episode_steps=args.max_episode_steps,
                )
                delta_debug = solver.ctrl.delta_debug_summary()
                actions_valid = _delta_pose_actions_valid(_recorder_actions(recorder))
                controller_failure_count = int(delta_debug.get("controller_failure_count", 0))
                qpos_fallback_count = int(delta_debug.get("qpos_fallback_count", 0))
                if episode_length > 0 and not actions_valid:
                    status = f"{status}|delta_action_invalid"
                should_save = (
                    episode_length > 0
                    and success
                    and controller_failure_count == 0
                    and qpos_fallback_count == 0
                    and actions_valid
                )
                if should_save:
                    out_path = recorder.save_episode(saved_episode_idx)
                    metadata = {
                        "collector": "scripted_delta_pose_expert",
                        "action_semantics": "delta_pose_actual",
                        "rotation_representation": "euler_xyz",
                        "translation_frame": "root_aligned",
                        "gripper_semantics": "absolute",
                        "control_mode": "bimanual_pd_ee_delta_pose",
                        "seed": int(reset_seed),
                        "length": int(episode_length),
                        "success": bool(success),
                        "status": str(status),
                        "delta_debug": delta_debug,
                        "first_pick_debug": getattr(solver, "first_pick_debug", None),
                        "debug": getattr(solver, "last_debug_info", None),
                    }
                    _write_episode_metadata(out_path, metadata)
                    print(
                        f"saved {out_path} "
                        f"(seed={reset_seed}, length={episode_length}, "
                        f"success={int(success)}, "
                        f"controller_failure_count={controller_failure_count}, "
                        f"qpos_fallback_count={qpos_fallback_count}, "
                        f"actions_valid={int(actions_valid)}, "
                        f"status={status})"
                    )
                    saved_episode_idx += 1
                else:
                    print(
                        f"skipped scripted_delta_pose_expert attempt={attempted_episode_idx} "
                        f"(seed={reset_seed}, length={episode_length}, success={int(success)}, "
                        f"controller_failure_count={controller_failure_count}, "
                        f"qpos_fallback_count={qpos_fallback_count}, "
                        f"actions_valid={int(actions_valid)}, "
                        f"status={status})"
                    )
                    if args.debug_scripted_expert and getattr(solver, "first_pick_debug", None):
                        print("scripted_delta_pose_expert_pick_debug=" + json.dumps(solver.first_pick_debug, ensure_ascii=False))
                    if args.debug_scripted_expert and getattr(solver, "last_debug_info", None):
                        print("scripted_delta_pose_expert_debug=" + json.dumps(solver.last_debug_info, ensure_ascii=False))
                attempted_episode_idx += 1
            if saved_episode_idx < args.episode_num and args.max_attempts is not None:
                print(
                    f"warning: requested {args.episode_num} delta episodes but only saved {saved_episode_idx} "
                    f"after {attempted_episode_idx} scripted_delta_pose_expert attempts"
                )
            return

        if args.collector == "policy_intervention":
            from sim.maniskill.policy_intervention_collector import PolicyInterventionCollector

            ckpt_path = _resolve_repo_path(args.ckpt_path)
            if ckpt_path is None:
                raise ValueError("--collector policy_intervention requires --ckpt-path.")

            collector = PolicyInterventionCollector(
                env,
                policy_kind=args.policy_kind,
                ckpt_path=str(ckpt_path),
                action_type=resolved_runtime["action_type"],
                debug=args.debug_policy_intervention,
            )
            saved_episode_idx = 0
            attempted_episode_idx = 0
            while saved_episode_idx < args.episode_num:
                reset_seed = args.seed + attempted_episode_idx
                success, episode_length, status = collector.collect_episode(
                    recorder=recorder,
                    seed=reset_seed,
                    max_episode_steps=args.max_episode_steps,
                )
                if success or not args.save_only_success:
                    if episode_length > 0:
                        out_path = recorder.save_episode(saved_episode_idx)
                        print(
                            f"saved {out_path} "
                            f"(seed={reset_seed}, length={episode_length}, "
                            f"success={int(success)}, status={status})"
                        )
                        saved_episode_idx += 1
                    else:
                        print(
                            f"skipped policy_intervention attempt={attempted_episode_idx} "
                            f"(seed={reset_seed}, length=0, success={int(success)}, status={status})"
                        )
                    if args.debug_policy_intervention and getattr(collector, "last_debug_info", None):
                        print(
                            "policy_intervention_debug="
                            + json.dumps(collector.last_debug_info, ensure_ascii=False)
                        )
                else:
                    print(
                        f"skipped policy_intervention attempt={attempted_episode_idx} "
                        f"(seed={reset_seed}, length={episode_length}, success=0, status={status})"
                    )
                    if args.debug_policy_intervention and getattr(collector, "last_debug_info", None):
                        print(
                            "policy_intervention_debug="
                            + json.dumps(collector.last_debug_info, ensure_ascii=False)
                        )
                attempted_episode_idx += 1
            return

        if args.collector == "policy_manual_intervention":
            from sim.maniskill.policy_manual_intervention_collector import (
                PolicyManualInterventionCollector,
            )

            ckpt_path = _resolve_repo_path(args.ckpt_path)
            if ckpt_path is None:
                raise ValueError("--collector policy_manual_intervention requires --ckpt-path.")

            collector = PolicyManualInterventionCollector(
                env,
                policy_kind=args.policy_kind,
                ckpt_path=str(ckpt_path),
                action_type=resolved_runtime["action_type"],
                debug=args.debug_policy_manual_intervention,
            )
            saved_episode_idx = 0
            attempted_episode_idx = 0
            while saved_episode_idx < args.episode_num:
                reset_seed = args.seed + attempted_episode_idx
                success, episode_length, status = collector.collect_episode(
                    recorder=recorder,
                    seed=reset_seed,
                    max_episode_steps=args.max_episode_steps,
                )
                replay_debug = None
                replay_success = None
                if episode_length > 0 and success:
                    try:
                        replay_success, replay_debug = _validate_action_replay(
                            task_name=runtime_task_name,
                            resolved_runtime=resolved_runtime,
                            seed=reset_seed,
                            actions=_recorder_actions(recorder),
                            max_episode_steps=args.max_episode_steps,
                        )
                    except Exception as exc:
                        replay_success = False
                        replay_debug = {
                            "seed": int(reset_seed),
                            "success": 0,
                            "error": repr(exc),
                        }
                    if not replay_success:
                        status = f"{status}|action_replay_failed"

                should_save = (success and (replay_success is not False)) or not args.save_only_success
                if should_save:
                    if episode_length > 0:
                        out_path = recorder.save_episode(saved_episode_idx)
                        metadata = {
                            "collector": "policy_manual_intervention",
                            "seed": int(reset_seed),
                            "length": int(episode_length),
                            "success": bool(success),
                            "replay_success": None if replay_success is None else bool(replay_success),
                            "status": str(status),
                            "action_replay": replay_debug,
                            "debug": getattr(collector, "last_debug_info", None),
                        }
                        _write_episode_metadata(out_path, metadata)
                        print(
                            f"saved {out_path} "
                            f"(seed={reset_seed}, length={episode_length}, "
                            f"success={int(success)}, "
                            f"replay_success={int(replay_success) if replay_success is not None else 'n/a'}, "
                            f"status={status})"
                        )
                        saved_episode_idx += 1
                    else:
                        print(
                            f"skipped policy_manual_intervention attempt={attempted_episode_idx} "
                            f"(seed={reset_seed}, length=0, success={int(success)}, status={status})"
                        )
                    if args.debug_policy_manual_intervention and getattr(collector, "last_debug_info", None):
                        print(
                            "policy_manual_intervention_debug="
                            + json.dumps(collector.last_debug_info, ensure_ascii=False)
                        )
                else:
                    print(
                        f"skipped policy_manual_intervention attempt={attempted_episode_idx} "
                        f"(seed={reset_seed}, length={episode_length}, success={int(success)}, "
                        f"replay_success={int(replay_success) if replay_success is not None else 'n/a'}, "
                        f"status={status})"
                    )
                    if replay_debug is not None:
                        print(
                            "policy_manual_intervention_replay_debug="
                            + json.dumps(replay_debug, ensure_ascii=False)
                        )
                    if args.debug_policy_manual_intervention and getattr(collector, "last_debug_info", None):
                        print(
                            "policy_manual_intervention_debug="
                            + json.dumps(collector.last_debug_info, ensure_ascii=False)
                        )
                attempted_episode_idx += 1
            return

        teacher_zarr_path, teacher_seed_path = _resolve_teacher_assets(args, resolved_profile_name)
        teacher_actions, teacher_episode_ends = _load_teacher_actions(teacher_zarr_path)
        teacher_seeds = _load_teacher_seeds(teacher_seed_path)

        saved_episode_idx = 0
        teacher_episode_idx = 0
        prev_end = 0
        num_teacher_episodes = len(teacher_episode_ends)

        while saved_episode_idx < args.episode_num and teacher_episode_idx < num_teacher_episodes:
            teacher_end = int(teacher_episode_ends[teacher_episode_idx])
            action_seq = teacher_actions[prev_end:teacher_end]
            prev_end = teacher_end

            reset_seed = (
                teacher_seeds[teacher_episode_idx]
                if teacher_seeds is not None and teacher_episode_idx < len(teacher_seeds)
                else args.seed + teacher_episode_idx
            )
            obs = env.reset(seed=int(reset_seed))
            recorder.start_episode(dict(obs.data))
            maybe_render()

            success = False
            episode_length = 0
            for action in action_seq[: args.max_episode_steps]:
                result = env.step(
                    ActionSpec.from_value(action, action_type=resolved_runtime["action_type"])
                )
                recorder.append_step(action, dict(result.observation.data))
                maybe_render()
                episode_length += 1
                if _to_bool_scalar(result.info.get("success", False)):
                    success = True
                    break
                if result.terminated or result.truncated:
                    break

            if success or not args.save_only_success:
                out_path = recorder.save_episode(saved_episode_idx)
                print(
                    f"saved {out_path} "
                    f"(teacher_episode={teacher_episode_idx}, seed={reset_seed}, "
                    f"length={episode_length}, success={int(success)})"
                )
                saved_episode_idx += 1
            else:
                print(
                    f"skipped teacher_episode={teacher_episode_idx} "
                    f"(seed={reset_seed}, length={episode_length}, success=0)"
                )

            teacher_episode_idx += 1

        if saved_episode_idx < args.episode_num:
            print(
                f"warning: requested {args.episode_num} episodes but only saved {saved_episode_idx} "
                f"from {num_teacher_episodes} teacher episodes"
            )
    finally:
        env.close()


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_collection(args)


if __name__ == "__main__":
    main()
