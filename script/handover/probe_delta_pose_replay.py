#!/usr/bin/env python3
"""Probe whether scripted joint-action demos can be replayed as delta gripper poses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import (
    ManiSkillTaskFactory,
    ManiSkillTrajectoryRecorder,
    apply_task_profile,
    resolve_runtime_control_mode,
    resolve_runtime_task_target,
)
from sim.maniskill.expert.handover_block_solver import HandoverBlockSolver
from sim.maniskill.expert.primitives import to_numpy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", type=str)
    parser.add_argument("--task-profile", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--replay-max-steps", type=int, default=None)
    parser.add_argument("--sim-backend", type=str, default=None)
    parser.add_argument("--render-backend", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="tmp/handover/delta_pose_probe")
    return parser


def _to_bool_scalar(value: Any) -> bool:
    return bool(to_numpy(value).reshape(-1)[0])


def _to_float_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(to_numpy(value), dtype=np.float64).reshape(shape)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _normalize_quat(q1)
    w2, x2, y2, z2 = _normalize_quat(q2)
    return _normalize_quat(
        np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    v = q[1:]
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(v_norm, float(q[0]))
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return (v / v_norm) * angle


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / angle
    half = 0.5 * angle
    return _normalize_quat(
        np.concatenate(
            [
                np.asarray([np.cos(half)], dtype=np.float64),
                axis * np.sin(half),
            ],
            axis=0,
        )
    )


def _pose_delta(prev_pose: np.ndarray, next_pose: np.ndarray) -> np.ndarray:
    prev_pose = np.asarray(prev_pose, dtype=np.float64).reshape(7)
    next_pose = np.asarray(next_pose, dtype=np.float64).reshape(7)
    pos_delta = next_pose[:3] - prev_pose[:3]
    q_prev = _normalize_quat(prev_pose[3:])
    q_next = _normalize_quat(next_pose[3:])
    if float(np.dot(q_prev, q_next)) < 0.0:
        q_next = -q_next
    q_delta = _quat_multiply(q_next, _quat_inverse(q_prev))
    rot_delta = _quat_to_rotvec(q_delta)
    return np.concatenate([pos_delta, rot_delta], axis=0).astype(np.float32)


def _apply_pose_delta(current_pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float64).reshape(7)
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    target_pose = current_pose.copy()
    target_pose[:3] += delta[:3]
    target_pose[3:] = _quat_multiply(_rotvec_to_quat(delta[3:]), current_pose[3:])
    return target_pose


def _eval_info(env_backend) -> dict[str, Any]:
    evaluator = getattr(env_backend.env.unwrapped, "evaluate", None)
    if not callable(evaluator):
        return {}
    raw = evaluator()
    keys = (
        "success",
        "spatial_success",
        "xy_aligned",
        "z_aligned",
        "right_gripper_open",
        "right_ee_clear",
        "box_static",
    )
    return {key: _to_bool_scalar(raw[key]) for key in keys if key in raw}


def _pose_tracking_error(ctrl, targets: dict[int, np.ndarray]) -> dict[str, float]:
    errors: dict[str, float] = {}
    for arm_index, target_pose in targets.items():
        current_pose = np.asarray(ctrl.current_gripper_pose(arm_index), dtype=np.float64)
        pos_error = float(np.linalg.norm(current_pose[:3] - target_pose[:3]))
        quat_error = 1.0 - float(
            np.clip(abs(np.dot(_normalize_quat(current_pose[3:]), _normalize_quat(target_pose[3:]))), 0.0, 1.0)
        )
        prefix = "left" if arm_index == 0 else "right"
        errors[f"{prefix}_pos_error"] = pos_error
        errors[f"{prefix}_quat_error"] = quat_error
    return errors


class DeltaPoseTrajectoryRecorder:
    """Recorder wrapper that stores delta gripper pose actions instead of joint actions."""

    def __init__(self, recorder: ManiSkillTrajectoryRecorder, solver: HandoverBlockSolver) -> None:
        self._recorder = recorder
        self._solver = solver
        self._prev_gripper_pose: dict[int, np.ndarray] = {}
        self.delta_actions: list[np.ndarray] = []
        self.delta_norms: list[np.ndarray] = []

    def _read_gripper_poses(self) -> dict[int, np.ndarray]:
        return {
            arm_index: np.asarray(self._solver.ctrl.current_gripper_pose(arm_index), dtype=np.float64).reshape(7)
            for arm_index in (0, 1)
        }

    def start_episode(self, initial_obs: dict[str, Any]) -> None:
        self._recorder.start_episode(initial_obs)
        self._prev_gripper_pose = self._read_gripper_poses()
        self.delta_actions = []
        self.delta_norms = []

    def append_step(self, action: Any, next_obs: dict[str, Any]) -> None:
        joint_action = np.asarray(action, dtype=np.float32).reshape(14)
        next_gripper_pose = self._read_gripper_poses()
        left_delta = _pose_delta(self._prev_gripper_pose[0], next_gripper_pose[0])
        right_delta = _pose_delta(self._prev_gripper_pose[1], next_gripper_pose[1])
        delta_action = np.concatenate(
            [
                left_delta,
                np.asarray([joint_action[6]], dtype=np.float32),
                right_delta,
                np.asarray([joint_action[13]], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        self.delta_actions.append(delta_action)
        self.delta_norms.append(
            np.asarray(
                [
                    np.linalg.norm(left_delta[:3]),
                    np.linalg.norm(left_delta[3:]),
                    np.linalg.norm(right_delta[:3]),
                    np.linalg.norm(right_delta[3:]),
                ],
                dtype=np.float32,
            )
        )
        self._recorder.append_step(delta_action, next_obs)
        self._prev_gripper_pose = next_gripper_pose

    def save_episode(self, episode_idx: int) -> Path:
        return self._recorder.save_episode(episode_idx)


def _resolve_runtime(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    runtime_env_id = args.task_name
    if args.task_profile is not None and args.task_name == args.task_profile:
        runtime_env_id = None
    runtime_task_name, resolved_profile_name = resolve_runtime_task_target(
        task_name=args.task_name,
        task_profile=args.task_profile,
        env_id=runtime_env_id,
    )
    runtime_kwargs = {
        "action_type": None,
        "obs_mode": None,
        "control_mode": None,
        "render_mode": None,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "reward_mode": None,
        "robot_uids": None,
        "sim_freq": None,
        "control_freq": None,
        "max_episode_steps": args.max_episode_steps,
    }
    _resolved_action_type, resolved_control_mode = resolve_runtime_control_mode(
        action_type=runtime_kwargs["action_type"],
        control_mode=runtime_kwargs["control_mode"],
    )
    runtime_kwargs["control_mode"] = resolved_control_mode
    profile_runtime_key = resolved_profile_name or runtime_task_name
    _, resolved_runtime, _profile_name = apply_task_profile(profile_runtime_key, runtime_kwargs)
    return runtime_task_name, resolved_runtime


def _make_env(task_name: str, runtime: dict[str, Any]):
    return ManiSkillTaskFactory().create_env(task_name=task_name, **runtime)


def _delta_action_to_joint_action(ctrl, delta_action: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    delta_action = np.asarray(delta_action, dtype=np.float64).reshape(14)
    debug: dict[str, Any] = {"ik": {}, "planner": {}, "source": {}}
    target_poses: dict[int, np.ndarray] = {}
    arm_qpos: dict[int, np.ndarray] = {}

    for arm_index, offset in ((0, 0), (1, 7)):
        current_gripper_pose = np.asarray(ctrl.current_gripper_pose(arm_index), dtype=np.float64)
        target_gripper_pose = _apply_pose_delta(current_gripper_pose, delta_action[offset : offset + 6])
        target_planner_pose = ctrl.gripper_pose_to_planner_pose(arm_index, target_gripper_pose)
        ik_qpos, ik_reason = ctrl.ik_to_pose_debug(arm_index, target_planner_pose, threshold=1e-3)
        arm_name = "left" if arm_index == 0 else "right"
        debug["ik"][arm_name] = ik_reason
        if ik_qpos is not None:
            arm_qpos[arm_index] = ctrl._arm_qpos_from_plan(ik_qpos)
            debug["source"][arm_name] = "ik"
        else:
            # MPLib IK is not reliable at the current UR5 home pose, while the
            # planner can still solve the same tiny target. Keep IK statistics,
            # but use this fallback so the probe can test full trajectory replay.
            plan_result, plan_reason = ctrl.plan_to_pose_debug(arm_index, target_planner_pose)
            debug["planner"][arm_name] = plan_reason
            if plan_result is None:
                return None, debug
            positions = np.asarray(plan_result["position"], dtype=np.float64)
            arm_qpos[arm_index] = ctrl._arm_qpos_from_plan(positions[-1])
            debug["source"][arm_name] = "planner"
        target_poses[arm_index] = target_gripper_pose

    joint_action = ctrl.compose_action(
        left_qpos=arm_qpos[0],
        left_gripper=float(delta_action[6]),
        right_qpos=arm_qpos[1],
        right_gripper=float(delta_action[13]),
    )
    ctrl.commanded_arm_qpos[0] = arm_qpos[0].astype(np.float64, copy=True)
    ctrl.commanded_gripper[0] = float(delta_action[6])
    ctrl.commanded_arm_qpos[1] = arm_qpos[1].astype(np.float64, copy=True)
    ctrl.commanded_gripper[1] = float(delta_action[13])
    debug["target_poses"] = target_poses
    return joint_action, debug


def _replay_delta_episode(
    *,
    task_name: str,
    runtime: dict[str, Any],
    seed: int,
    delta_actions: np.ndarray,
    max_episode_steps: int,
    replay_max_steps: int | None = None,
) -> dict[str, Any]:
    env = _make_env(task_name, runtime)
    solver = HandoverBlockSolver(env)
    try:
        env.reset(seed=int(seed))
        solver.ctrl.reset_episode(max_episode_steps)
        ik_successes = 0
        ik_attempts = 0
        adapter_successes = 0
        adapter_attempts = 0
        planner_fallbacks = 0
        pose_errors: list[dict[str, float]] = []
        terminated = False
        truncated = False
        success = False
        failure_reason = None

        step_limit = int(max_episode_steps)
        if replay_max_steps is not None:
            step_limit = min(step_limit, int(replay_max_steps))

        for step_idx, delta_action in enumerate(np.asarray(delta_actions, dtype=np.float32)[:step_limit]):
            joint_action, debug = _delta_action_to_joint_action(solver.ctrl, delta_action)
            ik_attempts += 2
            adapter_attempts += 2
            if joint_action is None:
                failure_reason = f"adapter_failed_at_step_{step_idx}:{debug}"
                break
            sources = debug.get("source", {})
            ik_successes += sum(1 for source in sources.values() if source == "ik")
            planner_fallbacks += sum(1 for source in sources.values() if source == "planner")
            adapter_successes += 2
            target_poses = debug.get("target_poses", {})
            result = env.step(ActionSpec.from_value(joint_action, action_type=runtime.get("action_type")))
            pose_errors.append(_pose_tracking_error(solver.ctrl, target_poses))
            success = _to_bool_scalar(result.info.get("success", False))
            terminated = bool(result.terminated)
            truncated = bool(result.truncated)
            if success or terminated or truncated:
                break

        final_info = _eval_info(env)
        success = bool(final_info.get("success", success))
        pos_errors = [
            max(item.get("left_pos_error", 0.0), item.get("right_pos_error", 0.0))
            for item in pose_errors
        ]
        quat_errors = [
            max(item.get("left_quat_error", 0.0), item.get("right_quat_error", 0.0))
            for item in pose_errors
        ]
        return {
            "success": bool(success),
            "steps": int(len(pose_errors)),
            "requested_replay_steps": int(step_limit),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_reason": failure_reason,
            "ik_success_fraction": float(ik_successes / max(1, ik_attempts)),
            "adapter_success_fraction": float(adapter_successes / max(1, adapter_attempts)),
            "planner_fallback_fraction": float(planner_fallbacks / max(1, adapter_attempts)),
            "pose_pos_error_mean": float(np.mean(pos_errors)) if pos_errors else None,
            "pose_pos_error_max": float(np.max(pos_errors)) if pos_errors else None,
            "pose_quat_error_mean": float(np.mean(quat_errors)) if quat_errors else None,
            "pose_quat_error_max": float(np.max(quat_errors)) if quat_errors else None,
            "final_info": final_info,
        }
    finally:
        env.close()


def _delta_stats(delta_actions: list[np.ndarray], delta_norms: list[np.ndarray]) -> dict[str, Any]:
    if not delta_actions:
        return {}
    actions = np.asarray(delta_actions, dtype=np.float32)
    norms = np.asarray(delta_norms, dtype=np.float32)
    return {
        "action_shape": list(actions.shape),
        "has_nan": bool(np.isnan(actions).any()),
        "abs_max": float(np.max(np.abs(actions))),
        "left_pos_norm_max": float(np.max(norms[:, 0])),
        "left_rot_norm_max": float(np.max(norms[:, 1])),
        "right_pos_norm_max": float(np.max(norms[:, 2])),
        "right_rot_norm_max": float(np.max(norms[:, 3])),
        "left_pos_norm_mean": float(np.mean(norms[:, 0])),
        "left_rot_norm_mean": float(np.mean(norms[:, 1])),
        "right_pos_norm_mean": float(np.mean(norms[:, 2])),
        "right_rot_norm_mean": float(np.mean(norms[:, 3])),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    task_name, runtime = _resolve_runtime(args)
    max_attempts = int(args.max_attempts) if args.max_attempts is not None else int(args.episodes) * 5
    env = _make_env(task_name, runtime)
    solver = HandoverBlockSolver(env)
    saved = 0
    attempted = 0
    episode_summaries: list[dict[str, Any]] = []
    all_delta_actions: list[np.ndarray] = []
    all_delta_norms: list[np.ndarray] = []

    try:
        while saved < int(args.episodes) and attempted < max_attempts:
            seed = int(args.seed) + attempted
            obs = env.reset(seed=seed)
            base_recorder = ManiSkillTrajectoryRecorder(output_dir)
            recorder = DeltaPoseTrajectoryRecorder(base_recorder, solver)
            recorder.start_episode(dict(obs.data))
            expert_success, expert_length, expert_status = solver.solve_episode(
                recorder=recorder,
                max_episode_steps=int(args.max_episode_steps),
            )
            delta_actions = np.asarray(recorder.delta_actions, dtype=np.float32)
            replay_result = None
            if expert_success and len(delta_actions) > 0:
                out_path = recorder.save_episode(saved)
                replay_result = _replay_delta_episode(
                    task_name=task_name,
                    runtime=runtime,
                    seed=seed,
                    delta_actions=delta_actions,
                    max_episode_steps=int(args.max_episode_steps),
                    replay_max_steps=args.replay_max_steps,
                )
                episode_summary = {
                    "episode": int(saved),
                    "seed": int(seed),
                    "path": str(out_path),
                    "expert_success": bool(expert_success),
                    "expert_length": int(expert_length),
                    "expert_status": str(expert_status),
                    "delta_stats": _delta_stats(recorder.delta_actions, recorder.delta_norms),
                    "delta_replay": replay_result,
                }
                (out_path.with_suffix(".json")).write_text(
                    json.dumps(episode_summary, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                episode_summaries.append(episode_summary)
                all_delta_actions.extend(recorder.delta_actions)
                all_delta_norms.extend(recorder.delta_norms)
                print(
                    f"saved {out_path} seed={seed} "
                    f"expert_success={int(expert_success)} "
                    f"delta_replay_success={int(bool(replay_result.get('success', False)))} "
                    f"ik_frac={replay_result.get('ik_success_fraction'):.3f}"
                )
                saved += 1
            else:
                print(
                    f"skipped attempt={attempted} seed={seed} "
                    f"expert_success={int(expert_success)} length={expert_length} status={expert_status}"
                )
            attempted += 1
    finally:
        env.close()

    replay_successes = [
        bool(item.get("delta_replay", {}).get("success", False))
        for item in episode_summaries
    ]
    ik_fracs = [
        float(item.get("delta_replay", {}).get("ik_success_fraction", 0.0))
        for item in episode_summaries
    ]
    summary = {
        "task_name": task_name,
        "task_profile": args.task_profile,
        "requested_episodes": int(args.episodes),
        "saved_episodes": int(saved),
        "attempted_episodes": int(attempted),
        "expert_success_count": int(saved),
        "delta_replay_success_count": int(sum(replay_successes)),
        "delta_replay_success_rate": float(np.mean(replay_successes)) if replay_successes else 0.0,
        "ik_success_fraction_mean": float(np.mean(ik_fracs)) if ik_fracs else 0.0,
        "delta_stats_all": _delta_stats(all_delta_actions, all_delta_norms),
        "episodes": episode_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
