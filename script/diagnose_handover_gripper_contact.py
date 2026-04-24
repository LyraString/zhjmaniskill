import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.maniskill import ManiSkillTaskFactory, apply_task_profile, resolve_task_profile
from sim.maniskill.expert.primitives import FixedRateMotionPrimitives
from sim.maniskill.tasks.handover_block_env import BOX_HALF_SIZE
from sim.maniskill.tasks import dual_ur5_wsg_env as dual_ur5_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose whether left-gripper finger contact happens before teacher close during state replay."
    )
    parser.add_argument("--task-profile", type=str, default="handover_block_native_legacy_rng_smoke")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--close-threshold", type=float, default=0.2)
    parser.add_argument("--near-threshold", type=float, default=0.01)
    parser.add_argument("--box-motion-threshold", type=float, default=0.005)
    parser.add_argument(
        "--legacy-demo-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "handover_block" / "demo_clean"),
    )
    parser.add_argument("--teacher-seed-path", type=str, default=None)
    parser.add_argument("--sim-backend", type=str, default=None)
    parser.add_argument("--render-backend", type=str, default=None)
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--disable-nonfinger-gripper-collision",
        action="store_true",
        help="Diagnostic only: remove wsg_50_base_link/gripper_left/gripper_right collision from a temporary URDF.",
    )
    parser.add_argument(
        "--shrink-finger-collision",
        action="store_true",
        help="Diagnostic only: scale finger_left/finger_right collision meshes in a temporary URDF.",
    )
    parser.add_argument(
        "--finger-collision-scale",
        type=float,
        default=0.8,
        help="Scale factor used with --shrink-finger-collision.",
    )
    parser.add_argument(
        "--use-finger-box-collision-proxy",
        action="store_true",
        help="Diagnostic only: replace finger_left/finger_right collision meshes with simple box proxies.",
    )
    parser.add_argument(
        "--finger-box-collision-scale",
        type=float,
        default=1.0,
        help="Scale factor used with --use-finger-box-collision-proxy.",
    )
    return parser


def _resolve_repo_path(path_str: str | None) -> Path | None:
    if path_str in (None, ""):
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_seed_path(profile_name: str, seed_path_arg: str | None) -> Path:
    seed_path = _resolve_repo_path(seed_path_arg)
    if seed_path is not None:
        return seed_path
    profile = resolve_task_profile(profile_name)
    if profile is None:
        raise KeyError(f"Unknown task profile: {profile_name}")
    teacher_cfg = (profile or {}).get("teacher_dataset", {})
    seed_path = _resolve_repo_path(teacher_cfg.get("seed_path"))
    if seed_path is None:
        raise ValueError(f"Task profile `{profile_name}` does not define teacher_dataset.seed_path")
    return seed_path


def _load_teacher_seeds(seed_path: Path) -> list[int]:
    raw = seed_path.read_text(encoding="utf-8").split()
    if not raw:
        raise ValueError(f"No seeds found in {seed_path}")
    return [int(value) for value in raw]


def _load_legacy_demo(hdf5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as root:
        left_gripper = np.asarray(root["endpose/left_gripper"], dtype=np.float64)
        joint_vector = np.asarray(root["joint_action/vector"], dtype=np.float64)
    return {
        "left_gripper": left_gripper,
        "joint_vector": joint_vector,
    }


def _format_float(value: float) -> str:
    return f"{value:.8g}"


def _scale_collision_mesh(collision: ET.Element, scale: float) -> bool:
    geometry = collision.find("geometry")
    if geometry is None:
        return False
    mesh = geometry.find("mesh")
    if mesh is None:
        return False

    raw_scale = mesh.get("scale")
    if raw_scale:
        values = [float(value) for value in raw_scale.split()]
        if len(values) == 1:
            values = values * 3
        if len(values) != 3:
            raise RuntimeError(f"Unexpected URDF mesh scale `{raw_scale}`")
        values = [value * scale for value in values]
    else:
        values = [scale, scale, scale]
    mesh.set("scale", " ".join(_format_float(value) for value in values))
    return True


def _replace_link_collisions_with_box(
    link: ET.Element,
    *,
    origin_xyz: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
) -> None:
    for collision in list(link.findall("collision")):
        link.remove(collision)

    collision = ET.SubElement(link, "collision")
    ET.SubElement(
        collision,
        "origin",
        {
            "xyz": " ".join(_format_float(value) for value in origin_xyz),
            "rpy": "0 0 0",
        },
    )
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(
        geometry,
        "box",
        {"size": " ".join(_format_float(value) for value in size_xyz)},
    )


def _make_diagnostic_robot_dir(
    source_robot_dir: Path,
    *,
    disable_nonfinger_collision: bool,
    finger_collision_scale: float | None,
    use_finger_box_collision_proxy: bool,
    finger_box_collision_scale: float,
) -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory(prefix="robotwin_ur5_wsg_diag_")
    temp_robot_dir = Path(temp_dir.name)

    config = yaml.safe_load((source_robot_dir / "config.yml").read_text(encoding="utf-8"))
    source_urdf = (source_robot_dir / config["urdf_path"]).resolve()
    suffixes = []
    if disable_nonfinger_collision:
        suffixes.append("no_nonfinger_collision")
    if finger_collision_scale is not None:
        suffixes.append(f"finger_collision_scale_{_format_float(finger_collision_scale).replace('.', 'p')}")
    if use_finger_box_collision_proxy:
        suffixes.append(f"finger_box_collision_proxy_{_format_float(finger_box_collision_scale).replace('.', 'p')}")
    patched_urdf_name = f"{source_urdf.stem}_{'_'.join(suffixes)}.urdf"
    patched_urdf = temp_robot_dir / patched_urdf_name

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    if disable_nonfinger_collision:
        for link_name in ("wsg_50_base_link", "gripper_left", "gripper_right"):
            link = root.find(f"./link[@name='{link_name}']")
            if link is None:
                raise RuntimeError(f"Failed to find link `{link_name}` in {source_urdf}")
            for collision in list(link.findall("collision")):
                link.remove(collision)

    if use_finger_box_collision_proxy:
        scale = float(finger_box_collision_scale)
        if scale <= 0.0:
            raise ValueError("--finger-box-collision-scale must be positive")
        base_origin = (-0.0085, 0.0, 0.094)
        base_size = (0.023, 0.03, 0.073)
        for link_name in ("finger_left", "finger_right"):
            link = root.find(f"./link[@name='{link_name}']")
            if link is None:
                raise RuntimeError(f"Failed to find link `{link_name}` in {source_urdf}")
            _replace_link_collisions_with_box(
                link,
                origin_xyz=base_origin,
                size_xyz=tuple(value * scale for value in base_size),
            )
    elif finger_collision_scale is not None:
        if finger_collision_scale <= 0.0:
            raise ValueError("--finger-collision-scale must be positive")
        scaled_count = 0
        for link_name in ("finger_left", "finger_right"):
            link = root.find(f"./link[@name='{link_name}']")
            if link is None:
                raise RuntimeError(f"Failed to find link `{link_name}` in {source_urdf}")
            for collision in link.findall("collision"):
                if _scale_collision_mesh(collision, finger_collision_scale):
                    scaled_count += 1
        if scaled_count == 0:
            raise RuntimeError("No finger collision mesh was found to scale")
    tree.write(patched_urdf, encoding="utf-8", xml_declaration=True)

    config["urdf_path"] = f"./{patched_urdf_name}"
    (temp_robot_dir / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    for name in ("meshes", "ur5.srdf"):
        source_path = source_robot_dir / name
        target_path = temp_robot_dir / name
        if source_path.exists():
            try:
                target_path.symlink_to(source_path, target_is_directory=source_path.is_dir())
            except OSError:
                if source_path.is_dir():
                    shutil.copytree(source_path, target_path)
                else:
                    shutil.copy2(source_path, target_path)
    return temp_dir


def _to_numpy(value, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if shape is not None:
        array = array.reshape(shape)
    return array


def _joint_name(joint) -> str:
    if hasattr(joint, "get_name"):
        return joint.get_name()
    return str(joint.name)


def _link_name(link) -> str:
    if hasattr(link, "get_name"):
        return link.get_name()
    return str(link.name)


def _validate_joint_order(raw_env) -> None:
    for arm_index, robot, arm_joints, gripper_joints in (
        (0, raw_env.left_robot, raw_env.left_arm_joints, raw_env.left_gripper),
        (1, raw_env.right_robot, raw_env.right_arm_joints, raw_env.right_gripper),
    ):
        active_names = [_joint_name(joint) for joint in robot.get_active_joints()]
        expected_names = [_joint_name(joint) for joint in arm_joints] + [
            _joint_name(joint) for joint, _, _ in gripper_joints
        ]
        if active_names[: len(expected_names)] != expected_names:
            raise RuntimeError(
                f"Unexpected active joint order for arm {arm_index}: "
                f"expected prefix {expected_names}, got {active_names}"
            )


def _apply_teacher_robot_state(raw_env, arm_index: int, arm_qpos: np.ndarray, gripper_value: float) -> None:
    robot = raw_env.left_robot if arm_index == 0 else raw_env.right_robot
    arm_joints = raw_env.left_arm_joints if arm_index == 0 else raw_env.right_arm_joints
    gripper_joints = raw_env.left_gripper if arm_index == 0 else raw_env.right_gripper

    qpos = robot.get_qpos().clone()
    qvel = robot.get_qvel().clone()
    qvel.zero_()

    active_joints = robot.get_active_joints()
    name_to_index = {_joint_name(joint): idx for idx, joint in enumerate(active_joints)}

    arm_qpos = np.asarray(arm_qpos, dtype=np.float64).reshape(len(arm_joints))
    for joint, target in zip(arm_joints, arm_qpos):
        qpos[:, name_to_index[_joint_name(joint)]] = float(target)

    gripper_scale = raw_env.embodiment_spec.gripper_scale(arm_index)
    real_gripper_val = float(gripper_scale[0] + gripper_value * (gripper_scale[1] - gripper_scale[0]))
    for joint, sign, offset in gripper_joints:
        qpos[:, name_to_index[_joint_name(joint)]] = real_gripper_val * float(sign) + float(offset)

    robot.set_qpos(qpos)
    robot.set_qvel(qvel)
    robot.set_joint_drive_targets(qpos[:, : len(arm_joints)], joints=arm_joints)
    robot.set_joint_drive_velocity_targets(
        torch.zeros((1, len(arm_joints)), device=raw_env.device, dtype=qvel.dtype),
        joints=arm_joints,
    )
    raw_env._set_gripper(torch.tensor([gripper_value], device=raw_env.device, dtype=torch.float32), arm_index=arm_index)


def _advance_one_control_step(raw_env) -> None:
    if raw_env.gpu_sim_enabled:
        raw_env.scene._gpu_apply_all()
    raw_env._before_control_step()
    for _ in range(raw_env._sim_steps_per_control):
        raw_env._before_simulation_step()
        raw_env.scene.step()
        raw_env._after_simulation_step()
    raw_env._after_control_step()
    if raw_env.gpu_sim_enabled:
        raw_env.scene._gpu_fetch_all()


def _find_link(robot, link_name: str):
    if hasattr(robot, "find_link_by_name"):
        link = robot.find_link_by_name(link_name)
        if link is not None:
            return link
    for link in robot.get_links():
        if _link_name(link) == link_name:
            return link
    raise RuntimeError(f"Failed to find link `{link_name}`")


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _point_to_box_metrics(point_world: np.ndarray, box_pose: np.ndarray, half_size: np.ndarray) -> dict[str, object]:
    box_center = np.asarray(box_pose[:3], dtype=np.float64)
    box_quat = np.asarray(box_pose[3:7], dtype=np.float64)
    rot = _quat_to_matrix(box_quat)
    point_local = rot.T @ (np.asarray(point_world, dtype=np.float64) - box_center)
    outside = np.maximum(np.abs(point_local) - half_size, 0.0)
    surface_distance = float(np.linalg.norm(outside))
    signed_margin = float(np.min(half_size - np.abs(point_local)))
    return {
        "local_pos": point_local.tolist(),
        "surface_distance": surface_distance,
        "signed_margin": signed_margin,
    }


def _matches_box_name(name: str) -> bool:
    return "box" in name.lower()


def _is_static_support_name(name: str) -> bool:
    lowered = name.lower()
    return "table" in lowered or "workspace" in lowered or "ground" in lowered or "wall" in lowered


def _extract_box_finger_contacts(
    raw_env, finger_names: tuple[str, str]
) -> tuple[dict[str, dict[str, object] | None], list[dict[str, object]], list[dict[str, object]], bool]:
    result: dict[str, dict[str, object] | None] = {name: None for name in finger_names}
    box_contacts: list[dict[str, object]] = []
    all_contacts: list[dict[str, object]] = []
    if not hasattr(raw_env.scene, "get_contacts"):
        return result, box_contacts, all_contacts, False
    try:
        contacts = raw_env.scene.get_contacts()
    except AttributeError:
        return result, box_contacts, all_contacts, False
    for contact in contacts:
        if not hasattr(contact, "bodies") or len(contact.bodies) < 2:
            continue
        name0 = contact.bodies[0].entity.name
        name1 = contact.bodies[1].entity.name
        point_positions = []
        for point in getattr(contact, "points", []):
            point_positions.append(np.round(_to_numpy(point.position).reshape(3), 5).tolist())
        all_contacts.append(
            {
                "pair": [name0, name1],
                "point_count": len(point_positions),
                "point_positions": point_positions[:4],
            }
        )
        name0_is_box = _matches_box_name(name0)
        name1_is_box = _matches_box_name(name1)
        if not (name0_is_box or name1_is_box):
            continue
        other_name = name1 if name0_is_box else name0
        box_contact = {
            "other": other_name,
            "pair": [name0, name1],
            "point_count": len(point_positions),
            "point_positions": point_positions[:4],
            "is_static_support": _is_static_support_name(other_name),
        }
        box_contacts.append(box_contact)
        if other_name not in result:
            continue
        result[other_name] = {
            "point_count": len(point_positions),
            "point_positions": point_positions[:4],
        }
    return result, box_contacts, all_contacts, True


def _capture_state(raw_env, ctrl: FixedRateMotionPrimitives, finger_links: dict[str, object]) -> dict[str, object]:
    box_pose = _to_numpy(raw_env.box.pose.raw_pose, (-1, 7))[0].astype(np.float64, copy=False)
    finger_pose = {
        name: _to_numpy(link.pose.raw_pose, (-1, 7))[0].astype(np.float64, copy=False)
        for name, link in finger_links.items()
    }
    contacts, box_contacts, all_contacts, contact_api_available = _extract_box_finger_contacts(
        raw_env, tuple(finger_links.keys())
    )
    return {
        "left_gripper": float(ctrl.current_gripper_value(0)),
        "box_pose": box_pose,
        "finger_pose": finger_pose,
        "contacts": contacts,
        "box_contacts": box_contacts,
        "all_contacts": all_contacts,
        "contact_api_available": bool(contact_api_available),
    }


def _first_index(records: list[dict[str, object]], predicate):
    for idx, record in enumerate(records):
        if predicate(record):
            return idx
    return None


def main() -> None:
    args = build_parser().parse_args()
    if args.shrink_finger_collision and args.use_finger_box_collision_proxy:
        raise ValueError("--shrink-finger-collision and --use-finger-box-collision-proxy are mutually exclusive")
    if args.shrink_finger_collision and args.finger_collision_scale <= 0.0:
        raise ValueError("--finger-collision-scale must be positive")
    if args.use_finger_box_collision_proxy and args.finger_box_collision_scale <= 0.0:
        raise ValueError("--finger-box-collision-scale must be positive")

    seed_path = _load_seed_path(args.task_profile, args.teacher_seed_path)
    teacher_seeds = _load_teacher_seeds(seed_path)
    reset_seed = teacher_seeds[args.episode_index]

    legacy_demo_dir = _resolve_repo_path(args.legacy_demo_dir)
    if legacy_demo_dir is None:
        raise ValueError("--legacy-demo-dir resolved to None")
    hdf5_path = legacy_demo_dir / "data" / f"episode{args.episode_index}.hdf5"
    old_demo = _load_legacy_demo(hdf5_path)

    temp_robot_dir = None
    original_ur5_wsg_dir = dual_ur5_module.UR5_WSG_DIR
    if (
        args.disable_nonfinger_gripper_collision
        or args.shrink_finger_collision
        or args.use_finger_box_collision_proxy
    ):
        temp_robot_dir = _make_diagnostic_robot_dir(
            original_ur5_wsg_dir,
            disable_nonfinger_collision=bool(args.disable_nonfinger_gripper_collision),
            finger_collision_scale=float(args.finger_collision_scale) if args.shrink_finger_collision else None,
            use_finger_box_collision_proxy=bool(args.use_finger_box_collision_proxy),
            finger_box_collision_scale=float(args.finger_box_collision_scale),
        )
        dual_ur5_module.UR5_WSG_DIR = Path(temp_robot_dir.name)

    runtime_kwargs = {
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "render_mode": args.render_mode,
    }
    _, resolved_runtime, _ = apply_task_profile(args.task_profile, runtime_kwargs)

    records: list[dict[str, object]] = []
    env = None
    try:
        env = ManiSkillTaskFactory().create_env(task_name=args.task_profile, **resolved_runtime)
        env.reset(seed=reset_seed)
        raw_env = env.env.unwrapped
        _validate_joint_order(raw_env)
        ctrl = FixedRateMotionPrimitives(env)
        finger_links = {
            "finger_left": _find_link(raw_env.left_robot, "finger_left"),
            "finger_right": _find_link(raw_env.left_robot, "finger_right"),
        }
        initial_box_pose = _to_numpy(raw_env.box.pose.raw_pose, (-1, 7))[0]
        box_half_size = _to_numpy(BOX_HALF_SIZE, (3,))

        max_steps = min(int(args.max_steps), len(old_demo["joint_vector"]))
        for step_idx in range(max_steps):
            teacher_vector = np.asarray(old_demo["joint_vector"][step_idx], dtype=np.float64)
            _apply_teacher_robot_state(raw_env, 0, teacher_vector[:6], float(teacher_vector[6]))
            _apply_teacher_robot_state(raw_env, 1, teacher_vector[7:13], float(teacher_vector[13]))
            _advance_one_control_step(raw_env)
            if getattr(env, "render_mode", None) == "human":
                env.render()

            current = _capture_state(raw_env, ctrl, finger_links)
            box_pose = np.asarray(current["box_pose"], dtype=np.float64)
            finger_pose = {
                name: np.asarray(pose, dtype=np.float64)
                for name, pose in current["finger_pose"].items()
            }
            finger_metrics = {
                name: _point_to_box_metrics(pose[:3], box_pose, box_half_size)
                for name, pose in finger_pose.items()
            }

            records.append(
                {
                    "idx": step_idx,
                    "teacher_left_gripper": float(old_demo["left_gripper"][step_idx]),
                    "current_left_gripper": float(current["left_gripper"]),
                    "current_box_pose": box_pose.tolist(),
                    "finger_left_pose": finger_pose["finger_left"].tolist(),
                    "finger_right_pose": finger_pose["finger_right"].tolist(),
                    "finger_left_metrics": finger_metrics["finger_left"],
                    "finger_right_metrics": finger_metrics["finger_right"],
                    "finger_left_contact": current["contacts"]["finger_left"],
                    "finger_right_contact": current["contacts"]["finger_right"],
                    "box_contacts": current["box_contacts"],
                    "all_contacts": current["all_contacts"],
                    "contact_api_available": bool(current["contact_api_available"]),
                    "box_disp": float(np.linalg.norm(box_pose[:3] - initial_box_pose[:3])),
                }
            )
    finally:
        if env is not None:
            env.close()
        dual_ur5_module.UR5_WSG_DIR = original_ur5_wsg_dir
        if temp_robot_dir is not None:
            temp_robot_dir.cleanup()

    first_teacher_close_idx = _first_index(records, lambda r: float(r["teacher_left_gripper"]) < args.close_threshold)
    first_current_close_idx = _first_index(records, lambda r: float(r["current_left_gripper"]) < args.close_threshold)
    first_box_motion_idx = _first_index(records, lambda r: float(r["box_disp"]) > args.box_motion_threshold)
    first_finger_left_near_idx = _first_index(
        records, lambda r: float(r["finger_left_metrics"]["surface_distance"]) <= args.near_threshold
    )
    first_finger_right_near_idx = _first_index(
        records, lambda r: float(r["finger_right_metrics"]["surface_distance"]) <= args.near_threshold
    )
    first_finger_left_contact_idx = _first_index(records, lambda r: r["finger_left_contact"] is not None)
    first_finger_right_contact_idx = _first_index(records, lambda r: r["finger_right_contact"] is not None)
    first_box_contact_idx = _first_index(records, lambda r: bool(r["box_contacts"]))
    first_robot_box_contact_idx = _first_index(
        records, lambda r: any(not bool(contact.get("is_static_support")) for contact in r["box_contacts"])
    )

    if first_teacher_close_idx is None:
        diagnosis = "teacher_never_closed"
    elif first_finger_left_contact_idx is not None and first_finger_left_contact_idx < first_teacher_close_idx:
        diagnosis = "finger_left_contacts_box_before_teacher_close"
    elif first_finger_right_contact_idx is not None and first_finger_right_contact_idx < first_teacher_close_idx:
        diagnosis = "finger_right_contacts_box_before_teacher_close"
    elif first_box_contact_idx is not None and first_box_contact_idx < first_teacher_close_idx:
        diagnosis = "non_finger_link_contacts_box_before_teacher_close"
    elif first_box_motion_idx is not None and first_box_motion_idx < first_teacher_close_idx:
        diagnosis = "box_moves_before_teacher_close_without_detected_finger_contact"
    else:
        diagnosis = "no_early_finger_contact_detected"

    center_idx = first_teacher_close_idx if first_teacher_close_idx is not None else min(len(records) - 1, max_steps // 2)
    window_start = max(0, center_idx - 20)
    window_end = min(len(records) - 1, center_idx + 20)

    summary = {
        "task_profile": args.task_profile,
        "episode_index": int(args.episode_index),
        "seed": int(reset_seed),
        "legacy_hdf5_path": str(hdf5_path),
        "record_count": len(records),
        "disable_nonfinger_gripper_collision": bool(args.disable_nonfinger_gripper_collision),
        "shrink_finger_collision": bool(args.shrink_finger_collision),
        "use_finger_box_collision_proxy": bool(args.use_finger_box_collision_proxy),
        "finger_collision_scale": float(args.finger_collision_scale) if args.shrink_finger_collision else None,
        "finger_box_collision_scale": float(args.finger_box_collision_scale)
        if args.use_finger_box_collision_proxy
        else None,
        "contact_api_available": bool(records[0]["contact_api_available"]) if records else False,
        "first_teacher_close_idx": first_teacher_close_idx,
        "first_current_close_idx": first_current_close_idx,
        "first_box_motion_idx": first_box_motion_idx,
        "first_finger_left_near_idx": first_finger_left_near_idx,
        "first_finger_right_near_idx": first_finger_right_near_idx,
        "first_finger_left_contact_idx": first_finger_left_contact_idx,
        "first_finger_right_contact_idx": first_finger_right_contact_idx,
        "first_box_contact_idx": first_box_contact_idx,
        "first_robot_box_contact_idx": first_robot_box_contact_idx,
        "diagnosis": diagnosis,
        "window_start": window_start,
        "window_end": window_end,
    }

    print(f"episode={summary['episode_index']} seed={summary['seed']} records={summary['record_count']}")
    print(
        "events "
        f"contact_api_available={summary['contact_api_available']} "
        f"teacher_close={first_teacher_close_idx} "
        f"current_close={first_current_close_idx} "
        f"box_motion={first_box_motion_idx} "
        f"finger_left_near={first_finger_left_near_idx} "
        f"finger_right_near={first_finger_right_near_idx} "
        f"finger_left_contact={first_finger_left_contact_idx} "
        f"finger_right_contact={first_finger_right_contact_idx} "
        f"box_contact={first_box_contact_idx} "
        f"robot_box_contact={first_robot_box_contact_idx}"
    )
    print(f"diagnosis={diagnosis}")
    print(
        "window "
        f"[{window_start}, {window_end}] "
        "(idx teacher_g current_g box_disp left_gap right_gap left_ct right_ct box_ct robot_box_ct box_pairs all_pairs box_xyz left_xyz right_xyz)"
    )
    for idx in range(window_start, window_end + 1):
        record = records[idx]
        box_xyz = np.asarray(record["current_box_pose"], dtype=np.float64)[:3]
        left_xyz = np.asarray(record["finger_left_pose"], dtype=np.float64)[:3]
        right_xyz = np.asarray(record["finger_right_pose"], dtype=np.float64)[:3]
        left_gap = float(record["finger_left_metrics"]["surface_distance"])
        right_gap = float(record["finger_right_metrics"]["surface_distance"])
        left_contact = 1 if record["finger_left_contact"] is not None else 0
        right_contact = 1 if record["finger_right_contact"] is not None else 0
        box_contacts = record["box_contacts"]
        box_contact = 1 if box_contacts else 0
        robot_box_contacts = [contact for contact in box_contacts if not bool(contact.get("is_static_support"))]
        robot_box_contact = 1 if robot_box_contacts else 0
        box_pairs = ",".join(str(contact["other"]) for contact in box_contacts[:3]) if box_contacts else "-"
        all_pairs = ";".join(
            f"{contact['pair'][0]}|{contact['pair'][1]}" for contact in record["all_contacts"][:4]
        )
        if not all_pairs:
            all_pairs = "-"
        print(
            f"{idx:03d} "
            f"{float(record['teacher_left_gripper']):.2f} "
            f"{float(record['current_left_gripper']):.2f} "
            f"{float(record['box_disp']):.4f} "
            f"{left_gap:.4f} "
            f"{right_gap:.4f} "
            f"{left_contact:d} "
            f"{right_contact:d} "
            f"{box_contact:d} "
            f"{robot_box_contact:d} "
            f"{box_pairs} "
            f"{all_pairs} "
            f"{np.round(box_xyz, 4).tolist()} "
            f"{np.round(left_xyz, 4).tolist()} "
            f"{np.round(right_xyz, 4).tolist()}"
        )

    if args.output_json:
        output_path = _resolve_repo_path(args.output_json)
        if output_path is None:
            raise ValueError("--output-json resolved to None")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "records": records,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
