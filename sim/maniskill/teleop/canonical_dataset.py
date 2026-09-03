"""Streaming canonical dataset writer for VR teleoperation episodes."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any, Mapping

import av
import h5py
import numpy as np


SCHEMA_NAME = "robotwin_canonical_teleop"
SCHEMA_VERSION = "1.0"
DEFAULT_CAMERA_NAMES = (
    "head_camera",
    "left_wrist_camera",
    "right_wrist_camera",
)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single_env_array(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def _copy_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_tree(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    return np.asarray(value).copy()


def _numeric_extra(raw_extra: Any) -> dict[str, np.ndarray]:
    if not isinstance(raw_extra, Mapping):
        return {}
    result: dict[str, np.ndarray] = {}
    for key, value in raw_extra.items():
        try:
            array = _single_env_array(value)
        except (TypeError, ValueError):
            continue
        if array.dtype.kind not in "biuf":
            continue
        result[str(key)] = array
    return result


def _pose_array(value: Any) -> np.ndarray:
    return _single_env_array(value).astype(np.float32, copy=False).reshape(7)


def capture_canonical_observation(
    raw_obs: Mapping[str, Any],
    base_env: Any,
    *,
    host_monotonic_ns: int | None = None,
    camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
) -> dict[str, Any]:
    """Copy one raw ManiSkill observation into the canonical CPU schema."""

    sensor_data = raw_obs.get("sensor_data")
    sensor_param = raw_obs.get("sensor_param")
    agent = raw_obs.get("agent")
    if not isinstance(sensor_data, Mapping) or not isinstance(sensor_param, Mapping):
        raise KeyError("Raw observation is missing sensor_data or sensor_param.")
    if not isinstance(agent, Mapping) or "qpos" not in agent:
        raise KeyError("Raw observation is missing agent.qpos.")

    cameras: dict[str, dict[str, np.ndarray]] = {}
    for camera_name in camera_names:
        images = sensor_data.get(camera_name)
        params = sensor_param.get(camera_name)
        if not isinstance(images, Mapping) or not isinstance(params, Mapping):
            raise KeyError(f"Raw observation is missing camera `{camera_name}`.")
        required_images = ("rgb", "position", "segmentation")
        required_params = ("intrinsic_cv", "extrinsic_cv", "cam2world_gl")
        missing = [key for key in required_images if key not in images]
        missing += [key for key in required_params if key not in params]
        if missing:
            raise KeyError(f"Camera `{camera_name}` is missing fields: {missing}")
        position = _single_env_array(images["position"])
        if position.dtype != np.int16 or position.shape[-1] != 3:
            raise ValueError(
                f"Camera `{camera_name}` position must be int16 [...,3] millimeters; "
                f"got shape={position.shape}, dtype={position.dtype}."
            )
        cameras[camera_name] = {
            "rgb": _single_env_array(images["rgb"]).astype(np.uint8, copy=False),
            "position_mm": position,
            "segmentation_id": _single_env_array(images["segmentation"])
            .astype(np.int16, copy=False)
            .squeeze(-1),
            "intrinsic_cv": _single_env_array(params["intrinsic_cv"]).astype(
                np.float32, copy=False
            ),
            "extrinsic_cv": _single_env_array(params["extrinsic_cv"]).astype(
                np.float32, copy=False
            ),
            "cam2world_gl": _single_env_array(params["cam2world_gl"]).astype(
                np.float32, copy=False
            ),
        }

    root_pose_world = np.stack(
        [base_env.get_arm_root_pose_array(arm_index) for arm_index in (0, 1)],
        axis=0,
    ).astype(np.float32, copy=False)
    ee_pose_world = np.stack(
        [base_env.get_arm_end_effector_pose_array(arm_index) for arm_index in (0, 1)],
        axis=0,
    ).astype(np.float32, copy=False)
    ee_pose_root = np.stack(
        [
            _pose_array(base_env.left_delta_pose_controller.ee_pose_at_base.raw_pose),
            _pose_array(base_env.right_delta_pose_controller.ee_pose_at_base.raw_pose),
        ],
        axis=0,
    )

    full_qpos = []
    full_qvel = []
    for arm_index in (0, 1):
        robot = base_env.get_arm_robot(arm_index)
        full_qpos.append(_single_env_array(robot.get_qpos()).astype(np.float32, copy=False))
        full_qvel.append(_single_env_array(robot.get_qvel()).astype(np.float32, copy=False))

    return {
        "host_monotonic_ns": np.int64(
            time.monotonic_ns() if host_monotonic_ns is None else host_monotonic_ns
        ),
        "cameras": cameras,
        "robot": {
            "agent_qpos": _single_env_array(agent["qpos"]).astype(np.float32, copy=False),
            "left_full_qpos": full_qpos[0],
            "right_full_qpos": full_qpos[1],
            "left_full_qvel": full_qvel[0],
            "right_full_qvel": full_qvel[1],
            "root_pose_world_wxyz": root_pose_world,
            "ee_pose_root_wxyz": ee_pose_root,
            "ee_pose_world_wxyz": ee_pose_world,
        },
        "task": _numeric_extra(raw_obs.get("extra")),
    }


class _H5SeriesWriter:
    def __init__(self, h5_file: h5py.File) -> None:
        self.h5_file = h5_file

    def append(self, prefix: str, values: Mapping[str, Any]) -> None:
        self._append_mapping(prefix.strip("/"), values)

    def _append_mapping(self, prefix: str, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                self._append_mapping(path, value)
            else:
                self._append_array(path, np.asarray(value))

    def _append_array(self, path: str, value: np.ndarray) -> None:
        if value.dtype.kind not in "biuf":
            raise TypeError(f"Only numeric arrays can be streamed to HDF5: {path}")
        if path in self.h5_file:
            dataset = self.h5_file[path]
            if dataset.shape[1:] != value.shape:
                raise ValueError(
                    f"Shape changed for `{path}`: {dataset.shape[1:]} -> {value.shape}"
                )
        else:
            parent_path, dataset_name = path.rsplit("/", 1)
            parent = self.h5_file.require_group(parent_path)
            compression = "lzf" if value.nbytes >= 256 else None
            chunks = (1, *value.shape) if value.ndim > 0 else (256,)
            dataset = parent.create_dataset(
                dataset_name,
                shape=(0, *value.shape),
                maxshape=(None, *value.shape),
                chunks=chunks,
                dtype=value.dtype,
                compression=compression,
            )
        index = int(dataset.shape[0])
        dataset.resize(index + 1, axis=0)
        dataset[index] = value


class _EpisodeStreamWriter:
    def __init__(
        self,
        partial_dir: Path,
        *,
        camera_names: tuple[str, ...],
        fps: int,
        rgb_crf: int,
        rgb_preset: str,
        queue_size: int,
    ) -> None:
        self.partial_dir = partial_dir
        self.camera_names = camera_names
        self.fps = int(fps)
        self.rgb_crf = int(rgb_crf)
        self.rgb_preset = str(rgb_preset)
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=int(queue_size))
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="teleop-dataset-writer", daemon=True)
        self._thread.start()

    def put(self, command: str, payload: Any) -> None:
        self._raise_if_failed()
        self._queue.put((command, _copy_tree(payload) if payload is not None else None))
        self._raise_if_failed()

    def finish(self, final_metadata: Mapping[str, Any]) -> None:
        if self._error is None:
            self.put("finish", dict(final_metadata))
        else:
            self._queue.put(("finish", dict(final_metadata)))
        self._thread.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Teleop dataset writer failed.") from self._error

    def _run(self) -> None:
        h5_file: h5py.File | None = None
        containers: dict[str, av.container.OutputContainer] = {}
        streams: dict[str, av.video.stream.VideoStream] = {}
        video_frame_index = 0
        observation_count = 0
        action_count = 0
        metadata: dict[str, Any] = {}
        failed = False
        try:
            self.partial_dir.mkdir(parents=True, exist_ok=False)
            h5_file = h5py.File(self.partial_dir / "data.hdf5", "w")
            h5_file.attrs["schema_name"] = SCHEMA_NAME
            h5_file.attrs["schema_version"] = SCHEMA_VERSION
            series = _H5SeriesWriter(h5_file)

            while True:
                command, payload = self._queue.get()
                if failed:
                    if command == "finish":
                        break
                    continue
                try:
                    if command == "observation":
                        observation = dict(payload)
                        camera_payload = observation["cameras"]
                        for camera_name in self.camera_names:
                            rgb = np.asarray(camera_payload[camera_name].pop("rgb"), dtype=np.uint8)
                            if camera_name not in containers:
                                container = av.open(
                                    str(self.partial_dir / f"{camera_name}.mp4"), mode="w"
                                )
                                stream = container.add_stream("libx264", rate=self.fps)
                                stream.width = int(rgb.shape[1])
                                stream.height = int(rgb.shape[0])
                                stream.pix_fmt = "yuv420p"
                                stream.options = {
                                    "crf": str(self.rgb_crf),
                                    "preset": self.rgb_preset,
                                }
                                containers[camera_name] = container
                                streams[camera_name] = stream
                            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                            frame.pts = video_frame_index
                            frame.time_base = Fraction(1, self.fps)
                            for packet in streams[camera_name].encode(frame):
                                containers[camera_name].mux(packet)
                        series.append("observations", observation)
                        observation_count += 1
                        video_frame_index += 1
                    elif command == "step":
                        series.append("actions", payload["action"])
                        series.append("transitions", payload["transition"])
                        action_count += 1
                        observation = dict(payload["next_observation"])
                        camera_payload = observation["cameras"]
                        for camera_name in self.camera_names:
                            rgb = np.asarray(camera_payload[camera_name].pop("rgb"), dtype=np.uint8)
                            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                            frame.pts = video_frame_index
                            frame.time_base = Fraction(1, self.fps)
                            for packet in streams[camera_name].encode(frame):
                                containers[camera_name].mux(packet)
                        series.append("observations", observation)
                        observation_count += 1
                        video_frame_index += 1
                    elif command == "finish":
                        metadata = dict(payload)
                        break
                    else:
                        raise ValueError(f"Unknown writer command: {command}")
                except BaseException as exc:
                    self._error = exc
                    failed = True
        except BaseException as exc:
            self._error = exc
        finally:
            for camera_name, stream in streams.items():
                try:
                    for packet in stream.encode(None):
                        containers[camera_name].mux(packet)
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
            for container in containers.values():
                try:
                    container.close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
            if h5_file is not None:
                h5_file.attrs["observation_count"] = int(observation_count)
                h5_file.attrs["action_count"] = int(action_count)
                h5_file.close()
            if self._error is None:
                metadata.update(
                    {
                        "schema_name": SCHEMA_NAME,
                        "schema_version": SCHEMA_VERSION,
                        "observation_count": int(observation_count),
                        "action_count": int(action_count),
                    }
                )
                (self.partial_dir / "episode.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )


class CanonicalTeleopDatasetRecorder:
    """Manage atomic episode directories under one canonical dataset root."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
        fps: int = 20,
        rgb_crf: int = 18,
        rgb_preset: str = "veryfast",
        queue_size: int = 8,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.episodes_dir = self.dataset_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.camera_names = tuple(camera_names)
        self.fps = int(fps)
        self.rgb_crf = int(rgb_crf)
        self.rgb_preset = str(rgb_preset)
        self.queue_size = int(queue_size)
        self._writer: _EpisodeStreamWriter | None = None
        self._episode_index: int | None = None
        self._partial_dir: Path | None = None
        self._recorded_action_count = 0
        self._write_dataset_info()

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def _write_dataset_info(self) -> None:
        info_path = self.dataset_dir / "dataset_info.json"
        expected = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "camera_names": list(self.camera_names),
            "rgb": {
                "codec": "H.264",
                "pixel_format": "yuv420p",
                "crf": self.rgb_crf,
                "fps": self.fps,
            },
            "geometry": {
                "field": "position_mm",
                "dtype": "int16",
                "compression": "lossless_lzf",
                "depth_derivation": "depth_mm = -position_mm[..., 2]",
            },
            "quaternion_order": {
                "robot_pose": "wxyz",
                "xr_controller_pose": "xyzw",
            },
        }
        if info_path.exists():
            existing = json.loads(info_path.read_text(encoding="utf-8"))
            if existing != expected:
                raise ValueError(f"Incompatible dataset schema at {info_path}")
            return
        info_path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def next_episode_index(self) -> int:
        indices = []
        for path in self.episodes_dir.iterdir():
            name = path.name.removeprefix(".").removesuffix(".partial")
            if not name.startswith("episode_"):
                continue
            try:
                indices.append(int(name.split("_", 1)[1]))
            except ValueError:
                continue
        return 0 if not indices else max(indices) + 1

    def next_seed(self, *, default: int = 0) -> int:
        """Return one greater than the largest seed in completed episodes."""

        seeds = []
        for metadata_path in self.episodes_dir.glob("episode_*/episode.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                seed = metadata.get("seed")
                if isinstance(seed, bool):
                    continue
                seeds.append(int(seed))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return int(default) if not seeds else max(seeds) + 1

    def start_episode(
        self,
        initial_observation: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
        episode_index: int | None = None,
    ) -> int:
        if self.recording:
            raise RuntimeError("An episode is already being recorded.")
        episode_index = self.next_episode_index() if episode_index is None else int(episode_index)
        partial_dir = self.episodes_dir / f".episode_{episode_index:06d}.partial"
        final_dir = self.episodes_dir / f"episode_{episode_index:06d}"
        if partial_dir.exists() or final_dir.exists():
            raise FileExistsError(f"Episode index {episode_index} already exists.")
        self._writer = _EpisodeStreamWriter(
            partial_dir,
            camera_names=self.camera_names,
            fps=self.fps,
            rgb_crf=self.rgb_crf,
            rgb_preset=self.rgb_preset,
            queue_size=self.queue_size,
        )
        self._episode_index = episode_index
        self._partial_dir = partial_dir
        self._recorded_action_count = 0
        self._writer.put("observation", initial_observation)
        self._initial_metadata = dict(metadata)
        return episode_index

    def append_step(
        self,
        *,
        action: Mapping[str, Any],
        transition: Mapping[str, Any],
        next_observation: Mapping[str, Any],
    ) -> None:
        if self._writer is None:
            raise RuntimeError("No episode is being recorded.")
        self._writer.put(
            "step",
            {
                "action": action,
                "transition": transition,
                "next_observation": next_observation,
            },
        )
        self._recorded_action_count += 1

    def save_episode(self, *, final_metadata: Mapping[str, Any] | None = None) -> Path:
        if self._writer is None or self._partial_dir is None or self._episode_index is None:
            raise RuntimeError("No episode is being recorded.")
        if self._recorded_action_count <= 0:
            self.discard_episode()
            raise ValueError("Cannot save a teleoperation episode with no actions.")
        metadata = dict(self._initial_metadata)
        metadata.update(dict(final_metadata or {}))
        writer = self._writer
        partial_dir = self._partial_dir
        episode_index = self._episode_index
        self._clear_active_writer()
        writer.finish(metadata)
        final_dir = self.episodes_dir / f"episode_{episode_index:06d}"
        partial_dir.rename(final_dir)
        return final_dir

    def discard_episode(self) -> None:
        if self._writer is None or self._partial_dir is None:
            return
        writer = self._writer
        partial_dir = self._partial_dir
        self._clear_active_writer()
        try:
            writer.finish({"discarded": True})
        finally:
            if partial_dir.exists():
                shutil.rmtree(partial_dir)

    def _clear_active_writer(self) -> None:
        self._writer = None
        self._episode_index = None
        self._partial_dir = None
        self._initial_metadata = {}
        self._recorded_action_count = 0

    def close(self) -> None:
        self.discard_episode()
