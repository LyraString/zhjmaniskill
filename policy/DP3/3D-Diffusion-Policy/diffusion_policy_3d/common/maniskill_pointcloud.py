"""Raw ManiSkill camera positions -> cropped world XYZ -> RoboTwin FPS.

The same function is used for offline VR conversion and online evaluation.
Unlike the earlier adapter, there is no per-camera sparse sampling stage.
"""
import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _world_points(raw, preprocessing):
    clouds = []
    lower = np.asarray(preprocessing["crop_min"], dtype=np.float32)
    upper = np.asarray(preprocessing["crop_max"], dtype=np.float32)
    for camera in preprocessing["cameras"]:
        images = raw["sensor_data"][camera]
        # Canonical VR/SAPIEN3 positions are OpenGL XYZ in millimetres.
        position = as_numpy(images["position"]).reshape(-1, 3).astype(np.float32) / 1000
        valid = as_numpy(images["segmentation"]).reshape(-1) != 0
        transform = as_numpy(raw["sensor_param"][camera]["cam2world_gl"]).reshape(4, 4)
        world = position[valid] @ transform[:3, :3].T + transform[:3, 3]
        world = world[((world >= lower) & (world <= upper)).all(axis=-1)]
        clouds.append(world)
    points = np.concatenate(clouds, axis=0).astype(np.float32, copy=False)
    count = int(preprocessing["points"])
    if count < 1:
        raise ValueError("Point count must be positive.")
    if len(points) < count:
        # Match RoboTwin's zero padding before FPS, including empty views.
        points = np.pad(points, ((0, count - len(points)), (0, 0)))
    if not np.isfinite(points).all():
        raise ValueError("Raw point cloud contains NaN or Inf.")
    return points


def encode_raw_observations(observations, preprocessing, *, device="cuda:0"):
    clouds = [_world_points(raw, preprocessing) for raw in observations]
    lengths = torch.as_tensor([len(points) for points in clouds], device=device)
    padded = np.zeros((len(clouds), max(map(len, clouds)), 3), dtype=np.float32)
    for i, points in enumerate(clouds):
        padded[i, :len(points)] = points
    with torch.no_grad():
        sampled, _ = sample_farthest_points(
            torch.as_tensor(padded, device=device), lengths=lengths,
            K=int(preprocessing["points"]), random_start_point=False,
        )
    states = np.stack([as_numpy(raw["agent"]["qpos"]).reshape(-1)
                       for raw in observations]).astype(np.float32)
    if states.shape != (len(clouds), 14) or not np.isfinite(states).all():
        raise ValueError("Expected finite 14D joint states.")
    return {"point_cloud": sampled.cpu().numpy(), "agent_pos": states}


def encode_raw_observation(raw, preprocessing, *, device="cuda:0"):
    return {key: value[0] for key, value in
            encode_raw_observations([raw], preprocessing, device=device).items()}


def raw_observation_from_h5(h5, frame, cameras):
    sensors, params = {}, {}
    for camera in cameras:
        source = h5[f"observations/cameras/{camera}"]
        sensors[camera] = {"position": source["position_mm"][frame],
                           "segmentation": source["segmentation_id"][frame]}
        params[camera] = {"cam2world_gl": source["cam2world_gl"][frame]}
    return {"sensor_data": sensors, "sensor_param": params,
            "agent": {"qpos": h5["observations/robot/agent_qpos"][frame]}}
