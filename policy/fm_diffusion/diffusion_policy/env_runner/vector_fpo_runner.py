from __future__ import annotations

import sys
from pathlib import Path


robotwin_root = Path(__file__).resolve().parents[4]
if str(robotwin_root) not in sys.path:
    sys.path.insert(0, str(robotwin_root))

from runners.vector_online_env_runner import VectorOnlineEnvRunner


class VectorFPORunner(VectorOnlineEnvRunner):
    """Compatibility wrapper for the legacy RoboTwin vector online runner."""

    def __init__(
        self,
        task_name,
        setting,
        head_camera_type,
        num_envs=2,
        base_seed=0,
        action_type="qpos",
        start_method="spawn",
        suppress_worker_output=True,
    ):
        super().__init__(
            factory_target="legacy.robotwin.task_factory.RoboTwinTaskFactory",
            task_name=task_name,
            setting=setting,
            head_camera_type=head_camera_type,
            num_envs=num_envs,
            base_seed=int(base_seed),
            action_type=action_type,
            start_method=start_method,
            bootstrap_cwd=str(robotwin_root),
            suppress_worker_output=bool(suppress_worker_output),
        )
