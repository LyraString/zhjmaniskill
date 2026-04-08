from __future__ import annotations

import sys
from pathlib import Path


robotwin_root = Path(__file__).resolve().parents[4]
if str(robotwin_root) not in sys.path:
    sys.path.insert(0, str(robotwin_root))

from legacy.robotwin import RoboTwinTaskFactory
from runners.online_env_runner import OnlineEnvRunner


class FPORunner(OnlineEnvRunner):
    """Compatibility wrapper for the legacy RoboTwin online training runner."""

    def __init__(
        self,
        task_name,
        setting,
        head_camera_type,
        base_seed=0,
        action_type="qpos",
    ):
        super().__init__(
            task_factory=RoboTwinTaskFactory(),
            task_name=task_name,
            setting=setting,
            head_camera_type=head_camera_type,
            base_seed=int(base_seed),
            action_type=action_type,
            bootstrap_cwd=str(robotwin_root),
        )
