"""Legacy RoboTwin single-environment backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.interfaces.env_backend import EnvBackend
from core.schemas.action import ActionSpec
from core.schemas.observation import Observation
from core.schemas.step_result import StepResult
from legacy.robotwin.action_adapter import convert_action_to_robotwin
from legacy.robotwin.obs_adapter import convert_robotwin_obs


@dataclass(slots=True)
class RoboTwinRuntimeState:
    """Mutable runtime bookkeeping for one legacy RoboTwin environment."""

    ep_id: int = 0
    last_seed: Optional[int] = None


class RoboTwinEnvBackend(EnvBackend):
    """Adapter that exposes a RoboTwin task through the shared EnvBackend API."""

    def __init__(
        self,
        *,
        task: Any,
        task_name: str,
        task_args: Dict[str, Any],
        base_seed: int = 0,
        action_type: str = "qpos",
    ) -> None:
        self._task = task
        self.task_name = task_name
        self.task_args = dict(task_args)
        self.base_seed = int(base_seed)
        self.action_type = action_type
        self.runtime = RoboTwinRuntimeState()

    def reset(self, *, seed: Optional[int] = None) -> Observation:
        max_seed = 2**32 - 1
        if seed is None:
            seed = ((100000 * (1 + self.base_seed)) + self.runtime.ep_id) % max_seed

        self._task.setup_demo(
            now_ep_num=self.runtime.ep_id,
            seed=int(seed),
            is_test=True,
            **self.task_args,
        )
        raw_obs = self._task.get_obs()

        self.runtime.last_seed = int(seed)
        obs = convert_robotwin_obs(
            raw_obs,
            metadata={
                "seed": int(seed),
                "ep_id": int(self.runtime.ep_id),
                "task_name": self.task_name,
                "backend": "robotwin_legacy",
            },
        )
        self.runtime.ep_id += 1
        return obs

    def step(self, action: ActionSpec) -> StepResult:
        robotwin_action = convert_action_to_robotwin(action)
        self._task.take_action(robotwin_action, action_type=self.action_type)

        raw_obs = self._task.get_obs()
        obs = convert_robotwin_obs(
            raw_obs,
            metadata={
                "seed": self.runtime.last_seed,
                "ep_id": int(max(0, self.runtime.ep_id - 1)),
                "task_name": self.task_name,
                "backend": "robotwin_legacy",
            },
        )

        terminated = bool(self._task.eval_success)
        truncated = bool((self._task.take_action_cnt >= self._task.step_lim) and (not terminated))
        reward = 1.0 if terminated else 0.0

        return StepResult.create(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "success": terminated,
                "step": int(self._task.take_action_cnt),
                "step_lim": int(self._task.step_lim),
                "plan_success": bool(getattr(self._task, "plan_success", True)),
            },
        )

    def close(self) -> None:
        if self._task is not None:
            self._task.close_env(clear_cache=False)
