"""Backend-agnostic multiprocessing vector runner."""

from __future__ import annotations

import contextlib
import importlib
import multiprocessing as mp
import os
import sys
from typing import Any, Dict

import numpy as np

from runners.online_env_runner import OnlineEnvRunner


@contextlib.contextmanager
def _maybe_suppress_worker_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _load_factory(factory_target: str):
    module_name, class_name = factory_target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    factory_cls = getattr(module, class_name)
    return factory_cls()


def _resolve_start_method(start_method: str) -> str:
    if start_method != "spawn":
        return start_method
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file not in (None, "<stdin>"):
        return start_method
    available = mp.get_all_start_methods()
    if "fork" in available:
        return "fork"
    return start_method


def _recv_or_raise(conn):
    payload = conn.recv()
    if (
        isinstance(payload, tuple)
        and len(payload) == 3
        and payload[0] == "__worker_init_error__"
    ):
        _, exc_type, message = payload
        raise RuntimeError(
            f"Vector env worker failed during initialization: {exc_type}: {message}"
        )
    return payload


def _worker_main(conn, cfg: Dict[str, Any]):
    suppress_worker_output = bool(cfg.get("suppress_worker_output", True))
    try:
        runner = OnlineEnvRunner(
            task_factory=_load_factory(str(cfg["factory_target"])),
            task_name=str(cfg["task_name"]),
            setting=str(cfg["setting"]),
            head_camera_type=str(cfg["head_camera_type"]),
            base_seed=int(cfg["base_seed"]),
            action_type=str(cfg["action_type"]),
            bootstrap_cwd=cfg.get("bootstrap_cwd"),
        )
    except Exception as exc:
        conn.send(("__worker_init_error__", type(exc).__name__, str(exc)))
        conn.close()
        return
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                with _maybe_suppress_worker_output(suppress_worker_output):
                    obs, info = runner.reset()
                conn.send((obs, info))
            elif cmd == "step":
                with _maybe_suppress_worker_output(suppress_worker_output):
                    obs, rew, terminated, truncated, info = runner.step(payload)
                conn.send((obs, rew, terminated, truncated, info))
            elif cmd == "close":
                conn.send(None)
                break
            else:
                raise ValueError(f"Unknown command: {cmd}")
    finally:
        runner.close()
        conn.close()


class VectorOnlineEnvRunner:
    """Multiprocessing vector runner for backend-based environments."""

    def __init__(
        self,
        *,
        factory_target: str,
        task_name: str,
        setting: str,
        head_camera_type: str,
        num_envs: int = 2,
        base_seed: int = 0,
        action_type: str = "qpos",
        start_method: str = "spawn",
        bootstrap_cwd: str | None = None,
        suppress_worker_output: bool = True,
    ) -> None:
        self.num_envs = int(num_envs)
        self.ctx = mp.get_context(_resolve_start_method(start_method))
        self.parents = []
        self.procs = []

        for env_id in range(self.num_envs):
            parent, child = self.ctx.Pipe()
            cfg = dict(
                factory_target=factory_target,
                task_name=task_name,
                setting=setting,
                head_camera_type=head_camera_type,
                base_seed=int(base_seed) * 10000 + env_id,
                action_type=action_type,
                bootstrap_cwd=bootstrap_cwd,
                suppress_worker_output=bool(suppress_worker_output),
            )
            proc = self.ctx.Process(target=_worker_main, args=(child, cfg), daemon=True)
            proc.start()
            child.close()
            self.parents.append(parent)
            self.procs.append(proc)

    def reset(self):
        for conn in self.parents:
            conn.send(("reset", None))
        out = [_recv_or_raise(conn) for conn in self.parents]
        obs_list = [item[0] for item in out]
        info_list = [item[1] for item in out]
        return obs_list, info_list

    def reset_at(self, env_ids):
        for env_id in env_ids:
            self.parents[env_id].send(("reset", None))
        out = {}
        for env_id in env_ids:
            obs, info = _recv_or_raise(self.parents[env_id])
            out[env_id] = (obs, info)
        return out

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions length ({len(actions)}) must match num_envs ({self.num_envs})"
            )
        for conn, action in zip(self.parents, actions):
            conn.send(("step", action))
        out = [_recv_or_raise(conn) for conn in self.parents]
        obs = [item[0] for item in out]
        rew = np.array([item[1] for item in out], dtype=np.float32)
        terminated = np.array([item[2] for item in out], dtype=np.bool_)
        truncated = np.array([item[3] for item in out], dtype=np.bool_)
        info = [item[4] for item in out]
        return obs, rew, terminated, truncated, info

    def close(self):
        for conn in self.parents:
            try:
                conn.send(("close", None))
                _ = conn.recv()
            except Exception:
                pass
            conn.close()
        for proc in self.procs:
            proc.join(timeout=1.0)
