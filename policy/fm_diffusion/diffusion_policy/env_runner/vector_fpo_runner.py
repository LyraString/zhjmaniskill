import multiprocessing as mp
import contextlib
import os
import numpy as np
from diffusion_policy.env_runner.fpo_runner import FPORunner


@contextlib.contextmanager
def _maybe_suppress_worker_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _worker_main(conn, cfg):
    suppress_worker_output = bool(cfg.get("suppress_worker_output", True))
    runner = FPORunner(
        task_name=cfg['task_name'],
        setting=cfg['setting'],
        head_camera_type=cfg['head_camera_type'],
        base_seed=cfg['base_seed'],
        action_type=cfg['action_type'],
    )
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == 'reset':
                with _maybe_suppress_worker_output(suppress_worker_output):
                    obs, info = runner.reset()
                conn.send((obs, info))
            elif cmd == 'step':
                with _maybe_suppress_worker_output(suppress_worker_output):
                    obs, rew, terminated, truncated, info = runner.step(payload)
                conn.send((obs, rew, terminated, truncated, info))
            elif cmd == 'close':
                conn.send(None)
                break
            else:
                raise ValueError(f"Unknown command: {cmd}")
    finally:
        runner.close()
        conn.close()

class VectorFPORunner:
    def __init__(self, task_name, setting, head_camera_type, num_envs=2, base_seed=0,
                 action_type='qpos', start_method='spawn', suppress_worker_output=True):
        self.num_envs = int(num_envs)
        self.ctx = mp.get_context(start_method)
        self.parents = []
        self.procs = []
        for env_id in range(self.num_envs):
            parent, child = self.ctx.Pipe()
            cfg = dict(
                task_name=task_name,
                setting=setting,
                head_camera_type=head_camera_type,
                base_seed=int(base_seed) * 10000 + env_id,
                action_type=action_type,
                suppress_worker_output=bool(suppress_worker_output),
            )
            p = self.ctx.Process(target=_worker_main, args=(child, cfg), daemon=True)
            p.start()
            child.close()
            self.parents.append(parent)
            self.procs.append(p)
        
    def reset(self):
        for c in self.parents:
            c.send(('reset', None))
        out = [c.recv() for c in self.parents]
        obs_list = [x[0] for x in out]
        info_list = [x[1] for x in out]
        return obs_list, info_list

    def reset_at(self, env_ids):
        for i in env_ids:
            self.parents[i].send(('reset', None))
        out = {}
        for i in env_ids:
            obs, info = self.parents[i].recv()
            out[i] = (obs, info)
        return out

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError(f"actions length ({len(actions)}) must match num_envs ({self.num_envs})")
        for c, a in zip(self.parents, actions):
            c.send(("step", a))
        out = [c.recv() for c in self.parents]
        obs = [x[0] for x in out]
        rew = np.array([x[1] for x in out], dtype=np.float32)
        terminated = np.array([x[2] for x in out], dtype=np.bool_)
        truncated = np.array([x[3] for x in out], dtype=np.bool_)
        info = [x[4] for x in out]
        return obs, rew, terminated, truncated, info
    
    def close(self):
        for c in self.parents:
            try:
                c.send(("close", None))
                _ = c.recv()
            except Exception:
                pass
            c.close()
        for p in self.procs:
            p.join(timeout=1.0)
