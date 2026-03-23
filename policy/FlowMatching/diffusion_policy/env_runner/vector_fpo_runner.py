import multiprocessing as mp
import numpy as np
from diffusion_policy.env_runner.fpo_runner import FPORunner


def _worker_main(conn, cfg):
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
                obs, info = runner.reset()
                conn.send((obs, info))
            elif cmd == 'step':
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
                 action_type='qpos', start_method='spawn'):
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
