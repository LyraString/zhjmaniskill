"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import os
import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# 确保导入 fm_diffusion 目录下的模块
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
CORE_DIR = CURRENT_DIR.parent / "fm_diffusion"
# 将 fm_diffusion 目录添加到路径最前面，确保优先级
sys.path.insert(0, str(CORE_DIR))

import hydra
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU training requires CUDA.")
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds visible CUDA devices "
                f"({torch.cuda.device_count()})."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        cfg.training.device = f"cuda:{local_rank}"
        OmegaConf.update(
            cfg,
            "training.distributed_world_size",
            world_size,
            merge=False,
            force_add=True,
        )

    rank = dist.get_rank() if distributed else 0
    try:
        cls = hydra.utils.get_class(cfg._target_)
        if distributed and not getattr(cls, "supports_distributed", False):
            raise RuntimeError(
                f"Workspace {cls.__name__} does not support DDP."
            )
        output_dir = HydraConfig.get().runtime.output_dir
        if distributed:
            shared_output_dir = [output_dir if rank == 0 else None]
            dist.broadcast_object_list(shared_output_dir, src=0)
            output_dir = shared_output_dir[0]
            workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
        else:
            workspace: BaseWorkspace = cls(cfg)
        if rank == 0:
            print(cfg.task.dataset.zarr_path, cfg.task_name)
        workspace.run()
        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
