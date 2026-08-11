"""Hydra entry point for ResFiT post-training."""

import pathlib
import sys


sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
CORE_DIR = CURRENT_DIR.parent / "fm_diffusion"
sys.path.insert(0, str(CORE_DIR))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(CURRENT_DIR.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf) -> None:
    OmegaConf.resolve(cfg)
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=HydraConfig.get().runtime.output_dir)
    workspace.run()


if __name__ == "__main__":
    main()
