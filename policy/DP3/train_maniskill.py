"""Train the existing RoboTwin DP3 workspace on ManiSkill VR data."""
import os
import sys
from pathlib import Path

import zarr
import hydra
from omegaconf import OmegaConf

POLICY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = POLICY_ROOT.parents[1]
DP3_ROOT = POLICY_ROOT / "3D-Diffusion-Policy"
sys.path.insert(0, str(DP3_ROOT))
from train_dp3 import TrainDP3Workspace

OmegaConf.register_new_resolver("repo_root", lambda: str(REPO_ROOT), replace=True)


@hydra.main(version_base="1.2", config_path="3D-Diffusion-Policy/diffusion_policy_3d/config",
            config_name="robotwin_dp3")
def main(cfg):
    OmegaConf.resolve(cfg)
    dataset = zarr.open_group(str(cfg.task.dataset.zarr_path), mode="r")
    if dataset.attrs.get("action_semantics") != cfg.task.maniskill.action_semantics:
        raise ValueError("Dataset action semantics do not match the DP3 checkpoint configuration.")
    if dataset.attrs.get("preprocessing") != OmegaConf.to_container(cfg.task.preprocessing):
        raise ValueError("Dataset point-cloud preprocessing does not match online inference.")
    # Legacy workspace checkpoint paths are relative to the DP3 policy folder.
    os.chdir(POLICY_ROOT)
    TrainDP3Workspace(cfg).run()


if __name__ == "__main__":
    main()
