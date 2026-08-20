import pathlib
import sys

import dill
import hydra
import torch


CURRENT_DIR = pathlib.Path(__file__).resolve().parent
CORE_DIR = CURRENT_DIR.parent / "fm_diffusion"
sys.path.insert(0, str(CORE_DIR))

from diffusion_policy.env_runner.dp_runner import DPRunner
from diffusion_policy.policy.resfit_policy import ResFiTPolicy
from diffusion_policy.workspace.resfit_robotworkspace import ResFiTWorkspace


class DP:
    def __init__(
        self,
        ckpt_file: str,
        n_obs_steps: int,
        n_action_steps: int,
    ) -> None:
        self.policy = self.get_policy(
            ckpt_file,
            n_action_steps=n_action_steps,
            device="cuda:0",
        )
        self.runner = DPRunner(
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )
        self.obs_cache = self.runner.obs

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def reset_obs(self):
        self.runner.reset_obs()
        self.obs_cache = self.runner.obs

    def get_action(self, observation=None):
        return self.runner.get_action(self.policy, observation)

    def get_last_obs(self):
        return self.runner.obs[-1]

    @staticmethod
    def get_policy(checkpoint, n_action_steps, device):
        checkpoint_path = pathlib.Path(checkpoint).expanduser()
        with checkpoint_path.open("rb") as file:
            payload = torch.load(file, pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = workspace_cls(cfg, output_dir=None)
        if not isinstance(workspace, ResFiTWorkspace):
            raise TypeError(
                "ResFiT checkpoint target did not create ResFiTWorkspace: "
                f"{type(workspace).__name__}."
            )

        base_checkpoint = pathlib.Path(str(cfg.training.base_checkpoint)).expanduser()
        if not base_checkpoint.is_file():
            raise FileNotFoundError(
                "Base Flow Matching checkpoint referenced by ResFiT was not found: "
                f"{base_checkpoint}"
            )
        with base_checkpoint.open("rb") as file:
            base_payload = torch.load(file, pickle_module=dill, map_location="cpu")
        workspace._validate_base_checkpoint(base_payload, cfg)
        workspace.load_payload(
            base_payload,
            exclude_keys=("optimizer",),
            include_keys=(),
        )
        workspace.load_payload(
            payload,
            exclude_keys=workspace.exclude_keys,
            include_keys=workspace.include_keys,
        )

        source = str(cfg.training.base_policy_source).lower()
        if source == "ema_model":
            if "ema_model" not in base_payload.get("state_dicts", {}):
                raise RuntimeError("Requested EMA base policy, but checkpoint has no ema_model.")
            base_policy = workspace.ema_model
        elif source == "model":
            base_policy = workspace.model
        else:
            raise ValueError("training.base_policy_source must be `model` or `ema_model`.")
        base_policy.n_action_steps = int(n_action_steps)
        normalizer = getattr(base_policy, "normalizer", None)
        if normalizer is None or "action" not in normalizer.params_dict:
            raise RuntimeError("Base policy checkpoint has no action normalizer statistics.")
        action_stats = normalizer["action"].get_input_stats()
        action_low = action_stats["min"].detach().to(device="cpu", dtype=torch.float32)
        action_high = action_stats["max"].detach().to(device="cpu", dtype=torch.float32)
        if action_low.shape != (14,) or action_high.shape != (14,):
            raise RuntimeError(
                "Base policy action normalizer must contain 14D min/max statistics, got "
                f"{tuple(action_low.shape)} and {tuple(action_high.shape)}."
            )
        if not torch.isfinite(action_low).all() or not torch.isfinite(action_high).all():
            raise RuntimeError("Base policy action normalizer bounds contain NaN or Inf values.")
        if not torch.all(action_low < action_high):
            raise RuntimeError("Base policy action normalizer requires min < max in every dimension.")

        policy = ResFiTPolicy(
            base_policy=base_policy,
            residual_actor=workspace.residual_actor,
            action_low=action_low,
            action_high=action_high,
        )
        policy.to(torch.device(device)).eval()
        return policy
