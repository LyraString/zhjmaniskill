if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    # sys.path.append(ROOT_DIR)
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import subprocess
import logging
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy

import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _resolve_amp_config(training_cfg) -> tuple[bool, torch.dtype]:
    amp_cfg = training_cfg.get("amp", {})
    enabled = bool(amp_cfg.get("enabled", False))
    dtype_name = str(amp_cfg.get("dtype", "bf16")).lower()
    if dtype_name in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif dtype_name in ("fp16", "float16"):
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported training.amp.dtype: {dtype_name}")
    return enabled, dtype


def _action_prediction_metrics(pred_action: torch.Tensor, gt_action: torch.Tensor) -> dict:
    if pred_action.shape != gt_action.shape:
        raise ValueError(
            "Action metrics require matching prediction and target shapes, got "
            f"prediction={tuple(pred_action.shape)} target={tuple(gt_action.shape)}"
        )
    if pred_action.shape[-1] != 14:
        return {}

    pred_action = pred_action.float()
    gt_action = gt_action.float()
    translation_dims = (0, 1, 2, 7, 8, 9)
    rotation_dims = (3, 4, 5, 10, 11, 12)
    gripper_dims = (6, 13)

    translation_mae = torch.nn.functional.l1_loss(
        pred_action[..., translation_dims], gt_action[..., translation_dims]
    )
    rotation_mae = torch.nn.functional.l1_loss(
        pred_action[..., rotation_dims], gt_action[..., rotation_dims]
    )
    gripper_accuracy = (
        (pred_action[..., gripper_dims] >= 0.5)
        == (gt_action[..., gripper_dims] >= 0.5)
    ).float().mean()
    return {
        "val_action_translation_mae_m": translation_mae.item(),
        "val_action_rotation_mae_rad": rotation_mae.item(),
        "val_gripper_accuracy": gripper_accuracy.item(),
    }


def _suppress_torch_compile_logs() -> None:
    for logger_name in (
        "torch._dynamo",
        "torch._inductor",
        "torch._inductor.select_algorithm",
        "torch.fx.experimental.symbolic_shapes",
        "triton",
        "triton.compiler",
        "triton.runtime",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _resolve_nvidia_smi_gpu_id(device: torch.device) -> str | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        first_device = visible_devices.split(",")[0].strip()
        if first_device and first_device not in ("-1", "none", "None"):
            return first_device
    if device.type == "cuda":
        return str(device.index if device.index is not None else torch.cuda.current_device())
    return None


def _query_gpu_system_metrics(device: torch.device) -> dict:
    gpu_id = _resolve_nvidia_smi_gpu_id(device)
    if gpu_id is None:
        return {}

    query = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return {}

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 5:
        return {}

    try:
        gpu_util, mem_used, mem_total, power_draw, temp = [float(part) for part in parts]
    except ValueError:
        return {}

    return {
        "system_gpu_util_percent": gpu_util,
        "system_gpu_mem_used_mb": mem_used,
        "system_gpu_mem_total_mb": mem_total,
        "system_gpu_power_w": power_draw,
        "system_gpu_temp_c": temp,
    }


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: BaseImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        # resume training
        if cfg.training.resume:
            # lastest_ckpt_path = self.get_checkpoint_path()
            # if lastest_ckpt_path.is_file():
            #     print(f"Resuming from checkpoint {lastest_ckpt_path}")
            #     self.load_checkpoint(path=lastest_ckpt_path)
            resume_ckpt_path = cfg.training.get("resume_ckpt_path", None)
            if resume_ckpt_path is not None:
                ckpt_path = pathlib.Path(resume_ckpt_path)
                if not ckpt_path.is_file():
                    raise FileNotFoundError(f"Specified resume checkpoint not found: {ckpt_path}")
                print(f"Resuming from specified checkpoint {ckpt_path}")
                self.load_checkpoint(path=ckpt_path)
            else:
                lastest_ckpt_path = self.get_checkpoint_path()
                if lastest_ckpt_path.is_file():
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                    self.load_checkpoint(path=lastest_ckpt_path)

        # Optimizer checkpoints also contain runtime implementation flags. Reapply
        # the current config so an older checkpoint cannot silently disable fused AdamW.
        if "fused" in cfg.optimizer:
            fused = bool(cfg.optimizer.fused)
            self.optimizer.defaults["fused"] = fused
            for param_group in self.optimizer.param_groups:
                param_group["fused"] = fused

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # compile UNet backbone for operator fusion
        tc_cfg = cfg.training.get("torch_compile", {})
        if tc_cfg.get("enabled", False):
            mode = tc_cfg.get("mode", "max-autotune")
            dynamic = tc_cfg.get("dynamic", False)
            fullgraph = tc_cfg.get("fullgraph", False)
            if tc_cfg.get("suppress_logs", True):
                _suppress_torch_compile_logs()
            print(f"[torch.compile] mode={mode}, dynamic={dynamic}, fullgraph={fullgraph}")
            self.model.model = torch.compile(
                self.model.model,
                mode=mode,
                dynamic=dynamic,
                fullgraph=fullgraph,
            )
            if self.ema_model is not None:
                self.ema_model.model = torch.compile(
                    self.ema_model.model,
                    mode=mode,
                    dynamic=dynamic,
                    fullgraph=fullgraph,
                )
            if tc_cfg.get("compile_obs_encoder", False) and hasattr(self.model, "obs_encoder"):
                print("[torch.compile] compiling obs_encoder")
                self.model.obs_encoder = torch.compile(
                    self.model.obs_encoder,
                    mode=mode,
                    dynamic=dynamic,
                    fullgraph=fullgraph,
                )
                if self.ema_model is not None and hasattr(self.ema_model, "obs_encoder"):
                    self.ema_model.obs_encoder = torch.compile(
                        self.ema_model.obs_encoder,
                        mode=mode,
                        dynamic=dynamic,
                        fullgraph=fullgraph,
                    )

        amp_enabled, amp_dtype = _resolve_amp_config(cfg.training)
        amp_enabled = bool(amp_enabled and device.type == "cuda")
        if amp_enabled:
            print(f"Using AMP autocast: dtype={amp_dtype}")

        # save batch for sampling
        train_sampling_batch = None
        fpo_checks_done = True
        fpo_smoke_status = None
        fpo_repro_status = None
        fpo_smoke_cfm_loss_mean = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        tb_enabled = bool(getattr(cfg.logging, "tensorboard", {}).get("enabled", True))
        tb_log_every = int(getattr(cfg.logging, "tensorboard", {}).get("log_every", 1))
        tb_flush_secs = int(getattr(cfg.logging, "tensorboard", {}).get("flush_secs", 10))
        tb_dir = os.path.join(self.output_dir, "tensorboard")
        tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=tb_flush_secs) if tb_enabled else None

        try:
            with JsonLogger(log_path) as json_logger:
                for local_epoch_idx in range(cfg.training.num_epochs):
                    step_log = dict()
                    # ========= train for this epoch ==========
                    if cfg.training.freeze_encoder:
                        self.model.obs_encoder.eval()
                        self.model.obs_encoder.requires_grad_(False)

                    train_losses = list()
                    with tqdm.tqdm(
                            train_dataloader,
                            desc=f"Training epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dataset.postprocess(batch, device)
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            # one-time FPO checks on the first train batch
                            if (not fpo_checks_done) and hasattr(self.model, "run_fpo_forward_smoke_test") and hasattr(self.model, "run_fpo_reproducibility_check"):
                                smoke_info = self.model.run_fpo_forward_smoke_test(
                                    batch=batch,
                                    num_fpo_samples=4,
                                    batch_size=min(2, batch["action"].shape[0]),
                                )
                                repro_info = self.model.run_fpo_reproducibility_check(
                                    batch=batch,
                                    num_fpo_samples=4,
                                    seed=seed,
                                    batch_size=min(2, batch["action"].shape[0]),
                                    atol=0.0,
                                    rtol=0.0,
                                )
                                print(f"[FPO CHECK] smoke test passed: {smoke_info}")
                                print(f"[FPO CHECK] reproducibility passed: {repro_info}")
                                fpo_smoke_status = smoke_info.get("status", "unknown")
                                fpo_repro_status = repro_info.get("status", "unknown")
                                fpo_smoke_cfm_loss_mean = smoke_info.get("cfm_loss_mean", None)
                                fpo_checks_done = True

                            # compute loss
                            with torch.autocast(
                                device_type=device.type,
                                dtype=amp_dtype,
                                enabled=amp_enabled,
                            ):
                                raw_loss = self.model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                            # step optimizer
                            if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()

                            # update ema
                            if cfg.training.use_ema:
                                ema.step(self.model)

                            # logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                "train_loss": raw_loss_cpu,
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                                "lr": lr_scheduler.get_last_lr()[0],
                            }
                            if fpo_smoke_status is not None:
                                step_log["fpo_smoke_status"] = fpo_smoke_status
                            if fpo_repro_status is not None:
                                step_log["fpo_repro_status"] = fpo_repro_status
                            if fpo_smoke_cfm_loss_mean is not None:
                                step_log["fpo_smoke_cfm_loss_mean"] = fpo_smoke_cfm_loss_mean

                            if (tb_writer is not None) and (((self.global_step + 1) % max(1, tb_log_every)) == 0):
                                tb_step = int(self.global_step)
                                tb_writer.add_scalar("train/train_loss_step", raw_loss_cpu, tb_step)
                                tb_writer.add_scalar("train/lr", float(step_log["lr"]), tb_step)

                            is_last_batch = batch_idx == (len(train_dataloader) - 1)
                            if not is_last_batch:
                                # log of last step is combined with validation and rollout
                                json_logger.log(step_log)
                                self.global_step += 1

                            if (cfg.training.max_train_steps
                                    is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                                break

                    # at the end of each epoch
                    # replace train_loss with epoch average
                    train_loss = np.mean(train_losses)
                    step_log["train_loss"] = train_loss
                    step_log["train_loss_std"] = float(np.std(train_losses))

                    # ========= eval for this epoch ==========
                    policy = self.model
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                    # run rollout
                    # if (self.epoch % cfg.training.rollout_every) == 0:
                    #     runner_log = env_runner.run(policy)
                    #     # log all
                    #     step_log.update(runner_log)

                    # run validation
                    if (self.epoch % cfg.training.val_every) == 0:
                        with torch.no_grad():
                            val_losses = list()
                            val_sampling_batch = None
                            with tqdm.tqdm(
                                    val_dataloader,
                                    desc=f"Validation epoch {self.epoch}",
                                    leave=False,
                                    mininterval=cfg.training.tqdm_interval_sec,
                            ) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dataset.postprocess(batch, device)
                                    if val_sampling_batch is None:
                                        val_sampling_batch = batch
                                    with torch.autocast(
                                        device_type=device.type,
                                        dtype=amp_dtype,
                                        enabled=amp_enabled,
                                    ):
                                        loss = self.model.compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps
                                            is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log["val_loss"] = val_loss
                            if val_sampling_batch is not None:
                                result = policy.predict_action(val_sampling_batch["obs"])
                                step_log.update(_action_prediction_metrics(
                                    result["action_pred"],
                                    val_sampling_batch["action"],
                                ))

                    # run diffusion sampling on a training batch
                    if (self.epoch % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = train_sampling_batch
                            obs_dict = batch["obs"]
                            gt_action = batch["action"]

                            result = policy.predict_action(obs_dict)
                            pred_action = result["action_pred"]
                            pred_action_float = pred_action.float()
                            gt_action_float = gt_action.float()
                            mse = torch.nn.functional.mse_loss(pred_action_float, gt_action_float)
                            mae = torch.nn.functional.l1_loss(pred_action_float, gt_action_float)
                            step_log["train_action_mse_error"] = mse.item()
                            step_log["train_action_mae_error"] = mae.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del pred_action_float
                            del gt_action_float
                            del mse
                            del mae

                    step_log.update(_query_gpu_system_metrics(device))

                    # checkpoint
                    if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                        # checkpointing
                        save_dir_name = cfg.training.get("checkpoint_run_name", None)
                        if save_dir_name in (None, ""):
                            save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                            save_dir_name = f"{save_name}-{seed}"
                        self.save_checkpoint(f"checkpoints/{save_dir_name}/{self.epoch + 1}.ckpt")  # TODO

                    if tb_writer is not None:
                        tb_epoch = int(self.epoch)
                        tb_writer.add_scalar("epoch/train_loss", float(step_log["train_loss"]), tb_epoch)
                        if "val_loss" in step_log:
                            tb_writer.add_scalar("epoch/val_loss", float(step_log["val_loss"]), tb_epoch)
                        for key in (
                            "val_action_translation_mae_m",
                            "val_action_rotation_mae_rad",
                            "val_gripper_accuracy",
                        ):
                            if key in step_log:
                                tb_writer.add_scalar(f"epoch/{key}", float(step_log[key]), tb_epoch)
                        if "train_action_mse_error" in step_log:
                            tb_writer.add_scalar(
                                "epoch/train_action_mse_error",
                                float(step_log["train_action_mse_error"]),
                                tb_epoch,
                            )
                        if "train_loss_std" in step_log:
                            tb_writer.add_scalar("epoch/train_loss_std", float(step_log["train_loss_std"]), tb_epoch)
                        if "train_action_mae_error" in step_log:
                            tb_writer.add_scalar(
                                "epoch/train_action_mae_error",
                                float(step_log["train_action_mae_error"]),
                                tb_epoch,
                            )
                        for key in (
                            "system_gpu_util_percent",
                            "system_gpu_mem_used_mb",
                            "system_gpu_mem_total_mb",
                            "system_gpu_power_w",
                            "system_gpu_temp_c",
                        ):
                            if key in step_log:
                                tb_writer.add_scalar(f"system/{key.removeprefix('system_')}", float(step_log[key]), tb_epoch)

                    # ========= eval end for this epoch ==========
                    policy.train()

                    # end of epoch
                    # log of last step is combined with validation and rollout
                    json_logger.log(step_log)
                    self.global_step += 1
                    self.epoch += 1
        finally:
            if tb_writer is not None:
                tb_writer.flush()
                tb_writer.close()


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def _collate_identity(batch):
    assert len(batch) == 1
    return batch[0]


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    mp_context = None
    if num_workers > 0:
        import multiprocessing

        mp_context = multiprocessing.get_context("spawn")
        fm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if fm_root not in existing_pythonpath.split(os.pathsep):
            os.environ["PYTHONPATH"] = (
                os.pathsep.join([fm_root, existing_pythonpath])
                if existing_pythonpath
                else fm_root
            )

    dataloader = DataLoader(
        dataset,
        collate_fn=_collate_identity,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        multiprocessing_context=mp_context,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
