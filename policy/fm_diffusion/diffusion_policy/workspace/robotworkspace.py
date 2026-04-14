if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    # sys.path.append(ROOT_DIR)
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
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
                            with tqdm.tqdm(
                                    val_dataloader,
                                    desc=f"Validation epoch {self.epoch}",
                                    leave=False,
                                    mininterval=cfg.training.tqdm_interval_sec,
                            ) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dataset.postprocess(batch, device)
                                    loss = self.model.compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps
                                            is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log["val_loss"] = val_loss

                    # run diffusion sampling on a training batch
                    if (self.epoch % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = train_sampling_batch
                            obs_dict = batch["obs"]
                            gt_action = batch["action"]

                            result = policy.predict_action(obs_dict)
                            pred_action = result["action_pred"]
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log["train_action_mse_error"] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse

                    # checkpoint
                    if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                        # checkpointing
                        save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                        self.save_checkpoint(f"checkpoints/{save_name}-{seed}/{self.epoch + 1}.ckpt")  # TODO

                    if tb_writer is not None:
                        tb_epoch = int(self.epoch)
                        tb_writer.add_scalar("epoch/train_loss", float(step_log["train_loss"]), tb_epoch)
                        if "val_loss" in step_log:
                            tb_writer.add_scalar("epoch/val_loss", float(step_log["val_loss"]), tb_epoch)
                        if "train_action_mse_error" in step_log:
                            tb_writer.add_scalar(
                                "epoch/train_action_mse_error",
                                float(step_log["train_action_mse_error"]),
                                tb_epoch,
                            )

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

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
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
