if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    # sys.path.append(ROOT_DIR)
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import io
import contextlib
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
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
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

from diffusion_policy.model.rl.value_mlp import CriticMLP
from collections import deque
import torch.nn.functional as F

from diffusion_policy.env_runner.fpo_runner import FPORunner
from diffusion_policy.env_runner.vector_fpo_runner import VectorFPORunner
from diffusion_policy.env_runner.maniskill_fpo_runner import ManiSkillFPORunner
from diffusion_policy.env_runner.maniskill_vector_fpo_runner import ManiSkillVectorFPORunner
from sim.maniskill.task_profiles import resolve_runtime_control_mode

# # Maniskill并行训练环境
# import gymnasium as gym
# import mani_skill.envs
# from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

OmegaConf.register_new_resolver("eval", eval, replace=True)

# --------------------------------新增开始--------------------------------
@contextlib.contextmanager
def _maybe_suppress_stdout(enabled: bool):
    if enabled:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    else:
        yield


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _format_online_progress(it, total, step_log):
    iter_idx = int(it) + 1
    total_iters = int(total)

    mean_reward = _safe_float(step_log.get("mean_reward", 0.0))
    mean_ep_len = _safe_float(step_log.get("mean_ep_len", 0.0))
    actor_loss = _safe_float(step_log.get("actor_loss", 0.0))
    critic_loss = _safe_float(step_log.get("critic_loss", 0.0))
    rho_mean = _safe_float(step_log.get("rho_mean", 0.0))
    clipped_ratio_frac = _safe_float(step_log.get("clipped_ratio_frac", 0.0))

    return (
        f"[online_fpo] iter {iter_idx}/{total_iters} | "
        f"reward={mean_reward:.4f} ep_len={mean_ep_len:.2f} | "
        f"actor={actor_loss:.4f} critic={critic_loss:.4f} | "
        f"rho={rho_mean:.4f} clip_frac={clipped_ratio_frac:.4f}"
    )
# --------------------------------新增结束--------------------------------


def _create_identity_field_normalizer(dim: int) -> SingleFieldLinearNormalizer:
    scale = torch.ones(dim, dtype=torch.float32)
    offset = torch.zeros(dim, dtype=torch.float32)
    input_stats = {
        "min": -torch.ones(dim, dtype=torch.float32),
        "max": torch.ones(dim, dtype=torch.float32),
        "mean": torch.zeros(dim, dtype=torch.float32),
        "std": torch.ones(dim, dtype=torch.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=input_stats,
    )


def _create_identity_normalizer_from_shape_meta(shape_meta: dict) -> LinearNormalizer:
    obs_shape_meta = shape_meta["obs"]
    point_cloud_dim = int(obs_shape_meta["point_cloud"]["shape"][-1])
    agent_pos_dim = int(obs_shape_meta["agent_pos"]["shape"][0])
    action_dim = int(shape_meta["action"]["shape"][0])

    normalizer = LinearNormalizer()
    normalizer["point_cloud"] = _create_identity_field_normalizer(point_cloud_dim)
    normalizer["agent_pos"] = _create_identity_field_normalizer(agent_pos_dim)
    normalizer["action"] = _create_identity_field_normalizer(action_dim)
    return normalizer

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

        # 初始化critic网络和优化器
        self.critic = hydra.utils.instantiate(cfg.critic)
        self.critic_optimizer = hydra.utils.instantiate(cfg.critic_optimizer, params=self.critic.parameters())

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

        env_backend = str(getattr(cfg.training, "env_backend", "robotwin_legacy"))
        skip_offline_dataset = bool(getattr(cfg.training, "online_only_no_dataset", False))

        dataset = None
        train_dataloader = None
        val_dataset = None
        val_dataloader = None
        if env_backend == "maniskill" and skip_offline_dataset:
            normalizer = _create_identity_normalizer_from_shape_meta(cfg.shape_meta)
        else:
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
        lr_scheduler = None
        if train_dataloader is not None:
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

        # 设备迁移
        self.critic.to(device)
        optimizer_to(self.critic_optimizer, device)

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

        # # 离线训练部分
        # with JsonLogger(log_path) as json_logger:
        #     for local_epoch_idx in range(cfg.training.num_epochs):
        #         step_log = dict()
        #         # ========= train for this epoch ==========
        #         if cfg.training.freeze_encoder:
        #             self.model.obs_encoder.eval()
        #             self.model.obs_encoder.requires_grad_(False)

        #         train_losses = list()
        #         with tqdm.tqdm(
        #                 train_dataloader,
        #                 desc=f"Training epoch {self.epoch}",
        #                 leave=False,
        #                 mininterval=cfg.training.tqdm_interval_sec,
        #         ) as tepoch:
        #             for batch_idx, batch in enumerate(tepoch):
        #                 batch = dataset.postprocess(batch, device)
        #                 if train_sampling_batch is None:
        #                     train_sampling_batch = batch

        #                 # one-time FPO checks on the first train batch
        #                 if (not fpo_checks_done) and hasattr(self.model, "run_fpo_forward_smoke_test") and hasattr(self.model, "run_fpo_reproducibility_check"):
        #                     smoke_info = self.model.run_fpo_forward_smoke_test(
        #                         batch=batch,
        #                         num_fpo_samples=4,
        #                         batch_size=min(2, batch["action"].shape[0]),
        #                     )
        #                     repro_info = self.model.run_fpo_reproducibility_check(
        #                         batch=batch,
        #                         num_fpo_samples=4,
        #                         seed=seed,
        #                         batch_size=min(2, batch["action"].shape[0]),
        #                         atol=0.0,
        #                         rtol=0.0,
        #                     )
        #                     print(f"[FPO CHECK] smoke test passed: {smoke_info}")
        #                     print(f"[FPO CHECK] reproducibility passed: {repro_info}")
        #                     fpo_smoke_status = smoke_info.get("status", "unknown")
        #                     fpo_repro_status = repro_info.get("status", "unknown")
        #                     fpo_smoke_cfm_loss_mean = smoke_info.get("cfm_loss_mean", None)
        #                     fpo_checks_done = True

        #                 # compute loss
        #                 raw_loss = self.model.compute_loss(batch)
        #                 loss = raw_loss / cfg.training.gradient_accumulate_every
        #                 loss.backward()

        #                 # step optimizer
        #                 if (self.global_step % cfg.training.gradient_accumulate_every == 0):
        #                     self.optimizer.step()
        #                     self.optimizer.zero_grad()
        #                     lr_scheduler.step()

        #                 # update ema
        #                 if cfg.training.use_ema:
        #                     ema.step(self.model)

        #                 # logging
        #                 raw_loss_cpu = raw_loss.item()
        #                 tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
        #                 train_losses.append(raw_loss_cpu)
        #                 step_log = {
        #                     "train_loss": raw_loss_cpu,
        #                     "global_step": self.global_step,
        #                     "epoch": self.epoch,
        #                     "lr": lr_scheduler.get_last_lr()[0],
        #                 }
        #                 if fpo_smoke_status is not None:
        #                     step_log["fpo_smoke_status"] = fpo_smoke_status
        #                 if fpo_repro_status is not None:
        #                     step_log["fpo_repro_status"] = fpo_repro_status
        #                 if fpo_smoke_cfm_loss_mean is not None:
        #                     step_log["fpo_smoke_cfm_loss_mean"] = fpo_smoke_cfm_loss_mean

        #                 is_last_batch = batch_idx == (len(train_dataloader) - 1)
        #                 if not is_last_batch:
        #                     # log of last step is combined with validation and rollout
        #                     json_logger.log(step_log)
        #                     self.global_step += 1

        #                 if (cfg.training.max_train_steps
        #                         is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
        #                     break

        #         # at the end of each epoch
        #         # replace train_loss with epoch average
        #         train_loss = np.mean(train_losses)
        #         step_log["train_loss"] = train_loss

        #         # ========= eval for this epoch ==========
        #         policy = self.model
        #         if cfg.training.use_ema:
        #             policy = self.ema_model
        #         policy.eval()

        #         # run rollout
        #         # if (self.epoch % cfg.training.rollout_every) == 0:
        #         #     runner_log = env_runner.run(policy)
        #         #     # log all
        #         #     step_log.update(runner_log)

        #         # run validation
        #         if (self.epoch % cfg.training.val_every) == 0:
        #             with torch.no_grad():
        #                 val_losses = list()
        #                 with tqdm.tqdm(
        #                         val_dataloader,
        #                         desc=f"Validation epoch {self.epoch}",
        #                         leave=False,
        #                         mininterval=cfg.training.tqdm_interval_sec,
        #                 ) as tepoch:
        #                     for batch_idx, batch in enumerate(tepoch):
        #                         batch = dataset.postprocess(batch, device)
        #                         loss = self.model.compute_loss(batch)
        #                         val_losses.append(loss)
        #                         if (cfg.training.max_val_steps
        #                                 is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
        #                             break
        #                 if len(val_losses) > 0:
        #                     val_loss = torch.mean(torch.tensor(val_losses)).item()
        #                     # log epoch average validation loss
        #                     step_log["val_loss"] = val_loss

        #         # run diffusion sampling on a training batch
        #         if (self.epoch % cfg.training.sample_every) == 0:
        #             with torch.no_grad():
        #                 # sample trajectory from training set, and evaluate difference
        #                 batch = train_sampling_batch
        #                 obs_dict = batch["obs"]
        #                 gt_action = batch["action"]

        #                 result = policy.predict_action(obs_dict)
        #                 pred_action = result["action_pred"]
        #                 mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        #                 step_log["train_action_mse_error"] = mse.item()
        #                 del batch
        #                 del obs_dict
        #                 del gt_action
        #                 del result
        #                 del pred_action
        #                 del mse

        #         # checkpoint
        #         if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
        #             # checkpointing
        #             save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
        #             self.save_checkpoint(f"checkpoints/{save_name}-{seed}/{self.epoch + 1}.ckpt")  # TODO

        #         # ========= eval end for this epoch ==========
        #         policy.train()

        #         # end of epoch
        #         # log of last step is combined with validation and rollout
        #         json_logger.log(step_log)
        #         self.global_step += 1
        #         self.epoch += 1

        # 在线训练部分

        num_envs = int(getattr(cfg.training, "num_envs", 1))
        if env_backend == "maniskill":
            sim_config = getattr(cfg.training, "sim_config", None)
            sim_config = OmegaConf.to_container(sim_config, resolve=True) if sim_config is not None else None
            task_maniskill_cfg = getattr(cfg.task, "maniskill", None)
            task_profile = getattr(task_maniskill_cfg, "profile", None) if task_maniskill_cfg is not None else None
            task_env_id = getattr(task_maniskill_cfg, "env_id", None) if task_maniskill_cfg is not None else None
            sim_backend = getattr(cfg.training, "sim_backend", None)
            render_backend = getattr(cfg.training, "render_backend", None)
            runtime_action_type, runtime_control_mode = resolve_runtime_control_mode(
                action_type=getattr(cfg.training, "action_type", "pd_joint_delta_pos"),
                control_mode=getattr(cfg.training, "control_mode", None),
            )
            if num_envs > 1:
                env = ManiSkillVectorFPORunner(
                    task_name=cfg.task.name,
                    task_profile=task_profile,
                    env_id=task_env_id,
                    num_envs=num_envs,
                    base_seed=cfg.training.seed,
                    action_type=runtime_action_type,
                    obs_mode=str(getattr(cfg.training, "obs_mode", "pointcloud+state_dict")),
                    control_mode=runtime_control_mode,
                    render_mode=getattr(cfg.training, "render_mode", None),
                    sim_backend=sim_backend,
                    render_backend=render_backend,
                    sim_freq=getattr(cfg.training, "sim_freq", None),
                    control_freq=getattr(cfg.training, "control_freq", None),
                    sim_config=sim_config,
                    robot_uids=getattr(cfg.training, "robot_uids", None),
                    reward_mode=getattr(cfg.training, "reward_mode", None),
                )
            else:
                env = ManiSkillFPORunner(
                    task_name=cfg.task.name,
                    task_profile=task_profile,
                    env_id=task_env_id,
                    base_seed=cfg.training.seed,
                    action_type=runtime_action_type,
                    obs_mode=str(getattr(cfg.training, "obs_mode", "pointcloud+state_dict")),
                    control_mode=runtime_control_mode,
                    render_mode=getattr(cfg.training, "render_mode", None),
                    sim_backend=sim_backend,
                    render_backend=render_backend,
                    sim_freq=getattr(cfg.training, "sim_freq", None),
                    control_freq=getattr(cfg.training, "control_freq", None),
                    sim_config=sim_config,
                    robot_uids=getattr(cfg.training, "robot_uids", None),
                    reward_mode=getattr(cfg.training, "reward_mode", None),
                )
        elif num_envs > 1:
            env = VectorFPORunner(
                task_name=cfg.task.name,
                setting=cfg.setting,
                head_camera_type=cfg.head_camera_type,
                num_envs=num_envs,
                base_seed=cfg.training.seed,
                action_type="qpos",
                start_method=str(getattr(cfg.training, "vec_env_start_method", "spawn")),
                suppress_worker_output=bool(getattr(cfg.training, "suppress_env_step_print", True)),
            )
        else:
            env = FPORunner(
                task_name=cfg.task.name,
                setting=cfg.setting,
                head_camera_type=cfg.head_camera_type,
                base_seed=cfg.training.seed,
                action_type="qpos",
            )

        try:
            with JsonLogger(log_path) as json_logger:
                online_iters = int(cfg.training.online_iters)
                with tqdm.tqdm(
                    range(online_iters),
                    desc="FPO Training",
                    dynamic_ncols=True,
                    mininterval=float(getattr(cfg.training, "tqdm_interval_sec", 1.0)),
                    leave=True,
                ) as pbar:
                    for it in pbar:
                        self.model.train()
                        self.critic.train()

                        rollout = self.collect_rollout(env, device)
                        update_log = self.update_fpo_from_rollout(rollout, device)

                        step_log = {
                            "iter": it,
                            "iter_total": online_iters,
                            "global_step": self.global_step,
                            **update_log,
                            "mean_reward": rollout["reward_mean"],
                            "mean_ep_len": rollout["ep_len_mean"],
                        }

                        if cfg.training.use_ema:
                            ema.step(self.model)
                        
                        if ((it + 1) % cfg.training.checkpoint_every) == 0:
                            # Use one-based labels in checkpoint names so they match
                            # the offline pretraining filenames (e.g. 2000.ckpt) and
                            # the human-readable online iteration count.
                            pretrain_epoch_label = int(self.epoch) + 1
                            online_iter_label = int(it) + 1
                            save_name = f"online_fpo-pretrain_ep{pretrain_epoch_label}"
                            self.save_checkpoint(
                                f"checkpoints/{save_name}-{seed}/{pretrain_epoch_label}-{online_iter_label}.ckpt"
                            )

                        progress_line = _format_online_progress(it, online_iters, step_log)
                        pbar.set_postfix({
                            "rew": f"{_safe_float(step_log['mean_reward']):.3f}",
                            "ep": f"{_safe_float(step_log['mean_ep_len']):.1f}",
                            "actor": f"{_safe_float(step_log['actor_loss']):.3f}",
                            "critic": f"{_safe_float(step_log['critic_loss']):.3f}",
                            "rho": f"{_safe_float(step_log['rho_mean']):.3f}",
                        })

                        if (tb_writer is not None) and (((it + 1) % max(1, tb_log_every)) == 0):
                            tb_step = int(self.global_step)
                            tb_writer.add_scalar("online/mean_reward", _safe_float(step_log["mean_reward"]), tb_step)
                            tb_writer.add_scalar("online/mean_ep_len", _safe_float(step_log["mean_ep_len"]), tb_step)
                            tb_writer.add_scalar("online/actor_loss", _safe_float(step_log["actor_loss"]), tb_step)
                            tb_writer.add_scalar("online/critic_loss", _safe_float(step_log["critic_loss"]), tb_step)
                            tb_writer.add_scalar("online/rho_mean", _safe_float(step_log["rho_mean"]), tb_step)
                            tb_writer.add_scalar("online/cfm_diff_mean", _safe_float(step_log["cfm_diff_mean"]), tb_step)
                            tb_writer.add_scalar("online/clipped_ratio_frac", _safe_float(step_log["clipped_ratio_frac"]), tb_step)
                            tb_writer.add_scalar("online/iter", float(it), tb_step)

                        tqdm.tqdm.write(progress_line)
                        
                        json_logger.log(step_log)
                        self.global_step += 1
                tqdm.tqdm.write("[online_fpo] training finished")
        
        finally:
            if tb_writer is not None:
                tb_writer.flush()
                tb_writer.close()
            env.close()

    # 提取历史观测并堆叠成一个新的观测字典
    def _stack_last_n_obs(self, obs_hist, n_steps):
        assert len(obs_hist) > 0

        keys = obs_hist[0].keys()
        out = {}
        for k in keys:
            arrs = [o[k] for o in obs_hist]
            if len(arrs) < n_steps:
                arrs = [arrs[0]] * (n_steps - len(arrs)) + arrs
            else:
                arrs = arrs[-n_steps:]
            out[k] = np.stack(arrs, axis=0)
        
        return out

    # 将观测数据转换为网络可接受的输入格式
    def _obs_to_policy_input(self, obs_to, device):
        obs_t = {}
        for k, v in obs_to.items():
            t = torch.from_numpy(v).to(device)
            if t.dtype == torch.uint8:
                t = t.float() / 255.0   # 将图像数据归一化到[0, 1]范围
            obs_t[k] = t.unsqueeze(0)
        
        return obs_t

    # 构建Critic网络的输入
    def _build_critic_input_from_obs(self, obs_input):
        conds = self.model.build_condition_from_obs(obs_input)
        gc = conds.get("global_cond", None)
        if gc is None:
            raise ValueError("Global condition is required for critic input construction.")
        
        return gc

    # 计算GAE函数
    def calculate_gae(self, rewards, values, truncs, discounts, bootstrap_values):

        gamma = self.cfg.training.gamma
        lmbda = self.cfg.training.lmbda
        discounting = float(getattr(self.cfg.training, "discounting", 1.0))

        all_adv = []
        all_ret = []

        for ep_rew, ep_val, ep_tr, ep_disc, ep_boot in zip(
            rewards, values, truncs, discounts, bootstrap_values
            ):
            T = len(ep_rew)
            adv = np.zeros(T, dtype=np.float32)
            ret = np.zeros(T, dtype=np.float32)

            next_value = float(ep_boot)
            gae_adv = 0.0

            # TD Error反向递归计算优势函数
            for t in reversed(range(T)):
                trunc_mask = 1.0 - float(ep_tr[t])
                eff_disc = ep_disc[t] * discounting
                delta = ep_rew[t] + gamma * next_value * eff_disc - ep_val[t]
                gae_adv = delta + gamma * lmbda * eff_disc * trunc_mask * gae_adv
                adv[t] = gae_adv
                ret[t] = adv[t] + ep_val[t] 
                next_value = ep_val[t]

            # 将每一个episode的优势函数添加到批次中
            all_adv.extend(adv.tolist())
            all_ret.extend(ret.tolist())

        return (
            torch.tensor(all_adv, dtype=torch.float32),
            torch.tensor(all_ret, dtype=torch.float32),
        )
    
    def collect_rollout(self, env, device):
        num_envs = int(getattr(self.cfg.training, "num_envs", 1))
        if num_envs <= 1:
            return self.collect_rollout_single(env, device)
        else:
            return self.collect_rollout_vector(env, device, num_envs)
    
    def collect_rollout_single(self, env, device):
        cfg_t = self.cfg.training
        suppress_env_step_print = bool(getattr(cfg_t, "suppress_env_step_print", True))
        show_rollout_tqdm = bool(getattr(cfg_t, "show_rollout_tqdm", True))
        rollout_tqdm_interval = float(getattr(cfg_t, "rollout_tqdm_interval_sec", 1.0))
        target_steps = int(cfg_t.timesteps_per_batch)
        max_ep_steps = int(cfg_t.max_timesteps_per_episode)
        n_obs_steps = int(self.cfg.n_obs_steps)
        base_num_fpo_samples = int(cfg_t.num_fpo_samples)
        max_total_fpo_samples = int(getattr(cfg_t, "max_total_fpo_samples_per_call", 0))
        num_fpo_samples = base_num_fpo_samples
        if max_total_fpo_samples > 0:
            num_fpo_samples = max(1, min(base_num_fpo_samples, max_total_fpo_samples))
        reward_scaling = float(getattr(cfg_t, "reward_scaling", 1.0))

        # 给GAE的episode-wise数据
        batch_rews = []
        batch_vals = []
        batch_dones = []
        batch_truncs = []
        batch_discounts = []
        batch_bootstrap_values = []

        flat_critic_obs = []
        flat_naction_target = []  # normalized action prediction used for FPO ratio recomputation
        flat_loss_eps = []
        flat_loss_t = []
        flat_init_cfm_loss = []
        flat_cond_data = []
        flat_cond_mask = []
        flat_global_cond = []

        steps = 0
        obs_hist = deque(maxlen=n_obs_steps)
        # --------------------------------新增开始--------------------------------
        rollout_pbar = None
        if show_rollout_tqdm:
            rollout_pbar = tqdm.tqdm(
                total=target_steps,
                desc="Rollout",
                dynamic_ncols=True,
                leave=False,
                mininterval=rollout_tqdm_interval,
                position=1,
            )

        try:
            while steps < target_steps:
                with _maybe_suppress_stdout(suppress_env_step_print):
                    obs, _ = env.reset()
                obs_hist.clear()
                obs_hist.append(obs)

                ep_rews = []
                ep_vals = []
                ep_dones = []
                ep_truncs = []
                ep_discounts = []
                done = False
                ep_step = 0

                while (not done) and (ep_step < max_ep_steps) and (steps < target_steps):
                    obs_to = self._stack_last_n_obs(obs_hist, n_obs_steps)
                    obs_input = self._obs_to_policy_input(obs_to, device)

                    with torch.inference_mode():
                        out = self.model.sample_action_with_info(
                            obs_input,
                            num_fpo_samples=num_fpo_samples,
                        )
                        critic_in = out["global_cond"]
                        v = self.critic(critic_in).squeeze(-1)                     # [1]

                    # 执行动作：默认取第一个动作
                    # out["action"]: [1, Ta, Da]
                    env_action = out["action"][0, 0].detach().cpu().numpy()

                    with _maybe_suppress_stdout(suppress_env_step_print):
                        next_obs, rew, terminated, truncated, _ = env.step(env_action)
                    done = bool(terminated or truncated)

                    trunc = float(truncated)
                    discount = 0.0 if bool(terminated) else 1.0
                    ep_truncs.append(trunc)
                    ep_discounts.append(discount)

                    ep_rews.append(float(rew) * reward_scaling)
                    ep_vals.append(float(v.item()))
                    ep_dones.append(float(done))

                    flat_critic_obs.append(critic_in.squeeze(0).detach().cpu())                     # [D]
                    flat_naction_target.append(out["naction_pred"].squeeze(0).detach().cpu())      # [T, Da], normalized
                    flat_loss_eps.append(out["loss_eps"].squeeze(0).detach().cpu())                 # [N, T, Da]
                    flat_loss_t.append(out["loss_t"].squeeze(0).detach().cpu())                     # [N]
                    flat_init_cfm_loss.append(out["initial_cfm_loss"].squeeze(0).detach().cpu())    # [N]

                    flat_cond_data.append(out["cond_data"].squeeze(0).detach().cpu())             # [T, Da]
                    flat_cond_mask.append(out["cond_mask"].squeeze(0).detach().cpu())             # [T, Da]
                    flat_global_cond.append(out["global_cond"].squeeze(0).detach().cpu())         # [D]

                    obs_hist.append(next_obs)
                    steps += 1
                    ep_step += 1

                    if rollout_pbar is not None:
                        rollout_pbar.update(1)

                # 最后一个obs的bootstrap value
                last_obs_to = self._stack_last_n_obs(obs_hist, n_obs_steps)
                last_obs_input = self._obs_to_policy_input(last_obs_to, device)
                with torch.inference_mode():
                    last_v = self.critic(self._build_critic_input_from_obs(last_obs_input)).squeeze(-1).item()
                batch_truncs.append(ep_truncs)
                batch_discounts.append(ep_discounts)
                batch_bootstrap_values.append(float(last_v))

                batch_rews.append(ep_rews)
                batch_vals.append(ep_vals)
                batch_dones.append(ep_dones)
        finally:
            if rollout_pbar is not None:
                rollout_pbar.close()
        # --------------------------------新增结束--------------------------------
        flat_rewards = [r for ep in batch_rews for r in ep]
        ep_lens = [len(ep) for ep in batch_rews if len(ep) > 0]

        reward_mean = float(np.mean(flat_rewards)) if len(flat_rewards) > 0 else 0.0
        ep_len_mean = float(np.mean(ep_lens)) if len(ep_lens) > 0 else 0.0

        rollout = {
            "batch_rews": batch_rews,
            "batch_vals": batch_vals,
            "batch_dones": batch_dones,
            "critic_obs": torch.stack(flat_critic_obs, dim=0),
            "naction_target": torch.stack(flat_naction_target, dim=0),
            "cond_data": torch.stack(flat_cond_data, dim=0),
            "cond_mask": torch.stack(flat_cond_mask, dim=0),
            "global_cond": torch.stack(flat_global_cond, dim=0),
            "reward_mean": reward_mean,
            "ep_len_mean": ep_len_mean,
            "batch_truncs": batch_truncs,
            "batch_discounts": batch_discounts,
            "batch_bootstrap_values": batch_bootstrap_values,
        }

        rollout["loss_eps"] = torch.stack(flat_loss_eps, dim=0)
        rollout["loss_t"] = torch.stack(flat_loss_t, dim=0)
        rollout["initial_cfm_loss"] = torch.stack(flat_init_cfm_loss, dim=0)

        return rollout

    def collect_rollout_vector(self, env, device, num_envs):
        cfg_t = self.cfg.training
        suppress_env_step_print = bool(getattr(cfg_t, "suppress_env_step_print", True))
        show_rollout_tqdm = bool(getattr(cfg_t, "show_rollout_tqdm", True))
        rollout_tqdm_interval = float(getattr(cfg_t, "rollout_tqdm_interval_sec", 1.0))
        target_steps = int(cfg_t.timesteps_per_batch)
        max_ep_steps = int(cfg_t.max_timesteps_per_episode)
        n_obs_steps = int(self.cfg.n_obs_steps)
        base_num_fpo_samples = int(cfg_t.num_fpo_samples)
        max_total_fpo_samples = int(getattr(cfg_t, "max_total_fpo_samples_per_call", 0))
        reward_scaling = float(getattr(cfg_t, "reward_scaling", 1.0))

        # 初始化并行环境
        obs_list, _ = env.reset()

        # 初始化每个环境独立历史与episode缓存
        obs_hist_list = [deque(maxlen=n_obs_steps) for _ in range(num_envs)]
        ep_rews = [[] for _ in range(num_envs)]
        ep_vals = [[] for _ in range(num_envs)]
        ep_dones = [[] for _ in range(num_envs)]
        ep_truncs = [[] for _ in range(num_envs)]
        ep_discounts = [[] for _ in range(num_envs)]
        ep_steps = [0 for _ in range(num_envs)]
        alive = [True for _ in range(num_envs)]

        for i in range(num_envs):
            obs_hist_list[i].append(obs_list[i])

        # 全局batch容器
        batch_rews, batch_vals, batch_dones = [], [], []
        batch_truncs, batch_discounts, batch_bootstrap_values = [], [], []
        flat_critic_obs, flat_naction_target = [], []  # normalized action predictions for current-loss recomputation
        flat_loss_eps, flat_loss_t, flat_init_cfm_loss = [], [], []
        flat_cond_data, flat_cond_mask, flat_global_cond = [], [], []

        steps = 0
        rollout_pbar = None
        if show_rollout_tqdm:
            rollout_pbar = tqdm.tqdm(
                total=target_steps,
                desc="Rollout",
                dynamic_ncols=True,
                leave=False,
                mininterval=rollout_tqdm_interval,
                position=1,
            )

        try:
            while steps < target_steps:
                active_ids = [i for i in range(num_envs) if alive[i]]
                if len(active_ids) == 0:
                    reset_out = env.reset_at(list(range(num_envs)))
                    for i in range(num_envs):
                        obs_i, _ = reset_out[i]
                        obs_hist_list[i].clear()
                        obs_hist_list[i].append(obs_i)
                        ep_steps[i] = 0
                        alive[i] = True
                    continue

                obs_to_batch = {}
                for i in active_ids:
                    obs_to_i = self._stack_last_n_obs(obs_hist_list[i], n_obs_steps)
                    for k, v in obs_to_i.items():
                        obs_to_batch.setdefault(k, []).append(v)
                for k in obs_to_batch:
                    obs_to_batch[k] = np.stack(obs_to_batch[k], axis=0)

                num_fpo_samples = base_num_fpo_samples
                if max_total_fpo_samples > 0:
                    per_env_budget = max_total_fpo_samples // max(1, len(active_ids))
                    num_fpo_samples = max(1, min(base_num_fpo_samples, per_env_budget))

                obs_input = {}
                for k, v in obs_to_batch.items():
                    t = torch.from_numpy(v).to(device)
                    if t.dtype == torch.uint8:
                        t = t.float() / 255.0
                    obs_input[k] = t

                with torch.inference_mode():
                    out = self.model.sample_action_with_info(
                        obs_input,
                        num_fpo_samples=num_fpo_samples,
                    )
                    critic_in = out["global_cond"]
                    v = self.critic(critic_in).squeeze(-1)

                # 构造完整actions，非active env保持空动作占位
                action_dim = out["action"].shape[-1]
                actions_all = [np.zeros((action_dim,), dtype=np.float32) for _ in range(num_envs)]
                for local_j, env_id in enumerate(active_ids):
                    actions_all[env_id] = out["action"][local_j, 0].detach().cpu().numpy()

                with _maybe_suppress_stdout(suppress_env_step_print):
                    next_obs_list, rew_arr, terminated_arr, truncated_arr, _ = env.step(actions_all)

                for local_j, env_id in enumerate(active_ids):
                    done = bool(terminated_arr[env_id] or truncated_arr[env_id])
                    trunc = float(bool(truncated_arr[env_id]))
                    discount = 0.0 if bool(terminated_arr[env_id]) else 1.0

                    ep_rews[env_id].append(float(rew_arr[env_id]) * reward_scaling)
                    ep_vals[env_id].append(float(v[local_j].item()))
                    ep_dones[env_id].append(float(done))
                    ep_truncs[env_id].append(trunc)
                    ep_discounts[env_id].append(discount)

                    flat_critic_obs.append(critic_in[local_j].detach().cpu())
                    flat_naction_target.append(out["naction_pred"][local_j].detach().cpu())       # [T, Da], normalized
                    flat_loss_eps.append(out["loss_eps"][local_j].detach().cpu())
                    flat_loss_t.append(out["loss_t"][local_j].detach().cpu())
                    flat_init_cfm_loss.append(out["initial_cfm_loss"][local_j].detach().cpu())
                    flat_cond_data.append(out["cond_data"][local_j].detach().cpu())
                    flat_cond_mask.append(out["cond_mask"][local_j].detach().cpu())
                    flat_global_cond.append(out["global_cond"][local_j].detach().cpu())

                    obs_hist_list[env_id].append(next_obs_list[env_id])
                    ep_steps[env_id] += 1
                    steps += 1

                    if rollout_pbar is not None:
                        rollout_pbar.update(1)

                    reached_max = ep_steps[env_id] >= max_ep_steps
                    if done or reached_max:
                        last_obs_to = self._stack_last_n_obs(obs_hist_list[env_id], n_obs_steps)
                        last_obs_input = self._obs_to_policy_input(last_obs_to, device)
                        with torch.inference_mode():
                            last_v = self.critic(self._build_critic_input_from_obs(last_obs_input)).squeeze(-1).item()

                        batch_rews.append(ep_rews[env_id])
                        batch_vals.append(ep_vals[env_id])
                        batch_dones.append(ep_dones[env_id])
                        batch_truncs.append(ep_truncs[env_id])
                        batch_discounts.append(ep_discounts[env_id])
                        batch_bootstrap_values.append(float(last_v))

                        ep_rews[env_id] = []
                        ep_vals[env_id] = []
                        ep_dones[env_id] = []
                        ep_truncs[env_id] = []
                        ep_discounts[env_id] = []
                        ep_steps[env_id] = 0

                        if done:
                            reset_out = env.reset_at([env_id])
                            obs_i, _ = reset_out[env_id]
                            obs_hist_list[env_id].clear()
                            obs_hist_list[env_id].append(obs_i)
                        else:
                            obs_hist_list[env_id].clear()
                            obs_hist_list[env_id].append(next_obs_list[env_id])

                if steps >= target_steps:
                    break

            # flush未结束episode，保证GAE输入完整
            for i in range(num_envs):
                if len(ep_rews[i]) == 0:
                    continue
                last_obs_to = self._stack_last_n_obs(obs_hist_list[i], n_obs_steps)
                last_obs_input = self._obs_to_policy_input(last_obs_to, device)
                with torch.inference_mode():
                    last_v = self.critic(self._build_critic_input_from_obs(last_obs_input)).squeeze(-1).item()

                batch_rews.append(ep_rews[i])
                batch_vals.append(ep_vals[i])
                batch_dones.append(ep_dones[i])
                batch_truncs.append(ep_truncs[i])
                batch_discounts.append(ep_discounts[i])
                batch_bootstrap_values.append(float(last_v))
        finally:
            if rollout_pbar is not None:
                rollout_pbar.close()

        flat_rewards = [r for ep in batch_rews for r in ep]
        ep_lens = [len(ep) for ep in batch_rews if len(ep) > 0]

        reward_mean = float(np.mean(flat_rewards)) if len(flat_rewards) > 0 else 0.0
        ep_len_mean = float(np.mean(ep_lens)) if len(ep_lens) > 0 else 0.0

        rollout = {
            "batch_rews": batch_rews,
            "batch_vals": batch_vals,
            "batch_dones": batch_dones,
            "critic_obs": torch.stack(flat_critic_obs, dim=0),
            "naction_target": torch.stack(flat_naction_target, dim=0),
            "cond_data": torch.stack(flat_cond_data, dim=0),
            "cond_mask": torch.stack(flat_cond_mask, dim=0),
            "global_cond": torch.stack(flat_global_cond, dim=0),
            "reward_mean": reward_mean,
            "ep_len_mean": ep_len_mean,
            "batch_truncs": batch_truncs,
            "batch_discounts": batch_discounts,
            "batch_bootstrap_values": batch_bootstrap_values,
        }

        rollout["loss_eps"] = torch.stack(flat_loss_eps, dim=0)
        rollout["loss_t"] = torch.stack(flat_loss_t, dim=0)
        rollout["initial_cfm_loss"] = torch.stack(flat_init_cfm_loss, dim=0)

        return rollout

    def update_fpo_from_rollout(self, rollout, device):
        """
        使用 rollout 做一次 FPO 更新（actor+critic）
        """
        cfg_t = self.cfg.training
        clip_coef = float(cfg_t.clip_coef)
        n_updates = int(cfg_t.n_updates_per_iteration)
        max_grad_norm = float(getattr(cfg_t, "max_grad_norm", 1.0))
        value_loss_coeff = float(getattr(cfg_t, "value_loss_coeff", 0.25))

        ratio_mode = str(getattr(cfg_t, "ratio_mode", "vanilla"))
        clamp_logratio = getattr(cfg_t, "clamp_logratio", None)
        if clamp_logratio is not None:
            clamp_logratio = float(clamp_logratio)
        rho_min = float(getattr(cfg_t, "rho_min", 1e-6))
        rho_max = float(getattr(cfg_t, "rho_max", 100.0))

        flat_trunc = []
        for ep in rollout["batch_truncs"]:
            flat_trunc.extend(ep)
        flat_trunc = torch.tensor(flat_trunc, dtype=torch.float32, device=device)

        # ---- GAE ----
        advantages, returns = self.calculate_gae(
            rollout["batch_rews"],
            rollout["batch_vals"],
            rollout["batch_truncs"],
            rollout["batch_discounts"],\
            rollout["batch_bootstrap_values"],
        )
        advantages = advantages.to(device)
        returns = returns.to(device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        critic_obs = rollout["critic_obs"].to(device)            # [S, D]
        naction_target = rollout["naction_target"].to(device)    # [S, T, Da], normalized
        loss_eps = rollout["loss_eps"].to(device)                # [S, N, T, Da]
        loss_t = rollout["loss_t"].to(device)                    # [S, N]
        init_loss = rollout["initial_cfm_loss"].to(device)       # [S, N]
        cond_data = rollout["cond_data"].to(device)              # [S, T, Da]
        cond_mask = rollout["cond_mask"].to(device)              # [S, T, Da]
        global_cond = rollout["global_cond"].to(device)          # [S, D]

        S = critic_obs.shape[0]
        idx_all = np.arange(S)
        # 先给一个简单minibatch策略
        num_minibatches = int(getattr(cfg_t, "num_minibatches", 4))
        minibatch_size = max(1, S // num_minibatches)

        actor_losses = []
        critic_losses = []
        ratio_means = []
        cfm_diffs = []

        clipped_ratio_fracs = []

        for _ in range(n_updates):
            np.random.shuffle(idx_all)
            for start in range(0, S, minibatch_size):
                end = start + minibatch_size
                idx_np = idx_all[start:end]
                if len(idx_np) == 0:
                    continue
                idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)

                mb_adv = advantages[idx]              # [B]
                mb_ret = returns[idx]                 # [B]

                mb_naction = naction_target[idx]      # [B, T, Da]
                mb_eps = loss_eps[idx]                # [B, N, T, Da]
                mb_t = loss_t[idx]                    # [B, N]
                mb_init = init_loss[idx]              # [B, N]

                mb_cond_data = cond_data[idx]         # [B, T, Da]
                mb_cond_mask = cond_mask[idx]         # [B, T, Da]
                mb_global = global_cond[idx]          # [B, D]

                B, N, T, Da = mb_eps.shape

                # 展平 B*N
                # Recompute current CFM loss on the same normalized action scale
                # used when collecting initial_cfm_loss during rollout.
                flat_naction = mb_naction.unsqueeze(1).expand(B, N, T, Da).reshape(B * N, T, Da)
                flat_eps = mb_eps.reshape(B * N, T, Da)
                flat_t = mb_t.reshape(B * N)
                flat_init = mb_init.reshape(B * N)

                flat_cond_data = mb_cond_data.unsqueeze(1).expand(B, N, T, Da).reshape(B * N, T, Da)
                flat_cond_mask = mb_cond_mask.unsqueeze(1).expand(B, N, T, Da).reshape(B * N, T, Da)
                flat_global = mb_global.unsqueeze(1).expand(B, N, mb_global.shape[-1]).reshape(B * N, mb_global.shape[-1])

                postdict = {
                    "cond_data": flat_cond_data,
                    "cond_mask": flat_cond_mask.bool(),
                    "local_cond": None,
                    "global_cond": flat_global,
                }

                curr = self.model.compute_cfm_loss_with_fixed_noise_t(
                    postdict,
                    flat_naction,
                    flat_eps,
                    flat_t,
                    return_per_sample=True,
                    postdict=True,
                )  # [B*N]

                logratio = (flat_init - curr).reshape(B, N)

                if clamp_logratio is not None:
                    logratio = torch.clamp(logratio, min=-clamp_logratio, max=clamp_logratio)

                cfm_diffs.append(float(logratio.mean().item()))

                if ratio_mode == "vanilla":
                    ratio = torch.exp(logratio.mean(dim=1))
                    ratio = torch.clamp(ratio, rho_min, rho_max)

                    clipped_frac = (torch.abs(ratio - 1.0) > clip_coef).float().mean()
                    clipped_ratio_fracs.append(float(clipped_frac.item()))

                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * mb_adv
                    actor_loss = (-torch.min(surr1, surr2)).mean()

                elif ratio_mode == "per_sample":
                    ratio = torch.exp(logratio)
                    ratio = torch.clamp(ratio, rho_min, rho_max)

                    adv = mb_adv.unsqueeze(1).expand_as(ratio)

                    clipped_frac = (torch.abs(ratio - 1.0) > clip_coef).float().mean()
                    clipped_ratio_fracs.append(float(clipped_frac.item()))

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv
                    actor_loss = (-torch.min(surr1, surr2)).mean()

                else:
                    raise ValueError(f"Unknown ratio_mode: {ratio_mode}")

                self.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                # mb_trunc = flat_trunc[idx]
                ret_pred = self.critic(critic_obs[idx]).squeeze(-1)
                # v_error = (mb_ret - v_pred) * (1.0 - mb_trunc)
                ret_error = mb_ret - ret_pred
                critic_loss = (ret_error.pow(2).mean()) * value_loss_coeff

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                ratio_means.append(float(ratio.mean().item()))

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "rho_mean": float(np.mean(ratio_means)) if ratio_means else 0.0,
            "cfm_diff_mean": float(np.mean(cfm_diffs)) if cfm_diffs else 0.0,
            "clipped_ratio_frac": float(np.mean(clipped_ratio_fracs)) if clipped_ratio_fracs else 0.0,
        }

    def rew_and_adv(self,):
        pass

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
