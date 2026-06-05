from typing import Dict
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class FPOImagePolicy(BaseImagePolicy):
    """
    基于FPO的图像条件动作预测策略。
    
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=10,  # ODE 积分步数
        obs_as_global_cond=True,    # 是否将观测作为全局条件（而不是局部条件）一般在Transformer中设置为False
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        sigma_min=0.0,  # 最小噪声水平
        ode_solver='euler',  # 'euler' 或 'rk4'
        **kwargs,
    ):
        super().__init__()

        # 解析形状
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # 获取特征维度
        obs_feature_dim = obs_encoder.output_shape()[0]

        # 创建用于速度场预测的模型
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min
        self.ode_solver = ode_solver
        self.kwargs = kwargs

    def build_condition_from_obs(self, obs_dict: Dict[str, torch.Tensor], *args) -> Dict:
        # 归一化输入
        nobs = self.normalizer.normalize(obs_dict)
        if len(args) == 1:   # 训练阶段，使用动作作为条件的一部分
            nactions = self.normalizer["action"].normalize(args[0])
            batch_size = nactions.shape[0]
            horizon = nactions.shape[1]

            # 处理不同的观测传递方式
            local_cond = None
            global_cond = None
            trajectory = nactions
            cond_data = trajectory
            
            if self.obs_as_global_cond:
                # 重塑 B, T, ... 为 B*T
                this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs)

                global_cond = nobs_features.reshape(batch_size, -1)
            else:
                # 重塑 B, T, ... 为 B*T
                this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs)

                nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                cond_data = torch.cat([nactions, nobs_features], dim=-1)
                trajectory = cond_data.detach()
            
            conds = dict(trajectory=trajectory, batch_size=batch_size, cond_data=cond_data,
                         local_cond=local_cond, global_cond=global_cond)
            
        elif len(args) == 0:  # 采样阶段，不使用动作条件
            value = next(iter(nobs.values()))
            B, To = value.shape[:2]
            T = self.horizon
            Da = self.action_dim
            Do = self.obs_feature_dim 
            To = self.n_obs_steps

            device = self.device
            dtype = self.dtype

            # 处理不同的观测传递方式
            local_cond = None
            global_cond = None
            if self.obs_as_global_cond:
                # 通过全局特征作为条件
                this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs)

                global_cond = nobs_features.reshape(B, -1)

                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                # 通过修复（inpainting）作为条件
                this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs)

                nobs_features = nobs_features.reshape(B, To, -1)
                cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_data[:, :To, Da:] = nobs_features
                cond_mask[:, :To, Da:] = True
        
            conds = dict(cond_data=cond_data, cond_mask=cond_mask,
                         local_cond=local_cond, global_cond=global_cond)
        else:
            raise ValueError("不支持的参数组合")

        return conds

    # ========= 流匹配采样 ============
    def ode_step_euler(self, x, t, dt, local_cond=None, global_cond=None):
        """单步欧拉积分。"""
        # 在当前状态预测速度
        v = self.model(x, t, local_cond=local_cond, global_cond=global_cond)
        # 欧拉步: x_{t+dt} = x_t + v_t * dt
        x_next = x + v * dt
        return x_next

    def ode_step_rk4(self, x, t, dt, local_cond=None, global_cond=None):
        """单步 RK4（四阶龙格库塔）积分。"""
        # k1 = v(x, t)
        k1 = self.model(x, t, local_cond=local_cond, global_cond=global_cond)
        
        # k2 = v(x + k1*dt/2, t + dt/2)
        x2 = x + k1 * dt / 2
        t2 = t + dt / 2
        k2 = self.model(x2, t2, local_cond=local_cond, global_cond=global_cond)
        
        # k3 = v(x + k2*dt/2, t + dt/2)
        x3 = x + k2 * dt / 2
        k3 = self.model(x3, t2, local_cond=local_cond, global_cond=global_cond)
        
        # k4 = v(x + k3*dt, t + dt)
        x4 = x + k3 * dt
        t4 = t + dt
        k4 = self.model(x4, t4, local_cond=local_cond, global_cond=global_cond)
        
        # RK4 组合
        x_next = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        return x_next

    def flow_matching_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        return_path=False,
    ):
        """
        使用流匹配 ODE 积分采样轨迹。
        
        Args:
            condition_data: 条件信息
            condition_mask: 条件掩码
            local_cond: 局部条件
            global_cond: 全局条件（例如图像特征）
            generator: 随机数生成器用于可复现性
            return_path: 是否返回采样路径
        """
        device = condition_data.device
        dtype = condition_data.dtype
        
        # 从 t=0 的噪声开始
        x = torch.randn(
            size=condition_data.shape,
            dtype=dtype,
            device=device,
            generator=generator,
        ) * (1.0 - self.sigma_min) + self.sigma_min * condition_data

        if return_path:
            x_path = [x.detach().clone()]

        # 从 t=0 到 t=1 积分
        dt = 1.0 / self.num_inference_steps
        
        # 为批处理创建时间张量
        batch_size = x.shape[0]
        
        for step in range(self.num_inference_steps):
            t = step * dt
            # 为批次中的所有元素创建相同值的时间张量
            t_tensor = torch.full((batch_size,), t * 999, device=device, dtype=torch.long)
            
            # 在每一步应用条件
            x[condition_mask] = condition_data[condition_mask]
            
            # ODE 积分步
            if self.ode_solver == 'euler':
                x = self.ode_step_euler(x, t_tensor, dt, local_cond, global_cond)
            elif self.ode_solver == 'rk4':
                x = self.ode_step_rk4(x, t_tensor, dt, local_cond, global_cond)
            else:
                raise ValueError(f"未知的 ODE 求解器: {self.ode_solver}")
            
            if return_path:
                x_path.append(x.detach().clone())
        
        # 最终强制应用条件
        x[condition_mask] = condition_data[condition_mask]
        
        if return_path:
            x_t_path = torch.stack(x_path, dim=1)  # B, num_steps+1, ...
            return x, x_t_path
        return x

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        从观测预测动作。
        
        Args:
            obs_dict: 包含观测的字典
        Returns:
            包含 'action' 和 'action_pred' 键的字典
        """
        assert "past_action" not in obs_dict  # 尚未实现
        
        # # 归一化输入
        # nobs = self.normalizer.normalize(obs_dict)
        # value = next(iter(nobs.values()))
        # B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # # build input
        # device = self.device
        # dtype = self.dtype

        # # 处理不同的观测传递方式
        # local_cond = None
        # global_cond = None
        # if self.obs_as_global_cond:
        #     # 通过全局特征作为条件
        #     this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        #     nobs_features = self.obs_encoder(this_nobs)
        #     # 重塑回 B, Do
        #     global_cond = nobs_features.reshape(B, -1)
        #     # 动作的空数据
        #     cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        #     cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        # else:
        #     # 通过修复（inpainting）作为条件
        #     this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        #     nobs_features = self.obs_encoder(this_nobs)
        #     # 重塑回 B, T, Do
        #     nobs_features = nobs_features.reshape(B, To, -1)
        #     cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
        #     cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        #     cond_data[:, :To, Da:] = nobs_features
        #     cond_mask[:, :To, Da:] = True

        conds_dict = self.build_condition_from_obs(obs_dict)
        cond_data, cond_mask, local_cond, global_cond = conds_dict["cond_data"], conds_dict["cond_mask"], conds_dict["local_cond"], conds_dict["global_cond"]

        # 运行流匹配采样
        nsample = self.flow_matching_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= Training (Flow Matching Loss) ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        计算条件流匹配损失。
        
        流匹配损失: E[||v_θ(x_t, t) - (x_1 - x_0)||^2]
        其中 x_t = t*x_1 + (1-t)*x_0
        """
        # # 归一化输入
        # assert "valid_mask" not in batch
        # nobs = self.normalizer.normalize(batch["obs"])
        # nactions = self.normalizer["action"].normalize(batch["action"])
        # batch_size = nactions.shape[0]
        # horizon = nactions.shape[1]

        # # 处理不同的观测传递方式
        # local_cond = None
        # global_cond = None
        # trajectory = nactions
        # cond_data = trajectory
        
        # if self.obs_as_global_cond:
        #     # 重塑 B, T, ... 为 B*T
        #     this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        #     nobs_features = self.obs_encoder(this_nobs)
        #     # 重塑回 B, Do
        #     global_cond = nobs_features.reshape(batch_size, -1)
        # else:
        #     # 重塑 B, T, ... 为 B*T
        #     this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        #     nobs_features = self.obs_encoder(this_nobs)
        #     # 重塑回 B, T, Do
        #     nobs_features = nobs_features.reshape(batch_size, horizon, -1)
        #     cond_data = torch.cat([nactions, nobs_features], dim=-1)
        #     trajectory = cond_data.detach()

        conds_dict = self.build_condition_from_obs(batch["obs"], batch["action"])
        trajectory, batch_size, cond_data, local_cond, global_cond = conds_dict["trajectory"], conds_dict["batch_size"], conds_dict["cond_data"], conds_dict["local_cond"], conds_dict["global_cond"]

        # 生成条件掩码
        condition_mask = self.mask_generator(trajectory.shape)

        # 流匹配: x_1 是数据, x_0 是噪声
        x_1 = trajectory  # 目标数据
        x_0 = torch.randn_like(x_1)  # 源噪声
        
        # 对批次中每个样本采样随机时间 t ~ U(0, 1)
        t = torch.rand(batch_size, device=trajectory.device)
        
        # 插值: x_t = t*x_1 + (1-t)*x_0
        t_expanded = t.view(batch_size, 1, 1)  # 扩展用于广播
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        
        # 计算目标速度: v_t = x_1 - x_0
        target_v = x_1 - x_0
        
        # 对 x_t 应用条件
        x_t[condition_mask] = cond_data[condition_mask]
        
        # 将时间转换为与扩散模型相同的格式（为了与模型兼容）
        # 将 t 从 [0,1] 缩放到 [0, 999] 以匹配模型的预期输入
        timesteps = (t * 999).long()
        
        # 预测速度场
        pred_v = self.model(x_t, timesteps, local_cond=local_cond, global_cond=global_cond)
        
        # 只在非条件部分计算损失
        loss_mask = ~condition_mask
        
        # 流匹配损失: 预测速度和目标速度之间的 MSE
        loss = F.mse_loss(pred_v, target_v, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        
        return loss

    # ========= FPO Configuration ============
    def compute_cfm_loss_with_fixed_noise_t(self, obs_dict: Dict[str, torch.Tensor],
                                            naction_target: torch.Tensor, noise: torch.Tensor,
                                            t: torch.Tensor, return_per_sample=False, 
                                            postdict=False, return_pred_v=False):
        # naction_target must stay in normalized action space.
        assert naction_target.shape == noise.shape
        if not self.obs_as_global_cond:
            raise NotImplementedError(
                "当前实现仅支持 obs_as_global_cond=True。"
            )

        if postdict:
            prep_dict = obs_dict
        else:
            prep_dict = self.build_condition_from_obs(obs_dict)
        cond_data, cond_mask, local_cond, global_cond = prep_dict["cond_data"], prep_dict["cond_mask"], prep_dict["local_cond"], prep_dict["global_cond"]

        x_1 = naction_target
        x_0 = noise.to(device=x_1.device, dtype=x_1.dtype)

        # 插值: x_t = t*x_1 + (1-t)*x_0
        t_expanded = t.to(device=x_1.device, dtype=x_1.dtype).view(-1, 1, 1)  # 扩展用于广播
        x_t = t_expanded * x_1 + (1.0 - t_expanded) * x_0
        target_v = x_1 - x_0

        x_t[cond_mask] = cond_data[cond_mask]

        timesteps = (t * 999).long()
        pred_v = self.model(x_t, timesteps, local_cond=local_cond, global_cond=global_cond)

        loss_mask = ~cond_mask

        loss = F.mse_loss(pred_v, target_v, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        # loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.reshape(loss.shape[0], -1).sum(dim=1)

        result = loss if return_per_sample else loss.mean()
        if return_pred_v:
            # The actor update reuses this velocity for KL instead of forwarding
            # the current policy a second time at the same fixed-noise points.
            return result, pred_v, loss_mask
        return result

    def compute_cfm_velocity_with_fixed_noise_t(self, obs_dict: Dict[str, torch.Tensor],
                                                naction_target: torch.Tensor, noise: torch.Tensor,
                                                t: torch.Tensor, postdict=False):
        # Return the predicted flow velocity for the same fixed-noise CFM points
        # used by FPO, plus the non-conditioned mask where losses are valid.
        assert naction_target.shape == noise.shape
        if not self.obs_as_global_cond:
            raise NotImplementedError(
                "当前实现仅支持 obs_as_global_cond=True。"
            )

        if postdict:
            prep_dict = obs_dict
        else:
            prep_dict = self.build_condition_from_obs(obs_dict)
        cond_data, cond_mask, local_cond, global_cond = prep_dict["cond_data"], prep_dict["cond_mask"], prep_dict["local_cond"], prep_dict["global_cond"]

        x_1 = naction_target
        x_0 = noise.to(device=x_1.device, dtype=x_1.dtype)

        t_expanded = t.to(device=x_1.device, dtype=x_1.dtype).view(-1, 1, 1)
        x_t = t_expanded * x_1 + (1.0 - t_expanded) * x_0
        x_t[cond_mask] = cond_data[cond_mask]

        timesteps = (t * 999).long()
        pred_v = self.model(x_t, timesteps, local_cond=local_cond, global_cond=global_cond)
        return pred_v, ~cond_mask

    # 采样动作 + 产出FPO训练所需信息
    def sample_action_with_info(self, obs_dict: Dict[str, torch.Tensor], num_fpo_samples,
                                return_path=False, compute_initial_cfm_loss=True):
        N = int(num_fpo_samples)
        conds_dict = self.build_condition_from_obs(obs_dict)
        cond_data, cond_mask, local_cond, global_cond = conds_dict["cond_data"], conds_dict["cond_mask"], conds_dict["local_cond"], conds_dict["global_cond"]

        x_t_path = None
        if return_path:
            nsample, x_t_path = self.flow_matching_sample(
                cond_data,
                cond_mask,
                local_cond,
                global_cond,
                return_path=True
            )
        else:
            nsample = self.flow_matching_sample(
                cond_data,
                cond_mask,
                local_cond,
                global_cond,
                return_path=False
            )
        naction_pred = nsample[..., :self.action_dim]
        # 反归一化！
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # 构建用于FPO训练的样本路径
        B, T, Da = naction_pred.shape
        x1 = naction_pred.detach().unsqueeze(1).expand(B, N, T, Da).reshape(B * N, T, Da)
        eps = torch.randn_like(x1)
        
        # t采样
        discretize_t = bool(self.kwargs.get("discretize_t_for_training", True))
        if discretize_t:
            idx = torch.randint(low=0, high=self.num_inference_steps, size=(B * N,), device=x1.device)
            t = idx.float() / float(self.num_inference_steps)
        else:
            t = torch.rand(B * N, device=x1.device)

        initial_cfm_loss = None
        if compute_initial_cfm_loss:
            # obs_tiled = dict_apply(obs_dict, lambda x: x.repeat_interleave(N, dim=0))
            def repeat_bn(tensor):
                if tensor is None: return None
                return tensor.repeat_interleave(N, dim=0)

            obs_tiled = {
                "cond_data": repeat_bn(conds_dict["cond_data"]),
                "cond_mask": repeat_bn(conds_dict["cond_mask"]),
                "local_cond": repeat_bn(conds_dict["local_cond"]),
                "global_cond": repeat_bn(conds_dict["global_cond"]),
            }

            # 计算初始CFM损失
            initial_cfm_loss = self.compute_cfm_loss_with_fixed_noise_t(
                obs_tiled,
                x1,
                eps,
                t,
                return_per_sample=True,
                postdict=True,
            )
            initial_cfm_loss = initial_cfm_loss.detach().reshape(B, N)

        return {
            "action": action,                         # unnormalized action used by env.step
            "action_pred": action_pred,               # unnormalized full-horizon action prediction
            "naction_pred": naction_pred.detach(),    # normalized full-horizon action prediction
            "x_t_path": x_t_path,
            "loss_eps": eps.reshape(B, N, T, Da),
            "loss_t": t.reshape(B, N),
            "initial_cfm_loss": initial_cfm_loss,
            "cond_data": cond_data,
            "cond_mask": cond_mask,
            "global_cond": global_cond,
        }



    # ========= 仅供调试用 ============
    def run_fpo_forward_smoke_test(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        num_fpo_samples: int = 4,
        batch_size: int = 2,
    ) -> Dict[str, object]:
        """检查 A/B/C 前向是否可运行且输出形状有效。"""
        assert "obs" in batch and "action" in batch
        assert batch_size > 0
        N = int(num_fpo_samples)
        assert N > 0

        obs = dict_apply(batch["obs"], lambda x: x[:batch_size])
        action = batch["action"][:batch_size]
        B = action.shape[0]

        with torch.no_grad():
            out = self.sample_action_with_info(obs, num_fpo_samples=N)

            required_keys = [
                "action",
                "action_pred",
                "x_t_path",
                "loss_eps",
                "loss_t",
                "initial_cfm_loss",
            ]
            for key in required_keys:
                assert key in out, f"sample_action_with_info 缺少字段: {key}"

            assert out["action"].shape[0] == B
            assert out["action_pred"].shape[0] == B
            assert out["loss_eps"].shape[0] == B
            assert out["loss_t"].shape == (B, N)
            assert out["initial_cfm_loss"].shape == (B, N)

            if out["x_t_path"] is not None:
                assert out["x_t_path"].shape[0] == B
                assert out["x_t_path"].shape[1] == self.num_inference_steps + 1

            for key in ["action_pred", "loss_eps", "loss_t", "initial_cfm_loss"]:
                assert torch.isfinite(out[key]).all(), f"{key} 含有 NaN/Inf"

            naction = self.normalizer["action"].normalize(action)
            noise = torch.randn_like(naction)
            t = torch.rand(B, device=naction.device, dtype=naction.dtype)
            per_sample_loss = self.compute_cfm_loss_with_fixed_noise_t(
                obs,
                naction,
                noise,
                t,
                return_per_sample=True,
            )

            assert per_sample_loss.shape == (B,)
            assert torch.isfinite(per_sample_loss).all(), "per_sample_loss 含有 NaN/Inf"

        return {
            "status": "ok",
            "batch_size": int(B),
            "num_fpo_samples": int(N),
            "action_shape": tuple(out["action"].shape),
            "action_pred_shape": tuple(out["action_pred"].shape),
            "x_t_path_shape": None if out["x_t_path"] is None else tuple(out["x_t_path"].shape),
            "loss_eps_shape": tuple(out["loss_eps"].shape),
            "loss_t_shape": tuple(out["loss_t"].shape),
            "initial_cfm_loss_shape": tuple(out["initial_cfm_loss"].shape),
            "cfm_loss_shape": tuple(per_sample_loss.shape),
            "cfm_loss_mean": float(per_sample_loss.mean().item()),
        }

    def run_fpo_reproducibility_check(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        num_fpo_samples: int = 4,
        seed: int = 1234,
        batch_size: int = 2,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> Dict[str, object]:
        """检查同一输入与随机种子下 C 的输出是否可复现。"""
        assert "obs" in batch
        assert batch_size > 0
        N = int(num_fpo_samples)
        assert N > 0

        obs = dict_apply(batch["obs"], lambda x: x[:batch_size])

        def _set_seed(value: int):
            random.seed(value)
            np.random.seed(value)
            torch.manual_seed(value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(value)

        with torch.no_grad():
            _set_seed(seed)
            out1 = self.sample_action_with_info(obs, num_fpo_samples=N)

            _set_seed(seed)
            out2 = self.sample_action_with_info(obs, num_fpo_samples=N)

            torch.testing.assert_close(out1["action_pred"], out2["action_pred"], atol=atol, rtol=rtol)
            torch.testing.assert_close(out1["loss_eps"], out2["loss_eps"], atol=atol, rtol=rtol)
            torch.testing.assert_close(out1["loss_t"], out2["loss_t"], atol=atol, rtol=rtol)
            torch.testing.assert_close(out1["initial_cfm_loss"], out2["initial_cfm_loss"], atol=atol, rtol=rtol)

            if out1["x_t_path"] is not None and out2["x_t_path"] is not None:
                torch.testing.assert_close(out1["x_t_path"], out2["x_t_path"], atol=atol, rtol=rtol)

        return {
            "status": "ok",
            "seed": int(seed),
            "batch_size": int(batch_size),
            "num_fpo_samples": int(N),
            "checked_tensors": [
                "action_pred",
                "loss_eps",
                "loss_t",
                "initial_cfm_loss",
                "x_t_path",
            ],
            "atol": float(atol),
            "rtol": float(rtol),
        }
