from typing import Dict
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


class FlowMatchingImagePolicy(BaseImagePolicy):
    """
    基于条件流匹配（Conditional Flow Matching）的图像条件动作预测策略。
    
    流匹配学习一个速度场 v_t(x)，将噪声转换为数据。
    - 训练：使用 CFM 损失学习速度场
    - 推理：从 t=0（噪声）到 t=1（数据）积分 ODE
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=10,  # ODE 积分步数
        obs_as_global_cond=True,
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
    ):
        """
        使用流匹配 ODE 积分采样轨迹。
        
        Args:
            condition_data: 条件信息
            condition_mask: 条件掩码
            local_cond: 局部条件
            global_cond: 全局条件（例如图像特征）
            generator: 随机数生成器用于可复现性
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
        
        # 最终强制应用条件
        x[condition_mask] = condition_data[condition_mask]
        
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
        
        # 归一化输入
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # 处理不同的观测传递方式
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # 通过全局特征作为条件
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # 重塑回 B, Do
            global_cond = nobs_features.reshape(B, -1)
            # 动作的空数据
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # 通过修复（inpainting）作为条件
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # 重塑回 B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

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
        # 归一化输入
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
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
            # 重塑回 B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # 重塑 B, T, ... 为 B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # 重塑回 B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

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
