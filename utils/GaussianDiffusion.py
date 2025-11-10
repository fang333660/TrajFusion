# utils/GaussianDiffusion.py (G²Diff 升级版)
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Tuple, Any
from types import SimpleNamespace
import random


def linear_beta_schedule(timesteps: int, start: float = 0.0001, end: float = 0.02) -> torch.Tensor:
    return torch.linspace(start, end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0.0, 0.999).float()


class GaussianDiffusion:
    def __init__(self, config: SimpleNamespace, model: torch.nn.Module,
                 device: Union[str, torch.device]):
        self.config = config
        self.model = model
        self.device = torch.device(device)
        diffusion_cfg = getattr(config, 'diffusion', SimpleNamespace())
        self.num_timesteps = int(getattr(diffusion_cfg, 'num_diffusion_timesteps', 1000))
        self.epsilon = 1e-8
        schedule_name = getattr(diffusion_cfg, 'beta_schedule', 'linear')
        beta_start = float(getattr(diffusion_cfg, 'beta_start', 0.0001))
        beta_end = float(getattr(diffusion_cfg, 'beta_end', 0.02))
        cosine_s = float(getattr(diffusion_cfg, 'cosine_s', 0.008))
        if schedule_name == 'linear':
            self.betas = linear_beta_schedule(self.num_timesteps, start=beta_start, end=beta_end)
        elif schedule_name == 'cosine':
            self.betas = cosine_beta_schedule(self.num_timesteps, s=cosine_s)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_name}")
        self.betas = self.betas.to(self.device).float()
        self.alphas = (1.0 - self.betas).float()
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).float()
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).float()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod + self.epsilon).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / (self.alphas_cumprod + self.epsilon)).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / (self.alphas_cumprod + self.epsilon) - 1).float()
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (
                1.0 - self.alphas_cumprod + self.epsilon)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=self.epsilon)).float()
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (
                1.0 - self.alphas_cumprod + self.epsilon)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1.0 - self.alphas_cumprod + self.epsilon)
        print("GaussianDiffusion initialized (G²Diff 升级版).")

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t.long())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).float()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x0, device=self.device)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_x0_from_noise(self, xt: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, xt.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape)
        return sqrt_recip_alphas_cumprod_t * xt - sqrt_recipm1_alphas_cumprod_t * noise_pred

    # [!!! 核心修改: 重写 p_losses 方法 !!!]
    def p_losses(self, x0: torch.Tensor, attributes: torch.Tensor,
                 precomputed_lra_batch: Optional[torch.Tensor] = None,
                 t: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None,
                 uncond_prob: float = 0.0,
                 config_data_ref: Optional[SimpleNamespace] = None
                 ) -> dict[str, torch.Tensor]:

        batch_size, channels, traj_len = x0.shape
        x0_dev = x0.to(self.device)
        attributes_dev = attributes.to(self.device)
        lra_batch_dev = precomputed_lra_batch.to(self.device) if precomputed_lra_batch is not None else None

        if t is None: t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        if noise is None: noise = torch.randn_like(x0_dev)

        xt = self.q_sample(x0=x0_dev, t=t, noise=noise)

        # --- 模型预测 (保持不变) ---
        # LRA dropout 和噪声注入 (已修复变量名冲突)
        final_lra_batch = lra_batch_dev
        if self.model.training and lra_batch_dev is not None:
            if random.random() < 0.30:  # lra_dropout_prob
                final_lra_batch = torch.zeros_like(lra_batch_dev)
            else:
                lra_noise = torch.randn_like(lra_batch_dev) * 0.05  # noise_injection_level
                final_lra_batch = lra_batch_dev + lra_noise

        predicted_noise = self.model(xt, t, attributes_dev,
                                     precomputed_lra_batch=final_lra_batch,
                                     uncond_prob=uncond_prob)

        # --- 基础损失计算 (main_loss) ---
        loss_type = getattr(self.config.training, 'loss_type', 'l1').lower()
        if loss_type == 'l1':
            unweighted_loss = F.l1_loss(predicted_noise, noise, reduction='none')
        elif loss_type == 'mse':
            unweighted_loss = F.mse_loss(predicted_noise, noise, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        road_adherence_weight = getattr(self.config.training, 'road_adherence_weight', 0.0)
        if road_adherence_weight > 0 and lra_batch_dev is not None:
            with torch.no_grad():
                norm_dist = lra_batch_dev[..., 0]  # Shape: [B, L]
                adherence_weights = (1.0 - norm_dist.clamp(0.0, 1.0)) ** 2
                final_weights = (1.0 + road_adherence_weight * adherence_weights).unsqueeze(1)  # Shape: [B, 1, L]
            main_loss = (unweighted_loss * final_weights).mean()
        else:
            main_loss = unweighted_loss.mean()

        # --- G²Diff 新增损失项 ---
        x0_pred = self.predict_x0_from_noise(xt, t, predicted_noise)

        # 1. 终点锚定损失 (L_anchor)
        endpoint_weight = getattr(self.config.training, 'endpoint_loss_weight', 0.0)
        if endpoint_weight > 0:
            endpoint_loss = F.mse_loss(x0_pred[:, :, -1], x0_dev[:, :, -1])
        else:
            endpoint_loss = torch.tensor(0.0, device=self.device)

        # 2. LRA引导的几何平滑损失 (L_geom)
        geom_weight = getattr(self.config.training, 'geom_loss_weight', 0.0)
        if geom_weight > 0 and lra_batch_dev is not None and traj_len > 2:
            # 提取距离特征 (假设它在LRA特征向量的第一个位置)
            # lra_batch_dev shape: [B, L, D_lra]
            # 我们需要 accel_pred 的形状是 [B, C, L-2]，所以 norm_dist 也需要匹配
            norm_dist = lra_batch_dev[:, 1:-1, 0]  # Shape: [B, L-2]

            # 计算加速度
            vel_pred = x0_pred[:, :, 1:] - x0_pred[:, :, :-1]
            accel_pred = vel_pred[:, :, 1:] - vel_pred[:, :, :-1]  # Shape: [B, C, L-2]

            # 计算调制权重
            # w_geom shape: [B, L-2] -> [B, 1, L-2] 以便广播
            w_geom = torch.exp(-5.0 * norm_dist).unsqueeze(1)

            # 计算加权几何损失 (使用MSE更稳定)
            geom_loss = (w_geom * accel_pred.pow(2)).mean()
        else:
            geom_loss = torch.tensor(0.0, device=self.device)

        # --- 汇总总损失 ---
        total_loss = main_loss + endpoint_weight * endpoint_loss + geom_weight * geom_loss

        # 返回一个包含所有损失项的字典，方便监控
        return {
            'total_loss': total_loss,
            'main_loss': main_loss.detach(),
            'endpoint_loss': endpoint_loss.detach(),
            'geom_loss': geom_loss.detach()
        }

    # --- p_mean_variance, p_sample, p_sample_loop 保持不变 ---
    def p_mean_variance(self, xt: torch.Tensor, t_tensor: torch.Tensor, attributes: torch.Tensor,
                        precomputed_lra_batch: Optional[torch.Tensor] = None,
                        clip_denoised: bool = True,
                        uncond_prob_for_model_call: float = 0.0,
                        config_data_ref_for_model_call: Optional[SimpleNamespace] = None
                        ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        pred_noise = self.model(xt, t_tensor, attributes,
                                precomputed_lra_batch=precomputed_lra_batch,
                                uncond_prob=uncond_prob_for_model_call)
        x0_pred = self.predict_x0_from_noise(xt, t_tensor, pred_noise)
        if clip_denoised: x0_pred = x0_pred.clamp(-1., 1.)
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t_tensor, xt.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t_tensor, xt.shape)
        posterior_mean = posterior_mean_coef1_t * x0_pred + posterior_mean_coef2_t * xt
        posterior_log_variance_clipped_t = self._extract(self.posterior_log_variance_clipped, t_tensor, xt.shape)
        return posterior_mean, None, posterior_log_variance_clipped_t

    def p_sample(self, xt: torch.Tensor, t_int: int, attributes: torch.Tensor,
                 precomputed_lra_for_sample: Optional[torch.Tensor] = None,
                 clip_denoised: bool = True,
                 guidance_scale_sampling: float = 0.0,
                 config_data_ref_for_model_call: Optional[SimpleNamespace] = None
                 ) -> torch.Tensor:
        b = xt.shape[0]
        t_tensor = torch.full((b,), t_int, device=self.device, dtype=torch.long)
        p_mean_var_kwargs = {
            "precomputed_lra_batch": precomputed_lra_for_sample,
            "clip_denoised": clip_denoised,
            "config_data_ref_for_model_call": config_data_ref_for_model_call
        }
        if guidance_scale_sampling == 0.0:
            posterior_mean, _, log_var = self.p_mean_variance(xt, t_tensor, attributes, uncond_prob_for_model_call=0.0,
                                                              **p_mean_var_kwargs)
        else:
            mean_cond, _, log_var_cond = self.p_mean_variance(xt, t_tensor, attributes, uncond_prob_for_model_call=0.0,
                                                              **p_mean_var_kwargs)
            mean_uncond, _, _ = self.p_mean_variance(xt, t_tensor, attributes, uncond_prob_for_model_call=1.0,
                                                     **p_mean_var_kwargs)
            posterior_mean = mean_uncond + guidance_scale_sampling * (mean_cond - mean_uncond)
            log_var = log_var_cond
        noise_sample = torch.randn_like(xt)
        nonzero_mask = (t_tensor > 0).float().view(b, *((1,) * (len(xt.shape) - 1)))
        return posterior_mean + nonzero_mask * (0.5 * log_var).exp() * noise_sample

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, attributes: torch.Tensor,
                      precomputed_lra_for_sampling_run: Optional[torch.Tensor] = None,
                      guidance_scale_sampling: float = 0.0,
                      config_data_ref_for_model_call: Optional[SimpleNamespace] = None
                      ) -> torch.Tensor:
        xt = torch.randn(shape, device=self.device)
        attributes_dev = attributes.to(self.device)
        lra_dev = precomputed_lra_for_sampling_run.to(
            self.device) if precomputed_lra_for_sampling_run is not None else None
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="DDPM Sampling loop", total=self.num_timesteps,
                      leave=False):
            xt = self.p_sample(xt, i, attributes_dev,
                               precomputed_lra_for_sample=lra_dev,
                               guidance_scale_sampling=guidance_scale_sampling,
                               config_data_ref_for_model_call=config_data_ref_for_model_call)
        return xt.clamp(-1., 1.)