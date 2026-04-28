import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm


def make_beta_schedule(
    schedule: str = 'linear',
    n_timesteps: int = 1000,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3
) -> np.ndarray:
    if schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timesteps, dtype=np.float64)
    elif schedule == 'cosine':
        timesteps = np.arange(n_timesteps + 1, dtype=np.float64) / n_timesteps + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = np.cos(alphas) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, 0, 0.999)
    elif schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timesteps, dtype=np.float64) ** 2
    else:
        raise ValueError(f'Unknown schedule: {schedule}')

    return betas


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    batch_size = t.shape[0]
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionNetwork(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        beta_schedule: dict,
        image_size: int = 32,
        channels: int = 3
    ):
        super().__init__()

        self.denoise_fn = denoise_fn
        self.image_size = image_size
        self.channels = channels

        self.set_new_noise_schedule(beta_schedule['train'], phase='train')
        self.test_schedule = beta_schedule['test']

    def set_new_noise_schedule(self, schedule_config: dict, phase: str = 'train'):
        betas = make_beta_schedule(**schedule_config)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        timesteps = len(betas)

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped',
                           to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                           to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                           to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        if phase == 'train':
            self.num_timesteps = timesteps
        else:
            self.num_test_timesteps = timesteps

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_input = torch.cat([cond, x_t], dim=1)
        noise_pred = self.denoise_fn(x_input, t)
        x_start = self.predict_start_from_noise(x_t, t, noise_pred)

        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)

        model_mean, model_log_variance = self.q_posterior(x_start, x_t, t)

        return model_mean, model_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        model_mean, model_log_variance = self.p_mean_variance(
            x_t, t, cond, clip_denoised=clip_denoised
        )

        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, int, int, int],
        return_intermediates: bool = False
    ) -> torch.Tensor:
        device = cond.device
        b = shape[0]

        img = torch.randn(shape, device=device)

        intermediates = [img] if return_intermediates else None

        self.set_new_noise_schedule(self.test_schedule, phase='test')

        for i in tqdm(reversed(range(self.num_test_timesteps)), desc='Sampling', total=self.num_test_timesteps, disable=True):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond, clip_denoised=True)

            if return_intermediates:
                intermediates.append(img)

        if hasattr(self, 'num_timesteps'):
            pass

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        batch_size: int = 1,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        batch_size = cond.shape[0]
        shape = (batch_size, self.channels, self.image_size, self.image_size)

        return self.p_sample_loop(cond, shape, return_intermediates)

    def forward(
        self,
        x_start: torch.Tensor,
        cond: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = x_start.shape[0]
        device = x_start.device

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        x_input = torch.cat([cond, x_noisy], dim=1)
        noise_pred = self.denoise_fn(x_input, t)

        return noise_pred, noise, t
