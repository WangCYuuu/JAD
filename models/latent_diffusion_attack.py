"""
Latent Diffusion Attack Model (JAD v2.0)
在 Latent Space 进行扩散，结合交叉注意力蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import math

from .unet import UNet


def _get_module_device(m: nn.Module) -> torch.device:
    for p in m.parameters():
        return p.device
    for b in m.buffers():
        return b.device
    return torch.device("cpu")
from .qkv_attention_extractor import QKVAttentionExtractor, CrossAttentionDistillationLoss


class VAEWrapper(nn.Module):
    """VAE 包装器（使用预训练的 VAE）"""
    def __init__(self, pretrained_path: str = None):
        super().__init__()
        
        # 加载预训练的 VAE
        if pretrained_path is None or pretrained_path == "null":
            # 直接使用简单 VAE
            print("Warning: Using simple VAE (not pretrained)")
            self.vae = SimpleVAE(
                in_channels=3,
                latent_channels=4,
                hidden_dims=[64, 128, 256, 512]
            )
        else:
            # 尝试加载预训练 VAE
            try:
                from diffusers import AutoencoderKL
                import os
                os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
                self.vae = AutoencoderKL.from_pretrained(
                    pretrained_path,
                    local_files_only=True,  # 只使用本地缓存，不联网
                    timeout=60  # 60秒超时
                )
                print(f"✓ Loaded pretrained VAE from local cache: {pretrained_path}")
            except Exception as e:
                # 如果下载失败，使用简单的 VAE
                print(f"Warning: Failed to load pretrained VAE ({str(e)[:100]})")
                print("Falling back to simple VAE")
                self.vae = SimpleVAE(
                    in_channels=3,
                    latent_channels=4,
                    hidden_dims=[64, 128, 256, 512]
                )
        
        # 冻结 VAE
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.scaling_factor = 0.18215
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码到 latent space
        
        Args:
            x: [B, 3, H, W] 图像 (0-1 或 -1-1)
        
        Returns:
            z: [B, 4, h, w] latent
        """
        # 归一化到 [-1, 1] (VAE 期望的输入)
        if x.min() >= 0:
            x = x * 2.0 - 1.0
        
        # 编码 (兼容不同版本的 diffusers)
        x = x.to(_get_module_device(self.vae))
        encoder_output = self.vae.encode(x)
        if hasattr(encoder_output, 'latent_dist'):
            z = encoder_output.latent_dist.sample()
        else:
            # 新版本直接返回分布对象
            z = encoder_output.sample()
        z = z * self.scaling_factor
        
        return z
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从 latent space 解码
        
        Args:
            z: [B, 4, h, w] latent
        
        Returns:
            x: [B, 3, H, W] 图像 (0-1)
        """
        # 反缩放
        z = z / self.scaling_factor
        
        # 解码
        z = z.to(_get_module_device(self.vae))
        x = self.vae.decode(z).sample
        
        # 归一化到 [0, 1]
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0, 1)
        
        return x


class SimpleVAE(nn.Module):
    """简单的 VAE 实现（如果无法加载预训练模型）"""
    def __init__(self, in_channels=3, latent_channels=4, hidden_dims=[64, 128, 256]):
        super().__init__()
        
        # Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_channels, 1)
        self.fc_logvar = nn.Conv2d(hidden_dims[-1], latent_channels, 1)
        
        # Decoder
        modules = []
        in_channels = latent_channels
        for h_dim in reversed(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        
        modules.append(nn.Conv2d(hidden_dims[0], 3, 3, padding=1))
        modules.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*modules)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        
        class Distribution:
            def __init__(self, mu):
                self.mu = mu
            def sample(self):
                return self.mu
        
        return Distribution(mu)
    
    def decode(self, z):
        x = self.decoder(z)
        
        class Output:
            def __init__(self, sample):
                self.sample = sample
        
        return Output(x)


class LatentDiffusionAttack(nn.Module):
    """
    基于 Latent Space 的对抗攻击扩散模型（改进版）
    
    核心改进：
    1. 在 latent space 工作（降维）
    2. 使用 Q,K,V 分离提取
    3. 采用交叉注意力蒸馏
    4. 推理时可选精炼
    """
    def __init__(
        self,
        vae_path: str = None,
        unet_config: dict = None,
        unet: nn.Module = None,
        beta_schedule: dict = None,
        qkv_extractor: QKVAttentionExtractor = None,
        scaling_factor: float = 0.18215
    ):
        super().__init__()
        
        # 1. VAE（编码/解码）
        self.vae = VAEWrapper(vae_path)
        self.scaling_factor = scaling_factor
        
        # 2. Latent U-Net
        default_unet_config = {
            'in_channels': 8,      # z_clean(4) + z_noisy(4)
            'out_channels': 4,     # latent 噪声
            'model_channels': 256,
            'num_res_blocks': 2,
            'channel_mult': (1, 2, 2, 4),
            'attention_resolutions': (32, 16, 8),
            'num_heads': 8
        }
        
        if unet is not None:
            # If an external unet is provided, use it directly.
            self.unet = unet
        else:
            # Otherwise, fall back to the default internal unet creation.
            if unet_config:
                default_unet_config.update(unet_config)
            self.unet = UNet(**default_unet_config)
        
        # 3. QKV 注意力提取器
        self.qkv_extractor = qkv_extractor
        
        # 4. 交叉注意力损失
        self.cross_attn_loss = CrossAttentionDistillationLoss(scale=1.0)
        
        # 5. 扩散调度器
        self._init_noise_schedule(beta_schedule)
    
    def _init_noise_schedule(self, config):
        """初始化噪声调度"""
        from .diffusion_network import make_beta_schedule, extract
        
        if config is None:
            config = {
                'train': {'schedule': 'linear', 'n_timesteps': 2000, 
                         'linear_start': 1e-6, 'linear_end': 0.01},
                'test': {'schedule': 'linear', 'n_timesteps': 100,
                        'linear_start': 1e-3, 'linear_end': 0.09}
            }
        
        # 训练调度
        train_schedule = config['train']
        betas = make_beta_schedule(**train_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(torch.tensor(alphas), dim=0)
        
        self.num_timesteps = len(betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1 - alphas_cumprod).float())
        
        # 保存测试调度配置
        self.test_schedule_config = config['test']
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        from .diffusion_network import extract
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward_train(
        self,
        clean_img: torch.Tensor,
        adv_img: torch.Tensor,
        use_attention: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        训练时的前向传播
        
        Args:
            clean_img: [B, 3, H, W] 干净图像 (0-1)
            adv_img: [B, 3, H, W] 对抗样本 (0-1)
            use_attention: 是否使用注意力蒸馏
        
        Returns:
            (total_loss, loss_dict)
        """
        B = clean_img.shape[0]
        device = clean_img.device
        
        # 🔵 编码到 latent space
        z_clean = self.vae.encode(clean_img)  # [B, 4, h, w]
        z_adv = self.vae.encode(adv_img)
        
        # 🟡 前向扩散（latent space）
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_adv)
        z_noisy = self.q_sample(z_adv, t, noise)
        
        # 🟢 噪声预测（latent space）
        z_input = torch.cat([z_clean, z_noisy], dim=1)  # [B, 8, h, w]
        noise_pred = self.unet(z_input, t)
        
        # 🟣 扩散损失
        loss_diff = F.mse_loss(noise_pred, noise)
        
        losses = {'diffusion': loss_diff.item()}
        total_loss = loss_diff
        
        # 🔴 注意力蒸馏损失（可选）
        if use_attention and self.qkv_extractor is not None:
            try:
                # 重建 latent
                z_adv_pred = self._predict_x0(z_noisy, t, noise_pred)
                
                # 解码到像素空间
                x_adv_pred = self.vae.decode(z_adv_pred)
                x_clean = self.vae.decode(z_clean)
                
                # 🔑 提取 Q, K, V
                qkv_adv = self.qkv_extractor.extract_qkv(x_adv_pred)
                qkv_clean = self.qkv_extractor.extract_qkv(x_clean)
                
                # 🔑 交叉注意力损失
                loss_attn = self.cross_attn_loss(qkv_adv, qkv_clean)
                
                losses['cross_attention'] = loss_attn.item()
                total_loss = total_loss + 0.3 * loss_attn
                
            except Exception as e:
                print(f'Warning: Attention loss failed: {e}')
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def _predict_x0(self, z_t, t, noise_pred):
        """从噪声预测原始 latent"""
        from .diffusion_network import extract
        
        # 确保必要的 buffer 已注册（先检查再使用）
        if not hasattr(self, 'alphas_cumprod'):
            alphas_cumprod = self.sqrt_alphas_cumprod ** 2
            self.register_buffer('alphas_cumprod', alphas_cumprod)
            self.register_buffer('sqrt_recip_alphas_cumprod',
                               torch.sqrt(1.0 / alphas_cumprod))
            self.register_buffer('sqrt_recipm1_alphas_cumprod',
                               torch.sqrt(1.0 / alphas_cumprod - 1))
        
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, z_t.shape) * z_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, z_t.shape) * noise_pred
        )
    
    @torch.no_grad()
    def sample(
        self,
        clean_img: torch.Tensor,
        num_steps: int = 100,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        采样对抗样本（latent space）
        
        Args:
            clean_img: [B, 3, H, W] 干净图像 (0-1)
            num_steps: 采样步数
        
        Returns:
            adv_img: [B, 3, H, W] 对抗样本 (0-1)
        """
        # 编码
        z_clean = self.vae.encode(clean_img)
        
        # 从噪声开始
        z_t = torch.randn_like(z_clean)
        
        intermediates = [] if return_intermediates else None
        
        # 逆向去噪
        for i in tqdm(reversed(range(num_steps)), desc='Sampling', total=num_steps, disable=True):
            t = torch.full((z_clean.shape[0],), i, device=z_clean.device, dtype=torch.long)
            
            # 预测噪声
            z_input = torch.cat([z_clean, z_t], dim=1)
            noise_pred = self.unet(z_input, t)
            
            # 更新 latent
            z_t = self._p_sample_step(z_t, t, noise_pred)
            
            if return_intermediates:
                intermediates.append(z_t)
        
        # 解码
        adv_img = self.vae.decode(z_t)
        
        if return_intermediates:
            return adv_img, intermediates
        
        return adv_img
    
    def _p_sample_step(self, z_t, t, noise_pred):
        """单步采样"""
        from .diffusion_network import extract
        
        # 预测 x_0
        z_0_pred = self._predict_x0(z_t, t, noise_pred)
        z_0_pred = torch.clamp(z_0_pred, -1, 1)
        
        # 计算后验均值
        # 获取 alpha_t 和 beta_t（正确的形状）
        sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, z_t.shape)
        alpha_t = sqrt_alpha_t ** 2
        beta_t = 1 - alpha_t
        
        model_mean = z_0_pred  # 简化
        
        # 添加噪声
        if t[0] > 0:
            noise = torch.randn_like(z_t)
            z_t = model_mean + torch.sqrt(beta_t) * noise
        else:
            z_t = model_mean
        
        return z_t
    
    def attack_with_refinement(
        self,
        clean_img: torch.Tensor,
        eps: float = 16/255,
        refinement_iters: int = 3,
        refinement_lr: float = 0.05
    ) -> torch.Tensor:
        """
        攻击 + 推理时精炼（完整版本）
        
        Args:
            clean_img: [B, 3, H, W] 干净图像
            eps: L∞ 约束
            refinement_iters: 精炼迭代次数
            refinement_lr: 精炼学习率
        
        Returns:
            对抗样本 [B, 3, H, W]
        """
        # 🔵 阶段1: 快速采样（latent diffusion）
        z_adv = self._latent_sample(clean_img)
        
        # 🟢 阶段2: 推理时精炼（类似 AttentionDistillation）
        if refinement_iters > 0 and self.qkv_extractor is not None:
            z_adv = self._refine_with_attention(
                z_adv,
                clean_img,
                iters=refinement_iters,
                lr=refinement_lr,
                eps=eps
            )
        
        # 🟣 阶段3: 解码并裁剪
        adv_img = self.vae.decode(z_adv)
        adv_img = self._clip_perturbation(adv_img, clean_img, eps)
        
        return adv_img
    
    def _latent_sample(self, clean_img):
        """Latent space 采样"""
        z_clean = self.vae.encode(clean_img)
        z_adv = torch.randn_like(z_clean)
        
        # 100 步去噪
        for i in reversed(range(100)):
            t = torch.full((z_clean.shape[0],), i, device=z_clean.device, dtype=torch.long)
            z_input = torch.cat([z_clean, z_adv], dim=1)
            noise_pred = self.unet(z_input, t)
            z_adv = self._p_sample_step(z_adv, t, noise_pred)
        
        return z_adv
    
    def _refine_with_attention(
        self,
        z_adv: torch.Tensor,
        clean_img: torch.Tensor,
        iters: int,
        lr: float,
        eps: float
    ) -> torch.Tensor:
        """
        使用注意力蒸馏精炼 latent
        类似 AttentionDistillation 的推理时优化
        
        Args:
            z_adv: [B, 4, h, w] 初始 latent
            clean_img: [B, 3, H, W] 干净图像
            iters: 迭代次数
            lr: 学习率
            eps: 扰动约束（用于正则化）
        """
        # 🔑 提取干净图像的 Q, K, V（固定）
        with torch.no_grad():
            qkv_clean = self.qkv_extractor.extract_qkv(clean_img)
        
        # 🔧 优化 latent
        z_adv = z_adv.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z_adv], lr=lr)
        
        for iter_idx in range(iters):
            optimizer.zero_grad()
            
            # 解码
            x_adv = self.vae.decode(z_adv)
            
            # 提取对抗样本的 Q, K, V
            qkv_adv = self.qkv_extractor.extract_qkv(x_adv)
            
            # 🔑 计算交叉注意力损失
            loss = self.cross_attn_loss(qkv_adv, qkv_clean)
            
            # 🔧 可选：添加扰动正则化
            perturbation = x_adv - clean_img
            linf_norm = torch.max(torch.abs(perturbation))
            reg_loss = F.relu(linf_norm - eps) * 0.1
            
            total_loss = loss + reg_loss
            
            # 反向传播（只更新 z_adv）
            total_loss.backward()
            optimizer.step()
        
        return z_adv.detach()
    
    def _clip_perturbation(
        self,
        adv_img: torch.Tensor,
        clean_img: torch.Tensor,
        eps: float
    ) -> torch.Tensor:
        """裁剪扰动到 eps 范围"""
        perturbation = adv_img - clean_img
        perturbation = torch.clamp(perturbation, -eps, eps)
        adv_img = clean_img + perturbation
        adv_img = torch.clamp(adv_img, 0, 1)
        return adv_img

