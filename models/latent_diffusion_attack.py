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
    def __init__(self, pretrained_path: str = None):
        super().__init__()
        

        if pretrained_path is None or pretrained_path == "null":
            print("Warning: Using simple VAE (not pretrained)")
            self.vae = SimpleVAE(
                in_channels=3,
                latent_channels=4,
                hidden_dims=[64, 128, 256, 512]
            )
        else:
            try:
                from diffusers import AutoencoderKL
                import os
                os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
                self.vae = AutoencoderKL.from_pretrained(
                    pretrained_path,
                    local_files_only=True,  
                    timeout=60  
                )
                print(f"✓ Loaded pretrained VAE from local cache: {pretrained_path}")
            except Exception as e:
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
        if x.min() >= 0:
            x = x * 2.0 - 1.0
        
        x = x.to(_get_module_device(self.vae))
        encoder_output = self.vae.encode(x)
        if hasattr(encoder_output, 'latent_dist'):
            z = encoder_output.latent_dist.sample()
        else:
            z = encoder_output.sample()
        z = z * self.scaling_factor
        
        return z
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        z = z / self.scaling_factor
        

        z = z.to(_get_module_device(self.vae))
        x = self.vae.decode(z).sample

        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0, 1)
        
        return x


class SimpleVAE(nn.Module):
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
        
       
        self.vae = VAEWrapper(vae_path)
        self.scaling_factor = scaling_factor
        

        default_unet_config = {
            'in_channels': 8,      
            'out_channels': 4,    
            'model_channels': 256,
            'num_res_blocks': 2,
            'channel_mult': (1, 2, 2, 4),
            'attention_resolutions': (32, 16, 8),
            'num_heads': 8
        }
        
        if unet is not None:
            self.unet = unet
        else:
            if unet_config:
                default_unet_config.update(unet_config)
            self.unet = UNet(**default_unet_config)
        
        self.qkv_extractor = qkv_extractor
        
        self.cross_attn_loss = CrossAttentionDistillationLoss(scale=1.0)
        
        self._init_noise_schedule(beta_schedule)
    
    def _init_noise_schedule(self, config):

        from .diffusion_network import make_beta_schedule, extract
        
        if config is None:
            config = {
                'train': {'schedule': 'linear', 'n_timesteps': 2000, 
                         'linear_start': 1e-6, 'linear_end': 0.01},
                'test': {'schedule': 'linear', 'n_timesteps': 100,
                        'linear_start': 1e-3, 'linear_end': 0.09}
            }
        
        train_schedule = config['train']
        betas = make_beta_schedule(**train_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(torch.tensor(alphas), dim=0)
        
        self.num_timesteps = len(betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1 - alphas_cumprod).float())
        
        self.test_schedule_config = config['test']
    
    def q_sample(self, x_start, t, noise=None):
  
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
        
        B = clean_img.shape[0]
        device = clean_img.device
        
       
        z_clean = self.vae.encode(clean_img)  # [B, 4, h, w]
        z_adv = self.vae.encode(adv_img)
        
       
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_adv)
        z_noisy = self.q_sample(z_adv, t, noise)
        
        
        z_input = torch.cat([z_clean, z_noisy], dim=1)  # [B, 8, h, w]
        noise_pred = self.unet(z_input, t)
        
        # 🟣 扩散损失
        loss_diff = F.mse_loss(noise_pred, noise)
        
        losses = {'diffusion': loss_diff.item()}
        total_loss = loss_diff
        

        if use_attention and self.qkv_extractor is not None:
            try:
               
                z_adv_pred = self._predict_x0(z_noisy, t, noise_pred)
                
              
                x_adv_pred = self.vae.decode(z_adv_pred)
                x_clean = self.vae.decode(z_clean)
                
               
                qkv_adv = self.qkv_extractor.extract_qkv(x_adv_pred)
                qkv_clean = self.qkv_extractor.extract_qkv(x_clean)
                
               
                loss_attn = self.cross_attn_loss(qkv_adv, qkv_clean)
                
                losses['cross_attention'] = loss_attn.item()
                total_loss = total_loss + 0.3 * loss_attn
                
            except Exception as e:
                print(f'Warning: Attention loss failed: {e}')
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def _predict_x0(self, z_t, t, noise_pred):

        from .diffusion_network import extract
        
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
       
        z_clean = self.vae.encode(clean_img)
        
        z_t = torch.randn_like(z_clean)
        
        intermediates = [] if return_intermediates else None
        
       
        for i in tqdm(reversed(range(num_steps)), desc='Sampling', total=num_steps, disable=True):
            t = torch.full((z_clean.shape[0],), i, device=z_clean.device, dtype=torch.long)
            
            z_input = torch.cat([z_clean, z_t], dim=1)
            noise_pred = self.unet(z_input, t)
            
            z_t = self._p_sample_step(z_t, t, noise_pred)
            
            if return_intermediates:
                intermediates.append(z_t)
        
        adv_img = self.vae.decode(z_t)
        
        if return_intermediates:
            return adv_img, intermediates
        
        return adv_img
    
    def _p_sample_step(self, z_t, t, noise_pred):
        from .diffusion_network import extract
        
        z_0_pred = self._predict_x0(z_t, t, noise_pred)
        z_0_pred = torch.clamp(z_0_pred, -1, 1)
        
        sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, z_t.shape)
        alpha_t = sqrt_alpha_t ** 2
        beta_t = 1 - alpha_t
        
        model_mean = z_0_pred  
        
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

        z_adv = self._latent_sample(clean_img)
        
        if refinement_iters > 0 and self.qkv_extractor is not None:
            z_adv = self._refine_with_attention(
                z_adv,
                clean_img,
                iters=refinement_iters,
                lr=refinement_lr,
                eps=eps
            )
        
        adv_img = self.vae.decode(z_adv)
        adv_img = self._clip_perturbation(adv_img, clean_img, eps)
        
        return adv_img
    
    def _latent_sample(self, clean_img):
        z_clean = self.vae.encode(clean_img)
        z_adv = torch.randn_like(z_clean)
        
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

        with torch.no_grad():
            qkv_clean = self.qkv_extractor.extract_qkv(clean_img)
        
        z_adv = z_adv.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z_adv], lr=lr)
        
        for iter_idx in range(iters):
            optimizer.zero_grad()
            
        
            x_adv = self.vae.decode(z_adv)

            qkv_adv = self.qkv_extractor.extract_qkv(x_adv)
            

            loss = self.cross_attn_loss(qkv_adv, qkv_clean)
            
            perturbation = x_adv - clean_img
            linf_norm = torch.max(torch.abs(perturbation))
            reg_loss = F.relu(linf_norm - eps) * 0.1
            
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
        
        return z_adv.detach()
    
    def _clip_perturbation(
        self,
        adv_img: torch.Tensor,
        clean_img: torch.Tensor,
        eps: float
    ) -> torch.Tensor:

        perturbation = adv_img - clean_img
        perturbation = torch.clamp(perturbation, -eps, eps)
        adv_img = clean_img + perturbation
        adv_img = torch.clamp(adv_img, 0, 1)
        return adv_img

