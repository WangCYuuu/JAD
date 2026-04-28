"""
JAD v2.0 Models Package

Core components:
- LatentDiffusionAttack: Main model
- QKVAttentionExtractor: Q,K,V extraction
- CrossAttentionDistillationLoss: Cross-attention loss
"""

from .latent_diffusion_attack import LatentDiffusionAttack, VAEWrapper
from .qkv_attention_extractor import (
    QKVAttentionExtractor,
    CrossAttentionDistillationLoss,
    create_qkv_extractor
)
from .unet import UNet
from .losses import DiffusionLoss

__all__ = [
    'LatentDiffusionAttack',
    'VAEWrapper',
    'QKVAttentionExtractor',
    'CrossAttentionDistillationLoss',
    'create_qkv_extractor',
    'UNet',
    'DiffusionLoss'
]

