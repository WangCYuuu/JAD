import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [B] or [B, 1]
        Returns:
            [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        if use_scale_shift_norm:
            self.time_emb = nn.Sequential(
                Swish(),
                nn.Linear(time_emb_dim, out_channels * 2)
            )
        else:
            self.time_emb = nn.Sequential(
                Swish(),
                nn.Linear(time_emb_dim, out_channels)
            )
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_emb_dim]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        emb = self.time_emb(time_emb)[:, :, None, None]
        if self.use_scale_shift_norm:
            scale, shift = emb.chunk(2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = self.norm2(h + emb)
        
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        h = self.norm(x)
        
        qkv = self.qkv(h)  # [B, 3C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        
        h = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        h = self.proj(h)
        
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,        
        out_channels: int = 3,       
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = (16, 8),
        channel_mult: Tuple[int] = (1, 2, 2, 4),
        num_heads: int = 8,
        dropout: float = 0.0,
        use_fp16: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.dtype = torch.float16 if use_fp16 else torch.float32
        

        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        

        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = [model_channels]
        ch = model_channels
        ds = 1  
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, out_ch, time_emb_dim, dropout=dropout)
                ]
                ch = out_ch
                channels.append(ch)
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.down_blocks.append(nn.ModuleList(layers))
            
            if level != len(channel_mult) - 1:
                self.down_samples.append(Downsample(ch))
                channels.append(ch)
                ds *= 2
            else:
                self.down_samples.append(nn.Identity())
        
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim, dropout=dropout),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, ch, time_emb_dim, dropout=dropout)
        ])
        
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                ich = channels.pop()
                layers = [
                    ResBlock(ch + ich, out_ch, time_emb_dim, dropout=dropout)
                ]
                ch = out_ch
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.up_blocks.append(nn.ModuleList(layers))
            
            if level != 0:
                self.up_samples.append(Upsample(ch))
                ds //= 2
            else:
                self.up_samples.append(nn.Identity())
        
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            Swish(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
       
        t_emb = self.time_embed(timesteps)
        
        h = self.input_conv(x.type(self.dtype))
        
        hs = [h]
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for layer in blocks:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:  # AttentionBlock
                    h = layer(h)
            hs.append(h)
            h = downsample(h)
        
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:  # AttentionBlock
                h = layer(h)
        
        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in blocks:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:  # AttentionBlock
                    h = layer(h)
            h = upsample(h)
        
        h = self.output_conv(h)
        
        return h.type(x.dtype)

