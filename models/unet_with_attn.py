import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

class UNet2DWithAttention(UNet2DModel):
    def __init__(
        self,
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D",) * 3,
        up_block_types=("UpBlock2D",) * 3,
        start_layer=2,
        end_layer=4,
        extract_mode="attn_probs",
        attn_res=None,  
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
        

        self.attn_adapter = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),  
            nn.Sigmoid()
        )
        
        self._attn_outputs = []
        self._return_attn = False

        self._register_attn_hooks()

    def _safe_reshape(self, attn):
        if attn.dim() == 4:
            return attn
        elif attn.dim() == 3:
            B, N, D = attn.shape
            H = W = int(N**0.5)
            return attn.permute(0, 2, 1).view(B, D, H, W)
        else:
            raise ValueError(f"Unsupported attention dimension: {attn.dim()}")

    def _register_attn_hooks(self):
        def hook_fn(module, input, output):
            if self._return_attn:
                if isinstance(output, (tuple, list)):
                    attn = output[1] if len(output) >= 2 else output[0]
                elif hasattr(output, 'attentions'):
                    attn = output.attentions
                else:
                    attn = output
                
                if isinstance(attn, (list, tuple)) and len(attn) > 0:
                    attn = attn[-1]
                
                if not isinstance(attn, torch.Tensor):
                    return
                
                if attn.dim() not in [3, 4]:
                    return
                
                self._attn_outputs.append(attn.detach().clone())

        for name, module in self.named_modules():
            if "Attention" in module.__class__.__name__:
                module.register_forward_hook(hook_fn)

    def forward(self, sample, timesteps, return_attn=False, **kwargs):
        self._return_attn = return_attn
        self._attn_outputs = []
        
        out = super().forward(sample, timesteps, **kwargs)
        
        if return_attn:
            adjusted_attn = []
            for attn in self._attn_outputs:
                attn_reshaped = self._safe_reshape(attn)
                assert attn_reshaped.size(1) == 512, f"Invalid channel dimension: {attn_reshaped.shape}"
                
                attn_resized = F.interpolate(attn_reshaped, (224, 224), mode='bilinear')
                attn_out = self.attn_adapter(attn_resized)
                adjusted_attn.append(attn_out)
            return out, adjusted_attn
        
        return out