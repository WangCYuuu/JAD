import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import math


class QKVAttentionExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str] = ['layer4'],
        num_heads: int = 8,
        head_dim: int = 64
    ):
        super().__init__()
        self.model = model
        self.target_layers = target_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.activations = {}
        self.qkv_projections = nn.ModuleDict()
        
        self._create_qkv_projections()
        self._register_hooks()
    
    def _create_qkv_projections(self):
        for layer_name in self.target_layers:
            layer = self._get_layer_by_name(layer_name)
            if layer is None:
                continue
            
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 32, 32)
                self.model(dummy_input)  
            
            if layer_name in self.activations:
                C = self.activations[layer_name].shape[1]
                
                embed_dim = self.num_heads * self.head_dim
                
                self.qkv_projections[f'{layer_name}_to_qkv'] = nn.Linear(C, embed_dim * 3)
                
                print(f'Created QKV projection for {layer_name}: {C} → {embed_dim}')
    
    def _get_layer_by_name(self, name: str):
        for n, module in self.model.named_modules():
            if n == name:
                return module
        return None
    
    def _register_hooks(self):
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(forward_hook(name))
    
    def extract_qkv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.activations.clear()
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(x)
        
        qkv_dict = {}
        
        for layer_name in self.target_layers:
            if layer_name not in self.activations:
                continue
            
            feat = self.activations[layer_name]  # [B, C, H, W]
            B, C, H, W = feat.shape
            

            feat_seq = feat.flatten(2).transpose(1, 2)  # [B, HW, C]
            

            qkv_proj = self.qkv_projections[f'{layer_name}_to_qkv']
            qkv = qkv_proj(feat_seq)  # [B, HW, 3*embed_dim]
            

            embed_dim = self.num_heads * self.head_dim
            qkv = qkv.reshape(B, H*W, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, HW, head_dim]
            
            q, k, v = qkv[0], qkv[1], qkv[2]
            
 
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            self_out = torch.matmul(attn_weights, v)
            

            qkv_dict[f'{layer_name}_q'] = q
            qkv_dict[f'{layer_name}_k'] = k
            qkv_dict[f'{layer_name}_v'] = v
            qkv_dict[f'{layer_name}_out'] = self_out
        
        return qkv_dict


class CrossAttentionDistillationLoss(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(
        self,
        qkv_adv: Dict[str, torch.Tensor],
        qkv_clean: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        total_loss = 0.0
        count = 0
        
        layer_names = set([k.rsplit('_', 1)[0] for k in qkv_adv.keys()])
        
        for layer_name in layer_names:
            Q_adv = qkv_adv.get(f'{layer_name}_q')
            K_adv = qkv_adv.get(f'{layer_name}_k')
            V_adv = qkv_adv.get(f'{layer_name}_v')
            self_out_adv = qkv_adv.get(f'{layer_name}_out')
            
            K_clean = qkv_clean.get(f'{layer_name}_k')
            V_clean = qkv_clean.get(f'{layer_name}_v')
            
            if any(x is None for x in [Q_adv, K_clean, V_clean, self_out_adv]):
                continue
            
            cross_out = F.scaled_dot_product_attention(
                Q_adv * self.scale,      
                K_clean,                
                V_clean,                 
            )
            

            loss = F.l1_loss(self_out_adv, cross_out.detach())
            
            total_loss += loss
            count += 1
        
        return total_loss / max(count, 1)


class MultiScaleQKVExtractor(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str] = ['layer2', 'layer3', 'layer4'],
        num_heads: int = 8
    ):
        super().__init__()
        self.extractors = nn.ModuleDict()
        
        for layer in target_layers:
            self.extractors[layer] = QKVAttentionExtractor(
                model,
                target_layers=[layer],
                num_heads=num_heads
            )
    
    def extract_qkv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_qkv = {}
        
        for layer_name, extractor in self.extractors.items():
            qkv = extractor.extract_qkv(x)
            all_qkv.update(qkv)
        
        return all_qkv


def create_qkv_extractor(
    model_arch: str = 'resnet50',
    weights_path: str = None,
    target_layers: List[str] = ['layer4'],
    num_heads: int = 8
):
    from torchvision import models
    

    if model_arch == 'resnet18':
        model = models.resnet18(num_classes=10)
    elif model_arch == 'resnet50':
        model = models.resnet50(num_classes=10)
    else:
        raise ValueError(f'Unknown architecture: {model_arch}')
    

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    model.eval()

    extractor = QKVAttentionExtractor(
        model=model,
        target_layers=target_layers,
        num_heads=num_heads
    )
    
    return extractor

