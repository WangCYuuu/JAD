import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class HierarchicalQKVExtractor(nn.Module):
    def __init__(
        self,
        cnn_model: nn.Module,
        vit_model: Optional[nn.Module] = None,
        cnn_low_layers: List[str] = ['layer2', 'layer3'],
        cnn_high_layers: List[str] = ['layer4'],
        vit_low_layers: List[int] = [6, 7, 8],
        vit_high_layers: List[int] = [9, 10, 11],
        num_heads: int = 8,
        head_dim: int = 64,
        target_size: Tuple[int, int] = (4, 4),
        cnn_normalize: Optional[Dict[str, List[float]]] = None,
        vit_normalize: Optional[Dict[str, List[float]]] = None
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim
        self.target_size = target_size

        if cnn_normalize is None:
            cnn_normalize = {
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010]
            }
        if vit_normalize is None:
            vit_normalize = {
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]
            }

        self.register_buffer('cnn_mean', torch.tensor(cnn_normalize['mean']).view(1, 3, 1, 1))
        self.register_buffer('cnn_std', torch.tensor(cnn_normalize['std']).view(1, 3, 1, 1))
        self.register_buffer('vit_mean', torch.tensor(vit_normalize['mean']).view(1, 3, 1, 1))
        self.register_buffer('vit_std', torch.tensor(vit_normalize['std']).view(1, 3, 1, 1))

        self.cnn_low_layers = cnn_low_layers
        self.cnn_high_layers = cnn_high_layers
        self.vit_low_layers = vit_low_layers if vit_model else []
        self.vit_high_layers = vit_high_layers if vit_model else []

        self.cnn_model = cnn_model
        self.cnn_activations = {}
        self.cnn_projections = nn.ModuleDict()

        self.vit_model = vit_model
        self.vit_qkv_cache = {}

        self._init_cnn_extractor()
        if vit_model:
            self._init_vit_extractor()

    def _init_cnn_extractor(self):
        all_cnn_layers = self.cnn_low_layers + self.cnn_high_layers

        def make_hook(name):
            def hook(module, input, output):
                self.cnn_activations[name] = output.detach()
            return hook

        for name, module in self.cnn_model.named_modules():
            if name in all_cnn_layers:
                module.register_forward_hook(make_hook(name))

        with torch.no_grad():
            device = next(self.cnn_model.parameters()).device
            model_class_name = self.cnn_model.__class__.__name__.lower()
            if 'swin' in model_class_name or 'vit' in model_class_name or 'vision_transformer' in model_class_name:
                dummy = torch.randn(1, 3, 224, 224).to(device)
            else:
                dummy = torch.randn(1, 3, 32, 32).to(device)

            dummy_norm = (dummy - self.cnn_mean.to(device)) / self.cnn_std.to(device)
            _ = self.cnn_model(dummy_norm)

        for layer_name in all_cnn_layers:
            if layer_name in self.cnn_activations:
                C = self.cnn_activations[layer_name].shape[1]
                self.cnn_projections[f'{layer_name}_qkv'] = nn.Linear(C, self.embed_dim * 3)
                print(f'CNN {layer_name}: {C} channels → QKV projection')

    def _init_vit_extractor(self):
        all_vit_layers = self.vit_low_layers + self.vit_high_layers

        layer_idx = 0
        for name, module in self.vit_model.named_modules():
            if name.endswith('.attn') and 'blocks' in name:
                if layer_idx in all_vit_layers:
                    self._modify_vit_attention(module, layer_idx)
                    print(f'ViT layer{layer_idx}: hooked')
                layer_idx += 1

    def _modify_vit_attention(self, attn_module, layer_idx):
        original_forward = attn_module.forward
        cache = self.vit_qkv_cache

        def new_forward(x, *args, **kwargs):
            B, N, C = x.shape

            if hasattr(attn_module, 'qkv'):
                qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                cache[f'layer{layer_idx}_q'] = q[:, :, 1:, :].detach()
                cache[f'layer{layer_idx}_k'] = k[:, :, 1:, :].detach()
                cache[f'layer{layer_idx}_v'] = v[:, :, 1:, :].detach()

            return original_forward(x, *args, **kwargs)

        attn_module.forward = new_forward

    def extract_hierarchical_qkv(
        self,
        x: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        result = {
            'low_level': {'cnn': {}, 'vit': {}},
            'high_level': {'cnn': {}, 'vit': {}}
        }

        self.cnn_activations.clear()
        x_cnn = (x - self.cnn_mean) / self.cnn_std
        with torch.no_grad():
            _ = self.cnn_model(x_cnn)

        for layer_name in self.cnn_low_layers:
            if layer_name in self.cnn_activations:
                qkv = self._cnn_activation_to_qkv(layer_name, self.cnn_activations[layer_name])
                result['low_level']['cnn'][layer_name] = qkv

        for layer_name in self.cnn_high_layers:
            if layer_name in self.cnn_activations:
                qkv = self._cnn_activation_to_qkv(layer_name, self.cnn_activations[layer_name])
                result['high_level']['cnn'][layer_name] = qkv

        if self.vit_model is not None:
            self.vit_qkv_cache.clear()
            x_vit = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x_vit = (x_vit - self.vit_mean) / self.vit_std

            with torch.no_grad():
                _ = self.vit_model(x_vit)

            for layer_idx in self.vit_low_layers:
                if f'layer{layer_idx}_q' in self.vit_qkv_cache:
                    qkv = self._vit_cache_to_qkv(layer_idx)
                    result['low_level']['vit'][f'layer{layer_idx}'] = qkv

            for layer_idx in self.vit_high_layers:
                if f'layer{layer_idx}_q' in self.vit_qkv_cache:
                    qkv = self._vit_cache_to_qkv(layer_idx)
                    result['high_level']['vit'][f'layer{layer_idx}'] = qkv

        return result

    def _cnn_activation_to_qkv(self, layer_name, activation):
        feat = activation
        B, C, H, W = feat.shape
        feat_seq = feat.flatten(2).transpose(1, 2)

        qkv = self.cnn_projections[f'{layer_name}_qkv'](feat_seq)
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self._resize_qkv(q, (H, W), self.target_size)
        k = self._resize_qkv(k, (H, W), self.target_size)
        v = self._resize_qkv(v, (H, W), self.target_size)

        return {'q': q, 'k': k, 'v': v}

    def _vit_cache_to_qkv(self, layer_idx):
        q = self.vit_qkv_cache[f'layer{layer_idx}_q']
        k = self.vit_qkv_cache[f'layer{layer_idx}_k']
        v = self.vit_qkv_cache[f'layer{layer_idx}_v']

        grid_size = int(math.sqrt(q.shape[2]))

        q = self._resize_qkv(q, (grid_size, grid_size), self.target_size)
        k = self._resize_qkv(k, (grid_size, grid_size), self.target_size)
        v = self._resize_qkv(v, (grid_size, grid_size), self.target_size)

        return {'q': q, 'k': k, 'v': v}

    def _resize_qkv(self, qkv, current_size, target_size):
        B, heads, HW, d = qkv.shape
        H, W = current_size

        if current_size == target_size:
            return qkv

        qkv = qkv.transpose(1, 2).reshape(B, HW, heads * d)
        qkv = qkv.view(B, H, W, heads * d)
        qkv = qkv.permute(0, 3, 1, 2)

        H_t, W_t = target_size
        qkv = F.interpolate(qkv, size=target_size, mode='bilinear', align_corners=False)

        qkv = qkv.view(B, heads, d, H_t * W_t)
        qkv = qkv.permute(0, 1, 3, 2)

        return qkv


class HierarchicalCrossAttentionLoss(nn.Module):
    def __init__(
        self,
        low_level_cnn_weight: float = 0.7,
        low_level_vit_weight: float = 0.3,
        high_level_cnn_weight: float = 0.3,
        high_level_vit_weight: float = 0.7,
        low_level_total_weight: float = 0.3,
        high_level_total_weight: float = 0.7,
        scale: float = 1.0
    ):
        super().__init__()

        self.low_cnn_w = low_level_cnn_weight
        self.low_vit_w = low_level_vit_weight
        self.high_cnn_w = high_level_cnn_weight
        self.high_vit_w = high_level_vit_weight

        self.low_total_w = low_level_total_weight
        self.high_total_w = high_level_total_weight

        self.scale = scale

    def forward(
        self,
        qkv_adv_hierarchical: Dict[str, Dict],
        qkv_clean_hierarchical: Dict[str, Dict]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}

        loss_low = self._compute_level_loss(
            qkv_adv_hierarchical['low_level'],
            qkv_clean_hierarchical['low_level'],
            cnn_weight=self.low_cnn_w,
            vit_weight=self.low_vit_w,
            level_name='low'
        )

        losses['low_level'] = loss_low.item() if loss_low is not None else 0.0

        loss_high = self._compute_level_loss(
            qkv_adv_hierarchical['high_level'],
            qkv_clean_hierarchical['high_level'],
            cnn_weight=self.high_cnn_w,
            vit_weight=self.high_vit_w,
            level_name='high'
        )

        losses['high_level'] = loss_high.item() if loss_high is not None else 0.0

        total_loss = None

        if loss_low is not None and loss_high is not None:
            total_loss = self.low_total_w * loss_low + self.high_total_w * loss_high
        elif loss_low is not None:
            total_loss = self.low_total_w * loss_low
        elif loss_high is not None:
            total_loss = self.high_total_w * loss_high
        else:
            total_loss = torch.tensor(0.0, requires_grad=False)

        losses['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, losses

    def _compute_level_loss(
        self,
        qkv_adv_level: Dict[str, Dict],
        qkv_clean_level: Dict[str, Dict],
        cnn_weight: float,
        vit_weight: float,
        level_name: str
    ) -> Optional[torch.Tensor]:
        total_loss = 0.0
        has_loss = False

        if 'cnn' in qkv_adv_level and len(qkv_adv_level['cnn']) > 0:
            cnn_loss = self._compute_arch_loss(
                qkv_adv_level['cnn'],
                qkv_clean_level['cnn']
            )

            if cnn_loss is not None:
                total_loss += cnn_weight * cnn_loss
                has_loss = True

        if 'vit' in qkv_adv_level and len(qkv_adv_level['vit']) > 0:
            vit_loss = self._compute_arch_loss(
                qkv_adv_level['vit'],
                qkv_clean_level['vit']
            )

            if vit_loss is not None:
                total_loss += vit_weight * vit_loss
                has_loss = True

        return total_loss if has_loss else None

    def _compute_arch_loss(
        self,
        qkv_adv: Dict[str, Dict],
        qkv_clean: Dict[str, Dict]
    ) -> Optional[torch.Tensor]:
        total_loss = 0.0
        count = 0

        for layer_name, qkv_dict_adv in qkv_adv.items():
            if layer_name not in qkv_clean:
                continue

            qkv_dict_clean = qkv_clean[layer_name]

            Q_adv = qkv_dict_adv['q']
            K_adv = qkv_dict_adv['k']
            V_adv = qkv_dict_adv['v']

            K_clean = qkv_dict_clean['k']
            V_clean = qkv_dict_clean['v']

            self_out = F.scaled_dot_product_attention(
                Q_adv * self.scale,
                K_adv,
                V_adv
            )

            cross_out = F.scaled_dot_product_attention(
                Q_adv * self.scale,
                K_clean,
                V_clean
            )

            loss = F.l1_loss(self_out, cross_out.detach())
            total_loss += loss
            count += 1

        return total_loss / count if count > 0 else None


class HierarchicalAttentionGuidedDiffusion(nn.Module):
    def __init__(
        self,
        latent_diffusion,
        hierarchical_extractor: HierarchicalQKVExtractor,
        hierarchical_loss: HierarchicalCrossAttentionLoss
    ):
        super().__init__()

        self.diffusion = latent_diffusion
        self.extractor = hierarchical_extractor
        self.hier_loss = hierarchical_loss

    def forward_train(
        self,
        clean_img: torch.Tensor,
        adv_img: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        z_clean = self.diffusion.vae.encode(clean_img)
        z_adv = self.diffusion.vae.encode(adv_img)

        B = z_clean.shape[0]
        device = z_clean.device

        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_adv)
        z_noisy = self.diffusion.q_sample(z_adv, t, noise)

        z_input = torch.cat([z_clean, z_noisy], dim=1)
        noise_pred = self.diffusion.unet(z_input, t)

        loss_diff = F.mse_loss(noise_pred, noise)

        losses = {'diffusion': loss_diff.item()}
        total_loss = 1.0 * loss_diff

        try:
            z_adv_pred = self.diffusion._predict_x0(z_noisy, t, noise_pred)
            x_adv_pred = self.diffusion.vae.decode(z_adv_pred)
            x_clean = self.diffusion.vae.decode(z_clean)

            qkv_adv_hier = self.extractor.extract_hierarchical_qkv(x_adv_pred)
            qkv_clean_hier = self.extractor.extract_hierarchical_qkv(x_clean)

            loss_hier, hier_losses = self.hier_loss(qkv_adv_hier, qkv_clean_hier)

            total_loss = total_loss + loss_hier
            losses.update(hier_losses)

        except Exception as e:
            print(f'Warning: Hierarchical attention loss failed: {e}')

        losses['total'] = total_loss.item()

        return total_loss, losses


def create_hierarchical_system(
    cnn_model,
    vit_model=None,
    config: dict = None
):
    if config is None:
        config = {
            'cnn_low_layers': ['layer2', 'layer3'],
            'cnn_high_layers': ['layer4'],
            'vit_low_layers': [6, 7, 8],
            'vit_high_layers': [9, 10, 11],
            'low_level_cnn_weight': 0.7,
            'low_level_vit_weight': 0.3,
            'high_level_cnn_weight': 0.3,
            'high_level_vit_weight': 0.7,
            'low_level_total_weight': 0.3,
            'high_level_total_weight': 0.7
        }

    extractor = HierarchicalQKVExtractor(
        cnn_model=cnn_model,
        vit_model=vit_model,
        cnn_low_layers=config.get('cnn_low_layers', ['layer2', 'layer3']),
        cnn_high_layers=config.get('cnn_high_layers', ['layer4']),
        vit_low_layers=config.get('vit_low_layers', [6, 7, 8]) if vit_model else [],
        vit_high_layers=config.get('vit_high_layers', [9, 10, 11]) if vit_model else [],
        num_heads=config.get('num_heads', 8),
        target_size=config.get('target_size', (4, 4)),
        cnn_normalize=config.get('cnn_normalize', None),
        vit_normalize=config.get('vit_normalize', None)
    )

    loss_fn = HierarchicalCrossAttentionLoss(
        low_level_cnn_weight=config.get('low_level_cnn_weight', 0.7),
        low_level_vit_weight=config.get('low_level_vit_weight', 0.3),
        high_level_cnn_weight=config.get('high_level_cnn_weight', 0.3),
        high_level_vit_weight=config.get('high_level_vit_weight', 0.7),
        low_level_total_weight=config.get('low_level_total_weight', 0.3),
        high_level_total_weight=config.get('high_level_total_weight', 0.7),
        scale=config.get('attention_scale', 1.0)
    )

    return extractor, loss_fn
