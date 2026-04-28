import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class CNNToQKVProjector(nn.Module):
    def __init__(
        self,
        cnn_model: nn.Module,
        target_layers: List[str] = ['layer4'],
        num_heads: int = 8,
        head_dim: int = 64
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.target_layers = target_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim

        self.activations = {}
        self.projections = nn.ModuleDict()

        self._create_projections()
        self._register_hooks()

    def _create_projections(self):
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32)
            self.cnn_model(dummy)

        for layer_name in self.target_layers:
            if layer_name in self.activations:
                C = self.activations[layer_name].shape[1]
                self.projections[f'{layer_name}_qkv'] = nn.Linear(C, self.embed_dim * 3)
                print(f'CNN layer {layer_name}: {C} → QKV({self.embed_dim}×3)')

    def _register_hooks(self):
        def hook(name):
            def fn(module, input, output):
                self.activations[name] = output.detach()
            return fn

        for name, module in self.cnn_model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(hook(name))

    def extract_qkv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.activations.clear()

        with torch.no_grad():
            _ = self.cnn_model(x)

        qkv_dict = {}

        for layer_name in self.target_layers:
            if layer_name not in self.activations:
                continue

            feat = self.activations[layer_name]
            B, C, H, W = feat.shape

            feat_seq = feat.flatten(2).transpose(1, 2)
            qkv = self.projections[f'{layer_name}_qkv'](feat_seq)
            qkv = qkv.reshape(B, H * W, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)

            q, k, v = qkv[0], qkv[1], qkv[2]

            qkv_dict[f'cnn_{layer_name}_q'] = q
            qkv_dict[f'cnn_{layer_name}_k'] = k
            qkv_dict[f'cnn_{layer_name}_v'] = v

        return qkv_dict


class ViTToQKVExtractor(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        target_layers: List[int] = [8, 9, 10, 11],
        resize_to: Tuple[int, int] = (32, 32)
    ):
        super().__init__()
        self.vit_model = vit_model
        self.target_layers = target_layers
        self.resize_to = resize_to

        self.qkv_cache = {}
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        layer_idx = 0

        for name, module in self.vit_model.named_modules():
            if 'attn' in name.lower() and 'drop' not in name.lower():
                if layer_idx in self.target_layers:
                    self._modify_attention_forward(module, layer_idx)
                layer_idx += 1

    def _modify_attention_forward(self, attn_module, layer_idx):
        original_forward = attn_module.forward
        cache = self.qkv_cache

        def new_forward(x):
            B, N, C = x.shape

            if hasattr(attn_module, 'qkv'):
                qkv = attn_module.qkv(x)
                qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                cache[f'vit_layer{layer_idx}_q'] = q[:, :, 1:, :].detach()
                cache[f'vit_layer{layer_idx}_k'] = k[:, :, 1:, :].detach()
                cache[f'vit_layer{layer_idx}_v'] = v[:, :, 1:, :].detach()

            return original_forward(x)

        attn_module.forward = new_forward

    def extract_qkv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.qkv_cache.clear()

        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        with torch.no_grad():
            _ = self.vit_model(x)

        processed_qkv = {}
        for key, value in self.qkv_cache.items():
            B, heads, N, d = value.shape
            grid_size = int(math.sqrt(N))

            value_spatial = value.transpose(1, 2).reshape(B * N, heads * d)
            value_spatial = value_spatial.view(B, grid_size, grid_size, heads * d)
            value_spatial = value_spatial.permute(0, 3, 1, 2)

            target_h, target_w = self.resize_to
            value_resized = F.interpolate(
                value_spatial,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )

            value_resized = value_resized.view(B, heads, d, target_h * target_w)
            value_resized = value_resized.permute(0, 1, 3, 2)

            processed_qkv[key] = value_resized

        return processed_qkv


class CrossArchitectureFusion(nn.Module):
    def __init__(
        self,
        cnn_extractor: CNNToQKVProjector,
        vit_extractor: Optional[ViTToQKVExtractor] = None,
        fusion_strategy: str = 'weighted_average',
        target_size: Tuple[int, int] = (4, 4),
        cnn_weight: float = 0.6,
        vit_weight: float = 0.4
    ):
        super().__init__()
        self.cnn_extractor = cnn_extractor
        self.vit_extractor = vit_extractor
        self.fusion_strategy = fusion_strategy
        self.target_size = target_size
        self.cnn_weight = cnn_weight
        self.vit_weight = vit_weight

        if fusion_strategy == 'gated':
            embed_dim = cnn_extractor.embed_dim
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )

    def extract_qkv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_qkv = {}

        cnn_qkv = self.cnn_extractor.extract_qkv(x)
        all_qkv.update(cnn_qkv)

        if self.vit_extractor is not None:
            vit_qkv = self.vit_extractor.extract_qkv(x)
            all_qkv.update(vit_qkv)

        if self.vit_extractor is not None:
            fused_qkv = self._fuse_qkv(cnn_qkv, vit_qkv)
            all_qkv.update(fused_qkv)

        return all_qkv

    def _fuse_qkv(
        self,
        cnn_qkv: Dict[str, torch.Tensor],
        vit_qkv: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        fused = {}

        cnn_keys = [k for k in cnn_qkv.keys() if 'cnn_' in k]
        vit_keys = [k for k in vit_qkv.keys() if 'vit_' in k]

        if len(cnn_keys) == 0 or len(vit_keys) == 0:
            return fused

        cnn_q_key = [k for k in cnn_keys if k.endswith('_q')][0]
        cnn_k_key = [k for k in cnn_keys if k.endswith('_k')][0]
        cnn_v_key = [k for k in cnn_keys if k.endswith('_v')][0]

        vit_q_key = [k for k in vit_keys if k.endswith('_q')][-1]
        vit_k_key = [k for k in vit_keys if k.endswith('_k')][-1]
        vit_v_key = [k for k in vit_keys if k.endswith('_v')][-1]

        cnn_q = cnn_qkv[cnn_q_key]
        cnn_k = cnn_qkv[cnn_k_key]
        cnn_v = cnn_qkv[cnn_v_key]

        vit_q = vit_qkv[vit_q_key]
        vit_k = vit_qkv[vit_k_key]
        vit_v = vit_qkv[vit_v_key]

        cnn_q = self._resize_to_target(cnn_q)
        cnn_k = self._resize_to_target(cnn_k)
        cnn_v = self._resize_to_target(cnn_v)

        vit_q = self._resize_to_target(vit_q)
        vit_k = self._resize_to_target(vit_k)
        vit_v = self._resize_to_target(vit_v)

        if self.fusion_strategy == 'weighted_average':
            fused['fused_q'] = self.cnn_weight * cnn_q + self.vit_weight * vit_q
            fused['fused_k'] = self.cnn_weight * cnn_k + self.vit_weight * vit_k
            fused['fused_v'] = self.cnn_weight * cnn_v + self.vit_weight * vit_v

        elif self.fusion_strategy == 'concat':
            fused['fused_q'] = torch.cat([cnn_q, vit_q], dim=1)
            fused['fused_k'] = torch.cat([cnn_k, vit_k], dim=1)
            fused['fused_v'] = torch.cat([cnn_v, vit_v], dim=1)

        elif self.fusion_strategy == 'gated':
            B, heads, HW, d = cnn_q.shape

            cnn_flat = cnn_q.mean(dim=(1, 2))
            vit_flat = vit_q.mean(dim=(1, 2))

            gate = self.gate(torch.cat([cnn_flat, vit_flat], dim=-1))
            gate = gate.view(B, 1, 1, d)

            fused['fused_q'] = gate * cnn_q + (1 - gate) * vit_q
            fused['fused_k'] = gate * cnn_k + (1 - gate) * vit_k
            fused['fused_v'] = gate * cnn_v + (1 - gate) * vit_v

        return fused

    def _resize_to_target(self, qkv: torch.Tensor) -> torch.Tensor:
        B, heads, HW, d = qkv.shape
        current_size = int(math.sqrt(HW))

        if current_size == self.target_size[0]:
            return qkv

        qkv_spatial = qkv.transpose(1, 2).reshape(B, HW, heads * d)
        qkv_spatial = qkv_spatial.view(B, current_size, current_size, heads * d)
        qkv_spatial = qkv_spatial.permute(0, 3, 1, 2)

        qkv_resized = F.interpolate(
            qkv_spatial,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )

        target_h, target_w = self.target_size
        qkv_resized = qkv_resized.view(B, heads, d, target_h * target_w)
        qkv_resized = qkv_resized.permute(0, 1, 3, 2)

        return qkv_resized


class MultiArchitectureCrossAttentionLoss(nn.Module):
    def __init__(
        self,
        fusion_mode: str = 'separate',
        cnn_weight: float = 0.6,
        vit_weight: float = 0.4,
        scale: float = 1.0
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.cnn_weight = cnn_weight
        self.vit_weight = vit_weight
        self.scale = scale

    def forward(
        self,
        qkv_adv: Dict[str, torch.Tensor],
        qkv_clean: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.fusion_mode == 'separate':
            return self._separate_loss(qkv_adv, qkv_clean)
        elif self.fusion_mode == 'fused':
            return self._fused_loss(qkv_adv, qkv_clean)
        elif self.fusion_mode == 'hierarchical':
            return self._hierarchical_loss(qkv_adv, qkv_clean)

    def _separate_loss(self, qkv_adv, qkv_clean):
        total_loss = 0.0
        count = 0

        cnn_loss = self._compute_cross_attention_for_type(qkv_adv, qkv_clean, 'cnn')
        if cnn_loss is not None:
            total_loss += self.cnn_weight * cnn_loss
            count += 1

        vit_loss = self._compute_cross_attention_for_type(qkv_adv, qkv_clean, 'vit')
        if vit_loss is not None:
            total_loss += self.vit_weight * vit_loss
            count += 1

        return total_loss / max(count, 1)

    def _fused_loss(self, qkv_adv, qkv_clean):
        K_clean_fused, V_clean_fused = self._fuse_kv(qkv_clean)
        Q_adv_fused, K_adv_fused, V_adv_fused = self._fuse_qkv(qkv_adv)

        self_out = F.scaled_dot_product_attention(
            Q_adv_fused, K_adv_fused, V_adv_fused,
            scale=1 / math.sqrt(Q_adv_fused.size(-1))
        )

        cross_out = F.scaled_dot_product_attention(
            Q_adv_fused, K_clean_fused, V_clean_fused,
            scale=1 / math.sqrt(Q_adv_fused.size(-1))
        )

        loss = F.l1_loss(self_out, cross_out.detach())

        return loss

    def _hierarchical_loss(self, qkv_adv, qkv_clean):
        total_loss = 0.0

        low_level_loss = self._compute_layer_loss(
            qkv_adv, qkv_clean,
            cnn_layers=['layer3'],
            vit_layers=[8, 9],
            cnn_w=0.7, vit_w=0.3
        )

        high_level_loss = self._compute_layer_loss(
            qkv_adv, qkv_clean,
            cnn_layers=['layer4'],
            vit_layers=[10, 11],
            cnn_w=0.4, vit_w=0.6
        )

        total_loss = 0.3 * low_level_loss + 0.7 * high_level_loss

        return total_loss

    def _compute_cross_attention_for_type(self, qkv_adv, qkv_clean, arch_type):
        loss = 0.0
        count = 0

        adv_keys = [k for k in qkv_adv.keys() if arch_type in k]

        for key_q in adv_keys:
            if not key_q.endswith('_q'):
                continue

            base_key = key_q.rsplit('_', 1)[0]
            key_k = f'{base_key}_k'
            key_v = f'{base_key}_v'

            if key_k not in qkv_adv or key_v not in qkv_adv:
                continue
            if key_k not in qkv_clean or key_v not in qkv_clean:
                continue

            Q_adv = qkv_adv[key_q]
            K_adv = qkv_adv[key_k]
            V_adv = qkv_adv[key_v]

            K_clean = qkv_clean[key_k]
            V_clean = qkv_clean[key_v]

            self_out = F.scaled_dot_product_attention(
                Q_adv * self.scale, K_adv, V_adv
            )

            cross_out = F.scaled_dot_product_attention(
                Q_adv * self.scale, K_clean, V_clean
            )

            loss += F.l1_loss(self_out, cross_out.detach())
            count += 1

        return loss / max(count, 1) if count > 0 else None

    def _fuse_kv(self, qkv_dict):
        cnn_keys = [k for k in qkv_dict.keys() if 'cnn' in k and ('_k' in k or '_v' in k)]
        vit_keys = [k for k in qkv_dict.keys() if 'vit' in k and ('_k' in k or '_v' in k)]

        if len(cnn_keys) == 0:
            vit_k = [v for k, v in qkv_dict.items() if 'vit' in k and '_k' in k][0]
            vit_v = [v for k, v in qkv_dict.items() if 'vit' in k and '_v' in k][0]
            return vit_k, vit_v

        if len(vit_keys) == 0:
            cnn_k = [v for k, v in qkv_dict.items() if 'cnn' in k and '_k' in k][0]
            cnn_v = [v for k, v in qkv_dict.items() if 'cnn' in k and '_v' in k][0]
            return cnn_k, cnn_v

        cnn_k = [v for k, v in qkv_dict.items() if 'cnn' in k and '_k' in k][0]
        cnn_v = [v for k, v in qkv_dict.items() if 'cnn' in k and '_v' in k][0]
        vit_k = [v for k, v in qkv_dict.items() if 'vit' in k and '_k' in k][-1]
        vit_v = [v for k, v in qkv_dict.items() if 'vit' in k and '_v' in k][-1]

        cnn_k = self._resize_to_target(cnn_k)
        cnn_v = self._resize_to_target(cnn_v)
        vit_k = self._resize_to_target(vit_k)
        vit_v = self._resize_to_target(vit_v)

        K_fused = self.cnn_weight * cnn_k + self.vit_weight * vit_k
        V_fused = self.cnn_weight * cnn_v + self.vit_weight * vit_v

        return K_fused, V_fused

    def _fuse_qkv(self, qkv_dict):
        K_fused, V_fused = self._fuse_kv(qkv_dict)

        cnn_q_keys = [k for k, v in qkv_dict.items() if 'cnn' in k and '_q' in k]
        vit_q_keys = [k for k, v in qkv_dict.items() if 'vit' in k and '_q' in k]

        if len(cnn_q_keys) > 0 and len(vit_q_keys) > 0:
            cnn_q = qkv_dict[cnn_q_keys[0]]
            vit_q = qkv_dict[vit_q_keys[-1]]

            cnn_q = self._resize_to_target(cnn_q)
            vit_q = self._resize_to_target(vit_q)

            Q_fused = self.cnn_weight * cnn_q + self.vit_weight * vit_q
        elif len(cnn_q_keys) > 0:
            Q_fused = qkv_dict[cnn_q_keys[0]]
        else:
            Q_fused = qkv_dict[vit_q_keys[-1]]

        return Q_fused, K_fused, V_fused

    def _compute_layer_loss(self, qkv_adv, qkv_clean, cnn_layers, vit_layers, cnn_w, vit_w):
        return 0.0


def create_cross_architecture_fusion(
    cnn_model,
    vit_model=None,
    cnn_layers=['layer4'],
    vit_layers=[11],
    fusion_strategy='weighted_average',
    cnn_weight=0.6,
    vit_weight=0.4
):
    cnn_extractor = CNNToQKVProjector(
        cnn_model, target_layers=cnn_layers
    )

    vit_extractor = None
    if vit_model is not None:
        vit_extractor = ViTToQKVExtractor(
            vit_model, target_layers=vit_layers
        )

    fusion = CrossArchitectureFusion(
        cnn_extractor=cnn_extractor,
        vit_extractor=vit_extractor,
        fusion_strategy=fusion_strategy,
        cnn_weight=cnn_weight,
        vit_weight=vit_weight
    )

    return fusion
