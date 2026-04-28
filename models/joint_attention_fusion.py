import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Dict, List, Tuple

import numpy as np

class GradCAMExtractor:

    def __init__(self, model, target_layers: List[str]):

        self.model = model

        self.target_layers = target_layers

        self.activations = {}

        self.gradients = {}

        self.hooks = []

        for name, module in model.named_modules():

            if name in target_layers:

                def forward_hook(name):

                    def hook(module, input, output):

                        self.activations[name] = output.detach()

                    return hook

                def backward_hook(name):

                    def hook(module, grad_input, grad_output):

                        self.gradients[name] = grad_output[0].detach()

                    return hook

                h1 = module.register_forward_hook(forward_hook(name))

                h2 = module.register_full_backward_hook(backward_hook(name))

                self.hooks.extend([h1, h2])

    def extract_attention(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:

        self.activations = {}

        self.gradients = {}

        images = images.detach().requires_grad_(True)

        outputs = self.model(images)

        one_hot = torch.zeros_like(outputs)

        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        outputs.backward(gradient=one_hot, retain_graph=False)

        attention_maps = {}

        for layer_name in self.target_layers:

            if layer_name in self.activations and layer_name in self.gradients:

                acts = self.activations[layer_name]

                grads = self.gradients[layer_name]

                weights = grads.mean(dim=(2, 3), keepdim=True)

                cam = (weights * acts).sum(dim=1)

                cam = F.relu(cam)

                B, H, W = cam.shape

                cam = cam.view(B, -1)

                cam_min = cam.min(dim=1, keepdim=True)[0]

                cam_max = cam.max(dim=1, keepdim=True)[0]

                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

                cam = cam.view(B, H, W)

                attention_maps[layer_name] = cam

        return attention_maps

    def remove_hooks(self):

        for hook in self.hooks:

            hook.remove()

class ViTGradCAMExtractor:

    def __init__(self, model, target_blocks: List[int]):

        self.model = model

        self.target_blocks = target_blocks

        self.activations = {}

        self.gradients = {}

        self.hooks = []

        for idx in target_blocks:

            if idx < len(model.blocks):

                block = model.blocks[idx]

                block_name = f'block{idx}'

                def forward_hook(name):

                    def hook(module, input, output):

                        self.activations[name] = output.detach()

                    return hook

                def backward_hook(name):

                    def hook(module, grad_input, grad_output):

                        self.gradients[name] = grad_output[0].detach()

                    return hook

                h1 = block.register_forward_hook(forward_hook(block_name))

                h2 = block.register_full_backward_hook(backward_hook(block_name))

                self.hooks.extend([h1, h2])

    def extract_attention(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:

        self.activations = {}

        self.gradients = {}

        images = images.detach().requires_grad_(True)

        outputs = self.model(images)

        one_hot = torch.zeros_like(outputs)

        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        outputs.backward(gradient=one_hot, retain_graph=False)

        attention_maps = {}

        for block_name in [f'block{idx}' for idx in self.target_blocks]:

            if block_name in self.activations and block_name in self.gradients:

                acts = self.activations[block_name]

                grads = self.gradients[block_name]

                acts = acts[:, 1:, :]

                grads = grads[:, 1:, :]

                weights = grads.mean(dim=2, keepdim=True)

                cam = (weights * acts).sum(dim=2)

                cam = F.relu(cam)

                B, num_patches = cam.shape

                patch_size = int(np.sqrt(num_patches))

                if patch_size * patch_size == num_patches:

                    cam = cam.view(B, patch_size, patch_size)

                    cam_flat = cam.view(B, -1)

                    cam_min = cam_flat.min(dim=1, keepdim=True)[0]

                    cam_max = cam_flat.max(dim=1, keepdim=True)[0]

                    cam = (cam_flat - cam_min) / (cam_max - cam_min + 1e-8)

                    cam = cam.view(B, patch_size, patch_size)

                    attention_maps[block_name] = cam

        return attention_maps

    def remove_hooks(self):

        for hook in self.hooks:

            hook.remove()

class JointAttentionFusion(nn.Module):

    def __init__(

        self,

        cnn_layers: List[str],

        vit_blocks: List[int],

        target_size: Tuple[int, int] = (7, 7)

    ):

        super().__init__()

        self.cnn_layers = cnn_layers

        self.vit_blocks = vit_blocks

        self.target_size = target_size

        self.cnn_layer_weights = self._compute_layer_weights(len(cnn_layers))

        self.vit_layer_weights = self._compute_layer_weights(len(vit_blocks))

    def _compute_layer_weights(self, num_layers):

        weights = torch.arange(1, num_layers + 1, dtype=torch.float32)

        weights = weights / weights.sum()

        return weights

    def forward(

        self,

        cnn_attention_maps: Dict[str, torch.Tensor],

        vit_attention_maps: Dict[str, torch.Tensor]

    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if cnn_attention_maps:

            B = list(cnn_attention_maps.values())[0].shape[0]

            device = list(cnn_attention_maps.values())[0].device

        elif vit_attention_maps:

            B = list(vit_attention_maps.values())[0].shape[0]

            device = list(vit_attention_maps.values())[0].device

        else:

            raise ValueError("Both CNN and ViT attention maps are empty")

        A_cnn = self._fuse_within_model(cnn_attention_maps, self.cnn_layer_weights, device)

        A_vit = self._fuse_within_model(vit_attention_maps, self.vit_layer_weights, device)

        if A_cnn is None and A_vit is None:

            zero_attn = torch.zeros(B, *self.target_size, device=device)

            zero_weight = torch.zeros(B, device=device)

            return zero_attn, zero_attn, zero_attn, zero_weight, zero_weight

        if A_cnn is not None:

            A_cnn = F.interpolate(

                A_cnn.unsqueeze(1),

                size=self.target_size,

                mode='bilinear',

                align_corners=False

            ).squeeze(1)

            A_cnn = self._normalize_attention(A_cnn)

        else:

            A_cnn = torch.zeros(B, *self.target_size, device=device)

        if A_vit is not None:

            A_vit = F.interpolate(

                A_vit.unsqueeze(1),

                size=self.target_size,

                mode='bilinear',

                align_corners=False

            ).squeeze(1)

            A_vit = self._normalize_attention(A_vit)

        else:

            A_vit = torch.zeros(B, *self.target_size, device=device)

        s_cnn = A_cnn.mean(dim=(1, 2))

        s_vit = A_vit.mean(dim=(1, 2))

        w_cnn = s_cnn / (s_cnn + s_vit + 1e-8)

        w_vit = s_vit / (s_cnn + s_vit + 1e-8)

        A_joint = (

            w_cnn.view(B, 1, 1) * A_cnn +

            w_vit.view(B, 1, 1) * A_vit

        )

        return A_joint, A_cnn, A_vit, w_cnn, w_vit

    def _fuse_within_model(

        self,

        attention_maps: Dict[str, torch.Tensor],

        weights: torch.Tensor,

        device

    ) -> torch.Tensor:

        if not attention_maps:

            return None

        weights = weights.to(device)

        fused = None

        for i, (name, attn_map) in enumerate(attention_maps.items()):

            if i < len(weights):

                weighted_map = weights[i] * attn_map

                if fused is None:

                    fused = weighted_map

                else:

                    if fused.shape[-2:] != weighted_map.shape[-2:]:

                        weighted_map = F.interpolate(

                            weighted_map.unsqueeze(1),

                            size=fused.shape[-2:],

                            mode='bilinear',

                            align_corners=False

                        ).squeeze(1)

                    fused = fused + weighted_map

        return fused

    def _normalize_attention(self, attn: torch.Tensor) -> torch.Tensor:

        B = attn.shape[0]

        attn_flat = attn.view(B, -1)

        attn_min = attn_flat.min(dim=1, keepdim=True)[0]

        attn_max = attn_flat.max(dim=1, keepdim=True)[0]

        attn_norm = (attn_flat - attn_min) / (attn_max - attn_min + 1e-8)

        return attn_norm.view_as(attn)

class JointAttentionLoss(nn.Module):

    def __init__(self, scale: float = 1.0):

        super().__init__()

        self.scale = scale

    def forward(

        self,

        A_joint_adv: torch.Tensor,

        A_joint_clean: torch.Tensor

    ) -> torch.Tensor:

        loss = F.mse_loss(A_joint_adv, A_joint_clean)

        A_adv_flat = A_joint_adv.view(A_joint_adv.shape[0], -1)

        A_clean_flat = A_joint_clean.view(A_joint_clean.shape[0], -1)

        cos_sim = F.cosine_similarity(A_adv_flat, A_clean_flat, dim=1).mean()

        total_loss = loss + self.scale * (1.0 - cos_sim)

        return total_loss

def create_joint_attention_system(

    cnn_model,

    vit_model,

    cnn_layers: List[str],

    vit_blocks: List[int],

    target_size: Tuple[int, int] = (7, 7),

    loss_scale: float = 1.0

):

    gradcam_extractor = GradCAMExtractor(cnn_model, cnn_layers)

    vit_gradcam_extractor = None

    if vit_model is not None:

        vit_gradcam_extractor = ViTGradCAMExtractor(vit_model, vit_blocks)

    fusion_module = JointAttentionFusion(cnn_layers, vit_blocks, target_size)

    loss_fn = JointAttentionLoss(scale=loss_scale)

    return gradcam_extractor, vit_gradcam_extractor, fusion_module, loss_fn
