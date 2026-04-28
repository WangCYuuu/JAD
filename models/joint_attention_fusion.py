"""
联合注意力融合模块
实现论文中的两步注意力融合策略：
1. 层级内融合（Layer-wise fusion）
2. 跨模型动态融合（Cross-model dynamic fusion）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class GradCAMExtractor:
    """
    Grad-CAM 注意力提取器
    从 CNN 模型提取注意力图
    """
    def __init__(self, model, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
        # 注册钩子
        for name, module in model.named_modules():
            if name in target_layers:
                # 前向钩子（保存激活）
                def forward_hook(name):
                    def hook(module, input, output):
                        self.activations[name] = output.detach()
                    return hook
                
                # 反向钩子（保存梯度）
                def backward_hook(name):
                    def hook(module, grad_input, grad_output):
                        self.gradients[name] = grad_output[0].detach()
                    return hook
                
                h1 = module.register_forward_hook(forward_hook(name))
                h2 = module.register_full_backward_hook(backward_hook(name))
                self.hooks.extend([h1, h2])
    
    def extract_attention(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取 Grad-CAM 注意力图
        
        Args:
            images: [B, 3, H, W]
            labels: [B]
        
        Returns:
            {layer_name: attention_map [B, H', W']}
        """
        self.activations = {}
        self.gradients = {}
        
        # 前向传播（确保是 leaf variable）
        images = images.detach().requires_grad_(True)
        outputs = self.model(images)
        
        # 反向传播（只对目标类）
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        outputs.backward(gradient=one_hot, retain_graph=False)
        
        # 计算 Grad-CAM
        attention_maps = {}
        for layer_name in self.target_layers:
            if layer_name in self.activations and layer_name in self.gradients:
                acts = self.activations[layer_name]  # [B, C, H, W]
                grads = self.gradients[layer_name]    # [B, C, H, W]
                
                # 全局平均池化梯度
                weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
                
                # 加权求和
                cam = (weights * acts).sum(dim=1)  # [B, H, W]
                
                # ReLU + 归一化到 [0, 1]
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
    """
    ViT Grad-CAM 提取器
    使用 Grad-CAM 方法从 ViT 提取注意力图
    """
    def __init__(self, model, target_blocks: List[int]):
        self.model = model
        self.target_blocks = target_blocks
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
        # 注册钩子到 ViT blocks
        for idx in target_blocks:
            if idx < len(model.blocks):
                block = model.blocks[idx]
                block_name = f'block{idx}'
                
                # 前向钩子
                def forward_hook(name):
                    def hook(module, input, output):
                        self.activations[name] = output.detach()
                    return hook
                
                # 反向钩子
                def backward_hook(name):
                    def hook(module, grad_input, grad_output):
                        self.gradients[name] = grad_output[0].detach()
                    return hook
                
                h1 = block.register_forward_hook(forward_hook(block_name))
                h2 = block.register_full_backward_hook(backward_hook(block_name))
                self.hooks.extend([h1, h2])
    
    def extract_attention(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        使用 Grad-CAM 提取 ViT 注意力图
        
        Args:
            images: [B, 3, H, W] (已归一化)
            labels: [B]
        
        Returns:
            {block_name: attention_map [B, H', W']}
        """
        self.activations = {}
        self.gradients = {}
        
        # 前向传播（确保是 leaf variable）
        images = images.detach().requires_grad_(True)
        outputs = self.model(images)
        
        # 反向传播
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        outputs.backward(gradient=one_hot, retain_graph=False)
        
        # 计算 Grad-CAM
        attention_maps = {}
        
        for block_name in [f'block{idx}' for idx in self.target_blocks]:
            if block_name in self.activations and block_name in self.gradients:
                acts = self.activations[block_name]  # [B, N, D] where N=num_patches+1
                grads = self.gradients[block_name]    # [B, N, D]
                
                # 去掉 class token
                acts = acts[:, 1:, :]  # [B, num_patches, D]
                grads = grads[:, 1:, :]  # [B, num_patches, D]
                
                # 计算权重（平均梯度）
                weights = grads.mean(dim=2, keepdim=True)  # [B, num_patches, 1]
                
                # 加权求和
                cam = (weights * acts).sum(dim=2)  # [B, num_patches]
                
                # ReLU
                cam = F.relu(cam)
                
                # Reshape 到 2D
                B, num_patches = cam.shape
                patch_size = int(np.sqrt(num_patches))
                
                if patch_size * patch_size == num_patches:
                    cam = cam.view(B, patch_size, patch_size)
                    
                    # 归一化到 [0, 1]
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
    """
    联合注意力融合
    实现两步融合策略：
    1. 层级内融合
    2. 跨模型动态融合
    """
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
        
        # 层级内融合权重（基于深度）
        self.cnn_layer_weights = self._compute_layer_weights(len(cnn_layers))
        self.vit_layer_weights = self._compute_layer_weights(len(vit_blocks))
    
    def _compute_layer_weights(self, num_layers):
        """计算层级权重（深层权重更大）"""
        weights = torch.arange(1, num_layers + 1, dtype=torch.float32)
        weights = weights / weights.sum()
        return weights
    
    def forward(
        self,
        cnn_attention_maps: Dict[str, torch.Tensor],
        vit_attention_maps: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        两步注意力融合
        
        Args:
            cnn_attention_maps: {layer_name: [B, H, W]}
            vit_attention_maps: {block_name: [B, H', W']}
        
        Returns:
            (joint_attention, A_cnn, A_vit, w_cnn, w_vit)
        """
        # 获取 batch size 和 device
        if cnn_attention_maps:
            B = list(cnn_attention_maps.values())[0].shape[0]
            device = list(cnn_attention_maps.values())[0].device
        elif vit_attention_maps:
            B = list(vit_attention_maps.values())[0].shape[0]
            device = list(vit_attention_maps.values())[0].device
        else:
            raise ValueError("Both CNN and ViT attention maps are empty")
        
        # Step 1: 层级内融合
        A_cnn = self._fuse_within_model(cnn_attention_maps, self.cnn_layer_weights, device)
        A_vit = self._fuse_within_model(vit_attention_maps, self.vit_layer_weights, device)
        
        # 处理空的情况
        if A_cnn is None and A_vit is None:
            # 返回全零的注意力图
            zero_attn = torch.zeros(B, *self.target_size, device=device)
            zero_weight = torch.zeros(B, device=device)
            return zero_attn, zero_attn, zero_attn, zero_weight, zero_weight
        
        # 插值到统一尺寸
        if A_cnn is not None:
            A_cnn = F.interpolate(
                A_cnn.unsqueeze(1), 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)  # [B, H_t, W_t]
            A_cnn = self._normalize_attention(A_cnn)
        else:
            A_cnn = torch.zeros(B, *self.target_size, device=device)
        
        if A_vit is not None:
            A_vit = F.interpolate(
                A_vit.unsqueeze(1), 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)  # [B, H_t, W_t]
            A_vit = self._normalize_attention(A_vit)
        else:
            A_vit = torch.zeros(B, *self.target_size, device=device)
        
        # Step 2: 跨模型动态融合
        # 计算平均强度
        s_cnn = A_cnn.mean(dim=(1, 2))  # [B]
        s_vit = A_vit.mean(dim=(1, 2))  # [B]
        
        # 动态权重
        w_cnn = s_cnn / (s_cnn + s_vit + 1e-8)  # [B]
        w_vit = s_vit / (s_cnn + s_vit + 1e-8)  # [B]
        
        # 融合
        A_joint = (
            w_cnn.view(B, 1, 1) * A_cnn +
            w_vit.view(B, 1, 1) * A_vit
        )  # [B, H_t, W_t]
        
        return A_joint, A_cnn, A_vit, w_cnn, w_vit
    
    def _fuse_within_model(
        self, 
        attention_maps: Dict[str, torch.Tensor], 
        weights: torch.Tensor,
        device
    ) -> torch.Tensor:
        """层级内融合"""
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
                    # 插值到相同尺寸
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
        """归一化到 [0, 1]"""
        B = attn.shape[0]
        attn_flat = attn.view(B, -1)
        attn_min = attn_flat.min(dim=1, keepdim=True)[0]
        attn_max = attn_flat.max(dim=1, keepdim=True)[0]
        attn_norm = (attn_flat - attn_min) / (attn_max - attn_min + 1e-8)
        return attn_norm.view_as(attn)


class JointAttentionLoss(nn.Module):
    """
    基于联合注意力图的损失函数
    引导对抗样本保持关键区域的注意力一致性
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(
        self,
        A_joint_adv: torch.Tensor,
        A_joint_clean: torch.Tensor
    ) -> torch.Tensor:
        """
        计算联合注意力损失
        
        Args:
            A_joint_adv: [B, H, W] 对抗样本的联合注意力
            A_joint_clean: [B, H, W] 干净图像的联合注意力
        
        Returns:
            loss: 标量
        """
        # L2 损失
        loss = F.mse_loss(A_joint_adv, A_joint_clean)
        
        # 可选：加入 cosine similarity loss
        # 展平
        A_adv_flat = A_joint_adv.view(A_joint_adv.shape[0], -1)
        A_clean_flat = A_joint_clean.view(A_joint_clean.shape[0], -1)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(A_adv_flat, A_clean_flat, dim=1).mean()
        
        # 总损失
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
    """
    创建联合注意力系统
    
    Args:
        cnn_model: CNN 模型（用于 Grad-CAM）
        vit_model: ViT 模型（用于 Grad-CAM）
        cnn_layers: CNN 层名称列表
        vit_blocks: ViT block 索引列表
        target_size: 注意力图目标尺寸
        loss_scale: 损失缩放因子
    
    Returns:
        (gradcam_extractor, vit_gradcam_extractor, fusion_module, loss_fn)
    """
    # 创建 CNN Grad-CAM 提取器
    gradcam_extractor = GradCAMExtractor(cnn_model, cnn_layers)
    
    # 创建 ViT Grad-CAM 提取器
    vit_gradcam_extractor = None
    if vit_model is not None:
        vit_gradcam_extractor = ViTGradCAMExtractor(vit_model, vit_blocks)
    
    # 创建融合模块
    fusion_module = JointAttentionFusion(cnn_layers, vit_blocks, target_size)
    
    # 创建损失函数
    loss_fn = JointAttentionLoss(scale=loss_scale)
    
    return gradcam_extractor, vit_gradcam_extractor, fusion_module, loss_fn

