"""
Q, K, V 分离的注意力提取器
改进版：借鉴 AttentionDistillation 的设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import math


class QKVAttentionExtractor(nn.Module):
    """
    从分类模型提取 Q, K, V 三元组
    类似 AttentionDistillation，但应用于分类模型而非 UNet
    """
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
        
        # 为每个目标层创建 Q, K, V 投影
        self._create_qkv_projections()
        self._register_hooks()
    
    def _create_qkv_projections(self):
        """为每个特征层创建 Q, K, V 投影矩阵"""
        for layer_name in self.target_layers:
            # 获取层的通道数
            layer = self._get_layer_by_name(layer_name)
            if layer is None:
                continue
            
            # 推断输出通道数
            # 对于 ResNet layer4: 512 通道
            # 需要通过一次前向传播来确定
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 32, 32)
                self.model(dummy_input)  # 触发 hook
            
            if layer_name in self.activations:
                C = self.activations[layer_name].shape[1]
                
                # 创建 Q, K, V 投影
                embed_dim = self.num_heads * self.head_dim
                
                self.qkv_projections[f'{layer_name}_to_qkv'] = nn.Linear(C, embed_dim * 3)
                
                print(f'Created QKV projection for {layer_name}: {C} → {embed_dim}')
    
    def _get_layer_by_name(self, name: str):
        """根据名称获取层"""
        for n, module in self.model.named_modules():
            if n == name:
                return module
        return None
    
    def _register_hooks(self):
        """注册前向钩子"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(forward_hook(name))
    
    def extract_qkv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取 Q, K, V 三元组
        
        Args:
            x: [B, C, H, W] 输入图像
        
        Returns:
            {
                'layer4_q': [B, num_heads, HW, head_dim],
                'layer4_k': [B, num_heads, HW, head_dim],
                'layer4_v': [B, num_heads, HW, head_dim],
                'layer4_out': [B, num_heads, HW, head_dim]  # 自注意力输出
            }
        """
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
            
            # Reshape 为序列形式
            feat_seq = feat.flatten(2).transpose(1, 2)  # [B, HW, C]
            
            # 投影到 QKV
            qkv_proj = self.qkv_projections[f'{layer_name}_to_qkv']
            qkv = qkv_proj(feat_seq)  # [B, HW, 3*embed_dim]
            
            # 分离 Q, K, V
            embed_dim = self.num_heads * self.head_dim
            qkv = qkv.reshape(B, H*W, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, HW, head_dim]
            
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # 计算自注意力输出
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            self_out = torch.matmul(attn_weights, v)
            
            # 保存
            qkv_dict[f'{layer_name}_q'] = q
            qkv_dict[f'{layer_name}_k'] = k
            qkv_dict[f'{layer_name}_v'] = v
            qkv_dict[f'{layer_name}_out'] = self_out
        
        return qkv_dict


class CrossAttentionDistillationLoss(nn.Module):
    """
    交叉注意力蒸馏损失
    借鉴 AttentionDistillation 的核心思想
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(
        self,
        qkv_adv: Dict[str, torch.Tensor],
        qkv_clean: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算交叉注意力蒸馏损失
        
        Args:
            qkv_adv: 对抗样本的 Q,K,V
            qkv_clean: 干净图像的 Q,K,V
        
        Returns:
            损失值
        """
        total_loss = 0.0
        count = 0
        
        # 遍历所有层
        layer_names = set([k.rsplit('_', 1)[0] for k in qkv_adv.keys()])
        
        for layer_name in layer_names:
            # 提取对抗样本的 Q, K, V
            Q_adv = qkv_adv.get(f'{layer_name}_q')
            K_adv = qkv_adv.get(f'{layer_name}_k')
            V_adv = qkv_adv.get(f'{layer_name}_v')
            self_out_adv = qkv_adv.get(f'{layer_name}_out')
            
            # 提取干净图像的 K, V
            K_clean = qkv_clean.get(f'{layer_name}_k')
            V_clean = qkv_clean.get(f'{layer_name}_v')
            
            if any(x is None for x in [Q_adv, K_clean, V_clean, self_out_adv]):
                continue
            
            # 🔑 核心：计算交叉注意力输出
            # 使用对抗样本的 Q，查询干净图像的 K, V
            cross_out = F.scaled_dot_product_attention(
                Q_adv * self.scale,      # 对抗样本的查询
                K_clean,                 # 干净图像的键
                V_clean,                 # 干净图像的值
            )
            
            # 损失：让对抗的自注意力输出 ≈ 交叉注意力输出
            # 这样对抗样本会"表现得像"干净图像
            loss = F.l1_loss(self_out_adv, cross_out.detach())
            
            total_loss += loss
            count += 1
        
        return total_loss / max(count, 1)


class MultiScaleQKVExtractor(nn.Module):
    """
    多尺度 Q,K,V 提取器
    从多个层提取，更全面
    """
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
        """提取多尺度 Q,K,V"""
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
    """
    创建 QKV 提取器的工厂函数
    """
    from torchvision import models
    
    # 加载模型
    if model_arch == 'resnet18':
        model = models.resnet18(num_classes=10)
    elif model_arch == 'resnet50':
        model = models.resnet50(num_classes=10)
    else:
        raise ValueError(f'Unknown architecture: {model_arch}')
    
    # 加载权重
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    model.eval()
    
    # 创建提取器
    extractor = QKVAttentionExtractor(
        model=model,
        target_layers=target_layers,
        num_heads=num_heads
    )
    
    return extractor

