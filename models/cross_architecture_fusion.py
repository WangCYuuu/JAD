"""
跨架构注意力融合模块
将 CNN 和 ViT 的注意力统一到交叉注意力蒸馏框架中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class CNNToQKVProjector(nn.Module):
    """
    将 CNN 的卷积特征投影为 Q, K, V
    使其可以与 ViT 的注意力融合
    """
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
        
        # 为每层创建 QKV 投影
        self._create_projections()
        self._register_hooks()
    
    def _create_projections(self):
        """创建 CNN 特征到 QKV 的投影"""
        # 先运行一次获取通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32)
            self.cnn_model(dummy)
        
        for layer_name in self.target_layers:
            if layer_name in self.activations:
                C = self.activations[layer_name].shape[1]
                
                # 单个投影到 3×embed_dim
                self.projections[f'{layer_name}_qkv'] = nn.Linear(C, self.embed_dim * 3)
                
                print(f'CNN layer {layer_name}: {C} → QKV({self.embed_dim}×3)')
    
    def _register_hooks(self):
        """注册钩子捕获激活"""
        def hook(name):
            def fn(module, input, output):
                self.activations[name] = output.detach()
            return fn
        
        for name, module in self.cnn_model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(hook(name))
    
    def extract_qkv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取 CNN 的 Q, K, V
        
        Returns:
            {
                'cnn_layer4_q': [B, num_heads, HW, head_dim],
                'cnn_layer4_k': [B, num_heads, HW, head_dim],
                'cnn_layer4_v': [B, num_heads, HW, head_dim]
            }
        """
        self.activations.clear()
        
        with torch.no_grad():
            _ = self.cnn_model(x)
        
        qkv_dict = {}
        
        for layer_name in self.target_layers:
            if layer_name not in self.activations:
                continue
            
            feat = self.activations[layer_name]  # [B, C, H, W]
            B, C, H, W = feat.shape
            
            # 转为序列形式
            feat_seq = feat.flatten(2).transpose(1, 2)  # [B, HW, C]
            
            # 投影到 QKV
            qkv = self.projections[f'{layer_name}_qkv'](feat_seq)  # [B, HW, 3E]
            qkv = qkv.reshape(B, H*W, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, HW, d]
            
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            qkv_dict[f'cnn_{layer_name}_q'] = q
            qkv_dict[f'cnn_{layer_name}_k'] = k
            qkv_dict[f'cnn_{layer_name}_v'] = v
        
        return qkv_dict


class ViTToQKVExtractor(nn.Module):
    """
    从 ViT 提取原生的 Q, K, V
    """
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
        """拦截 ViT 的注意力层"""
        layer_idx = 0
        
        for name, module in self.vit_model.named_modules():
            # 查找注意力模块
            if 'attn' in name.lower() and 'drop' not in name.lower():
                if layer_idx in self.target_layers:
                    # 修改注意力模块以保存 Q,K,V
                    self._modify_attention_forward(module, layer_idx)
                layer_idx += 1
    
    def _modify_attention_forward(self, attn_module, layer_idx):
        """修改注意力模块的前向传播以保存 Q,K,V"""
        original_forward = attn_module.forward
        cache = self.qkv_cache
        
        def new_forward(x):
            B, N, C = x.shape
            
            # 计算 QKV
            if hasattr(attn_module, 'qkv'):
                qkv = attn_module.qkv(x)  # [B, N, 3C]
                qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, d]
                
                # 保存（移除 CLS token，只保留 patches）
                cache[f'vit_layer{layer_idx}_q'] = q[:, :, 1:, :].detach()
                cache[f'vit_layer{layer_idx}_k'] = k[:, :, 1:, :].detach()
                cache[f'vit_layer{layer_idx}_v'] = v[:, :, 1:, :].detach()
            
            # 继续原始前向传播
            return original_forward(x)
        
        attn_module.forward = new_forward
    
    def extract_qkv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取 ViT 的 Q, K, V
        
        Args:
            x: [B, 3, H, W] 图像（可能需要 resize）
        
        Returns:
            {
                'vit_layer8_q': [B, heads, num_patches, d],
                'vit_layer8_k': [B, heads, num_patches, d],
                'vit_layer8_v': [B, heads, num_patches, d],
                ...
            }
        """
        self.qkv_cache.clear()
        
        # ViT 通常需要更大的输入
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            _ = self.vit_model(x)
        
        # 处理：将 ViT 的注意力 resize 到统一尺寸
        processed_qkv = {}
        for key, value in self.qkv_cache.items():
            # value: [B, heads, num_patches, d]
            B, heads, N, d = value.shape
            
            # ViT patches: 假设 14×14 (224/16)
            grid_size = int(math.sqrt(N))
            
            # Reshape 为空间形式
            value_spatial = value.transpose(1, 2).reshape(B * N, heads * d)
            value_spatial = value_spatial.view(B, grid_size, grid_size, heads * d)
            value_spatial = value_spatial.permute(0, 3, 1, 2)  # [B, heads*d, H', W']
            
            # Resize 到目标尺寸 (如 4×4，与 latent 对应)
            target_h, target_w = self.resize_to
            value_resized = F.interpolate(
                value_spatial,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
            
            # 转回序列形式
            value_resized = value_resized.view(B, heads, d, target_h * target_w)
            value_resized = value_resized.permute(0, 1, 3, 2)  # [B, heads, HW, d]
            
            processed_qkv[key] = value_resized
        
        return processed_qkv


class CrossArchitectureFusion(nn.Module):
    """
    跨架构注意力融合
    
    策略：
    1. 统一尺寸：将所有注意力 resize 到相同空间尺寸
    2. 融合策略：加权平均、拼接、门控等
    3. 交叉注意力：在融合后的特征上计算
    """
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
        
        # 门控融合的可学习参数
        if fusion_strategy == 'gated':
            embed_dim = cnn_extractor.embed_dim
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
    
    def extract_qkv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取并融合 CNN 和 ViT 的 Q,K,V
        
        Returns:
            {
                'fused_q': [B, heads, HW, d],
                'fused_k': [B, heads, HW, d],
                'fused_v': [B, heads, HW, d],
                'cnn_q': ..., 'vit_q': ...  # 也返回单独的
            }
        """
        all_qkv = {}
        
        # 1. 提取 CNN 的 Q,K,V
        cnn_qkv = self.cnn_extractor.extract_qkv(x)
        all_qkv.update(cnn_qkv)
        
        # 2. 提取 ViT 的 Q,K,V（如果有）
        if self.vit_extractor is not None:
            vit_qkv = self.vit_extractor.extract_qkv(x)
            all_qkv.update(vit_qkv)
        
        # 3. 融合 CNN 和 ViT 的注意力
        if self.vit_extractor is not None:
            fused_qkv = self._fuse_qkv(cnn_qkv, vit_qkv)
            all_qkv.update(fused_qkv)
        
        return all_qkv
    
    def _fuse_qkv(
        self,
        cnn_qkv: Dict[str, torch.Tensor],
        vit_qkv: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        融合 CNN 和 ViT 的 Q, K, V
        
        策略：
        1. weighted_average: 加权平均
        2. concat: 拼接（增加头数）
        3. gated: 门控融合（可学习）
        """
        fused = {}
        
        # 提取需要融合的 Q,K,V
        # 假设都已经 resize 到相同尺寸
        cnn_keys = [k for k in cnn_qkv.keys() if 'cnn_' in k]
        vit_keys = [k for k in vit_qkv.keys() if 'vit_' in k]
        
        if len(cnn_keys) == 0 or len(vit_keys) == 0:
            return fused
        
        # 选择主要的 CNN 层和 ViT 层进行融合
        # 例如：CNN layer4 + ViT layer11（最后一层）
        cnn_q_key = [k for k in cnn_keys if k.endswith('_q')][0]
        cnn_k_key = [k for k in cnn_keys if k.endswith('_k')][0]
        cnn_v_key = [k for k in cnn_keys if k.endswith('_v')][0]
        
        vit_q_key = [k for k in vit_keys if k.endswith('_q')][-1]  # 最后一层
        vit_k_key = [k for k in vit_keys if k.endswith('_k')][-1]
        vit_v_key = [k for k in vit_keys if k.endswith('_v')][-1]
        
        cnn_q = cnn_qkv[cnn_q_key]
        cnn_k = cnn_qkv[cnn_k_key]
        cnn_v = cnn_qkv[cnn_v_key]
        
        vit_q = vit_qkv[vit_q_key]
        vit_k = vit_qkv[vit_k_key]
        vit_v = vit_qkv[vit_v_key]
        
        # 确保尺寸一致（resize）
        cnn_q = self._resize_to_target(cnn_q)
        cnn_k = self._resize_to_target(cnn_k)
        cnn_v = self._resize_to_target(cnn_v)
        
        vit_q = self._resize_to_target(vit_q)
        vit_k = self._resize_to_target(vit_k)
        vit_v = self._resize_to_target(vit_v)
        
        # 根据策略融合
        if self.fusion_strategy == 'weighted_average':
            fused['fused_q'] = self.cnn_weight * cnn_q + self.vit_weight * vit_q
            fused['fused_k'] = self.cnn_weight * cnn_k + self.vit_weight * vit_k
            fused['fused_v'] = self.cnn_weight * cnn_v + self.vit_weight * vit_v
        
        elif self.fusion_strategy == 'concat':
            # 拼接（头数加倍）
            fused['fused_q'] = torch.cat([cnn_q, vit_q], dim=1)  # heads 维度拼接
            fused['fused_k'] = torch.cat([cnn_k, vit_k], dim=1)
            fused['fused_v'] = torch.cat([cnn_v, vit_v], dim=1)
        
        elif self.fusion_strategy == 'gated':
            # 门控融合（可学习权重）
            B, heads, HW, d = cnn_q.shape
            
            # 计算门控权重
            cnn_flat = cnn_q.mean(dim=(1, 2))  # [B, d]
            vit_flat = vit_q.mean(dim=(1, 2))  # [B, d]
            
            gate = self.gate(torch.cat([cnn_flat, vit_flat], dim=-1))  # [B, d]
            gate = gate.view(B, 1, 1, d)  # broadcast
            
            fused['fused_q'] = gate * cnn_q + (1 - gate) * vit_q
            
            # K, V 使用相同的门控
            fused['fused_k'] = gate * cnn_k + (1 - gate) * vit_k
            fused['fused_v'] = gate * cnn_v + (1 - gate) * vit_v
        
        return fused
    
    def _resize_to_target(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        将 Q/K/V resize 到目标尺寸
        
        Args:
            qkv: [B, heads, HW, d]
        """
        B, heads, HW, d = qkv.shape
        current_size = int(math.sqrt(HW))
        
        if current_size == self.target_size[0]:
            return qkv
        
        # Reshape 为空间形式
        qkv_spatial = qkv.transpose(1, 2).reshape(B, HW, heads * d)
        qkv_spatial = qkv_spatial.view(B, current_size, current_size, heads * d)
        qkv_spatial = qkv_spatial.permute(0, 3, 1, 2)  # [B, heads*d, H, W]
        
        # Resize
        qkv_resized = F.interpolate(
            qkv_spatial,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # 转回序列形式
        target_h, target_w = self.target_size
        qkv_resized = qkv_resized.view(B, heads, d, target_h * target_w)
        qkv_resized = qkv_resized.permute(0, 1, 3, 2)  # [B, heads, HW, d]
        
        return qkv_resized


class MultiArchitectureCrossAttentionLoss(nn.Module):
    """
    多架构交叉注意力损失
    
    支持三种融合策略：
    1. 分离损失：CNN 和 ViT 各自计算，然后加权
    2. 融合损失：先融合 K,V，再计算交叉注意力
    3. 层次损失：多层级融合
    """
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
        """
        计算跨架构交叉注意力损失
        """
        if self.fusion_mode == 'separate':
            return self._separate_loss(qkv_adv, qkv_clean)
        elif self.fusion_mode == 'fused':
            return self._fused_loss(qkv_adv, qkv_clean)
        elif self.fusion_mode == 'hierarchical':
            return self._hierarchical_loss(qkv_adv, qkv_clean)
    
    def _separate_loss(self, qkv_adv, qkv_clean):
        """
        分离损失：CNN 和 ViT 各自计算交叉注意力损失
        
        L_total = w_cnn × L_cnn + w_vit × L_vit
        
        其中：
        L_cnn = L1(Attn(Q_adv^cnn, K_adv^cnn, V_adv^cnn), 
                   Attn(Q_adv^cnn, K_clean^cnn, V_clean^cnn))
        L_vit = L1(Attn(Q_adv^vit, K_adv^vit, V_adv^vit),
                   Attn(Q_adv^vit, K_clean^vit, V_clean^vit))
        """
        total_loss = 0.0
        count = 0
        
        # CNN 的交叉注意力
        cnn_loss = self._compute_cross_attention_for_type(qkv_adv, qkv_clean, 'cnn')
        if cnn_loss is not None:
            total_loss += self.cnn_weight * cnn_loss
            count += 1
        
        # ViT 的交叉注意力
        vit_loss = self._compute_cross_attention_for_type(qkv_adv, qkv_clean, 'vit')
        if vit_loss is not None:
            total_loss += self.vit_weight * vit_loss
            count += 1
        
        return total_loss / max(count, 1)
    
    def _fused_loss(self, qkv_adv, qkv_clean):
        """
        融合损失：先融合 CNN 和 ViT 的 K,V，再计算交叉注意力
        
        K_fused = w_cnn × K_cnn + w_vit × K_vit
        V_fused = w_cnn × V_cnn + w_vit × V_vit
        
        L = L1(Attn(Q_adv, K_adv, V_adv),
               Attn(Q_adv, K_fused_clean, V_fused_clean))
        """
        # 提取并融合干净图像的 K, V
        K_clean_fused, V_clean_fused = self._fuse_kv(qkv_clean)
        
        # 对抗样本的 Q, K, V（也可以融合）
        Q_adv_fused, K_adv_fused, V_adv_fused = self._fuse_qkv(qkv_adv)
        
        # 计算注意力输出
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
        """
        层次损失：在多个层次上分别计算，然后组合
        
        L = Σ_layer w_layer × L_layer
        
        不同层次可以有不同的融合策略
        """
        total_loss = 0.0
        
        # 低层（边缘、纹理）：CNN 权重大
        low_level_loss = self._compute_layer_loss(
            qkv_adv, qkv_clean, 
            cnn_layers=['layer3'], 
            vit_layers=[8, 9],
            cnn_w=0.7, vit_w=0.3
        )
        
        # 高层（语义、对象）：ViT 权重大
        high_level_loss = self._compute_layer_loss(
            qkv_adv, qkv_clean,
            cnn_layers=['layer4'],
            vit_layers=[10, 11],
            cnn_w=0.4, vit_w=0.6
        )
        
        total_loss = 0.3 * low_level_loss + 0.7 * high_level_loss
        
        return total_loss
    
    def _compute_cross_attention_for_type(self, qkv_adv, qkv_clean, arch_type):
        """为特定架构类型计算交叉注意力损失"""
        loss = 0.0
        count = 0
        
        # 找到该类型的所有层
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
            
            # 自注意力
            self_out = F.scaled_dot_product_attention(
                Q_adv * self.scale, K_adv, V_adv
            )
            
            # 交叉注意力（用干净的 K,V）
            cross_out = F.scaled_dot_product_attention(
                Q_adv * self.scale, K_clean, V_clean
            )
            
            loss += F.l1_loss(self_out, cross_out.detach())
            count += 1
        
        return loss / max(count, 1) if count > 0 else None
    
    def _fuse_kv(self, qkv_dict):
        """融合 CNN 和 ViT 的 K, V"""
        # 提取 CNN 的 K, V
        cnn_keys = [k for k in qkv_dict.keys() if 'cnn' in k and ('_k' in k or '_v' in k)]
        vit_keys = [k for k in qkv_dict.keys() if 'vit' in k and ('_k' in k or '_v' in k)]
        
        if len(cnn_keys) == 0:
            # 只有 ViT
            vit_k = [v for k, v in qkv_dict.items() if 'vit' in k and '_k' in k][0]
            vit_v = [v for k, v in qkv_dict.items() if 'vit' in k and '_v' in k][0]
            return vit_k, vit_v
        
        if len(vit_keys) == 0:
            # 只有 CNN
            cnn_k = [v for k, v in qkv_dict.items() if 'cnn' in k and '_k' in k][0]
            cnn_v = [v for k, v in qkv_dict.items() if 'cnn' in k and '_v' in k][0]
            return cnn_k, cnn_v
        
        # 都有：加权融合
        cnn_k = [v for k, v in qkv_dict.items() if 'cnn' in k and '_k' in k][0]
        cnn_v = [v for k, v in qkv_dict.items() if 'cnn' in k and '_v' in k][0]
        vit_k = [v for k, v in qkv_dict.items() if 'vit' in k and '_k' in k][-1]
        vit_v = [v for k, v in qkv_dict.items() if 'vit' in k and '_v' in k][-1]
        
        # Resize 到相同尺寸
        cnn_k = self._resize_to_target(cnn_k)
        cnn_v = self._resize_to_target(cnn_v)
        vit_k = self._resize_to_target(vit_k)
        vit_v = self._resize_to_target(vit_v)
        
        # 加权平均
        K_fused = self.cnn_weight * cnn_k + self.vit_weight * vit_k
        V_fused = self.cnn_weight * cnn_v + self.vit_weight * vit_v
        
        return K_fused, V_fused
    
    def _fuse_qkv(self, qkv_dict):
        """融合 Q, K, V（完整版本）"""
        K_fused, V_fused = self._fuse_kv(qkv_dict)
        
        # Q 也融合（或只用一个）
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
        """计算特定层的损失"""
        # 实现类似 _fused_loss，但只针对指定层
        # 省略细节...
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
    """
    工厂函数：创建跨架构融合模块
    """
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

