"""
损失函数模块
包含扩散损失、注意力蒸馏损失和对抗损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DiffusionLoss(nn.Module):
    """扩散模型的噪声预测损失"""
    def __init__(self, loss_type: str = 'l2'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self, 
        noise_pred: torch.Tensor, 
        noise_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noise_pred: [B, C, H, W] - 预测的噪声
            noise_true: [B, C, H, W] - 真实的噪声
        """
        if self.loss_type == 'l1':
            return F.l1_loss(noise_pred, noise_true)
        elif self.loss_type == 'l2':
            return F.mse_loss(noise_pred, noise_true)
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(noise_pred, noise_true)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')


class AttentionDistillationLoss(nn.Module):
    """
    注意力蒸馏损失
    鼓励生成的对抗样本保持与原图相似的注意力模式
    """
    def __init__(self, loss_type: str = 'l1', temperature: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
    
    def forward(
        self,
        attn_adv: Dict[str, torch.Tensor],
        attn_clean: Dict[str, torch.Tensor],
        detach_target: bool = True
    ) -> torch.Tensor:
        """
        计算注意力蒸馏损失
        
        Args:
            attn_adv: 对抗样本的注意力图 {layer_name: [B, H, W]}
            attn_clean: 干净图像的注意力图 {layer_name: [B, H, W]}
            detach_target: 是否 detach 目标注意力
        
        Returns:
            注意力蒸馏损失
        """
        total_loss = 0.0
        count = 0
        
        for layer_name in attn_adv.keys():
            if layer_name not in attn_clean:
                continue
            
            attn_a = attn_adv[layer_name]
            attn_c = attn_clean[layer_name]
            
            if detach_target:
                attn_c = attn_c.detach()
            
            # 温度缩放
            attn_a = attn_a / self.temperature
            attn_c = attn_c / self.temperature
            
            # 计算损失
            if self.loss_type == 'l1':
                loss = F.l1_loss(attn_a, attn_c)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(attn_a, attn_c)
            elif self.loss_type == 'kl':
                # KL 散度（需要归一化）
                attn_a_norm = F.softmax(attn_a.flatten(1), dim=1)
                attn_c_norm = F.softmax(attn_c.flatten(1), dim=1)
                loss = F.kl_div(
                    attn_a_norm.log(), 
                    attn_c_norm, 
                    reduction='batchmean'
                )
            elif self.loss_type == 'cosine':
                # 余弦相似度损失
                attn_a_flat = attn_a.flatten(1)
                attn_c_flat = attn_c.flatten(1)
                loss = 1 - F.cosine_similarity(attn_a_flat, attn_c_flat, dim=1).mean()
            else:
                raise ValueError(f'Unknown loss type: {self.loss_type}')
            
            total_loss += loss
            count += 1
        
        return total_loss / max(count, 1)


class AdversarialLoss(nn.Module):
    """
    对抗损失
    鼓励生成的样本能够欺骗目标模型
    """
    def __init__(self, use_logits: bool = True):
        super().__init__()
        self.use_logits = use_logits
    
    def forward(
        self,
        outputs: torch.Tensor,
        true_labels: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            outputs: [B, num_classes] - 模型输出（logits 或概率）
            true_labels: [B] - 真实标签
            targeted: 是否为目标攻击
            target_labels: [B] - 目标标签（仅用于目标攻击）
        
        Returns:
            对抗损失（越小越好 = 攻击成功）
        """
        if targeted:
            # 目标攻击：最大化目标类别的概率
            if target_labels is None:
                raise ValueError('Target labels must be provided for targeted attack')
            
            if self.use_logits:
                # 最小化目标类别的交叉熵
                loss = F.cross_entropy(outputs, target_labels)
            else:
                # 最大化目标类别的概率
                target_probs = outputs.gather(1, target_labels.unsqueeze(1)).squeeze(1)
                loss = -torch.log(target_probs + 1e-8).mean()
        else:
            # 非目标攻击：降低真实类别的概率
            if self.use_logits:
                # 最大化除真实类别外其他类别的 logits
                # 等价于最小化真实类别的概率
                loss = -F.cross_entropy(outputs, true_labels)
            else:
                # 最小化真实类别的概率
                true_probs = outputs.gather(1, true_labels.unsqueeze(1)).squeeze(1)
                loss = torch.log(true_probs + 1e-8).mean()
        
        return loss


class PerceptualLoss(nn.Module):
    """
    感知损失（基于预训练网络的特征）
    可选：保持对抗样本与原图在感知上的相似性
    """
    def __init__(self, feature_layers: list = [2, 7, 12, 21]):
        super().__init__()
        # 使用预训练的 VGG 提取特征
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features
        
        self.layers = nn.ModuleList()
        self.feature_layers = feature_layers
        
        last_layer = 0
        for layer_idx in feature_layers:
            self.layers.append(vgg[last_layer:layer_idx])
            last_layer = layer_idx
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] - 生成的图像
            y: [B, 3, H, W] - 目标图像
        """
        # Resize 到 VGG 输入尺寸
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        
        loss = 0.0
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    """
    def __init__(
        self,
        diffusion_weight: float = 1.0,
        attention_cnn_weight: float = 0.3,
        attention_vit_weight: float = 0.3,
        adversarial_weight: float = 0.1,
        perceptual_weight: float = 0.0,
        attention_loss_type: str = 'l1',
        use_adversarial: bool = False
    ):
        super().__init__()
        
        self.diffusion_weight = diffusion_weight
        self.attention_cnn_weight = attention_cnn_weight
        self.attention_vit_weight = attention_vit_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight
        self.use_adversarial = use_adversarial
        
        # 初始化各个损失
        self.diffusion_loss = DiffusionLoss(loss_type='l2')
        self.attention_loss = AttentionDistillationLoss(loss_type=attention_loss_type)
        self.adversarial_loss = AdversarialLoss(use_logits=True) if use_adversarial else None
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
        attn_adv: Optional[Dict[str, torch.Tensor]] = None,
        attn_clean: Optional[Dict[str, torch.Tensor]] = None,
        model_outputs: Optional[torch.Tensor] = None,
        true_labels: Optional[torch.Tensor] = None,
        x_adv: Optional[torch.Tensor] = None,
        x_clean: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算组合损失
        
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        
        # 1. 扩散损失（必需）
        loss_diff = self.diffusion_loss(noise_pred, noise_true)
        losses['diffusion'] = loss_diff.item()
        total_loss = self.diffusion_weight * loss_diff
        
        # 2. 注意力蒸馏损失
        if attn_adv is not None and attn_clean is not None:
            # 分离 CNN 和 ViT 的注意力
            attn_adv_cnn = {k: v for k, v in attn_adv.items() if 'cnn' in k}
            attn_clean_cnn = {k: v for k, v in attn_clean.items() if 'cnn' in k}
            attn_adv_vit = {k: v for k, v in attn_adv.items() if 'vit' in k or 'layer' in k}
            attn_clean_vit = {k: v for k, v in attn_clean.items() if 'vit' in k or 'layer' in k}
            
            # CNN 注意力损失
            if len(attn_adv_cnn) > 0 and self.attention_cnn_weight > 0:
                loss_attn_cnn = self.attention_loss(attn_adv_cnn, attn_clean_cnn)
                losses['attention_cnn'] = loss_attn_cnn.item()
                total_loss += self.attention_cnn_weight * loss_attn_cnn
            
            # ViT 注意力损失
            if len(attn_adv_vit) > 0 and self.attention_vit_weight > 0:
                loss_attn_vit = self.attention_loss(attn_adv_vit, attn_clean_vit)
                losses['attention_vit'] = loss_attn_vit.item()
                total_loss += self.attention_vit_weight * loss_attn_vit
        
        # 3. 对抗损失
        if self.use_adversarial and model_outputs is not None and true_labels is not None:
            loss_adv = self.adversarial_loss(model_outputs, true_labels, targeted=False)
            losses['adversarial'] = loss_adv.item()
            total_loss += self.adversarial_weight * loss_adv
        
        # 4. 感知损失
        if self.perceptual_weight > 0 and x_adv is not None and x_clean is not None:
            loss_perc = self.perceptual_loss(x_adv, x_clean)
            losses['perceptual'] = loss_perc.item()
            total_loss += self.perceptual_weight * loss_perc
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses

