import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torchvision import models

from models.latent_diffusion_attack import LatentDiffusionAttack
from models.joint_attention_fusion import (
    create_joint_attention_system,
    GradCAMExtractor,
    ViTGradCAMExtractor,
    JointAttentionFusion,
    JointAttentionLoss
)
from data.dataset import get_dataloader
from utils.lr_scheduler import create_scheduler

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()

def load_victim_models(config, device):

    models_dict = {}

    cnn_config = config['joint_attention']['cnn_model']
    arch = cnn_config['arch']

    if arch == 'resnet50':
        cnn_model = models.resnet50(num_classes=10)
    elif arch == 'resnet18':
        cnn_model = models.resnet18(num_classes=10)
    elif arch == 'vgg16':
        cnn_model = models.vgg16(num_classes=10)
    else:
        raise ValueError(f'Unknown CNN arch: {arch}')

    if arch in ['resnet18', 'resnet50']:
        cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 10)
    elif arch == 'vgg16':
        cnn_model.classifier[6] = nn.Linear(4096, 10)

    if os.path.exists(cnn_config['weights_path']):
        cnn_model.load_state_dict(torch.load(cnn_config['weights_path'], map_location='cpu'))
        print(f'✓ Loaded CNN from {cnn_config["weights_path"]}')
    else:
        print(f'⚠ CNN weights not found')

    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    models_dict['cnn'] = cnn_model

    vit_config = config['joint_attention'].get('vit_model', {})
    if vit_config.get('enabled', False):
        try:
            import timm
            vit_model = timm.create_model(
                vit_config['arch'],
                pretrained=False,
                num_classes=10
            )

            if os.path.exists(vit_config['weights_path']):
                vit_model.load_state_dict(torch.load(vit_config['weights_path'], map_location='cpu'))
                print(f'✓ Loaded ViT from {vit_config["weights_path"]}')
            else:
                print(f'⚠ ViT weights not found')

            vit_model = vit_model.to(device)
            vit_model.eval()
            models_dict['vit'] = vit_model
        except Exception as e:
            print(f'⚠ Failed to load ViT: {e}')
            models_dict['vit'] = None
    else:
        models_dict['vit'] = None

    return models_dict

def train_epoch(
    diffusion_model,
    gradcam_extractor,
    vit_gradcam_extractor,
    fusion_module,
    attention_loss_fn,
    train_loader,
    optimizer,
    device,
    epoch,
    writer,
    config,
    ema=None
):

    diffusion_model.train()
    fusion_module.train()

    total_loss = 0.0
    loss_dict_total = {}

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for i, batch in enumerate(pbar):
        if 'cond_image' in batch and 'gt_image' in batch:
            clean_img = batch['cond_image'].to(device)
            adv_img = batch['gt_image'].to(device)
        else:
            clean_img = batch['clean_image'].to(device)
            adv_img = batch['adv_image'].to(device)

        labels = batch.get('label', torch.zeros(clean_img.shape[0], dtype=torch.long)).to(device)

        z_clean = diffusion_model.vae.encode(clean_img)
        z_adv = diffusion_model.vae.encode(adv_img)

        B = z_clean.shape[0]
        t = torch.randint(0, diffusion_model.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_adv)
        z_noisy = diffusion_model.q_sample(z_adv, t, noise)

        z_input = torch.cat([z_clean, z_noisy], dim=1)
        noise_pred = diffusion_model.unet(z_input, t)

        loss_diff = F.mse_loss(noise_pred, noise)

        w_diff = config.get('loss_weights', {}).get('diffusion', 1.0)
        losses = {'diffusion': loss_diff.item()}
        total_loss_batch = w_diff * loss_diff

        if gradcam_extractor is not None:
            try:

                z_adv_pred = diffusion_model._predict_x0(z_noisy, t, noise_pred)

                z_adv_scaled = z_adv_pred / diffusion_model.scaling_factor
                z_clean_scaled = z_clean / diffusion_model.scaling_factor

                vae_out_adv = diffusion_model.vae.decode(z_adv_scaled)
                vae_out_clean = diffusion_model.vae.decode(z_clean_scaled)

                x_adv_pred = vae_out_adv.sample if hasattr(vae_out_adv, 'sample') else vae_out_adv
                x_clean = vae_out_clean.sample if hasattr(vae_out_clean, 'sample') else vae_out_clean

                x_adv_pred = (x_adv_pred + 1.0) / 2.0
                x_clean = (x_clean + 1.0) / 2.0

                cnn_attn_adv = gradcam_extractor.extract_attention(x_adv_pred, labels)
                cnn_attn_clean = gradcam_extractor.extract_attention(x_clean, labels)

                vit_attn_adv = {}
                vit_attn_clean = {}
                if vit_gradcam_extractor is not None:
                    x_adv_pred_224 = F.interpolate(x_adv_pred, size=(224, 224), mode='bilinear', align_corners=False)
                    x_clean_224 = F.interpolate(x_clean, size=(224, 224), mode='bilinear', align_corners=False)

                    x_adv_pred_224_norm = (x_adv_pred_224 - 0.5) / 0.5
                    x_clean_224_norm = (x_clean_224 - 0.5) / 0.5

                    vit_attn_adv = vit_gradcam_extractor.extract_attention(x_adv_pred_224_norm, labels)
                    vit_attn_clean = vit_gradcam_extractor.extract_attention(x_clean_224_norm, labels)

                A_joint_adv, A_cnn_adv, A_vit_adv, w_cnn_adv, w_vit_adv = fusion_module(
                    cnn_attn_adv, vit_attn_adv
                )
                A_joint_clean, A_cnn_clean, A_vit_clean, w_cnn_clean, w_vit_clean = fusion_module(
                    cnn_attn_clean, vit_attn_clean
                )

                loss_attention = attention_loss_fn(A_joint_adv, A_joint_clean)

                w_attn = config.get('loss_weights', {}).get('joint_attention', 0.01)
                total_loss_batch = total_loss_batch + w_attn * loss_attention

                losses['joint_attention'] = loss_attention.item()

                if i == 0 and epoch % 10 == 0:
                    print(f'\n🔍 Joint attention fusion:')
                    print(f'  CNN dynamic weight: {w_cnn_adv.mean():.3f}')
                    print(f'  ViT dynamic weight: {w_vit_adv.mean():.3f}')
                    print(f'  Attention loss: {loss_attention.item():.4f}')

            except Exception as e:
                if i == 0:
                    import traceback
                    print(f'\n⚠ Joint attention loss failed: {e}')
                    traceback.print_exc()

        losses['total'] = total_loss_batch.item()

        optimizer.zero_grad()
        total_loss_batch.backward()

        if config['train'].get('grad_clip', 0) > 0:
            nn.utils.clip_grad_norm_(
                diffusion_model.parameters(),
                config['train']['grad_clip']
            )

        optimizer.step()

        if ema is not None:
            global_step = epoch * len(train_loader) + i
            if global_step > config['train']['ema'].get('start_iter', 500):
                ema.update()

        total_loss += total_loss_batch.item()
        for key, value in losses.items():
            loss_dict_total[key] = loss_dict_total.get(key, 0) + value

        pbar.set_postfix({
            'diff': f"{losses.get('diffusion', 0):.4f}",
            'attn': f"{losses.get('joint_attention', 0):.4f}"
        })

        global_step = epoch * len(train_loader) + i
        if global_step % config['train'].get('log_iter', 100) == 0:
            for key, value in losses.items():
                writer.add_scalar(f'train/{key}', value, global_step)

    avg_loss = total_loss / len(train_loader)
    for key in loss_dict_total:
        loss_dict_total[key] /= len(train_loader)

    return avg_loss, loss_dict_total

@torch.no_grad()
def validate(diffusion_model, val_loader, device, epoch, writer):
    diffusion_model.eval()
    total_loss = 0.0

    for batch in tqdm(val_loader, desc='Validation'):
        if 'cond_image' in batch and 'gt_image' in batch:
            clean_img = batch['cond_image'].to(device)
            adv_img = batch['gt_image'].to(device)
        else:
            clean_img = batch['clean_image'].to(device)
            adv_img = batch['adv_image'].to(device)

        z_clean = diffusion_model.vae.encode(clean_img)
        z_adv = diffusion_model.vae.encode(adv_img)

        B = z_clean.shape[0]
        t = torch.randint(0, diffusion_model.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_adv)
        z_noisy = diffusion_model.q_sample(z_adv, t, noise)

        z_input = torch.cat([z_clean, z_noisy], dim=1)
        noise_pred = diffusion_model.unet(z_input, t)

        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    writer.add_scalar('val/diffusion', avg_loss, epoch)

    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='JAD v2.0 Joint Attention Fusion Training')
    parser.add_argument('--config', type=str, default='config/train_joint_attention.json')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    print('='*80)
    print('JAD v2.0 Joint Attention Fusion Training')
    print('='*80)
    print(f'Device: {device}')
    print(f'Config: {args.config}')
    print('')
    print('🏗️ Attention Fusion Strategy:')
    print('  Step 1: Layer-wise fusion within CNN and ViT')
    print('  Step 2: Dynamic cross-model fusion')
    print('  CNN: Grad-CAM attention maps')
    print('  ViT: Self-attention to class token')
    print('='*80 + '\n')

    seed = config['seed']
    if seed == -1:
        import time
        seed = int(time.time()) % (2**32)

    torch.manual_seed(seed)
    np.random.seed(seed)

    exp_dir = os.path.join(config['path']['base_dir'], config['name'])
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)

    writer = SummaryWriter(os.path.join(exp_dir, 'tb_logger'))

    print('Loading data...')
    train_loader, val_loader = get_dataloader(config, phase='train')
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset) if val_loader else 0}')

    print('\nLoading victim models...')
    victim_models = load_victim_models(config, device)

    print('\nCreating joint attention fusion system...')
    ja_config = config['joint_attention']

    gradcam_extractor, vit_gradcam_extractor, fusion_module, attention_loss_fn = create_joint_attention_system(
        cnn_model=victim_models['cnn'],
        vit_model=victim_models.get('vit'),
        cnn_layers=ja_config['cnn_model']['layers'],
        vit_blocks=ja_config.get('vit_model', {}).get('blocks', []),
        target_size=tuple(ja_config['fusion_config']['target_size']),
        loss_scale=ja_config['loss_config'].get('scale', 1.0)
    )

    fusion_module = fusion_module.to(device)
    print('✓ Joint attention system created (CNN + ViT Grad-CAM)')

    print('\nCreating Latent Diffusion model...')
    diffusion_model = LatentDiffusionAttack(
        vae_path=config['model']['vae']['pretrained'],
        unet_config=config['model']['unet'],
        beta_schedule=config['model']['beta_schedule'],
        qkv_extractor=None,
        scaling_factor=config['model']['vae']['scaling_factor']
    ).to(device)

    trainable_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params / 1e6:.2f}M')

    optimizer = optim.AdamW(
        diffusion_model.unet.parameters(),
        **config['train']['optimizer']
    )

    scheduler_type = config['train']['lr_scheduler'].get('type', 'warmup_cosine')
    scheduler = create_scheduler(optimizer, config, scheduler_type=scheduler_type)
    print(f'✓ LR scheduler: {scheduler_type}')

    ema = None
    if config['train']['ema']['enabled']:
        ema = EMA(diffusion_model, decay=config['train']['ema']['decay'])
        print(f'✓ EMA enabled (decay={config["train"]["ema"]["decay"]})')

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f'\nResuming from {args.resume}')
        checkpoint = torch.load(args.resume)
        diffusion_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print('\n' + '='*80)
    print('Start Joint Attention Training...')
    print('='*80 + '\n')

    for epoch in range(start_epoch, config['train']['n_epochs']):
        train_loss, train_dict = train_epoch(
            diffusion_model,
            gradcam_extractor,
            vit_gradcam_extractor,
            fusion_module,
            attention_loss_fn,
            train_loader,
            optimizer,
            device,
            epoch,
            writer,
            config,
            ema
        )

        print(f'\n📊 Epoch {epoch}:')
        print(f'  Total Loss: {train_loss:.4f}')
        print(f'  Diffusion: {train_dict.get("diffusion", 0):.4f}')
        print(f'  Joint Attention: {train_dict.get("joint_attention", 0):.4f}')

        if val_loader and epoch % config['train']['val_epoch'] == 0:
            if ema:
                ema.apply_shadow()

            val_loss = validate(diffusion_model, val_loader, device, epoch, writer)

            if ema:
                ema.restore()

            print(f'  Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(exp_dir, 'checkpoints', 'best_model_joint_attention.pth')

                torch.save({
                    'epoch': epoch,
                    'model': diffusion_model.state_dict(),
                    'ema': ema.shadow if ema else None,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': config
                }, save_path)

                print(f'  ✅ Saved best model (val_loss: {val_loss:.4f})')

        if (epoch + 1) % config['train']['save_checkpoint_epoch'] == 0:
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'epoch_{epoch}_joint_attention.pth')
            torch.save({
                'epoch': epoch,
                'model': diffusion_model.state_dict(),
                'ema': ema.shadow if ema else None,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config
            }, checkpoint_path)
            print(f'  ✅ Saved checkpoint')

        if scheduler_type == 'adaptive':
            if val_loader and epoch % config['train']['val_epoch'] == 0:
                scheduler.step(val_loss=val_loss, epoch=epoch)
            else:
                scheduler.step(epoch=epoch)
        else:
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        print(f'  LR: {current_lr:.6f}')
        print('='*80)

    print('\n🎉 Training completed!')
    print(f'Best val loss: {best_val_loss:.4f}')

    if gradcam_extractor:
        gradcam_extractor.remove_hooks()
    if vit_gradcam_extractor:
        vit_gradcam_extractor.remove_hooks()

    writer.close()

if __name__ == '__main__':
    main()
