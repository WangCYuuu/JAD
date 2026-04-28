# JAD : Joint Attention Distillation for Black-box Attacks

This repository contains the attack code and checkpoints for the paper:

**Latent Danger Zone: Distilling Unified Attention for Cross-Architecture Black-box Attacks**

> **Implementation Note.**  
> This repository provides the officially maintained **JAD v2.0** implementation, with an optimized hierarchical QKV-based attention distillation pipeline.

---

## Overview

This project provides a latent-diffusion-based black-box attack pipeline with unified attention distillation across heterogeneous victim architectures (e.g., CNN + ViT).

Main components include:

- **Hierarchical greedy black-box attack evaluation**
- **Cross-architecture attention / QKV distillation modules**
- **CIFAR-10 and ImageNet-oriented attack config templates**


---

## Repository Structure

- `attack_hierarchical_greedy.py` – greedy-query black-box attack script.
- `models/` – core model modules (latent diffusion, U-Net, attention extraction/fusion, losses).
- `config/` – attack configs and ablation configs.
- `checkpoints/` – pretrained checkpoints for attack.

---

## Environment

Recommended: Python 3.9+ with CUDA-enabled PyTorch.

Typical dependencies:

- `torch`
- `torchvision`
- `numpy`
- `tqdm`
- `diffusers`
- `timm` (for some victim models)

Install dependencies according to your platform/CUDA version.

---

## Required Downloads (Checkpoint + Model Assets)

To run the attack, please download the required checkpoint from:

https://drive.google.com/drive/folders/13Arvu7NDvWecjnUNpf53d03oYxr616f9?usp=drive_link

In addition, you need to prepare the following model asset folders locally and place them under `models/`:

- `models/sd-vae-ft-mse`
- `models/stable-diffusion-v1-5`

These folders are required by the attack pipeline and are not included in this repository.

The adversarial pair file data/adv_pairs/cifar10_dual_arch_eps16.pth has also been uploaded to cloud storage. Please download it yourself and place it at: 

- `data/adv_pairs/cifar10_dual_arch_eps16.pth`

After downloading the checkpoint, place it in your local `checkpoints/` directory (or any path you prefer), and pass the path via `--model`.

---

## Attack

Example:

```bash
python attack_hierarchical_greedy.py \
  --config config/attack_cifar10.json \
  --model /path/to/checkpoints/best_model_joint_attention.pth \
  --victim densenet169 \
  --victim_weights /path/to/victim_weights.pth
```

Notes:

1. `--model` should be the downloaded pretrained diffusion checkpoint.
2. `--victim_weights` should be a trained victim classifier checkpoint.
3. Make sure the VAE path used in the code/config is valid in your environment.

---

## Version Notice

This repository provides the officially maintained **JAD v2.0** implementation.

JAD v2.0 preserves the core idea of joint attention distillation and introduces a hierarchical QKV-based distillation module for faster attack evaluation, lower memory overhead, and better extensibility.

---

## Citation

If you find this repository useful, please cite:

**Latent Danger Zone: Distilling Unified Attention for Cross-Architecture Black-box Attacks**

@ARTICLE{11474508,
  author={Li, Yang and Wang, Chenyu and Wang, Tingrui and Wang, Yongwei and Li, Haonan and Liu, Zhunga and Pan, Quan},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={Latent Danger Zone: Distilling Unified Attention for Cross-Architecture Black-box Attacks}, 
  year={2026},
  volume={},
  number={},
  pages={1-18},
  keywords={Weapons;Feedback;Circuits;MIMICs;Millimeter wave integrated circuits;Monolithic integrated circuits;Pixel;Multiaccess communication;Communication systems;Network architecture;Black-box Adversarial Attacks;Latent Diffusion Models;Cross-architecture Transferability;Generative Adversarial Example Generation},
  doi={10.1109/TDSC.2026.3680667}}

