import argparse
import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, models, transforms

from models.latent_diffusion_attack import LatentDiffusionAttack


def _get_victim_preprocess(arch: str, device: torch.device):
    if arch in ["resnet18", "resnet50", "densenet169"]:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
        return mean, std, (32, 32)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return mean, std, (224, 224)


def normalize_for_model(images: torch.Tensor, arch: str):
    mean, std, input_size = _get_victim_preprocess(arch, images.device)
    if images.shape[-2:] != input_size:
        images = F.interpolate(images, size=input_size, mode="bilinear", align_corners=False)
    return (images - mean) / std


def denormalize_from_model(images: torch.Tensor, arch: str, target_size=(32, 32)):
    mean, std, _ = _get_victim_preprocess(arch, images.device)
    x01 = images * std + mean
    x01 = torch.clamp(x01, 0, 1)
    if target_size is not None and x01.shape[-2:] != tuple(target_size):
        x01 = F.interpolate(x01, size=target_size, mode="bilinear", align_corners=False)
    return x01


def load_victim_model(victim_cfg, device):
    arch = victim_cfg["arch"]
    if arch in ["swin_t", "swint"]:
        arch = "swin_tiny_patch4_window7_224"
    weights_path = victim_cfg.get("weights_path")
    num_classes = int(victim_cfg.get("num_classes", 10))

    if arch in ["resnet50", "resnet18"]:
        if arch == "resnet50":
            model = models.resnet50(num_classes=num_classes)
        else:
            model = models.resnet18(num_classes=num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == "densenet169":
        model = models.densenet169(num_classes=num_classes)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        try:
            import timm
        except Exception as e:
            raise ImportError("timm is required to load this victim model") from e
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)

    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict, strict=True)
        print(f"✓ Loaded victim model ({arch}) from {weights_path}")
    elif weights_path:
        print(f"⚠ Victim weights not found: {weights_path}")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_greedy(
    diffusion_model: LatentDiffusionAttack,
    clean_img_01: torch.Tensor,
    victim_model: torch.nn.Module,
    victim_arch: str,
    labels: torch.Tensor,
    num_steps: int,
    num_candidates: int,
    base_seed: int = 0,
    score_type: str = "ce",
):
    """Greedy candidate search sampling.

    Returns:
        adv_img_01: [B,3,32,32] in [0,1]
        query_count: int, total number of victim forward calls made inside sampling
    """
    device = clean_img_01.device

    z_clean = diffusion_model.vae.encode(clean_img_01)
    z_t = torch.randn_like(z_clean)

    ce = nn.CrossEntropyLoss(reduction="none")
    total_queries = 0

    def _compute_scores(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if score_type == "ce":
            return ce(logits, y)
        if score_type == "p_true":
            probs = torch.softmax(logits, dim=1)
            return -probs[torch.arange(logits.shape[0], device=logits.device), y]
        if score_type == "margin":
            true_logits = logits[torch.arange(logits.shape[0], device=logits.device), y]
            tmp = logits.clone()
            tmp[torch.arange(logits.shape[0], device=logits.device), y] = -1e9
            max_other = tmp.max(dim=1).values
            return -(true_logits - max_other)
        raise ValueError(f"Unknown score_type: {score_type}")

    for i in reversed(range(num_steps)):
        t = torch.full((z_clean.shape[0],), i, device=device, dtype=torch.long)

        best_scores = torch.full((z_clean.shape[0],), -1e9, device=device)
        best_z_next = None

        for c in range(num_candidates):
            torch.manual_seed(base_seed + i * 1000 + c)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(base_seed + i * 1000 + c)

            z_input = torch.cat([z_clean, z_t], dim=1)
            noise_pred = diffusion_model.unet(z_input, t)
            z_next = diffusion_model._p_sample_step(z_t, t, noise_pred)

            x_next_01 = diffusion_model.vae.decode(z_next)
            x_next_norm = normalize_for_model(x_next_01, victim_arch)
            logits = victim_model(x_next_norm)
            total_queries += 1

            scores = _compute_scores(logits, labels)  # [B], larger is better

            if best_z_next is None:
                best_z_next = z_next.clone()
                best_scores = scores
            else:
                improve = scores > best_scores
                if improve.any():
                    best_scores = torch.where(improve, scores, best_scores)
                    best_z_next = torch.where(improve.view(-1, 1, 1, 1), z_next, best_z_next)

        z_t = best_z_next

    adv_img_01 = diffusion_model.vae.decode(z_t)
    return adv_img_01, total_queries


def attack_greedy(
    diffusion_model,
    victim_model,
    test_loader,
    victim_arch,
    save_dir,
    eps,
    max_queries,
    num_steps,
    num_candidates,
    score_type,
    save_successful,
    save_successful_limit,
):
    total_tested = 0
    correctly_classified = 0
    misclassified = 0
    success_count = 0
    queries_list = []
    successful_attacks = []

    global_idx = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Attacking")):
            images = images.to(next(victim_model.parameters()).device)
            labels = labels.to(images.device)

            batch_size = len(images)
            batch_global_indices = torch.arange(global_idx, global_idx + batch_size, device=images.device)
            global_idx += batch_size
            total_tested += batch_size

            # 原始预测：只攻击 clean 正确分类样本
            outputs = victim_model(normalize_for_model(images, victim_arch))
            preds = outputs.argmax(dim=1)
            correct_mask = (preds == labels)

            correctly_classified += correct_mask.sum().item()
            misclassified += (~correct_mask).sum().item()

            if correct_mask.sum() == 0:
                continue

            remaining_images_norm = images[correct_mask].clone()
            remaining_labels = labels[correct_mask].clone()
            remaining_indices = batch_global_indices[correct_mask].clone()

            # 转为 [0,1] 给 diffusion
            remaining_images_01 = denormalize_from_model(remaining_images_norm, victim_arch, target_size=(32, 32))
            remaining_images_01_for_attack = remaining_images_01.clone()

            used_queries_for_this_batch = 0

            for query in range(1, max_queries + 1):
                if len(remaining_images_01_for_attack) == 0:
                    break

                adv_images_01, q_used = sample_greedy(
                    diffusion_model=diffusion_model,
                    clean_img_01=remaining_images_01_for_attack,
                    victim_model=victim_model,
                    victim_arch=victim_arch,
                    labels=remaining_labels,
                    num_steps=num_steps,
                    num_candidates=num_candidates,
                    base_seed=query,
                    score_type=score_type,
                )
                used_queries_for_this_batch += q_used

                # L∞ clip in [0,1]
                adv_noise = adv_images_01 - remaining_images_01_for_attack
                adv_noise = torch.clamp(adv_noise, -eps, eps)
                adv_images_01 = torch.clamp(remaining_images_01_for_attack + adv_noise, 0, 1)

                adv_outputs = victim_model(normalize_for_model(adv_images_01, victim_arch))
                adv_preds = adv_outputs.argmax(dim=1)
                used_queries_for_this_batch += int(adv_outputs.shape[0])

                success_mask = (adv_preds != remaining_labels)
                num_success = success_mask.sum().item()

                if num_success > 0:
                    success_count += num_success
                    queries_list.extend([query] * num_success)

                    if save_successful and len(successful_attacks) < save_successful_limit:
                        idxs = torch.where(success_mask)[0]
                        remaining_slots = max(save_successful_limit - len(successful_attacks), 0)
                        if remaining_slots > 0:
                            idxs = idxs[:remaining_slots]
                            for i in idxs:
                                successful_attacks.append(
                                    {
                                        "clean": remaining_images_01_for_attack[i].cpu(),
                                        "adv": adv_images_01[i].cpu(),
                                        "true_label": remaining_labels[i].item(),
                                        "pred_label": adv_preds[i].item(),
                                        "queries": query,
                                        "method": f"greedy_candidates_{num_candidates}",
                                        "global_index": int(remaining_indices[i].item()),
                                    }
                                )

                fail_mask = ~success_mask
                remaining_images_01_for_attack = remaining_images_01_for_attack[fail_mask]
                remaining_labels = remaining_labels[fail_mask]
                remaining_indices = remaining_indices[fail_mask]

            if correctly_classified >= 10000:
                break

    model_acc = correctly_classified / total_tested if total_tested > 0 else 0
    success_rate = success_count / correctly_classified if correctly_classified > 0 else 0
    avg_queries = np.mean(queries_list) if queries_list else 0
    median_queries = np.median(queries_list) if queries_list else 0

    os.makedirs(save_dir, exist_ok=True)

    results = {
        "version": "2.0-hierarchical-greedy",
        "total_tested": total_tested,
        "correctly_classified": correctly_classified,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_queries": avg_queries,
        "median_queries": median_queries,
        "queries_list": queries_list,
        "num_steps": num_steps,
        "num_candidates": num_candidates,
        "score_type": score_type,
        "eps": float(eps),
    }

    with open(os.path.join(save_dir, "attack_results_hierarchical_greedy.json"), "w") as f:
        json.dump(results, f, indent=2)

    if successful_attacks:
        torch.save(successful_attacks, os.path.join(save_dir, "successful_attacks_hierarchical_greedy.pth"))

        clean_images = torch.stack([x["clean"] for x in successful_attacks], dim=0)
        adv_images = torch.stack([x["adv"] for x in successful_attacks], dim=0)
        true_labels = torch.tensor([x["true_label"] for x in successful_attacks], dtype=torch.long)
        global_indices = torch.tensor([x["global_index"] for x in successful_attacks], dtype=torch.long)

        torch.save(
            {
                "clean": clean_images,
                "adv": adv_images,
                "labels": true_labels,
                "indices": global_indices,
                "victim_arch": victim_arch,
                "eps": float(eps),
                "version": "2.0-hierarchical-greedy",
                "num_candidates": int(num_candidates),
            },
            os.path.join(save_dir, "successful_samples_clean_adv_greedy.pth"),
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="JAD v2.0 Hierarchical Greedy Attack")
    parser.add_argument("--config", type=str, default="config/attack_cifar10.json")
    parser.add_argument("--model", type=str, required=True, help="Path to hierarchical model checkpoint")

    parser.add_argument("--victim", type=str, default="densenet169", help="victim arch: densenet169/resnet50/... or timm name")
    parser.add_argument("--victim_weights", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=80)
    parser.add_argument("--max_queries", type=int, default=1000)
    parser.add_argument("--score_type", type=str, default="ce", choices=["ce", "margin", "p_true"])
    parser.add_argument("--eps", type=float, default=4 / 255, help="L-inf perturbation budget in [0,1], e.g. 8/255")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for dataloader, overrides config if set")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.model, map_location="cpu")
    train_config = checkpoint.get("config", {})

    if train_config and "model" in train_config:
        diffusion_model = LatentDiffusionAttack(
            vae_path="/home/wcy/code/JADnew/models/stable-diffusion-v1-5/vae",
            unet_config=train_config["model"]["unet"],
            beta_schedule=train_config["model"]["beta_schedule"],
            qkv_extractor=None,
            scaling_factor=train_config["model"]["vae"]["scaling_factor"],
        ).to(device)
    else:
        diffusion_model = LatentDiffusionAttack(
            vae_path="/home/wcy/code/JADnew/models/stable-diffusion-v1-5/vae"
        ).to(device)

    diffusion_model.load_state_dict(checkpoint["model"], strict=False)
    diffusion_model = diffusion_model.to(device)

    if "ema" in checkpoint and checkpoint["ema"]:
        for name, param in diffusion_model.named_parameters():
            if name in checkpoint["ema"]:
                param.data = checkpoint["ema"][name].to(param.device)

    diffusion_model.eval()

    test_dataset = datasets.CIFAR10(
        root="./data/CIFAR-10",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["attack"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    victim_cfg = {
        "arch": args.victim,
        "weights_path": args.victim_weights,
        "num_classes": args.num_classes,
    }
    victim_model = load_victim_model(victim_cfg, device)

    eps = args.eps

    base_save_dir = config["output"]["save_dir"]
    attack_tag = os.path.splitext(os.path.basename(args.model))[0]
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_save_dir, attack_tag, run_ts, args.victim)

    save_successful = config.get("output", {}).get("save_successful", False)
    save_successful_limit = int(config.get("output", {}).get("save_successful_limit", 100))

    results = attack_greedy(
        diffusion_model=diffusion_model,
        victim_model=victim_model,
        test_loader=test_loader,
        victim_arch=args.victim,
        save_dir=save_dir,
        eps=eps,
        max_queries=args.max_queries,
        num_steps=args.num_steps,
        num_candidates=args.num_candidates,
        score_type=args.score_type,
        save_successful=save_successful,
        save_successful_limit=save_successful_limit,
    )

    print("✓ Done.")
    print(json.dumps({k: results[k] for k in ["success_rate", "avg_queries", "median_queries"]}, indent=2))


if __name__ == "__main__":
    main()
