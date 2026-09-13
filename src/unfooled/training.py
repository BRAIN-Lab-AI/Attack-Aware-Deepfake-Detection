from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .attacks import ATTACK_FUNCS
from .config import ExperimentConfig
from .evaluation import evaluate
from .losses import batch_pos_weight, dice_loss_per_sample, sobel_edges
from .masks import robust_face_bbox_mask
from .utils import denorm_to_pil, norm_batch, tensorize_pil_list

LAMBDA_MASK_BASE = 0.8
EDGE_WEIGHT = 0.1
CONSISTENCY_WEIGHT = 0.05
SIZE_WEIGHT = 0.15
CLEAN_MASK_WEIGHT = 0.12


def build_optimizer(model):
    head_parameters = list(model.res_adapter.parameters()) + list(
        model.mask_head.parameters()
    )
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_parameter_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": 1e-4, "weight_decay": 1e-4},
            {"params": head_parameters, "lr": 3e-4, "weight_decay": 1e-4},
        ]
    )


@torch.no_grad()
def red_team_select(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    image_size: int,
    max_k: int = 2,
):
    pil_images = denorm_to_pil(inputs)
    names = random.sample(list(ATTACK_FUNCS.keys()), k=max_k)
    scores, tensors, image_sets = [], [], []
    for name in names:
        attacked_images = [ATTACK_FUNCS[name](image) for image in pil_images]
        tensor = tensorize_pil_list(
            attacked_images, resize_to=(image_size, image_size)
        ).to(device)
        attacked_inputs = norm_batch(tensor)
        logits, _ = model(attacked_inputs)
        classification_loss = F.binary_cross_entropy_with_logits(logits, labels)
        scores.append(classification_loss.detach())
        tensors.append(attacked_inputs)
        image_sets.append(attacked_images)
    worst_index = int(torch.argmax(torch.stack(scores)))
    return tensors[worst_index], image_sets[worst_index], names[worst_index]


def train(
    model,
    train_loader,
    val_loader,
    config: ExperimentConfig,
    device: torch.device,
    optimizer=None,
):
    if optimizer is None:
        optimizer = build_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}")
        last_loss = float("nan")
        for inputs, mask_batch, labels, _ in progress:
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask_batch = mask_batch.to(device)

            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                clean_inputs = (
                    inputs
                    if inputs.shape[-1] == config.image_size
                    else F.interpolate(inputs, (config.image_size, config.image_size))
                )
                _, clean_mask_logits = model(clean_inputs)
                clean_predicted_masks = torch.sigmoid(clean_mask_logits)

            clean_mask_ground_truth = mask_batch.clone()
            if clean_mask_ground_truth.shape[-2:] != clean_mask_logits.shape[-2:]:
                clean_mask_ground_truth = F.interpolate(
                    clean_mask_ground_truth,
                    size=clean_mask_logits.shape[-2:],
                    mode="nearest",
                )
            valid_clean = (
                clean_mask_ground_truth.sum(dim=(1, 2, 3)) > 0
            ).float()
            clean_positive_weight = batch_pos_weight(clean_mask_ground_truth)
            clean_bce = F.binary_cross_entropy_with_logits(
                clean_mask_logits,
                clean_mask_ground_truth,
                reduction="none",
                pos_weight=clean_positive_weight,
            ).mean(dim=(1, 2, 3))
            clean_dice = dice_loss_per_sample(
                clean_mask_logits, clean_mask_ground_truth
            )
            clean_mask_loss = 0.5 * clean_bce + 0.5 * clean_dice
            clean_mask_loss = (clean_mask_loss * valid_clean).sum() / (
                valid_clean.sum() + 1e-6
            )

            with torch.no_grad():
                candidate_count = (
                    1 if epoch == 1 else (2 if config.mode == "FAST" else 3)
                )
                attacked_inputs, attacked_images, attack_name = red_team_select(
                    model,
                    inputs,
                    labels,
                    device,
                    config.image_size,
                    max_k=candidate_count,
                )
            attacked_masks = []
            for image in attacked_images:
                mask = robust_face_bbox_mask(image)
                # Kept byte-for-byte equivalent to the notebook's attacked-prior scaling.
                attacked_masks.append(
                    torch.from_numpy(mask.astype(np.float32))[None, ...] / 255.0
                )
            attacked_masks = torch.stack(attacked_masks, 0).to(device)

            lambda_mask = LAMBDA_MASK_BASE * min(1.0, epoch / 2)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, mask_logits = model(attacked_inputs)
                predicted_masks = torch.sigmoid(mask_logits)

                if attacked_masks.shape[-2:] != mask_logits.shape[-2:]:
                    attacked_masks = F.interpolate(
                        attacked_masks, size=mask_logits.shape[-2:], mode="nearest"
                    )
                if clean_predicted_masks.shape[-2:] != predicted_masks.shape[-2:]:
                    clean_predicted_masks = F.interpolate(
                        clean_predicted_masks,
                        size=predicted_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                valid = (attacked_masks.sum(dim=(1, 2, 3)) > 0).float()
                positive_weight = batch_pos_weight(attacked_masks)
                bce_per_sample = F.binary_cross_entropy_with_logits(
                    mask_logits,
                    attacked_masks,
                    reduction="none",
                    pos_weight=positive_weight,
                ).mean(dim=(1, 2, 3))
                dice_per_sample = dice_loss_per_sample(mask_logits, attacked_masks)
                predicted_edges = sobel_edges(predicted_masks).mean(dim=(1, 2, 3))
                target_edges = sobel_edges(attacked_masks).mean(dim=(1, 2, 3))
                edge_per_sample = (predicted_edges - target_edges).abs()
                predicted_area = predicted_masks.mean(dim=(1, 2, 3))
                target_area = attacked_masks.mean(dim=(1, 2, 3))
                size_per_sample = (predicted_area - target_area).abs()

                mask_loss_per_sample = (
                    0.6 * bce_per_sample
                    + 0.4 * dice_per_sample
                    + EDGE_WEIGHT * edge_per_sample
                    + SIZE_WEIGHT * size_per_sample
                )
                mask_loss = (mask_loss_per_sample * valid).sum() / (
                    valid.sum() + 1e-6
                )
                consistency_loss = F.l1_loss(
                    predicted_masks, clean_predicted_masks
                )
                classification_loss = F.binary_cross_entropy_with_logits(
                    logits, labels
                )
                loss = (
                    classification_loss
                    + lambda_mask * mask_loss
                    + CONSISTENCY_WEIGHT * consistency_loss
                    + CLEAN_MASK_WEIGHT * clean_mask_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            last_loss = loss.item()
            progress.set_postfix({"loss": f"{last_loss:.4f}", "atk": attack_name})

        validation_metrics, _ = evaluate(
            model, val_loader, device, config.image_size, tta_n=1
        )
        history.append(
            {"epoch": epoch, "loss": last_loss, "validation": validation_metrics}
        )
        print("Val:", {key: round(value, 3) for key, value in validation_metrics.items()})
    return history, optimizer


def mask_warmup(
    model,
    train_loader,
    device: torch.device,
    image_size: int,
    steps: int = 150,
    learning_rate: float = 1e-3,
):
    """Notebook utility retained for fidelity; it is not called by the main run."""
    model.train()
    for module in [model.fc, model.l1, model.l2, model.l3, model.l4]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(model.mask_head.parameters()) + list(model.res_adapter.parameters()),
        lr=learning_rate,
    )
    iterator = iter(train_loader)
    completed = 0
    while completed < steps:
        try:
            inputs, target_masks, labels, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, target_masks, labels, _ = next(iterator)
        keep = (labels >= 0.5) & (target_masks.sum(dim=(1, 2, 3)) > 0)
        if keep.sum() == 0:
            continue
        inputs = inputs[keep].to(device)
        target_masks = target_masks[keep].to(device)
        if inputs.shape[-1] != image_size:
            inputs = F.interpolate(inputs, (image_size, image_size))
        optimizer.zero_grad(set_to_none=True)
        _, mask_logits = model(inputs)
        if target_masks.shape[-2:] != mask_logits.shape[-2:]:
            target_masks = F.interpolate(
                target_masks, size=mask_logits.shape[-2:], mode="nearest"
            )
        bce = F.binary_cross_entropy_with_logits(mask_logits, target_masks)
        dice = dice_loss_per_sample(mask_logits, target_masks)
        (bce + dice.mean()).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed += 1
    for parameter in model.parameters():
        parameter.requires_grad_(True)
