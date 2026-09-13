from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .attacks import ATTACK_FUNCS
from .masks import robust_face_bbox_mask
from .utils import denorm_to_pil, norm_batch, pil_gamma, pil_jpeg_roundtrip, tensorize_pil_list


def mask_iou_bundle(
    predicted_mask: torch.Tensor, ground_truth: torch.Tensor, threshold: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    if ground_truth.shape[-2:] != predicted_mask.shape[-2:]:
        ground_truth = F.interpolate(
            ground_truth, size=predicted_mask.shape[-2:], mode="nearest"
        )
    prediction = predicted_mask[:, 0].clamp(0, 1)
    target = ground_truth[:, 0].clamp(0, 1)
    prediction_binary = prediction >= threshold
    target_binary = target >= 0.5
    intersection = (prediction_binary & target_binary).flatten(1).sum(1).float()
    union = (prediction_binary | target_binary).flatten(1).sum(1).float() + 1e-6
    hard_iou = intersection / union
    soft_intersection = (prediction * target).flatten(1).sum(1)
    soft_union = (prediction + target - prediction * target).flatten(1).sum(1) + 1e-6
    soft_iou = soft_intersection / soft_union
    return hard_iou, soft_iou


def ece_binary(labels, probabilities, n_bins: int = 15) -> float:
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.where(predictions == 1, probabilities, 1.0 - probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        members = (confidence >= bins[index]) & (
            confidence
            < (bins[index + 1] if index < n_bins - 1 else confidence <= bins[index + 1])
        )
        if members.sum() == 0:
            continue
        accuracy = (predictions[members] == labels[members]).mean()
        ece += members.mean() * abs(accuracy - confidence[members].mean())
    return float(ece)


def evaluate(model, loader, device: torch.device, image_size: int, tta_n: int = 1):
    model.eval()
    all_probabilities, all_labels, hard_ious, soft_ious = [], [], [], []
    with torch.no_grad():
        for inputs, masks, labels, _ in tqdm(loader, desc="eval", leave=False):
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            logits_list, masks_list = [], []
            for tta_index in range(max(1, int(tta_n))):
                if tta_index == 0:
                    transformed = (
                        inputs
                        if inputs.shape[-1] == image_size
                        else F.interpolate(inputs, (image_size, image_size))
                    )
                    logits, mask_logits = model(transformed)
                else:
                    pil_images = denorm_to_pil(inputs)
                    jittered = []
                    for image in pil_images:
                        width, height = image.size
                        delta_width, delta_height = int(0.02 * width), int(0.02 * height)
                        left = random.randint(0, max(0, delta_width))
                        top = random.randint(0, max(0, delta_height))
                        transformed_image = image.crop(
                            (left, top, width - left, height - top)
                        ).resize((image_size, image_size), Image.BICUBIC)
                        transformed_image = pil_jpeg_roundtrip(
                            transformed_image, quality=random.randint(70, 90)
                        )
                        transformed_image = pil_gamma(
                            transformed_image, gamma=random.uniform(0.95, 1.05)
                        )
                        jittered.append(transformed_image)
                    transformed = norm_batch(
                        tensorize_pil_list(
                            jittered, resize_to=(image_size, image_size)
                        ).to(device)
                    )
                    logits, mask_logits = model(transformed)
                if mask_logits.shape[-2:] != (image_size, image_size):
                    mask_logits = F.interpolate(
                        mask_logits,
                        size=(image_size, image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                logits_list.append(logits)
                masks_list.append(torch.sigmoid(mask_logits))

            aggregated_logits = torch.stack(logits_list, 0).mean(0)
            aggregated_masks = torch.stack(masks_list, 0).amax(0)
            probabilities = torch.sigmoid(aggregated_logits).detach().cpu().numpy()
            all_probabilities += probabilities.tolist()
            all_labels += labels.cpu().numpy().tolist()

            hard_iou, soft_iou = mask_iou_bundle(
                aggregated_masks, masks, threshold=0.3
            )
            for index in range(labels.shape[0]):
                if labels[index] >= 0.5 and masks[index].sum() > 0:
                    hard_ious.append(float(hard_iou[index].item()))
                    soft_ious.append(float(soft_iou[index].item()))

    auc = (
        roc_auc_score(all_labels, all_probabilities)
        if len(set(all_labels)) > 1
        else float("nan")
    )
    average_precision = (
        average_precision_score(all_labels, all_probabilities)
        if len(set(all_labels)) > 1
        else float("nan")
    )
    accuracy = float(
        np.mean(
            (
                (np.array(all_probabilities) >= 0.5).astype(int)
                == np.array(all_labels)
            ).astype(np.float32)
        )
    )
    metrics = {
        "AUC": auc,
        "AP": average_precision,
        "ACC": accuracy,
        "ECE": ece_binary(all_labels, all_probabilities, n_bins=15),
        "IoU@0.3": float(np.mean(hard_ious)) if hard_ious else float("nan"),
        "IoU_soft": float(np.mean(soft_ious)) if soft_ious else float("nan"),
    }
    return metrics, (all_labels, all_probabilities)


def eval_attack(
    model,
    loader,
    attack_name: str,
    device: torch.device,
    image_size: int,
    with_iou: bool = True,
):
    model.eval()
    all_probabilities, all_labels = [], []
    hard_ious, soft_ious = [], []
    with torch.no_grad():
        for inputs, _, labels, _ in tqdm(
            loader, desc=f"atk:{attack_name}", leave=False
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            images = [ATTACK_FUNCS[attack_name](image) for image in denorm_to_pil(inputs)]
            transformed = norm_batch(
                tensorize_pil_list(images, resize_to=(image_size, image_size)).to(device)
            )
            logits, mask_logits = model(transformed)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            all_probabilities += probabilities.tolist()
            all_labels += labels.cpu().numpy().tolist()

            if with_iou:
                attacked_masks = []
                for image in images:
                    mask = robust_face_bbox_mask(image)
                    attacked_masks.append(
                        torch.from_numpy(mask.astype(np.float32))[None, ...] / 255.0
                    )
                attacked_masks = torch.stack(attacked_masks, 0).to(device)
                predicted_masks = torch.sigmoid(mask_logits)
                hard_iou, soft_iou = mask_iou_bundle(
                    predicted_masks, attacked_masks, threshold=0.3
                )
                for index in range(labels.shape[0]):
                    if labels[index] >= 0.5 and attacked_masks[index].sum() > 0:
                        hard_ious.append(float(hard_iou[index].item()))
                        soft_ious.append(float(soft_iou[index].item()))

    auc = (
        roc_auc_score(all_labels, all_probabilities)
        if len(set(all_labels)) > 1
        else float("nan")
    )
    accuracy = float(
        np.mean(
            (
                (np.array(all_probabilities) >= 0.5).astype(int)
                == np.array(all_labels)
            ).astype(np.float32)
        )
    )
    output = {
        "AUC": auc,
        "ACC": accuracy,
        "ECE": ece_binary(all_labels, all_probabilities, n_bins=15),
    }
    if with_iou:
        output["IoU@0.3"] = float(np.mean(hard_ious)) if hard_ious else float("nan")
        output["IoU_soft"] = float(np.mean(soft_ious)) if soft_ious else float("nan")
    return output


def eval_surveillance(model, loader, device: torch.device, image_size: int):
    def low_light(image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(0.6)
        image = ImageEnhance.Contrast(image).enhance(0.9)
        image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        return pil_jpeg_roundtrip(image, quality=40)

    model.eval()
    all_probabilities, all_labels = [], []
    with torch.no_grad():
        for inputs, _, labels, _ in tqdm(loader, desc="surveillance", leave=False):
            images = [low_light(image) for image in denorm_to_pil(inputs.to(device))]
            transformed = norm_batch(
                tensorize_pil_list(images, resize_to=(image_size, image_size)).to(device)
            )
            logits, _ = model(transformed)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            all_probabilities += probabilities.tolist()
            all_labels += labels.cpu().numpy().tolist()
    auc = (
        roc_auc_score(all_labels, all_probabilities)
        if len(set(all_labels)) > 1
        else float("nan")
    )
    accuracy = float(
        np.mean(
            (
                (np.array(all_probabilities) >= 0.5).astype(int)
                == np.array(all_labels)
            ).astype(np.float32)
        )
    )
    return {
        "AUC": auc,
        "ACC": accuracy,
        "ECE": ece_binary(all_labels, all_probabilities, n_bins=15),
    }


def evaluate_robustness_suite(
    model,
    loader,
    device: torch.device,
    image_size: int,
    tta_n: int,
    with_attack_iou: bool = True,
):
    clean_metrics, _ = evaluate(model, loader, device, image_size, tta_n=tta_n)
    results = {"clean": clean_metrics}
    for attack_name in ATTACK_FUNCS:
        results[attack_name] = eval_attack(
            model,
            loader,
            attack_name,
            device,
            image_size,
            with_iou=with_attack_iou,
        )
    worst_accuracy = min(metrics["ACC"] for metrics in results.values())
    worst_auc = min(metrics["AUC"] for metrics in results.values())
    delta_auc = {
        name: metrics["AUC"] - results["clean"]["AUC"]
        for name, metrics in results.items()
        if name != "clean"
    }
    return {
        "splits": results,
        "summary": {
            "worst_ACC": worst_accuracy,
            "worst_AUC": worst_auc,
            "delta_AUC": delta_auc,
        },
        "surveillance": eval_surveillance(model, loader, device, image_size),
    }
