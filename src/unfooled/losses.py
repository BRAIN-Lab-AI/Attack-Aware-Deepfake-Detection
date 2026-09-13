from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def dice_loss_per_sample(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    numerator = 2 * (probabilities * target).flatten(1).sum(1) + eps
    denominator = (
        probabilities.flatten(1).sum(1) + target.flatten(1).sum(1) + eps
    )
    return 1.0 - numerator / denominator


def sobel_edges(inputs: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=torch.float32,
        device=inputs.device,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
        dtype=torch.float32,
        device=inputs.device,
    ).unsqueeze(0)
    gradient_x = F.conv2d(inputs, kernel_x, padding=1)
    gradient_y = F.conv2d(inputs, kernel_y, padding=1)
    return torch.sqrt(gradient_x * gradient_x + gradient_y * gradient_y + 1e-6)


def batch_pos_weight(
    mask_batch: torch.Tensor,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    foreground_fraction = float(mask_batch.mean().item())
    weight = (1.0 - foreground_fraction) / (foreground_fraction + eps)
    return torch.tensor(
        [float(np.clip(weight, min_weight, max_weight))], device=mask_batch.device
    )

