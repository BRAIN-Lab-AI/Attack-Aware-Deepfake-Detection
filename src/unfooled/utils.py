from __future__ import annotations

import io
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

IM_MEAN = torch.tensor([0.485, 0.456, 0.406])
IM_STD = torch.tensor([0.229, 0.224, 0.225])


def seed_everything(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_to_square(image: Image.Image, fill: int = 0) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    output = Image.new("RGB", (side, side), color=(fill, fill, fill))
    output.paste(image, ((side - width) // 2, (side - height) // 2))
    return output


def pil_jpeg_roundtrip(
    image: Image.Image, quality: int = 75, subsampling: str = "4:2:0"
) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=int(quality), subsampling=subsampling)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def pil_gamma(image: Image.Image, gamma: float = 1.0) -> Image.Image:
    array = np.asarray(image).astype(np.float32) / 255.0
    array = np.clip(array, 1e-6, 1.0) ** float(gamma)
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def norm_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = IM_MEAN.to(batch.device)[None, :, None, None]
    std = IM_STD.to(batch.device)[None, :, None, None]
    return (batch - mean) / std


def denorm_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = IM_MEAN.to(batch.device)[None, :, None, None]
    std = IM_STD.to(batch.device)[None, :, None, None]
    return batch * std + mean


def tensorize_pil_list(
    images: list[Image.Image], resize_to: tuple[int, int] | None = None
) -> torch.Tensor:
    batch = torch.stack([transforms.ToTensor()(image) for image in images], dim=0)
    if resize_to is not None:
        batch = F.interpolate(batch, size=resize_to, mode="bilinear", align_corners=False)
    return batch


def denorm_to_pil(batch: torch.Tensor) -> list[Image.Image]:
    batch = denorm_batch(batch).clamp(0, 1)
    return [to_pil_image(item.cpu()) for item in batch]
