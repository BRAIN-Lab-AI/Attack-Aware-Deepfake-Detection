from __future__ import annotations

import random
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .masks import robust_face_bbox_mask
from .utils import pil_gamma, pil_jpeg_roundtrip


def attack_jpeg_realign(image: Image.Image) -> Image.Image:
    width, height = image.size
    dx, dy = random.randint(-3, 3), random.randint(-3, 3)
    canvas = Image.new("RGB", (width + 8, height + 8), (0, 0, 0))
    canvas.paste(image, (4 + dx, 4 + dy))
    shifted = canvas.crop((4, 4, 4 + width, 4 + height))
    return pil_jpeg_roundtrip(
        shifted, quality=random.randint(35, 85), subsampling="4:2:0"
    )


def attack_resample_warp(image: Image.Image) -> Image.Image:
    width, height = image.size
    angle = random.uniform(-3, 3)
    scale = random.uniform(0.96, 1.04)
    transformed = image.rotate(angle, resample=Image.BICUBIC)
    transformed = transformed.resize(
        (int(width * scale), int(height * scale)), Image.BICUBIC
    )
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    offset_x = (width - transformed.size[0]) // 2 + random.randint(-2, 2)
    offset_y = (height - transformed.size[1]) // 2 + random.randint(-2, 2)
    canvas.paste(transformed, (offset_x, offset_y))
    return canvas


def attack_denoise_regrain(image: Image.Image) -> Image.Image:
    array = np.array(image.convert("RGB"))
    array = cv2.fastNlMeansDenoisingColored(array, None, 7, 7, 7, 21)
    noise = np.random.normal(
        0, random.uniform(1.0, 3.0), array.shape
    ).astype(np.float32)
    array = np.clip(array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array).filter(
        ImageFilter.UnsharpMask(radius=1, percent=60, threshold=2)
    )


def attack_seam_smooth(image: Image.Image) -> Image.Image:
    mask = robust_face_bbox_mask(image)
    if mask.mean() == 0:
        return image
    kernel_size = random.choice([3, 5, 7])
    blurred = image.filter(ImageFilter.GaussianBlur(radius=kernel_size))
    edges = cv2.Canny((mask * 255).astype(np.uint8), 0, 1)
    edges = cv2.dilate(
        edges, np.ones((kernel_size, kernel_size), np.uint8), iterations=1
    )
    alpha = cv2.GaussianBlur(
        edges.astype(np.float32), (0, 0), sigmaX=kernel_size
    )
    if alpha.max() > 0:
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-6)
    alpha = np.clip(alpha[..., None], 0, 1)
    source = np.array(image).astype(np.float32) / 255.0
    target = np.array(blurred).astype(np.float32) / 255.0
    output = source * (1 - alpha) + target * alpha
    return Image.fromarray((output * 255.0).clip(0, 255).astype(np.uint8))


def attack_color_gamma(image: Image.Image) -> Image.Image:
    output = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    output = ImageEnhance.Contrast(output).enhance(random.uniform(0.9, 1.1))
    output = ImageEnhance.Color(output).enhance(random.uniform(0.9, 1.1))
    return pil_gamma(output, gamma=random.uniform(0.9, 1.1))


def attack_social_transcode(image: Image.Image) -> Image.Image:
    width, height = image.size
    target = random.choice([720, 640, 512, 384])
    transformed = image.resize((target, int(height * (target / width))), Image.BICUBIC)
    transformed = ImageOps.pad(
        transformed, (384, 384), method=Image.BICUBIC, color=(0, 0, 0)
    )
    return pil_jpeg_roundtrip(transformed, quality=random.randint(55, 80))


ATTACK_FUNCS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "jpeg": attack_jpeg_realign,
    "warp": attack_resample_warp,
    "regrain": attack_denoise_regrain,
    "seam": attack_seam_smooth,
    "gamma": attack_color_gamma,
    "transcode": attack_social_transcode,
}

