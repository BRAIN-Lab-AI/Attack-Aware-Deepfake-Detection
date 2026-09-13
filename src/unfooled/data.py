from __future__ import annotations

import io
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from datasets import disable_caching as hf_disable_caching
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import ExperimentConfig
from .masks import FaceMaskCache, soften_mask
from .utils import IM_MEAN, IM_STD, pad_to_square


def cache_hf_images(
    dataset_id: str,
    split: str,
    column: str = "image",
    n: int = 1000,
    out_dir: str | Path = ".",
) -> list[str]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    paths: list[str] = []
    count = 0
    for record in dataset:
        if count >= n:
            break
        image = record.get(column, None)
        if image is None:
            image = next(iter(record.values()))
        try:
            if isinstance(image, Image.Image):
                converted = image.convert("RGB")
            elif isinstance(image, dict) and "bytes" in image:
                converted = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
            elif isinstance(image, (bytes, bytearray)):
                converted = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                converted = Image.fromarray(np.array(image)).convert("RGB")
        except Exception:
            continue
        converted = pad_to_square(converted).resize((384, 384), Image.BICUBIC)
        file_path = output_dir / f"{uuid.uuid4().hex}.jpg"
        converted.save(file_path, "JPEG", quality=95, subsampling="4:2:0")
        paths.append(str(file_path))
        count += 1
        if count % 100 == 0:
            print("  saved", count)
    return paths


def download_datasets(config: ExperimentConfig) -> tuple[list[str], list[str]]:
    config.fake_dir.mkdir(parents=True, exist_ok=True)
    config.real_dir.mkdir(parents=True, exist_ok=True)
    hf_disable_caching()
    fake_paths = cache_hf_images(
        config.fake_dataset, "train", "image", config.n_fake, config.fake_dir
    )
    real_paths = cache_hf_images(
        config.real_dataset, "train", "image", config.n_real, config.real_dir
    )
    return fake_paths, real_paths


class DiskDataset(Dataset):
    def __init__(
        self,
        fake_dir: str | Path,
        real_dir: str | Path,
        mask_cache_dir: str | Path,
        split: str = "train",
        seed: int = 2025,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        image_size: int = 256,
    ):
        fake_dir, real_dir = Path(fake_dir), Path(real_dir)
        paths = [(str(fake_dir / name), 1) for name in os.listdir(fake_dir)]
        paths += [(str(real_dir / name), 0) for name in os.listdir(real_dir)]
        paths = [
            (path, label)
            for path, label in paths
            if path.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        rng = random.Random(seed)
        rng.shuffle(paths)
        count = len(paths)
        val_count = int(count * val_frac)
        test_count = int(count * test_frac)
        if split == "train":
            self.samples = paths[: count - val_count - test_count]
        elif split == "val":
            self.samples = paths[count - val_count - test_count : count - test_count]
        else:
            self.samples = paths[count - test_count :]
        self.split = split
        self.image_size = image_size
        self.mask_cache = FaceMaskCache(mask_cache_dir)
        self.train_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.9, 1.1), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            ]
        )
        self.to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IM_MEAN.tolist(), std=IM_STD.tolist()),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image_transformed = (
            self.train_aug(image) if self.split == "train" else self.eval_transform(image)
        )
        if label == 1:
            mask = self.mask_cache.get(path, image)
            mask = soften_mask(mask, sigma=2.0)
        else:
            mask = np.zeros((image.height, image.width), dtype=np.float32)
        mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0)[
            None, ...
        ]
        image_tensor = self.to_tensor_norm(image_transformed)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        return image_tensor, mask_tensor, label_tensor, path


def build_dataloaders(config: ExperimentConfig):
    common = dict(
        fake_dir=config.fake_dir,
        real_dir=config.real_dir,
        mask_cache_dir=config.mask_cache_dir,
        seed=config.seed,
        val_frac=config.val_frac,
        test_frac=config.test_frac,
        image_size=config.image_size,
    )
    train_set = DiskDataset(split="train", **common)
    val_set = DiskDataset(split="val", **common)
    test_set = DiskDataset(split="test", **common)
    loader_common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_common)
    val_loader = DataLoader(val_set, shuffle=False, **loader_common)
    test_loader = DataLoader(test_set, shuffle=False, **loader_common)
    return train_loader, val_loader, test_loader

