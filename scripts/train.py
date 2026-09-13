#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image

from unfooled.config import make_config
from unfooled.data import build_dataloaders, download_datasets
from unfooled.evaluation import evaluate_robustness_suite
from unfooled.model import UnFooledNet
from unfooled.training import train
from unfooled.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train the UnFooled v2.3 model.")
    parser.add_argument("--mode", choices=["FAST", "PRO"], default="FAST")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--download", action="store_true", help="Cache the datasets before training."
    )
    parser.add_argument(
        "--skip-test", action="store_true", help="Skip final robustness evaluation."
    )
    return parser.parse_args()


def json_ready_config(config) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.as_dict().items()
    }


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="`rcond` parameter will change",
        category=FutureWarning,
        module="insightface",
    )
    Image.MAX_IMAGE_PIXELS = 933120000
    config = make_config(args.mode, args.data_root)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(json_ready_config(config))

    if args.download:
        fake_paths, real_paths = download_datasets(config)
        print(f"cached: {len(fake_paths)} fake, {len(real_paths)} real")
    if not config.fake_dir.exists() or not config.real_dir.exists():
        raise FileNotFoundError(
            "Dataset directories are missing. Run scripts/download_data.py or pass --download."
        )

    train_loader, val_loader, test_loader = build_dataloaders(config)
    print(
        {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        }
    )
    model = UnFooledNet().to(device)
    print(
        "params (M):",
        round(sum(parameter.numel() for parameter in model.parameters()) / 1e6, 2),
    )
    history, optimizer = train(model, train_loader, val_loader, config, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "unfooled_v2_3_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": json_ready_config(config),
            "history": history,
        },
        checkpoint_path,
    )
    print("Saved checkpoint:", checkpoint_path)

    if not args.skip_test:
        results = evaluate_robustness_suite(
            model,
            test_loader,
            device,
            config.image_size,
            config.tta_n,
            with_attack_iou=config.eval_iou_on_attacks,
        )
        results_path = args.output_dir / "test_metrics.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        print("Saved metrics:", results_path)


if __name__ == "__main__":
    main()

