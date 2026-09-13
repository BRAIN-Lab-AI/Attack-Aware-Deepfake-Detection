#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image

from unfooled.config import make_config
from unfooled.data import build_dataloaders
from unfooled.evaluation import evaluate_robustness_suite
from unfooled.model import AttackAwareDeepfakeDetector
from unfooled.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mode", choices=["FAST", "PRO"], default="FAST")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/metrics.json")
    )
    parser.add_argument("--no-attack-iou", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Image.MAX_IMAGE_PIXELS = 933120000
    config = make_config(args.mode, args.data_root)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = build_dataloaders(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AttackAwareDeepfakeDetector(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    results = evaluate_robustness_suite(
        model,
        test_loader,
        device,
        config.image_size,
        config.tta_n,
        with_attack_iou=not args.no_attack_iou,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("Saved metrics:", args.output)


if __name__ == "__main__":
    main()
