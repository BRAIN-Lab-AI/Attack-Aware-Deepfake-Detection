#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unfooled.config import make_config
from unfooled.data import download_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Cache the notebook's two datasets.")
    parser.add_argument("--mode", choices=["FAST", "PRO"], default="FAST")
    parser.add_argument("--data-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="The secret `HF_TOKEN` does not exist in your Colab secrets.",
        category=UserWarning,
        module="huggingface_hub",
    )
    config = make_config(args.mode, args.data_root)
    fake_paths, real_paths = download_datasets(config)
    print(f"cached: {len(fake_paths)} fake, {len(real_paths)} real")


if __name__ == "__main__":
    main()

