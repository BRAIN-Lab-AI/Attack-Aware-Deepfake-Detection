from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Notebook v2.3 defaults, exposed without changing their values."""

    mode: str = "FAST"
    fake_dataset: str = "OpenRL/DeepFakeFace"
    real_dataset: str = "nielsr/CelebA-faces"
    data_root: Path = Path("data/unfooled_data_v2_2")
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 2025
    image_size: int = 256
    eval_iou_on_attacks: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False

    @property
    def n_fake(self) -> int:
        return 800 if self.mode == "FAST" else 4000

    @property
    def n_real(self) -> int:
        return 800 if self.mode == "FAST" else 4000

    @property
    def epochs(self) -> int:
        return 2 if self.mode == "FAST" else 5

    @property
    def batch_size(self) -> int:
        return 32 if self.mode == "FAST" else 48

    @property
    def tta_n(self) -> int:
        return 3 if self.mode == "FAST" else 5

    @property
    def fake_dir(self) -> Path:
        return self.data_root / "fake"

    @property
    def real_dir(self) -> Path:
        return self.data_root / "real"

    @property
    def mask_cache_dir(self) -> Path:
        return self.data_root / "mask_cache_v2_2"

    def as_dict(self) -> dict:
        values = asdict(self)
        values.update(
            n_fake=self.n_fake,
            n_real=self.n_real,
            epochs=self.epochs,
            batch_size=self.batch_size,
            tta_n=self.tta_n,
        )
        return values


def make_config(mode: str = "FAST", data_root: str | Path | None = None) -> ExperimentConfig:
    mode = mode.upper()
    if mode not in {"FAST", "PRO"}:
        raise ValueError("mode must be 'FAST' or 'PRO'")
    kwargs = {"mode": mode}
    if data_root is not None:
        kwargs["data_root"] = Path(data_root)
    return ExperimentConfig(**kwargs)

