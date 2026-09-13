"""Attack-aware deepfake detection package."""

from .config import ExperimentConfig
from .model import AttackAwareDeepfakeDetector

__all__ = ["AttackAwareDeepfakeDetector", "ExperimentConfig"]
