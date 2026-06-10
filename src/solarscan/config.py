"""Typed configuration loaded from configs/*.yaml (pydantic-validated)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class SiteConfig(BaseModel):
    module_rated_kw: float = 0.4
    peak_sun_hours: float = 4.5


class ClassifierConfig(BaseModel):
    backbone: str = "convnext_nano"
    input_size: tuple[int, int] = (64, 64)
    num_classes: int = 12
    pretrained: bool = True
    class_balanced_sampling: bool = True


class TrainConfig(BaseModel):
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 4
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    synthetic_augmentation: bool = True
    amp: bool = True  # mixed precision on the A4000


class DataConfig(BaseModel):
    root: Path = Path("data/raw/infrared-solar-modules")


class Config(BaseModel):
    site: SiteConfig = Field(default_factory=SiteConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load a config YAML; falls back to all-defaults when no file is given."""
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(raw)
