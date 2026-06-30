"""Training configuration: dataclasses + YAML loader.

Mirrors the style of model/config.py (frozen dataclasses, nested unpack).
Loads from configs/training/<name>.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    shard_dir: str
    seq_len: int
    val_shard: int


@dataclass(frozen=True)
class TrainParams:
    batch_size: int
    total_steps: int
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    grad_accum: int = 1


@dataclass(frozen=True)
class OptimParams:
    lr_peak: float
    lr_min: float
    b1: float
    b2: float
    eps: float


@dataclass(frozen=True)
class CkptConfig:
    save_every: int
    output_dir: str


@dataclass(frozen=True)
class LogConfig:
    log_file: str
    log_every: int
    eval_every: int
    eval_batches: int


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    dataset: DatasetConfig
    train: TrainParams
    optim: OptimParams
    ckpt: CkptConfig
    log: LogConfig


_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "training"


def load_training_config(
    name: str,
    configs_dir: Path | None = None,
) -> TrainConfig:
    """Load a TrainConfig from {configs_dir}/{name}.yaml.

    Args:
        name: Config basename (e.g. "model_25m"); .yaml suffix optional.
        configs_dir: Override configs directory. Defaults to configs/training.

    Raises:
        KeyError: If config file not found.
    """
    cfg_dir = configs_dir or _CONFIGS_DIR
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    path = cfg_dir / name
    if not path.exists():
        raise KeyError(f"Config '{name}' not found in {cfg_dir}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return TrainConfig(
        model_name=raw["model_name"],
        dataset=DatasetConfig(**raw["dataset"]),
        train=TrainParams(**raw["train"]),
        optim=OptimParams(**raw["optim"]),
        ckpt=CkptConfig(**raw["ckpt"]),
        log=LogConfig(**raw["log"]),
    )
