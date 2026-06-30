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
        name: Config basename (e.g. "model_25m") or full path
              (e.g. "configs/training/model_25m.yaml"). If it contains a
              directory separator, treated as a full path and loaded as-is.
              Otherwise, looked up under `configs_dir` (or default).
        configs_dir: Override configs directory. Defaults to configs/training.

    Raises:
        KeyError: If config file not found.
    """
    name_path = Path(name)
    if "/" in name or name_path.is_absolute() or name_path.exists():
        path = name_path
    else:
        if not name.endswith(".yaml"):
            name = f"{name}.yaml"
        cfg_dir = configs_dir or _CONFIGS_DIR
        path = cfg_dir / name
    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    if not path.exists():
        raise KeyError(f"Config '{path}' not found")
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return TrainConfig(
        model_name=raw["model_name"],
        dataset=DatasetConfig(**raw["dataset"]),
        train=TrainParams(**raw["train"]),
        optim=OptimParams(**raw["optim"]),
        ckpt=CkptConfig(**raw["ckpt"]),
        log=LogConfig(**raw["log"]),
    )
