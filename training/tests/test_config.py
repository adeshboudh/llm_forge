"""TrainConfig dataclass + YAML loader tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from training.config import (
    CkptConfig,
    DatasetConfig,
    LogConfig,
    OptimParams,
    TrainParams,
    load_training_config,
)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "training"


def test_load_model_25m():
    cfg = load_training_config("model_25m", configs_dir=_CONFIGS_DIR)
    assert cfg.model_name == "model_25m"
    assert cfg.dataset.shard_dir == "/kaggle/input/datasets/adeshboudh/llm-forge-tokens-v1/"
    assert cfg.dataset.seq_len == 1024
    assert cfg.dataset.val_shard == 214
    assert cfg.train.batch_size == 128
    assert cfg.train.total_steps == 9766
    assert cfg.train.warmup_steps == 200
    assert cfg.train.weight_decay == 0.1
    assert cfg.train.grad_clip == 1.0
    assert cfg.train.grad_accum == 1
    assert cfg.optim.lr_peak == 3.0e-4
    assert cfg.optim.lr_min == 3.0e-5
    assert cfg.optim.b1 == 0.9
    assert cfg.optim.b2 == 0.95
    assert cfg.optim.eps == 1.0e-8
    assert cfg.ckpt.save_every == 500
    assert cfg.ckpt.output_dir == "/kaggle/working/ckpt/"
    assert cfg.log.log_file == "/kaggle/working/train_log.jsonl"
    assert cfg.log.log_every == 1
    assert cfg.log.eval_every == 500
    assert cfg.log.eval_batches == 50


def test_load_smoke_test():
    cfg = load_training_config("smoke_test", configs_dir=_CONFIGS_DIR)
    assert cfg.model_name == "model_25m"
    assert cfg.dataset.shard_dir == "./data/shards_smoke/"
    assert cfg.dataset.seq_len == 128
    assert cfg.dataset.val_shard == 3
    assert cfg.train.batch_size == 4
    assert cfg.train.total_steps == 5
    assert cfg.train.warmup_steps == 2
    assert cfg.log.eval_every == 2
    assert cfg.log.eval_batches == 2


def test_unknown_config_raises():
    with pytest.raises(KeyError):
        load_training_config("nonexistent", configs_dir=_CONFIGS_DIR)


def test_yaml_suffix_optional():
    a = load_training_config("model_25m", configs_dir=_CONFIGS_DIR)
    b = load_training_config("model_25m.yaml", configs_dir=_CONFIGS_DIR)
    assert a == b


def test_full_path_accepted():
    """--config configs/training/model_25m.yaml must work (CLI passes full path)."""
    a = load_training_config("model_25m", configs_dir=_CONFIGS_DIR)
    b = load_training_config(str(_CONFIGS_DIR / "model_25m.yaml"))
    assert a == b
    # Same path without .yaml suffix should also work.
    c = load_training_config(str(_CONFIGS_DIR / "model_25m"))
    assert a == c


def test_config_is_frozen():
    cfg = load_training_config("model_25m", configs_dir=_CONFIGS_DIR)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train.batch_size = 999


def test_nested_dataclasses_hydrate():
    cfg = load_training_config("model_25m", configs_dir=_CONFIGS_DIR)
    assert isinstance(cfg.dataset, DatasetConfig)
    assert isinstance(cfg.train, TrainParams)
    assert isinstance(cfg.optim, OptimParams)
    assert isinstance(cfg.ckpt, CkptConfig)
    assert isinstance(cfg.log, LogConfig)
