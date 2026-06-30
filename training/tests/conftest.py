"""Shared fixtures for training tests.

Generates 4 toy shards × 10k tokens of uint16 random data + matching
metadata.json in a pytest tmp_path/. No real shards needed locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.config import (
    CkptConfig,
    DatasetConfig,
    LogConfig,
    OptimParams,
    TrainConfig,
    TrainParams,
)


@pytest.fixture
def toy_shards(tmp_path) -> Path:
    """4 shards × 10k tokens, uint16, fixed seed + matching metadata.json."""
    rng = np.random.default_rng(42)
    n_shards = 4
    tokens_per_shard = 10_000
    records = []
    for i in range(n_shards):
        tokens = rng.integers(0, 32768, size=tokens_per_shard, dtype=np.uint16)
        np.save(tmp_path / f"shard_{i:05d}.npy", tokens)
        records.append({
            "index": i,
            "filename": f"shard_{i:05d}.npy",
            "tokens": tokens_per_shard,
            "size_mb": round(tokens.nbytes / 1e6, 2),
        })
    metadata = {
        "dataset_version": "smoke_test",
        "vocab_size": 32768,
        "total_tokens": n_shards * tokens_per_shard,
        "total_shards": n_shards,
        "shard_size": tokens_per_shard,
        "shards": records,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


@pytest.fixture
def toy_config(toy_shards) -> TrainConfig:
    """A TrainConfig pointing at the toy_shards dir; mirrors smoke_test.yaml."""
    ckpt_dir = toy_shards / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    return TrainConfig(
        model_name="model_25m",
        dataset=DatasetConfig(
            shard_dir=str(toy_shards),
            seq_len=128,
            val_shard=3,
        ),
        train=TrainParams(
            batch_size=4,
            total_steps=5,
            warmup_steps=2,
            weight_decay=0.1,
            grad_clip=1.0,
            grad_accum=1,
        ),
        optim=OptimParams(
            lr_peak=3.0e-4,
            lr_min=3.0e-5,
            b1=0.9,
            b2=0.95,
            eps=1.0e-8,
        ),
        ckpt=CkptConfig(
            save_every=2,
            output_dir=str(ckpt_dir),
        ),
        log=LogConfig(
            log_file=str(toy_shards / "train_log.jsonl"),
            log_every=1,
            eval_every=2,
            eval_batches=2,
        ),
    )
