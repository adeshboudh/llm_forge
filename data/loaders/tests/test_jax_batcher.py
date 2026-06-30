"""JAXBatcher tests — host-side np batching over ShardedTokenDataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.loaders.jax_batcher import JAXBatcher


def _write_shards(root: Path, n_shards: int, tokens_per_shard: int, start_token: int = 0) -> None:
    """Write deterministic toy shards + metadata.json under root."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    records = []
    for i in range(n_shards):
        tokens = rng.integers(0, 32768, size=tokens_per_shard, dtype=np.uint16)
        np.save(root / f"shard_{i:05d}.npy", tokens)
        records.append(
            {
                "index": i,
                "filename": f"shard_{i:05d}.npy",
                "tokens": tokens_per_shard,
                "size_mb": round(tokens.nbytes / 1e6, 2),
            }
        )
    metadata = {
        "dataset_version": "smoke",
        "vocab_size": 32768,
        "total_tokens": n_shards * tokens_per_shard,
        "total_shards": n_shards,
        "shard_size": tokens_per_shard,
        "shards": records,
    }
    (root / "metadata.json").write_text(json.dumps(metadata))


@pytest.fixture
def shards_dir(tmp_path):
    _write_shards(tmp_path, n_shards=4, tokens_per_shard=10_000)
    return tmp_path


def test_train_iter_yields_correct_shapes(shards_dir):
    batcher = JAXBatcher(
        shard_dir=shards_dir,
        seq_len=128,
        batch_size=4,
        val_shard=3,
        seed=0,
    )
    for input_ids, target_ids in batcher.train_iter():
        assert input_ids.shape == (4, 128)
        assert target_ids.shape == (4, 128)
        assert input_ids.dtype == np.int32
        assert target_ids.dtype == np.int32
        break


def test_train_excludes_val_shard(shards_dir):
    """Val shard must not appear in train shards list."""
    batcher = JAXBatcher(
        shard_dir=shards_dir,
        seq_len=128,
        batch_size=4,
        val_shard=3,
        seed=0,
    )
    train_paths = [p.name for p in batcher._train_dataset._shards]
    assert "shard_00003.npy" not in train_paths
    assert len(train_paths) == 3


def test_val_iter_cycles_infinitely(shards_dir):
    batcher = JAXBatcher(
        shard_dir=shards_dir,
        seq_len=128,
        batch_size=4,
        val_shard=3,
        seed=0,
    )
    n = 0
    for input_ids, _ in batcher.val_iter():
        assert input_ids.shape == (4, 128)
        n += 1
        if n >= 100:
            break
    assert n == 100  # would loop forever otherwise; we cap the test


def test_skip_tokens_advances_cursor(shards_dir):
    """skip_tokens(n) must skip n windows from train stream."""
    batcher = JAXBatcher(
        shard_dir=shards_dir,
        seq_len=128,
        batch_size=4,
        val_shard=3,
        seed=0,
    )
    it_a = batcher.train_iter()
    next(it_a)  # discard first batch; compare second batch below

    batcher_b = JAXBatcher(
        shard_dir=shards_dir,
        seq_len=128,
        batch_size=4,
        val_shard=3,
        seed=0,
    )
    # Skip one batch worth of tokens (batch_size × seq_len = 4 × 128 = 512)
    batcher_b.skip_tokens(4 * 128)
    second_b = next(batcher_b.train_iter())

    # After skipping first batch, second_b's first batch should match it_a's second batch.
    second_a = next(it_a)
    np.testing.assert_array_equal(second_b[0], second_a[0])


def test_empty_shard_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        JAXBatcher(shard_dir=tmp_path, seq_len=128, batch_size=4, val_shard=0)


def test_repr(shards_dir):
    batcher = JAXBatcher(
        shard_dir=shards_dir,
        seq_len=128,
        batch_size=4,
        val_shard=3,
        seed=0,
    )
    s = repr(batcher)
    assert "JAXBatcher" in s
    assert "train_shards=3" in s
    assert "val_shard=3" in s
