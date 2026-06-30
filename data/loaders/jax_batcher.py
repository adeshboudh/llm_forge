"""Host-side JAX-friendly batcher wrapping ShardedTokenDataset.

Reads toy/real uint16 shards via the Phase-2 ShardedTokenDataset, builds two
streams (train excludes val_shard, val only contains val_shard) and yields
fixed-size (B, seq_len) int32 numpy batches ready to feed pjit.

Stays on host (no jax arrays here) — pjit input shardings handle the device split.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from data.loaders.npy_loader import ShardedTokenDataset


class JAXBatcher:
    """Host numpy batcher over uint16 .npy shards.

    Args:
        shard_dir: Directory containing shard_*.npy + metadata.json.
        seq_len: Sequence length per sample (matches ModelConfig.max_seq_len).
        batch_size: Per-host batch size; pjit splits this across 8 cores.
        val_shard: Index of the shard to reserve for validation (excluded from train).
        seed: Seed for shard order shuffle (resume-deterministic).
    """

    def __init__(
        self,
        shard_dir: str | Path,
        seq_len: int,
        batch_size: int,
        val_shard: int,
        seed: int = 0,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_shard = val_shard
        self.seed = seed
        self._skip_tokens: int = 0

        # Two datasets: train excludes val_shard; val only includes val_shard.
        self._train_dataset = self._build_train_dataset()
        self._val_dataset = self._build_val_dataset()

    def train_iter(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (input_ids, target_ids) int32 batches of shape (B, seq_len)."""
        random.seed(self.seed)
        skip_samples = self._skip_tokens // self.seq_len
        buf_in: list[np.ndarray] = []
        buf_tgt: list[np.ndarray] = []
        for i, (input_ids, target_ids) in enumerate(self._train_dataset):
            if i < skip_samples:
                continue
            buf_in.append(np.asarray(input_ids, dtype=np.int32))
            buf_tgt.append(np.asarray(target_ids, dtype=np.int32))
            if len(buf_in) == self.batch_size:
                yield np.stack(buf_in), np.stack(buf_tgt)
                buf_in, buf_tgt = [], []

    def val_iter(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Cycle val shard infinitely, yielding (B, seq_len) int32 batches."""
        cycled = itertools.cycle(self._val_dataset)
        buf_in: list[np.ndarray] = []
        buf_tgt: list[np.ndarray] = []
        for input_ids, target_ids in cycled:
            buf_in.append(np.asarray(input_ids, dtype=np.int32))
            buf_tgt.append(np.asarray(target_ids, dtype=np.int32))
            if len(buf_in) == self.batch_size:
                yield np.stack(buf_in), np.stack(buf_tgt)
                buf_in, buf_tgt = [], []

    def skip_tokens(self, n: int) -> None:
        """Advance the train cursor by n tokens (for --resume)."""
        self._skip_tokens = n

    def __repr__(self) -> str:
        return (
            f"JAXBatcher(shard_dir={self.shard_dir}, "
            f"seq_len={self.seq_len}, batch_size={self.batch_size}, "
            f"train_shards={len(self._train_dataset._shards)}, "
            f"val_shard={self.val_shard})"
        )

    def _build_train_dataset(self) -> ShardedTokenDataset:
        return self._build_dataset(exclude_shards=[self.val_shard])

    def _build_val_dataset(self) -> ShardedTokenDataset:
        return self._build_dataset(include_shards=[self.val_shard])

    def _build_dataset(
        self,
        exclude_shards: list[int] | None = None,
        include_shards: list[int] | None = None,
    ) -> ShardedTokenDataset:
        ds = ShardedTokenDataset(
            shard_dir=self.shard_dir,
            seq_len=self.seq_len,
            shuffle_shards=False,
            progress=False,
        )
        all_shards = ds._shards
        if exclude_shards is not None:
            excluded_names = {f"shard_{i:05d}.npy" for i in exclude_shards}
            ds._shards = [p for p in all_shards if p.name not in excluded_names]
        elif include_shards is not None:
            included_names = {f"shard_{i:05d}.npy" for i in include_shards}
            ds._shards = [p for p in all_shards if p.name in included_names]
        return ds
