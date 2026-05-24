"""
PyTorch Dataset for loading uint16 .npy token shards at training time.

Loads shards lazily — one shard in memory at a time. Slices fixed-length
windows from the flat token array. No padding — every token is real.

Usage:
    from data.loaders.npy_loader import ShardedTokenDataset

    dataset = ShardedTokenDataset(
        shard_dir="data/shards/",
        seq_len=1024,
        shuffle_shards=True,   # shuffle shard order each epoch
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=4)

    for input_ids, target_ids in loader:
        # input_ids  : (B, seq_len)  uint16 → cast to long in model
        # target_ids : (B, seq_len)  = input_ids shifted by 1
        loss = model(input_ids, labels=target_ids)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


class ShardedTokenDataset:
    """
    Iterable dataset over uint16 .npy shards. Yields (input, target) pairs.

    Each item: a (seq_len,) window from the flat token stream.
    Target = input shifted left by 1 (standard next-token prediction).

    Compatible with torch.utils.data.DataLoader.

    Args:
        shard_dir:      Directory containing shard_*.npy + metadata.json.
        seq_len:        Sequence length for training (e.g. 1024, 2048).
        shuffle_shards: Shuffle shard order each epoch (default True).
        stride:         Step between windows (default = seq_len, no overlap).
                        stride < seq_len → overlapping windows (more data,
                        but correlated — not recommended for pretraining).
    """

    def __init__(
        self,
        shard_dir:      str | Path,
        seq_len:        int = 1024,
        shuffle_shards: bool = True,
        stride:         int | None = None,
    ) -> None:
        self.shard_dir      = Path(shard_dir)
        self.seq_len        = seq_len
        self.shuffle_shards = shuffle_shards
        self.stride         = stride if stride is not None else seq_len

        self._shards, self._metadata = self._load_metadata()
        self._total_sequences = self._estimate_total_sequences()

    # ------------------------------------------------------------------
    # Iterable protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        """Iterate through all shards, yielding (input, target) pairs."""
        shard_paths = list(self._shards)
        if self.shuffle_shards:
            random.shuffle(shard_paths)

        for shard_path in shard_paths:
            tokens = np.load(shard_path).astype(np.int64)  # uint16 → int64 for loss

            # Slice non-overlapping windows of seq_len + 1 (need target too)
            window = self.seq_len + 1
            for start in range(0, len(tokens) - window + 1, self.stride):
                chunk = tokens[start : start + window]
                yield chunk[:-1], chunk[1:]  # input, target

    def __len__(self) -> int:
        """Approximate total sequences across all shards."""
        return self._total_sequences

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def total_tokens(self) -> int:
        return self._metadata.get("total_tokens", 0)

    @property
    def vocab_size(self) -> int:
        return self._metadata.get("vocab_size", 32_768)

    def __repr__(self) -> str:
        return (
            f"ShardedTokenDataset("
            f"shards={len(self._shards)}, "
            f"total_tokens={self.total_tokens:,}, "
            f"seq_len={self.seq_len}, "
            f"~sequences={self._total_sequences:,})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_metadata(self) -> tuple[list[Path], dict[str, Any]]:
        meta_path = self.shard_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.shard_dir}. "
                f"Run the data pipeline first."
            )

        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        shard_paths = sorted(self.shard_dir.glob("shard_*.npy"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard_*.npy files found in {self.shard_dir}."
            )

        expected = metadata.get("total_shards", len(shard_paths))
        if len(shard_paths) != expected:
            raise ValueError(
                f"metadata.json says {expected} shards, "
                f"found {len(shard_paths)} on disk. "
                f"Dataset may be incomplete."
            )

        return shard_paths, metadata

    def _estimate_total_sequences(self) -> int:
        """Total non-overlapping seq_len windows across all shards."""
        total_tokens = self._metadata.get("total_tokens", 0)
        return max(0, (total_tokens - 1) // self.stride)
