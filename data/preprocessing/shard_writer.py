"""
Shard writer — accumulates token IDs and flushes to uint16 .npy files.

Shard format:
    filename : shard_{index:05d}.npy
    dtype    : uint16  (IDs 0–32767 fit; saves 50% vs int32)
    shape    : (N,)    flat token array — training loop slices into windows
    size     : ~100MB per shard by default (50M tokens × 2 bytes)

Metadata (metadata.json written at end):
    {
      "dataset_version" : "v1-bpe32k-fineweb10BT",
      "vocab_size"      : 32768,
      "total_tokens"    : 10_234_567_890,
      "total_shards"    : 205,
      "shard_size"      : 50_000_000,   # tokens per shard (last may be smaller)
      "tokenizer_path"  : "tokenizer.json",
      "created_at"      : "2025-05-24T...",
      "shards"          : [
          {"index": 0, "filename": "shard_00000.npy", "tokens": 50000000},
          ...
      ]
    }

Dataset versioning rule:
    Never overwrite existing shards. Bump version (v2, v3) if tokenizer changes.
    Model DataConfig declares which version it consumes.

Usage:
    writer = ShardWriter(output_dir="data/shards/", shard_size=50_000_000)
    for token_ids in tokenize_stream:
        writer.add(token_ids)
    metadata = writer.finalize(dataset_version="v1-bpe32k-fineweb10BT")
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tqdm import tqdm


class ShardWriter:
    """
    Writes a flat uint16 token stream to numbered .npy shard files.

    Args:
        output_dir:  Directory to write shard files. Created if absent.
        shard_size:  Number of tokens per shard (default 50M → ~100MB).
        vocab_size:  Expected vocab size for validation (default 32_768).
        progress:    Show tqdm bar over total tokens written (default True).
        token_budget: Total tokens expected (sets tqdm total). None = unknown.
    """

    def __init__(
        self,
        output_dir:  str | Path,
        shard_size:  int = 50_000_000,
        vocab_size:  int = 32_768,
        progress:    bool = True,
        token_budget: int | None = None,
    ) -> None:
        self.output_dir  = Path(output_dir)
        self.shard_size  = shard_size
        self.vocab_size  = vocab_size
        self.progress    = progress
        self.token_budget = token_budget

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: list[int] = []
        self._shard_index:  int = 0
        self._total_tokens: int = 0
        self._shard_records: list[dict[str, Any]] = []

        # Guard against overwriting existing shards
        existing = sorted(self.output_dir.glob("shard_*.npy"))
        if existing:
            raise FileExistsError(
                f"Shards already exist in {self.output_dir}. "
                f"Increment dataset version — never overwrite shards.\n"
                f"First existing: {existing[0].name}"
            )

        # Progress bar — updates via update() per token, closes on finalize()
        self._bar: tqdm | None = None
        if self.progress:
            self._bar = tqdm(
                total=token_budget,
                desc="shards",
                unit="tok",
                mininterval=1.0,
                dynamic_ncols=True,
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, token_ids: list[int]) -> None:
        """
        Add token IDs for one document to the internal buffer.
        Flushes to disk automatically when buffer reaches shard_size.

        Args:
            token_ids: Token IDs for one document (incl. <|endoftext|>).
        """
        self._validate_ids(token_ids)
        self._buffer.extend(token_ids)

        while len(self._buffer) >= self.shard_size:
            shard_tokens = self._buffer[:self.shard_size]
            self._buffer  = self._buffer[self.shard_size:]
            self._flush(shard_tokens)

    def finalize(self, dataset_version: str = "v1") -> dict[str, Any]:
        """
        Flush remaining buffer (partial last shard) and write metadata.json.

        Args:
            dataset_version: Version string e.g. "v1-bpe32k-fineweb10BT".

        Returns:
            Metadata dict (also written to metadata.json).

        Must be called exactly once after all add() calls.
        """
        if self._buffer:
            self._flush(self._buffer)
            self._buffer = []

        if self._total_tokens == 0:
            raise ValueError("No tokens written — finalize() called on empty writer.")

        if self._bar is not None:
            self._bar.n = self._total_tokens
            self._bar.refresh()
            self._bar.close()
            self._bar = None

        metadata = self._build_metadata(dataset_version)
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"\n[shard_writer] finalized\n"
            f"  shards       : {self._shard_index}\n"
            f"  total tokens : {self._total_tokens:,}\n"
            f"  output_dir   : {self.output_dir}\n"
            f"  metadata     : {meta_path}"
        )
        return metadata

    @property
    def tokens_written(self) -> int:
        """Total tokens flushed to disk (excludes current buffer)."""
        return self._total_tokens - len(self._buffer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush(self, tokens: list[int]) -> None:
        """Write one shard to disk as uint16 .npy."""
        arr  = np.array(tokens, dtype=np.uint16)
        path = self.output_dir / f"shard_{self._shard_index:05d}.npy"
        np.save(path, arr)

        n = len(tokens)
        self._shard_records.append({
            "index":    self._shard_index,
            "filename": path.name,
            "tokens":   n,
            "size_mb":  round(path.stat().st_size / 1e6, 2),
        })
        self._total_tokens += n
        self._shard_index  += 1

        if self._bar is not None:
            self._bar.update(n)
            self._bar.set_postfix(
                shards=self._shard_index,
                last=f"{path.name}",
            )

    def _validate_ids(self, ids: list[int]) -> None:
        """Raise if any ID is out of uint16 range."""
        if not ids:
            return
        lo, hi = min(ids), max(ids)
        if lo < 0 or hi >= self.vocab_size:
            bad = [i for i in ids if i < 0 or i >= self.vocab_size]
            raise ValueError(
                f"Token IDs out of range [0, {self.vocab_size}): "
                f"min={lo} max={hi}  bad_sample={bad[:5]}"
            )

    def _build_metadata(self, dataset_version: str) -> dict[str, Any]:
        return {
            "dataset_version": dataset_version,
            "vocab_size":      self.vocab_size,
            "total_tokens":    self._total_tokens,
            "total_shards":    self._shard_index,
            "shard_size":      self.shard_size,
            "created_at":      datetime.now(timezone.utc).isoformat(),
            "shards":          self._shard_records,
        }
