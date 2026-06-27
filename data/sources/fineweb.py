"""
FineWeb-Edu document streamer.

Streams raw text documents from HuggingFace, yielding one document at a time.
No tokenization here — pure text out. Caller decides encoding.

Dataset: HuggingFaceFW/fineweb-edu  config: sample-10BT
Text field: "text"
Total: ~10B tokens (pre-tokenized estimate), ~38GB compressed

Usage:
    from data.sources.fineweb import FineWebEduStream

    stream = FineWebEduStream(token_budget=1_000_000_000)
    for doc in stream:
        print(doc.text[:100], doc.char_count)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from tqdm import tqdm


@dataclass(frozen=True)
class Document:
    text: str
    char_count: int
    doc_id: int        # sequential index within this stream session


class FineWebEduStream:
    """
    Streams FineWeb-Edu documents from HuggingFace.

    Stops when token_budget (measured in chars) is exhausted.
    Uses HuggingFace streaming — no full download required.

    Args:
        token_budget: Stop after this many characters (proxy for tokens).
                      1B chars ≈ 250M tokens (rough estimate).
                      For actual token budgets see pipeline.py.
        dataset:      HuggingFace dataset path.
        config:       Dataset config / subset.
        split:        Dataset split (default "train").
        min_doc_len:  Skip documents shorter than this many chars.
        log_every:    Log progress every N documents. 0 = silent.
    """

    DATASET   = "HuggingFaceFW/fineweb-edu"
    CONFIG    = "sample-10BT"
    SPLIT     = "train"
    TEXT_FIELD = "text"

    def __init__(
        self,
        token_budget:  int = 10_000_000_000,   # 10B chars ≈ full dataset
        dataset:       str | None = None,
        config:        str | None = None,
        split:         str = "train",
        min_doc_len:   int = 100,
        log_every:     int = 100_000,
        progress:      bool = True,
    ) -> None:
        self.token_budget = token_budget
        self.dataset      = dataset or self.DATASET
        self.config       = config  or self.CONFIG
        self.split        = split
        self.min_doc_len  = min_doc_len
        self.log_every    = log_every
        self.progress     = progress

    def __iter__(self) -> Iterator[Document]:
        """Yield Document objects until token_budget exhausted."""
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError(
                "'datasets' not installed. Run: pip install datasets"
            ) from err

        ds = load_dataset(
            self.dataset,
            self.config,
            split=self.split,
            streaming=True,
        )

        chars_seen = 0
        docs_seen  = 0
        skipped    = 0
        t0         = time.time()

        # Total dataset size is unknown with streaming — use chars_budget.
        # Wrap the source row iterator; bar updates per yielded doc.
        row_iter = ds
        if self.progress:
            row_iter = tqdm(
                ds,
                desc="fineweb",
                unit="doc",
                mininterval=1.0,
                dynamic_ncols=True,
            )

        for row in row_iter:
            text: str = row.get(self.TEXT_FIELD, "") or ""

            if len(text) < self.min_doc_len:
                skipped += 1
                continue

            yield Document(text=text, char_count=len(text), doc_id=docs_seen)

            chars_seen += len(text)
            docs_seen  += 1

            if self.progress and self.log_every and docs_seen % self.log_every == 0:
                elapsed = time.time() - t0
                rate    = chars_seen / elapsed / 1e6  # MB/s
                pct     = 100 * chars_seen / self.token_budget
                if isinstance(row_iter, tqdm):
                    row_iter.set_postfix(
                        chars=f"{chars_seen / 1e9:.3f}B",
                        pct=f"{pct:.1f}%",
                        mb_s=f"{rate:.1f}",
                    )

            if chars_seen >= self.token_budget:
                break

        if self.progress and isinstance(row_iter, tqdm):
            row_iter.close()

        elapsed = time.time() - t0
        print(
            f"  [fineweb] done: {docs_seen:,} docs  "
            f"{chars_seen / 1e9:.3f}B chars  "
            f"{skipped:,} skipped  "
            f"{elapsed:.1f}s"
        )
