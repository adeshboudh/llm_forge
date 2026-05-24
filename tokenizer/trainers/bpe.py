"""
BPE Tokenizer Trainer.

Byte-level BPE (GPT-2 style):
  - Pre-tokenize with regex, convert to UTF-8 bytes
  - Base vocab: 256 raw bytes → IDs 5–260
  - Learn merges until vocab_size reached → IDs 261–32767
  - Special tokens occupy IDs 0–4 (locked forever)

Vocab layout (vocab_size=32_768):
    0–4     : special tokens  (<|endoftext|>, <|pad|>, <|unk|>, <|bos|>, <|eos|>)
    5–260   : raw bytes (0x00–0xFF)
    261–32767 : learned BPE merges (32,507 merges)

Usage:
    from tokenizer.trainers.bpe import BPETrainer

    trainer = BPETrainer(vocab_size=32_768)
    trainer.train(iter(["Hello world", "foo bar baz", ...]))
    trainer.save("tokenizer/")
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GPT-2 pre-tokenization pattern — splits on contractions, words, digits,
# punctuation, and whitespace before running BPE.
_GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
)

# Special token IDs — LOCKED. Never reassign after first tokenization.
SPECIAL_TOKENS: dict[str, int] = {
    "<|endoftext|>": 0,
    "<|pad|>":        1,
    "<|unk|>":        2,
    "<|bos|>":        3,
    "<|eos|>":        4,
}

_N_SPECIAL   = len(SPECIAL_TOKENS)   # 5
_N_BYTES     = 256
_BYTE_OFFSET = _N_SPECIAL            # byte b → ID (5 + b)
_MERGE_START = _N_SPECIAL + _N_BYTES # 261 — first merge token ID


# ---------------------------------------------------------------------------
# Core BPE helpers (module-level for clarity)
# ---------------------------------------------------------------------------

def _pretokenize(text: str) -> list[str]:
    """Split text into pre-tokens with GPT-2 regex."""
    return _GPT2_PAT.findall(text)


def _get_pair_counts(
    vocab: dict[tuple[int, ...], int],
) -> dict[tuple[int, int], int]:
    """Count adjacent pair frequencies across all words in vocab."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for word, freq in vocab.items():
        for a, b in zip(word, word[1:]):
            counts[(a, b)] += freq
    return counts


def _merge_pair(
    vocab: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
) -> dict[tuple[int, ...], int]:
    """Replace every occurrence of pair in vocab with new_id."""
    a, b = pair
    new_vocab: dict[tuple[int, ...], int] = {}
    for word, freq in vocab.items():
        merged: list[int] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                merged.append(new_id)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        new_vocab[tuple(merged)] = freq
    return new_vocab


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BPETrainer:
    """
    Trains a byte-level BPE tokenizer from raw text.

    Args:
        vocab_size: Target vocab size including special tokens.
                    Max 32_768 (uint16-safe). Default: 32_768.
        log_every:  Print merge progress every N merges. 0 = silent.
    """

    def __init__(self, vocab_size: int = 32_768, log_every: int = 1_000) -> None:
        if vocab_size > 32_768:
            raise ValueError(
                f"vocab_size={vocab_size} exceeds uint16 max 32_768"
            )
        if vocab_size < _MERGE_START + 1:
            raise ValueError(
                f"vocab_size must be > {_MERGE_START} "
                f"(5 special + 256 bytes + at least 1 merge)"
            )

        self.vocab_size = vocab_size
        self.log_every  = log_every
        self.n_merges   = vocab_size - _MERGE_START  # 32_768 - 261 = 32_507

        # Populated after train()
        self.merges: list[tuple[int, int]] = []
        self.token_to_id: dict[bytes, int] = {}
        self.id_to_token: dict[int, bytes] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, texts: Iterator[str]) -> None:
        """
        Train BPE on an iterator of raw text strings.

        Args:
            texts: Yields one document / line at a time.
                   Stream from dataset — no need to load all into RAM.
        """
        print("Building word frequency table...")
        word_freq = self._count_word_freq(texts)
        print(f"  {len(word_freq):,} unique pre-tokens found")

        # Represent each word as tuple of token IDs.
        # Initially every byte b maps to ID (_BYTE_OFFSET + b).
        byte_vocab: dict[tuple[int, ...], int] = {
            tuple(_BYTE_OFFSET + b for b in word.encode("utf-8")): freq
            for word, freq in word_freq.items()
        }

        # Pre-populate id_to_token with base byte tokens so merge lookups
        # never need the fallback bytes([id - _BYTE_OFFSET]) hack.
        for b in range(_N_BYTES):
            self.id_to_token[_BYTE_OFFSET + b] = bytes([b])

        print(f"Learning {self.n_merges:,} merges...")
        next_id = _MERGE_START

        for i in range(self.n_merges):
            counts = _get_pair_counts(byte_vocab)

            if not counts:
                print(f"  No pairs left at merge {i}. Stopping early.")
                break

            best_pair  = max(counts, key=lambda p: counts[p])
            best_count = counts[best_pair]

            if best_count < 2:
                print(f"  All pairs freq < 2 at merge {i}. Stopping early.")
                break

            self.merges.append(best_pair)
            byte_vocab = _merge_pair(byte_vocab, best_pair, next_id)
            next_id += 1

            # Update id_to_token incrementally (id_to_token has all byte tokens
            # pre-populated, so both constituents are always present here).
            merged_bytes = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]
            self.id_to_token[next_id - 1] = merged_bytes

            if self.log_every and (i + 1) % self.log_every == 0:
                print(
                    f"  [{i+1:>6,}/{self.n_merges:,}]  "
                    f"{self.id_to_token[best_pair[0]]!r} + {self.id_to_token[best_pair[1]]!r}"
                    f"  freq={best_count:,}  → id={next_id - 1}"
                )

        print(f"Training complete. {len(self.merges):,} merges learned.")
        self._build_lookup_tables()

    def get_vocab(self) -> dict[str, int]:
        """
        Return full {token_repr: id} vocab dict for serialization.
        Special tokens included. Byte tokens use latin-1 repr.
        """
        vocab: dict[str, int] = {}
        vocab.update(SPECIAL_TOKENS)
        for token_bytes, token_id in self.token_to_id.items():
            vocab[token_bytes.decode("latin-1")] = token_id
        return vocab

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _count_word_freq(self, texts: Iterator[str]) -> dict[str, int]:
        """Stream texts and count pre-token frequencies."""
        freq: dict[str, int] = defaultdict(int)
        n = 0
        for text in texts:
            for word in _pretokenize(text):
                freq[word] += 1
            n += 1
            if n % 50_000 == 0:
                print(f"  {n:,} texts processed, {len(freq):,} unique pre-tokens")
        return dict(freq)

    def _build_lookup_tables(self) -> None:
        """Build token_to_id and id_to_token from base bytes + merges."""
        self.token_to_id = {}
        self.id_to_token = {}

        # Base byte tokens
        for b in range(_N_BYTES):
            token_bytes = bytes([b])
            token_id    = _BYTE_OFFSET + b
            self.token_to_id[token_bytes] = token_id
            self.id_to_token[token_id]    = token_bytes

        # Merge tokens (id_to_token already partially built during training)
        for i, (a, b) in enumerate(self.merges):
            token_id     = _MERGE_START + i
            merged_bytes = self.id_to_token[a] + self.id_to_token[b]
            self.token_to_id[merged_bytes] = token_id
            self.id_to_token[token_id]     = merged_bytes
