"""
BPE Encoder — text → token ID sequence.

Algorithm:
  1. Split on special tokens first (preserve their IDs).
  2. Pre-tokenize each text chunk with GPT-2 regex.
  3. Convert each pre-token to byte IDs (base vocab IDs 5–260).
  4. Greedily apply the lowest-rank BPE merge repeatedly until no
     applicable pair remains (same merge order as training).
  5. Concatenate all token ID lists.

ID layout (must match bpe.py):
    0–4     : special tokens
    5–260   : raw bytes
    261+    : BPE merge tokens

Usage:
    from tokenizer.trainers.bpe import BPETrainer
    from tokenizer.runtime.encode import BPEEncoder

    trainer = BPETrainer(vocab_size=32_768)
    trainer.train(texts)

    encoder = BPEEncoder.from_trainer(trainer)
    ids = encoder.encode("Hello, world!")
    ids = encoder.encode("doc1<|endoftext|>doc2", add_special_tokens=True)
"""

from __future__ import annotations

import re
from typing import Sequence

from tokenizer.trainers.bpe import (
    SPECIAL_TOKENS,
    _BYTE_OFFSET,
    _MERGE_START,
    _GPT2_PAT,
)

# ---------------------------------------------------------------------------
# Special-token splitter
# ---------------------------------------------------------------------------

def _build_special_pat(special_tokens: dict[str, int]) -> re.Pattern[str]:
    """Regex that splits text on any special token string (longest match first)."""
    tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    escaped = [re.escape(t) for t in tokens_sorted]
    return re.compile("(" + "|".join(escaped) + ")")


# ---------------------------------------------------------------------------
# Core word encoder (the hot path)
# ---------------------------------------------------------------------------

def _encode_word(word: str, merges_rank: dict[tuple[int, int], int]) -> list[int]:
    """
    Encode a single pre-token (no spaces, punctuation chunk, etc.) to token IDs.

    Steps:
      - Convert UTF-8 bytes to base token IDs.
      - Repeatedly find the pair with the lowest merge rank and apply it
        everywhere in the sequence, until no known pair remains.

    Args:
        word: Single pre-token string (output of GPT-2 regex).
        merges_rank: {(a, b): rank} where rank = training order index.

    Returns:
        List of token IDs.
    """
    # Base: each byte → its token ID
    ids: list[int] = [_BYTE_OFFSET + b for b in word.encode("utf-8")]

    while len(ids) >= 2:
        # Find the adjacent pair with the lowest merge rank
        best_rank = float("inf")
        best_pair: tuple[int, int] | None = None

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            rank = merges_rank.get(pair, float("inf"))
            if rank < best_rank:
                best_rank = rank
                best_pair = pair

        if best_pair is None or best_rank == float("inf"):
            break  # No applicable merges left

        new_id = _MERGE_START + best_rank  # merge rank == its offset from _MERGE_START

        # Apply merge — replace ALL occurrences of best_pair left-to-right
        new_ids: list[int] = []
        i = 0
        a, b = best_pair
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids

    return ids


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class BPEEncoder:
    """
    Encodes text strings to BPE token ID sequences.

    Args:
        merges: Ordered merge list from BPETrainer (list of (a, b) pairs).
        special_tokens: {token_str: token_id} mapping. Defaults to the
                        locked special tokens defined in bpe.py.
    """

    def __init__(
        self,
        merges: list[tuple[int, int]],
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.special_tokens: dict[str, int] = (
            special_tokens if special_tokens is not None else dict(SPECIAL_TOKENS)
        )
        # rank lookup: (a, b) → index in merges list
        self.merges_rank: dict[tuple[int, int], int] = {
            pair: i for i, pair in enumerate(merges)
        }
        self._special_pat = _build_special_pat(self.special_tokens)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_trainer(cls, trainer: object) -> "BPEEncoder":
        """Build encoder directly from a trained BPETrainer instance."""
        return cls(
            merges=trainer.merges,  # type: ignore[attr-defined]
            special_tokens=trainer.special_tokens if hasattr(trainer, "special_tokens")
                           else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Encode a single text string to a list of token IDs.

        Args:
            text: Raw input string. May contain special tokens as literal text.
            add_bos: Prepend <|bos|> token (ID 3).
            add_eos: Append  <|eos|> token (ID 4).

        Returns:
            List of int token IDs (all values 0–32767).
        """
        ids: list[int] = []

        if add_bos:
            ids.append(self.special_tokens["<|bos|>"])

        # Split on special tokens, preserving them as capture groups.
        chunks = self._special_pat.split(text)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                for word in _GPT2_PAT.findall(chunk):
                    ids.extend(_encode_word(word, self.merges_rank))

        if add_eos:
            ids.append(self.special_tokens["<|eos|>"])

        return ids

    def encode_batch(
        self,
        texts: Sequence[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        """
        Encode a list of strings. Returns a list of token ID lists.
        No padding — sequences may have different lengths.

        For padded tensors, pad to max length with <|pad|> (ID 1).
        """
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def encode_as_uint16(self, text: str, **kwargs: bool) -> list[int]:
        """
        Encode and assert all IDs fit in uint16 (0–32767).
        Use this before writing .npy shards.
        """
        ids = self.encode(text, **kwargs)
        if any(i < 0 or i > 32_767 for i in ids):
            bad = [i for i in ids if i < 0 or i > 32_767]
            raise ValueError(f"Token IDs out of uint16 range: {bad[:5]}")
        return ids
