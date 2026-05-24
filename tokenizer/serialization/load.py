"""
Tokenizer deserialization — load saved BPE from disk.

Reads tokenizer.json (the canonical file written by save.py) and
reconstructs a ready-to-use Tokenizer with both encoder and decoder.
Does NOT require the BPETrainer — only the saved JSON.

Usage:
    from tokenizer.serialization.load import load_tokenizer

    tok = load_tokenizer("tokenizer/saved/tokenizer.json")
    ids  = tok.encode("Hello, world!")
    text = tok.decode(ids)

    # Or use encoder / decoder separately
    ids  = tok.encoder.encode(text, add_bos=True)
    text = tok.decoder.decode(ids, skip_special_tokens=False)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenizer.trainers.bpe import _BYTE_OFFSET, _MERGE_START, _N_BYTES
from tokenizer.runtime.encode import BPEEncoder
from tokenizer.runtime.decode import BPEDecoder

_SUPPORTED_VERSIONS = {"1.0"}


# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------

@dataclass
class Tokenizer:
    """
    Thin wrapper around BPEEncoder + BPEDecoder loaded from disk.

    Attributes:
        encoder: BPEEncoder — use for text → token IDs.
        decoder: BPEDecoder — use for token IDs → text.
        vocab_size: int — total vocabulary size.
        special_tokens: dict[str, int] — special token map.
    """

    encoder: BPEEncoder
    decoder: BPEDecoder
    vocab_size: int
    special_tokens: dict[str, int]

    # Convenience pass-throughs so callers don't need to unwrap

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to token IDs."""
        return self.encoder.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.decoder.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: list[str], **kwargs: bool) -> list[list[int]]:
        return self.encoder.encode_batch(texts, **kwargs)

    def decode_batch(self, batch: list[list[int]], **kwargs: bool) -> list[str]:
        return self.decoder.decode_batch(batch, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"merges={len(self.encoder.merges_rank)}, "
            f"special_tokens={list(self.special_tokens)})"
        )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_tokenizer(path: str | Path) -> Tokenizer:
    """
    Load a saved tokenizer from tokenizer.json.

    Args:
        path: Path to tokenizer.json written by save_tokenizer().

    Returns:
        Tokenizer with .encode() and .decode() ready to use.

    Raises:
        FileNotFoundError: tokenizer.json missing.
        ValueError: JSON malformed, unknown version, or corrupt merges.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    _validate(data, path)

    special_tokens: dict[str, int] = data["special_tokens"]
    raw_merges: list[list[int]]    = data["merges"]
    vocab_size: int                = data["vocab_size"]

    merges: list[tuple[int, int]] = [tuple(pair) for pair in raw_merges]  # type: ignore[misc]
    id_to_token = _rebuild_id_to_token(merges)

    encoder = BPEEncoder(merges=merges, special_tokens=special_tokens)
    decoder = BPEDecoder(id_to_token=id_to_token, special_tokens=special_tokens)

    return Tokenizer(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(data: dict, path: Path) -> None:
    """Raise ValueError on any schema problem."""
    required = {"version", "vocab_size", "special_tokens", "merges"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"tokenizer.json missing fields: {missing}  ({path})")

    version = data["version"]
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported tokenizer version {version!r}. "
            f"Supported: {_SUPPORTED_VERSIONS}"
        )

    if not isinstance(data["merges"], list):
        raise ValueError("'merges' must be a list")

    if data["vocab_size"] != _MERGE_START + len(data["merges"]):
        raise ValueError(
            f"vocab_size mismatch: got {data['vocab_size']}, "
            f"expected {_MERGE_START + len(data['merges'])} "
            f"(261 base + {len(data['merges'])} merges)"
        )


def _rebuild_id_to_token(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """
    Reconstruct id_to_token from merges list.

    Mirrors BPETrainer._build_lookup_tables() — kept here so load.py
    has zero dependency on BPETrainer at runtime.

    Layout:
        _BYTE_OFFSET + b  (5–260)  → bytes([b])   for b in 0..255
        _MERGE_START + i  (261+)   → constituent bytes concatenated
    """
    id_to_token: dict[int, bytes] = {}

    # Base byte tokens
    for b in range(_N_BYTES):
        id_to_token[_BYTE_OFFSET + b] = bytes([b])

    # Merge tokens — must be built in order (each may depend on prior merges)
    for i, (a, b) in enumerate(merges):
        token_id = _MERGE_START + i
        try:
            merged = id_to_token[a] + id_to_token[b]
        except KeyError as exc:
            raise ValueError(
                f"Corrupt merges: merge {i} references unknown token ID {exc}. "
                f"Merges must be stored in training order."
            ) from exc
        id_to_token[token_id] = merged

    return id_to_token
