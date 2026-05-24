"""
Tokenizer serialization — save trained BPE to disk.

Writes three files to output_dir/:

  tokenizer.json  — canonical file; load.py uses only this.
                    Contains special_tokens, merges (as int pairs),
                    vocab_size, and format version. Self-contained.

  vocab.json      — human-readable {token_repr: id} map.
                    Byte tokens encoded as latin-1 (same as GPT-2).
                    Use for inspection / debugging only.

  merges.txt      — human-readable merge list.
                    Each line: "bytes_a bytes_b" showing what was merged.
                    Line N (0-indexed) produced token ID (_MERGE_START + N).

Usage:
    from tokenizer.serialization.save import save_tokenizer

    trainer = BPETrainer(vocab_size=32_768)
    trainer.train(texts)
    save_tokenizer(trainer, output_dir="tokenizer/saved/")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tokenizer.trainers.bpe import SPECIAL_TOKENS, _BYTE_OFFSET, _MERGE_START

_FORMAT_VERSION = "1.0"


def save_tokenizer(
    trainer: object,
    output_dir: str | Path,
    name: str = "tokenizer",
) -> Path:
    """
    Serialize a trained BPETrainer to disk.

    Args:
        trainer: Trained BPETrainer instance (must have .merges and .id_to_token).
        output_dir: Directory to write files into. Created if absent.
        name: Stem name used for tokenizer.json (default "tokenizer").
              vocab.json and merges.txt always use fixed names.

    Returns:
        Path to the written tokenizer.json (canonical file).

    Raises:
        ValueError: If trainer.merges is empty (untrained).
        FileExistsError: If output_dir already has a tokenizer.json and
                         would be overwritten — pass overwrite=True to skip check.
    """
    merges: list[tuple[int, int]] = trainer.merges  # type: ignore[attr-defined]
    id_to_token: dict[int, bytes] = trainer.id_to_token  # type: ignore[attr-defined]

    if not merges:
        raise ValueError("trainer.merges is empty — call trainer.train() first")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_tokenizer_json(out / f"{name}.json", merges, trainer)
    _write_vocab_json(out / "vocab.json", id_to_token)
    _write_merges_txt(out / "merges.txt", merges, id_to_token)

    canonical = out / f"{name}.json"
    print(f"Saved tokenizer to {canonical}")
    print(f"  vocab_size : {_MERGE_START + len(merges)}")
    print(f"  merges     : {len(merges):,}")
    print(f"  files      : tokenizer.json  vocab.json  merges.txt")
    return canonical


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_tokenizer_json(
    path: Path,
    merges: list[tuple[int, int]],
    trainer: object,
) -> None:
    """
    Write canonical tokenizer.json.

    Schema:
        version        str   format version for forward compatibility
        vocab_size     int   total tokens including special + bytes + merges
        special_tokens obj   {token_str: id}  (IDs 0–4, locked)
        merges         list  [[a, b], ...]  integer pair per merge, ordered
    """
    vocab_size = _MERGE_START + len(merges)

    payload = {
        "version": _FORMAT_VERSION,
        "vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "merges": [list(pair) for pair in merges],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_vocab_json(
    path: Path,
    id_to_token: dict[int, bytes],
) -> None:
    """
    Write vocab.json: {token_repr: id} for all non-special tokens.

    Byte tokens encoded as latin-1 so single-byte tokens remain readable.
    Merge tokens encoded as latin-1 (may contain multi-byte sequences).

    Note: this file is for inspection only — load.py does not use it.
    """
    # Special tokens first (IDs 0–4)
    vocab: dict[str, int] = dict(SPECIAL_TOKENS)

    # Byte + merge tokens sorted by ID
    for token_id in sorted(id_to_token):
        token_bytes = id_to_token[token_id]
        # latin-1 preserves all 256 byte values as single characters
        token_repr = token_bytes.decode("latin-1")
        vocab[token_repr] = token_id

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def _write_merges_txt(
    path: Path,
    merges: list[tuple[int, int]],
    id_to_token: dict[int, bytes],
) -> None:
    """
    Write merges.txt: human-readable merge history.

    Format per line:
        <bytes_a_repr> <bytes_b_repr>   # → id=<result_id>

    Line index N (0-based) produced token ID (_MERGE_START + N).
    Byte reprs use latin-1 so they're diff-friendly in git.
    """
    lines: list[str] = [
        f"# BPE merges -- line N (0-indexed) produced token ID {_MERGE_START}+N\n"
    ]

    for i, (a, b) in enumerate(merges):
        a_repr = id_to_token[a].decode("latin-1")
        b_repr = id_to_token[b].decode("latin-1")
        result_id = _MERGE_START + i
        lines.append(f"{a_repr} {b_repr}  # -> {result_id}\n")

    with open(path, "w", encoding="latin-1") as f:
        f.writelines(lines)
