"""
BPE Decoder — token ID sequence → text.

Algorithm:
  1. For each token ID, look up its bytes in id_to_token.
  2. Special tokens (IDs 0–4): either skip or emit their string repr.
  3. Concatenate all byte chunks.
  4. Decode as UTF-8 (with configurable error handling).

Byte-level BPE guarantees valid UTF-8 on decode because:
  - Pre-tokenizer never splits mid-codepoint.
  - Every merge joins adjacent bytes from the same pre-token.
  - So concatenating token bytes always yields valid UTF-8 sequences.

Usage:
    from tokenizer.runtime.decode import BPEDecoder

    decoder = BPEDecoder.from_trainer(trainer)
    text = decoder.decode([264, 269])
    text = decoder.decode(ids, skip_special_tokens=False)  # keep <|endoftext|>
"""

from __future__ import annotations

from tokenizer.trainers.bpe import (
    SPECIAL_TOKENS,
    _BYTE_OFFSET,
    _N_BYTES,
)


class BPEDecoder:
    """
    Decodes BPE token ID sequences back to text strings.

    Args:
        id_to_token: {token_id: bytes} mapping for all non-special tokens.
                     Must cover IDs 5–260 (bytes) and 261+ (merges).
        special_tokens: {token_str: token_id}. Defaults to locked special tokens.
    """

    def __init__(
        self,
        id_to_token: dict[int, bytes],
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.id_to_token: dict[int, bytes] = id_to_token
        self.special_tokens: dict[str, int] = (
            special_tokens if special_tokens is not None else dict(SPECIAL_TOKENS)
        )
        # Reverse: token_id → token_str (for special tokens only)
        self._id_to_special: dict[int, str] = {
            v: k for k, v in self.special_tokens.items()
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_trainer(cls, trainer: object) -> "BPEDecoder":
        """Build decoder directly from a trained BPETrainer instance."""
        return cls(
            id_to_token=trainer.id_to_token,  # type: ignore[attr-defined]
            special_tokens=getattr(trainer, "special_tokens", None),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> str:
        """
        Decode a list of token IDs to a text string.

        Args:
            ids: Token ID list (e.g. output of BPEEncoder.encode).
            skip_special_tokens: If True (default), special tokens (IDs 0–4)
                are silently dropped. If False, their string repr is emitted
                (e.g. '<|endoftext|>').
            errors: UTF-8 decode error handling — 'replace', 'ignore', 'strict'.
                    Default 'replace' substitutes invalid bytes with U+FFFD.

        Returns:
            Decoded string.
        """
        byte_chunks: list[bytes] = []

        for token_id in ids:
            if token_id in self._id_to_special:
                if not skip_special_tokens:
                    # Emit special token as its string repr encoded to UTF-8
                    byte_chunks.append(self._id_to_special[token_id].encode("utf-8"))
                # skip_special_tokens=True → just continue
                continue

            token_bytes = self.id_to_token.get(token_id)
            if token_bytes is None:
                # Unknown ID — emit replacement character bytes
                byte_chunks.append("?".encode("utf-8"))
                continue

            byte_chunks.append(token_bytes)

        return b"".join(byte_chunks).decode("utf-8", errors=errors)

    def decode_batch(
        self,
        batch: list[list[int]],
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> list[str]:
        """Decode a batch of token ID lists. Returns list of strings."""
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens, errors=errors)
            for ids in batch
        ]

    def decode_single_token(self, token_id: int) -> bytes:
        """
        Return the raw bytes for a single token ID.
        Useful for streaming / incremental decode.
        Special tokens return their UTF-8 encoded string repr.
        """
        if token_id in self._id_to_special:
            return self._id_to_special[token_id].encode("utf-8")
        return self.id_to_token.get(token_id, b"?")
