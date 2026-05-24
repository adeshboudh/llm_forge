"""
Tokenizer test suite — BPETrainer, BPEEncoder, BPEDecoder, save/load.

Run:
    pytest tokenizer/tests/test_bpe.py -v
    pytest tokenizer/tests/test_bpe.py::test_roundtrip_unicode -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tokenizer.trainers.bpe import (
    BPETrainer,
    SPECIAL_TOKENS,
    _BYTE_OFFSET,
    _MERGE_START,
    _N_BYTES,
)
from tokenizer.runtime.encode import BPEEncoder
from tokenizer.runtime.decode import BPEDecoder
from tokenizer.serialization.save import save_tokenizer
from tokenizer.serialization.load import load_tokenizer


# ---------------------------------------------------------------------------
# Shared fixtures — trained once per session, reused across tests
# ---------------------------------------------------------------------------

_CORPUS = (
    ["hello world", "hello there", "world peace", "foo bar baz",
     "the quick brown fox", "jumps over the lazy dog"] * 300
)
_VOCAB_SIZE = 400  # small enough for fast tests, big enough for real merges


@pytest.fixture(scope="module")
def trainer() -> BPETrainer:
    t = BPETrainer(vocab_size=_VOCAB_SIZE, log_every=0)
    t.train(iter(_CORPUS))
    return t


@pytest.fixture(scope="module")
def encoder(trainer: BPETrainer) -> BPEEncoder:
    return BPEEncoder.from_trainer(trainer)


@pytest.fixture(scope="module")
def decoder(trainer: BPETrainer) -> BPEDecoder:
    return BPEDecoder.from_trainer(trainer)


@pytest.fixture(scope="module")
def saved_tok(trainer: BPETrainer):
    """Save to temp dir and load back — shared across serialization tests."""
    with tempfile.TemporaryDirectory() as tmp:
        out = save_tokenizer(trainer, tmp)
        tok = load_tokenizer(out)
        yield tok, tmp, out


# ---------------------------------------------------------------------------
# BPETrainer
# ---------------------------------------------------------------------------

class TestBPETrainer:

    def test_vocab_size_too_large(self):
        with pytest.raises(ValueError, match="uint16"):
            BPETrainer(vocab_size=32_769)

    def test_vocab_size_too_small(self):
        with pytest.raises(ValueError, match="vocab_size must be"):
            BPETrainer(vocab_size=10)

    def test_special_token_ids_locked(self):
        assert SPECIAL_TOKENS["<|endoftext|>"] == 0
        assert SPECIAL_TOKENS["<|pad|>"]       == 1
        assert SPECIAL_TOKENS["<|unk|>"]       == 2
        assert SPECIAL_TOKENS["<|bos|>"]       == 3
        assert SPECIAL_TOKENS["<|eos|>"]       == 4

    def test_byte_offset(self):
        assert _BYTE_OFFSET == 5          # first byte token ID
        assert _BYTE_OFFSET + 255 == 260  # last byte token ID

    def test_merge_start(self):
        assert _MERGE_START == 261        # 5 special + 256 bytes

    def test_train_produces_merges(self, trainer: BPETrainer):
        assert len(trainer.merges) > 0

    def test_merge_count(self, trainer: BPETrainer):
        expected = _VOCAB_SIZE - _MERGE_START
        # May be less if corpus exhausted early
        assert len(trainer.merges) <= expected

    def test_id_to_token_covers_bytes(self, trainer: BPETrainer):
        for b in range(_N_BYTES):
            token_id = _BYTE_OFFSET + b
            assert token_id in trainer.id_to_token
            assert trainer.id_to_token[token_id] == bytes([b])

    def test_merge_tokens_in_id_to_token(self, trainer: BPETrainer):
        for i in range(len(trainer.merges)):
            token_id = _MERGE_START + i
            assert token_id in trainer.id_to_token
            assert isinstance(trainer.id_to_token[token_id], bytes)
            assert len(trainer.id_to_token[token_id]) >= 2  # always a merge

    def test_get_vocab_size(self, trainer: BPETrainer):
        vocab = trainer.get_vocab()
        # special + byte + merge tokens
        expected = len(SPECIAL_TOKENS) + _N_BYTES + len(trainer.merges)
        assert len(vocab) == expected

    def test_get_vocab_special_tokens(self, trainer: BPETrainer):
        vocab = trainer.get_vocab()
        for token_str, token_id in SPECIAL_TOKENS.items():
            assert vocab[token_str] == token_id

    def test_early_stop_graceful(self):
        """Tiny corpus runs out of pairs before n_merges — must not crash."""
        t = BPETrainer(vocab_size=32_768, log_every=0)
        t.train(iter(["ab", "ab"]))  # freq=2, exactly 1 pair possible
        assert len(t.merges) == 1


# ---------------------------------------------------------------------------
# BPEEncoder
# ---------------------------------------------------------------------------

class TestBPEEncoder:

    def test_encode_returns_list(self, encoder: BPEEncoder):
        ids = encoder.encode("hello")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_empty_string(self, encoder: BPEEncoder):
        assert encoder.encode("") == []

    def test_all_ids_in_uint16_range(self, encoder: BPEEncoder):
        ids = encoder.encode("hello world foo bar baz")
        assert all(0 <= i <= 32_767 for i in ids)

    def test_special_token_in_text(self, encoder: BPEEncoder):
        ids = encoder.encode("hello<|endoftext|>world")
        assert 0 in ids  # <|endoftext|> → ID 0

    def test_special_token_id_exact(self, encoder: BPEEncoder):
        ids = encoder.encode("<|endoftext|>")
        assert ids == [0]

    def test_add_bos(self, encoder: BPEEncoder):
        ids = encoder.encode("hello", add_bos=True)
        assert ids[0] == SPECIAL_TOKENS["<|bos|>"]  # 3

    def test_add_eos(self, encoder: BPEEncoder):
        ids = encoder.encode("hello", add_eos=True)
        assert ids[-1] == SPECIAL_TOKENS["<|eos|>"]  # 4

    def test_add_bos_eos_both(self, encoder: BPEEncoder):
        ids = encoder.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == 3
        assert ids[-1] == 4

    def test_encode_batch_matches_individual(self, encoder: BPEEncoder):
        texts = ["hello", "world", "hello world"]
        batch = encoder.encode_batch(texts)
        for text, ids in zip(texts, batch):
            assert ids == encoder.encode(text)

    def test_encode_as_uint16_valid(self, encoder: BPEEncoder):
        ids = encoder.encode_as_uint16("hello world")
        assert all(0 <= i <= 32_767 for i in ids)

    def test_merges_compress(self, encoder: BPEEncoder):
        """Trained merges should reduce token count vs raw bytes."""
        text = "hello"
        ids = encoder.encode(text)
        raw_bytes = len(text.encode("utf-8"))
        assert len(ids) < raw_bytes  # merges must have fired

    def test_unicode_handled(self, encoder: BPEEncoder):
        """Non-ASCII text must encode without crash."""
        ids = encoder.encode("café résumé naïve")
        assert all(0 <= i <= 32_767 for i in ids)
        assert len(ids) > 0

    def test_multiple_special_tokens(self, encoder: BPEEncoder):
        ids = encoder.encode("<|bos|>hello<|eos|>")
        assert ids[0] == 3   # <|bos|>
        assert ids[-1] == 4  # <|eos|>


# ---------------------------------------------------------------------------
# BPEDecoder
# ---------------------------------------------------------------------------

class TestBPEDecoder:

    def test_decode_returns_string(self, encoder: BPEEncoder, decoder: BPEDecoder):
        ids = encoder.encode("hello")
        assert isinstance(decoder.decode(ids), str)

    def test_decode_empty(self, decoder: BPEDecoder):
        assert decoder.decode([]) == ""

    def test_skip_special_tokens_default(self, encoder: BPEEncoder, decoder: BPEDecoder):
        ids = encoder.encode("hello<|endoftext|>world")
        out = decoder.decode(ids)  # skip_special_tokens=True by default
        assert "<|endoftext|>" not in out
        assert "hello" in out
        assert "world" in out

    def test_keep_special_tokens(self, encoder: BPEEncoder, decoder: BPEDecoder):
        ids = encoder.encode("hello<|endoftext|>world")
        out = decoder.decode(ids, skip_special_tokens=False)
        assert "<|endoftext|>" in out

    def test_unknown_id_returns_question_mark(self, decoder: BPEDecoder):
        out = decoder.decode([99_999])  # ID way out of range
        assert "?" in out  # fallback, no crash

    def test_decode_batch_matches_individual(
        self, encoder: BPEEncoder, decoder: BPEDecoder
    ):
        texts = ["hello", "world"]
        batch = decoder.decode_batch(encoder.encode_batch(texts))
        for text, out in zip(texts, batch):
            assert out == decoder.decode(encoder.encode(text))

    def test_decode_single_token_byte(self, decoder: BPEDecoder):
        b = decoder.decode_single_token(_BYTE_OFFSET)  # byte 0x00
        assert b == bytes([0])

    def test_decode_single_token_special(self, decoder: BPEDecoder):
        b = decoder.decode_single_token(0)
        assert b == b"<|endoftext|>"


# ---------------------------------------------------------------------------
# Round-trip (encode → decode == original)
# ---------------------------------------------------------------------------

class TestRoundTrip:

    @pytest.mark.parametrize("text", [
        "hello world",
        "hello there",
        "foo bar baz",
        "the quick brown fox jumps over the lazy dog",
        "Hello World!",
        "  leading spaces",
        "trailing spaces  ",
        "multiple   spaces",
        "newline\nhere",
        "tab\there",
    ])
    def test_roundtrip_ascii(self, encoder: BPEEncoder, decoder: BPEDecoder, text: str):
        assert decoder.decode(encoder.encode(text)) == text

    @pytest.mark.parametrize("text", [
        "café",
        "résumé",
        "naïve",
        "日本語",
        "中文",
        "한국어",
        "αβγδ",
        "emoji: 🔥",
    ])
    def test_roundtrip_unicode(self, encoder: BPEEncoder, decoder: BPEDecoder, text: str):
        """Byte-level BPE must round-trip all Unicode via UTF-8."""
        assert decoder.decode(encoder.encode(text)) == text

    def test_roundtrip_empty(self, encoder: BPEEncoder, decoder: BPEDecoder):
        assert decoder.decode(encoder.encode("")) == ""

    def test_roundtrip_bos_eos_stripped(self, encoder: BPEEncoder, decoder: BPEDecoder):
        """BOS/EOS added by encoder must be stripped by decoder (skip_special_tokens=True)."""
        text = "hello world"
        ids = encoder.encode(text, add_bos=True, add_eos=True)
        assert ids[0] == 3 and ids[-1] == 4
        assert decoder.decode(ids) == text

    def test_roundtrip_multiple_docs(self, encoder: BPEEncoder, decoder: BPEDecoder):
        """Documents joined with <|endoftext|> round-trip with skip_special=False."""
        docs = ["hello world", "foo bar"]
        joined = "<|endoftext|>".join(docs)
        ids = encoder.encode(joined)
        out = decoder.decode(ids, skip_special_tokens=False)
        assert out == joined


# ---------------------------------------------------------------------------
# Serialization: save + load
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_save_creates_three_files(self, saved_tok):
        _, tmp, _ = saved_tok
        assert (Path(tmp) / "tokenizer.json").exists()
        assert (Path(tmp) / "vocab.json").exists()
        assert (Path(tmp) / "merges.txt").exists()

    def test_tokenizer_json_schema(self, saved_tok):
        _, tmp, _ = saved_tok
        data = json.loads((Path(tmp) / "tokenizer.json").read_text())
        assert "version"        in data
        assert "vocab_size"     in data
        assert "special_tokens" in data
        assert "merges"         in data

    def test_tokenizer_json_special_tokens(self, saved_tok):
        _, tmp, _ = saved_tok
        data = json.loads((Path(tmp) / "tokenizer.json").read_text())
        assert data["special_tokens"] == SPECIAL_TOKENS

    def test_vocab_json_size_matches(self, saved_tok):
        _, tmp, out = saved_tok
        tok_data  = json.loads((Path(tmp) / "tokenizer.json").read_text())
        vocab     = json.loads((Path(tmp) / "vocab.json").read_text())
        assert len(vocab) == tok_data["vocab_size"]

    def test_merges_txt_line_count(self, saved_tok, trainer: BPETrainer):
        _, tmp, _ = saved_tok
        lines = (Path(tmp) / "merges.txt").read_text(encoding="latin-1").splitlines()
        non_comment = [l for l in lines if not l.startswith("#")]
        assert len(non_comment) == len(trainer.merges)

    def test_loaded_vocab_size(self, saved_tok, trainer: BPETrainer):
        tok, _, _ = saved_tok
        assert tok.vocab_size == _MERGE_START + len(trainer.merges)

    def test_loaded_special_tokens(self, saved_tok):
        tok, _, _ = saved_tok
        assert tok.special_tokens == SPECIAL_TOKENS

    def test_loaded_encode_matches_inmemory(
        self, saved_tok, encoder: BPEEncoder
    ):
        tok, _, _ = saved_tok
        texts = ["hello world", "foo bar", "the quick brown fox"]
        for text in texts:
            assert tok.encode(text) == encoder.encode(text)

    def test_loaded_roundtrip(self, saved_tok):
        tok, _, _ = saved_tok
        texts = ["hello world", "foo bar baz", "Hello World!"]
        for text in texts:
            assert tok.decode(tok.encode(text)) == text

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_tokenizer("/nonexistent/path/tokenizer.json")

    def test_load_bad_version(self, saved_tok):
        _, tmp, out = saved_tok
        data = json.loads(Path(out).read_text())
        data["version"] = "9.9"
        bad = Path(tmp) / "bad_version.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Unsupported tokenizer version"):
            load_tokenizer(bad)

    def test_load_vocab_size_mismatch(self, saved_tok):
        _, tmp, out = saved_tok
        data = json.loads(Path(out).read_text())
        data["vocab_size"] = data["vocab_size"] + 999
        bad = Path(tmp) / "bad_size.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="vocab_size mismatch"):
            load_tokenizer(bad)

    def test_load_missing_field(self, saved_tok):
        _, tmp, out = saved_tok
        data = json.loads(Path(out).read_text())
        del data["merges"]
        bad = Path(tmp) / "bad_fields.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing fields"):
            load_tokenizer(bad)

    def test_save_untrained_raises(self):
        bad = BPETrainer(vocab_size=300)
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="train()"):
                save_tokenizer(bad, tmp)
