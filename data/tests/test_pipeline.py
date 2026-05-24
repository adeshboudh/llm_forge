"""
Data pipeline tests — no network, no GPU, no HuggingFace.

Tests cover:
  - DocumentTokenizer: encode correctness, EOT prepend, stream
  - ShardWriter: flush logic, uint16 validation, metadata, overwrite guard
  - ShardedTokenDataset: load, iterate, window slicing, metadata validation

Run:
    pytest data/tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal stub tokenizer — avoids loading real BPE for unit tests
# ---------------------------------------------------------------------------

@dataclass
class _StubTokenizer:
    """
    Minimal tokenizer stub.
    Encodes each char as its ASCII value + 10 (keeps IDs > 4 to avoid
    colliding with special token IDs 0–4).
    """
    vocab_size: int = 32_768
    special_tokens: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                "<|endoftext|>": 0, "<|pad|>": 1,
                "<|unk|>": 2, "<|bos|>": 3, "<|eos|>": 4,
            }

    def encode(self, text: str, **_) -> list[int]:
        return [min(ord(c) + 10, 32_767) for c in text]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_tok():
    return _StubTokenizer()


@pytest.fixture
def doc_tok(stub_tok):
    from data.preprocessing.tokenize_dataset import DocumentTokenizer
    return DocumentTokenizer(stub_tok, add_eot=True)


def _fake_docs(texts: list[str]):
    """Yield fake Document objects."""
    @dataclass
    class Doc:
        text: str
        char_count: int
        doc_id: int

    for i, t in enumerate(texts):
        yield Doc(text=t, char_count=len(t), doc_id=i)


# ---------------------------------------------------------------------------
# DocumentTokenizer
# ---------------------------------------------------------------------------

class TestDocumentTokenizer:

    def test_eot_prepended(self, doc_tok):
        ids = doc_tok.encode_document("hello")
        assert ids[0] == 0  # <|endoftext|>

    def test_eot_not_prepended_when_disabled(self, stub_tok):
        from data.preprocessing.tokenize_dataset import DocumentTokenizer
        dt = DocumentTokenizer(stub_tok, add_eot=False)
        ids = dt.encode_document("hello")
        assert ids[0] != 0

    def test_encode_document_nonempty(self, doc_tok):
        ids = doc_tok.encode_document("abc")
        assert len(ids) > 1  # EOT + at least one token

    def test_all_ids_uint16(self, doc_tok):
        ids = doc_tok.encode_document("hello world " * 20)
        assert all(0 <= i <= 32_767 for i in ids)

    def test_encode_stream_yields_per_doc(self, doc_tok):
        docs = list(_fake_docs(["hello", "world", "foo"]))
        results = list(doc_tok.encode_stream(iter(docs), log_every=0))
        assert len(results) == 3

    def test_encode_stream_each_starts_with_eot(self, doc_tok):
        docs = list(_fake_docs(["hello", "world"]))
        for ids in doc_tok.encode_stream(iter(docs), log_every=0):
            assert ids[0] == 0


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------

class TestShardWriter:

    def _make_writer(self, tmp, shard_size=10):
        from data.preprocessing.shard_writer import ShardWriter
        return ShardWriter(output_dir=tmp, shard_size=shard_size)

    def test_flush_creates_shard(self, tmp_path):
        w = self._make_writer(tmp_path, shard_size=5)
        w.add(list(range(5)))   # exactly one shard
        w.finalize("v-test")
        shards = list(tmp_path.glob("shard_*.npy"))
        assert len(shards) >= 1

    def test_partial_last_shard_flushed(self, tmp_path):
        w = self._make_writer(tmp_path, shard_size=10)
        w.add(list(range(7)))   # less than shard_size → flushed on finalize
        meta = w.finalize("v-test")
        assert meta["total_tokens"] == 7

    def test_multiple_shards(self, tmp_path):
        w = self._make_writer(tmp_path, shard_size=5)
        w.add(list(range(13)))  # 13 tokens → 2 full shards + 1 partial
        meta = w.finalize("v-test")
        assert meta["total_shards"] == 3
        assert meta["total_tokens"] == 13

    def test_shard_dtype_uint16(self, tmp_path):
        w = self._make_writer(tmp_path, shard_size=5)
        w.add([1, 2, 3, 4, 5])
        w.finalize("v-test")
        shard = np.load(tmp_path / "shard_00000.npy")
        assert shard.dtype == np.uint16

    def test_shard_content_correct(self, tmp_path):
        tokens = [10, 20, 30, 40, 50]
        w = self._make_writer(tmp_path, shard_size=10)
        w.add(tokens)
        w.finalize("v-test")
        shard = np.load(tmp_path / "shard_00000.npy")
        assert list(shard) == tokens

    def test_metadata_written(self, tmp_path):
        w = self._make_writer(tmp_path, shard_size=5)
        w.add(list(range(5)))
        w.finalize("v1-test")
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["dataset_version"] == "v1-test"
        assert meta["total_tokens"] == 5
        assert meta["total_shards"] == 1
        assert len(meta["shards"]) == 1

    def test_overwrite_guard(self, tmp_path):
        w = self._make_writer(tmp_path, shard_size=5)
        w.add(list(range(5)))
        w.finalize("v-test")
        # Try creating another writer in same dir
        from data.preprocessing.shard_writer import ShardWriter
        with pytest.raises(FileExistsError, match="never overwrite"):
            ShardWriter(output_dir=tmp_path, shard_size=5)

    def test_invalid_token_id_raises(self, tmp_path):
        from data.preprocessing.shard_writer import ShardWriter
        w = ShardWriter(output_dir=tmp_path, shard_size=10, vocab_size=100)
        with pytest.raises(ValueError, match="out of range"):
            w.add([50, 200])  # 200 >= vocab_size=100

    def test_finalize_empty_raises(self, tmp_path):
        w = self._make_writer(tmp_path, shard_size=5)
        with pytest.raises(ValueError, match="empty"):
            w.finalize("v-test")

    def test_shard_filenames_zero_padded(self, tmp_path):
        w = self._make_writer(tmp_path, shard_size=3)
        w.add(list(range(9)))  # 3 shards
        w.finalize("v-test")
        names = sorted(p.name for p in tmp_path.glob("shard_*.npy"))
        assert names[0] == "shard_00000.npy"
        assert names[1] == "shard_00001.npy"
        assert names[2] == "shard_00002.npy"


# ---------------------------------------------------------------------------
# ShardedTokenDataset
# ---------------------------------------------------------------------------

def _write_test_shards(tmp_path: Path, tokens: list[int], shard_size: int = 20):
    """Helper: write fake shards + metadata for loader tests."""
    from data.preprocessing.shard_writer import ShardWriter
    w = ShardWriter(output_dir=tmp_path, shard_size=shard_size)
    w.add(tokens)
    w.finalize("v-loader-test")


class TestShardedTokenDataset:

    def test_load_requires_metadata(self, tmp_path):
        from data.loaders.npy_loader import ShardedTokenDataset
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            ShardedTokenDataset(tmp_path, seq_len=4)

    def test_load_requires_shards(self, tmp_path):
        (tmp_path / "metadata.json").write_text('{"total_shards": 1}')
        from data.loaders.npy_loader import ShardedTokenDataset
        with pytest.raises(FileNotFoundError, match="shard_"):
            ShardedTokenDataset(tmp_path, seq_len=4)

    def test_iteration_yields_pairs(self, tmp_path):
        tokens = list(range(100))
        _write_test_shards(tmp_path, tokens, shard_size=50)
        from data.loaders.npy_loader import ShardedTokenDataset
        ds = ShardedTokenDataset(tmp_path, seq_len=8, shuffle_shards=False)
        pairs = list(ds)
        assert len(pairs) > 0
        inp, tgt = pairs[0]
        assert len(inp) == 8
        assert len(tgt) == 8

    def test_target_is_input_shifted(self, tmp_path):
        tokens = list(range(50))
        _write_test_shards(tmp_path, tokens, shard_size=50)
        from data.loaders.npy_loader import ShardedTokenDataset
        ds = ShardedTokenDataset(tmp_path, seq_len=4, shuffle_shards=False)
        inp, tgt = next(iter(ds))
        assert list(tgt) == list(inp[1:]) + [inp[-1] + 1]  # shifted by 1

    def test_repr(self, tmp_path):
        _write_test_shards(tmp_path, list(range(100)), shard_size=50)
        from data.loaders.npy_loader import ShardedTokenDataset
        ds = ShardedTokenDataset(tmp_path, seq_len=8)
        r = repr(ds)
        assert "ShardedTokenDataset" in r
        assert "seq_len=8" in r

    def test_shard_count_mismatch_raises(self, tmp_path):
        _write_test_shards(tmp_path, list(range(100)), shard_size=50)
        # Corrupt metadata to claim wrong shard count
        meta = json.loads((tmp_path / "metadata.json").read_text())
        meta["total_shards"] = 99
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        from data.loaders.npy_loader import ShardedTokenDataset
        with pytest.raises(ValueError, match="metadata.json says"):
            ShardedTokenDataset(tmp_path, seq_len=4)

    def test_metadata_accessible(self, tmp_path):
        _write_test_shards(tmp_path, list(range(100)), shard_size=50)
        from data.loaders.npy_loader import ShardedTokenDataset
        ds = ShardedTokenDataset(tmp_path, seq_len=8)
        assert ds.total_tokens == 100
        assert ds.vocab_size == 32_768
        assert ds.metadata["dataset_version"] == "v-loader-test"
