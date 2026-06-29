"""ModelConfig + load_model_config tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from model.config import ModelConfig, load_model_config


def test_load_model_25m():
    cfg = load_model_config("model_25m")
    assert cfg.name == "model_25m"
    assert cfg.n_layers == 4
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.n_kv_heads == 4
    assert cfg.d_ff == 1280
    assert cfg.vocab_size == 32768
    assert cfg.max_seq_len == 1024
    assert cfg.theta_base == 10000.0
    assert cfg.tied_embeddings is True
    assert cfg.attention == "gqa"


def test_load_model_125m():
    cfg = load_model_config("model_125m")
    assert cfg.n_layers == 12
    assert cfg.d_model == 768
    assert cfg.n_heads == 12
    assert cfg.n_kv_heads == 4
    assert cfg.d_ff == 2048
    assert cfg.max_seq_len == 1024


def test_load_model_350m():
    cfg = load_model_config("model_350m")
    assert cfg.n_layers == 24
    assert cfg.d_model == 1024
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 8
    assert cfg.d_ff == 2816
    assert cfg.max_seq_len == 2048


def test_load_unknown_raises():
    with pytest.raises(KeyError, match="not found"):
        load_model_config("model_999m")


def test_d_head_derived():
    cfg = load_model_config("model_25m")
    assert cfg.d_head == cfg.d_model // cfg.n_heads  # 64


def test_d_kv_heads_divides_n_heads():
    for name in ("model_25m", "model_125m", "model_350m"):
        cfg = load_model_config(name)
        assert cfg.n_heads % cfg.n_kv_heads == 0


def test_config_frozen():
    cfg = load_model_config("model_25m")
    with pytest.raises(Exception):
        cfg.n_layers = 99  # frozen dataclass
