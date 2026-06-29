"""End-to-end smoke tests — full LM forward for each size, JIT compiles."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model.config import load_model_config
from model.lm import LM


def _forward_smoke(name, batch=2, seq_len=128):
    cfg = load_model_config(name)
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (batch, seq_len), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (batch, seq_len), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss = model.apply(params, input_ids, target_ids)
    return float(loss), cfg


def test_25m_forward_pass():
    loss, cfg = _forward_smoke("model_25m", batch=2, seq_len=128)
    expected = float(jnp.log(cfg.vocab_size))
    assert abs(loss - expected) < 1.5


def test_125m_forward_pass():
    loss, cfg = _forward_smoke("model_125m", batch=2, seq_len=128)
    assert np.isfinite(loss)


def test_350m_forward_pass():
    # 350M is slow on CPU — mark skip if init takes >30s
    if jax.default_backend() == "cpu":
        pytest.skip("350M forward is too slow on CPU; run on GPU/TPU")
    loss, _ = _forward_smoke("model_350m", batch=1, seq_len=128)
    assert np.isfinite(loss)


def test_all_variants_run_on_smallest_model():
    cfg = load_model_config("model_25m")
    for variant in ("mha", "mqa", "gqa"):
        c = dataclasses.replace(cfg, attention=variant)
        model = LM(config=c)
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (2, 64), 0, c.vocab_size)
        target_ids = jax.random.randint(key, (2, 64), 0, c.vocab_size)
        params = model.init(key, input_ids, target_ids)
        loss = model.apply(params, input_ids, target_ids)
        assert np.isfinite(float(loss)), f"{variant} produced non-finite loss"


def test_jit_compiles():
    cfg = load_model_config("model_25m")
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 32), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 32), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)

    @jax.jit
    def jit_forward(p, x, y):
        return model.apply(p, x, y)

    loss = jit_forward(params, input_ids, target_ids)
    # Force computation
    float(loss)
    assert np.isfinite(float(loss))
