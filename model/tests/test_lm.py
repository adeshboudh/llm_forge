"""Full LM tests — forward pass, loss, tied emb, gradient flow."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model.config import load_model_config
from model.lm import LM


def _cfg(name="model_25m", attention="gqa"):
    cfg = load_model_config(name)
    # Shrink for fast unit tests: n_layers=2, d_model=64, n_heads=4, d_ff=128
    return dataclasses.replace(
        cfg, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
        d_ff=128, max_seq_len=64,
    )


def test_forward_returns_scalar_loss():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss = model.apply(params, input_ids, target_ids)
    assert loss.shape == ()  # scalar


def test_loss_near_neg_log_vocab_at_init():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss = float(model.apply(params, input_ids, target_ids))
    expected = float(jnp.log(cfg.vocab_size))
    assert abs(loss - expected) < 1.5  # ±1.5 around 10.4


def test_logits_shape():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss, logits = model.apply(params, input_ids, target_ids, return_logits=True)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_gradients_flow_to_all_params():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)

    def loss(p):
        return model.apply(p, input_ids, target_ids)

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.any(leaf != 0) for leaf in leaves), "some param got zero grad"


def test_tied_embeddings():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 4), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 4), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    # tok_emb: (vocab, D); lm_head not in params (tied)
    assert "tok_emb" in params["params"]
    assert "lm_head" not in params["params"]


def test_tokens_within_vocab_run():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    good_in = jnp.array([[0, 1, 2, 3]])
    good_tgt = jnp.array([[1, 2, 3, 4]])
    params = model.init(key, good_in, good_tgt)
    loss = model.apply(params, good_in, good_tgt)
    assert jnp.isfinite(loss)


def test_attention_variant_selectable():
    for variant in ("mha", "mqa", "gqa"):
        cfg = dataclasses.replace(_cfg(), attention=variant)
        model = LM(config=cfg)
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (2, 8), 0, cfg.vocab_size)
        target_ids = jax.random.randint(key, (2, 8), 0, cfg.vocab_size)
        params = model.init(key, input_ids, target_ids)
        loss = model.apply(params, input_ids, target_ids)
        assert jnp.isfinite(loss)


def test_forward_runs_at_specd_seq_len():
    cfg = dataclasses.replace(_cfg(), max_seq_len=32)
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (1, 32), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (1, 32), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss = model.apply(params, input_ids, target_ids)
    assert jnp.isfinite(loss)


def test_param_count_smoke():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (1, 4), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (1, 4), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    total = sum(p.size for p in jax.tree_util.tree_leaves(params))
    # Crude: should be much less than 1M after our mini-shrink
    assert total > 0
