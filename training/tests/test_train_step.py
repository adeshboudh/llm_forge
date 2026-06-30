"""train_step / eval_step tests — pjit boundaries, bf16 cast, dtype assertions."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from model.config import load_model_config
from model.lm import LM
from training.state import create_train_state
from training.train_step import eval_step, make_loss_fn, set_model, train_step


def _model_cfg():
    cfg = load_model_config("model_25m")
    return dataclasses.replace(
        cfg,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=64,
    )


@pytest.fixture(autouse=True)
def _set_step_model():
    """Ensure train_step knows which LM to call inside the jit closure."""
    cfg = _model_cfg()
    set_model(LM(config=cfg))
    yield


def test_loss_fn_returns_finite_scalar(toy_config):
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    key = jax.random.PRNGKey(0)
    state = create_train_state(key, model, toy_config, model_cfg)
    input_ids = jax.random.randint(key, (2, 16), 0, model_cfg.vocab_size, dtype=jnp.int32)
    target_ids = jax.random.randint(key, (2, 16), 0, model_cfg.vocab_size, dtype=jnp.int32)
    loss_fn = make_loss_fn(model)
    loss, grads = loss_fn(state.params, input_ids, target_ids)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_grads_are_fp32(toy_config):
    """Mixed precision: grads return as fp32 (master copy dtype)."""
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    state = create_train_state(jax.random.PRNGKey(0), model, toy_config, model_cfg)
    input_ids = jnp.zeros((2, 16), dtype=jnp.int32)
    target_ids = jnp.zeros((2, 16), dtype=jnp.int32)
    loss_fn = make_loss_fn(model)
    _, grads = loss_fn(state.params, input_ids, target_ids)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert leaf.dtype == jnp.float32, f"grad leaf dtype={leaf.dtype}"


def test_bf16_compute_path_matches_fp32_within_tolerance(toy_config):
    """bf16 cast of 2D params should not change loss by > 5% vs all-fp32 at small batch."""
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    key = jax.random.PRNGKey(0)
    state = create_train_state(key, model, toy_config, model_cfg)
    input_ids = jax.random.randint(key, (1, 16), 0, model_cfg.vocab_size, dtype=jnp.int32)
    target_ids = jax.random.randint(key, (1, 16), 0, model_cfg.vocab_size, dtype=jnp.int32)

    loss_fn = make_loss_fn(model)
    loss_bf16 = loss_fn(state.params, input_ids, target_ids)[0]

    def loss_fp32_fn(params):
        return model.apply(params, input_ids, target_ids)

    loss_fp32 = loss_fp32_fn(state.params)

    assert abs(float(loss_bf16) - float(loss_fp32)) / float(loss_fp32) < 0.05


def test_train_step_returns_new_state_loss_gradnorm(toy_config):
    """train_step (jit w/ sharding): returns (new_state, loss, metrics)."""
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    key = jax.random.PRNGKey(0)
    state = create_train_state(key, model, toy_config, model_cfg)
    input_ids = jax.random.randint(key, (4, 64), 0, model_cfg.vocab_size, dtype=jnp.int32)
    target_ids = jax.random.randint(key, (4, 64), 0, model_cfg.vocab_size, dtype=jnp.int32)
    new_state, loss, metrics = train_step(state, input_ids, target_ids)
    assert jnp.isfinite(loss)
    assert loss.shape == ()
    assert "grad_norm" in metrics
    assert float(metrics["grad_norm"]) > 0
    assert int(new_state.step) == int(state.step) + 1


def test_eval_step_returns_finite_loss(toy_config):
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    key = jax.random.PRNGKey(0)
    state = create_train_state(key, model, toy_config, model_cfg)
    input_ids = jax.random.randint(key, (4, 64), 0, model_cfg.vocab_size, dtype=jnp.int32)
    target_ids = jax.random.randint(key, (4, 64), 0, model_cfg.vocab_size, dtype=jnp.int32)
    loss = eval_step(state, input_ids, target_ids)
    assert jnp.isfinite(loss)
    assert loss.shape == ()
