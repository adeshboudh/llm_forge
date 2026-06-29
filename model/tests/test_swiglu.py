"""SwiGLU MLP tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model.config import load_model_config
from model.mlp.swiglu import SwiGLUMLP, compute_d_ff


def test_compute_d_ff():
    # round(8/3 * D / 256) * 256
    assert compute_d_ff(512) == 1280   # round(1365.33/256)*256 = 5*256 = 1280
    assert compute_d_ff(768) == 2048   # round(2048/256)*256 = 8*256 = 2048
    assert compute_d_ff(1024) == 2816  # round(2730.67/256)*256 = 11*256 = 2816


def test_shape_preserved():
    mod = SwiGLUMLP(d_model=64, d_ff=128)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 64))
    params = mod.init(key, x)
    out = mod.apply(params, x)
    assert out.shape == (2, 16, 64)


def test_param_count():
    mod = SwiGLUMLP(d_model=64, d_ff=128)
    params = mod.init(jax.random.PRNGKey(0), jnp.ones((1, 1, 64)))
    p = params["params"]
    # W_gate: (64, 128), W_up: (64, 128), W_down: (128, 64)
    expected = 2 * 64 * 128 + 128 * 64
    actual = p["W_gate"].size + p["W_up"].size + p["W_down"].size
    assert actual == expected


def test_zero_input_zero_output():
    mod = SwiGLUMLP(d_model=64, d_ff=128)
    key = jax.random.PRNGKey(0)
    params = mod.init(key, jnp.ones((1, 1, 64)))
    out = mod.apply(params, jnp.zeros((1, 4, 64)))
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_gradients_flow():
    mod = SwiGLUMLP(d_model=64, d_ff=128)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 8, 64))
    params = mod.init(key, x)

    def loss(p):
        return jnp.sum(mod.apply(p, x))

    grads = jax.grad(loss)(params)["params"]
    for name in ("W_gate", "W_up", "W_down"):
        assert jnp.all(grads[name] != 0), f"{name} got zero grad"


def test_d_ff_config_match():
    for name, expected in [("model_25m", 1280), ("model_125m", 2048), ("model_350m", 2816)]:
        cfg = load_model_config(name)
        assert compute_d_ff(cfg.d_model) == expected, f"{name}: formula"
        assert cfg.d_ff == expected, f"{name}: yaml"
        assert cfg.d_ff == compute_d_ff(cfg.d_model), f"{name}: mismatch"
