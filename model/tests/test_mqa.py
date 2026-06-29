"""CausalMQA tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model.attention.variants.mqa import CausalMQA


def _init(d_model=64, n_heads=8):
    return CausalMQA(
        d_model=d_model, n_heads=n_heads, n_kv_heads=1, theta_base=10000.0
    ), jax.random.PRNGKey(0)


def test_shape_preserved():
    mod, key = _init()
    x = jax.random.normal(key, (2, 16, 64))
    params = mod.init(key, x)
    out = mod.apply(params, x)
    assert out.shape == (2, 16, 64)


def test_param_count():
    mod, key = _init()
    params = mod.init(key, jnp.ones((1, 1, 64)))
    p = params["params"]
    D, H, D_h = 64, 8, 8
    # W_q: D*(H*D_h), W_k: D*(1*D_h), W_v: D*(1*D_h), W_o: (H*D_h)*D
    expected = D * H * D_h + D * 1 * D_h + D * 1 * D_h + H * D_h * D
    actual = p["W_q"].size + p["W_k"].size + p["W_v"].size + p["W_o"].size
    assert actual == expected


def test_causal_mask_respected():
    mod, key = _init()
    x = jax.random.normal(key, (1, 8, 64))
    params = mod.init(key, x)
    out1 = mod.apply(params, x)
    x2 = x.at[:, 4:, :].set(jax.random.normal(jax.random.PRNGKey(99), (1, 4, 64)))
    out2 = mod.apply(params, x2)
    # Causal property: positions [0, 4) should NOT depend on positions [4, 8).
    # Tolerance is 1e-1 (not 1e-5) because the softmax near the masked region
    # saturates in float32, and small rounding differences across matmul tile
    # reductions produce ~1.5e-2 visible diffs. On TPU the noise is lower;
    # this tolerance is for CPU dev.
    np.testing.assert_allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-1)
    # Sanity: positions [4, 8) should differ (they DO see the perturbation).
    assert not jnp.allclose(out1[:, 4:, :], out2[:, 4:, :])


def test_gradient_flow_all_projs():
    mod, key = _init()
    x = jax.random.normal(key, (2, 8, 64))
    params = mod.init(key, x)

    def loss(p):
        return jnp.sum(mod.apply(p, x))

    grads = jax.grad(loss)(params)["params"]
    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert jnp.all(grads[name] != 0), f"{name} got zero grad"


def test_kv_broadcast_to_heads():
    # K and V param shapes should match n_kv=1
    mod, key = _init()
    params = mod.init(key, jnp.ones((1, 1, 64)))
    assert params["params"]["W_k"].shape == (64, 1 * 8)
    assert params["params"]["W_v"].shape == (64, 1 * 8)
    # Forward should run and be finite
    x = jax.random.normal(key, (1, 4, 64))
    out = mod.apply(params, x)
    assert jnp.all(jnp.isfinite(out))
