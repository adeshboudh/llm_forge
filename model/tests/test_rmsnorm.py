"""RMSNorm tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model.normalization.rmsnorm import RMSNorm


def _init(dim=8, eps=1e-6):
    return RMSNorm(dim=dim, eps=eps)


def _make_params(dim=8):
    mod = _init(dim)
    key = jax.random.PRNGKey(0)
    return mod.init(key, jnp.ones((1, 4, dim)))


def test_zero_input_zero_output():
    mod = _init()
    params = _make_params()
    out = mod.apply(params, jnp.zeros((1, 4, 8)))
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_output_rms_is_one():
    mod = _init()
    params = _make_params()
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 8))
    out = mod.apply(params, x)
    rms = jnp.sqrt(jnp.mean(out**2, axis=-1))
    np.testing.assert_allclose(rms, jnp.ones_like(rms), atol=1e-4)


def test_scale_param_applied():
    mod = _init()
    params = _make_params()
    # Set scale to 2.0
    params["params"]["scale"] = jnp.ones(8) * 2.0
    x = jax.random.normal(jax.random.PRNGKey(2), (1, 4, 8))
    out = mod.apply(params, x)
    rms = jnp.sqrt(jnp.mean(out**2, axis=-1))
    np.testing.assert_allclose(rms, 2.0 * jnp.ones_like(rms), atol=1e-4)


def test_shape_preserved():
    mod = _init()
    params = _make_params()
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 16, 8))
    out = mod.apply(params, x)
    assert out.shape == (2, 16, 8)


def test_param_count():
    params = _make_params()
    assert params["params"]["scale"].shape == (8,)
    assert params["params"]["scale"].size == 8


def test_gradients_flow_to_scale():
    mod = _init()
    params = _make_params()
    x = jax.random.normal(jax.random.PRNGKey(4), (1, 4, 8))

    def loss(s):
        p = {"params": {"scale": s}}
        out = mod.apply(p, x)
        return jnp.sum(out)

    grad = jax.grad(loss)(params["params"]["scale"])
    assert jnp.all(grad != 0)


def test_eps_applied():
    # With large eps, output should be smaller (input gets divided by sqrt(ms+eps))
    mod_small = _init(eps=1e-6)
    mod_large = _init(eps=0.5)
    x = jnp.ones((1, 4, 8)) * 0.1
    p_small = _make_params()
    p_large = mod_large.init(jax.random.PRNGKey(0), x)
    out_small = mod_small.apply(p_small, x)
    out_large = mod_large.apply(p_large, x)
    # With zero input, different eps shouldn't change zero output, so use nonzero input
    # RMS of out_small should be larger (smaller denominator)
    rms_s = float(jnp.sqrt(jnp.mean(out_small**2)))
    rms_l = float(jnp.sqrt(jnp.mean(out_large**2)))
    assert rms_s > rms_l
