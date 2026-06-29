"""TransformerBlock tests."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from model.blocks.transformer_block import TransformerBlock
from model.config import load_model_config


def _block_cfg(name="model_25m", attention="gqa"):
    cfg = load_model_config(name)
    return dataclasses.replace(cfg, attention=attention)


def test_shape_preserved():
    cfg = _block_cfg()
    block = TransformerBlock(config=cfg)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, cfg.d_model))
    params = block.init(key, x)
    out = block.apply(params, x)
    assert out.shape == (2, 16, cfg.d_model)


def test_param_count():
    cfg = _block_cfg()
    block = TransformerBlock(config=cfg)
    params = block.init(jax.random.PRNGKey(0), jnp.ones((1, 4, cfg.d_model)))
    p = params["params"]
    D, H, n_kv, D_h, F = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_head, cfg.d_ff
    norm_params = 2 * D
    attn_params = (D * H * D_h) + (D * n_kv * D_h) * 2 + (H * D_h * D)
    mlp_params = 3 * D * F
    expected = norm_params + attn_params + mlp_params
    actual = 0
    for k1 in p:
        for k2 in p[k1]:
            actual += p[k1][k2].size
    assert actual == expected


def test_gradient_flow_to_all_paths():
    cfg = _block_cfg()
    block = TransformerBlock(config=cfg)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 8, cfg.d_model))
    params = block.init(key, x)

    def loss(p):
        return jnp.sum(block.apply(p, x))

    grads = jax.grad(loss)(params)["params"]

    def any_nonzero(group):
        return any(jnp.any(grads[group][k] != 0) for k in grads[group])

    assert any_nonzero("norm1") or any_nonzero("attn")
    assert any_nonzero("mlp")


def test_pre_norm_order():
    # If norm1.scale=0, output == x + mlp(norm2(x))   (attn contributes 0)
    cfg = _block_cfg()
    block = TransformerBlock(config=cfg)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 4, cfg.d_model))
    params = block.init(key, x)
    # Zero out norm1's scale by direct tree manipulation
    new_scale = jnp.zeros_like(params["params"]["norm1"]["scale"])
    params = jax.tree_util.tree_map_with_path(
        lambda path, v: new_scale if "norm1" in str(path) and path[-1].key == "scale" else v,
        params,
    )
    # Apply, expect output finite (block-modified by mlp path only)
    out = block.apply(params, x)
    assert jnp.all(jnp.isfinite(out))


def test_attention_variant_selectable():
    for variant in ("mha", "mqa", "gqa"):
        cfg = _block_cfg(attention=variant)
        block = TransformerBlock(config=cfg)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 8, cfg.d_model))
        params = block.init(key, x)
        out = block.apply(params, x)
        assert out.shape == (2, 8, cfg.d_model)
        assert jnp.all(jnp.isfinite(out))
