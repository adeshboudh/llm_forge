"""TrainState creation + weight decay mask + orbax save/restore round-trip."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model.config import load_model_config
from model.lm import LM
from training.state import create_train_state, restore, save, weight_decay_mask


def _model_cfg():
    """Tiny LM config for fast tests (mirrors model/tests/test_lm.py::_cfg)."""
    import dataclasses

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


def test_create_train_state_params_are_fp32(toy_config):
    cfg = toy_config
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    key = jax.random.PRNGKey(0)
    state = create_train_state(key, model, cfg, model_cfg)
    for leaf in jax.tree_util.tree_leaves(state.params):
        assert leaf.dtype == jnp.float32, f"param leaf dtype={leaf.dtype}"
    assert int(state.step) == 0


def test_create_train_state_has_opt_state(toy_config):
    state = create_train_state(
        jax.random.PRNGKey(0),
        LM(config=_model_cfg()),
        toy_config,
        _model_cfg(),
    )
    leaves = jax.tree_util.tree_leaves(state.opt_state)
    assert len(leaves) >= 2
    for leaf in leaves:
        assert leaf.dtype in (jnp.float32, jnp.int32), f"opt_state dtype={leaf.dtype}"


def test_weight_decay_mask_skips_1d_and_tok_emb(toy_config):
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (1, 8), 0, model_cfg.vocab_size)
    params = model.init(key, input_ids, input_ids)["params"]
    mask = weight_decay_mask(params)
    flat_mask = jax.tree_util.tree_leaves(mask)
    flat_params = jax.tree_util.tree_leaves(params)
    flat_params_with_path = jax.tree_util.tree_leaves_with_path(params)
    flat_mask_with_path = jax.tree_util.tree_leaves_with_path(mask)
    assert len(flat_mask) == len(flat_params)
    for (path_p, p), (_path_m, m) in zip(flat_params_with_path, flat_mask_with_path, strict=False):
        path_str = str(path_p)
        if "tok_emb" in path_str:
            assert m is False, f"tok_emb should NOT be decayed, path={path_str}"
        if p.ndim <= 1:
            assert m is False, f"1D param should NOT be decayed, path={path_str}"
    any_decayed = any(
        m
        for (_, p), (_, m) in zip(flat_params_with_path, flat_mask_with_path, strict=False)
        if p.ndim >= 2
    )
    assert any_decayed, "no 2D params marked for weight decay — mask is wrong"


def test_save_restore_round_trips_params(toy_config, tmp_path):
    cfg = toy_config
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    key = jax.random.PRNGKey(0)
    state = create_train_state(key, model, cfg, model_cfg)

    ckpt_path = Path(cfg.ckpt.output_dir) / "test_ckpt"
    save(state, ckpt_path)

    placeholder = create_train_state(
        jax.random.PRNGKey(1),
        model,
        cfg,
        model_cfg,
    )
    restored = restore(ckpt_path, placeholder)

    for a, b in zip(
        jax.tree_util.tree_leaves(state.params),
        jax.tree_util.tree_leaves(restored.params),
        strict=False,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert int(restored.step) == int(state.step)


def test_restore_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        restore(tmp_path / "nonexistent_ckpt", None)


def test_apply_gradients_advances_step(toy_config):
    state = create_train_state(
        jax.random.PRNGKey(0),
        LM(config=_model_cfg()),
        toy_config,
        _model_cfg(),
    )
    grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    new_state = state.apply_gradients(grads=grads)
    assert int(new_state.step) == int(state.step) + 1


def test_lr_schedule_clamps_warmup_when_total_steps_shorter(toy_config):
    """--max-steps 50 with warmup_steps=200 must not raise; warmup is clamped."""
    import dataclasses

    from training.state import _make_lr_schedule

    cfg = dataclasses.replace(
        toy_config,
        train=dataclasses.replace(toy_config.train, total_steps=50, warmup_steps=200),
    )
    sched = _make_lr_schedule(cfg)
    assert float(sched(0)) == 0.0
    assert float(sched(50 // 2)) > 0.0
    assert float(sched(cfg.train.total_steps - 1)) < float(cfg.optim.lr_peak)
