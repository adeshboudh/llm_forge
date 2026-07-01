"""Tests for safetensors export: round-trip + dtype + shape + key stability."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest
from safetensors.numpy import load_file

from model.config import load_model_config
from model.lm import LM
from training.export import (
    export_state_params,
    load_params_safetensors,
    save_params_safetensors,
)
from training.state import create_train_state


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


def test_save_and_load_round_trip(toy_config, tmp_path):
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    state = create_train_state(jax.random.PRNGKey(0), model, toy_config, model_cfg)

    path = tmp_path / "params.safetensors"
    save_params_safetensors(state.params, path)
    assert path.exists()

    flat = load_params_safetensors(path)
    original = jax.tree_util.tree_flatten_with_path(state.params)[0]
    original_dict = {
        "/".join(str(p.key) for p in path_): np.asarray(leaf) for path_, leaf in original
    }
    assert set(flat.keys()) == set(original_dict.keys())
    for k in flat:
        np.testing.assert_array_equal(flat[k], original_dict[k])


def test_safetensors_dtype_preserved(toy_config, tmp_path):
    """Master params are fp32; safetensors should preserve dtype."""
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    state = create_train_state(jax.random.PRNGKey(0), model, toy_config, model_cfg)
    path = tmp_path / "params.safetensors"
    save_params_safetensors(state.params, path)

    flat = load_params_safetensors(path)
    for name, arr in flat.items():
        assert arr.dtype == np.float32, f"{name} got {arr.dtype}, expected float32"


def test_keys_have_dotted_path(toy_config, tmp_path):
    """Keys should look like 'params/blocks_0/mlp/W_up', not opaque IDs."""
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    state = create_train_state(jax.random.PRNGKey(0), model, toy_config, model_cfg)
    path = tmp_path / "params.safetensors"
    save_params_safetensors(state.params, path)

    flat = load_file(str(path))
    sample_keys = list(flat.keys())[:5]
    for k in sample_keys:
        assert k.startswith("params/"), f"key {k!r} missing 'params/' prefix"
        assert "/" in k, f"key {k!r} should be a dotted path"


def test_export_state_params_helper(toy_config, tmp_path):
    """export_state_params is sugar for save_params_safetensors(state.params, ...)."""
    model_cfg = _model_cfg()
    model = LM(config=model_cfg)
    state = create_train_state(jax.random.PRNGKey(0), model, toy_config, model_cfg)
    path = tmp_path / "params.safetensors"
    export_state_params(state, path)
    assert path.exists()
    flat = load_file(str(path))
    assert len(flat) > 0


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_params_safetensors(tmp_path / "nope.safetensors")
