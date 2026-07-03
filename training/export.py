"""Portable params export via safetensors (HuggingFace standard).

Orbax checkpoints (`training.state.save`) keep params + AdamW state
together and need the same model code to restore — great for resume,
inconvenient for inference serving or cross-tool use.

This module writes just the model parameters (fp32 master) to a
single .safetensors file with stable flat key names like
`"params/blocks_0/mlp/W_up"`. Any tool that reads safetensors
(HuggingFace transformers, safetensors.torch, raw numpy) can load
the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
from safetensors.numpy import load_file, save_file


def _flatten_params(params: Any) -> dict[str, np.ndarray]:
    """Flatten a (nested) flax params pytree to {dotted_name: ndarray}.

    Leaves are converted to host-side numpy arrays (single device, no
    sharding) — the caller is responsible for gathering before export.
    """
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    out: dict[str, np.ndarray] = {}
    for path, leaf in flat:
        if not hasattr(leaf, "shape"):
            continue
        key = "/".join(
            str(p.key) if hasattr(p, "key") else str(p) for p in path
        )
        out[key] = np.asarray(leaf)
    return out


def save_params_safetensors(params: Any, path: str | Path) -> None:
    """Write a flax params pytree to a .safetensors file.

    Args:
        params: Flax params pytree (e.g. ``state.params``).
        path: Destination ``.safetensors`` file. Parent dirs are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = _flatten_params(params)
    save_file(flat, str(path))


def load_params_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load a .safetensors file into a flat {name: ndarray} dict.

    Returns the flat dict; the caller is responsible for unflattening
    into the same tree structure used at save time (use
    ``jax.tree_util.tree_unflatten`` with the structure from the model).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"params file not found: {path}")
    return load_file(str(path))


def load_params_into_model(path: str | Path, model) -> dict:
    """Load a .safetensors file and reshape it into a flax-apply-ready dict.

    Two gotchas handled here so callers don't trip on them:

    1. The saved keys carry a leading ``params/`` prefix (because
       ``state.params`` is wrapped in a flax collection when saved).
       We strip that prefix so the unflatten aligns with the structure
       returned by ``model.init(...)['params']``.
    2. ``model.apply`` expects ``{'params': pytree}``, not the bare pytree.

    Args:
        path: Path to ``params.safetensors``.
        model: A flax model instance (e.g. ``LM(config=cfg)``).

    Returns:
        A flax-apply-ready dict ``{'params': <pytree>}``.
    """
    import jax
    import jax.numpy as jnp
    import jax.tree_util as jtu

    flat = load_params_safetensors(path)
    flat = {k.removeprefix("params/"): v for k, v in flat.items()}

    sample = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 4), dtype=jnp.int32), jnp.zeros((1, 4), dtype=jnp.int32))["params"]
    init_leaves = jax.tree_util.tree_flatten_with_path(sample)
    structure = init_leaves[1]
    init_keys = ["/".join(str(p.key) for p in p_) for p_, _ in init_leaves[0]]
    missing = [k for k in init_keys if k not in flat]
    if missing:
        raise KeyError(
            f"safetensors is missing {len(missing)} param(s); first: {missing[:3]}. "
            "Did the file come from a different model config?"
        )
    leaves = [jnp.asarray(flat[k]) for k in init_keys]
    pytree = jtu.tree_unflatten(structure, leaves)
    sample_leaves = jtu.tree_leaves(sample)
    loaded_leaves = jtu.tree_leaves(pytree)
    for name, s, l in zip(init_keys, sample_leaves, loaded_leaves):
        if tuple(s.shape) != tuple(l.shape):
            raise ValueError(
                f"shape mismatch for {name!r}: saved={tuple(s.shape)} vs "
                f"model={tuple(l.shape)}. Check that the safetensors was "
                f"produced from the same model config."
            )
    return {"params": pytree}


def export_state_params(state, path: str | Path) -> None:
    """Convenience: pull ``state.params`` and write to .safetensors."""
    save_params_safetensors(state.params, path)
