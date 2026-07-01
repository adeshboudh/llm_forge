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


def export_state_params(state, path: str | Path) -> None:
    """Convenience: pull ``state.params`` and write to .safetensors."""
    save_params_safetensors(state.params, path)
