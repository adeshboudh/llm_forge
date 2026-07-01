"""TrainState: create, save, restore via orbax + weight decay mask.

TrainState carries the fp32 master params + AdamW opt_state + step.
Orbax writes to /kaggle/working/ckpt/<step>/ (async on TPU).
"""

from __future__ import annotations

from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state

from model.config import ModelConfig
from training.config import TrainConfig


def weight_decay_mask(params: dict) -> dict:
    """Return a pytree mask matching `params`: True where decay applies, False otherwise.

    Rules:
      - 1D params (norm scales, etc): False (never decay)
      - 2D params named "tok_emb":   False (Llama convention)
      - all other 2D weights:         True (decay at config.train.weight_decay)
    """
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    masked = {}
    for path, val in flat:
        path_str = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        if val.ndim <= 1:
            decay = False
        elif "tok_emb" in path_str:
            decay = False
        else:
            decay = True
        target = masked
        for p in path[:-1]:
            key = p.key if hasattr(p, "key") else str(p)
            target = target.setdefault(key, {})
        last = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        target[last] = decay
    return masked


def create_train_state(
    key: jax.Array,
    model: nn.Module,
    config: TrainConfig,
    model_cfg: ModelConfig,
) -> train_state.TrainState:
    """Initialize a TrainState: fp32 master params + AdamW opt_state + step=0."""
    dummy_input = jnp.zeros((1, model_cfg.max_seq_len), dtype=jnp.int32)
    dummy_target = jnp.zeros((1, model_cfg.max_seq_len), dtype=jnp.int32)
    params = model.init(key, dummy_input, dummy_target)

    lr_schedule = _make_lr_schedule(config)
    tx = _make_optax(config, lr_schedule)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def save(state: train_state.TrainState, path: Path | str) -> None:
    """Save state to `path` via orbax PyTreeCheckpointer (async on TPU)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    target = {"params": state.params, "opt_state": state.opt_state, "step": int(state.step)}
    # orbax>=0.6 deprecated `force=`; use positional `directory` (was `dir`).
    ocp.PyTreeCheckpointer().save(path, target)


def restore(
    path: Path | str,
    placeholder: train_state.TrainState | None,
) -> train_state.TrainState:
    """Restore state from `path` into `placeholder` (must be same shapes)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if placeholder is None:
        raise ValueError("restore() requires a placeholder TrainState for shapes + apply_fn + tx")
    restored = ocp.PyTreeCheckpointer().restore(path)
    return placeholder.replace(
        params=restored["params"],
        opt_state=restored["opt_state"],
        step=int(restored["step"]),
    )


def _make_lr_schedule(config: TrainConfig):
    import optax

    warmup_steps = min(config.train.warmup_steps, max(1, config.train.total_steps // 2))

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optim.lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=config.train.total_steps,
        end_value=config.optim.lr_min,
    )


def _make_optax(config: TrainConfig, lr_schedule):
    import optax

    tx = optax.chain(
        optax.clip_by_global_norm(config.train.grad_clip),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.optim.b1,
            b2=config.optim.b2,
            eps=config.optim.eps,
            weight_decay=config.train.weight_decay,
            mask=weight_decay_mask,
        ),
    )
    return tx
