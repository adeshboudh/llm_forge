"""Training step + eval step — bf16 compute, fp32 master, grad_norm metric.

Mixed-precision boundary: TrainState.params is fp32; inside the loss closure,
2D+ params are cast to bf16 for matmul throughput on TPU v5e MXU. 1D params
(norm scales) stay fp32 to avoid quantization noise in the residual stream.
Grads auto-promote to fp32 (vjp through astype), so AdamW moments accumulate
precisely.

The train loop calls train_step(state, input_ids, target_ids) per step.
train.py sets the global _MODULE_MODEL before the first call (so the jit'd closure
can re-cast params inside a value_and_grad context that needs the model
object, not state.apply_fn).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from model.lm import LM
from training.tpu import (
    make_input_sharding,
    make_loss_sharding,
    make_mesh,
    make_param_sharding,
    setup_devices,
)

_MODULE_MODEL: LM | None = None


def set_model(model: LM) -> None:
    global _MODULE_MODEL
    _MODULE_MODEL = model


def _bf16_cast(params):
    """Cast 2D+ params to bf16; keep 1D params fp32."""
    return jax.tree_util.tree_map(
        lambda p: p.astype(jnp.bfloat16) if p.ndim >= 2 else p,
        params,
    )


def make_loss_fn(model: LM):
    """Return a (params, input_ids, target_ids) -> (loss, grads) closure.

    Used by tests that want to assert on grads without going through train_step.
    """

    @jax.value_and_grad
    def loss_fn(params, input_ids, target_ids):
        return model.apply(_bf16_cast(params), input_ids, target_ids)

    return loss_fn


def _grad_norm(grads):
    leaves = jax.tree_util.tree_leaves(grads)
    return jnp.sqrt(sum(jnp.square(g).sum() for g in leaves))


@jax.value_and_grad
def _loss_for_vg(params, input_ids, target_ids):
    """Shared loss body for train. Uses _MODULE_MODEL (set by train.py)."""
    assert _MODULE_MODEL is not None, "call training.train_step.set_model(model) first"
    return _MODULE_MODEL.apply(_bf16_cast(params), input_ids, target_ids)


def _loss_no_grad(params, input_ids, target_ids):
    """Eval path: no grad. Uses _MODULE_MODEL (set by train.py)."""
    assert _MODULE_MODEL is not None, "call training.train_step.set_model(model) first"
    return _MODULE_MODEL.apply(_bf16_cast(params), input_ids, target_ids)


_MESH = make_mesh(setup_devices())
_INPUT_SHARDING = make_input_sharding(_MESH)
_PARAM_SHARDING = make_param_sharding(_MESH)
_LOSS_SHARDING = make_loss_sharding(_MESH)


@jax.jit(
    in_shardings=(_PARAM_SHARDING, _INPUT_SHARDING, _INPUT_SHARDING),
    out_shardings=None,
)
def _train_loss_and_grad(params, input_ids, target_ids):
    """Sharded train loss + grad. Input batch is split across cores; params + loss are replicated."""
    return _loss_for_vg(params, input_ids, target_ids)


@jax.jit(
    in_shardings=(_PARAM_SHARDING, _INPUT_SHARDING, _INPUT_SHARDING),
    out_shardings=None,
)
def _eval_loss_jit(params, input_ids, target_ids):
    return _loss_no_grad(params, input_ids, target_ids)


def train_step(state, input_ids, target_ids):
    """One training step. Returns (new_state, loss, metrics)."""
    loss, grads = _train_loss_and_grad(state.params, input_ids, target_ids)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, {"grad_norm": _grad_norm(grads)}


def eval_step(state, input_ids, target_ids):
    """Eval: returns scalar loss only (no grads, no param update)."""
    return _eval_loss_jit(state.params, input_ids, target_ids)
