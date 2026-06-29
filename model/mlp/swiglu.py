"""SwiGLU MLP — Llama-style feedforward.

    gate = silu(x @ W_gate)
    up   = x @ W_up
    h    = gate * up
    y    = h @ W_down

No biases. d_ff defaults to round(8/3 * d_model / 256) * 256 (Llama ratio,
rounded to a multiple of 256 for hardware friendliness).
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def compute_d_ff(d_model: int, multiple_of: int = 256) -> int:
    """Pick d_ff = round(8/3 * d_model / multiple_of) * multiple_of."""
    raw = (8.0 / 3.0) * d_model
    rounded = round(raw / multiple_of)
    return rounded * multiple_of


class SwiGLUMLP(nn.Module):
    """SwiGLU feedforward block.

    Args:
        d_model: Input/output feature dim.
        d_ff:    Intermediate dim.
    """
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        W_gate = self.param(
            "W_gate",
            nn.initializers.normal(stddev=0.02),
            (self.d_model, self.d_ff),
        )
        W_up = self.param(
            "W_up",
            nn.initializers.normal(stddev=0.02),
            (self.d_model, self.d_ff),
        )
        W_down = self.param(
            "W_down",
            nn.initializers.normal(stddev=0.02),
            (self.d_ff, self.d_model),
        )
        gate = jax.nn.silu(x @ W_gate)
        up = x @ W_up
        h = gate * up
        return h @ W_down
