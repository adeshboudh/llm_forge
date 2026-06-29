"""RMSNorm — Root Mean Square Layer Normalization (no mean subtraction).

    out = scale * x / sqrt(mean(x^2) + eps)

Llama convention: no bias, learnable scale only.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale (no bias).

    Args:
        dim: Feature dimension (last axis).
        eps: Numerical stability constant. Default 1e-6.
    """
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (self.dim,),
        )
        ms = jnp.mean(x * x, axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(ms + self.eps)
        return scale.astype(x.dtype) * x_norm
