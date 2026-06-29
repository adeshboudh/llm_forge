"""RoPE — Rotary Position Embedding.

Pure functions (no Flax module state). Applied to Q and K only — V is not
rotated (Llama convention).

Convention (GPT-NeoX / Llama style):
    inv_freq[i]   = 1 / theta_base ** (2i / D_h)        for i in 0..D_h/2
    angle[t, i]   = position[t] * inv_freq[i]
    cos, sin      : (T, D_h/2)
    For each pair (a = x[..., 2i], b = x[..., 2i+1]):
        a' = a * cos - b * sin
        b' = a * sin + b * cos
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _build_cos_sin(
    seq_len: int,
    d_head: int,
    theta_base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pre-compute (cos, sin) tables of shape (T, D_h/2)."""
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, d_head, 2, dtype=jnp.float32) / d_head))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, inv_freq)              # (T, D_h/2)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _rotate(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotation to last dim of x. x: (..., D_h), cos/sin: (T, D_h/2)."""
    x1 = x[..., 0::2]   # even indices: (..., D_h/2)
    x2 = x[..., 1::2]   # odd indices
    # Broadcast cos/sin over leading dims. cos: (T, D_h/2) -> (1, T, 1, D_h/2)
    # so the T axis aligns with the T axis of x (B, T, H, D_h).
    cos_b = jnp.expand_dims(cos, axis=(0, 2))
    sin_b = jnp.expand_dims(sin, axis=(0, 2))
    out1 = x1 * cos_b - x2 * sin_b
    out2 = x1 * sin_b + x2 * cos_b
    # interleave back: (..., 2i) = out1[..., i], (..., 2i+1) = out2[..., i]
    stacked = jnp.stack((out1, out2), axis=-1)         # (..., D_h/2, 2)
    return stacked.reshape(x.shape)


def apply_rope(
    q: jnp.ndarray,
    k: jnp.ndarray,
    theta_base: float = 10000.0,
    positions: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE to Q and K. V is NOT touched (caller passes only q, k).

    Args:
        q, k: (B, T, H, D_h) float32.
        theta_base: RoPE base frequency (default 10000.0).
        positions: Optional (T,) int positions. Defaults to arange(T).

    Returns:
        (q_rot, k_rot): same shapes as inputs.
    """
    B, T, H, D_h = q.shape
    if positions is None:
        positions = jnp.arange(T, dtype=jnp.float32)
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, D_h, 2, dtype=jnp.float32) / D_h))
    angles = jnp.outer(positions.astype(jnp.float32), inv_freq)   # (T, D_h/2)
    cos = jnp.cos(angles).astype(q.dtype)
    sin = jnp.sin(angles).astype(q.dtype)
    return _rotate(q, cos, sin), _rotate(k, cos, sin)


def invert_rope(
    q: jnp.ndarray,
    k: jnp.ndarray,
    theta_base: float = 10000.0,
    positions: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse of apply_rope — rotates by -angle. Used for tests."""
    B, T, H, D_h = q.shape
    if positions is None:
        positions = jnp.arange(T, dtype=jnp.float32)
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, D_h, 2, dtype=jnp.float32) / D_h))
    angles = jnp.outer(positions.astype(jnp.float32), inv_freq)
    cos = jnp.cos(-angles).astype(q.dtype)
    sin = jnp.sin(-angles).astype(q.dtype)
    return _rotate(q, cos, sin), _rotate(k, cos, sin)
