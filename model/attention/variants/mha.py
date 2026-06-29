"""Causal Multi-Head Attention (MHA) — baseline variant.

    n_kv_heads == n_heads   (every Q head has its own K, V)

Standard MHA: RoPE on Q,K; causal mask; softmax; scale 1/sqrt(d_head).
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from model.embeddings.rope import apply_rope


class CausalMHA(nn.Module):
    """Causal Multi-Head Attention.

    Args:
        d_model:    Input/output feature dim.
        n_heads:    Number of Q heads.
        n_kv_heads: Number of KV heads. For MHA, == n_heads.
        theta_base: RoPE base frequency.
    """
    d_model: int
    n_heads: int
    n_kv_heads: int
    theta_base: float = 10000.0

    def setup(self) -> None:
        assert self.n_kv_heads == self.n_heads, (
            "CausalMHA requires n_kv_heads == n_heads"
        )
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        self.W_q = self.param("W_q", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_heads * self.d_head))
        self.W_k = self.param("W_k", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_kv_heads * self.d_head))
        self.W_v = self.param("W_v", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_kv_heads * self.d_head))
        self.W_o = self.param("W_o", nn.initializers.normal(stddev=0.02),
                              (self.n_heads * self.d_head, self.d_model))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, D = x.shape
        H, D_h = self.n_heads, self.d_head
        q = (x @ self.W_q).reshape(B, T, H, D_h)
        k = (x @ self.W_k).reshape(B, T, H, D_h)   # n_kv == H for MHA
        v = (x @ self.W_v).reshape(B, T, H, D_h)

        q, k = apply_rope(q, k, theta_base=self.theta_base)

        # (B, T, H, D_h) -> (B, H, T, D_h) for einsum
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # (B, H, T, D_h) @ (B, H, D_h, T) -> (B, H, T, T)
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D_h)
        mask = jnp.tril(jnp.ones((T, T)))                # (T, T)
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)     # (B, H, T, D_h)
        out = out.reshape(B, T, H * D_h)
        return out @ self.W_o
