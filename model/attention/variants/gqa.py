"""Causal Grouped-Query Attention (GQA) — N KV heads, each serves H/N Q heads.

    n_kv_heads divides n_heads evenly   (n_kv=H -> MHA; n_kv=1 -> MQA)

Production standard (Llama 2/3, Mistral, Mixtral): KV cache is n_kv_heads/H
the size of MHA, with negligible quality drop.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from model.embeddings.rope import apply_rope


class CausalGQA(nn.Module):
    """Causal Grouped-Query Attention.

    Args:
        d_model:    Input/output feature dim.
        n_heads:    Number of Q heads.
        n_kv_heads: Number of KV heads. Must divide n_heads.
        theta_base: RoPE base frequency.
    """
    d_model: int
    n_heads: int
    n_kv_heads: int
    theta_base: float = 10000.0

    def setup(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_kv_heads ({self.n_kv_heads}) must divide n_heads ({self.n_heads})"
        )
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
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
        H, n_kv, D_h = self.n_heads, self.n_kv_heads, self.d_head
        q = (x @ self.W_q).reshape(B, T, H, D_h)
        k = (x @ self.W_k).reshape(B, T, n_kv, D_h)
        v = (x @ self.W_v).reshape(B, T, n_kv, D_h)

        # repeat_interleave: each kv head serves n_rep q heads
        # (B, T, n_kv, D_h) -> (B, T, n_kv, n_rep, D_h) -> (B, T, H, D_h)
        k = jnp.repeat(k[:, :, :, None, :], self.n_rep, axis=3).reshape(B, T, H, D_h)
        v = jnp.repeat(v[:, :, :, None, :], self.n_rep, axis=3).reshape(B, T, H, D_h)

        q, k = apply_rope(q, k, theta_base=self.theta_base)

        # Transpose to (B, H, T, D_h) for einsum
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D_h)
        mask = jnp.tril(jnp.ones((T, T)))
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D_h)
        return out @ self.W_o
