"""Causal Multi-Query Attention (MQA) — single KV head shared by all Q heads.

    n_kv_heads == 1   (all heads share one K, one V)

Memory-efficient: K/V cache size is 1/H compared to MHA. Quality drop is
small for small-medium models. Used in PaLM, GPT-J, Falcon.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from model.embeddings.rope import apply_rope


class CausalMQA(nn.Module):
    """Causal Multi-Query Attention.

    Args:
        d_model:    Input/output feature dim.
        n_heads:    Number of Q heads.
        n_kv_heads: Must be 1 for MQA.
        theta_base: RoPE base frequency.
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    theta_base: float = 10000.0

    def setup(self) -> None:
        assert self.n_kv_heads == 1, "CausalMQA requires n_kv_heads == 1"
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        self.W_q = self.param(
            "W_q", nn.initializers.normal(stddev=0.02), (self.d_model, self.n_heads * self.d_head)
        )
        # Single KV head: shape (D, D_h)
        self.W_k = self.param(
            "W_k", nn.initializers.normal(stddev=0.02), (self.d_model, self.d_head)
        )
        self.W_v = self.param(
            "W_v", nn.initializers.normal(stddev=0.02), (self.d_model, self.d_head)
        )
        self.W_o = self.param(
            "W_o", nn.initializers.normal(stddev=0.02), (self.n_heads * self.d_head, self.d_model)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, D = x.shape
        H, D_h = self.n_heads, self.d_head
        q = (x @ self.W_q).reshape(B, T, H, D_h)
        k = (x @ self.W_k).reshape(B, T, 1, D_h)  # (B, T, 1, D_h)
        v = (x @ self.W_v).reshape(B, T, 1, D_h)

        # Broadcast KV across all H heads
        k = jnp.broadcast_to(k, (B, T, H, D_h))
        v = jnp.broadcast_to(v, (B, T, H, D_h))

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
