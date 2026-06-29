"""TransformerBlock — pre-norm residual block.

    h   = x + attn(norm1(x))
    out = h + mlp(norm2(h))

Variant selectable via config.attention: "mha" | "mqa" | "gqa".
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from model.attention.variants.mha import CausalMHA
from model.attention.variants.mqa import CausalMQA
from model.attention.variants.gqa import CausalGQA
from model.config import ModelConfig
from model.mlp.swiglu import SwiGLUMLP
from model.normalization.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """One transformer decoder block (Llama-style).

    Attributes:
        config: ModelConfig providing d_model, n_heads, n_kv_heads, attention, etc.
    """
    config: ModelConfig

    def setup(self) -> None:
        cfg = self.config
        self.norm1 = RMSNorm(dim=cfg.d_model)
        self.norm2 = RMSNorm(dim=cfg.d_model)
        if cfg.attention == "mha":
            self.attn = CausalMHA(d_model=cfg.d_model, n_heads=cfg.n_heads,
                                  n_kv_heads=cfg.n_heads, theta_base=cfg.theta_base)
        elif cfg.attention == "mqa":
            self.attn = CausalMQA(d_model=cfg.d_model, n_heads=cfg.n_heads,
                                  n_kv_heads=1, theta_base=cfg.theta_base)
        elif cfg.attention == "gqa":
            self.attn = CausalGQA(d_model=cfg.d_model, n_heads=cfg.n_heads,
                                  n_kv_heads=cfg.n_kv_heads, theta_base=cfg.theta_base)
        else:
            raise ValueError(f"unknown attention variant: {cfg.attention}")
        self.mlp = SwiGLUMLP(d_model=cfg.d_model, d_ff=cfg.d_ff)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x + self.attn(self.norm1(x))
        out = h + self.mlp(self.norm2(h))
        return out
