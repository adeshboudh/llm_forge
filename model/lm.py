"""Full causal LM — Llama-style with tied embeddings.

x      = tok_emb[input_ids]
for block in blocks: x = block(x)
x      = final_norm(x)
logits = x @ tok_emb.T          # tied lm_head
loss   = mean(cross_entropy(logits, target_ids))
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from model.blocks.transformer_block import TransformerBlock
from model.config import ModelConfig
from model.normalization.rmsnorm import RMSNorm


class LM(nn.Module):
    """Causal transformer LM (Llama-style).

    Attributes:
        config: ModelConfig — sized presets from configs/models/{name}.yaml.
    """

    config: ModelConfig

    def setup(self) -> None:
        cfg = self.config
        self.tok_emb = self.param(
            "tok_emb",
            nn.initializers.normal(stddev=cfg.init.embed_std),
            (cfg.vocab_size, cfg.d_model),
        )
        self.blocks = [TransformerBlock(config=cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(dim=cfg.d_model)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        target_ids: jnp.ndarray,
        *,
        return_logits: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning scalar cross-entropy loss.

        Args:
            input_ids:  (B, T) int in [0, vocab).
            target_ids: (B, T) int — typically input_ids shifted-L by 1.
            return_logits: If True, also return logits (B, T, V).

        Returns:
            loss (scalar) or (loss, logits) if return_logits=True.
        """
        x = self.tok_emb[input_ids]  # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.tok_emb.T  # tied (B, T, V)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_p = jnp.take_along_axis(
            log_probs,
            target_ids[..., None],
            axis=-1,
        ).squeeze(-1)
        loss = jnp.mean(-log_p)
        if return_logits:
            return loss, logits
        return loss
