"""Model configuration: dataclass + YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InitConfig:
    embed_std: float = 0.02
    hidden_std: float = 0.02
    norm_scale: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    name: str
    target_params: int
    architecture: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    theta_base: float
    tied_embeddings: bool
    attention: str
    init: InitConfig

    @property
    def d_head(self) -> int:
        """Per-head dimension. Must divide d_model evenly."""
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """Q heads per KV head (GQA repeat factor)."""
        return self.n_heads // self.n_kv_heads

    def to_hf_dict(self) -> dict:
        """Serialize to a HuggingFace-Llama-style config dict.

        Keys mirror LlamaConfig (transformers>=4.40) so the file is
        loadable by HF `LlamaForCausalLM.from_pretrained(...)` after
        parameter reshaping (future work in Phase 6).
        """
        return {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": self.d_model,
            "intermediate_size": self.d_ff,
            "num_attention_heads": self.n_heads,
            "num_hidden_layers": self.n_layers,
            "num_key_value_heads": self.n_kv_heads,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_seq_len,
            "rope_theta": self.theta_base,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": self.tied_embeddings,
            "attention_bias": False,
            "mlp_bias": False,
            "torch_dtype": "float32",
            "transformers_version": "4.40+",
        }


_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "models"


def load_model_config(
    name: str,
    configs_dir: Path | None = None,
) -> ModelConfig:
    """Load a ModelConfig from {configs_dir}/{name}.yaml.

    Args:
        name: Config basename (e.g. "model_25m"); .yaml suffix optional.
        configs_dir: Override configs directory. Defaults to repo configs/models.

    Raises:
        KeyError: If config file not found.
    """
    cfg_dir = configs_dir or _CONFIGS_DIR
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    path = cfg_dir / name
    if not path.exists():
        raise KeyError(f"Config '{name}' not found in {cfg_dir}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    init_raw = raw.pop("init", {})
    init = InitConfig(**init_raw)
    return ModelConfig(**raw, init=init)
