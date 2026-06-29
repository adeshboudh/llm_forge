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
