"""Export a trained Llama-style model to a HuggingFace-compatible directory.

HF expects this layout::

    output_dir/
    ├── config.json              # architecture
    ├── params.safetensors       # model weights (fp32 master)
    ├── tokenizer.json           # 32k BPE tokenizer
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── generation_config.json   # bos / eos / max_length
    └── README.md                # training metadata

The result is loadable by ``huggingface_hub.snapshot_download``
and (after a future weight-reshape step in Phase 6) by
``transformers.LlamaForCausalLM.from_pretrained``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from model.config import ModelConfig
from training.export import save_params_safetensors

_TOKENIZER_SRC = Path(__file__).resolve().parent.parent / "tokenizer"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _copy_tokenizer(output_dir: Path) -> None:
    """Copy the 32k BPE tokenizer artifacts into the HF repo."""
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        src = _TOKENIZER_SRC / name
        if not src.exists():
            continue
        shutil.copy2(src, output_dir / name)


def _write_generation_config(output_dir: Path, cfg: ModelConfig) -> None:
    _write_json(
        output_dir / "generation_config.json",
        {
            "bos_token_id": 3,  # <|bos|>
            "eos_token_id": 4,  # <|eos|>
            "pad_token_id": 1,  # <|pad|>
            "max_length": cfg.max_seq_len,
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
        },
    )


def _write_readme(
    output_dir: Path,
    cfg: ModelConfig,
    *,
    tokens_trained: int,
    num_steps: int,
    final_loss: float | None,
    repo_id: str | None,
) -> None:
    body = [
        "---",
        "license: apache-2.0",
        "tags:",
        "  - llm",
        "  - llama",
        "  - jax",
        "  - flax",
        "  - pretraining",
        "  - from-scratch",
        "datasets:",
        "  - HuggingFaceFW/fineweb-edu",
        "language:",
        "  - en",
        "library_name: jax",
        f"model_name: {repo_id or cfg.name}",
        "---",
        "",
        f"# {repo_id or cfg.name}",
        "",
        f"Llama-style transformer trained from scratch in JAX/Flax.",
        "",
        f"- **Architecture:** {cfg.architecture}",
        f"- **Target size:** {cfg.target_params / 1e6:.0f}M parameters",
        f"- **Layers:** {cfg.n_layers}",
        f"- **d_model:** {cfg.d_model}",
        f"- **Heads:** {cfg.n_heads} Q / {cfg.n_kv_heads} KV (GQA repeat={cfg.n_rep})",
        f"- **d_ff:** {cfg.d_ff}",
        f"- **Vocab:** {cfg.vocab_size} (BPE 32k)",
        f"- **Max seq len:** {cfg.max_seq_len}",
        f"- **RoPE theta:** {cfg.theta_base}",
        f"- **Tied embeddings:** {cfg.tied_embeddings}",
        "",
        "## Training",
        "",
        f"- **Tokens trained:** {tokens_trained:,}",
        f"- **Steps:** {num_steps:,}",
    ]
    if final_loss is not None:
        body.append(f"- **Final loss:** {final_loss:.4f}")
    body.extend(
        [
            "",
            "## Files",
            "",
            "- `params.safetensors` — model weights (fp32)",
            "- `config.json` — architecture (HF Llama format)",
            "- `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json` — 32k BPE",
            "- `generation_config.json` — default sampling parameters",
            "",
            "## Usage (JAX/Flax, this repo)",
            "",
            "```python",
            "from huggingface_hub import snapshot_download",
            "from model.config import load_model_config",
            "from model.lm import LM",
            "from training.export import load_params_safetensors",
            "",
            f'path = snapshot_download("{repo_id or cfg.name}")',
            f"cfg = load_model_config('{cfg.name}')  # or load_model_config('model_<size>')",
            "model = LM(config=cfg)",
            "flat = load_params_safetensors(f'{path}/params.safetensors')",
            "```",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(body) + "\n")


def export_hf(
    model_cfg: ModelConfig,
    params: Any,
    output_dir: str | Path,
    *,
    tokens_trained: int = 0,
    num_steps: int = 0,
    final_loss: float | None = None,
    repo_id: str | None = None,
    include_tokenizer: bool = True,
) -> Path:
    """Write a HuggingFace-compatible directory for one model size.

    Args:
        model_cfg: Architecture config (d_model, n_layers, etc.).
        params: Flax params pytree (``state.params``).
        output_dir: Destination directory; created if missing.
        tokens_trained: Number of tokens the model has seen (for README).
        num_steps: Training step count (for README).
        final_loss: Last training loss (for README; optional).
        repo_id: Target HF repo name (e.g. ``adesh01/llm_forge-25m``).
        include_tokenizer: If True, copy the 32k BPE tokenizer artifacts.

    Returns:
        The output directory as a ``Path``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "config.json", model_cfg.to_hf_dict())
    save_params_safetensors(params, output_dir / "params.safetensors")
    _write_generation_config(output_dir, model_cfg)
    if include_tokenizer:
        _copy_tokenizer(output_dir)
    _write_readme(
        output_dir,
        model_cfg,
        tokens_trained=tokens_trained,
        num_steps=num_steps,
        final_loss=final_loss,
        repo_id=repo_id,
    )
    return output_dir


def push_to_hub(
    output_dir: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "Upload trained model",
    token: str | None = None,
) -> str:
    """Upload an exported HF directory to the Hub.

    Args:
        output_dir: Local directory written by ``export_hf``.
        repo_id: e.g. ``adesh01/llm_forge-25m``.
        private: If True, create a private repo. Default public.
        commit_message: Git commit message.
        token: HF write token. Resolution order:
            1. ``token=`` argument
            2. ``HF_TOKEN`` env var
            3. ``huggingface-cli login`` cache

    Returns:
        The repo URL on the Hub.

    Raises:
        RuntimeError: With setup instructions if no token is found.
    """
    import os

    from huggingface_hub import HfApi

    if token is None:
        token = os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError(
            "No HF token found. On Kaggle, add a 'HF_TOKEN' secret "
            "(https://huggingface.co/settings/tokens, write scope) and "
            "expose it in the notebook:\n"
            "  from kaggle_secrets import UserSecretsClient\n"
            "  import os; os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\n"
            "Or set os.environ['HF_TOKEN'] = 'hf_...' inline."
        )

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"
