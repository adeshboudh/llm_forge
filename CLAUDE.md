# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision

Pet project for learning LLM internals end-to-end: pretraining, post-training, and inference at small, iterable scale. Target hardware: Kaggle TPU v5e-8 (128GB HBM). Model sizes: 25M → 125M → 350M params.

## Build Phases (strict sequential dependency)

```
Phase 1  →  tokenizer/          BPE 32k, encode/decode, tests
Phase 2  →  data/               FineWeb-Edu pipeline, .npy shards → Kaggle
Phase 3  →  model/              MHA → MQA → GQA → RoPE → SwiGLU → RMSNorm
Phase 4  →  training/           Pretrain loop, AdamW, cosine LR, TPU
Phase 5  →  posttraining/       SFT → LoRA → DPO → RLHF → GRPO
Phase 6  →  inference/          KV cache, sampling, quantization, vLLM, llama.cpp
```

Each phase requires the previous to be working and tested.

## Commands

No build tooling exists yet. When pyproject.toml is created, standard commands will be:

```bash
# install
pip install -e ".[dev]"

# lint / format
ruff check .
ruff format .

# tests (all)
pytest

# single test file
pytest tokenizer/tests/test_bpe.py -v

# single test
pytest tokenizer/tests/test_bpe.py::test_name -v
```

CI scripts will live in `ci/` (lint.sh, format.sh, test.sh, benchmark.sh).

## Architecture

### Module Map
- `tokenizer/` — Custom BPE trainer + runtime encoder/decoder. Vocab size 32,768 (uint16-safe).
- `data/` — FineWeb-Edu ingestion, cleaning, tokenization, shard writing. Shards saved as uint16 `.npy` to Kaggle Datasets.
- `model/` — Model components. Start with `model/attention/variants/mha.py`, then MQA → GQA.
- `training/` — Train loops, optimizers, LR schedulers, checkpointing, TPU setup.
- `posttraining/` — SFT, LoRA/QLoRA, DPO, RLHF/PPO, GRPO with verifiable rewards.
- `inference/` — Generation strategies, quantization, serving (FastAPI), integrations (vLLM, llama.cpp).
- `configs/` — YAML configs for tokenizer, datasets, model sizes, training, inference.
- `experiments/reports/` and `experiments/failed_experiments/` — committed markdown summaries.

### Critical Boundary: kv_cache Split

| Location | What lives here |
|----------|----------------|
| `model/attention/kv_cache/` | Cache **data structure** — key/value tensors, static vs dynamic allocation |
| `inference/kv_cache/` | Cache **management** — paged attention, eviction, multi-request scheduling |

Rule: "what the cache *is*" → `model/`. "How to manage cache across concurrent requests" → `inference/`.

### Special Token IDs (locked after first tokenization)

```python
special_tokens = {
    "<|endoftext|>": 0,   # document separator
    "<|pad|>":        1,
    "<|unk|>":        2,
    "<|bos|>":        3,
    "<|eos|>":        4,
}
# BPE merges fill IDs 5 → 32767
```

These IDs must never change after `.npy` shards are written — changing them invalidates the entire dataset.

### Dataset Versioning

Kaggle Datasets versioned by tokenizer. Never overwrite shards. Format: `llm-forge-tokens-v{N}`. Model configs declare which dataset version they consume via `DataConfig.dataset_version`.

### File Splitting Rule

Split a file when it exceeds ~300 lines **or** has two clearly separable responsibilities. Pre-splitting empty files into micro-modules is premature abstraction — build flat first, refactor when the seam is natural.

## Notebooks

`notebooks/` holds Kaggle `.ipynb` files as read-only reference. Do not edit them — they document experiments run on Kaggle.

## Experiments

`experiments/runs/`, `experiments/logs/`, `experiments/checkpoints/` are git-ignored (runtime artifacts). `experiments/reports/` and `experiments/failed_experiments/` are committed — failed experiment notes are the most valuable long-term asset.
