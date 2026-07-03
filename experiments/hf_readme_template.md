---
license: apache-2.0
tags:
  - llm
  - llama
  - jax
  - flax
  - pretraining
  - from-scratch
datasets:
  - HuggingFaceFW/fineweb-edu
language:
  - en
library_name: jax
model_name: adesh01/llm_forge-25m
---

# adesh01/llm_forge-25m

Llama-style transformer trained from scratch in JAX/Flax.

- **Architecture:** llama
- **Target size:** 25M parameters
- **Layers:** 4
- **d_model:** 512
- **Heads:** 8 Q / 4 KV (GQA repeat=2)
- **d_ff:** 1280
- **Vocab:** 32768 (BPE 32k)
- **Max seq len:** 1024
- **RoPE theta:** 10000.0
- **Tied embeddings:** True

## Training

- **Tokens trained:** 1,280,049,152
- **Steps:** 9,766
- **Final loss:** 3.8750

## Files

- `params.safetensors` — model weights (fp32)
- `config.json` — architecture (HF Llama format)
- `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json` — 32k BPE
- `generation_config.json` — default sampling parameters

## Usage (JAX/Flax, this repo)

```python
from huggingface_hub import snapshot_download
from model.config import load_model_config
from model.lm import LM
from training.export import load_params_safetensors

path = snapshot_download("adesh01/llm_forge-25m")
cfg = load_model_config('model_25m')  # or load_model_config('model_<size>')
model = LM(config=cfg)
flat = load_params_safetensors(f'{path}/params.safetensors')
```
