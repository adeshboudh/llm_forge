# llm-forge

Learning LLM internals end-to-end — pretraining, post-training, inference.

Compute: Kaggle TPU v5e-8, Lightning AI. Scale: 25M → 125M → 350M params.

## Structure

| Folder | Purpose |
|--------|---------|
| `notes/` | Phase-by-phase learning notes (feeds the website) |
| `code/` | Reference implementations (run on Kaggle / Lightning AI) |
| `notebooks/` | Kaggle `.ipynb` files — read-only reference |
| `experiments/` | Reports + failed experiment notes (committed) |
| `configs/` | YAML configs referenced in Kaggle runs |
| `web/` | Next.js site built from `notes/` |

## Phases

```
Phase 1  →  tokenizer     BPE 32k, encode/decode
Phase 2  →  data          FineWeb-Edu pipeline, .npy shards
Phase 3  →  model         MHA → MQA → GQA → RoPE → SwiGLU → RMSNorm
Phase 4  →  training      Pretrain loop, AdamW, cosine LR, TPU
Phase 5  →  posttraining  SFT → LoRA → DPO → RLHF → GRPO
Phase 6  →  inference     KV cache, sampling, quantization, vLLM
```
