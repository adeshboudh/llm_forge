# llm-forge — Project Structure

> A long-term pet project for learning LLM internals end-to-end:
> pretraining, post-training, and inference — at small, iterable scale.

---

## Compute & Scale Targets

| Axis | Choice |
|------|--------|
| Hardware | Kaggle TPU v5e-8 (128GB HBM total) |
| Model sizes | 25M → 125M → 350M params |
| Tokenizer | Custom BPE, 32,768 vocab (2¹⁵, uint16-safe) |
| Pretraining data | FineWeb-Edu `sample-10BT`, 2B token slice |
| Post-training data | GSM8K / small domain-specific, verifiable rewards |

---

## Folder Structure

```
llm-forge/
│
├── README.md
├── ROADMAP.md                          # phased learning milestones
├── STRUCTURE.md                        # kv_cache boundary decisions + notes
├── LICENSE
├── .gitignore
├── pyproject.toml                      # dependency + build management
│
├── requirements/
│   ├── base.txt
│   ├── train.txt
│   ├── inference.txt
│   └── dev.txt
│
├── configs/                            # global experiment configs (YAML)
│   ├── tokenizer/
│   │   ├── bpe_32k.yaml
│   │   └── bpe_50k.yaml
│   ├── datasets/
│   │   ├── tinystories.yaml            # smoke tests, fast iteration
│   │   ├── fineweb_edu.yaml
│   │   └── openwebtext.yaml
│   ├── models/
│   │   ├── 25m.yaml
│   │   ├── 125m.yaml
│   │   ├── 350m.yaml
│   │   └── 1b.yaml
│   ├── training/
│   │   ├── pretrain.yaml
│   │   ├── finetune.yaml
│   │   └── tpu_v5e.yaml
│   └── inference/
│       ├── fp16.yaml
│       ├── int8.yaml
│       └── gguf.yaml
│
├── docs/                               # deep technical notes per domain
│   ├── architecture.md
│   ├── tokenizer.md
│   ├── training.md
│   ├── inference.md
│   ├── optimization.md
│   ├── tpu.md
│   ├── quantization.md
│   ├── scaling-laws.md
│   └── experiments/                    # one dated .md per experiment run
│
├── notebooks/                          # Kaggle .ipynb files, read-only reference
│   ├── tokenizer_experiments/
│   ├── training_debug/
│   ├── tpu_tests/
│   └── profiling/
│
├── scripts/                            # CLI utility scripts
│   ├── train_tokenizer.sh
│   ├── preprocess_data.sh
│   ├── launch_tpu.sh
│   ├── benchmark.sh
│   └── export_gguf.sh
│
│
│   ┌─────────────────────────────────────────────────────────┐
│   │  PHASE 1 — Tokenizer                                     │
│   └─────────────────────────────────────────────────────────┘
│
├── tokenizer/
│   ├── README.md
│   ├── trainers/
│   │   ├── base.py
│   │   ├── bpe.py
│   │   ├── sentencepiece.py
│   │   └── unigram.py
│   ├── vocab/
│   │   ├── merges.py
│   │   ├── special_tokens.py           # <|endoftext|>=0, <|pad|>=1, <|unk|>=2, <|bos|>=3, <|eos|>=4
│   │   └── vocab_builder.py
│   ├── preprocessing/
│   │   ├── cleaners.py
│   │   ├── normalization.py
│   │   └── unicode_handling.py
│   ├── runtime/
│   │   ├── encode.py
│   │   ├── decode.py
│   │   └── fast_tokenizer.py
│   ├── serialization/
│   │   ├── save.py
│   │   └── load.py
│   └── tests/
│       ├── test_bpe.py
│       ├── test_encode_decode.py
│       └── test_special_tokens.py
│
│
│   ┌─────────────────────────────────────────────────────────┐
│   │  PHASE 2 — Data Pipeline                                 │
│   └─────────────────────────────────────────────────────────┘
│
├── data/
│   ├── README.md
│   ├── sources/
│   │   ├── fineweb.py
│   │   ├── openwebtext.py
│   │   ├── cosmopedia.py
│   │   └── tinystories.py
│   ├── cleaning/
│   │   ├── deduplication.py
│   │   ├── filtering.py
│   │   ├── language_filter.py
│   │   └── quality_scoring.py
│   ├── preprocessing/
│   │   ├── tokenize_dataset.py
│   │   ├── sequence_packing.py
│   │   ├── shard_writer.py             # saves uint16 .npy shards to Kaggle Dataset
│   │   └── metadata.py
│   ├── loaders/
│   │   ├── npy_loader.py
│   │   ├── streaming_loader.py
│   │   ├── packed_loader.py
│   │   └── distributed_loader.py
│   ├── formats/
│   │   ├── npy_format.py
│   │   ├── mmap_format.py
│   │   └── indexed_dataset.py
│   └── tests/
│       ├── test_packing.py
│       ├── test_sharding.py
│       └── test_loader.py
│
│
│   ┌─────────────────────────────────────────────────────────┐
│   │  PHASE 3 — Model Architecture  (core curriculum)         │
│   └─────────────────────────────────────────────────────────┘
│
├── model/
│   ├── README.md
│   ├── configs/
│   │   ├── base_config.py
│   │   └── registry.py
│   ├── embeddings/
│   │   ├── token_embedding.py
│   │   ├── positional.py
│   │   ├── rotary.py                   # RoPE
│   │   └── alibi.py
│   ├── attention/
│   │   ├── base_attention.py
│   │   ├── masking.py
│   │   ├── attention_ops.py
│   │   ├── variants/
│   │   │   ├── mha.py                  # ← start here
│   │   │   ├── mqa.py                  # reduce query heads
│   │   │   ├── gqa.py                  # production standard (LLaMA 2+)
│   │   │   ├── sliding_window.py
│   │   │   └── flash_attention.py      # optimization of MHA, not a sibling concept
│   │   ├── kv_cache/                   # cache DATA STRUCTURE lives here (not inference)
│   │   │   ├── cache.py                # static / dynamic cache
│   │   │   ├── paged_cache.py          # paged attention (memory layout)
│   │   │   └── cache_manager.py
│   │   └── tests/
│   │       ├── test_attention.py
│   │       ├── test_masking.py
│   │       └── test_kv_cache.py
│   ├── mlp/
│   │   ├── gelu.py
│   │   ├── swiglu.py
│   │   └── feedforward.py
│   ├── normalization/
│   │   ├── layernorm.py
│   │   ├── rmsnorm.py
│   │   └── fused_norm.py
│   ├── blocks/
│   │   ├── transformer_block.py
│   │   └── residual.py
│   ├── architectures/
│   │   ├── gpt2.py
│   │   ├── llama.py
│   │   └── tiny_transformer.py         # smoke-test architecture, no frills
│   ├── losses/
│   │   ├── cross_entropy.py
│   │   └── auxiliary_losses.py
│   ├── initialization/
│   │   ├── weight_init.py
│   │   └── scaling_rules.py
│   └── tests/
│       ├── test_transformer.py
│       ├── test_shapes.py
│       └── test_forward_pass.py
│
│
│   ┌─────────────────────────────────────────────────────────┐
│   │  PHASE 4 — Pretraining                                   │
│   └─────────────────────────────────────────────────────────┘
│
├── training/
│   ├── README.md
│   ├── loops/
│   │   ├── pretrain_loop.py
│   │   ├── finetune_loop.py
│   │   └── eval_loop.py
│   ├── optimizers/
│   │   ├── adamw.py
│   │   ├── lion.py
│   │   └── optimizer_factory.py
│   ├── schedulers/
│   │   ├── cosine.py
│   │   ├── warmup.py
│   │   └── lr_factory.py
│   ├── checkpointing/
│   │   ├── save.py
│   │   ├── restore.py
│   │   ├── safetensors_io.py
│   │   └── shard_checkpoints.py
│   ├── precision/
│   │   ├── bf16.py
│   │   ├── fp16.py
│   │   └── loss_scaling.py
│   ├── distributed/
│   │   ├── data_parallel.py
│   │   ├── tensor_parallel.py
│   │   ├── fsdp.py
│   │   └── sharding.py
│   ├── profiling/
│   │   ├── throughput.py
│   │   ├── memory.py
│   │   ├── flop_counter.py
│   │   └── compile_time.py
│   ├── logging/
│   │   ├── wandb_logger.py
│   │   ├── tensorboard_logger.py
│   │   └── metrics.py
│   ├── tpu/
│   │   ├── jax_setup.py
│   │   ├── mesh.py
│   │   ├── pjit_utils.py
│   │   └── kaggle_tpu.py               # Kaggle TPU v5e-8 init helpers
│   └── tests/
│       ├── test_checkpointing.py
│       ├── test_training_step.py
│       └── test_distributed.py
│
│
│   ┌─────────────────────────────────────────────────────────┐
│   │  PHASE 5 — Post-Training  (needs a trained model first)  │
│   └─────────────────────────────────────────────────────────┘
│
├── posttraining/
│   ├── sft/
│   │   ├── datasets/
│   │   ├── templates/
│   │   ├── trainer.py
│   │   └── eval.py
│   ├── lora/
│   │   ├── layers.py
│   │   ├── qlora.py
│   │   ├── merge.py
│   │   └── trainer.py
│   ├── dpo/
│   │   ├── preference_dataset.py
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── rlhf/
│   │   ├── reward_model/
│   │   ├── ppo/
│   │   ├── rollout/
│   │   └── trainer.py
│   └── grpo/
│       ├── sampler.py
│       ├── rewards.py                  # verifiable rewards (GSM8K-style)
│       └── trainer.py
│
│
│   ┌─────────────────────────────────────────────────────────┐
│   │  PHASE 6 — Inference                                     │
│   └─────────────────────────────────────────────────────────┘
│
├── inference/
│   ├── README.md
│   ├── generation/
│   │   ├── greedy.py
│   │   ├── beam_search.py
│   │   ├── topk.py
│   │   ├── topp.py
│   │   ├── speculative.py
│   │   └── penalties.py
│   ├── runtime/
│   │   ├── engine.py
│   │   ├── scheduler.py
│   │   ├── token_streamer.py
│   │   └── batching.py
│   ├── kv_cache/                       # serving-side cache MANAGEMENT only
│   │   ├── static_cache.py             # (cache data structure → model/attention/kv_cache/)
│   │   ├── dynamic_cache.py
│   │   └── paged_attention.py          # memory allocation for serving
│   ├── quantization/
│   │   ├── int8.py
│   │   ├── gptq.py
│   │   ├── awq.py
│   │   ├── gguf_export.py
│   │   └── calibration.py
│   ├── serving/
│   │   ├── fastapi_server.py
│   │   ├── websocket_server.py
│   │   ├── request_schema.py
│   │   └── docker/                     # Dockerfile + serving configs live here
│   ├── integrations/
│   │   ├── vllm_backend.py
│   │   ├── llama_cpp_backend.py
│   │   └── transformers_backend.py
│   ├── benchmarks/
│   │   ├── latency.py
│   │   ├── throughput.py
│   │   ├── memory_usage.py
│   │   └── cache_efficiency.py
│   └── tests/
│       ├── test_sampling.py
│       ├── test_generation.py
│       └── test_quantization.py
│
│
│   ┌─────────────────────────────────────────────────────────┐
│   │  Supporting Infrastructure                               │
│   └─────────────────────────────────────────────────────────┘
│
├── evals/
│   ├── perplexity/
│   ├── lm_harness/
│   ├── benchmarks/
│   ├── safety/
│   └── reasoning/
│
├── profiling/
│   ├── memory/
│   ├── kernels/
│   ├── tpu/
│   ├── latency/
│   └── traces/
│
├── experiments/
│   ├── runs/                           # git-ignored, runtime artifacts
│   ├── logs/                           # git-ignored
│   ├── checkpoints/                    # git-ignored
│   ├── reports/                        # .md summaries, committed
│   └── failed_experiments/             # what didn't work + why, committed
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── distributed/
│   ├── inference/
│   └── regression/
│
├── ci/
│   ├── lint.sh
│   ├── format.sh
│   ├── test.sh
│   └── benchmark.sh
│
└── research/
    ├── papers/
    ├── reproductions/
    ├── scaling_laws/
    ├── moe/                            # future: Mixture of Experts
    ├── multimodal/                     # future scope marker
    └── future_work/
```

---

## Key Boundaries

### kv_cache Split (important — do not blur this line)

| Location | Responsibility |
|----------|----------------|
| `model/attention/kv_cache/` | Cache **data structure** — how keys/values are stored, static vs dynamic allocation, the cache tensor itself |
| `inference/kv_cache/` | Cache **management for serving** — paged attention memory allocation, eviction policy, multi-request scheduling |

Rule of thumb: if it's about *what the cache is*, it's in `model/`. If it's about *how to manage cache across concurrent requests*, it's in `inference/`.

### What Lives in `experiments/`

| Subfolder | Git status | Contents |
|-----------|------------|----------|
| `runs/` | ignored | W&B run artifacts, raw metrics |
| `logs/` | ignored | Training logs, loss curves |
| `checkpoints/` | ignored | Model weights (too large) |
| `reports/` | committed | Dated `.md` summaries of what you found |
| `failed_experiments/` | committed | What didn't work and why — most valuable over time |

---

## Dataset Versioning Convention

Kaggle Datasets must be versioned by tokenizer. Never overwrite shards.

```
llm-forge-tokens-v1   →  bpe32k  · fineweb-edu · 2B tokens
llm-forge-tokens-v2   →  bpe32k  · fineweb-edu · 5B tokens   (future)
llm-forge-tokens-v3   →  unigram32k · fineweb-edu · 2B tokens (if tokenizer changes)
```

Model configs declare which dataset version they consume:

```python
@dataclass
class DataConfig:
    dataset_version: str = "v1-bpe32k-fineweb2B"
    tokenizer_path: str = "./tokenizer/tokenizer.json"
    vocab_size: int = 32_768
    shard_dir: str = "/kaggle/input/llm-forge-tokens-v1/"
    total_tokens: int = 2_000_000_000
```

---

## Special Token IDs (locked — never change after first tokenization)

```python
special_tokens = {
    "<|endoftext|>": 0,   # document separator
    "<|pad|>":        1,   # padding
    "<|unk|>":        2,   # unknown
    "<|bos|>":        3,   # beginning of sequence
    "<|eos|>":        4,   # end of sequence
}
# BPE merges fill IDs 5 → 32767
```

Changing these after `.npy` shards are written invalidates the entire dataset.

---

## Build Order (Phases)

```
Phase 1  →  tokenizer/          Train BPE 32k, encode/decode, tests
Phase 2  →  data/               FineWeb-Edu pipeline, .npy shards to Kaggle
Phase 3  →  model/              MHA → MQA → GQA → RoPE → SwiGLU → RMSNorm
Phase 4  →  training/           Pretrain loop, AdamW, cosine LR, TPU setup
Phase 5  →  posttraining/       SFT → LoRA → DPO → RLHF → GRPO
Phase 6  →  inference/          KV cache, sampling, quantization, vLLM, llama.cpp
```

Each phase requires the previous phase to be working and tested before starting.

---

## File Splitting Rule

> Split a file when it exceeds ~300 lines **or** has two clearly separable responsibilities.
> Pre-splitting empty files into micro-modules is premature abstraction.
> The structure above is a **future map** — build flat first, refactor when the seam is natural.
