# llm-forge — Progress Log

Bird's-eye view of what's done, open questions, decisions made, and blockers per phase.
Updated manually after each significant milestone.

---

## Phase 1 — Tokenizer ✅ COMPLETE

**Status:** All code written, tested, trained. Tokenizer committed to git.

### What's done

| Artifact              | Location                                      | Notes                                 |
| --------------------- | --------------------------------------------- | ------------------------------------- |
| Python BPE trainer    | `tokenizer/trainers/bpe.py`                   | Reference implementation, educational |
| Rust BPE trainer      | `tokenizer/trainers/bpe_rust/`                | ~50× faster, parallel via rayon       |
| Corpus downloader     | `tokenizer/trainers/download_corpus.py`       | Streams FineWeb-Edu → stdout          |
| Encoder               | `tokenizer/runtime/encode.py`                 | GPT-2 pretokenization, greedy merge   |
| Decoder               | `tokenizer/runtime/decode.py`                 | byte-level, skip_special_tokens       |
| Serialization         | `tokenizer/serialization/save.py` + `load.py` | tokenizer.json canonical format       |
| Training script       | `tokenizer/train_tokenizer.py`                | Python trainer CLI (slow, use Rust)   |
| Tests                 | `tokenizer/tests/test_bpe.py`                 | 68 tests, all pass                    |
| **Trained tokenizer** | `tokenizer/saved/tokenizer.json`              | **32k vocab, 1B chars FineWeb-Edu**   |
| Config                | `configs/tokenizer/bpe_32k.yaml`              | Reference YAML, no code wiring        |

### Training run details

- Dataset: FineWeb-Edu `sample-10BT`, 1B chars, 210,983 docs
- Unique pre-tokens: 1,408,047
- Merges learned: 32,507
- Wall time: ~7.8 hours (8-core laptop, Rust trainer)
- Tool: `download_corpus.py | bpe-trainer --output-dir tokenizer/saved/`

### Decisions made

- **Byte-level BPE, GPT-2 pretokenization** — handles all Unicode via UTF-8 bytes
- **vocab_size = 32,768** — 2¹⁵, fits uint16 (max shard token ID = 32,767)
- **Special tokens LOCKED at IDs 0–4** — never change after first .npy shard write
- **Vocab layout**: 0–4 specials, 5–260 bytes, 261–32,767 BPE merges
- **Rust trainer for actual training** — Python trainer kept as learning reference only
- **Tokenizer committed to git** (2MB) — 8h training artifact, too important to lose

### Open questions

- None blocking. Tokenizer is locked.

---

## Phase 2 — Data Pipeline ✅ COMPLETE (with overshoot — see note)

**Status:** All code written, tested, shards generated and pushed to Kaggle.
First run overshot the 10B budget by 7.5% (mid-shard flush race). Hard-cap fix landed afterward; future runs land at exactly the budget.

### What's done

| Artifact           | Location                                 | Notes                                   |
| ------------------ | ---------------------------------------- | --------------------------------------- |
| FineWeb-Edu source | `data/sources/fineweb.py`                | Streaming, char-budget stop, tqdm bar   |
| Document tokenizer | `data/preprocessing/tokenize_dataset.py` | Prepends EOT (ID=0), tqdm bar           |
| Shard writer       | `data/preprocessing/shard_writer.py`     | uint16 .npy, 50M tokens/shard, tqdm bar |
| Data loader        | `data/loaders/npy_loader.py`             | Lazy shard load, (input, target) pairs |
| Pipeline CLI       | `data/pipeline.py`                       | Hard-cap, resume support, tqdm bars     |
| Tests              | `data/tests/test_pipeline.py`            | 23 tests, all pass                      |
| Dataset config     | `configs/datasets/fineweb_edu.yaml`      | Token budgets per model size            |
| Push script        | `scripts/push_shards_to_kaggle.sh`       | kaggle CLI wrapper, tar mode            |

### Shard plan

**One canonical 10B-token shard set.** Smaller training runs consume a prefix of it.

| Version                       | Tokens      | Shards | For model   | Source                          |
| ----------------------------- | ----------- | ------ | ----------- | ------------------------------- |
| `v1-bpe32k-fineweb10B-overshoot` | 10,750,000,000 | 215    | 350M params | full uploaded set (7.5% over)  |
| 25M training                  | first 1B    | 20     | 25M params  | first ~20 shards                |
| 125M training                 | first 5B    | 100    | 125M params | first ~100 shards               |

**Why one set:** shards are flat uint16 token streams in deterministic stream order. The loader slices by `seq_len`/`total_tokens` at training time. Running 3 separate shard jobs = 3× disk, 3× upload, 3× version churn for no benefit. Subset = prefix of the same file set.

### Generated shard set — `llm-forge-tokens-v1-overshoot`

- **Tokens:** 10,750,000,000 (215 shards × 50M; 7.5% over the 10B target)
- **Disk:** ~21.5 GB
- **Format:** uint16 `.npy`, 50M tokens/shard, `metadata.json` with full shard index
- **Pushed to:** Kaggle Dataset `llm-forge-tokens-v1-overshoot`
- **Mount path on Kaggle:** `/kaggle/input/llm-forge-tokens-v1-overshoot/`
- **Why overshoot:** initial pipeline flushed a full shard after `total_tokens` already passed the budget. Fixed by hard-cap in `pipeline.py` — last doc is now truncated to fit.
- **Acceptable for training:** yes. 25M/125M subsets unaffected. 350M will use first 200 shards (10B) — last 15 shards are dead weight, can be ignored at training time.

### Future shard sets

- `v2-bpe32k-fineweb10BT` (planned) — clean 10B / 200 shards using the hard-cap fix
- Triggered only if v1-overshoot causes issues (e.g. reproducibility complaints). Otherwise keep v1.

### Decisions made

- **Flat uint16 .npy** (not padded sequences) — seq_len controlled at training time, more flexible
- **50M tokens/shard = ~100MB/file** — Kaggle-friendly chunk size
- **FileExistsError guard** in ShardWriter (default) — prevents accidental shard overwrite
- **`--skip-existing` resume** — opt-in flag, picks up from `shard_{N:05d}.npy` after partial run
- **EOT prepend per doc** — document boundary marked by `<|endoftext|>` (ID=0)
- **Token budget** measured in output tokens (not chars) — pipeline counts post-BPE tokens
- **Hard-cap on budget** — last doc truncated to never overshoot, final shard may be partial
- **tqdm progress bars** — every multi-hour loop (`FineWebEduStream`, `DocumentTokenizer`, `ShardWriter`, `ShardedTokenDataset`) shows live throughput + ETA
- **kaggle push via CLI** — `scripts/push_shards_to_kaggle.sh` (hard-links shards, tar mode upload)

---

## Phase 3 — Model Architecture ✅ COMPLETE

**Status:** All code written, tested. Llama-style LM forward pass works for all three sizes (25M/125M/350M) with all three attention variants (MHA/MQA/GQA).

### What's done

| Artifact               | Location                                   | Notes                              |
| ---------------------- | ------------------------------------------ | ---------------------------------- |
| Config loader          | `model/config.py`                          | ModelConfig dataclass + YAML load  |
| RoPE                   | `model/embeddings/rope.py`                 | Q,K only; V not rotated            |
| RMSNorm                | `model/normalization/rmsnorm.py`           | No bias, learnable scale           |
| SwiGLU MLP             | `model/mlp/swiglu.py`                      | d_ff = round(8/3·D/256)·256         |
| CausalMHA              | `model/attention/variants/mha.py`          | Baseline, n_kv=n_heads             |
| CausalMQA              | `model/attention/variants/mqa.py`          | n_kv=1, shared KV                  |
| CausalGQA              | `model/attention/variants/gqa.py`          | Configurable n_kv, repeat_interleave |
| TransformerBlock       | `model/blocks/transformer_block.py`        | Pre-norm, residuals, variant select |
| LM                     | `model/lm.py`                              | Tied emb, scalar loss, forward     |
| Summary CLI            | `model/summary.py`                         | Param count breakdown               |
| Tests                  | `model/tests/test_*.py`                    | Shape + smoke + gradient sanity    |
| Configs                | `configs/models/model_{25m,125m,350m}.yaml` | Three size presets                 |

### Model sizes

| Model    | n_layers | d_model | n_heads | n_kv  | d_ff  | ≈ params |
| -------- | -------- | ------- | ------- | ----- | ----- | -------- |
| model_25m  | 4        | 512     | 8       | 4     | 1280  | ~27.8M   |
| model_125m | 12       | 768     | 12      | 4     | 3072  | ~129M    |
| model_350m | 24       | 1024    | 16      | 8     | 2816  | ~316M    |

Note: d_ff values are computed from `compute_d_ff(d_model)` per Llama 8/3·D formula rounded to multiple of 256. `model_125m.yaml` had d_ff=3072 (spec error) and was corrected to 2048.

### Decisions made

- **JAX/Flax** for native TPU support
- **Llama-style arch**: pre-norm RMSNorm, SwiGLU, RoPE on Q/K only, tied emb, no biases
- **Standalone variants** (not one unified class): the diff is the lesson
- **Tied lm_head**: saves V·D params
- **d_head=64**: held constant across sizes (Llama convention)
- **GQA default** for all production configs; MHA/MQA exercisable via config

### Open questions

- None for Phase 3. Phase 4 will wrap in Optax + train loop.

---

## Phase 4 — Pretraining 🔓 Data ready, blocked on Phase 3

## Phase 5 — Post-Training 🔒 BLOCKED on Phase 4

## Phase 6 — Inference 🔒 BLOCKED on Phase 4

---

## Decisions Log (cross-cutting)

| Decision                                | Rationale                                                |
| --------------------------------------- | -------------------------------------------------------- |
| FineWeb-Edu as pretraining data         | High-quality educational text, good for small models     |
| 25M → 125M → 350M model sizes           | Iterable scale on Kaggle TPU v5e-8 (128GB HBM)           |
| uint16 .npy shards                      | 50% smaller than int32, all token IDs fit (0–32767)      |
| kv_cache split: model/ vs inference/    | Data structure vs serving management — clean boundary    |
| Special tokens locked at 0–4            | Changing them invalidates all .npy shards                |
| Dataset versioning: never overwrite     | Reproducibility — v1, v2, etc. by tokenizer version      |
| Flat token stream, no padding           | seq_len decided at training time, not dataset creation   |
| Kaggle Dataset for shards, git for code | Shards too large for git; tokenizer small enough (2MB)   |
| Rust BPE trainer alongside Python       | Python = learning reference; Rust = actual training tool |

---

## Kaggle Artifacts

| Kaggle Dataset                       | Contents                                          | Status         |
| ------------------------------------ | ------------------------------------------------- | -------------- |
| `llm-forge-tokenizer-v1`             | `tokenizer.json`, `vocab.json`, `merges.txt`      | ⏳ Not uploaded |
| `llm-forge-tokens-v1-overshoot`      | 215 × `shard_*.npy` + `metadata.json` (10.75B)    | ✅ Pushed       |
