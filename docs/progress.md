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

## Phase 2 — Data Pipeline ✅ Code complete / ⏳ Shards not yet generated

**Status:** All pipeline code written and tested. Shards need to be generated on target machine.

### What's done

| Artifact           | Location                                 | Notes                                   |
| ------------------ | ---------------------------------------- | --------------------------------------- |
| FineWeb-Edu source | `data/sources/fineweb.py`                | Streaming, char-budget stop             |
| Document tokenizer | `data/preprocessing/tokenize_dataset.py` | Prepends EOT (ID=0) per doc             |
| Shard writer       | `data/preprocessing/shard_writer.py`     | uint16 .npy, 50M tokens/shard           |
| Data loader        | `data/loaders/npy_loader.py`             | Lazy shard load, (input, target) pairs  |
| Pipeline CLI       | `data/pipeline.py`                       | Wires all components, token-budget stop |
| Tests              | `data/tests/test_pipeline.py`            | 23 tests, all pass                      |
| Dataset config     | `configs/datasets/fineweb_edu.yaml`      | Token budgets per model size            |

### Shard plan

**One canonical 10B-token shard set.** Smaller training runs consume a prefix of it.

| Version                 | Token budget | For model   | Source                          |
| ----------------------- | ------------ | ----------- | ------------------------------- |
| `v1-bpe32k-fineweb10BT` | 10B tokens   | 350M params | full shard set                  |
| 25M training            | first 1B     | 25M params  | first ~20 shards of 10B set     |
| 125M training           | first 5B     | 125M params | first ~100 shards of 10B set    |

**Why one set:** shards are flat uint16 token streams in deterministic stream order. The loader slices by `seq_len`/`total_tokens` at training time. Running 3 separate shard jobs = 3× disk, 3× upload, 3× version churn for no benefit. Subset = prefix of the same file set.

**Status:** ⏳ Not generated (single 10B run, then subset at training time).

### To generate shards

```bash
# On target machine (Kaggle / Lightning AI — needs network + time)
python data/pipeline.py \
    --tokenizer tokenizer/saved/tokenizer.json \
    --output-dir data/shards/ \
    --token-budget 10_000_000_000 \
    --dataset-version v1-bpe32k-fineweb10BT
```

Then upload `data/shards/` to Kaggle Dataset `llm-forge-tokens-v1`.

### Decisions made

- **Flat uint16 .npy** (not padded sequences) — seq_len controlled at training time, more flexible
- **50M tokens/shard = ~100MB/file** — Kaggle-friendly chunk size
- **FileExistsError guard** in ShardWriter — prevents accidental shard overwrite
- **EOT prepend per doc** — document boundary marked by `<|endoftext|>` (ID=0)
- **Token budget** measured in output tokens (not chars) — pipeline counts post-BPE tokens

### Open questions / blockers

- **Shard generation not started yet** — needs ~4–8 CPU hours on network-connected machine
- **Kaggle upload** — manual step after shard generation

---

## Phase 3 — Model Architecture 🔜 NEXT

**Status:** Not started. Start with `model/attention/variants/mha.py`.

### Plan

1. `model/attention/variants/mha.py` — Multi-Head Attention baseline
2. `model/attention/variants/mqa.py` — Multi-Query Attention
3. `model/attention/variants/gqa.py` — Grouped-Query Attention (production standard)
4. `model/embeddings/rotary.py` — RoPE positional encoding
5. `model/mlp/swiglu.py` — SwiGLU activation
6. `model/normalization/rmsnorm.py` — RMSNorm
7. Wire into `model/blocks/transformer_block.py`

---

## Phase 4 — Pretraining 🔒 BLOCKED on Phase 3

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

## Kaggle Artifacts Plan

| Kaggle Dataset           | Contents                                     | Status           |
| ------------------------ | -------------------------------------------- | ---------------- |
| `llm-forge-tokenizer-v1` | `tokenizer.json`, `vocab.json`, `merges.txt` | ⏳ Not uploaded  |
| `llm-forge-tokens-v1`    | `shard_*.npy` + `metadata.json` (10B tokens) | ⏳ Not generated |
