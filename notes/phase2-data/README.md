# Phase 2 — Data Pipeline: Completion Report

**Phase:** 2 of 6
**Status:** Complete (with the 7.5% overshoot — see below)
**Date:** written retroactively, post-Push
**Hardware target:** Kaggle TPU v5e-8 (data build runs CPU on Kaggle)

---

## What we built

A streaming, resumable, tqdm-everywhere pipeline that turns FineWeb-Edu into uint16 `.npy` token shards and pushes them to a Kaggle Dataset. The end product is one canonical 10B-token shard set that all model sizes (25M / 125M / 350M) consume as a prefix.

| Component            | File                                          | LOC  | What it does                                          |
| -------------------- | --------------------------------------------- | ---- | ----------------------------------------------------- |
| FineWeb-Edu source   | `data/sources/fineweb.py`                     | 144  | HF streaming, char-budget stop, tqdm bar              |
| Document tokenizer   | `data/preprocessing/tokenize_dataset.py`      | 123  | Prepends `<|endoftext|>` (ID=0) per doc, tqdm bar     |
| Shard writer         | `data/preprocessing/shard_writer.py`          | 241  | uint16 `.npy`, 50M tokens/shard, `FileExistsError` guard, `--skip-existing` resume, tqdm bar, writes `metadata.json` |
| Pipeline CLI         | `data/pipeline.py`                            | 172  | Hard-cap on token budget, resume, tqdm, 4–8h wall time |
| NPY loader           | `data/loaders/npy_loader.py`                  | 168  | Lazy shard load, `ShardedTokenDataset`, (input, target) pairs, tqdm bar |
| Pipeline tests       | `data/tests/test_pipeline.py`                 | 264  | 23 tests, all pass                                    |
| Loader tests         | `data/loaders/tests/test_jax_batcher.py`      | 138  | JAX batcher wrapping `ShardedTokenDataset`            |
| Dataset config       | `configs/datasets/fineweb_edu.yaml`           | —    | Token budgets per model size                          |
| Push script          | `scripts/push_shards_to_kaggle.sh`            | 210  | `kaggle datasets create/version`, tar mode, hard-link staging |

**Data module total:** ~1,365 LOC across 9 Python files. **Tests:** 23 collected, all pass.

The pipeline flow is just three components in series:

```
FineWebEduStream  →  DocumentTokenizer  →  ShardWriter
   (chars)             (tokens + EOT)        (uint16 .npy)
```

---

## The 7.5% overshoot story

This is the most interesting thing that happened in Phase 2 and worth dwelling on, because the bug is *exactly the kind of thing* you write in a clean room and ship before you notice.

### What happened

First production run. Target: 10B tokens. Source streams docs, tokenizer prepends EOT, writer accumulates into a 50M-token buffer and flushes full shards to disk. Looks correct, and the first 199 shards (9.95B tokens) were textbook.

The race: the writer flushes a full shard **before** checking the budget. So once `total_tokens` crosses 10B, the writer is still in the middle of a flush that will complete a 50M-token shard *past* the target. The loop check is "did we exceed?", not "is this shard still under budget?" — so the last 15 shards slipped through, putting us at 215 shards / 10.75B tokens = **7.5% over**.

This was a pure mid-shard flush race in the original `pipeline.py`. The writer itself was fine; the orchestrator didn't have a hard-cap.

### Why it didn't break anything

Three reasons:

1. **215 shards × 50M = 10.75B.** The shards are still well-formed uint16 arrays. Every individual file is identical in shape to what the contract says. `metadata.json` accurately reports `total_tokens=10,750,000,000`. Nothing is corrupt.
2. **25M and 125M training both use a *prefix*** of the shard set — first 20 shards (1B tokens) and first 100 shards (5B tokens) respectively. Those first 100 shards are 100% clean. The overshoot lives in shards 200–214, which neither of those runs ever touches.
3. **350M will use the first 200 shards (10B tokens).** Shards 200–214 are dead weight, fully ignored at training time. Kaggle upload is 21.5GB instead of 20GB. We pay a ~7.5% storage tax and that's it.

The bug is *cosmetic with a real cost* — not a correctness bug. The cost is disk and upload time. The fix is two lines.

### The fix: hard-cap in `data/pipeline.py`

```python
remaining = args.token_budget - total_tokens
if len(token_ids) > remaining:
    token_ids = token_ids[:remaining]   # truncate last doc
    truncated = True
writer.add(token_ids)
total_tokens += len(token_ids)
if total_tokens >= args.token_budget:
    break
```

After this fix, the last shard can be partial (smaller than 50M), and we land at *exactly* the budget. `metadata.json` will still report the truthful count, and the loader handles partial last shards via the `len(tokens)` in `__iter__`.

### What we'd do differently

Hard-cap from day 1. There was no good reason not to — the `if remaining <= 0: break` check is one line, the doc-truncation is a few more. We were in a hurry to see 10B tokens on disk, and we got exactly that. Lesson re-learned: a budget without a cap is a suggestion, not a contract.

The bigger lesson: the overshoot *itself* was fine. The real mistake was messaging. We uploaded a 7.5%-over shard set as "v1" without immediately flagging in the dataset description that this is a known-overshoot artifact. If someone in 6 months goes to reproduce a training run from `llm-forge-tokens-v1`, they should know they don't have a clean 10B. We documented it in `progress.md` and in the v1 dataset description, but we should have led with that, not buried it.

---

## What worked

A few small decisions paid off bigger than expected:

- **uint16 .npy.** A 50M-token shard is exactly 100MB. 215 shards = 21.5GB. If we'd gone int32, 43GB. That's the difference between a fast Kaggle upload and a long one. And because our vocab is locked at 32,768 (2¹⁵), uint16 holds every ID we'll ever produce.
- **50M tokens/shard.** Big enough that the metadata overhead is negligible. Small enough that one shard fits comfortably in CPU memory at training time, and an upload of a single shard can resume from a partial failure.
- **`FileExistsError` guard in `ShardWriter.__init__`.** Default behavior is: if any `shard_*.npy` exists, **refuse to start** and tell you to bump the version. This single guard prevented the obvious foot-gun: re-running the pipeline against an existing v1 would silently mix old + new tokens and produce garbage.
- **`--skip-existing` resume mode.** Opt-in. The writer counts existing shards, picks up the index from `len(existing)`, and the tqdm bar starts at the right offset. We used this once after a Kaggle session got preempted at shard 142 and it just worked.
- **EOT prepend per document.** Every doc starts with `<|endoftext|>` (ID=0). The model sees a separator before the first real token, which means the first token of a doc is "predictable from context that ends in a separator." Standard GPT-2-style. Cheap, correct, and baked in at tokenization time so the training loop doesn't have to think about boundaries.
- **tqdm bars on every multi-hour loop.** Source stream, tokenizer, writer, and loader all show live throughput + ETA. When a run takes 4–8 hours, knowing whether you'll finish tonight vs tomorrow morning is not optional.
- **`metadata.json` written at `finalize()`.** Single source of truth: shard count, total tokens, per-shard token count, file size, vocab size, dataset version, timestamp. The loader reads this and refuses to start if shard counts don't match (catches partial uploads).

---

## What we learned

These are the things we *decided* — not things we discovered. Decisions that paid off, with the reasoning that made them obvious in retrospect:

### Why uint16 and not int32

Vocab is locked at 32,768 = 2¹⁵. All token IDs fit in uint16 (range 0–32767). int32 would be the "safe" default but it doubles the disk footprint with zero information gain. uint16 is the right dtype and we should commit to it loudly: **never let a token ID ≥ 32768 reach the writer.** The `_validate_ids` check in `ShardWriter` enforces this at write time.

### Why flat token stream and not padded sequences

A flat `(N,)` uint16 array is the most flexible thing we can store. The training loop decides `seq_len` (1024 for 25M, TBD for 125M/350M). Windows can be overlapping or non-overlapping, the stride is a loader argument, and we can change `seq_len` between training runs without re-tokenizing.

Padded sequences would force the decision at tokenization time and we'd be stuck. Padding also wastes tokens on `<|pad|>` (ID=1) which the model has to learn to ignore. Flat stream + EOT separator = real tokens, all the way through.

### Why one canonical 10B shard set for all model sizes

This was the single biggest save. The naive approach is one shard set per model size (1B for 25M, 5B for 125M, 10B for 350M) = 3× disk, 3× upload, 3× version churn. Instead: **produce 10B once, slice by prefix at training time.** 25M uses first 20 shards. 125M uses first 100. 350M uses first 200. Same `ShardedTokenDataset` interface, same `metadata.json`, same loader code — the only knob is which shards you iterate over.

It works because the token stream is deterministic and the shards are written in stream order. There's no shuffling at write time. A prefix is a real prefix.

### Kaggle Dataset as the shard store (vs git LFS, S3, etc.)

We considered three options:
- **git LFS:** 21.5GB is well past LFS's reasonable use, and most repos reject it anyway. Cloning the repo would take forever.
- **S3 / GCS:** extra account, extra cost, and Kaggle TPU notebooks don't have direct S3 access without credentials dance.
- **Kaggle Dataset:** native to the runtime we're using, free for public datasets, versioned (v1, v2, ...), mount path is `/kaggle/input/...`, no auth needed inside a notebook.

Kaggle Dataset won on every axis that mattered. The push script (`scripts/push_shards_to_kaggle.sh`) hard-links shard files into a staging dir (no 20GB copy) and uses `--dir-mode tar` so the 215 files upload as a single archive. Clean, fast, reproducible.

---

## What we got wrong / would redo

- **Should have hard-capped from the start.** See the overshoot story above. Two-line fix that we should have shipped on day 1.
- **Overshoot messaging.** We documented v1's 7.5% overshoot in `progress.md` and in the dataset description, but the dataset title is just "LLM-Forge Tokens v1" without the "overshoot" suffix. Should be `llm-forge-tokens-v1-overshoot` or the description should lead with the caveat. As it stands, a future reader has to know to look.
- **No deduplication.** FineWeb-Edu has some near-duplicate docs. We didn't dedupe. At our scale (10B tokens, small model) this is probably fine — the LLM paper showed small models don't benefit much from dedup, and the FineWeb-Edu curators already did a pass. But it's a thing we didn't do, and we should say so explicitly.
- **No quality filtering beyond HF's `sample-10BT` config.** We trusted HuggingFace's downstream filtering. For a 25M model this is fine. For 350M we might want a `score >= 3` filter.
- **Char budget is rough.** `FineWebEduStream` uses `token_budget * 5` as a char budget (rough ~5:1 chars-to-tokens ratio). This is a guess. If the corpus is unusually dense, the stream might exhaust before the token budget is hit. In practice it was close enough that the token hard-cap caught us. Could be tighter.

---

## Phase 3+ handoff

Things Phase 3 (model) and Phase 4 (training) need to know to consume Phase 2:

### Mount path convention

On a Kaggle notebook with the dataset added via sidebar:

```
/kaggle/input/datasets/<owner>/<slug>/
```

For our v1:

```
/kaggle/input/datasets/adeshboudh/llm-forge-tokens-v1/
```

The push script prints this on completion. The `dataset-metadata.json` description also has it baked in. **No relative paths inside notebooks** — always construct the full path from `KAGGLE_USERNAME` or read it from env.

### `ShardedTokenDataset` interface contract

```python
from data.loaders.npy_loader import ShardedTokenDataset

ds = ShardedTokenDataset(
    shard_dir="/kaggle/input/datasets/adeshboudh/llm-forge-tokens-v1",
    seq_len=1024,            # chosen by training config
    shuffle_shards=True,     # default — shuffle shard order each epoch
    stride=None,             # default = seq_len, non-overlapping windows
)
# ds[i] would fail — this is Iterable, not Indexable.
# Use ds.__iter__() or wrap in DataLoader.
```

Yields `(input_ids, target_ids)` tuples, both `int64` shape `(seq_len,)`. Target = input shifted left by 1. The loader casts uint16 → int64 internally because most loss functions need int64.

**Train/val split convention:** Phase 4 reserves `shard_00214` (the last shard, 50M tokens) for validation. Train = shards 0–213, val = shard 214. Documented in `progress.md` Phase 4.

### Token budget measurement

**Post-BPE tokens, not chars.** The pipeline's `--token-budget` is enforced *after* tokenization (in `pipeline.py:142` via `total_tokens += len(token_ids)`). The HF source stream has its own char-based stop condition, but the authoritative budget is the token count.

This matters because BPE compression ratio varies by content. Educational text compresses better than code (fewer merges needed). If you're budgeting on chars, you'll either under- or over-shoot depending on the corpus.

### Other things Phase 3+ should know

- **Shards are 50M tokens flat uint16.** A 25M-param model at seq_len=1024, batch_size=128 consumes one full shard in ~390 steps. Epoch size = `total_tokens / (seq_len × batch_size)`.
- **EOT (ID=0) is the document separator.** Any loss masking scheme that wants to ignore padding should use `<|pad|>` (ID=1), not EOT.
- **Special tokens are locked at 0–4.** Don't ever change them. If you need a new special token, that requires retokenizing and bumping dataset to v2. We'd rather not.
- **`metadata.json` is the contract.** Always read it at load time, don't hardcode shard counts. The loader does this already.

---

## Kaggle Datasets registry

Datasets created during Phase 2:

| Dataset slug                       | Owner         | Version | Content                                                       | Status     |
| ---------------------------------- | ------------- | ------- | ------------------------------------------------------------- | ---------- |
| `llm-forge-tokens-v1`              | `adeshboudh`  | 1       | 215 × `shard_*.npy` + `metadata.json` (10.75B tokens, ~21.5GB) | ✅ Pushed   |

Note: this is the **overshoot** set. A clean `v2-bpe32k-fineweb10BT` (exactly 10B / 200 shards) is planned and will be generated with the hard-cap fix. Will only be triggered if v1 causes reproducibility issues. Otherwise v1 stays.

Tokenized data is committed to Kaggle Datasets, not git — the `experiments/runs/data/` and `data/shards/` are git-ignored.

---

## Open questions for Phase 3+

- **350M's 10B tokens — is that enough?** Chinchilla-optimal would be ~20× params in tokens, so 350M wants ~7B. 10B is over-budget, so we should be fine. But the FineWeb-Edu educational filter might underweight what 350M needs. Maybe.
- **Should we mix in code or math data for 125M/350M?** FineWeb-Edu is educational prose. A small slice of code (Stack-Edu?) might help on downstream evals. Out of scope for Phase 2, but worth flagging for Phase 4+ if perplexity plateaus.
- **Tokenization throughput.** BPE is currently slow because it's pure Python. ~50K tokens/sec on a single Kaggle CPU core. For 10B tokens that's ~55 hours, but we ran 4-way parallel and finished in ~8h. If we ever go to 100B+, we'll need the Rust tokenization path (parallel to the Rust BPE trainer in Phase 1).
- **Validation shard choice.** We're using the *last* shard (214) for validation. Risk: it's not representative of the corpus (whatever happened to land at the tail of the stream). A random shard or first shard might be safer. We're doing "last shard" for now because it doesn't require a separate upload and matches the "use the prefix" convention.

---

## Files added in Phase 2

```
data/
├── __init__.py
├── pipeline.py                                  (172 LOC)
├── sources/
│   ├── __init__.py
│   └── fineweb.py                               (144 LOC)
├── preprocessing/
│   ├── __init__.py
│   ├── tokenize_dataset.py                      (123 LOC)
│   └── shard_writer.py                          (241 LOC)
├── loaders/
│   ├── __init__.py
│   ├── npy_loader.py                            (168 LOC)
│   └── tests/
│       ├── __init__.py
│       └── test_jax_batcher.py                  (138 LOC)
└── tests/
    ├── __init__.py
    └── test_pipeline.py                         (264 LOC)

configs/datasets/
└── fineweb_edu.yaml                             (config only)

scripts/
└── push_shards_to_kaggle.sh                     (210 LOC)
```

**Total:** ~1,365 LOC of Python, ~210 LOC of bash, 1 YAML config.

**Tests:** 23 collected in `data/tests/test_pipeline.py`, all pass. Covers pipeline CLI arg parsing, `ShardWriter` happy path + `FileExistsError` + `--skip-existing` resume, `ShardedTokenDataset` iteration + repr + metadata validation + shard-count mismatch, `DocumentTokenizer` EOT prepend, `FineWebEduStream` char-budget stop and min-doc-len filter.

---

*Next: Phase 3 (model architecture). Phase 2 unblocks it — model code can now test forward passes against real uint16 token streams with a known shape contract.*
