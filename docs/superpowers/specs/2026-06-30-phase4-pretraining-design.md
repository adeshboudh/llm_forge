# Phase 4 — Pretraining (Design Spec)

**Date:** 2026-06-30
**Status:** Approved (all 6 design sections user-approved)
**Target:** Train `model_25m` end-to-end (1B tokens, FineWeb-Edu) on Kaggle TPU v5e-8.

---

## 1. Scope & Success Criterion

**Goal:** Build a complete pretraining stack — data loader, optimizer, scheduler, checkpointing, TPU mesh, train loop, CLI — and actually train `model_25m` to convergence on 1B tokens.

**Done means:**
- `model_25m` is being trained / has been trained on Kaggle TPU v5e-8 across 8 cores via pjit data-parallel.
- Framework is reusable for 125M/350M via config swap (no code changes).
- All training tests pass on CPU with toy shards (no real shards required locally).
- Phase 3 model code is untouched — reuse `model.LM.forward` as the loss function.

**Out of scope:**
- Larger-runs (125M/350M actual training): config-only later runs, framework is verified by 25M only in this phase.
- Real multi-host training (Kaggle uses a single VM): `training/tpu.py` builds a 1D mesh over 8 local devices only.
- Full inference / generation: smoke generation only; full samplers land in Phase 6.
- Post-training (SFT/DPO/etc.): Phase 5.

---

## 2. Architecture & Data Flow

### 2.1 Flow

```
configs/training/model_25m.yaml
        │
        ▼
training.train (CLI argparse)
        │
        ├─▶ training.config.load_training_config()   → TrainConfig
        ├─▶ data.loaders.jax_batcher.JAXBatcher       → host np (B, seq_len) int32
        ├─▶ training.tpu.setup_tpu() + make_mesh()     → Mesh, PartitionSpecs
        ├─▶ training.state.create_train_state()       → TrainState (fp32 master, AdamW state)
        │       └─ optax.chain(clip, adamw(cosine_warmup, wd_mask))
        ├─▶ training.train_step.train_step (pjit)     → (new_state, loss, grad_norm)
        ├─▶ training.state.save() each N steps        → /kaggle/working/ckpt/<step>/ (orbax async)
        ├─▶ eval_step every eval_every                  → val loss from val_shard
        └─▶ JSONL log every step                        → /kaggle/working/train_log.jsonl
```

### 2.2 Key Boundary

PyTorch-trained shards on host → `JAXBatcher` produces NumPy arrays on host → pjit shards them across 8 TPU cores by batch axis. The model (`model/lm.py`) is untouched — Phase 3 forward pass is reused as-is. Only training scaffolding is new.

### 2.3 TrainState Contents

A `flax.training.TrainState` containing:
- `params`: fp32 master copy (PyTree matching `model/lm.py` setup)
- `opt_state`: AdamW (m, v, count) in fp32
- `step`: int32
- (No RNG field — dropout isn't used; TrainState default has no `key`. Posttraining in Phase 5 may extend the dataclass.)

### 2.4 Routing

Single-process, single-host. Kaggle TPU v5e-8 presents 8 local devices to one VM. Mesh is 1D with axis name `"batch"` over those 8 devices. No multi-host concerns.

---

## 3. Precision (Mixed bf16 compute + fp32 master)

### 3.1 Cast Boundary

The bf16 cast lives inside a **loss closure** in `train_step.py`. The master params stay fp32; only an ephemeral bf16 copy is made for the forward/backward pass:

```python
def make_loss_fn(model: LM):
    @jax.value_and_grad
    def loss_fn(params, input_ids, target_ids):
        params_bf16 = jax.tree_util.tree_map(
            lambda p: p.astype(jnp.bfloat16) if p.ndim >= 2 else p,
            params,
        )
        return model.apply(params_bf16, input_ids, target_ids)
    return loss_fn
```

### 3.2 Dtype Rules

| Tensor | dtype | Where |
|--------|-------|-------|
| `state.params` (master) | fp32 | `TrainState.params`, optimizer updates |
| `state.opt_state` (AdamW m, v, count) | fp32 | optax internals |
| `params_bf16` (compute copy) | bf16 | ephemeral inside `loss_fn`, dies after backward |
| `input_ids`, `target_ids` | int32 | host loader → device |
| `grads` | fp32 | vjp aut-promotes through `astype` |
| `loss` | fp32 | `jnp.mean` reduction → scalar |

### 3.3 Cast Rule for params_bf16

`p.ndim >= 2` → cast to bf16. 1D tensors (norm scales; no biases in this Llama architecture) stay fp32. Rationale: 1D norm scales are tiny (1024 floats = 4KB), and bf16 quantization noise on a parameter that multiplies the entire residual stream can destabilize training. Llama/Mistral follow the same convention.

### 3.4 Why Not Pure BF16

v5e's **MXU** (matmul unit) requires bf16×bf16 → fp32-accumulate. But HBM and the scalar/vector units handle fp32 natively. Mixed precision gives bf16 matmul throughput (2-3× over fp32) without sacrificing AdamW second-moment precision. Pure bf16 end-to-end is known to destabilize AdamW runs past ~500M tokens due to precision loss in `v ← 0.95v + 0.05g²`. The 50MB HBM cost of keeping a fp32 master + bf16 compute copy is negligible for 25M.

### 3.5 HBM Budget at 25M (per core, data-parallel = 1/8 of params)

| Item | Size |
|------|------|
| fp32 master params (shard) | ~12.5MB |
| bf16 compute copy (shard) | ~6.25MB |
| AdamW m + v (shard) | ~25MB |
| grads fp32 (shard) | ~12.5MB |
| activations @ micro_batch=16, seq=1024, 4 layers | ~200MB |
| **Total** | **~260MB / 16GB HBM per core** |

98% free.

### 3.6 Loss Reduction

`jnp.mean(-log_p)` across all `(B, T)` positions. pjit reduces across cores automatically → scalar fp32 loss. No manual cross-core reduction.

---

## 4. TPU Strategy (Data-Parallel pjit, 1D Mesh)

### 4.1 Components

- `training/tpu.py`:
  - `setup_devices()` — `jax.devices()` returns device list. Same call works on CPU (1 CPU) and Kaggle TPU v5e-8 (8 TPU devices). No branching.
  - `make_mesh()` — `jax.sharding.Mesh(devices, ("batch",))` (1D)
  - `make_input_sharding()` / `make_param_sharding()` / `make_loss_sharding()` — three `NamedSharding` helpers for pjit inputs/outputs.

### 4.2 PartitionSpecs

| Tensor | PartitionSpec | Behavior |
|--------|--------------|----------|
| `input_ids`, `target_ids` | `("batch", None)` | Split across batch dim; replicated across seq |
| `params` (all leaves) | `(None)` | Replicated; data-parallel keeps full master copy per core |
| `loss` | `()` | Scalar; replicated |
| `grads` | `(None)` | fp32 grads replicated, optax update on master |

### 4.3 pjit vs jit

`train_step` is `pjit`-wrapped (annotates input/output sharding). `eval_step` uses `jax.jit` (no batch sharding needed — val loss is mean over val batches on host). Both are decorated once at module level.

### 4.4 Per-core batch derivation

No `micro_batch` field in TrainConfig. pjit derives per-core batch from `batch_size // len(devices)`. On 8 TPU cores with `batch_size=128`, each core processes 16 samples. On 1 CPU with `batch_size=128`, the single core processes 128 (works but wasteful — for CPU smoke use `smoke_test.yaml` with `batch_size=4`).

---

## 5. Data Loader

### 5.1 Reuse Strategy

Reuse `ShardedTokenDataset` (Phase 2, torch-style iterable) wrapped in a thin JAX-friendly batcher. No tf.data, no new deps.

### 5.2 `data/loaders/jax_batcher.py` (~60 lines)

```python
class JAXBatcher:
    def __init__(self, shard_dir, seq_len, batch_size, val_shard, seed=0): ...
    def train_iter(self) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
    def val_iter(self) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
    def skip_tokens(self, n: int) -> None: ...
```

**Rules:**
- Builds two `ShardedTokenDataset`s internally:
  - train: all shards **except** `val_shard`
  - val: only `val_shard`
- `__iter__` yields `(input_ids, target_ids)` as `np.int32` arrays of shape `(B, seq_len)`.
- `val_iter` cycles infinitely — caller decides how many batches to draw (via `eval_batches` in config).
- Stays on host (returns NumPy arrays). pjit input specs handle device sharding.
- `skip_tokens(n)` advances shard cursor by `n` tokens — supports resume.
- **Seeded shuffle**: `random.seed(seed)` is called inside `__init__` so `shuffle_shards=True` produces deterministic shard order per seed. Resume correctness depends on this — `skip_tokens(n)` slices shard N's token stream at offset `n`, and the same seed reproduces the same shard order.

### 5.3 Val Split

- Reserved shard: `shard_00214.npy` (50M tokens, 0.5% of 10.75B). Never fed to training.
- Eval cadence: every `eval_every` (=500) train steps, draw `eval_batches` (=50) val batches → mean val loss → perplexity = `exp(val_loss)`.
- Cheap, deterministic, gives honest signal that the model is learning, not memorizing.

---

## 6. Optimizer + Scheduler

### 6.1 Optimizer Chain

```python
tx = optax.chain(
    optax.clip_by_global_norm(config.grad_clip),       # 1.0
    optax.adamw(
        learning_rate=lr_schedule,
        b1=config.b1, b2=config.b2, eps=config.eps,
        weight_decay=config.weight_decay,              # 0.1
        mask=weight_decay_mask,
    ),
)
```

### 6.2 LR Schedule

```python
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=config.lr_peak,        # 3e-4
    warmup_steps=config.warmup_steps, # 200
    decay_steps=config.total_steps,    # 9766
    end_value=config.lr_min,          # 3e-5 (10% of peak)
)
```

Shape: linear 0 → 3e-4 over 200 steps, then cosine 3e-4 → 3e-5 over remaining 9566 steps.

### 6.3 Weight Decay Mask

Skip decay on: 1D params (norm scales), `tok_emb` (tied embeddings — Llama convention; decaying embeddings hurts perplexity). Apply decay 0.1 to: 2D weights (Q/K/V/O proj, w_gate, w_up, w_down).

### 6.4 Hyperparameters for 25M @ 1B tokens

| Param | Value | Source |
|-------|-------|--------|
| `lr_peak` | 3e-4 | standard for ~25M params |
| `lr_min` | 3e-5 | 10% of peak (Llama) |
| `warmup_steps` | 200 | ~2% of total steps |
| `b1` `b2` | 0.9, 0.95 | Llama |
| `eps` | 1e-8 | standard |
| `weight_decay` | 0.1 | Llama |
| `grad_clip` | 1.0 | Llama |
| `batch_size` | 128 (16 × 8 cores) | 1B/9766 ≈ 1024 tok/step/core |
| `seq_len` | 1024 | matches `max_seq_len` in ModelConfig |
| `val_shard` | 214 | last shard of v1 dataset (50M tokens, 0.5%) |
| `total_steps` | 9766 | 1B / (128 × 1024) |
| `save_every` | 500 | orbax async checkpoint |
| `eval_every` | 500 | val loss + perplexity |
| `eval_batches` | 50 | 50 × 128 × 1024 = 6.5M tokens per eval |

---

## 7. Checkpointing (Orbax + /kaggle/working)

- `training/state.py` exposes `save(state, path)` and `restore(path, placeholder_state)`.
- Backend: `orbax.checkpoint.PyTreeCheckpointer`. Async on TPU — training doesn't block on writes.
- Storage path: `/kaggle/working/ckpt/<step>/`.
- Kaggle's `/kaggle/working` persists across session reconnects (verifies via resume); does NOT persist across different notebook sessions, so download final checkpoints if you want them long-term.
- Restore: load against a `placeholder_state` from the same config. Orbax raises on shape mismatch — surfaces cleanly, doesn't corrupt state.

---

## 8. Logging (stdout tqdm + JSONL)

- Per-step `tqdm` bar: step, loss, lr, tok/s, ETA.
- Per-step JSONL line at `/kaggle/working/train_log.jsonl`:
  ```json
  {"step": 1234, "loss": 4.2, "val_loss": null, "lr": 2.8e-4, "grad_norm": 0.55, "tok/s": 52345, "ts": "2026-06-30T..."}
  ```
  `val_loss` populated every `eval_every` steps; `null` otherwise.
- Diffable, greppable, no extra deps. Plot loss curve with 5 lines of matplotlib in the Kaggle notebook.

---

## 9. Resume Protocol

```
python -m training.train --config configs/training/model_25m.yaml --resume /kaggle/working/ckpt/005000
```

1. `create_train_state()` initializes a fresh state (correct shapes from config).
2. `state = restore(ckpt_path, state)` overwrites params + opt_state + step.
3. Dataloader advanced via `skip_tokens(step × batch_size × seq_len)` — slices the flat token stream. Order-preserving because shards are deterministic and `shuffle_shards` uses a seeded RNG.
4. Train loop resumes from `state.step`.

Why token-offset instead of replaying shards: deterministic and cheap. No re-iteration to find our place.

---

## 10. Error Handling

### 10.1 TPU Preemption / OOM

- Wrap train loop in `try/except RuntimeError`.
- On exception: save emergency checkpoint to `/kaggle/working/ckpt/emergency_<step>/` (async orbax), log `{"event": "preemption", "step": N}`, exit cleanly.
- User restarts notebook, uses `--resume` against the emergency checkpoint.

### 10.2 Loss NaN

- Every step: `if not jnp.isfinite(loss): save emergency checkpoint, log, raise`.
- NaN propagates and silently ruins a run otherwise; better to fail loud at step N than discover step N+5000.

### 10.3 Dataset Errors

- `ShardedTokenDataset` already raises `FileNotFoundError` / `ValueError` on missing metadata/shard mismatch. Phase 4 inherits these. No silent retries.

### 10.4 Orbax Restore Mismatch

- Restore loads against a placeholder_state from the same config. If shapes don't match (wrong model checkpoint), orbax raises — surface it cleanly, don't let it corrupt state.

---

## 11. Module Layout

### 11.1 Files

```
training/
  config.py        ~70 lines   TrainConfig dataclass + load_training_config() + helper dataclasses
  state.py         ~90 lines   create_train_state(), save(), restore() via orbax
  train_step.py   ~110 lines   pjit train_step + jit eval_step + bf16 loss closure + wd_mask
  tpu.py           ~50 lines   setup_devices(), make_mesh(), make_input_sharding(), make_param_sharding(), make_loss_sharding(), tpu_context()
  train.py        ~120 lines   train loop + argparse CLI + JSONL logger + emergency save
  summary.py       ~50 lines   echo TrainConfig + devices + params + ETA
  __init__.py
  tests/
    conftest.py    toy_shards/toy_model/toy_config fixtures
    test_config.py
    test_state.py
    test_train_step.py
    test_train_smoke.py    end-to-end 5-step run

data/loaders/
  jax_batcher.py   ~60 lines   JAXBatcher wrapping ShardedTokenDataset
  tests/
    test_jax_batcher.py

configs/training/
  smoke_test.yaml
  model_25m.yaml

notebooks/phase4-training/
  train_25m.ipynb   ~6 cells: install, train, plot loss, load ckpt, sample
```

### 11.2 Why This Split

Each file = one responsibility, all well under the 300-line limit (CLAUDE.md rule). Tests map 1:1 per module. Mirrors Phase 3's `model/{config,lm,blocks,...}` layout, so it's idiomatic to this repo. Each file holds in your head.

Alternative (8 micro-files with `optim.py`, `scheduler.py`, etc.) rejected: premature abstraction. Each "component" would be ~30 lines of optax glue — adds import overhead, harder to follow than reading one `train_step`. Violates "no premature splitting into micro-modules" from CLAUDE.md.

---

## 12. TrainConfig Schema

```yaml
# configs/training/model_25m.yaml
model_name: model_25m            # loads ModelConfig via existing load_model_config()

dataset:
  shard_dir: /kaggle/input/datasets/adeshboudh/llm-forge-tokens-v1/
  seq_len: 1024
  val_shard: 214                 # shard_00214.npy = val

train:
  batch_size: 128                # per-host; → 16/core × 8 cores (pjit derives per-core from devices)
  total_steps: 9766              # 1B / (128 × 1024)
  warmup_steps: 200
  weight_decay: 0.1
  grad_clip: 1.0
  grad_accum: 1                  # HBM permits full batch at 25M

optim:
  lr_peak: 3.0e-4
  lr_min: 3.0e-5
  b1: 0.9
  b2: 0.95
  eps: 1.0e-8

ckpt:
  save_every: 500
  output_dir: /kaggle/working/ckpt/

log:
  log_file: /kaggle/working/train_log.jsonl
  log_every: 1
  eval_every: 500
  eval_batches: 50
```

### 12.1 Smoke Test Config

```yaml
# configs/training/smoke_test.yaml
model_name: model_25m
dataset:
  shard_dir: ./data/shards_smoke/     # populated by conftest fixture
  seq_len: 128
  val_shard: 3                        # shard_00003.npy
train:
  batch_size: 4
  total_steps: 5
  warmup_steps: 2
  weight_decay: 0.1
  grad_clip: 1.0
  grad_accum: 1
optim:
  lr_peak: 3.0e-4
  lr_min: 3.0e-5
  b1: 0.9
  b2: 0.95
  eps: 1.0e-8
ckpt:
  save_every: 2
  output_dir: ./data/shards_smoke/ckpt/
log:
  log_file: ./data/shards_smoke/train_log.jsonl
  log_every: 1
  eval_every: 2
  eval_batches: 2
```

---

## 13. CLI & Makefile

### 13.1 CLI

```
usage: training.train [-h] --config CONFIG [--resume RESUME] [--smoke] [--max-steps N]

required:
  --config CONFIG     Path to training YAML (e.g. configs/training/model_25m.yaml)

optional:
  --resume RESUME     Path to orbax checkpoint dir to restore from
  --smoke             Run 5 steps on CPU toy shards, exit
  --max-steps N       Override total_steps (for short Kaggle test runs)
```

### 13.2 Behavior Matrix

| Flag combo | What runs |
|-----------|-----------|
| `--config configs/training/model_25m.yaml` | Full 9766-step run on whatever devices present |
| `--config configs/training/smoke_test.yaml --smoke` | 5 steps, toy shards, CPU |
| `--config configs/training/model_25m.yaml --max-steps 100` | First 100 steps only |
| `--resume /kaggle/working/ckpt/005000 --config ...` | Resume from step 5000 |

### 13.3 New Make Targets

```make
train-smoke:  python -m training.train --config configs/training/smoke_test.yaml --smoke
train-25m:    python -m training.train --config configs/training/model_25m.yaml
train-test:   uv run pytest training/tests/ -v
train-summary: python -m training.summary --config configs/training/model_25m.yaml
```

---

## 14. Testing Strategy

### 14.1 Toy Shards Fixture (no real shards needed locally)

`conftest.py` generates 4 shards × 10k tokens of uint16 random data `[0..32767]` with a fixed seed + matching `metadata.json` in a pytest `tmp_path/`. Reproducible across CPU/GPU/TPU, ~80KB, auto-cleaned by pytest. No dependence on real shards existing anywhere.

### 14.2 Test Matrix

| Test file | What it asserts |
|-----------|----------------|
| `training/tests/test_config.py` | TrainConfig loads from YAML; nested dataclasses hydrate; unknown config key raises; frozen raises on mutation; `val_shard` integer in shard range |
| `data/loaders/test_jax_batcher.py` | Shapes `(B, seq_len)` int32; train excludes val shard; val cycles infinitely; `skip_tokens(n)` advances correctly; empty shard dir raises |
| `training/tests/test_state.py` | `create_train_state()` returns params in fp32 (assert on a leaf `.dtype`); AdamW mask correctly skips norm scales + tok_emb (query shape of weight_decay_updates); orbax save+restore round-trips params exactly |
| `training/tests/test_train_step.py` | Loss is finite + scalar; grads are fp32 (assert on a leaf grad `.dtype`); grad_norm metric > 0; bf16 compute path doesn't change loss by > 5% vs fp32 forward at small batch; `eval_step` returns finite loss |
| `training/tests/test_train_smoke.py` | 5 train steps with `toy_shards`/`toy_model`/`toy_config` on CPU — loss decreases; ckpt files exist; JSONL log has 5 lines; `--resume` round-trip from step 3 continues; CLI `--help` works |

### 14.3 Deps to Add to pyproject.toml

- `optax>=0.2.0` — optimizer + scheduler + clip
- `orbax-checkpoint>=0.5.0` — checkpointing

No new test deps (pytest already present).

---

## 15. Notebook (Kaggle)

`notebooks/phase4-training/train_25m.ipynb` (~6 cells, read-only reference per CLAUDE.md):

1. `%pip install -e .` + mount Kaggle dataset `/kaggle/input/datasets/adeshboudh/llm-forge-tokens-v1/`
2. `!python -m training.train --config configs/training/model_25m.yaml --max-steps 50` (sanity)
3. `!python -m training.train --config configs/training/model_25m.yaml` (real run)
4. Plot loss curve from `train_log.jsonl` (matplotlib inline)
5. Load final checkpoint via `orbax.restore`
6. Generate 3 samples from a fixed prompt using `model.LM.apply(..., return_logits=True)` + argmax+temperature sampling. Smoke-level; full generation lands in Phase 6.

---

## 16. What Phase 4 Reuses (no changes)

- `model/lm.py` `LM.__call__` — scalar loss, reused verbatim
- `model/config.py` `load_model_config()` — TrainConfig embeds `model_name` and calls this
- `model/attention/variants/*.py`, `model/blocks/transformer_block.py`, `model/embeddings/rope.py`, `model/normalization/rmsnorm.py`, `model/mlp/swiglu.py` — forward path unchanged
- `data/loaders/npy_loader.py` `ShardedTokenDataset` — wrapped, not replaced
- `configs/models/model_25m.yaml` — pointed to by TrainConfig

---

## 17. What Phase 4 Adds (new only)

- `training/config.py` — `TrainConfig` + nested dataclasses + `load_training_config()`
- `data/loaders/jax_batcher.py` — `JAXBatcher` wrapping `ShardedTokenDataset`
- `training/tpu.py` — `setup_tpu()`, `make_mesh()`, `make_partition_specs()`, `tpu_context()`
- `training/state.py` — `create_train_state()`, `save()`, `restore()`
- `training/train_step.py` — `train_step` (pjit), `eval_step` (jit), `_loss_fn`, `_weight_decay_mask`
- `training/train.py` — train loop + CLI + JSONL logger + emergency save
- `training/summary.py` — config echo + device/param summary
- `training/tests/{conftest,test_config,test_state,test_train_step,test_train_smoke}.py`
- `data/loaders/test_jax_batcher.py`
- `configs/training/{smoke_test,model_25m}.yaml`
- `notebooks/phase4-training/train_25m.ipynb`
- 2 new deps in `pyproject.toml`: `optax`, `orbax-checkpoint`
- Make targets: `train-smoke`, `train-25m`, `train-test`, `train-summary`

---

## 18. Open Items (deferred)

- 125M/350M training runs: framework is reusable via config swap; actual runs scheduled later on Kaggle.
- FSDP (weight sharding): not needed at 25M; would be when 350M HBM pressure forces it. Single line change to PartitionSpec when that day comes.
- Tensorboard / W&B integration: JSONL is sufficient for single-run analysis; richer dashboards come in Phase 5 when experiment variants matter.
- Muon optimizer experiment: noted; not in scope for the conservative 25M learner run.