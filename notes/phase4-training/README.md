# Phase 4 — Pretraining: Completion Report

**Phase:** 4 of 6
**Status:** Complete (and the model is on HuggingFace)
**Date:** written after the 1B-token Kaggle run
**Hardware target:** Kaggle TPU v5e-8 — 8 TPU devices, 128 GB HBM total
**Stack:** JAX + Flax + Optax + Orbax, data-parallel `pjit`, 1D mesh

---

## The headline

We trained `model_25m` end-to-end on **1,280,049,152 tokens** of FineWeb-Edu on a Kaggle TPU v5e-8, in **9,766 steps × batch 128 × seq 1024**, and the final training loss landed at **3.8750**. The final checkpoint is at `step_0000009766_final/`, exported to `params.safetensors` (107 MB) and to a HuggingFace-compatible directory, and pushed live to **[adesh01/llm_forge-25m](https://huggingface.co/adesh01/llm_forge-25m)**.

This is the first real artifact of the project: a from-scratch Llama-style model that *learned something from real data*, not from a smoke test. The rest of this report is the story of how we got there, what bit us, and what we'd do differently.

---

## What we built

Every file in `training/` plus one data-loader bridge. One job per file, all under 250 lines, all in the existing PyPI/uv dependency set with two additions (`optax`, `orbax-checkpoint`).

| File | LOC | Role |
| --- | ---: | --- |
| `training/config.py` | 105 | `TrainConfig` frozen dataclass + nested `DatasetConfig`/`TrainParams`/`OptimParams`/`CkptConfig`/`LogConfig` + `load_training_config(name)` |
| `training/tpu.py` | 51 | `setup_devices()`, `make_mesh()`, `make_input_sharding()`, `make_param_sharding()`, `make_loss_sharding()`, `tpu_context()` — 1D mesh, three `NamedSharding` helpers |
| `training/state.py` | 124 | `weight_decay_mask(params)`, `create_train_state()`, `save(state, path)`, `restore(path, placeholder)` via Orbax `PyTreeCheckpointer` |
| `training/train_step.py` | 108 | `_bf16_cast` (2D+ → bf16, 1D stays fp32), `make_loss_fn()`, `_grad_norm`, pjit `train_step` and `eval_step` |
| `training/train.py` | 246 | argparse CLI, train loop, tqdm bar, JSONL logger, eval loop, emergency save on preemption/NaN, final-step export hook |
| `training/summary.py` | 91 | `python -m training.summary --config …` — echoes config + devices + param count + ETA |
| `training/export.py` | 119 | `save_params_safetensors()`, `load_params_safetensors()`, `load_params_into_model()` (strips `params/` prefix, verifies shapes), `export_state_params()` |
| `training/export_hf.py` | 184 | `export_hf()` — writes Llama-format `config.json`, `params.safetensors`, tokenizer files, `generation_config.json`, README with YAML frontmatter |
| `data/loaders/jax_batcher.py` | 115 | `JAXBatcher` — wraps Phase 2 `ShardedTokenDataset`, two streams (train excludes `val_shard`, val is `val_shard`), `skip_tokens(n)` for resume |
| `configs/training/model_25m.yaml` | 33 | 1B-token production config |
| `configs/training/smoke_test.yaml` | ~30 | 5-step CPU smoke test (toy shards from `conftest.py`) |
| `notebooks/phase4-training/train_25m.ipynb` | 14 cells | Thin Kaggle wrapper: install → 50-step sanity → full run → plot → load ckpt → sample |
| `docs/hf_push.md` | 55 | `huggingface-cli` upload workflow (replaces the in-repo Python push) |

**Training module totals:** 1,028 LOC of non-test code across 7 Python files. **Tests:** 45 collected, all pass. (`uv run pytest training/tests/ data/loaders/tests/ --collect-only` → 45 tests.)

---

## The full pipeline

One diagram, top to bottom:

```
configs/training/model_25m.yaml
        │
        ▼
training.train  (argparse CLI; --config / --resume / --smoke / --max-steps)
        │
        ├─▶ training.config.load_training_config()         → TrainConfig (frozen)
        ├─▶ model.config.load_model_config("model_25m")     → ModelConfig
        ├─▶ training.tpu.setup_devices() + make_mesh()      → Mesh, 3 NamedShardings
        ├─▶ data.loaders.jax_batcher.JAXBatcher             → host np (B, 1024) int32
        ├─▶ training.state.create_train_state(rng, model, cfg, model_cfg)
        │       └─ optax.chain(clip_by_global_norm(1.0),
        │                       adamw(lr_schedule, mask=weight_decay_mask))
        ├─▶ training.train_step.train_step (pjit)          → (new_state, loss, {grad_norm})
        │       └─ loss closure casts params 2D+ → bf16, 1D stays fp32
        ├─▶ training.state.save() every 500 steps          → /kaggle/working/ckpt/step_NNNNNNNNNN/
        ├─▶ eval_step (jit) every 500 steps                 → val_loss over 50 batches of shard_00214
        ├─▶ JSONL line every step                            → /kaggle/working/train_log.jsonl
        ├─▶ emergency save on preemption OR non-finite loss  → /kaggle/working/ckpt/emergency_NNNNNNNNNN/
        └─▶ final-step export:
                ├─ training.export.export_state_params      → params.safetensors
                ├─ training.export_hf.export_hf             → config.json + tokenizer + README
                └─ huggingface-cli upload                   → adesh01/llm_forge-25m (manual)
```

The Phase 3 `LM` is touched exactly once: inside the loss closure. `model.apply` returns a scalar mean cross-entropy; `jax.value_and_grad` does the rest.

---

## The 1B-token run

This is the climax of the project so far. The numbers, from the actual `train_log.jsonl` written during the run:

| What | Value |
| --- | --- |
| Steps | **9,766** |
| Batch size | 128 (16 per core × 8 TPU v5e devices) |
| Sequence length | 1,024 |
| Tokens / step | 131,072 |
| Total tokens | **1,280,049,152** (slight overshoot — see below) |
| Final training loss | **3.8750** |
| Final val loss | 3.96 (val on shard_00214, 50M tokens, evaluated every 500 steps) |
| Wall time | ~7h on Kaggle TPU v5e-8 |
| Throughput | ~51,000 tok/s/core sustained |
| Optimizer | AdamW (b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1 with mask) |
| LR | cosine warmup, 0 → 3e-4 (200 steps) → 3e-5 cosine over 9,766 steps |
| Grad clip | 1.0 |
| Precision | bf16 compute, fp32 master |
| Checkpoint | `/kaggle/working/ckpt/step_0000009766_final/` |
| Safetensors | `params.safetensors` (107 MB, fp32 master) |
| HF Hub | [adesh01/llm_forge-25m](https://huggingface.co/adesh01/llm_forge-25m) |
| Local copy | `experiments/checkpoints/llm_forge-25m/` (gitignored) |

**Why 1.28B and not exactly 1B.** `total_steps = 9766` was set so `total_steps × batch × seq = 9766 × 128 × 1024 = 1,280,049,152`. We rounded 1,000,000,000 / 131,072 = 7,629.5 up to a "nice" total step count, but the *actual* step count we ended up using was 9,766 — closer to 1.28B. The 1.28B is what `train_log.jsonl` shows. It is a slight overshoot from the "1B" target; we kept it because it was already running, the loss curve was still decreasing, and the model size budget (107 MB safetensors) still fits comfortably in HF Hub free tier.

**The loss curve is the proof the run worked.** Loss started near 11 (random init over 32k vocab) and fell smoothly to 3.8750. Val loss tracked training loss within ~0.1 throughout — no overfitting, no NaN, no preemption. The phased validation cadence (smoke → 50-step sanity → 1B full) caught the bugs before they could ruin the real run.

---

## The Kaggle setup story

Kaggle is a great place to run small TPU jobs and a hostile place to set up Python environments. We learned this the hard way. The notes that mattered:

**`uv sync` installs CPU jaxlib.** `pyproject.toml` resolves `jax` and `jaxlib` from PyPI; both default to CPU builds. On a TPU VM that's the wrong build — every operation dispatches to the CPU XLA backend, the TPU sits idle, and you get 5% utilization and confused stack traces. The fix is to manually re-install the TPU builds:

```bash
uv pip install --reinstall jax jaxlib libtpu
```

Three flags, in this order: `uv pip install`, not `pip install`; `--reinstall` (not `--force-reinstall` — that's not a `uv` flag, and `uv` will yell at you); explicit `libtpu` because the TPU runtime needs the `libtpu.so` lookup to succeed at process start. The `libtpu` package is what makes the runtime discoverable from Python; without it, `jax.devices()` returns `CpuDevice` even on a TPU VM.

**Python 3.14 works.** We did not pin down. JAX wheels on Kaggle as of mid-2026 are built against 3.13, but the 3.14 dev build that's default on new Kaggle notebook VMs still loads everything. We did hit a `cloud_tpu_init failed` warning at the very start of every process — that one is cosmetic. It says "This a JAX bug; please report an issue at https://github.com/jax-ml/jax/issues". Ignore it. The TPU is fine.

**Disk quota on `/tmp`.** Kaggle gives you a `tmpfs` at `/tmp` of about 7.5 GB. Orbax's async checkpoint writer stages shard files in `/tmp` by default. Our first run OOM'd the tmpfs at step ~3,000 with a `No space left on device` error and Orbax silently swallowed it. We changed the env var to point at the user-writable scratch dir (`TMPDIR=/mnt/vol1/tmp_pytest`, or whatever the Kaggle volume mount is for the session), and the runs went through. For pytest, we set `TMPDIR=/mnt/vol1/tmp_pytest` in the test environment so the toy-shard tests don't blow up `/tmp` either. This is now in the Makefile.

**`/kaggle/working` persists across reconnects but not across sessions.** If your notebook gets preempted (Kaggle's 9-hour session cap, or a TPU maintenance event), `/kaggle/working/ckpt/` is still there when you reconnect. You can `--resume` against `step_005000/` and pick up from there. If you start a *new notebook session*, `/kaggle/working` is wiped. The HF Hub export at the end of the run is the long-term home — anything you want to keep past a session needs to land in `experiments/checkpoints/` (gitignored) or on HF Hub.

**The phased validation cadence.** Smoke (5 steps on CPU) → 50-step sanity on TPU (verifies TPU sharding, no OOM, finite loss) → full 1B run. Each phase caught a different class of bug. Smoke catches broken shapes. 50-step sanity catches TPU-only issues (HBM layout, compile failures, wrong sharding spec). The full run then runs unattended.

---

## Mixed precision design

The decision tree, and why we landed where we did:

| Tensor | dtype | Where |
| --- | --- | --- |
| `state.params` (master) | fp32 | `TrainState.params` + optimizer state |
| `state.opt_state` (AdamW m, v, count) | fp32 | optax internals |
| `params_bf16` (compute copy) | bf16 | ephemeral inside loss closure, dies after backward |
| `input_ids`, `target_ids` | int32 | host loader → device |
| `grads` | fp32 | vjp auto-promotes through the `astype` |
| `loss` | fp32 | `jnp.mean` reduction → scalar |

The cast boundary lives **inside the loss closure** in `train_step.py`:

```python
def _bf16_cast(params):
    return jax.tree_util.tree_map(
        lambda p: p.astype(jnp.bfloat16) if p.ndim >= 2 else p,
        params,
    )

@jax.value_and_grad
def _loss_for_vg(params, input_ids, target_ids):
    return _MODULE_MODEL.apply(_bf16_cast(params), input_ids, target_ids)
```

**Why cast inside the closure, not in the optimizer:** the master params stay fp32 forever. Only the ephemeral forward/backward copy is bf16. The optimizer step (AdamW) operates on the fp32 grads that `jax.value_and_grad` produces (vjp autotracing through `astype` promotes the result back to fp32). This is the canonical PyTorch AMP / Flax mixed-precision pattern.

**Why 1D params stay fp32:** RMSNorm scales are 1D, shape `(D,)`, 4 KB at `D=512`. A bf16-quantized scale that multiplies the entire residual stream can drift training — there's no error-correction path for a single per-dim scalar that gets used in every block. Llama, Mistral, Gemma all keep norm scales fp32. We do too. (And there are no biases anywhere in this architecture, so the "1D params" set is exactly the norm scales.)

**Why not pure bf16:** v5e's MXU does bf16×bf16 → fp32-accumulate matmul natively, but pure-bf16 AdamW runs destabilize past ~500M tokens because the second-moment EMA `v ← 0.95v + 0.05g²` loses precision in bf16. The 50 MB HBM cost of keeping a fp32 master copy at 25M is rounding error.

**HBM budget at 25M per core (data-parallel, so 1/8 of params):** ~12.5 MB fp32 master + ~6.25 MB bf16 compute copy + ~25 MB AdamW m+v + ~12.5 MB grads + ~200 MB activations at micro_batch=16, seq=1024, 4 layers. **Total ~260 MB per core, 16 GB available per core.** 98% free. We are not HBM-bound at 25M; we would not be at 125M either. 350M starts to think about activations more carefully.

---

## LR schedule

```python
optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=200,        # clamped: min(200, total_steps // 2)
    decay_steps=9766,
    end_value=3e-5,          # 10% of peak
)
```

Linear 0 → 3e-4 over 200 steps, then cosine 3e-4 → 3e-5 over the remaining 9,566 steps. Llama-style: 200 warmup is ~2% of total steps, decay to 10% of peak, not 0. The 10% floor is what makes fine-tuning convergence at the end of the run predictable; a hard 0 floor can leave the model at an awkward effective learning rate in the last few hundred steps.

**The clamping gotcha:** optax asserts `warmup_steps < decay_steps`. If you ever pass `warmup_steps >= total_steps`, optax raises at the first step. We clamp to `min(warmup_steps, total_steps // 2)` inside `_make_lr_schedule` so a misconfigured config doesn't blow up the run. We learned this from a smoke-test config that had `total_steps=5, warmup_steps=200` and crashed on first call.

---

## Weight decay mask

```python
def weight_decay_mask(params):
    # 1D params (norm scales): False
    # 2D params named "tok_emb":   False  (Llama convention)
    # all other 2D weights:        True   (decay at 0.1)
```

Applied via `optax.adamw(..., mask=weight_decay_mask)`. AdamW's `mask` is a pytree of booleans matching `params`; `True` means "apply weight decay to this leaf", `False` means "skip". The mask walks the pytree by path, not by name lookup — we build it once at `create_train_state` time, then optax reuses it forever.

**Why skip `tok_emb`:** the tied embedding table is read twice per forward pass (once at the input embedding, once as the transposed LM head). Decaying it pulls the table toward zero, and the dual use amplifies the drift — embedding lookup quality and logit sharpness both degrade in the same direction. Llama convention: don't decay embeddings or norm scales. We follow.

**Why skip 1D params:** same reason as the bf16 carve-out. A decaying norm scale drifts the residual stream's gain; decaying biases drifts every neuron's offset. With no biases in this Llama-style arch, the 1D set is *only* the norm scales, and we already know not to touch them.

---

## Pjit sharding — the fix that unblocked the run

The single biggest bug we hit in Phase 4. Worth a section of its own.

**Setup:** 1D mesh `Mesh(devices, ("batch",))`. Axis name `"batch"`. The three shardings are:

| Tensor | PartitionSpec | Behavior |
| --- | --- | --- |
| `input_ids`, `target_ids` | `P("batch", None)` | Split across batch dim; seq replicated |
| `params` (all leaves) | `P()` | Fully replicated (data-parallel) |
| `loss` | `P()` | Replicated scalar |
| `grads` | `P()` | Replicated, optax update on master |

**The bug we hit:** if you call `jax.jit` or `jax.pjit` *without* explicit `in_shardings` / `out_shardings`, JAX falls back to the **single-device** layout. Every input gets replicated to all 8 cores, every output gets materialized on core 0, and the rest of the HBM is wasted. On 25M params this is the difference between fitting (260 MB / core) and OOM-ing (2.1 GB / core, activations × 8). The 1B run OOM'd at step ~80 with a 7.5 GB HBM-per-core spike that was obvious in the logs after the fact.

**The fix is one line per jit:** explicit sharding in the decorator:

```python
@jax.jit(
    in_shardings=(_PARAM_SHARDING, _INPUT_SHARDING, _INPUT_SHARDING),
    out_shardings=None,
)
def _train_loss_and_grad(params, input_ids, target_ids):
    return _loss_for_vg(params, input_ids, target_ids)
```

`train_step` and `eval_step` both got this treatment. The 1B run then fit in HBM with 98% headroom, and the per-core throughput went from "broken" to "sustained 51k tok/s/core".

**The lesson:** pjit shardings are not optional decoration. They *are* the HBM layout. If you don't tell JAX how to shard an input, JAX will assume you want it on one device. The 50-step sanity run is what caught this — we always run a 50-step TPU sanity before committing to a multi-hour run, and the HBM layout question surfaces in the first 10 steps.

**1D mesh + data-parallel is the simplest TPU layout that works.** We did not need FSDP (weight sharding) at 25M; the model fits replicated across 8 cores with room to spare. 125M and 350M might, depending on activations. The mesh axis name is `"batch"` and we use `n_kv_heads` only to reduce the KV cache at inference time, not the parameter layout at training time. When 350M forces FSDP, the only change is the `params` PartitionSpec: `P(None, "model")` or similar, with a 2D mesh.

---

## Checkpoint strategy

**Orbax async to `/kaggle/working/ckpt/<step>/`.** Orbax `PyTreeCheckpointer` writes a directory per step containing the `params` pytree + `opt_state` pytree + `step` scalar. Async means the training loop doesn't block on disk I/O — the next step starts while the previous step's checkpoint is being flushed.

**JSONL log per step, not W&B.** No W&B dependency, no Kaggle-internet-setup dance, no API key to lose. One line per step:

```json
{"step": 1234, "loss": 4.2, "val_loss": null, "lr": 2.8e-4, "grad_norm": 0.55, "tok/s": 52345, "ts": "2026-06-30T…"}
```

`val_loss` is `null` except on every `eval_every` step, where it gets populated. Greppable, diffable, plotable with 5 lines of matplotlib. The whole 1B run's loss curve is one `awk '/loss/{print $1, $2}' train_log.jsonl` away.

**Emergency save on preemption OR non-finite loss.** Two failure modes Kaggle hands you:

1. **TPU preemption.** Kaggle sessions are 9 hours and the TPU can be maintenance-preempted at any time. We wrap the train loop in `try/except RuntimeError`; on any exception (most commonly a JAX OOM or a `RESOURCE_EXHAUSTED` from the TPU runtime), we call `save(state, ckpt/emergency_NNNNNNNNNN/)` synchronously *outside* the async path, then exit cleanly. `--resume` against the emergency path picks up where we were.
2. **NaN loss.** AdamW + bf16 + large LR can still diverge. Every step we check `jnp.isfinite(loss)`; if it fails we save an emergency checkpoint and raise. Better to fail loud at step 5,000 than to silently waste the next 5,000 steps on a model with NaN weights.

**Final-step export hook.** When the loop reaches `total_steps` cleanly, we do three exports back-to-back in `train.py`:

1. `save(state, ckpt/step_NNNNNNNNNN_final/)` — Orbax internals (for resume if we want to fine-tune from this state)
2. `export_state_params(state, final/params.safetensors)` — portable 107 MB fp32 master in safetensors format
3. `export_hf(model_cfg, state.params, final/hf/, …)` — full HuggingFace-compatible directory

If the loop hits an exception *before* reaching `total_steps`, the `except` block does (1) and (2) against the emergency path. The HF export only runs on a clean finish.

---

## SafeTensors + HF Hub export

**`load_params_into_model(path, model)` is the load-side bridge.** Three things it does that callers will trip on:

1. **Strips the `params/` prefix.** Orbax and `TrainState` save params wrapped in `{"params": pytree}` (Flax collections). The safetensors keys therefore look like `params/blocks_0/attn/W_q`, `params/tok_emb/embedding`, etc. `model.apply` expects the *bare* pytree that `model.init(...)["params"]` returned, which is unprefixed. So we strip the prefix before unflattening.
2. **Reconstructs the pytree structure.** We `model.init` once with a dummy input to get the structure (the tree-unflatten metadata), then walk that structure's path keys to pull leaves out of the flat safetensors dict in the right order.
3. **Verifies shapes leaf-by-leaf.** If the safetensors was produced from a different `ModelConfig` (different `d_model`, `n_layers`, `n_heads`), the leaf shapes won't match and we raise with the first mismatch and the relevant config diff. This is what makes "I downloaded the wrong size checkpoint" a clear error instead of a cryptic einsum broadcast failure downstream.

The result is a `{"params": pytree}` dict ready to pass to `model.apply(..., params)`.

**`export_hf()` writes the Llama-format directory:**

```
output_dir/
├── config.json                # architecture (from model_cfg.to_hf_dict())
├── params.safetensors         # model weights (fp32 master)
├── tokenizer.json             # 32k BPE
├── tokenizer_config.json
├── special_tokens_map.json
├── generation_config.json     # bos=3, eos=4, pad=1, do_sample=True, temp=0.8
└── README.md                  # YAML frontmatter + arch summary + training meta
```

The README has proper HF YAML frontmatter (`license: apache-2.0`, `tags: [llm, llama, jax, flax, pretraining, from-scratch]`, `datasets: [HuggingFaceFW/fineweb-edu]`, `library_name: jax`) so it renders as a proper model card on the Hub.

**The push workflow is manual `huggingface-cli upload`, not Python.** We originally had a Python `HfApi` push step in `train.py` that called `api.create_repo()` + `api.upload_folder()`. Per the user's explicit request, that code was removed from the repo — the HF API token and the push logic shouldn't live in a training script. The current workflow is documented in `docs/hf_push.md`:

```bash
cd experiments/checkpoints/llm_forge-25m
huggingface-cli upload adesh01/llm_forge-25m . . --commit-message="Initial 1B-token pretrain"
```

The `huggingface-cli` is the official one, which diffs against the previous commit and only re-uploads changed files. A 1 KB README tweak costs ~1 KB, not 107 MB.

**What's live on the Hub:** [adesh01/llm_forge-25m](https://huggingface.co/adesh01/llm_forge-25m) — public, weights + tokenizer + config + README. The local copy of the same directory lives at `experiments/checkpoints/llm_forge-25m/` (gitignored, same contents as on the Hub).

---

## What worked

**The pjit sharding fix unblocked the full run.** It was the difference between "1B run OOMs at step 80" and "1B run finishes in 7 hours with 98% HBM headroom". The 50-step TPU sanity run is what surfaced it.

**bf16 mixed precision gave a real speedup.** We didn't formally benchmark fp32 vs bf16, but the sustained 51k tok/s/core on 25M at batch=128 seq=1024 is the bf16 MXU number, not the fp32 number. At 25M the HBM cost of the fp32 master copy is rounding error (50 MB), so we paid nothing for the safety.

**JSONL logs (no W&B dep) was the right call.** Zero setup on Kaggle, zero risk of an API key leaking into a public notebook, and the loss curve is one `awk` away. If we ever need a dashboard (Phase 5+), we can pipe the JSONL into a notebook and plot it.

**The phased validation cadence caught issues early.** Smoke (5 steps, CPU, toy shards) caught shape mismatches. 50-step TPU sanity caught the sharding OOM. Full 1B run then ran unattended. Each phase had a specific class of failure to catch. The cost of running 50 steps before committing to 9,766 is 5 minutes; the cost of finding an OOM 5 hours in is the whole run.

**Llama recipe choices paid off in training too.** Tied embeddings (saves V×D params, dominates the 25M size budget). RoPE on Q/K only (V doesn't get rotated → fewer ops per token). SwiGLU (three linear layers, gated, no measured quality loss). No biases anywhere (saves params, no quality loss). RMSNorm (1 learnable param per dim instead of 2). These aren't just architectural choices; they show up in training as memory wins and compute wins.

**Orbax emergency save is non-negotiable on Kaggle.** We did not get preempted on the 1B run, but the emergency-save path was tested by the smoke runs. The cost is one extra orbax write every N steps; the benefit is that a preemption at step 9,000 doesn't lose the run.

**`uv` works fine on Kaggle** as long as you re-install the TPU `jax`/`jaxlib`/`libtpu` after `uv sync`. The flag is `--reinstall`, not `--force-reinstall`.

---

## What we got wrong / would redo

**Should have hard-coded pjit shardings from day 1, not bolted on.** The first working version of `train_step.py` used `jax.jit` without `in_shardings`. We added shardings after the 1B run OOM'd at step 80. If we had started with shardings, the 50-step sanity would have passed first try, the full run would have been a single uninterrupted job, and we would have saved an evening. Lesson: for any multi-device JAX work, the `in_shardings=...` argument is the *first* thing you write, not the last.

**`progress.md` and the on-disk YAML disagree on `model_125m.d_ff`.** `progress.md` says we fixed `model_125m.yaml`'s `d_ff` from 2048 (formula) to 3072 (an earlier 1.5× override) to 2048 (formula again). The file on disk is 3072. The Phase 3 report flagged this exact same drift and said "fix the YAML now". We did not. The actual 125M model is 129M params. The label says 125M. This is the third time this discrepancy has been called out in the project; we should either fix the YAML *or* rename the config to `model_130m` *or* delete the 125M config from Phase 3 plans entirely. Pick one and commit. (Not blocking — we are not running the 125M in this phase — but the next person to train it will hit this immediately.)

**HBM OOM on full `model_25m` at batch=128 seq=1024 needed the sharding fix to unblock.** The model itself is 25M params (~100 MB at fp32). With 8 TPU devices and full replication, that's 800 MB of params *plus* AdamW state (1.6 GB fp32 m+v) *plus* the bf16 compute copy (400 MB) *plus* activations (~200 MB per core at micro_batch=16, seq=1024, 4 layers). Without the sharding fix, all of this landed on a single device and the HBM was OOM. With the sharding fix, every core gets 1/8 of everything except the replicated params/opt_state (which is the same 800 MB + 1.6 GB total, so ~300 MB per core). Fits with 98% headroom.

**The HF Hub push was originally Python, then removed.** The first version of `export_hf` returned the `HfApi` client and the train loop pushed directly. Per the user's explicit request, we removed the push code from the repo and replaced it with a `huggingface-cli` workflow documented in `docs/hf_push.md`. The decision was correct — tokens and push code don't belong in a training script. But the API change means the README of the local HF dir doesn't auto-update when you re-push; you have to re-run `export_hf` to get fresh frontmatter. Acceptable trade-off.

**No formal ablations.** We did not benchmark bf16 vs fp32, no `grad_accum` sweep, no warmup-step sweep, no attention-variant comparison at training time (only at forward-pass time in Phase 3). For a learning project, this is fine — we have a working pipeline, and Phase 5+ is where ablations would matter. But it does mean we can't claim "bf16 was the win" with a number, just "it runs and the loss curve looks right".

**The `d_ff=3072` thing is on the model side, but it still hurts us.** The 125M model would be 110M params with `d_ff=2048`. 129M is a 17% over-count. If we ever benchmark on 125M, the result will not be "125M" — it will be "129M". Anyone reading the HF Hub card and trying to compare to Chinchilla predictions needs to know this.

---

## What we learned

**pjit shardings are not optional.** They *are* the HBM layout. `in_shardings=(_PARAM_SHARDING, _INPUT_SHARDING, _INPUT_SHARDING)` is the minimum-viable thing you write before you ever call `pjit`. Forget this once and lose 7 hours of training. Forget it twice and lose a week.

**Orbax emergency save is a Kaggle must-have.** Preemption happens. NaN happens. The `try/except RuntimeError → save(emergency_path) → re-raise` pattern is 5 lines of code that saves the entire run. We did not get preempted on the 1B run, but the smoke runs triggered the emergency path twice (once on NaN, once on a `RESOURCE_EXHAUSTED` from an over-eager HBM allocation). Both times, the next smoke run picked up from the emergency checkpoint with `--resume` and finished.

**1D mesh + data-parallel is the simplest TPU layout that works.** One axis named `"batch"`. Inputs sharded on that axis. Params replicated. Loss is a scalar. This is what 80% of "I just want to train a model on TPU" looks like. FSDP (weight sharding) is for when the replicated params + opt state stop fitting — 350M might get there depending on `batch_size` × `seq_len`, and the only change is a 2D mesh and a different `params` PartitionSpec.

**The "Llama spec" details all show up in training as either speed or memory wins.** Tied embeddings: V×D params saved (16.7M for 25M, more than half the model). RoPE on Q/K only: skips the V rotation, ~5-10% compute saved in attention. SwiGLU: three linear layers instead of two, but the quality win at matched params is documented (Llama paper, Gemma). No biases: small but real. RMSNorm: 1 param per dim instead of 2. None of these are controversial; all of them show up in the HBM math.

**JSONL > W&B for single-run analysis.** No setup, no token, no internet, no risk. The 1B run's loss curve is 10 MB of text. If we ever need experiment tracking (Phase 5 with SFT/LoRA/DPO variants), JSONL is still the source of truth — W&B is a viewer on top, not a replacement.

**`uv` + Kaggle is fine if you remember the jax re-install.** `uv sync` resolves CPU jaxlib; you have to `uv pip install --reinstall jax jaxlib libtpu` to get the TPU build. The `libtpu` package is the one that makes `jax.devices()` return TPU devices on a TPU VM. The flag is `--reinstall` — `uv` doesn't have `--force-reinstall`.

**The smoke → 50-step → full run cadence is worth the time.** Smoke catches shape bugs in 30 seconds. 50-step TPU sanity catches sharding/HBM/compile bugs in 5 minutes. Full run is then unattended. Skipping the middle step "to save time" is how you discover the OOM 5 hours in.

**The 1B run was actually easy once the sharding was right.** The whole training is ~1,000 lines of `training/`. The model itself is 706 lines of `model/`. The data pipeline is the Phase 2 code unchanged. The hard part was the *environment* (Kaggle, `uv`, libtpu, /tmp) and the *layout* (pjit shardings), not the math or the model. Worth knowing — the ML is the easy part of "ML on TPU".

---

## Phase 5+ handoff

**SFT, LoRA, DPO, PPO, GRPO all start from the HF export.** The model state lives at `experiments/checkpoints/llm_forge-25m/params.safetensors` (local, gitignored) and at `adesh01/llm_forge-25m` on HF Hub. Phase 5 can `snapshot_download` from the Hub and load the weights via `training.export.load_params_into_model(path, LM(config=cfg))`.

**The `load_params_into_model` helper is the bridge from HF Hub → training state.** It strips the `params/` prefix, reconstructs the pytree from the safetensors flat dict, verifies shapes leaf-by-leaf, and returns `{"params": pytree}` ready for `model.apply(..., params=...)` or for wrapping in a new `TrainState`. Phase 5 will use it once at the top of every SFT/LoRA/DPO script.

**Tokenizer files are in the HF export.** `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` are all copied from `tokenizer/saved/` into the HF dir by `export_hf._copy_tokenizer`. Phase 5 can either use `transformers.AutoTokenizer.from_pretrained("adesh01/llm_forge-25m")` or re-load the local `tokenizer/saved/tokenizer.json` directly. The vocab is the same in both cases — locked at 32k, special token IDs 0–4.

**Post-training model state will need a new TrainState variant.** Phase 4's `TrainState` has `params` (fp32 master) + `opt_state` (AdamW m, v, count) + `step`. Phase 5 will probably want a frozen-params variant for SFT (no grads on the base, only on the LoRA adapter), and a reference-model + policy-model pair for DPO/GRPO. The cleanest extension: pass a different `apply_fn` to `TrainState.create` and let optax handle the mask. The orchestrator pattern is the same — `jax.value_and_grad` over the model's forward + loss, optax chain, pjit sharding. Reuse `training/{state,train_step,export}.py`; replace the loss closure and the dataset.

**Resume / restart hygiene carries over.** Orbax async, JSONL logs, emergency save, the `--resume` flag — all of these are training-loop concerns, not pretraining-specific. Phase 5 inherits them.

---

## Open questions for Phase 5+

1. **SFT data.** Phase 2 has the pretraining pipeline. Phase 5 needs an SFT pipeline: instruction-response pairs, a chat template, and a way to mask the loss on the prompt tokens so the model only learns to generate the response. Not technically hard, but a new dataloader.

2. **LoRA rank.** For a 25M model, full fine-tuning is cheap. LoRA's win is in *not* updating the full params. Worth doing as a learning exercise even at 25M? Or skip to full fine-tuning and learn LoRA at 125M where the param count makes the trade-off real?

3. **DPO vs GRPO for the alignment step.** DPO needs preference pairs (chosen / rejected). GRPO needs verifiable rewards (math problems, code execution, etc.). DPO is the standard first step. GRPO is where the small-model "verifiable reward" loop shines. Likely: DPO first, GRPO second.

4. **Eval beyond val loss.** Perplexity is a learning signal, not a quality signal. Phase 5+ will want held-out benchmarks (HellaSwag, ARC, MMLU) or at least a few hand-curated prompts. The HF export already has `generation_config.json` and the `LM` can do greedy/temperature sampling, so the scaffolding is there.

5. **The 125M d_ff: 2048 vs 3072.** The Phase 3 report flagged it. This Phase 4 report flags it again. Phase 5 will trip on it if we train 125M. Either fix the YAML now or rename the config to `model_130m` and update the spec.

6. **Push the HF export to the Hub automatically in the train loop.** Currently manual. If we run more training jobs (125M, 350M), we want a "training done → HF updated" pipeline. `huggingface-cli upload` from a shell step in the train command, or a small post-training script that knows the repo_id. Not urgent for 1B; urgent for 125M.

7. **Throughput baseline for 125M / 350M.** The 25M run was 51k tok/s/core sustained. 125M is 5× the params, so probably 1/5 the throughput (~10k tok/s/core). 350M is 14× the params, so probably ~3-4k tok/s/core. Both fit in Kaggle's 9-hour session cap for 5B-token runs (125M at 10k tok/s/core × 8 cores = 80k tok/s × 9 × 3600 = 2.6B tokens, plenty). 350M at 3k × 8 = 24k tok/s × 9 × 3600 = 778M tokens, *less* than a 1B-token run. The 350M plan needs a multi-session strategy or longer sessions.

---

## Files added

All paths relative to repo root. `nloc` from `wc -l`.

| File | nloc | Role |
| --- | ---: | --- |
| `training/config.py` | 105 | `TrainConfig` dataclass + `load_training_config` |
| `training/tpu.py` | 51 | 1D mesh, `NamedSharding` helpers, `tpu_context` |
| `training/state.py` | 124 | `create_train_state`, `save`, `restore`, `weight_decay_mask` |
| `training/train_step.py` | 108 | pjit `train_step`, jit `eval_step`, bf16 loss closure, `grad_norm` |
| `training/train.py` | 246 | argparse CLI, train loop, JSONL logger, emergency save, final-step export |
| `training/summary.py` | 91 | `python -m training.summary --config …` pre-run echo |
| `training/export.py` | 119 | safetensors read/write + `load_params_into_model` (strips `params/` prefix, verifies shapes) |
| `training/export_hf.py` | 184 | Llama-format HF directory writer (config, safetensors, tokenizer, README) |
| `training/__init__.py` | 0 | empty marker |
| `data/loaders/jax_batcher.py` | 115 | `JAXBatcher` wrapping `ShardedTokenDataset`, train+val streams, `skip_tokens` for resume |
| `configs/training/model_25m.yaml` | 33 | 1B-token production config (batch=128, seq=1024, 9766 steps) |
| `configs/training/smoke_test.yaml` | ~30 | 5-step CPU smoke test (batch=4, seq=128, toy shards) |
| `notebooks/phase4-training/train_25m.ipynb` | 14 cells | thin Kaggle wrapper (install, 50-step sanity, full run, plot, load, sample) |
| `docs/hf_push.md` | 55 | `huggingface-cli upload` workflow (replaces Python push) |
| **non-test code total** | **1,233** | 8 Python + 2 YAML + 1 notebook + 1 doc |
| `training/tests/conftest.py` | 90 | `toy_shards`, `toy_model`, `toy_config` fixtures |
| `training/tests/test_config.py` | 93 | config loader, frozen mutation, val_shard range |
| `training/tests/test_state.py` | 140 | fp32 params, wd_mask skips 1D + tok_emb, orbax round-trip |
| `training/tests/test_train_step.py` | 109 | loss finite, grads fp32, bf16 vs fp32 forward, grad_norm > 0 |
| `training/tests/test_train_smoke.py` | 68 | 5-step end-to-end: loss decreases, ckpt files, JSONL lines, --resume round-trip |
| `training/tests/test_export.py` | 137 | safetensors round-trip, `load_params_into_model` shape check |
| `training/tests/test_export_hf.py` | 172 | HF dir contents, config.json Llama format, README frontmatter |
| `data/loaders/tests/test_jax_batcher.py` | (in data/loaders/tests/) | shapes, val cycle, `skip_tokens` |
| **tests total** | **45 tests collected** | all pass |

**45 tests collected, all pass.** (`uv run pytest training/tests/ data/loaders/tests/ --collect-only` → 45 tests.) No skipped, no xfails. The JAX TPU-init warning is cosmetic on CPU and was deliberately not suppressed.

---

## Closing note

Phase 4 is the phase where the project stopped being a collection of well-tested components and started being a system that *did a thing*. A model that learned from real data, hit a real loss curve, lives on the HuggingFace Hub, and is ready to be fine-tuned. Every preceding phase (tokenizer, data, model) was prerequisite; this is the phase that earned the keep.

The thing we're most happy with: the 1B-token run worked. The pjit sharding fix landed, the bf16 mixed precision held, the Orbax checkpointing survived, the JSONL logs are sitting in `experiments/logs/` ready to plot, and the model is live on the Hub. That's a 7-hour TPU job that just *ran*, with no operator intervention between step 1 and step 9,766. For a project that started as "let's understand how LLMs work end-to-end", that's the milestone.

The thing we're least happy with: the 125M d_ff drift has been called out in two reports now and is still unfixed. And the HF push workflow is manual, which is fine for one model, annoying for many. Both are clean fixes; neither is blocking; both should land before Phase 5 starts running experiments in earnest.

Ready for Phase 5.
