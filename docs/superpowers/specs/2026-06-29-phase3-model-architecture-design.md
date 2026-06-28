# Phase 3 — Model Architecture Design

**Status:** Approved (visual companion review, sections 1–4)
**Date:** 2026-06-29
**Owner:** adeshboudh
**Scope:** Build the Llama-style transformer model that Phase 4 will pretrain.

## 1. Goal

Build three numerically tested standalone attention variants (MHA, MQA, GQA),
the supporting Llama-style components (RoPE, SwiGLU, RMSNorm), wire them into
a full causal language model whose forward pass returns a scalar cross-entropy
loss. End state: a runnable LM that Phase 4 wraps with Optax + a train loop.

**Non-goals:** training, checkpointing, inference, mixed precision. Those belong
to Phases 4 and 6. This phase delivers a forward pass and a regression test
harness.

## 2. Stack

- **Framework:** JAX + Flax (`flax.linen` modules everywhere)
- **Parameter management:** Flax `nn.Module` with `setup()`/`__call__`
- **Optimizer:** not in this phase (Optax is the planned choice for Phase 4)
- **Dtype:** float32 only in Phase 3 — mixed precision deferred to Phase 4

## 3. Package layout

```
model/
├── __init__.py
├── config.py                       # ModelConfig dataclass + load_model_config()
├── attention/
│   ├── __init__.py
│   └── variants/
│       ├── __init__.py
│       ├── mha.py                  # CausalMHA(nn.Module)
│       ├── mqa.py                  # CausalMQA(nn.Module)
│       └── gqa.py                  # CausalGQA(nn.Module)
├── embeddings/
│   ├── __init__.py
│   └── rope.py                     # apply_rope(q, k, theta_base)
├── mlp/
│   ├── __init__.py
│   └── swiglu.py                   # SwiGLUMLP(nn.Module)
├── normalization/
│   ├── __init__.py
│   └── rmsnorm.py                  # RMSNorm(nn.Module)
├── blocks/
│   ├── __init__.py
│   └── transformer_block.py        # TransformerBlock(nn.Module)
└── lm.py                           # LM(nn.Module) — full causal LM, tied emb, scalar loss

configs/models/
├── model_25m.yaml
├── model_125m.yaml
└── model_350m.yaml

model/tests/
├── __init__.py
├── test_rope.py
├── test_rmsnorm.py
├── test_swiglu.py
├── test_mha.py
├── test_mqa.py
├── test_gqa.py
├── test_transformer_block.py
├── test_lm.py
└── test_smoke.py
```

**Why one variant per file (learning stack):** each variant owns its
Q/K/V/O projections — ~80 lines per file. The duplication between MHA, MQA, and
GQA is the lesson: diff the three files to see exactly what changes. A shared
helper module would hide the teaching artifact.

## 4. Module internals

### 4.1 RoPE — `model/embeddings/rope.py`

**Signature:** `apply_rope(q, k, theta_base=10000.0) -> (q_rot, k_rot)`

**Shapes:** `q, k : (B, T, H, D_h) → (B, T, H, D_h)` (rotated in-place)

**Math:**
```
inv_freq[i]   = 1.0 / (theta_base ** (2i / D_h))   for i in 0..D_h/2
positions     : (T,)
angles[t,i]   = positions[t] * inv_freq[i]          # (T, D_h/2)
cos, sin       : (T, D_h/2) each
q[... 2i]      = q[... 2i] * cos - q[... 2i+1] * sin
q[... 2i+1]   = q[... 2i+1] * cos + q[... 2i] * sin
# same for k
```

**Conventions:**
- Rotates Q and K only — V is **not** rotated (Llama convention).
- `theta_base = 10000.0` (Llama default, configurable per model).
- Pure function (no Flax module state) — just jnp ops.

**Tests:** identity at pos 0; rotation angle = pos·inv_freq; shape preserved;
roundtrip with inverse RoPE returns identity; V not rotated when forwarded.

### 4.2 RMSNorm — `model/normalization/rmsnorm.py`

**Signature:** `RMSNorm.__call__(x) -> x'` with attribute `dim: int`, `eps: float = 1e-6`

**Shapes:** `x : (B, T, D) → (B, T, D)`

**Params:** `scale : (D,)`, initialized to `1.0`

**Math:**
```
ms      = mean(x**2, axis=-1, keepdim=True) + eps
x_norm  = x * rsqrt(ms)                  # no mean subtraction (vs LayerNorm)
out     = scale * x_norm
```

**Tests:** zero input → zero output; output RMS ≈ 1.0; scale=2 → RMS=2.0; shape
preserved; exactly D params; gradient flows to scale; eps works at 0.1.

### 4.3 SwiGLU MLP — `model/mlp/swiglu.py`

**Signature:** `SwiGLUMLP.__call__(x) -> y` with attributes `d_model: int`, `d_ff: int`

**Shapes:** `x : (B, T, D) → (B, T, D)`

**Params:** `W_gate : (D, I)`, `W_up : (D, I)`, `W_down : (I, D)`. No biases.

**Math:**
```
gate = silu(x @ W_gate)        # silu(x) = x * sigmoid(x)
up   = x @ W_up
h    = gate * up
y    = h @ W_down
d_ff = round(8/3 * D / 256) * 256    # Llama ratio, rounded to multiple of 256
```

**Tests:** shape preserved; param count = `2·D·I + I·D`; zero input → zero
output; all three weights receive non-zero gradient; `d_ff` matches formula.

### 4.4 Attention variants — `model/attention/variants/{mha,mqa,gqa}.py`

**Common signature:**
```python
class CausalX(nn.Module):
    d_model:    int
    n_heads:    int
    n_kv_heads: int   # = n_heads for MHA, = 1 for MQA, configurable for GQA
    theta_base: float = 10000.0

    def __call__(self, x: Array) -> Array:
        # x: (B, T, D) → output: (B, T, D)
```

**Params (all three):** `W_q : (D, H·D_h)`, `W_k : (D, n_kv·D_h)`,
`W_v : (D, n_kv·D_h)`, `W_o : (H·D_h, D)`. No biases.

**MHA** (`n_kv = n_heads`):
```
Q = x @ W_q  → (B, T, H, D_h)
K = x @ W_k  → (B, T, H, D_h)
V = x @ W_v  → (B, T, H, D_h)
[Q, K] = apply_rope(Q, K, theta_base)
scores = Q @ K^T / sqrt(D_h)        # (B, H, T, T)
scores += causal_mask                # -inf above diagonal (jnp.tril)
attn = softmax(scores, axis=-1)
out = attn @ V                       # (B, T, H, D_h)
out = out.reshape(B, T, H·D_h) @ W_o # (B, T, D)
```

**MQA** (`n_kv = 1`):
```
K, V : (B, T, 1, D_h)        # single KV head with explicit head dim
K = repeat(K, H, axis=head)  # (B, T, H, D_h) — each of H Q heads shares the same KV
V = repeat(V, H, axis=head)
# rest identical to MHA
```

**GQA** (`n_kv configurable`, require `H % n_kv == 0`):
```
K, V : (B, T, n_kv, D_h)
K = repeat_interleave(K, H // n_kv, axis=head)    # (B, T, H, D_h)
V = repeat_interleave(V, H // n_kv, axis=head)
# rest identical to MHA
```

**Convention:** RoPE applied to Q and K only (not V). Scaling by `1/sqrt(D_h)`.
Causal mask = `jnp.tril(ones((T, T)))` with `-inf` in upper triangle.

**Validation:** `n_kv` must divide `n_heads` cleanly. ValueError raised at
module construction if not (enforced in GQA; MHA hardcodes n_kv=n_heads; MQA
hardcodes n_kv=1).

**Tests (shared):** shape preserved; param count =
`D·H·D_h·2 + D·n_kv·D_h·2` i.e. `2·D² + 2·D²·(n_kv/H)`; causal mask respected
(perturbing token t+1 onward doesn't move output[:t+1]); gradients flow to all
four projections; uniform input → uniform attention weights → uniform output
per head.

**Tests (variant-specific):**
- `test_mqa::test_kv_broadcast_to_heads` — (B,T,1,D_h) → (B,T,H,D_h) via repeat
- `test_gqa::test_repeat_interleave` — n_kv=4, H=8 → each KV head serves 2 Q heads
- `test_gqa::test_invalid_n_kv_modulo` — H=8, n_kv=3 raises ValueError

### 4.5 TransformerBlock — `model/blocks/transformer_block.py`

**Signature:**
```python
class TransformerBlock(nn.Module):
    config: ModelConfig

    def __call__(self, x: Array) -> Array:
        # (B, T, D) → (B, T, D)
```

**Holds:** `RMSNorm` (norm1), attention (MHA/MQA/GQA per `config.attention`),
`RMSNorm` (norm2), `SwiGLUMLP`.

**Variant selection:** TransformerBlock's `setup()` dispatches on
`config.attention`:
```python
def setup(self):
    self.norm1 = RMSNorm(self.config.d_model)
    self.norm2 = RMSNorm(self.config.d_model)
    if self.config.attention == "mha":
        self.attn = CausalMHA(self.config.d_model, self.config.n_heads,
                              self.config.n_heads, self.config.theta_base)
    elif self.config.attention == "mqa":
        self.attn = CausalMQA(self.config.d_model, self.config.n_heads,
                              1, self.config.theta_base)
    elif self.config.attention == "gqa":
        self.attn = CausalGQA(self.config.d_model, self.config.n_heads,
                              self.config.n_kv_heads, self.config.theta_base)
    else:
        raise ValueError(f"unknown attention: {self.config.attention}")
    self.mlp = SwiGLUMLP(self.config.d_model, self.config.d_ff)
```

**Math (pre-norm, residual):**
```
h   = x + attn(norm1(x))              # norm1 = RMSNorm
out = h + mlp(norm2(h))               # norm2 = RMSNorm
```

**Tests:** shape preserved; param count = 2 norms + 1 attn + 1 mlp; residual
path active (zero attn + zero mlp → block ≈ identity — modulo norms); gradients
flow through both residual paths; pre-norm order verified (set norm1.scale=0,
output == x + mlp(norm2(x))).

### 4.6 LM — `model/lm.py`

**Signature:**
```python
class LM(nn.Module):
    config: ModelConfig

    def __call__(self, input_ids: Array, target_ids: Array,
                 *, return_logits: bool = False) -> Array | tuple[Array, Array]:
        # input_ids : (B, T) int in [0, vocab)
        # target_ids: (B, T) int — shifted input, e.g. input[:, 1:]
        # returns: scalar loss (mean cross-entropy) or (loss, logits)
```

**Params:** `tok_emb : (vocab, D)`. `lm_head` is tied to `tok_emb.T` — no
separate params. RMSNorm scales (per block) + attention/swiglu weights per
block are owned by their modules.

**Math:**
```
x      = tok_emb[input_ids]                  # gather → (B, T, D)
for block in blocks: x = block(x)
x      = final_norm(x)                       # RMSNorm
logits = x @ tok_emb.T                       # tied → (B, T, vocab)
loss   = mean(cross_entropy(logits, target_ids))
```

**Initialization:**
- `tok_emb ~ N(0, 0.02)`
- All weight matrices (`W_q`, `W_k`, `W_v`, `W_o`, `W_gate`, `W_up`, `W_down`)
  `~ N(0, 0.02)`
- `RMSNorm.scale` initialized to `1.0`
- No biases anywhere

**Config field:** `config.attention` ∈ `{mha, mqa, gqa}` — selects which
variant the block uses. Each variant honored; tests exercise all three.

**Tests:** forward returns scalar loss; loss near `log(vocab) ≈ 10.4 ± 0.5` at
init; logits shape `(B, T, V)` when requested; gradients flow to **all**
parameter leaves; param count ≤ `target_params + 5%`; `lm_head == tok_emb.T`
(tied); input_ids ≥ vocab raises ValueError; running with attention variant
from config works; runs at `max_seq_len` declared in YAML.

## 5. Config schema

**Loader:** `model/config.py` exposes
```python
@dataclass(frozen=True)
class ModelConfig:
    name: str
    target_params: int
    architecture: str            # "llama"
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    theta_base: float
    tied_embeddings: bool
    attention: str               # "mha" | "mqa" | "gqa"
    init: InitConfig

def load_model_config(name: str, configs_dir: Path = configs/models) -> ModelConfig: ...
```

**YAML file (e.g. `configs/models/model_25m.yaml`):**
```yaml
name: model_25m
target_params: 25_000_000
architecture: llama
n_layers: 4
d_model: 512
n_heads: 8
n_kv_heads: 4
d_ff: 1280
vocab_size: 32768
max_seq_len: 1024
theta_base: 10000.0
tied_embeddings: true
attention: gqa
init:
  embed_std: 0.02
  hidden_std: 0.02
  norm_scale: 1.0
```

## 6. Model size table

| Hyperparam        | 25M        | 125M       | 350M       |
| ----------------- | ---------- | ---------- | ---------- |
| n_layers (N)      | 4          | 12         | 24         |
| d_model (D)       | 512        | 768        | 1024       |
| n_heads (H)       | 8          | 12         | 16         |
| d_head (D_h)      | 64         | 64         | 64         |
| n_kv_heads        | 4          | 4          | 8          |
| d_ff (I)          | 1280       | 3072       | 2816       |
| theta_base        | 10000      | 10000      | 10000      |
| vocab (V)          | 32768      | 32768      | 32768      |
| max_seq_len        | 1024       | 1024       | 2048       |
| attention default | gqa        | gqa        | gqa        |
| ≈ params          | 27.8M      | 129M       | 316M       |
| token budget      | 1B         | 5B         | 10B        |

Param count formula (Llama-style, tied emb, no biases):
```
emb      : V · D
per_blk  : 2·D² + 2·D²·(n_kv/H) + 3·D·I
total    : V·D + N · per_blk
```
Counts are approximate (±5% from d_ff rounding + tied weights); exact count
computed by `make model-summary NAME=model_25m` after implementation.

## 7. Test conventions

- **RNG:** `jax.random.PRNGKey(seed=0)` for deterministic inits.
- **Dtype:** float32 (no mixed precision in Phase 3).
- **Shape assertions:** fail fast; print actual vs expected on mismatch.
- **Gradient sanity:** `jax.grad(loss_fn)(params)` non-zero for every leaf.
- **Structure:** one `pytest` class per module file; one `assert` per test.
- **Fixtures:** shared RNG + shape fixtures in `model/tests/conftest.py`.
- **Reference integration:** deliberately omitted — tests are
  shape + smoke + gradient sanity, not cross-check against HuggingFace (see
  spec section 1 for rationale).

## 8. Test inventory

### `model/tests/test_rope.py`
- `test_identity_at_position_zero`
- `test_rotation_angle_matches`
- `test_shape_preserved`
- `test_bijection` (forward + inverse = identity)
- `test_consistency_over_v` (V not rotated)

### `model/tests/test_rmsnorm.py`
- `test_zero_input_zero_output`
- `test_output_rms_is_one`
- `test_scale_param_applied`
- `test_shape_preserved`
- `test_param_count` (exactly D)
- `test_gradients_flow_to_scale`
- `test_eps_applied`

### `model/tests/test_swiglu.py`
- `test_shape_preserved`
- `test_param_count` (= 2·D·I + I·D)
- `test_silu_gate_matches_formula`
- `test_gradients_flow`
- `test_d_ff_config`

### `model/tests/test_mha.py`, `test_mqa.py`, `test_gqa.py`
Shared:
- `test_shape_preserved`
- `test_param_count`
- `test_causal_mask_respected`
- `test_gradient_flow_all_projs`
- `test_uniform_input_uniform_out`

Variant-specific:
- `test_mqa::test_kv_broadcast_to_heads`
- `test_gqa::test_repeat_interleave`
- `test_gqa::test_invalid_n_kv_modulo`

### `model/tests/test_transformer_block.py`
- `test_shape_preserved`
- `test_param_count`
- `test_residual_path`
- `test_gradient_flow_to_all_paths`
- `test_pre_norm_order`

### `model/tests/test_lm.py`
- `test_forward_returns_scalar_loss`
- `test_loss_near_neg_log_vocab_at_init`
- `test_logits_shape`
- `test_gradients_flow_to_all_params`
- `test_param_count_smoke`
- `test_tied_embeddings`
- `test_tokens_out_of_range_raise`
- `test_attention_variant_selectable`
- `test_forward_runs_at_specd_seq_len`

### `model/tests/test_smoke.py`
- `test_25m_forward_pass`
- `test_125m_forward_pass`
- `test_350m_forward_pass` (skip on CPU if >30s)
- `test_all_variants_run_on_smallest_model`
- `test_jit_compiles`

## 9. Make targets

```
make model-test       # pytest model/tests/ -v
make model-summary    # uv run python -m model.summary --name model_25m
                      #   prints param count, breakdown per module, config echo
```

## 10. Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    "numpy>=2.0",
    "datasets>=3.0",
    "tqdm>=4.66",
    "jax>=0.4.20",
    "flax>=0.8.0",
]
```

JAX installed without accelerator extras by default (CPU). On the Kaggle TPU
box, install `jax[tpu]` via a separate Kaggle cell — documented in the
notebook, not baked into `pyproject.toml`, so local CPU dev keeps working.

## 11. Non-goals / explicit exclusions

- **Training step** — Phase 4 wraps forward loss in Optax + train loop.
- **Mixed precision / bfloat16** — Phase 4 concern.
- **KV cache** — Phase 6 (`model/attention/kv_cache/` per CLAUDE.md boundary).
- **Position embeddings other than RoPE** — none; RoPE is the only scheme.
- **Reference implementation cross-check** — out of scope for Phase 3 tests.
- **Checkpointing** — Phase 4.
- **Inference shapes** — Phase 6.

## 12. Open questions

- None blocking. All resolved during brainstorming sections 1–4.

## 13. Success criteria

Phase 3 is complete when:

1. `make model-test` passes — every test enumerated in section 8 green.
2. `make model-summary NAME=model_25m` prints a param count within ±5% of
   the target declared in the YAML.
3. The same command runs successfully for `model_125m` and `model_350m`.
4. `model/lm.py` forward returns a finite scalar loss ≈ `log(vocab)` at init
   for all three sizes.
5. Gradients flow to **every** parameter leaf (verified by tests).
6. All three attention variants (MHA, MQA, GQA) run end-to-end inside the LM
   via `config.attention`.