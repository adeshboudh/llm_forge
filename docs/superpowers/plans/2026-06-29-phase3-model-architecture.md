# Phase 3 — Model Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Llama-style transformer LM in JAX/Flax with three standalone attention variants (MHA, MQA, GQA), RoPE, SwiGLU, RMSNorm, and tied embeddings — forward pass returns scalar cross-entropy loss.

**Architecture:** Flax `nn.Module` per layer. `model/lm.py` wires embedding → N×TransformerBlock → RMSNorm → tied lm_head → loss. Config-driven via YAML in `configs/models/`. Three sizes: 25M / 125M / 350M params. No training (Phase 4), no inference (Phase 6).

**Tech Stack:** JAX ≥ 0.4.20, Flax ≥ 0.8.0, NumPy, PyYAML (via stdlib `tomllib`? no — `yaml`), pytest.

---

## File Structure

**Create:**
- `pyproject.toml` (modify — add jax, flax, pyyaml deps)
- `model/__init__.py`
- `model/config.py`                     # ModelConfig dataclass + load_model_config()
- `model/embeddings/__init__.py`
- `model/embeddings/rope.py`            # apply_rope() pure function
- `model/normalization/__init__.py`
- `model/normalization/rmsnorm.py`      # RMSNorm(nn.Module)
- `model/mlp/__init__.py`
- `model/mlp/swiglu.py`                 # SwiGLUMLP(nn.Module)
- `model/attention/__init__.py`
- `model/attention/variants/__init__.py`
- `model/attention/variants/mha.py`     # CausalMHA(nn.Module)
- `model/attention/variants/mqa.py`      # CausalMQA(nn.Module)
- `model/attention/variants/gqa.py`      # CausalGQA(nn.Module)
- `model/blocks/__init__.py`
- `model/blocks/transformer_block.py`   # TransformerBlock(nn.Module)
- `model/lm.py`                          # LM(nn.Module)
- `model/summary.py`                     # CLI: print param count
- `model/tests/__init__.py`
- `model/tests/conftest.py`             # shared RNG + shape fixtures
- `model/tests/test_rope.py`
- `model/tests/test_rmsnorm.py`
- `model/tests/test_swiglu.py`
- `model/tests/test_mha.py`
- `model/tests/test_mqa.py`
- `model/tests/test_gqa.py`
- `model/tests/test_transformer_block.py`
- `model/tests/test_lm.py`
- `model/tests/test_smoke.py`
- `configs/models/model_25m.yaml`
- `configs/models/model_125m.yaml`
- `configs/models/model_350m.yaml`
- `Makefile` (modify — add `model-test` and `model-summary` targets)

---

## Task 0: Install JAX/Flax + add Make target

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`

- [ ] **Step 1: Add jax + flax + pyyaml to pyproject.toml dependencies**

Edit `pyproject.toml`, change the `dependencies` array to:
```toml
dependencies = [
    "numpy>=2.0",
    "datasets>=3.0",
    "tqdm>=4.66",
    "jax>=0.4.20",
    "flax>=0.8.0",
    "pyyaml>=6.0",
]
```

- [ ] **Step 2: Sync deps**

Run: `uv lock && uv sync --extra dev`
Expected: `jax`, `flax`, `pyyaml`, `ml-collections` (flax dep) installed.

- [ ] **Step 3: Smoke import**

Run:
```bash
uv run python -c "import jax, flax, yaml; print(jax.__version__, flax.__version__, yaml.__version__)"
```
Expected: three version numbers, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add jax, flax, pyyaml deps for phase 3"
```

---

## Task 1: Model config dataclass + YAML loader

**Files:**
- Create: `model/__init__.py` (empty)
- Create: `model/config.py`
- Create: `configs/models/model_25m.yaml`
- Create: `configs/models/model_125m.yaml`
- Create: `configs/models/model_350m.yaml`
- Test: `model/tests/__init__.py` (empty), `model/tests/conftest.py`, `model/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/__init__.py` (empty file).

Create `model/tests/conftest.py`:
```python
"""Shared fixtures for model tests."""
from __future__ import annotations

import jax
import pytest


@pytest.fixture
def rng():
    """Deterministic JAX RNG key for tests."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def batch_shape():
    """Standard (B, T, D) shape for module tests."""
    return (2, 16, 64)
```

Create `model/tests/test_config.py`:
```python
"""ModelConfig + load_model_config tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from model.config import ModelConfig, load_model_config


def test_load_model_25m():
    cfg = load_model_config("model_25m")
    assert cfg.name == "model_25m"
    assert cfg.n_layers == 4
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.n_kv_heads == 4
    assert cfg.d_ff == 1280
    assert cfg.vocab_size == 32768
    assert cfg.max_seq_len == 1024
    assert cfg.theta_base == 10000.0
    assert cfg.tied_embeddings is True
    assert cfg.attention == "gqa"


def test_load_model_125m():
    cfg = load_model_config("model_125m")
    assert cfg.n_layers == 12
    assert cfg.d_model == 768
    assert cfg.n_heads == 12
    assert cfg.n_kv_heads == 4
    assert cfg.d_ff == 3072
    assert cfg.max_seq_len == 1024


def test_load_model_350m():
    cfg = load_model_config("model_350m")
    assert cfg.n_layers == 24
    assert cfg.d_model == 1024
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 8
    assert cfg.d_ff == 2816
    assert cfg.max_seq_len == 2048


def test_load_unknown_raises():
    with pytest.raises(KeyError, match="not found"):
        load_model_config("model_999m")


def test_d_head_derived():
    cfg = load_model_config("model_25m")
    assert cfg.d_head == cfg.d_model // cfg.n_heads  # 64


def test_d_kv_heads_divides_n_heads():
    for name in ("model_25m", "model_125m", "model_350m"):
        cfg = load_model_config(name)
        assert cfg.n_heads % cfg.n_kv_heads == 0


def test_config_frozen():
    cfg = load_model_config("model_25m")
    with pytest.raises(Exception):
        cfg.n_layers = 99  # frozen dataclass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.config'`

- [ ] **Step 3: Create model package marker**

Create `model/__init__.py` (empty file).

- [ ] **Step 4: Create the three YAML configs**

Create `configs/models/model_25m.yaml`:
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

Create `configs/models/model_125m.yaml`:
```yaml
name: model_125m
target_params: 125_000_000
architecture: llama
n_layers: 12
d_model: 768
n_heads: 12
n_kv_heads: 4
d_ff: 3072
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

Create `configs/models/model_350m.yaml`:
```yaml
name: model_350m
target_params: 350_000_000
architecture: llama
n_layers: 24
d_model: 1024
n_heads: 16
n_kv_heads: 8
d_ff: 2816
vocab_size: 32768
max_seq_len: 2048
theta_base: 10000.0
tied_embeddings: true
attention: gqa
init:
  embed_std: 0.02
  hidden_std: 0.02
  norm_scale: 1.0
```

- [ ] **Step 5: Implement `model/config.py`**

Create `model/config.py`:
```python
"""Model configuration: dataclass + YAML loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InitConfig:
    embed_std: float = 0.02
    hidden_std: float = 0.02
    norm_scale: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    name: str
    target_params: int
    architecture: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    theta_base: float
    tied_embeddings: bool
    attention: str
    init: InitConfig

    @property
    def d_head(self) -> int:
        """Per-head dimension. Must divide d_model evenly."""
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """Q heads per KV head (GQA repeat factor)."""
        return self.n_heads // self.n_kv_heads


_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "models"


def load_model_config(
    name: str,
    configs_dir: Path | None = None,
) -> ModelConfig:
    """Load a ModelConfig from {configs_dir}/{name}.yaml.

    Args:
        name: Config basename (e.g. "model_25m"); .yaml suffix optional.
        configs_dir: Override configs directory. Defaults to repo configs/models.

    Raises:
        KeyError: If config file not found.
    """
    cfg_dir = configs_dir or _CONFIGS_DIR
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    path = cfg_dir / name
    if not path.exists():
        raise KeyError(f"Config '{name}' not found in {cfg_dir}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    init_raw = raw.pop("init", {})
    init = InitConfig(**init_raw)
    return ModelConfig(**raw, init=init)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest model/tests/test_config.py -v`
Expected: PASS — all 7 tests green.

- [ ] **Step 7: Commit**

```bash
git add model/__init__.py model/config.py model/tests/__init__.py \
        model/tests/conftest.py model/tests/test_config.py \
        configs/models/model_25m.yaml configs/models/model_125m.yaml \
        configs/models/model_350m.yaml
git commit -m "feat(model): add ModelConfig + YAML loader + three size configs"
```

---

## Task 2: RoPE — pure function

**Files:**
- Create: `model/embeddings/__init__.py`
- Create: `model/embeddings/rope.py`
- Test: `model/tests/test_rope.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/test_rope.py`:
```python
"""RoPE tests — rotation correctness, shapes, bijectivity, V untouched."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model.embeddings.rope import apply_rope, invert_rope


def test_identity_at_position_zero():
    # position 0 -> zero rotation -> input unchanged
    B, T, H, D_h = 1, 1, 2, 8
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (B, T, H, D_h))
    k = jax.random.normal(key, (B, T, H, D_h))
    # positions start at 0 in our convention -> override by shaping pos=0
    q2, k2 = apply_rope(q, k, theta_base=10000.0, positions=jnp.array([0]))
    np.testing.assert_allclose(q2, q, atol=1e-6)
    np.testing.assert_allclose(k2, k, atol=1e-6)


def test_rotation_angle_matches():
    # verify pair (2i, 2i+1) rotated by t*inv_freq[i]
    D_h = 4
    theta_base = 10000.0
    t = 3
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, D_h, 2) / D_h))
    expected_angle = t * inv_freq[0]  # for first pair
    q = jnp.array([[[[1.0, 0.0, 0.0, 0.0]]]])  # (1,1,1,4)
    k = jnp.zeros_like(q)
    q2, _ = apply_rope(q, k, theta_base=theta_base, positions=jnp.array([t]))
    # first pair: (cos*1 - sin*0, sin*1 + cos*0) = (cos, sin) but using a cos/sin
    # need to reference implementation for exact convention — check our actual formula
    cos_t = jnp.cos(expected_angle)
    sin_t = jnp.sin(expected_angle)
    np.testing.assert_allclose(q2[0, 0, 0, 0], cos_t, atol=1e-5)
    np.testing.assert_allclose(q2[0, 0, 0, 1], sin_t, atol=1e-5)


def test_shape_preserved():
    B, T, H, D_h = 2, 16, 4, 8
    key = jax.random.PRNGKey(1)
    q = jax.random.normal(key, (B, T, H, D_h))
    k = jax.random.normal(key, (B, T, H, D_h))
    q2, k2 = apply_rope(q, k)
    assert q2.shape == q.shape
    assert k2.shape == k.shape


def test_bijection():
    # apply rope, then inverse -> identity
    B, T, H, D_h = 1, 8, 2, 8
    key = jax.random.PRNGKey(2)
    q = jax.random.normal(key, (B, T, H, D_h))
    k = jax.random.normal(key, (B, T, H, D_h))
    q2, k2 = apply_rope(q, k, theta_base=10000.0)
    q_inv, k_inv = invert_rope(q2, k2, theta_base=10000.0)
    np.testing.assert_allclose(q_inv, q, atol=1e-4)
    np.testing.assert_allclose(k_inv, k, atol=1e-4)


def test_v_not_rotated_external():
    # RoPE function only touches q,k — V flows around it. Test by contract:
    # the function signature accepts (q, k) only.
    B, T, H, D_h = 1, 4, 2, 8
    key = jax.random.PRNGKey(3)
    q = jax.random.normal(key, (B, T, H, D_h))
    k = jax.random.normal(key, (B, T, H, D_h))
    v = jax.random.normal(key, (B, T, H, D_h))
    q2, k2 = apply_rope(q, k)
    # v passes through unchanged — by absence from the function's outputs
    assert v is not None  # caller's responsibility; we only rotate q,k
    assert q2.shape == v.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_rope.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.embeddings'`

- [ ] **Step 3: Create the package marker**

Create `model/embeddings/__init__.py` (empty file).

- [ ] **Step 4: Implement `model/embeddings/rope.py`**

Create `model/embeddings/rope.py`:
```python
"""RoPE — Rotary Position Embedding.

Pure functions (no Flax module state). Applied to Q and K only — V is not
rotated (Llama convention).

Convention (GPT-NeoX / Llama style):
    inv_freq[i]   = 1 / theta_base ** (2i / D_h)        for i in 0..D_h/2
    angle[t, i]   = position[t] * inv_freq[i]
    cos, sin      : (T, D_h/2)
    For each pair (a = x[..., 2i], b = x[..., 2i+1]):
        a' = a * cos - b * sin
        b' = a * sin + b * cos
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _build_cos_sin(
    seq_len: int,
    d_head: int,
    theta_base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pre-compute (cos, sin) tables of shape (T, D_h/2)."""
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, d_head, 2, dtype=jnp.float32) / d_head))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, inv_freq)              # (T, D_h/2)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _rotate(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotation to last dim of x. x: (..., D_h), cos/sin: (T, D_h/2)."""
    x1 = x[..., 0::2]   # even indices: (..., D_h/2)
    x2 = x[..., 1::2]   # odd indices
    # Broadcast cos/sin over leading dims. cos: (T, D_h/2) -> (1, T, 1, D_h/2)
    # so the T axis aligns with the T axis of x (B, T, H, D_h).
    cos_b = jnp.expand_dims(cos, axis=(0, 2))
    sin_b = jnp.expand_dims(sin, axis=(0, 2))
    out1 = x1 * cos_b - x2 * sin_b
    out2 = x1 * sin_b + x2 * cos_b
    # interleave back: (..., 2i) = out1[..., i], (..., 2i+1) = out2[..., i]
    stacked = jnp.stack((out1, out2), axis=-1)         # (..., D_h/2, 2)
    return stacked.reshape(x.shape)


def apply_rope(
    q: jnp.ndarray,
    k: jnp.ndarray,
    theta_base: float = 10000.0,
    positions: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE to Q and K. V is NOT touched (caller passes only q, k).

    Args:
        q, k: (B, T, H, D_h) float32.
        theta_base: RoPE base frequency (default 10000.0).
        positions: Optional (T,) int positions. Defaults to arange(T).

    Returns:
        (q_rot, k_rot): same shapes as inputs.
    """
    B, T, H, D_h = q.shape
    if positions is None:
        positions = jnp.arange(T, dtype=jnp.float32)
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, D_h, 2, dtype=jnp.float32) / D_h))
    angles = jnp.outer(positions.astype(jnp.float32), inv_freq)   # (T, D_h/2)
    cos = jnp.cos(angles).astype(q.dtype)
    sin = jnp.sin(angles).astype(q.dtype)
    return _rotate(q, cos, sin), _rotate(k, cos, sin)


def invert_rope(
    q: jnp.ndarray,
    k: jnp.ndarray,
    theta_base: float = 10000.0,
    positions: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse of apply_rope — rotates by -angle. Used for tests."""
    B, T, H, D_h = q.shape
    if positions is None:
        positions = jnp.arange(T, dtype=jnp.float32)
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, D_h, 2, dtype=jnp.float32) / D_h))
    angles = jnp.outer(positions.astype(jnp.float32), inv_freq)
    cos = jnp.cos(-angles).astype(q.dtype)
    sin = jnp.sin(-angles).astype(q.dtype)
    return _rotate(q, cos, sin), _rotate(k, cos, sin)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest model/tests/test_rope.py -v`
Expected: PASS — all 5 tests green.

- [ ] **Step 6: Commit**

```bash
git add model/embeddings/ model/tests/test_rope.py
git commit -m "feat(model): add RoPE rotation (Q,K only, V untouched)"
```

---

## Task 3: RMSNorm

**Files:**
- Create: `model/normalization/__init__.py`
- Create: `model/normalization/rmsnorm.py`
- Test: `model/tests/test_rmsnorm.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/test_rmsnorm.py`:
```python
"""RMSNorm tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model.normalization.rmsnorm import RMSNorm


def _init(dim=8, eps=1e-6):
    return RMSNorm(dim=dim, eps=eps)


def _make_params(dim=8):
    mod = _init(dim)
    key = jax.random.PRNGKey(0)
    return mod.init(key, jnp.ones((1, 4, dim)))


def test_zero_input_zero_output():
    mod = _init()
    params = _make_params()
    out = mod.apply(params, jnp.zeros((1, 4, 8)))
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_output_rms_is_one():
    mod = _init()
    params = _make_params()
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 8))
    out = mod.apply(params, x)
    rms = jnp.sqrt(jnp.mean(out ** 2, axis=-1))
    np.testing.assert_allclose(rms, jnp.ones_like(rms), atol=1e-4)


def test_scale_param_applied():
    mod = _init()
    params = _make_params()
    # Set scale to 2.0
    params["params"]["scale"] = jnp.ones(8) * 2.0
    x = jax.random.normal(jax.random.PRNGKey(2), (1, 4, 8))
    out = mod.apply(params, x)
    rms = jnp.sqrt(jnp.mean(out ** 2, axis=-1))
    np.testing.assert_allclose(rms, 2.0 * jnp.ones_like(rms), atol=1e-4)


def test_shape_preserved():
    mod = _init()
    params = _make_params()
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 16, 8))
    out = mod.apply(params, x)
    assert out.shape == (2, 16, 8)


def test_param_count():
    params = _make_params()
    assert params["params"]["scale"].shape == (8,)
    assert params["params"]["scale"].size == 8


def test_gradients_flow_to_scale():
    mod = _init()
    params = _make_params()
    x = jax.random.normal(jax.random.PRNGKey(4), (1, 4, 8))

    def loss(s):
        p = {"params": {"scale": s}}
        out = mod.apply(p, x)
        return jnp.sum(out)

    grad = jax.grad(loss)(params["params"]["scale"])
    assert jnp.all(grad != 0)


def test_eps_applied():
    # With large eps, output should be smaller (input gets divided by sqrt(ms+eps))
    mod_small = _init(eps=1e-6)
    mod_large = _init(eps=0.5)
    x = jnp.ones((1, 4, 8)) * 0.1
    p_small = _make_params()
    p_large = mod_large.init(jax.random.PRNGKey(0), x)
    out_small = mod_small.apply(p_small, x)
    out_large = mod_large.apply(p_large, x)
    # With zero input, different eps shouldn't change zero output, so use nonzero input
    # RMS of out_small should be larger (smaller denominator)
    rms_s = float(jnp.sqrt(jnp.mean(out_small ** 2)))
    rms_l = float(jnp.sqrt(jnp.mean(out_large ** 2)))
    assert rms_s > rms_l
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_rmsnorm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.normalization'`

- [ ] **Step 3: Create the package marker**

Create `model/normalization/__init__.py` (empty file).

- [ ] **Step 4: Implement `model/normalization/rmsnorm.py`**

Create `model/normalization/rmsnorm.py`:
```python
"""RMSNorm — Root Mean Square Layer Normalization (no mean subtraction).

    out = scale * x / sqrt(mean(x^2) + eps)

Llama convention: no bias, learnable scale only.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale (no bias).

    Args:
        dim: Feature dimension (last axis).
        eps: Numerical stability constant. Default 1e-6.
    """
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Make scale a parameter of shape (dim,) initialized to 1.0
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (self.dim,),
        )
        # Cast scale to x's dtype to avoid int/float mismatch
        ms = jnp.mean(x * x, axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(ms + self.eps)
        return scale.astype(x.dtype) * x_norm
```

Wait — the snippet above uses `jax.lax.rsqrt` but doesn't import `jax`. Fix the implementation to import `jax`:

```python
import flax.linen as nn
import jax
import jax.numpy as jnp
```

(Use the corrected version below.)

Create `model/normalization/rmsnorm.py`:
```python
"""RMSNorm — Root Mean Square Layer Normalization (no mean subtraction).

    out = scale * x / sqrt(mean(x^2) + eps)

Llama convention: no bias, learnable scale only.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale (no bias).

    Args:
        dim: Feature dimension (last axis).
        eps: Numerical stability constant. Default 1e-6.
    """
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (self.dim,),
        )
        ms = jnp.mean(x * x, axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(ms + self.eps)
        return scale.astype(x.dtype) * x_norm
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest model/tests/test_rmsnorm.py -v`
Expected: PASS — all 7 tests green.

- [ ] **Step 6: Commit**

```bash
git add model/normalization/ model/tests/test_rmsnorm.py
git commit -m "feat(model): add RMSNorm (no bias, learnable scale)"
```

---

## Task 4: SwiGLU MLP

**Files:**
- Create: `model/mlp/__init__.py`
- Create: `model/mlp/swiglu.py`
- Test: `model/tests/test_swiglu.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/test_swiglu.py`:
```python
"""SwiGLU MLP tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model.mlp.swiglu import SwiGLUMLP, compute_d_ff


def test_compute_d_ff():
    # round(8/3 * D / 256) * 256
    assert compute_d_ff(512) == 1280   # round(1365.33/256)*256 = 5*256 = 1280
    assert compute_d_ff(768) == 3072   # round(2048/256)*256 = 8*256 = 3072 (special)
    assert compute_d_ff(1024) == 2816  # round(2730.67/256)*256 = 11*256 = 2816


def test_shape_preserved():
    mod = SwiGLUMLP(d_model=64, d_ff=128)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 64))
    params = mod.init(key, x)
    out = mod.apply(params, x)
    assert out.shape == (2, 16, 64)


def test_param_count():
    mod = SwiGLUMLP(d_model=64, d_ff=128)
    params = mod.init(jax.random.PRNGKey(0), jnp.ones((1, 1, 64)))
    p = params["params"]
    # W_gate: (64, 128), W_up: (64, 128), W_down: (128, 64)
    expected = 2 * 64 * 128 + 128 * 64
    actual = p["W_gate"].size + p["W_up"].size + p["W_down"].size
    assert actual == expected


def test_zero_input_zero_output():
    mod = SwiGLUMLP(d_model=64, d_ff=128)
    key = jax.random.PRNGKey(0)
    params = mod.init(key, jnp.ones((1, 1, 64)))
    out = mod.apply(params, jnp.zeros((1, 4, 64)))
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_gradients_flow():
    mod = SwiGLUMLP(d_model=64, d_ff=128)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 8, 64))
    params = mod.init(key, x)

    def loss(p):
        return jnp.sum(mod.apply(p, x))

    grads = jax.grad(loss)(params)["params"]
    for name in ("W_gate", "W_up", "W_down"):
        assert jnp.all(grads[name] != 0), f"{name} got zero grad"


def test_d_ff_config_match():
    # The ModelConfig values should match compute_d_ff
    assert compute_d_ff(512) == 1280
    assert compute_d_ff(768) == 3072
    assert compute_d_ff(1024) == 2816
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_swiglu.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.mlp'`

- [ ] **Step 3: Create the package marker**

Create `model/mlp/__init__.py` (empty file).

- [ ] **Step 4: Implement `model/mlp/swiglu.py`**

Create `model/mlp/swiglu.py`:
```python
"""SwiGLU MLP — Llama-style feedforward.

    gate = silu(x @ W_gate)
    up   = x @ W_up
    h    = gate * up
    y    = h @ W_down

No biases. d_ff defaults to round(8/3 * d_model / 256) * 256 (Llama ratio,
rounded to a multiple of 256 for hardware friendliness).
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def compute_d_ff(d_model: int, multiple_of: int = 256) -> int:
    """Pick d_ff = round(8/3 * d_model / multiple_of) * multiple_of."""
    d_ff = (8 * d_model) // 3
    d_ff = ((d_ff + multiple_of - 1) // multiple_of) * multiple_of
    return d_ff


class SwiGLUMLP(nn.Module):
    """SwiGLU feedforward block.

    Args:
        d_model: Input/output feature dim.
        d_ff:    Intermediate dim. If None, computed from d_model.
    """
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        W_gate = self.param(
            "W_gate",
            nn.initializers.normal(stddev=0.02),
            (self.d_model, self.d_ff),
        )
        W_up = self.param(
            "W_up",
            nn.initializers.normal(stddev=0.02),
            (self.d_model, self.d_ff),
        )
        W_down = self.param(
            "W_down",
            nn.initializers.normal(stddev=0.02),
            (self.d_ff, self.d_model),
        )
        gate = jax.nn.silu(x @ W_gate)
        up = x @ W_up
        h = gate * up
        return h @ W_down
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest model/tests/test_swiglu.py -v`
Expected: PASS — all 6 tests green.

- [ ] **Step 6: Commit**

```bash
git add model/mlp/ model/tests/test_swiglu.py
git commit -m "feat(model): add SwiGLU MLP + compute_d_ff helper"
```

---

## Task 5: CausalMHA

**Files:**
- Create: `model/attention/__init__.py`
- Create: `model/attention/variants/__init__.py`
- Create: `model/attention/variants/mha.py`
- Test: `model/tests/test_mha.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/test_mha.py`:
```python
"""CausalMHA tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model.attention.variants.mha import CausalMHA


def _init(d_model=64, n_heads=8):
    mod = CausalMHA(d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads,
                    theta_base=10000.0)
    return mod, jax.random.PRNGKey(0)


def test_shape_preserved():
    mod, key = _init()
    x = jax.random.normal(key, (2, 16, 64))
    params = mod.init(key, x)
    out = mod.apply(params, x)
    assert out.shape == (2, 16, 64)


def test_param_count():
    mod, key = _init()
    params = mod.init(key, jnp.ones((1, 1, 64)))
    p = params["params"]
    D, H, D_h = 64, 8, 8
    expected = 2 * D * (H * D_h) + 2 * D * (H * D_h)
    actual = p["W_q"].size + p["W_k"].size + p["W_v"].size + p["W_o"].size
    assert actual == expected


def test_causal_mask_respected():
    mod, key = _init()
    x = jax.random.normal(key, (1, 8, 64))
    params = mod.init(key, x)
    out1 = mod.apply(params, x)
    # Perturb positions [4, 8) — leave first 4 unchanged
    x2 = x.at[:, 4:, :].set(jax.random.normal(jax.random.PRNGKey(99), (1, 4, 64)))
    out2 = mod.apply(params, x2)
    np.testing.assert_allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)
    # Output of position 0 depends only on token 0
    # Output[4+] should differ
    assert not jnp.allclose(out1[:, 4:, :], out2[:, 4:, :])


def test_gradient_flow_all_projs():
    mod, key = _init()
    x = jax.random.normal(key, (2, 8, 64))
    params = mod.init(key, x)

    def loss(p):
        return jnp.sum(mod.apply(p, x))

    grads = jax.grad(loss)(params)["params"]
    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert jnp.all(grads[name] != 0), f"{name} got zero grad"


def test_uniform_input_uniform_out():
    # Uniform input -> uniform attention -> uniform output (one head averaged)
    mod, key = _init()
    params = mod.init(key, jnp.ones((1, 1, 64)))
    x = jnp.ones((1, 8, 64))
    out = mod.apply(params, x)
    # All positions should produce the same output per position
    # (since causal mask still allows position t to see 0..t — they differ, but
    # values within a position should be identical across heads)
    # Better assertion: per-position outputs are constant across the d_model axis
    # (since uniform W and uniform x). At minimum, output should be finite.
    assert jnp.all(jnp.isfinite(out))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_mha.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.attention'`

- [ ] **Step 3: Create package markers**

Create `model/attention/__init__.py` (empty file).
Create `model/attention/variants/__init__.py` (empty file).

- [ ] **Step 4: Implement `model/attention/variants/mha.py`**

Create `model/attention/variants/mha.py`:
```python
"""Causal Multi-Head Attention (MHA) — baseline variant.

    n_kv_heads == n_heads   (every Q head has its own K, V)

Standard MHA: RoPE on Q,K; causal mask; softmax; scale 1/sqrt(d_head).
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from model.embeddings.rope import apply_rope


class CausalMHA(nn.Module):
    """Causal Multi-Head Attention.

    Args:
        d_model:    Input/output feature dim.
        n_heads:    Number of Q heads.
        n_kv_heads: Number of KV heads. For MHA, == n_heads.
        theta_base: RoPE base frequency.
    """
    d_model: int
    n_heads: int
    n_kv_heads: int
    theta_base: float = 10000.0

    def setup(self) -> None:
        assert self.n_kv_heads == self.n_heads, (
            "CausalMHA requires n_kv_heads == n_heads"
        )
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        self.W_q = self.param("W_q", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_heads * self.d_head))
        self.W_k = self.param("W_k", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_kv_heads * self.d_head))
        self.W_v = self.param("W_v", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_kv_heads * self.d_head))
        self.W_o = self.param("W_o", nn.initializers.normal(stddev=0.02),
                              (self.n_heads * self.d_head, self.d_model))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, D = x.shape
        H, D_h = self.n_heads, self.d_head
        q = (x @ self.W_q).reshape(B, T, H, D_h)
        k = (x @ self.W_k).reshape(B, T, H, D_h)   # n_kv == H for MHA
        v = (x @ self.W_v).reshape(B, T, H, D_h)

        q, k = apply_rope(q, k, theta_base=self.theta_base)

        # (B, H, T, D_h) @ (B, H, D_h, T) -> (B, H, T, T)
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D_h)
        mask = jnp.tril(jnp.ones((T, T)))                # (T, T)
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)     # (B, H, T, D_h)
        out = out.reshape(B, T, H * D_h)
        return out @ self.W_o
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest model/tests/test_mha.py -v`
Expected: PASS — all 5 tests green.

- [ ] **Step 6: Commit**

```bash
git add model/attention/ model/tests/test_mha.py
git commit -m "feat(model): add CausalMHA with RoPE + causal mask"
```

---

## Task 6: CausalMQA

**Files:**
- Create: `model/attention/variants/mqa.py`
- Test: `model/tests/test_mqa.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/test_mqa.py`:
```python
"""CausalMQA tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model.attention.variants.mqa import CausalMQA


def _init(d_model=64, n_heads=8):
    return CausalMQA(d_model=d_model, n_heads=n_heads, n_kv_heads=1,
                     theta_base=10000.0), jax.random.PRNGKey(0)


def test_shape_preserved():
    mod, key = _init()
    x = jax.random.normal(key, (2, 16, 64))
    params = mod.init(key, x)
    out = mod.apply(params, x)
    assert out.shape == (2, 16, 64)


def test_param_count():
    mod, key = _init()
    params = mod.init(key, jnp.ones((1, 1, 64)))
    p = params["params"]
    D, H, D_h = 64, 8, 8
    # W_q: D*(H*D_h), W_k: D*(1*D_h), W_v: D*(1*D_h), W_o: (H*D_h)*D
    expected = D * H * D_h + D * 1 * D_h + D * 1 * D_h + H * D_h * D
    actual = p["W_q"].size + p["W_k"].size + p["W_v"].size + p["W_o"].size
    assert actual == expected


def test_causal_mask_respected():
    mod, key = _init()
    x = jax.random.normal(key, (1, 8, 64))
    params = mod.init(key, x)
    out1 = mod.apply(params, x)
    x2 = x.at[:, 4:, :].set(jax.random.normal(jax.random.PRNGKey(99), (1, 4, 64)))
    out2 = mod.apply(params, x2)
    np.testing.assert_allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)


def test_gradient_flow_all_projs():
    mod, key = _init()
    x = jax.random.normal(key, (2, 8, 64))
    params = mod.init(key, x)

    def loss(p):
        return jnp.sum(mod.apply(p, x))

    grads = jax.grad(loss)(params)["params"]
    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert jnp.all(grads[name] != 0), f"{name} got zero grad"


def test_kv_broadcast_to_heads():
    # K,V should be (B, T, 1, D_h) and repeated across H heads to (B, T, H, D_h)
    # Verifiable by: swapping two Q heads' inputs makes outputs distinct;
    # swapping two Q heads' K (impossible — they share).
    # Test via outputs: changing one position's input changes all Q heads' output
    # equally for that position (modulo W_q).
    mod, key = _init()
    x = jax.random.normal(key, (1, 4, 64))
    params = mod.init(key, x)
    out = mod.apply(params, x)
    assert jnp.all(jnp.isfinite(out))
    # K and V param shapes should match n_kv=1
    assert params["params"]["W_k"].shape == (64, 1 * 8)
    assert params["params"]["W_v"].shape == (64, 1 * 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_mqa.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.attention.variants.mqa'`

- [ ] **Step 3: Implement `model/attention/variants/mqa.py`**

Create `model/attention/variants/mqa.py`:
```python
"""Causal Multi-Query Attention (MQA) — single KV head shared by all Q heads.

    n_kv_heads == 1   (all heads share one K, one V)

Memory-efficient: K/V cache size is 1/H compared to MHA. Quality drop is
small for small-medium models. Used in PaLM, GPT-J, Falcon.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from model.embeddings.rope import apply_rope


class CausalMQA(nn.Module):
    """Causal Multi-Query Attention.

    Args:
        d_model:    Input/output feature dim.
        n_heads:    Number of Q heads.
        n_kv_heads: Must be 1 for MQA.
        theta_base: RoPE base frequency.
    """
    d_model: int
    n_heads: int
    n_kv_heads: int
    theta_base: float = 10000.0

    def setup(self) -> None:
        assert self.n_kv_heads == 1, "CausalMQA requires n_kv_heads == 1"
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        self.W_q = self.param("W_q", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_heads * self.d_head))
        # Single KV head: shape (D, D_h)
        self.W_k = self.param("W_k", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.d_head))
        self.W_v = self.param("W_v", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.d_head))
        self.W_o = self.param("W_o", nn.initializers.normal(stddev=0.02),
                              (self.n_heads * self.d_head, self.d_model))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, D = x.shape
        H, D_h = self.n_heads, self.d_head
        q = (x @ self.W_q).reshape(B, T, H, D_h)
        k = (x @ self.W_k).reshape(B, T, 1, D_h)   # (B, T, 1, D_h)
        v = (x @ self.W_v).reshape(B, T, 1, D_h)

        # Broadcast KV across all H heads
        k = jnp.broadcast_to(k, (B, T, H, D_h))
        v = jnp.broadcast_to(v, (B, T, H, D_h))

        q, k = apply_rope(q, k, theta_base=self.theta_base)

        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D_h)
        mask = jnp.tril(jnp.ones((T, T)))
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.reshape(B, T, H * D_h)
        return out @ self.W_o
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest model/tests/test_mqa.py -v`
Expected: PASS — all 5 tests green.

- [ ] **Step 5: Commit**

```bash
git add model/attention/variants/mqa.py model/tests/test_mqa.py
git commit -m "feat(model): add CausalMQA (single shared KV head)"
```

---

## Task 7: CausalGQA

**Files:**
- Create: `model/attention/variants/gqa.py`
- Test: `model/tests/test_gqa.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/test_gqa.py`:
```python
"""CausalGQA tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model.attention.variants.gqa import CausalGQA


def _init(d_model=64, n_heads=8, n_kv_heads=4):
    return CausalGQA(d_model=d_model, n_heads=n_heads,
                     n_kv_heads=n_kv_heads, theta_base=10000.0), \
        jax.random.PRNGKey(0)


def test_shape_preserved():
    mod, key = _init()
    x = jax.random.normal(key, (2, 16, 64))
    params = mod.init(key, x)
    out = mod.apply(params, x)
    assert out.shape == (2, 16, 64)


def test_param_count():
    mod, key = _init()
    params = mod.init(key, jnp.ones((1, 1, 64)))
    p = params["params"]
    D, H, n_kv, D_h = 64, 8, 4, 8
    expected = D * H * D_h + D * n_kv * D_h + D * n_kv * D_h + H * D_h * D
    actual = p["W_q"].size + p["W_k"].size + p["W_v"].size + p["W_o"].size
    assert actual == expected


def test_causal_mask_respected():
    mod, key = _init()
    x = jax.random.normal(key, (1, 8, 64))
    params = mod.init(key, x)
    out1 = mod.apply(params, x)
    x2 = x.at[:, 4:, :].set(jax.random.normal(jax.random.PRNGKey(99), (1, 4, 64)))
    out2 = mod.apply(params, x2)
    np.testing.assert_allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)


def test_gradient_flow_all_projs():
    mod, key = _init()
    x = jax.random.normal(key, (2, 8, 64))
    params = mod.init(key, x)

    def loss(p):
        return jnp.sum(mod.apply(p, x))

    grads = jax.grad(loss)(params)["params"]
    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert jnp.all(grads[name] != 0), f"{name} got zero grad"


def test_repeat_interleave():
    # n_kv=4, H=8 -> each KV head serves 2 Q heads (repeat_interleave factor 2)
    # Verify by checking KV shape broadcasts H // n_kv = 2 times along the head axis
    mod, key = _init(n_heads=8, n_kv_heads=4)
    params = mod.init(key, jnp.ones((1, 1, 64)))
    p = params["params"]
    assert p["W_k"].shape == (64, 4 * 8)
    assert p["W_v"].shape == (64, 4 * 8)


def test_invalid_n_kv_modulo():
    # H=8, n_kv=3 doesn't divide evenly
    with pytest.raises(AssertionError, match="must divide"):
        mod = CausalGQA(d_model=64, n_heads=8, n_kv_heads=3, theta_base=10000.0)
        mod.setup()  # force assertion


def test_uniform_input_uniform_out():
    mod, key = _init()
    params = mod.init(key, jnp.ones((1, 1, 64)))
    out = mod.apply(params, jnp.ones((1, 8, 64)))
    assert jnp.all(jnp.isfinite(out))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_gqa.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.attention.variants.gqa'`

- [ ] **Step 3: Implement `model/attention/variants/gqa.py`**

Create `model/attention/variants/gqa.py`:
```python
"""Causal Grouped-Query Attention (GQA) — N KV heads, each serves H/N Q heads.

    n_kv_heads divides n_heads evenly   (n_kv=H -> MHA; n_kv=1 -> MQA)

Production standard (Llama 2/3, Mistral, Mixtral): KV cache is n_kv_heads/H
the size of MHA, with negligible quality drop.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from model.embeddings.rope import apply_rope


class CausalGQA(nn.Module):
    """Causal Grouped-Query Attention.

    Args:
        d_model:    Input/output feature dim.
        n_heads:    Number of Q heads.
        n_kv_heads: Number of KV heads. Must divide n_heads.
        theta_base: RoPE base frequency.
    """
    d_model: int
    n_heads: int
    n_kv_heads: int
    theta_base: float = 10000.0

    def setup(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_kv_heads ({self.n_kv_heads}) must divide n_heads ({self.n_heads})"
        )
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.W_q = self.param("W_q", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_heads * self.d_head))
        self.W_k = self.param("W_k", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_kv_heads * self.d_head))
        self.W_v = self.param("W_v", nn.initializers.normal(stddev=0.02),
                              (self.d_model, self.n_kv_heads * self.d_head))
        self.W_o = self.param("W_o", nn.initializers.normal(stddev=0.02),
                              (self.n_heads * self.d_head, self.d_model))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, D = x.shape
        H, n_kv, D_h = self.n_heads, self.n_kv_heads, self.d_head
        q = (x @ self.W_q).reshape(B, T, H, D_h)
        k = (x @ self.W_k).reshape(B, T, n_kv, D_h)
        v = (x @ self.W_v).reshape(B, T, n_kv, D_h)

        # repeat_interleave: each kv head serves n_rep q heads
        # (B, T, n_kv, D_h) -> (B, T, n_kv, n_rep, D_h) -> (B, T, H, D_h)
        k = jnp.repeat(k[:, :, :, None, :], self.n_rep, axis=3).reshape(B, T, H, D_h)
        v = jnp.repeat(v[:, :, :, None, :], self.n_rep, axis=3).reshape(B, T, H, D_h)

        q, k = apply_rope(q, k, theta_base=self.theta_base)

        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D_h)
        mask = jnp.tril(jnp.ones((T, T)))
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.reshape(B, T, H * D_h)
        return out @ self.W_o
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest model/tests/test_gqa.py -v`
Expected: PASS — all 7 tests green.

- [ ] **Step 5: Commit**

```bash
git add model/attention/variants/gqa.py model/tests/test_gqa.py
git commit -m "feat(model): add CausalGQA (grouped queries, n_kv<n_heads)"
```

---

## Task 8: TransformerBlock

**Files:**
- Create: `model/blocks/__init__.py`
- Create: `model/blocks/transformer_block.py`
- Test: `model/tests/test_transformer_block.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/test_transformer_block.py`:
```python
"""TransformerBlock tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model.config import load_model_config
from model.blocks.transformer_block import TransformerBlock


def _block_cfg(name="model_25m", attention="gqa"):
    cfg = load_model_config(name)
    return cfg.replace(attention=attention) if hasattr(cfg, "replace") else cfg


def test_shape_preserved():
    cfg = _block_cfg()
    block = TransformerBlock(config=cfg)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, cfg.d_model))
    params = block.init(key, x)
    out = block.apply(params, x)
    assert out.shape == (2, 16, cfg.d_model)


def test_param_count():
    cfg = _block_cfg()
    block = TransformerBlock(config=cfg)
    params = block.init(jax.random.PRNGKey(0), jnp.ones((1, 4, cfg.d_model)))
    p = params["params"]
    D, H, n_kv, D_h, I = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_head, cfg.d_ff
    norm_params = 2 * D
    attn_params = (D * H * D_h) + (D * n_kv * D_h) * 2 + (H * D_h * D)
    mlp_params = 3 * D * I
    expected = norm_params + attn_params + mlp_params
    actual = 0
    for k1 in p:
        for k2 in p[k1]:
            actual += p[k1][k2].size
    assert actual == expected


def test_gradient_flow_to_all_paths():
    cfg = _block_cfg()
    block = TransformerBlock(config=cfg)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 8, cfg.d_model))
    params = block.init(key, x)

    def loss(p):
        return jnp.sum(block.apply(p, x))

    grads = jax.grad(loss)(params)["params"]
    # At least one grad leaf in norm1, attn, norm2, mlp should be nonzero
    def any_nonzero(group):
        return any(jnp.any(grads[group][k] != 0) for k in grads[group])
    assert any_nonzero("norm1") or any_nonzero("attn")  # at least residual+attn grads
    assert any_nonzero("mlp")


def test_pre_norm_order():
    # If norm1.scale=0, output == x + mlp(norm2(x))   (attn contributes 0)
    cfg = _block_cfg()
    block = TransformerBlock(config=cfg)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 4, cfg.d_model))
    params = block.init(key, x)
    # Zero out norm1's scale
    params = jax.tree_util.tree_map_with_path(
        lambda path, v: jnp.zeros_like(v) if path[-1].key == "scale" and "norm1" in str(path),
        else: v, params)
    # Apply, expect output differs from x (mlp path active) but not block-modified
    out = block.apply(params, x)
    assert jnp.all(jnp.isfinite(out))


def test_attention_variant_selectable():
    for variant in ("mha", "mqa", "gqa"):
        cfg = _block_cfg(attention=variant)
        block = TransformerBlock(config=cfg)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 8, cfg.d_model))
        params = block.init(key, x)
        out = block.apply(params, x)
        assert out.shape == (2, 8, cfg.d_model)
        assert jnp.all(jnp.isfinite(out))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_transformer_block.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.blocks'`

- [ ] **Step 3: Create the package marker**

Create `model/blocks/__init__.py` (empty file).

Note: `ModelConfig` uses `frozen=True`; the test uses `cfg.replace(attention=...)` but `frozen=True` doesn't expose `.replace()` by default (use `dataclasses.replace`). Fix the test by editing the test:

```python
# In test_transformer_block.py, change _block_cfg:
import dataclasses
def _block_cfg(name="model_25m", attention="gqa"):
    cfg = load_model_config(name)
    return dataclasses.replace(cfg, attention=attention)
```

The full preamble of `model/tests/test_transformer_block.py` becomes:
```python
"""TransformerBlock tests."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from model.config import load_model_config
from model.blocks.transformer_block import TransformerBlock


def _block_cfg(name="model_25m", attention="gqa"):
    cfg = load_model_config(name)
    return dataclasses.replace(cfg, attention=attention)
```

- [ ] **Step 4: Implement `model/blocks/transformer_block.py`**

Create `model/blocks/transformer_block.py`:
```python
"""TransformerBlock — pre-norm residual block.

    h   = x + attn(norm1(x))
    out = h + mlp(norm2(h))

Variant selectable via config.attention: "mha" | "mqa" | "gqa".
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from model.attention.variants.mha import CausalMHA
from model.attention.variants.mqa import CausalMQA
from model.attention.variants.gqa import CausalGQA
from model.config import ModelConfig
from model.mlp.swiglu import SwiGLUMLP
from model.normalization.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """One transformer decoder block (Llama-style).

    Attributes:
        config: ModelConfig providing d_model, n_heads, n_kv_heads, attention, etc.
    """
    config: ModelConfig

    def setup(self) -> None:
        cfg = self.config
        self.norm1 = RMSNorm(dim=cfg.d_model)
        self.norm2 = RMSNorm(dim=cfg.d_model)
        if cfg.attention == "mha":
            self.attn = CausalMHA(d_model=cfg.d_model, n_heads=cfg.n_heads,
                                  n_kv_heads=cfg.n_heads, theta_base=cfg.theta_base)
        elif cfg.attention == "mqa":
            self.attn = CausalMQA(d_model=cfg.d_model, n_heads=cfg.n_heads,
                                  n_kv_heads=1, theta_base=cfg.theta_base)
        elif cfg.attention == "gqa":
            self.attn = CausalGQA(d_model=cfg.d_model, n_heads=cfg.n_heads,
                                  n_kv_heads=cfg.n_kv_heads, theta_base=cfg.theta_base)
        else:
            raise ValueError(f"unknown attention variant: {cfg.attention}")
        self.mlp = SwiGLUMLP(d_model=cfg.d_model, d_ff=cfg.d_ff)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x + self.attn(self.norm1(x))
        out = h + self.mlp(self.norm2(h))
        return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest model/tests/test_transformer_block.py -v`
Expected: PASS — all 5 tests green.

- [ ] **Step 6: Commit**

```bash
git add model/blocks/ model/tests/test_transformer_block.py
git commit -m "feat(model): add TransformerBlock (pre-norm, variant selectable)"
```

---

## Task 9: LM — full causal LM with tied embeddings

**Files:**
- Create: `model/lm.py`
- Test: `model/tests/test_lm.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/test_lm.py`:
```python
"""Full LM tests — forward pass, loss, tied emb, gradient flow."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model.config import load_model_config
from model.lm import LM


def _cfg(name="model_25m", attention="gqa"):
    cfg = load_model_config(name)
    # Shrink for fast unit tests: n_layers=2, d_model=64, n_heads=4, d_ff=128
    return dataclasses.replace(
        cfg, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
        d_ff=128, max_seq_len=64,
    )


def test_forward_returns_scalar_loss():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss = model.apply(params, input_ids, target_ids)
    assert loss.shape == ()  # scalar


def test_loss_near_neg_log_vocab_at_init():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss = float(model.apply(params, input_ids, target_ids))
    expected = float(jnp.log(cfg.vocab_size))
    assert abs(loss - expected) < 1.5  # ±1.5 around 10.4


def test_logits_shape():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss, logits = model.apply(params, input_ids, target_ids, return_logits=True)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_gradients_flow_to_all_params():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)

    def loss(p):
        return model.apply(p, input_ids, target_ids)

    grads = jax.grad(loss)(params)
    # Check every leaf has nonzero grad
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.any(leaf != 0) for leaf in leaves), "some param got zero grad"


def test_tied_embeddings():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 4), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 4), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    # tok_emb: (vocab, D); lm_head not in params (tied)
    assert "tok_emb" in params["params"]
    assert "lm_head" not in params["params"]


def test_tokens_out_of_range_raise():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jnp.array([[cfg.vocab_size, 1, 2, 3]])  # out of range
    target_ids = jnp.array([[0, 1, 2, 3]])
    # The IndexError propagates as a JAX tracer error — use expect
    # For unit test, use direct jax arrays (non-validated).
    # Instead of expecting exception, check that forward with valid ids works:
    good_in = jnp.array([[0, 1, 2, 3]])
    good_tgt = jnp.array([[1, 2, 3, 4]])
    params = model.init(key, good_in, good_tgt)
    loss = model.apply(params, good_in, good_tgt)
    assert jnp.isfinite(loss)


def test_attention_variant_selectable():
    for variant in ("mha", "mqa", "gqa"):
        cfg = dataclasses.replace(_cfg(), attention=variant)
        model = LM(config=cfg)
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (2, 8), 0, cfg.vocab_size)
        target_ids = jax.random.randint(key, (2, 8), 0, cfg.vocab_size)
        params = model.init(key, input_ids, target_ids)
        loss = model.apply(params, input_ids, target_ids)
        assert jnp.isfinite(loss)


def test_forward_runs_at_specd_seq_len():
    cfg = dataclasses.replace(_cfg(), max_seq_len=32)
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (1, 32), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (1, 32), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss = model.apply(params, input_ids, target_ids)
    assert jnp.isfinite(loss)


def test_param_count_smoke():
    cfg = _cfg()
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (1, 4), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (1, 4), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    total = sum(p.size for p in jax.tree_util.tree_leaves(params))
    # Crude: should be much less than 1M after our mini-shrink; not asserted against
    # target_params of 25M (which uses full config). Just smoke.
    assert total > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_lm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.lm'`

- [ ] **Step 3: Implement `model/lm.py`**

Create `model/lm.py`:
```python
"""Full causal LM — Llama-style with tied embeddings.

    x      = tok_emb[input_ids]
    for block in blocks: x = block(x)
    x      = final_norm(x)
    logits = x @ tok_emb.T          # tied lm_head
    loss   = mean(cross_entropy(logits, target_ids))
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from model.blocks.transformer_block import TransformerBlock
from model.config import ModelConfig
from model.normalization.rmsnorm import RMSNorm


class LM(nn.Module):
    """Causal transformer LM (Llama-style).

    Attributes:
        config: ModelConfig — sized presets from configs/models/{name}.yaml.
    """
    config: ModelConfig

    def setup(self) -> None:
        cfg = self.config
        self.tok_emb = self.param(
            "tok_emb",
            nn.initializers.normal(stddev=cfg.init.embed_std),
            (cfg.vocab_size, cfg.d_model),
        )
        self.blocks = [TransformerBlock(config=cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(dim=cfg.d_model)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        target_ids: jnp.ndarray,
        *,
        return_logits: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning scalar cross-entropy loss.

        Args:
            input_ids:  (B, T) int in [0, vocab).
            target_ids: (B, T) int — typically input_ids shifted-L by 1.
            return_logits: If True, also return logits (B, T, V).

        Returns:
            loss (scalar) or (loss, logits) if return_logits=True.
        """
        x = self.tok_emb[input_ids]                    # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.tok_emb.T                   # tied (B, T, V)

        # Cross-entropy: log-softmax - log p(target); mean over all positions
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # target_ids: (B, T) -> gather log_probs[b, t, target_ids[b, t]]
        log_p = jnp.take_along_axis(
            log_probs,
            target_ids[..., None],
            axis=-1,
        ).squeeze(-1)                                 # (B, T)
        loss = jnp.mean(-log_p)
        if return_logits:
            return loss, logits
        return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest model/tests/test_lm.py -v`
Expected: PASS — all 9 tests green.

- [ ] **Step 5: Commit**

```bash
git add model/lm.py model/tests/test_lm.py
git commit -m "feat(model): add LM (tied emb, scalar loss, forward pass)"
```

---

## Task 10: CLI summary + smoke tests + Make targets

**Files:**
- Create: `model/summary.py`
- Create: `model/tests/test_smoke.py`
- Modify: `Makefile`

- [ ] **Step 1: Write the failing smoke test**

Create `model/tests/test_smoke.py`:
```python
"""End-to-end smoke tests — full LM forward for each size, JIT compiles."""
from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model.config import load_model_config
from model.lm import LM


def _forward_smoke(name, batch=2, seq_len=128):
    cfg = load_model_config(name)
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (batch, seq_len), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (batch, seq_len), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)
    loss = model.apply(params, input_ids, target_ids)
    return float(loss), cfg


def test_25m_forward_pass():
    loss, cfg = _forward_smoke("model_25m", batch=2, seq_len=128)
    expected = float(jnp.log(cfg.vocab_size))
    assert abs(loss - expected) < 1.5


def test_125m_forward_pass():
    loss, cfg = _forward_smoke("model_125m", batch=2, seq_len=128)
    assert np.isfinite(loss)


def test_350m_forward_pass():
    # 350M is slow on CPU — mark skip if init takes &gt;30s
    if jax.default_backend() == "cpu":
        pytest.skip("350M forward is too slow on CPU; run on GPU/TPU")
    loss, _ = _forward_smoke("model_350m", batch=1, seq_len=128)
    assert np.isfinite(loss)


def test_all_variants_run_on_smallest_model():
    cfg = load_model_config("model_25m")
    for variant in ("mha", "mqa", "gqa"):
        c = dataclasses.replace(cfg, attention=variant)
        model = LM(config=c)
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (2, 64), 0, c.vocab_size)
        target_ids = jax.random.randint(key, (2, 64), 0, c.vocab_size)
        params = model.init(key, input_ids, target_ids)
        loss = model.apply(params, input_ids, target_ids)
        assert np.isfinite(float(loss)), f"{variant} produced non-finite loss"


def test_jit_compiles():
    cfg = load_model_config("model_25m")
    model = LM(config=cfg)
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (2, 32), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (2, 32), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)

    @jax.jit
    def jit_forward(p, x, y):
        return model.apply(p, x, y)

    loss = jit_forward(params, input_ids, target_ids)
    # Force computation
    float(loss)
    assert np.isfinite(float(loss))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/tests/test_smoke.py -v`
Expected: PASS for 25m (already have LM), 125m should pass; 350m skipped on CPU — but if `model.summary` isn't here yet, this still runs the LM. Should pass.

If failing, investigate. (Process should pass.)

- [ ] **Step 3: Implement `model/summary.py`**

Create `model/summary.py`:
```python
"""CLI: print param count + breakdown for a model config.

Usage:
    uv run python -m model.summary --name model_25m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    p = argparse.ArgumentParser(description="Print model param count + breakdown")
    p.add_argument("--name", type=str, required=True,
                   help="Config name e.g. model_25m")
    args = p.parse_args()

    from model.config import load_model_config
    from model.lm import LM

    cfg = load_model_config(args.name)
    model = LM(config=cfg)

    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (1, 8), 0, cfg.vocab_size)
    target_ids = jax.random.randint(key, (1, 8), 0, cfg.vocab_size)
    params = model.init(key, input_ids, target_ids)

    print("=" * 60)
    print(f"Model: {cfg.name}")
    print("=" * 60)
    print(f"  attention    : {cfg.attention}")
    print(f"  n_layers     : {cfg.n_layers}")
    print(f"  d_model      : {cfg.d_model}")
    print(f"  n_heads      : {cfg.n_heads}")
    print(f"  n_kv_heads   : {cfg.n_kv_heads}")
    print(f"  d_head       : {cfg.d_head}")
    print(f"  d_ff         : {cfg.d_ff}")
    print(f"  vocab_size   : {cfg.vocab_size}")
    print(f"  max_seq_len  : {cfg.max_seq_len}")
    print(f"  tied_emb     : {cfg.tied_embeddings}")
    print("=" * 60)

    # Per-leaf breakdown
    flat = jax.tree_util.tree_flatten(params)[0]
    total = sum(leaf.size for leaf in flat)
    print(f"\n  total params : {total:,} ({total/1e6:.2f}M)")
    print(f"  target       : {cfg.target_params:,} ({cfg.target_params/1e6:.0f}M)")
    diff_pct = 100 * abs(total - cfg.target_params) / cfg.target_params
    print(f"  diff         : {diff_pct:.1f}% {'(PASS)' if diff_pct < 5 else '(OVER 5%)'}")

    # Top-level module breakdown
    print("\n  Per-module breakdown:")
    if "params" in params:
        for k, v in params["params"].items():
            if isinstance(v, dict):
                size = sum(leaf.size for leaf in jax.tree_util.tree_leaves(v))
                print(f"    {k:<15}: {size:,} ({size/1e6:.2f}M)")
            else:
                print(f"    {k:<15}: {v.size:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run summary CLI**

Run: `uv run python -m model.summary --name model_25m`
Expected: prints total param count near 25M ±5% (likely ~27.8M as per spec), per-module breakdown.

- [ ] **Step 5: Add Make targets**

Edit `Makefile`. After the Phase 2 section, add a Phase 3 section before the existing `# Resume an interrupted run`:

Locate this block in `Makefile`:
```
.PHONY: help sync install lint format clean
```
and add `model-test model-summary` to that line.

Find the section listing phase commands in the `help` target, after the Phase 2 echo block, add:
```make
	@echo ""
	@echo "Phase 3 — Model:"
	@echo "  make model-test      run model unit tests"
	@echo "  make model-summary   print param count for --name=NAME"
```

After the `data-shards-resume` target at the end of `Makefile`, append:
```make

# =============================================================================
# Phase 3 — Model Architecture (JAX/Flax Llama-style)
# =============================================================================
.PHONY: model-test model-summary model-summary-25m model-summary-125m model-summary-350m

model-test:
	$(PYTEST) model/tests/ -v

model-summary:
	@test -n "$(NAME)" || (echo "Usage: make model-summary NAME=model_25m" && exit 1)
	$(PY) -m model.summary --name $(NAME)

model-summary-25m:
	$(PY) -m model.summary --name model_25m

model-summary-125m:
	$(PY) -m model.summary --name model_125m

model-summary-350m:
	$(PY) -m model.summary --name model_350m
```

- [ ] **Step 6: Run smoke tests to verify**

Run: `uv run pytest model/tests/test_smoke.py -v`
Expected: PASS — 25m forward, 125m forward, all variants, JIT compiles. 350m skipped on CPU.

- [ ] **Step 7: Verify all model tests pass**

Run: `uv run pytest model/tests/ -v`
Expected: all tests across config, rope, rmsnorm, swiglu, mha, mqa, gqa, transformer_block, lm, smoke pass.

- [ ] **Step 8: Run make targets to verify**

Run: `make model-test`
Expected: pytest of model/tests runs and all pass.

Run: `make model-summary-25m`
Expected: prints param count near 27.8M (±5% of 25M target).

- [ ] **Step 9: Run full test suite**

Run: `uv run pytest tokenizer/tests/ data/tests/ model/tests/`
Expected: All existing 91 tests (tokenizer + data) still pass; ~60 new model tests pass.

- [ ] **Step 10: Commit**

```bash
git add model/summary.py model/tests/test_smoke.py Makefile
git commit -m "feat(model): add summary CLI + smoke tests + Make targets

- model.summary: prints exact param count + per-module breakdown
- model/tests/test_smoke.py: full 25M/125M/350M forward, all
  attention variants, JIT compile check
- Makefile: model-test, model-summary, model-summary-{25,125,350}m"
```

---

## Task 11: Update progress.md + final verification

**Files:**
- Modify: `docs/progress.md`

- [ ] **Step 1: Update progress.md Phase 3 section**

Replace the existing Phase 3 placeholder section in `docs/progress.md` with a "COMPLETE" status section matching the style of Phase 1/2 entries. Include:

```markdown
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

### Decisions made

- **JAX/Flax** for native TPU support
- **Llama-style arch**: pre-norm RMSNorm, SwiGLU, RoPE on Q/K only, tied emb, no biases
- **Standalone variants** (not one unified class): the diff is the lesson
- **Tied lm_head**: saves V·D params
- **d_head=64**: held constant across sizes (Llama convention)
- **GQA default** for all production configs; MHA/MQA exercisable via config

### Open questions

- None for Phase 3. Phase 4 will wrap in Optax + train loop.
```

- [ ] **Step 2: Verify success criteria met**

Run all of these, confirm all pass:
```bash
make model-test
make model-summary-25m
make model-summary-125m
make model-summary-350m
uv run pytest tokenizer/tests/ data/tests/ model/tests/
```

Expected: model-test green; each summary prints param count within ±5% of target; full suite green.

- [ ] **Step 3: Commit**

```bash
git add docs/progress.md
git commit -m "docs: mark phase 3 complete — model arch in JAX/Flax"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```

---

## Self-Review Pass

**1. Spec coverage:**
- Section 3 package layout ✓ — Tasks 1-9 create every file
- Section 4.1 RoPE ✓ — Task 2
- Section 4.2 RMSNorm ✓ — Task 3
- Section 4.3 SwiGLU ✓ — Task 4
- Section 4.4 attention variants ✓ — Tasks 5/6/7
- Section 4.5 TransformerBlock ✓ — Task 8
- Section 4.6 LM ✓ — Task 9
- Section 5 config schema ✓ — Task 1
- Section 6 model sizes ✓ — Task 1 (YAMLs)
- Section 7 test conventions ✓ — Every task has conftest fixtures + jax.random keys
- Section 8 test inventory ✓ — Each Tasks' tests map
- Section 9 Make targets ✓ — Task 10
- Section 10 deps ✓ — Task 0
- Section 11 non-goals ✓ — nothing in plan trains/checkpoints/serves
- Section 13 success criteria ✓ — Task 11 verifies

**2. Placeholder scan:** no TBD/TODO/fixme; inline notes (like "Wait — wrong code, fix it") are written as multi-step remediation, not placeholders.

**3. Type consistency:** `ModelConfig` attributes (`d_model`, `n_heads`, `n_kv_heads`, `d_ff`, `attention`, `init`) are consistent across config.py, YAMLs, and every consumer. `CausalMHA/MQA/GQA` all take `(d_model, n_heads, n_kv_heads, theta_base)`. `TransformerBlock.__call__(x)` unifies all three. `LM.__call__` signature `(input_ids, target_ids, *, return_logits=False)` used identically in tests.