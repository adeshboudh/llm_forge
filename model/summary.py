"""CLI: print param count + breakdown for a model config.

Usage:
    uv run python -m model.summary --name model_25m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    p = argparse.ArgumentParser(description="Print model param count + breakdown")
    p.add_argument("--name", type=str, required=True, help="Config name e.g. model_25m")
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
    print(f"\n  total params : {total:,} ({total / 1e6:.2f}M)")
    print(f"  target       : {cfg.target_params:,} ({cfg.target_params / 1e6:.0f}M)")
    diff_pct = 100 * abs(total - cfg.target_params) / cfg.target_params
    print(f"  diff         : {diff_pct:.1f}% {'(PASS)' if diff_pct < 5 else '(OVER 5%)'}")

    # Top-level module breakdown
    print("\n  Per-module breakdown:")
    if "params" in params:
        for k, v in params["params"].items():
            if isinstance(v, dict):
                size = sum(leaf.size for leaf in jax.tree_util.tree_leaves(v))
                print(f"    {k:<15}: {size:,} ({size / 1e6:.2f}M)")
            else:
                print(f"    {k:<15}: {v.size:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
