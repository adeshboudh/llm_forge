"""Print a one-shot pre-run summary: config echo + devices + params + ETA.

Usage:
    python -m training.summary --config configs/training/model_25m.yaml
"""

from __future__ import annotations

import argparse
import sys

import jax

from model.config import load_model_config
from model.lm import LM
from training.config import load_training_config
from training.tpu import setup_devices


def summarize(config_path: str) -> None:
    cfg = load_training_config(config_path)
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"config:        {config_path}")
    print(f"model_name:    {cfg.model_name}")

    print("\n-- dataset --")
    print(f"shard_dir:     {cfg.dataset.shard_dir}")
    print(f"seq_len:       {cfg.dataset.seq_len}")
    print(f"val_shard:     {cfg.dataset.val_shard}")

    print("\n-- train --")
    print(f"batch_size:    {cfg.train.batch_size}")
    print(f"total_steps:   {cfg.train.total_steps}")
    print(f"warmup_steps:  {cfg.train.warmup_steps}")
    print(f"weight_decay:  {cfg.train.weight_decay}")
    print(f"grad_clip:     {cfg.train.grad_clip}")
    print(f"grad_accum:    {cfg.train.grad_accum}")

    print("\n-- optim --")
    print(f"lr_peak:       {cfg.optim.lr_peak}")
    print(f"lr_min:        {cfg.optim.lr_min}")
    print(f"b1, b2, eps:   {cfg.optim.b1}, {cfg.optim.b2}, {cfg.optim.eps}")

    print("\n-- ckpt / log --")
    print(f"save_every:    {cfg.ckpt.save_every}")
    print(f"output_dir:    {cfg.ckpt.output_dir}")
    print(f"log_file:      {cfg.log.log_file}")
    print(f"eval_every:    {cfg.log.eval_every}")
    print(f"eval_batches:  {cfg.log.eval_batches}")

    print("\n-- devices --")
    devices = setup_devices()
    print(f"count:         {len(devices)}")
    print(f"platforms:     {[d.platform for d in devices]}")

    print("\n-- model params --")
    model_cfg = load_model_config(cfg.model_name)
    print(f"n_layers:      {model_cfg.n_layers}")
    print(f"d_model:       {model_cfg.d_model}")
    print(f"n_heads:       {model_cfg.n_heads}")
    print(f"n_kv_heads:    {model_cfg.n_kv_heads}")
    print(f"d_ff:          {model_cfg.d_ff}")
    print(f"vocab_size:    {model_cfg.vocab_size}")
    key = jax.random.PRNGKey(0)
    dummy_in = jax.numpy.zeros((1, model_cfg.max_seq_len), dtype=jax.numpy.int32)
    dummy_tgt = jax.numpy.zeros((1, model_cfg.max_seq_len), dtype=jax.numpy.int32)
    params = LM(config=model_cfg).init(key, dummy_in, dummy_tgt)
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"≈ params:      {total_params / 1e6:.2f}M")

    per_step_tok = cfg.train.batch_size * cfg.dataset.seq_len
    total_tok = per_step_tok * cfg.train.total_steps
    print("\n-- ETA (rough) --")
    print(f"per-step tok:  {per_step_tok:,}")
    print(f"total tok:     {total_tok:,}")
    print(f"steps:         {cfg.train.total_steps}")
    print("(multiply by measured tok/s once known)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    summarize(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
