"""Training CLI + train loop + JSONL logger + emergency save.

Usage:
    python -m training.train --config configs/training/model_25m.yaml
    python -m training.train --config configs/training/smoke_test.yaml --smoke
    python -m training.train --config ... --max-steps 100
    python -m training.train --config ... --resume /kaggle/working/ckpt/<step>/
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from data.loaders.jax_batcher import JAXBatcher
from model.config import load_model_config
from model.lm import LM
from training.config import TrainConfig, load_training_config
from training.state import create_train_state, restore, save
from training.tpu import setup_devices
from training.train_step import eval_step, set_model, train_step


def train(config: TrainConfig, resume_path: Path | None = None) -> list[float]:
    """Run the training loop. Returns a list of per-step losses.

    Steps:
      - Load ModelConfig via model_name
      - Initialize / restore TrainState
      - Pre-allocate the JAXBatcher (train + val streams)
      - For each step: pull a batch, step optimizer, log JSONL, save ckpt, eval.
    """
    devices = setup_devices()
    print(f"devices: {len(devices)} (types: {[d.platform for d in devices]})")

    model_cfg = load_model_config(config.model_name)
    model = LM(config=model_cfg)
    set_model(model)

    key = jax.random.PRNGKey(0)
    if resume_path is not None:
        state = create_train_state(key, model, config, model_cfg)
        state = restore(resume_path, state)
        print(f"restored from {resume_path} at step {int(state.step)}")
    else:
        state = create_train_state(key, model, config, model_cfg)
        print(f"initialized fresh state at step {int(state.step)}")

    batcher = JAXBatcher(
        shard_dir=config.dataset.shard_dir,
        seq_len=config.dataset.seq_len,
        batch_size=config.train.batch_size,
        val_shard=config.dataset.val_shard,
    )
    if resume_path is not None:
        skip = int(state.step) * config.train.batch_size * config.dataset.seq_len
        batcher.skip_tokens(skip)
        print(f"skipping {skip} tokens on train stream (resume)")

    log_path = Path(config.log.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")

    losses: list[float] = []
    total_steps = config.train.total_steps
    train_iter = batcher.train_iter()

    try:
        with tqdm(total=total_steps, desc="train", unit="step") as pbar:
            for step in range(int(state.step), total_steps):
                t0 = time.time()
                try:
                    input_ids_np, target_ids_np = next(train_iter)
                except StopIteration:
                    print(f"train stream exhausted at step {step}; wrapping")
                    train_iter = batcher.train_iter()
                    input_ids_np, target_ids_np = next(train_iter)
                input_ids = jnp.asarray(input_ids_np, dtype=jnp.int32)
                target_ids = jnp.asarray(target_ids_np, dtype=jnp.int32)

                state, loss, metrics = train_step(state, input_ids, target_ids)
                loss_f = float(loss)
                losses.append(loss_f)
                dt_ms = (time.time() - t0) * 1000
                tok_per_s = int(
                    config.train.batch_size * config.dataset.seq_len / max(dt_ms / 1000, 1e-9)
                )

                lr_at = _lr_at(config, step)

                val_loss = None
                if step > 0 and step % config.log.eval_every == 0:
                    val_loss = _eval_loss(state, batcher, config)

                row = {
                    "step": step,
                    "loss": loss_f,
                    "val_loss": val_loss,
                    "lr": lr_at,
                    "grad_norm": float(metrics["grad_norm"]),
                    "tok/s": tok_per_s,
                    "ts": dt.datetime.now(dt.UTC).isoformat(),
                }
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()

                pbar.set_postfix(loss=f"{loss_f:.4f}", lr=f"{lr_at:.2e}", tok_per_s=tok_per_s)
                pbar.update(1)

                if not jnp.isfinite(loss):
                    save(state, Path(config.ckpt.output_dir) / f"emergency_{step:010d}")
                    log_file.write(
                        json.dumps({"event": "nan", "step": step, "loss": loss_f}) + "\n"
                    )
                    log_file.flush()
                    raise RuntimeError(f"non-finite loss at step {step}: {loss_f}")

                if step > 0 and step % config.ckpt.save_every == 0:
                    save(state, Path(config.ckpt.output_dir) / f"step_{step:010d}")

            save(state, Path(config.ckpt.output_dir) / f"step_{total_steps:010d}_final")
    except RuntimeError as e:
        save(state, Path(config.ckpt.output_dir) / f"emergency_{int(state.step):010d}")
        log_file.write(
            json.dumps({"event": "preemption", "step": int(state.step), "err": str(e)}) + "\n"
        )
        log_file.flush()
        print(f"emergency checkpoint saved at step {int(state.step)}: {e}", file=sys.stderr)
    finally:
        log_file.close()

    return losses


def _lr_at(config: TrainConfig, step: int) -> float:
    import optax

    warmup_steps = min(config.train.warmup_steps, max(1, config.train.total_steps // 2))

    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optim.lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=config.train.total_steps,
        end_value=config.optim.lr_min,
    )
    return float(sched(step))


def _eval_loss(state, batcher: JAXBatcher, config: TrainConfig) -> float:
    val_iter = batcher.val_iter()
    total, n = 0.0, 0
    for input_ids_np, target_ids_np in val_iter:
        if n >= config.log.eval_batches:
            break
        loss = eval_step(
            state,
            jnp.asarray(input_ids_np, dtype=jnp.int32),
            jnp.asarray(target_ids_np, dtype=jnp.int32),
        )
        total += float(loss)
        n += 1
    return total / max(n, 1)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="training.train", description="llm_forge pretrainer")
    p.add_argument("--config", required=True, help="Path to training YAML")
    p.add_argument("--resume", default=None, help="Path to orbax checkpoint dir to restore from")
    p.add_argument("--smoke", action="store_true", help="Run on toy shards (overwrites shard_dir)")
    p.add_argument("--max-steps", type=int, default=None, help="Override total_steps")
    return p.parse_args(argv)


def _bootstrap_smoke_shards(shard_dir: Path) -> None:
    """Write 4 toy shards + metadata so --smoke works without pytest."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(4):
        tokens = rng.integers(0, 32768, size=10_000, dtype=np.uint16)
        np.save(shard_dir / f"shard_{i:05d}.npy", tokens)
    meta = {
        "dataset_version": "smoke",
        "vocab_size": 32768,
        "total_tokens": 40_000,
        "total_shards": 4,
        "shard_size": 10_000,
        "shards": [],
    }
    (shard_dir / "metadata.json").write_text(json.dumps(meta))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_training_config(args.config)
    if args.max_steps is not None:
        cfg = dataclasses.replace(
            cfg, train=dataclasses.replace(cfg.train, total_steps=args.max_steps)
        )
    if args.smoke:
        _bootstrap_smoke_shards(Path(cfg.dataset.shard_dir))
    resume_path = Path(args.resume) if args.resume else None
    train(cfg, resume_path=resume_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
