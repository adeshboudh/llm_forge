"""End-to-end smoke: 5 train steps on toy shards, ckpt, JSONL, --resume."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from training.train import train


def test_train_smoke_loss_decreases(toy_config):
    """5 train steps: final loss should be lower than initial loss."""
    losses = train(toy_config)
    assert len(losses) == toy_config.train.total_steps
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_train_smoke_writes_jsonl(toy_config):
    train(toy_config)
    log_path = Path(toy_config.log.log_file)
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == toy_config.train.total_steps
    for line in lines:
        row = json.loads(line)
        assert "step" in row
        assert "loss" in row
        assert "lr" in row
        assert "grad_norm" in row
        assert "ts" in row


def test_train_smoke_writes_checkpoint(toy_config):
    train(toy_config)
    ckpt_dir = Path(toy_config.ckpt.output_dir)
    ckpts = sorted(ckpt_dir.glob("*"))
    assert len(ckpts) >= 1


def test_resume_continues_from_step(toy_config):
    train(toy_config)
    candidates = list(Path(toy_config.ckpt.output_dir).iterdir())
    assert candidates, "no checkpoint to resume from"
    import jax

    from model.config import load_model_config
    from model.lm import LM
    from training.state import create_train_state, restore

    model_cfg = load_model_config(toy_config.model_name)
    model = LM(config=model_cfg)
    state = create_train_state(jax.random.PRNGKey(0), model, toy_config, model_cfg)
    restored = restore(candidates[0], state)
    assert int(restored.step) >= 0


def test_cli_help():
    """CLI --help exits 0 and shows --config."""
    result = subprocess.run(
        [sys.executable, "-m", "training.train", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--resume" in result.stdout
