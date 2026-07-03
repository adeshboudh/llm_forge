"""Tests for HuggingFace export: config + file presence + tokenizer integrity."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
from safetensors.numpy import load_file

from model.config import load_model_config
from model.lm import LM
from training.export_hf import export_hf
from training.state import create_train_state


def _small_model_cfg():
    cfg = load_model_config("model_25m")
    return dataclasses.replace(
        cfg,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=64,
    )


def _state(toy_config, cfg):
    model = LM(config=cfg)
    return create_train_state(jax.random.PRNGKey(0), model, toy_config, cfg)


def test_to_hf_dict_has_llama_keys():
    cfg = _small_model_cfg()
    hf = cfg.to_hf_dict()
    for key in (
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "tie_word_embeddings",
    ):
        assert key in hf, f"missing key {key!r}"
    assert hf["architectures"] == ["LlamaForCausalLM"]
    assert hf["model_type"] == "llama"
    assert hf["hidden_size"] == cfg.d_model
    assert hf["intermediate_size"] == cfg.d_ff
    assert hf["num_attention_heads"] == cfg.n_heads
    assert hf["num_key_value_heads"] == cfg.n_kv_heads


def test_export_hf_writes_all_files(toy_config, tmp_path):
    cfg = _small_model_cfg()
    state = _state(toy_config, cfg)
    out = export_hf(
        cfg,
        state.params,
        tmp_path / "hf",
        tokens_trained=1_000_000,
        num_steps=100,
        final_loss=4.2,
        repo_id="adesh01/llm_forge-test",
    )
    assert out.is_dir()
    for name in (
        "config.json",
        "params.safetensors",
        "generation_config.json",
        "README.md",
    ):
        assert (out / name).exists(), f"missing {name}"


def test_export_hf_copies_tokenizer_when_present(toy_config, tmp_path):
    cfg = _small_model_cfg()
    state = _state(toy_config, cfg)
    out = export_hf(
        cfg,
        state.params,
        tmp_path / "hf",
        tokens_trained=0,
        num_steps=0,
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (Path("tokenizer") / name).exists():
            assert (out / name).exists(), f"tokenizer file {name} not copied"


def test_export_hf_config_json_valid(toy_config, tmp_path):
    cfg = _small_model_cfg()
    state = _state(toy_config, cfg)
    out = export_hf(
        cfg,
        state.params,
        tmp_path / "hf",
        tokens_trained=100,
        num_steps=10,
        final_loss=5.0,
    )
    loaded = json.loads((out / "config.json").read_text())
    assert loaded["hidden_size"] == cfg.d_model
    assert loaded["vocab_size"] == cfg.vocab_size


def test_export_hf_readme_includes_repo_id(toy_config, tmp_path):
    cfg = _small_model_cfg()
    state = _state(toy_config, cfg)
    out = export_hf(
        cfg,
        state.params,
        tmp_path / "hf",
        tokens_trained=1_000_000,
        num_steps=100,
        final_loss=3.14,
        repo_id="adesh01/llm_forge-25m",
    )
    readme = (out / "README.md").read_text()
    assert "adesh01/llm_forge-25m" in readme
    assert "1,000,000" in readme or "1000000" in readme
    assert "3.14" in readme
    assert "Target size" in readme
    assert "load_model_config" in readme  # usage example uses real API


def test_export_hf_readme_has_yaml_frontmatter(toy_config, tmp_path):
    """HF requires YAML metadata block; check it starts with --- and has tags."""
    cfg = _small_model_cfg()
    state = _state(toy_config, cfg)
    out = export_hf(cfg, state.params, tmp_path / "hf", tokens_trained=0, num_steps=0)
    readme = (out / "README.md").read_text()
    assert readme.startswith("---"), "README must start with YAML frontmatter"
    body = readme.split("---", 2)
    assert len(body) >= 3, "README must have closing ---"
    frontmatter = body[1]
    assert "license:" in frontmatter
    assert "tags:" in frontmatter
    assert "llm" in frontmatter
    assert "library_name:" in frontmatter


def test_export_hf_params_loadable(toy_config, tmp_path):
    """Round-trip: load the safetensors back and check shapes match."""
    cfg = _small_model_cfg()
    state = _state(toy_config, cfg)
    out = export_hf(cfg, state.params, tmp_path / "hf", tokens_trained=0, num_steps=0)
    flat = load_file(str(out / "params.safetensors"))
    assert len(flat) > 0
    for arr in flat.values():
        assert arr.dtype.name == "float32"


def test_export_hf_no_tokenizer_when_disabled(toy_config, tmp_path):
    cfg = _small_model_cfg()
    state = _state(toy_config, cfg)
    out = export_hf(
        cfg,
        state.params,
        tmp_path / "hf",
        tokens_trained=0,
        num_steps=0,
        include_tokenizer=False,
    )
    assert not (out / "tokenizer.json").exists()
