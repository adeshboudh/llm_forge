# llm_forge — phase build commands
# Phase 1: tokenizer (BPE 32k)
# Phase 2: data pipeline (FineWeb-Edu → uint16 shards)
# Phase 3+: not yet wired
#
# All commands run via `uv` — venv at .venv is managed by uv sync.
# See pyproject.toml for deps.

UV       := uv
PY       := $(UV) run python
PYTEST   := $(UV) run pytest
RUFF     := $(UV) run ruff

# ----- shared -----
.PHONY: help sync install lint format clean model-test model-summary model-summary-25m model-summary-125m model-summary-350m train-smoke train-25m train-test train-summary

help:
	@echo "llm_forge Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install     install dev deps (editable)"
	@echo "  make lint        ruff check"
	@echo "  make format      ruff format"
	@echo "  make clean       remove caches + artifacts"
	@echo ""
	@echo "Phase 1 — Tokenizer:"
	@echo "  make tok-train           train BPE 32k on FineWeb-Edu (1B chars, ~1hr)"
	@echo "  make tok-train-fast      download corpus + run Rust BPE trainer"
	@echo "  make tok-test            run tokenizer unit tests"
	@echo "  make tok-encode TEXT=... round-trip encode/decode check"
	@echo "  make tok-download-corpus  stream FineWeb-Edu to stdout (1B chars)"
	@echo ""
	@echo "Phase 2 — Data Pipeline:"
	@echo "  make data-test           run pipeline unit tests"
	@echo "  make data-smoke          tiny end-to-end shard run (1k tokens)"
	@echo "  make data-shards-10b     build 10B-token shards (canonical set; subsets used for 1B/5B training)"
	@echo "  make data-shards-resume  resume an interrupted 10B run from existing shards"
	@echo "  make kaggle-push        upload data/shards/ as a new Kaggle dataset"
	@echo "  make kaggle-version     upload as a new version of an existing dataset"
	@echo ""
	@echo "Phase 3 — Model:"
	@echo "  make model-test      run model unit tests"
	@echo "  make model-summary   print param count for --name=NAME"
	@echo ""
	@echo "Phase 4 — Training:"
	@echo "  make train-smoke         5-step CPU run with toy shards"
	@echo "  make train-25m           full 1B-token run (Kaggle TPU v5e-8)"
	@echo "  make train-test          run training unit tests"
	@echo "  make train-summary CONFIG=configs/training/model_25m.yaml   pre-run summary"

install:
	$(UV) sync --extra dev

sync:
	$(UV) sync --extra dev

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/shards data/shards_smoke

# =============================================================================
# Phase 1 — Tokenizer (BPE 32k)
# =============================================================================
.PHONY: tok-test tok-train tok-train-fast tok-encode tok-download-corpus

tok-test:
	$(PYTEST) tokenizer/tests/ -v

tok-download-corpus:
	$(PY) tokenizer/trainers/download_corpus.py --char-budget 1000000000

tok-train:
	$(PY) tokenizer/train_tokenizer.py \
		--output-dir tokenizer/saved/ \
		--vocab-size 32768 \
		--char-budget 1000000000

tok-train-fast:
	$(PY) tokenizer/trainers/download_corpus.py --char-budget 1000000000 \
		| ./tokenizer/trainers/bpe_rust/target/release/bpe-trainer \
			--output-dir tokenizer/saved/

tok-encode:
	$(PY) -c "from tokenizer.serialization.load import load_tokenizer; \
tok = load_tokenizer('tokenizer/saved/tokenizer.json'); \
ids = tok.encode('$(TEXT)'); \
print('ids:', ids); \
print('decoded:', tok.decode(ids))"

# =============================================================================
# Phase 2 — Data Pipeline (FineWeb-Edu → uint16 .npy shards)
# =============================================================================
# One canonical 10B shard set. Smaller training runs (25M, 125M) consume
# a prefix of this set via ShardedTokenDataset's token slicing.
.PHONY: data-test data-smoke data-shards-10b data-shards-resume kaggle-push kaggle-version

data-test:
	$(PYTEST) data/tests/ -v

# Tiny end-to-end run to validate the pipeline locally (no real HF download)
data-smoke:
	$(PY) -c "from data.preprocessing.shard_writer import ShardWriter; \
from data.preprocessing.tokenize_dataset import DocumentTokenizer; \
from tokenizer.serialization.load import load_tokenizer; \
import os, tempfile; \
tmp = tempfile.mkdtemp(prefix='shards_smoke_'); \
tok = load_tokenizer('tokenizer/saved/tokenizer.json'); \
dt = DocumentTokenizer(tok, add_eot=True); \
w = ShardWriter(output_dir=tmp, shard_size=10000, vocab_size=tok.vocab_size); \
texts = ['hello world ' * 50, 'goodbye world ' * 30, 'lorem ipsum ' * 100]; \
[tokens := dt.encode_document(t) or w.add(dt.encode_document(t)) for t in texts]; \
meta = w.finalize(dataset_version='v-smoke'); \
print('smoke OK ->', tmp, meta)"

# Canonical 10B-token shard set (~4-8h on Kaggle CPU).
# 25M model trains on first 1B (first ~20 shards).
# 125M model trains on first 5B (first ~100 shards).
# 350M model trains on all 10B (all ~200 shards).
data-shards-10b:
	$(PY) data/pipeline.py \
		--tokenizer tokenizer/saved/tokenizer.json \
		--output-dir data/shards/ \
		--token-budget 10000000000 \
		--dataset-version v1-bpe32k-fineweb10BT

# Resume an interrupted run. Continues from shard_{N:05d}.npy
# where N = count of existing shards. Re-runnable.
data-shards-resume:
	$(PY) data/pipeline.py \
		--tokenizer tokenizer/saved/tokenizer.json \
		--output-dir data/shards/ \
		--token-budget 10000000000 \
		--dataset-version v1-bpe32k-fineweb10BT \
		--skip-existing

# Push data/shards/ to Kaggle as a new dataset.
# Requires KAGGLE_USERNAME + KAGGLE_KEY (or ~/.kaggle/kaggle.json).
kaggle-push:
	bash scripts/push_shards_to_kaggle.sh create

# Push a new version of an existing Kaggle dataset.
kaggle-version:
	bash scripts/push_shards_to_kaggle.sh version

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

# =============================================================================
# Phase 4 — Pretraining (JAX/Flax + optax + orbax on Kaggle TPU v5e-8)
# =============================================================================
.PHONY: train-smoke train-25m train-test train-summary

train-smoke:
	$(PY) -m training.train --config configs/training/smoke_test.yaml --smoke

train-25m:
	$(PY) -m training.train --config configs/training/model_25m.yaml

train-test:
	$(PYTEST) training/tests/ -v

train-summary:
	@test -n "$(CONFIG)" || (echo "Usage: make train-summary CONFIG=configs/training/model_25m.yaml" && exit 1)
	$(PY) -m training.summary --config $(CONFIG)
