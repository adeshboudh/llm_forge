#!/usr/bin/env bash
# push_shards_to_kaggle.sh — upload data/shards/ as a Kaggle Dataset.
#
# Two modes:
#   1. CREATE  — first time, makes a new dataset
#   2. VERSION — creates a new version of an existing dataset
#
# Reads KAGGLE_USERNAME / KAGGLE_KEY from env (or ~/.kaggle/kaggle.json).
#
# Usage:
#   # First push (creates dataset)
#   ./scripts/push_shards_to_kaggle.sh create
#
#   # New version (after regenerating shards)
#   ./scripts/push_shards_to_kaggle.sh version
#
# Env overrides:
#   DATASET_SLUG  — kaggle dataset name (default: llm-forge-tokens-v1)
#   SHARDS_DIR    — local shard directory (default: data/shards)
#   NOTE          — version note (default: auto-generated)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODE="${1:-create}"  # create | version

DATASET_SLUG="${DATASET_SLUG:-llm-forge-tokens-v1}"
SHARDS_DIR="${SHARDS_DIR:-data/shards}"
STAGE_DIR="${STAGE_DIR:-./.kaggle_stage_${DATASET_SLUG}}"

# Read username from env or kaggle.json
if [[ -z "${KAGGLE_USERNAME:-}" ]]; then
    if [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
        KAGGLE_USERNAME=$(python3 -c "import json; print(json.load(open('${HOME}/.kaggle/kaggle.json'))['username'])")
        export KAGGLE_USERNAME
    else
        echo "ERROR: KAGGLE_USERNAME not set and ~/.kaggle/kaggle.json not found."
        echo "Get a kaggle API token at https://www.kaggle.com/settings → API → Create New Token"
        exit 1
    fi
fi

if [[ -z "${KAGGLE_KEY:-}" ]]; then
    if [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
        KAGGLE_KEY=$(python3 -c "import json; print(json.load(open('${HOME}/.kaggle/kaggle.json'))['key'])")
        export KAGGLE_KEY
    else
        echo "ERROR: KAGGLE_KEY not set and ~/.kaggle/kaggle.json not found."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if [[ ! -d "${SHARDS_DIR}" ]]; then
    echo "ERROR: ${SHARDS_DIR} not found. Run 'make data-shards-10b' first."
    exit 1
fi

SHARD_COUNT=$(find "${SHARDS_DIR}" -maxdepth 1 -name 'shard_*.npy' | wc -l)
if [[ "${SHARD_COUNT}" -eq 0 ]]; then
    echo "ERROR: no shard_*.npy files in ${SHARDS_DIR}"
    exit 1
fi

TOTAL_SIZE=$(du -sh "${SHARDS_DIR}" | cut -f1)
echo "============================================================"
echo "Kaggle Push — ${MODE} mode"
echo "============================================================"
echo "  dataset     : ${KAGGLE_USERNAME}/${DATASET_SLUG}"
echo "  shards dir  : ${SHARDS_DIR}"
echo "  shard count : ${SHARD_COUNT}"
echo "  total size  : ${TOTAL_SIZE}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Verify kaggle CLI available
# ---------------------------------------------------------------------------

if ! command -v kaggle >/dev/null 2>&1; then
    echo "Installing kaggle CLI..."
    pip install -q kaggle
fi

# ---------------------------------------------------------------------------
# Stage shards (hard-link to avoid 20GB copy)
# ---------------------------------------------------------------------------

echo "Staging shards in ${STAGE_DIR}..."
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

# Hard-link shard files (instant, no disk usage)
find "${SHARDS_DIR}" -maxdepth 1 -name 'shard_*.npy' \
    -exec ln {} "${STAGE_DIR}/" \;

# Copy metadata.json (small, just copy)
if [[ -f "${SHARDS_DIR}/metadata.json" ]]; then
    cp "${SHARDS_DIR}/metadata.json" "${STAGE_DIR}/"
else
    echo "WARNING: no metadata.json in ${SHARDS_DIR} — generating..."
    python3 -c "
import json, numpy as np, glob
from datetime import datetime, timezone
from pathlib import Path

shard_dir = Path('${SHARDS_DIR}')
shards = sorted(shard_dir.glob('shard_*.npy'))
records, total = [], 0
for p in shards:
    n = int(np.load(p, mmap_mode='r').shape[0])
    records.append({'index': len(records), 'filename': p.name,
                    'tokens': n, 'size_mb': round(p.stat().st_size / 1e6, 2)})
    total += n

meta = {
    'dataset_version': '${DATASET_SLUG}',
    'vocab_size': 32768,
    'total_tokens': total,
    'total_shards': len(shards),
    'shard_size': 50_000_000,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'shards': records,
}
with open('${STAGE_DIR}/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'wrote metadata.json: {total:,} tokens / {len(shards)} shards')
"
fi

# Sanity check
STAGED_COUNT=$(find "${STAGE_DIR}" -maxdepth 1 -name 'shard_*.npy' | wc -l)
if [[ "${STAGED_COUNT}" -ne "${SHARD_COUNT}" ]]; then
    echo "ERROR: staging lost files (${STAGED_COUNT} vs ${SHARD_COUNT})"
    exit 1
fi
echo "  staged: ${STAGED_COUNT} shard files + metadata.json"
echo ""

# ---------------------------------------------------------------------------
# Write dataset metadata
# ---------------------------------------------------------------------------

NOTE="${NOTE:-Token shards for LLM-Forge pretraining (BPE 32k, FineWeb-Edu)}"

cat > "${STAGE_DIR}/dataset-metadata.json" <<EOF
{
  "title": "LLM-Forge Tokens v1 (FineWeb-Edu, BPE 32k)",
  "id": "${KAGGLE_USERNAME}/${DATASET_SLUG}",
  "description": "uint16 .npy token shards for llm_forge pretraining.\n\nSource: HuggingFaceFW/fineweb-edu (sample-10BT config).\nTokenizer: BPE 32k trained on 1B chars FineWeb-Edu (see llm-forge-tokenizer-v1).\nShard size: 50M tokens (~100MB per .npy file).\nTotal: ${SHARD_COUNT} shards, ${TOTAL_SIZE}.\n\nLoad with: from data.loaders.npy_loader import ShardedTokenDataset\n  ds = ShardedTokenDataset('/kaggle/input/${DATASET_SLUG}', seq_len=1024)",
  "licenses": [{"name": "odc-by"}],
  "keywords": ["llm", "pretraining", "tokens", "fineweb-edu", "bpe"],
  "collaborators": []
}
EOF

# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

case "${MODE}" in
    create)
        echo "Creating new dataset: ${KAGGLE_USERNAME}/${DATASET_SLUG}"
        echo "NOTE: --dir-mode tar bundles files into a single archive."
        echo "  - Faster upload for many small files"
        echo "  - Files still extract to individual shard_*.npy on download"
        echo ""
        kaggle datasets create \
            -p "${STAGE_DIR}" \
            --dir-mode tar \
            --public
        ;;

    version)
        echo "Creating new version of: ${KAGGLE_USERNAME}/${DATASET_SLUG}"
        kaggle datasets version \
            -p "${STAGE_DIR}" \
            --dir-mode tar \
            --message "${NOTE}"
        ;;

    *)
        echo "ERROR: unknown mode '${MODE}'. Use 'create' or 'version'."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Push complete"
echo "============================================================"
echo "  dataset URL: https://www.kaggle.com/datasets/${KAGGLE_USERNAME}/${DATASET_SLUG}"
echo ""
echo "To use in a kaggle notebook:"
echo "  1. Sidebar → + Add data → Your datasets → ${DATASET_SLUG}"
echo "  2. Mount path: /kaggle/input/${DATASET_SLUG}/"
echo ""
echo "Cleaning up staging dir..."
rm -rf "${STAGE_DIR}"
echo "Done."
