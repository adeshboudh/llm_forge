"""
Data pipeline — stream FineWeb-Edu, tokenize, write uint16 .npy shards.

Run on Kaggle / Lightning AI (CPU, ~4–8 hours for full 10BT):

    python data/pipeline.py \\
        --tokenizer  tokenizer/saved/tokenizer.json \\
        --output-dir data/shards/ \\
        --token-budget 10_000_000_000 \\
        --dataset-version v1-bpe32k-fineweb10BT

For a 1B token smoke-test run:

    python data/pipeline.py \\
        --tokenizer  tokenizer/saved/tokenizer.json \\
        --output-dir data/shards_1b/ \\
        --token-budget 1_000_000_000 \\
        --dataset-version v1-bpe32k-fineweb1B

Token budget is measured in OUTPUT tokens (after BPE), not chars.
Pipeline streams until accumulated token count reaches the budget.

Output: upload data/shards/ to Kaggle Dataset 'llm-forge-tokens-v1'.

Requirements:
    pip install datasets numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build uint16 .npy token shards from FineWeb-Edu")
    p.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer.json (from Phase 1 training)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write shard_*.npy + metadata.json",
    )
    p.add_argument(
        "--token-budget",
        type=int,
        default=10_000_000_000,
        help="Stop after this many output tokens (default: 10B = full dataset)",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=50_000_000,
        help="Tokens per shard file (default: 50M → ~100MB per shard)",
    )
    p.add_argument(
        "--dataset-version",
        type=str,
        default="v1-bpe32k-fineweb10BT",
        help="Version tag written to metadata.json (default: v1-bpe32k-fineweb10BT)",
    )
    p.add_argument(
        "--dataset-config",
        type=str,
        default="sample-10BT",
        help="FineWeb-Edu HuggingFace config (default: sample-10BT)",
    )
    p.add_argument(
        "--min-doc-len",
        type=int,
        default=100,
        help="Skip documents shorter than N chars (default: 100)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Repo root on sys.path so relative imports work
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import numpy as np  # noqa: F401 — validate numpy available early

    from tokenizer.serialization.load import load_tokenizer
    from data.sources.fineweb import FineWebEduStream
    from data.preprocessing.tokenize_dataset import DocumentTokenizer
    from data.preprocessing.shard_writer import ShardWriter

    print("=" * 64)
    print("LLM-Forge Data Pipeline")
    print("=" * 64)
    print(f"  tokenizer      : {args.tokenizer}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  token_budget   : {args.token_budget:,}")
    print(f"  shard_size     : {args.shard_size:,} tokens (~{args.shard_size*2/1e6:.0f} MB)")
    print(f"  dataset_version: {args.dataset_version}")
    print("=" * 64)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tok = load_tokenizer(args.tokenizer)
    print(f"  {tok}")

    # Init pipeline components
    # FineWebEduStream uses char budget internally; we stop by token count
    # so set char_budget very high and rely on DocumentTokenizer token tracking
    source = FineWebEduStream(
        token_budget=args.token_budget * 5,   # chars >> tokens (rough 5× ratio)
        config=args.dataset_config,
        min_doc_len=args.min_doc_len,
    )
    doc_tok = DocumentTokenizer(tok, add_eot=True)
    writer  = ShardWriter(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        vocab_size=tok.vocab_size,
    )

    print("\nStreaming and tokenizing...")
    t0           = time.time()
    total_tokens = 0

    for token_ids in doc_tok.encode_stream(source):
        writer.add(token_ids)
        total_tokens += len(token_ids)

        if total_tokens >= args.token_budget:
            print(f"\n  Token budget reached: {total_tokens:,} tokens")
            break

    metadata = writer.finalize(dataset_version=args.dataset_version)
    elapsed  = time.time() - t0

    print(f"\nPipeline complete in {elapsed/3600:.2f} hours")
    print(f"  total_tokens : {metadata['total_tokens']:,}")
    print(f"  total_shards : {metadata['total_shards']}")
    print(f"\nUpload '{args.output_dir}' to Kaggle Dataset '{args.dataset_version}'")


if __name__ == "__main__":
    main()
