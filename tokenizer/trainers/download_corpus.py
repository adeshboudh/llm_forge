"""
Download raw text from FineWeb-Edu and write to stdout (or file).
Pipe output directly to bpe-trainer (Rust) for fast BPE training.

Usage:
    # Pipe to Rust trainer
    python tokenizer/trainers/download_corpus.py \
        --char-budget 1_000_000_000 \
        | ./tokenizer/trainers/bpe_rust/target/release/bpe-trainer \
            --output-dir tokenizer/saved/

    # Save to file first (then train separately)
    python tokenizer/trainers/download_corpus.py \
        --char-budget 1_000_000_000 \
        --output /tmp/corpus.txt

    # Then train:
    ./tokenizer/trainers/bpe_rust/target/release/bpe-trainer \
        --input /tmp/corpus.txt \
        --output-dir tokenizer/saved/

Documents are separated by <|endoftext|> on its own line.
The Rust trainer splits on this marker and pretokenizes each doc separately.

Requirements:
    pip install datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download FineWeb-Edu corpus for BPE training")
    p.add_argument(
        "--char-budget",
        type=int,
        default=1_000_000_000,
        help="Stop after N chars of text (default: 1B)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write to file instead of stdout",
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
        help="Skip docs shorter than N chars (default: 100)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=50_000,
        help="Print progress every N docs (default: 50000, 0=silent)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    print(
        f"Streaming HuggingFaceFW/fineweb-edu config={args.dataset_config}...",
        file=sys.stderr,
    )
    print(
        f"  char_budget  : {args.char_budget:,}",
        file=sys.stderr,
    )

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.dataset_config,
        split="train",
        streaming=True,
    )

    total_chars = 0
    total_docs  = 0

    try:
        for doc in ds:
            text = doc.get("text", "")
            if len(text) < args.min_doc_len:
                continue

            out.write(text)
            out.write("\n<|endoftext|>\n")

            total_chars += len(text)
            total_docs  += 1

            if args.log_every and total_docs % args.log_every == 0:
                print(
                    f"  {total_docs:>8,} docs  {total_chars / 1e9:.3f}B chars"
                    f"  ({100 * total_chars / args.char_budget:.1f}%)",
                    file=sys.stderr,
                )

            if total_chars >= args.char_budget:
                print(
                    f"  Budget reached: {total_chars:,} chars across {total_docs:,} docs",
                    file=sys.stderr,
                )
                break

    finally:
        if args.output:
            out.close()

    print(
        f"Done. {total_docs:,} docs  {total_chars / 1e9:.3f}B chars",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
