"""
Train BPE tokenizer on FineWeb-Edu (streaming) and save to disk.

Run on Lightning AI / Kaggle / any machine with HuggingFace access:

    python tokenizer/train_tokenizer.py \\
        --output-dir tokenizer/saved/ \\
        --vocab-size 32768 \\
        --char-budget 1_000_000_000   # ~1GB text (~700MB after BPE training)

Requirements:
    pip install datasets

The script does NOT need GPU/TPU — runs on CPU.
Expected time: 30–90 min for 1B chars depending on machine.

Output (upload these to Kaggle Dataset 'llm-forge-tokenizer-v1'):
    tokenizer/saved/tokenizer.json   ← canonical
    tokenizer/saved/vocab.json
    tokenizer/saved/merges.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BPE tokenizer on FineWeb-Edu"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="tokenizer/saved",
        help="Directory to write tokenizer files (default: tokenizer/saved/)",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=32_768,
        help="Target vocabulary size including special tokens (default: 32768)",
    )
    p.add_argument(
        "--char-budget",
        type=int,
        default=1_000_000_000,
        help="Stop streaming after this many characters (default: 1B ≈ 1GB text)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name (default: HuggingFaceFW/fineweb-edu)",
    )
    p.add_argument(
        "--dataset-config",
        type=str,
        default="sample-10BT",
        help="Dataset config/subset (default: sample-10BT)",
    )
    p.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name containing raw text (default: text)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=2_000,
        help="Print merge progress every N merges (default: 2000, 0=silent)",
    )
    return p.parse_args()


def stream_texts(dataset_name: str, config: str, text_field: str, char_budget: int):
    """
    Stream text documents from HuggingFace dataset until char_budget exhausted.

    Yields:
        str: One document at a time.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading dataset '{dataset_name}' config='{config}' (streaming)...")
    ds = load_dataset(dataset_name, config, split="train", streaming=True)

    chars_seen = 0
    docs_seen  = 0
    log_every  = 100_000  # log every 100k docs

    for row in ds:
        text = row.get(text_field, "")
        if not text:
            continue

        yield text

        chars_seen += len(text)
        docs_seen  += 1

        if docs_seen % log_every == 0:
            pct = 100 * chars_seen / char_budget
            print(
                f"  streamed {docs_seen:>8,} docs  "
                f"{chars_seen / 1e9:.3f}B chars  "
                f"({pct:.1f}% of budget)"
            )

        if chars_seen >= char_budget:
            print(f"  char budget reached: {chars_seen:,} chars across {docs_seen:,} docs")
            break


def main() -> None:
    args = parse_args()

    # Repo root detection — script runs from any cwd
    script_dir = Path(__file__).parent
    repo_root  = script_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tokenizer.trainers.bpe import BPETrainer
    from tokenizer.serialization.save import save_tokenizer

    print("=" * 60)
    print("BPE Tokenizer Training")
    print("=" * 60)
    print(f"  vocab_size   : {args.vocab_size:,}")
    print(f"  char_budget  : {args.char_budget:,} (~{args.char_budget/1e9:.1f}GB text)")
    print(f"  dataset      : {args.dataset} / {args.dataset_config}")
    print(f"  output_dir   : {args.output_dir}")
    print("=" * 60)

    t0 = time.time()

    trainer = BPETrainer(vocab_size=args.vocab_size, log_every=args.log_every)

    texts = stream_texts(
        dataset_name=args.dataset,
        config=args.dataset_config,
        text_field=args.text_field,
        char_budget=args.char_budget,
    )

    trainer.train(texts)

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed/60:.1f} min")

    save_tokenizer(trainer, output_dir=args.output_dir)

    print("\nDone. Upload these to Kaggle Dataset 'llm-forge-tokenizer-v1':")
    out = Path(args.output_dir)
    for f in ["tokenizer.json", "vocab.json", "merges.txt"]:
        fpath = out / f
        size  = fpath.stat().st_size / 1e6 if fpath.exists() else 0
        print(f"  {fpath}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
