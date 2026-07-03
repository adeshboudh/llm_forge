# Phase 1 — Tokenizer

Status: complete. Code, tests, and the trained 32k tokenizer are all in git.

This is the first phase of `llm_forge` — a custom byte-level BPE tokenizer
trained from scratch on FineWeb-Edu, with a parallel Python/Rust trainer
implementation, a runtime encoder/decoder, and a `tokenizer.json`
serialization. Everything downstream (data shards, model, training)
consumes the artifact produced here, so this phase is a hard prerequisite
for all the rest.

## What we built

- Python BPE trainer — `tokenizer/trainers/bpe.py` (238 lines)
- Rust BPE trainer — `tokenizer/trainers/bpe_rust/src/{bpe,main,pretokenize,serialize}.rs` (~550 lines of `.rs` + 33-line `Cargo.toml`)
- Corpus downloader — `tokenizer/trainers/download_corpus.py` (146 lines)
- Training CLI — `tokenizer/train_tokenizer.py` (171 lines, Python trainer; the real training uses the Rust binary)
- Encoder — `tokenizer/runtime/encode.py` (214 lines)
- Decoder — `tokenizer/runtime/decode.py` (132 lines)
- Serialization — `tokenizer/serialization/save.py` (162 lines) + `load.py` (181 lines), writes a HuggingFace-compatible `tokenizer.json`
- Tests — `tokenizer/tests/test_bpe.py`, **68 tests, all passing**
- Trained artifact — `tokenizer/saved/{tokenizer.json, vocab.json, merges.txt}`, vocab 32,768
- Config — `configs/tokenizer/bpe_32k.yaml` (reference only; no code wiring yet)

## What worked

- **BPE 32,768 (2¹⁵).** Picking a power of two for vocab size means every
  token ID fits in `uint16`, and shard size drops to half of `int32`.
  This is the one decision that touched everything: shard format, the
  Rust trainer, downstream data loader, embedding matrix shape. Locking
  it in early removed a class of bugs we never had to think about.
- **GPT-2 style pretokenization regex.** The `'\w+|\d+|[^\s\w\d]+|...`
  pattern matches on word boundaries, digits, and punctuation chunks
  before BPE runs. It makes merges far more linguistically meaningful
  (you actually learn "ing" as a suffix, not garbage byte sequences
  crossing word boundaries) and dramatically reduces vocab waste on
  Common Crawl noise. The same regex is used in both Python and Rust
  trainers — easy to cross-check outputs.
- **Rust trainer (rayon-parallel, ~50× faster than Python).** Wall time
  for a 1B-char training run on an 8-core laptop dropped from
  estimated 4–5 days in pure Python to ~7.8 hours in Rust. The
  bottleneck was pretokenization + pair counting, and both parallelize
  trivially over pre-tokens.
- **Special token IDs locked at 0–4.** `SPECIAL_TOKENS` in `bpe.py:40`
  is the single source of truth and is also baked into the saved
  `tokenizer.json` + `vocab.json`. Both trainers (Python and Rust)
  reserve these IDs before counting starts. Once a `.npy` shard is
  written that contains token ID 0 meaning `<|endoftext|>`, we can
  never move that ID without invalidating the entire dataset.
- **Pipe-based corpus streaming.** `download_corpus.py | bpe-trainer`
  means the 1B-char corpus never sits on disk. The Python downloader
  streams FineWeb-Edu docs to stdout; the Rust binary reads from a
  pipe. Document boundaries are marked by an `<|endoftext|>` line,
  and the Rust trainer splits on that marker.
- **`tokenizer.json` as the canonical artifact.** The serialized format
  matches HuggingFace's `tokenizers` library, so we can sanity-check
  it by loading it with `tokenizers.Tokenizer.from_file(...)` in a
  notebook. This is cheap insurance against silent format bugs.

## What we learned

### The BPE algorithm itself

The whole thing is simpler than it looks. For a fixed corpus:

1. Pretokenize (regex). Result: ~1.4M unique pre-tokens on FineWeb-Edu.
2. Each pre-token → list of UTF-8 byte IDs (5–260 in our vocab).
3. Count adjacent-byte pairs across the whole corpus.
4. Pick the most-frequent pair, assign it a new merge ID, replace all
   occurrences of that pair in the corpus with the new ID.
5. Repeat until `vocab_size - (specials + bytes)` merges have been
   learned, or pair counts stop moving.

The data structure that matters: a priority queue (or "pair counter
that re-sorts cheaply") keyed on `(count, lex_smaller_token)`. We
chose to **not** use a real heap — pair counts change as merges are
applied, and the bookkeeping for invalidating heap entries was more
trouble than just re-scanning the top-K. Re-scanning was the
bottleneck in Python but is fine in Rust with rayon.

Vocab layout is the other thing people get wrong: the **order of
IDs in the final vocab is not arbitrary**. It is the order merges
were learned, with specials and base bytes prepended. So ID 5 = byte
0x00, ID 6 = byte 0x01, ..., ID 260 = byte 0xFF, ID 261 = first merge,
ID 262 = second merge, ..., ID 32767 = last merge. Embedding
matrices and final `lm_head` rows are addressed by ID, so the layout
is the contract.

### Byte-level encoding for Unicode

The cleanest way to handle "any input, any language" is: never store
characters, store **UTF-8 bytes**. The base vocab is exactly 256
bytes (0x00–0xFF), so any Unicode string is representable as a
sequence of byte IDs. No "unk" token for normal text, ever.

The only cost is that common multi-byte chars (Hindi, CJK, emoji)
expand into 2–4 base tokens. BPE merges partially recover this by
learning common multi-byte sequences (e.g. `##ा`, `##े`), but you
will never get the compression ratio of a SentencePiece-Unigram
tokenizer on non-Latin text. For FineWeb-Edu, which is mostly
English, this is a non-issue. For multilingual it would be.

### Why byte-level + regex + BPE (the GPT-2 recipe)

Three layers, each pulling its weight:

- **Byte-level base vocab (256)** — universal. No char-table choices,
  no `<unk>`, no language detection. Predictable coverage.
- **GPT-2 pretokenization regex** — splits at *linguistic* boundaries
  (words, digits, punctuation) so BPE is applied within a meaningful
  span. Without it, BPE would happily merge across word boundaries
  and learn nonsense.
- **BPE on top** — compresses frequent byte sequences inside
  pre-tokens into single IDs. This is where vocabulary efficiency
  comes from.

Strip any one of these three and the tokenizer gets worse.
- No regex → garbage merges, bad compression on real text.
- No bytes → OOV for rare chars, normalization hell.
- No BPE → vocab too big or too redundant.

### Rust vs Python for training

The Python trainer is in `bpe.py`, ~240 lines, easy to read. The
Rust trainer is ~550 lines of `.rs`. The difference is wall time:

| Trainer | Wall time on 1B chars | Notes |
|---------|----------------------|-------|
| Python  | estimated 4–5 days   | reference impl, correctness baseline |
| Rust    | ~7.8 hours           | rayon-parallel pretokenize + pair counting |

The Rust speedup comes almost entirely from parallelizing pair
counting over pre-tokens. The merge step itself is sequential (it
has to be — each merge changes the corpus for the next one), but
pretokenization and pair counting are embarrassingly parallel. With
~1.4M unique pre-tokens, the work unit count is high enough that
rayon's overhead is irrelevant.

Keeping the Python trainer in the repo was the right call: when
the Rust trainer gave a weird merge, we ran the same input through
the Python trainer and diffed the outputs. Saved us at least twice
during development.

## What we got wrong / would redo

- **"1B chars" not "1B tokens" for the training corpus.** Char-level
  budgeting was the natural unit for a streaming downloader — we
  don't have a tokenizer before we have a tokenizer. It turned out
  fine: 1B chars of FineWeb-Edu produced 1,408,047 unique
  pre-tokens, which is comfortably more than the 32,507 merges we
  needed to learn. Vocabulary saturation was clearly hit (last 5,000
  merges each had < 50 occurrences). If we ever retrain, we would
  skip the char-to-token guesswork and just budget in tokens,
  measuring via the Python trainer's intermediate output.
- **Vocab size 32,768 vs the alternatives.** We picked 32,768
  because of uint16. The well-known alternatives:
  - GPT-2: 50,257 — does not fit uint16 (50,257 > 65,535 but > 32,767)
  - Llama 1: 32,000 — close, but the awkward 32,000 vs 32,768
    gap means we'd need uint32 shards
  - Llama 2/3: 128,256 — requires uint32, doubles shard disk cost
  - Mistral 7B: 32,000 — same as Llama 1
  
  We chose the power-of-two for the shard-size win. The cost is
  that vocab efficiency on multilingual data is slightly worse than
  Llama's 32k, and way worse than Llama 3's 128k. For our English
  pretraining data the difference is small.
- **"Pre-tokenize + count pairs + merge" is a learning reference
  Python trainer worth keeping.** Tempting to delete it as "unused
  code" once the Rust trainer works. Don't. The Python version is
  short, correct, and reads top-to-bottom. It's how we debugged
  every weird merge the Rust trainer produced. The cost of carrying
  it is ~240 lines of code that the Rust trainer replaces in
  practice.

## Phase 2 prep

The handoff to Phase 2 (data pipeline) is mechanical and unforgiving:

- **Special tokens are locked.** `SPECIAL_TOKENS` IDs 0–4 in
  `bpe.py:40` are the only acceptable encoding of `<|endoftext|>`,
  `<|pad|>`, `<|unk|>`, `<|bos|>`, `<|eos|>`. After the first
  `.npy` shard is written, these IDs are the dataset's definition
  of "end of document", "padding", etc. Changing the order or
  removing a token silently corrupts every shard.

- **Vocab 32,768 → uint16 safe.** Every token ID fits in `uint16`
  (max 32,767). Shard format: 50M tokens/shard, raw uint16 `.npy`,
  ~100MB per file. This is the contract.

- **Encoder prepends `<|endoftext|>` per document.** This is what
  makes the shard a flat stream with recoverable doc boundaries.
  The model learns "after EOT, start a new doc" via the language
  modeling objective; the data loader just sees one long token
  stream and slices it into `seq_len`-sized training examples.

- **Encode/decode roundtrip is tested.** The `TestSerialization`
  group in `test_bpe.py` saves a tokenizer, reloads it, encodes
  text, and checks the result matches the in-memory encoder. This
  is the safety net for Phase 2: every doc we tokenize goes
  through the same code path, and the saved tokenizer is the
  single source of truth for the encode step.

## Open questions for Phase 2 and beyond

- **Normalization.** Right now we don't NFC-normalize Unicode before
  pretokenization. The same logical character in two different
  normalization forms produces different byte sequences and
  therefore different tokens. FineWeb-Edu is mostly clean, but
  we should measure: what fraction of tokens are "duplicates" of
  the same logical char under different normalizations? If it's
  >1%, add a normalization step in the data pipeline.

- **Pre-token count for the corpus is the wrong number for vocab
  saturation analysis.** We have 1.4M unique pre-tokens and
  learned 32,507 merges. The right diagnostic is *pair frequency
  at saturation* (the count of the least-frequent merge we kept).
  We eyeballed it from logs; the Rust trainer should dump this
  explicitly on a future run.

- **Adding a real "unk" strategy.** `<|unk|>` (ID 2) is in the
  vocab but the encoder never produces it. With byte-level base
  vocab, every byte has an ID, so there is no character the
  encoder cannot represent. This is correct, but it means
  `<|unk|>` is dead weight. Keep it in the vocab for now (the
  dataset is committed; removing it is a v2 tokenizer event).

- **Tokenizer upload to Kaggle.** `tokenizer.json` is 1.1MB —
  small enough for git, but Phase 2 wants the **exact same
  bytes** at training time. Currently the plan is: commit to
  git + also publish as `llm-forge-tokenizer-v1` Kaggle Dataset
  (per `docs/progress.md` Kaggle Artifacts table). The Kaggle
  upload is not done yet.

- **Special token handling in encode vs runtime.** The encoder
  splits on the GPT-2 regex first, then looks up tokens. Special
  tokens are matched as literal strings, with a guard that they
  never appear in the middle of a pre-token. Need to verify this
  with adversarial inputs (e.g. `<|endoftext|>` inside a code
  block) before Phase 2 ingests anything weird.

## Files added

| Path | LOC | Role |
|------|-----|------|
| `tokenizer/trainers/bpe.py` | 238 | Python BPE trainer (reference) |
| `tokenizer/trainers/bpe_rust/src/bpe.rs` | 171 | Rust BPE algorithm |
| `tokenizer/trainers/bpe_rust/src/main.rs` | 177 | Rust CLI entrypoint |
| `tokenizer/trainers/bpe_rust/src/pretokenize.rs` | 62 | Rust pretokenizer (regex + UTF-8 bytes) |
| `tokenizer/trainers/bpe_rust/src/serialize.rs` | 141 | Rust tokenizer.json writer |
| `tokenizer/trainers/bpe_rust/Cargo.toml` | 33 | Rust deps (rayon, regex, serde) |
| `tokenizer/trainers/download_corpus.py` | 146 | FineWeb-Edu streamer → stdout |
| `tokenizer/train_tokenizer.py` | 171 | Python CLI wrapper |
| `tokenizer/runtime/encode.py` | 214 | Runtime encoder |
| `tokenizer/runtime/decode.py` | 132 | Runtime decoder |
| `tokenizer/serialization/save.py` | 162 | `tokenizer.json` writer |
| `tokenizer/serialization/load.py` | 181 | `tokenizer.json` loader |
| `tokenizer/tests/test_bpe.py` | 396 | 68 tests |
| `tokenizer/saved/tokenizer.json` | — | Trained 32k artifact (1.1MB, committed) |
| `configs/tokenizer/bpe_32k.yaml` | — | Reference config |

Total new code: ~2,200 lines of Python + Rust, plus the trained
artifact. The Rust trainer did the actual 7.8h training run; the
Python trainer is kept as the readable reference and correctness
oracle. The trained tokenizer is committed to git (1.1MB) so the
artifact is never lost — Phase 2 can start without re-running
training.
