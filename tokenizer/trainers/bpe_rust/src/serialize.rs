//! Serialize trained BPE to disk.
//!
//! Outputs (same format as Python save.py — load.py compatible):
//!   tokenizer.json  — canonical; the only file load.py reads
//!   vocab.json      — human inspection: token_repr → id
//!   merges.txt      — human inspection: integer pair per line
//!
//! tokenizer.json schema:
//! {
//!   "version": "1.0",
//!   "vocab_size": 32768,
//!   "special_tokens": { "<|endoftext|>": 0, ... },
//!   "merges": [[a, b], ...]   ← integer pairs, NOT strings
//! }

use crate::bpe::BpeTrainer;
use crate::pretokenize::{BYTE_OFFSET, MERGE_START_ID};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write as IoWrite};
use std::path::Path;

/// Special tokens — LOCKED forever after first .npy shard write.
const SPECIAL_TOKENS: &[(&str, u32)] = &[
    ("<|endoftext|>", 0),
    ("<|pad|>", 1),
    ("<|unk|>", 2),
    ("<|bos|>", 3),
    ("<|eos|>", 4),
];

/// Write all three output files.
pub fn save(trainer: &BpeTrainer, output_dir: &Path) -> io::Result<()> {
    write_tokenizer_json(trainer, output_dir)?;
    write_vocab_json(trainer, output_dir)?;
    write_merges_txt(trainer, output_dir)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// tokenizer.json — canonical, load.py reads this
// ---------------------------------------------------------------------------

fn write_tokenizer_json(trainer: &BpeTrainer, output_dir: &Path) -> io::Result<()> {
    let special: Map<String, Value> = SPECIAL_TOKENS
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();

    // Merges as [[a, b], ...] integer arrays
    let merges: Vec<[u32; 2]> = trainer.merges.iter().map(|&(a, b)| [a, b]).collect();

    let doc = json!({
        "version":       "1.0",
        "vocab_size":    trainer.vocab_size,
        "special_tokens": special,
        "merges":        merges,
    });

    let path = output_dir.join("tokenizer.json");
    let content = serde_json::to_string_pretty(&doc)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, content)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// vocab.json — human inspection only
// ---------------------------------------------------------------------------

/// Reconstruct id_to_token from merge list.
/// Mirrors load.py _rebuild_id_to_token(). Needed for vocab.json.
fn build_id_to_token(merges: &[(u32, u32)]) -> HashMap<u32, Vec<u8>> {
    let mut id_to_token: HashMap<u32, Vec<u8>> = HashMap::new();

    // Byte tokens: IDs 5–260
    for b in 0u8..=255 {
        id_to_token.insert(b as u32 + BYTE_OFFSET, vec![b]);
    }

    // Merged tokens: IDs 261+
    let mut next_id = MERGE_START_ID;
    for &(a, b) in merges {
        let ta = id_to_token.get(&a).cloned().unwrap_or_default();
        let tb = id_to_token.get(&b).cloned().unwrap_or_default();
        let merged: Vec<u8> = ta.into_iter().chain(tb).collect();
        id_to_token.insert(next_id, merged);
        next_id += 1;
    }

    id_to_token
}

fn write_vocab_json(trainer: &BpeTrainer, output_dir: &Path) -> io::Result<()> {
    let id_to_token = build_id_to_token(&trainer.merges);

    let mut vocab: Map<String, Value> = Map::new();

    // Special tokens first
    for &(name, id) in SPECIAL_TOKENS {
        vocab.insert(name.to_string(), json!(id));
    }

    // Byte + merged tokens, sorted by ID
    let mut ids: Vec<u32> = id_to_token.keys().copied().collect();
    ids.sort_unstable();
    for id in ids {
        let bytes = &id_to_token[&id];
        // Lossy UTF-8: non-UTF-8 byte seqs show as replacement chars.
        // This file is for inspection only — load.py does not read it.
        let repr = String::from_utf8_lossy(bytes).into_owned();
        vocab.insert(repr, json!(id));
    }

    let path = output_dir.join("vocab.json");
    let content = serde_json::to_string_pretty(&vocab)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, content)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// merges.txt — human inspection only
// ---------------------------------------------------------------------------

fn write_merges_txt(trainer: &BpeTrainer, output_dir: &Path) -> io::Result<()> {
    let path = output_dir.join("merges.txt");
    let mut f = fs::File::create(path)?;

    writeln!(f, "# BPE merges -- bpe-trainer v{}", env!("CARGO_PKG_VERSION"))?;
    writeln!(f, "# Format: token_a token_b  (space-separated IDs)")?;
    writeln!(f, "# Merge N produces ID {} + N - 1", MERGE_START_ID)?;
    writeln!(f, "#")?;

    for &(a, b) in &trainer.merges {
        writeln!(f, "{} {}", a, b)?;
    }

    Ok(())
}
