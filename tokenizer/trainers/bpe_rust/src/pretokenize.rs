//! GPT-2 style pretokenization.
//!
//! Pipeline:
//!   raw text
//!     → split on "<|endoftext|>" (document boundary — skip separator)
//!     → apply GPT-2 regex → "words" (pre-tokens)
//!     → encode each word as byte token IDs (byte + BYTE_OFFSET)
//!     → accumulate into word → frequency map
//!
//! ID layout (mirrors Python bpe.py):
//!   0–4    : special tokens (never produced here)
//!   5–260  : byte tokens (byte b → ID b + 5)
//!   261+   : BPE merge tokens (assigned during training)

use fancy_regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Byte tokens start at ID 5 (0-4 reserved for specials).
pub const BYTE_OFFSET: u32 = 5;

/// First merge gets this ID: 5 specials + 256 bytes = 261.
pub const MERGE_START_ID: u32 = 261;

/// GPT-2 pretokenization pattern.
/// Splits contractions, words, digits, punctuation, whitespace.
/// Negative lookahead (?!\S) requires fancy-regex (not the std regex crate).
static GPT2_PAT: OnceLock<Regex> = OnceLock::new();

fn gpt2_regex() -> &'static Regex {
    GPT2_PAT.get_or_init(|| {
        Regex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        )
        .expect("Invalid GPT-2 regex — this is a bug")
    })
}

/// Build word-frequency table from raw text.
///
/// Returns: `HashMap<Vec<u32>, u64>` — token sequence → count.
/// Each word is a sequence of byte token IDs (b + BYTE_OFFSET).
pub fn build_word_freqs(text: &str) -> HashMap<Vec<u32>, u64> {
    let re = gpt2_regex();
    let mut freqs: HashMap<Vec<u32>, u64> = HashMap::new();

    // Split on document separator first — don't pretokenize across doc boundaries.
    // The separator itself is not text, so we skip it.
    for chunk in text.split("<|endoftext|>") {
        for m in re.find_iter(chunk).flatten() {
            let word = m.as_str().as_bytes();
            if word.is_empty() {
                continue;
            }
            // Each byte → byte token ID
            let token_ids: Vec<u32> = word.iter().map(|&b| b as u32 + BYTE_OFFSET).collect();
            *freqs.entry(token_ids).or_insert(0) += 1;
        }
    }

    freqs
}
