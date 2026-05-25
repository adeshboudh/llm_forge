//! BPE training core.
//!
//! Algorithm (identical to Python bpe.py, just ~50x faster):
//!
//!   1. Count all adjacent token pairs across all words (weighted by word freq).
//!   2. Find most-frequent pair. Tie-break: lexicographic on (a, b) for determinism.
//!   3. Assign new ID (261, 262, ...). Apply merge to every word.
//!   4. Repeat for n_merges = vocab_size - 261 steps.
//!
//! Data structures:
//!   word_freqs  : HashMap<Vec<u32>, u64>   — word (token seq) → frequency
//!   pair_counts : HashMap<(u32, u32), u64> — pair → total count across all words
//!
//! The `apply_merge` step rebuilds word_freqs from scratch each iteration.
//! This is O(V) per merge (V = unique word count). With V ≈ 1.5M and 32,507
//! merges, this is ~49B operations — fast in Rust (~5–20 minutes), slow in
//! Python (~8–24 hours).

use std::collections::HashMap;

pub struct BpeTrainer {
    pub vocab_size: usize,
    /// Ordered merge history: [(token_a, token_b), ...].
    /// Merge i produces ID (261 + i). Written to tokenizer.json.
    pub merges: Vec<(u32, u32)>,
    log_every: usize,
    next_id: u32,
}

impl BpeTrainer {
    pub fn new(vocab_size: usize, log_every: usize) -> Self {
        use crate::pretokenize::MERGE_START_ID;
        Self {
            vocab_size,
            merges: Vec::with_capacity(vocab_size - MERGE_START_ID as usize),
            log_every,
            next_id: MERGE_START_ID,
        }
    }

    /// Train BPE on a word-frequency table (consumed in-place).
    pub fn train(&mut self, mut word_freqs: HashMap<Vec<u32>, u64>) {
        let n_merges = self.vocab_size.saturating_sub(self.next_id as usize);

        for i in 0..n_merges {
            let pair_counts = count_pairs(&word_freqs);

            if pair_counts.is_empty() {
                eprintln!("  No more pairs at step {}. Stopping early.", i);
                break;
            }

            // Best pair: highest count, tie-break lexicographic (deterministic output)
            let best_pair = pair_counts
                .iter()
                .max_by(|(p1, &c1), (p2, &c2)| c1.cmp(&c2).then(p1.cmp(p2)))
                .map(|(&pair, _)| pair)
                .unwrap();

            let new_id = self.next_id;
            self.next_id += 1;
            apply_merge(&mut word_freqs, best_pair, new_id);
            self.merges.push(best_pair);

            if self.log_every > 0 && (i + 1) % self.log_every == 0 {
                let (a, b) = best_pair;
                eprintln!(
                    "  [{:>6}/{:>6}] merge ({}, {}) -> {}",
                    i + 1,
                    n_merges,
                    a,
                    b,
                    new_id
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count all adjacent token pairs across all words, weighted by word frequency.
fn count_pairs(word_freqs: &HashMap<Vec<u32>, u64>) -> HashMap<(u32, u32), u64> {
    let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
    for (tokens, &freq) in word_freqs {
        for window in tokens.windows(2) {
            *counts.entry((window[0], window[1])).or_insert(0) += freq;
        }
    }
    counts
}

/// Replace every non-overlapping occurrence of `pair` in each word with `new_id`.
/// Drains and rebuilds word_freqs so merged sequences can coalesce their counts.
fn apply_merge(word_freqs: &mut HashMap<Vec<u32>, u64>, pair: (u32, u32), new_id: u32) {
    let old: Vec<(Vec<u32>, u64)> = word_freqs.drain().collect();
    for (tokens, freq) in old {
        let merged = merge_sequence(&tokens, pair, new_id);
        *word_freqs.entry(merged).or_insert(0) += freq;
    }
}

/// Replace all non-overlapping occurrences of `pair` in `tokens` with `new_id`.
/// Scans left-to-right; after a match, skips 2 positions (no overlapping merges).
fn merge_sequence(tokens: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            result.push(new_id);
            i += 2;
        } else {
            result.push(tokens[i]);
            i += 1;
        }
    }
    result
}
