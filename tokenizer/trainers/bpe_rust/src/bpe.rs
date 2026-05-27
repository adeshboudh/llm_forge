//! BPE training core — parallel version using rayon.
//!
//! Algorithm (identical to Python bpe.py, just ~50x faster single-thread,
//! ~200-400x faster with 8 cores via rayon):
//!
//!   1. Count all adjacent token pairs (parallel fold+reduce across all words).
//!   2. Find most-frequent pair. Tie-break: lexicographic for determinism.
//!   3. Assign new ID (261, 262, ...). Apply merge (parallel map over all words).
//!   4. Repeat for n_merges = vocab_size - 261 steps.
//!
//! Parallelism:
//!   count_pairs  — rayon par_iter fold+reduce → no lock contention, perfect scaling
//!   apply_merge  — rayon par_iter map (merge_sequence) → sequential coalesce
//!
//! Data structures:
//!   word_freqs  : HashMap<Vec<u32>, u64>   — word (token seq) → frequency
//!   pair_counts : HashMap<(u32, u32), u64> — pair → total count across all words

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

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

        let pb = ProgressBar::new(n_merges as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] \
                     [{bar:45.cyan/blue}] {pos:>6}/{len} merges  ETA {eta_precise}",
                )
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.enable_steady_tick(Duration::from_millis(200));

        for i in 0..n_merges {
            // Parallel pair count
            let pair_counts = count_pairs_parallel(&word_freqs);

            if pair_counts.is_empty() {
                pb.abandon_with_message(format!("No more pairs at step {i}. Stopped early."));
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

            // Parallel merge application
            apply_merge_parallel(&mut word_freqs, best_pair, new_id);
            self.merges.push(best_pair);
            pb.inc(1);

            // Print occasional merge details above the bar (doesn't corrupt it)
            if self.log_every > 0 && (i + 1) % self.log_every == 0 {
                let (a, b) = best_pair;
                pb.println(format!(
                    "  [{:>6}/{:>6}]  ({}, {}) -> {}",
                    i + 1, n_merges, a, b, new_id,
                ));
            }
        }

        pb.finish_with_message(format!("{} merges done", self.merges.len()));
    }
}

// ---------------------------------------------------------------------------
// Parallel helpers
// ---------------------------------------------------------------------------

/// Count all adjacent token pairs, weighted by word frequency.
///
/// Uses rayon fold+reduce:
///   - Each thread builds its own local HashMap (no contention)
///   - Reduce merges thread-local maps into one final map
///
/// Scales linearly with core count up to ~unique_words / cache_line contention.
fn count_pairs_parallel(word_freqs: &HashMap<Vec<u32>, u64>) -> HashMap<(u32, u32), u64> {
    word_freqs
        .par_iter()
        .fold(
            HashMap::new,
            |mut local, (tokens, &freq)| {
                for window in tokens.windows(2) {
                    *local.entry((window[0], window[1])).or_insert(0) += freq;
                }
                local
            },
        )
        .reduce(
            HashMap::new,
            |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            },
        )
}

/// Apply merge: replace all occurrences of `pair` in each word with `new_id`.
///
/// Parallel map: each word's merge_sequence runs on a separate thread.
/// Sequential coalesce: rebuild HashMap (words may converge post-merge).
fn apply_merge_parallel(
    word_freqs: &mut HashMap<Vec<u32>, u64>,
    pair: (u32, u32),
    new_id: u32,
) {
    // Drain into Vec so rayon can index into it
    let old: Vec<(Vec<u32>, u64)> = word_freqs.drain().collect();

    // Parallel merge step — merge_sequence is pure, perfect for par_iter
    let merged: Vec<(Vec<u32>, u64)> = old
        .into_par_iter()
        .map(|(tokens, freq)| (merge_sequence(&tokens, pair, new_id), freq))
        .collect();

    // Sequential coalesce: two words can converge to same sequence after merge
    for (seq, freq) in merged {
        *word_freqs.entry(seq).or_insert(0) += freq;
    }
}

/// Replace all non-overlapping occurrences of `pair` in `tokens` with `new_id`.
/// Left-to-right scan; skips 2 positions after match (no overlapping merges).
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
