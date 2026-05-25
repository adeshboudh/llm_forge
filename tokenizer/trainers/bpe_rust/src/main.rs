//! bpe-trainer — fast BPE tokenizer trainer for llm-forge.
//!
//! Reads raw text from stdin or --input file.
//! Writes tokenizer.json (+ vocab.json, merges.txt) compatible with load.py.
//!
//! Usage:
//!
//!   # Pipe from Python downloader
//!   python tokenizer/trainers/download_corpus.py | \
//!       ./target/release/bpe-trainer --output-dir tokenizer/saved/
//!
//!   # From file
//!   ./target/release/bpe-trainer \
//!       --input /tmp/corpus.txt \
//!       --vocab-size 32768 \
//!       --output-dir tokenizer/saved/
//!
//! Output is identical to Python train_tokenizer.py. load.py loads both.

mod bpe;
mod pretokenize;
mod serialize;

use clap::Parser;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Fast BPE tokenizer trainer.
/// Reads text from stdin (pipe) or --input file. Outputs load.py-compatible tokenizer.json.
#[derive(Parser)]
#[command(name = "bpe-trainer", version, about)]
struct Args {
    /// Vocabulary size. Default: 32768 (2^15, uint16-safe). Must be > 261.
    #[arg(long, default_value_t = 32_768)]
    vocab_size: usize,

    /// Input text file. Reads from stdin if omitted.
    #[arg(long, short)]
    input: Option<PathBuf>,

    /// Output directory for tokenizer.json, vocab.json, merges.txt.
    #[arg(long, short, default_value = "tokenizer/saved")]
    output_dir: PathBuf,

    /// Print merge progress every N merges (0 = silent).
    #[arg(long, default_value_t = 1_000)]
    log_every: usize,
}

fn main() {
    let args = Args::parse();

    assert!(
        args.vocab_size > 261,
        "vocab_size must be > 261 (5 specials + 256 bytes + at least 1 merge)"
    );

    // --- Header ---
    println!("================================================================");
    println!("BPE Tokenizer Training (Rust)");
    println!("================================================================");
    println!("  vocab_size : {}", args.vocab_size);
    println!("  n_merges   : {}", args.vocab_size - 261);
    println!("  output_dir : {}", args.output_dir.display());
    println!(
        "  input      : {}",
        args.input
            .as_deref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "stdin".to_string())
    );
    println!("================================================================");

    let t_start = Instant::now();

    // --- Read input ---
    println!("\nReading text...");
    let t0 = Instant::now();
    let text = read_input(args.input.as_deref()).expect("Failed to read input");
    println!(
        "  {:.2}GB chars read in {:.1}s",
        text.len() as f64 / 1e9,
        t0.elapsed().as_secs_f32()
    );

    // --- Pretokenize ---
    println!("\nPretokenizing (GPT-2 regex)...");
    let t1 = Instant::now();
    let word_freqs = pretokenize::build_word_freqs(&text);
    println!(
        "  {} unique pre-tokens in {:.1}s",
        word_freqs.len(),
        t1.elapsed().as_secs_f32()
    );

    // --- BPE training ---
    let n_merges = args.vocab_size - 261;
    println!("\nLearning {} BPE merges...", n_merges);
    let t2 = Instant::now();
    let mut trainer = bpe::BpeTrainer::new(args.vocab_size, args.log_every);
    trainer.train(word_freqs);
    println!(
        "  {} merges learned in {:.1}s ({:.1} min)",
        trainer.merges.len(),
        t2.elapsed().as_secs_f32(),
        t2.elapsed().as_secs_f32() / 60.0
    );

    // --- Serialize ---
    println!("\nWriting output to {}...", args.output_dir.display());
    fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");
    serialize::save(&trainer, &args.output_dir).expect("Failed to write tokenizer files");
    println!("  tokenizer.json  (canonical — load with load.py)");
    println!("  vocab.json      (human inspection)");
    println!("  merges.txt      (human inspection)");

    let elapsed = t_start.elapsed();
    println!("\n================================================================");
    println!(
        "Done in {:.1}s ({:.1} min)",
        elapsed.as_secs_f32(),
        elapsed.as_secs_f32() / 60.0
    );
    println!("================================================================");
}

// ---------------------------------------------------------------------------
// Input reader
// ---------------------------------------------------------------------------

fn read_input(path: Option<&Path>) -> io::Result<String> {
    match path {
        Some(p) => fs::read_to_string(p),
        None => {
            // Read all stdin into string
            let mut text = String::new();
            io::stdin().lock().read_to_string(&mut text)?;
            Ok(text)
        }
    }
}
