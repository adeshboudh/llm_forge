//! bpe-trainer — fast BPE tokenizer trainer for llm-forge.
//!
//! Reads raw text from stdin or --input file.
//! Writes tokenizer.json (+ vocab.json, merges.txt) compatible with load.py.
//!
//! Usage:
//!
//!   # Pipe from Python downloader
//!   python3 tokenizer/trainers/download_corpus.py | \
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
    eprintln!("================================================================");
    eprintln!("BPE Tokenizer Training (Rust)");
    eprintln!("================================================================");
    eprintln!("  vocab_size : {}", args.vocab_size);
    eprintln!("  n_merges   : {}", args.vocab_size - 261);
    eprintln!("  output_dir : {}", args.output_dir.display());
    eprintln!(
        "  input      : {}",
        args.input
            .as_deref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "stdin".to_string())
    );
    eprintln!("================================================================");

    let t_start = Instant::now();

    // --- Read input ---
    eprintln!("\nReading text...");
    let t0 = Instant::now();
    let text = read_input_with_progress(args.input.as_deref()).expect("Failed to read input");
    eprintln!(
        "  {:.3}B chars read in {:.1}s",
        text.len() as f64 / 1e9,
        t0.elapsed().as_secs_f32()
    );

    // --- Pretokenize ---
    eprintln!("\nPretokenizing (GPT-2 regex)...");
    let t1 = Instant::now();
    let word_freqs = pretokenize::build_word_freqs(&text);
    eprintln!(
        "  {} unique pre-tokens in {:.1}s",
        word_freqs.len(),
        t1.elapsed().as_secs_f32()
    );

    // --- BPE training ---
    let n_merges = args.vocab_size - 261;
    eprintln!("\nLearning {} BPE merges...", n_merges);
    let t2 = Instant::now();
    let mut trainer = bpe::BpeTrainer::new(args.vocab_size, args.log_every);
    trainer.train(word_freqs, t2);
    eprintln!(
        "  {} merges learned in {:.1}s ({:.1} min)",
        trainer.merges.len(),
        t2.elapsed().as_secs_f32(),
        t2.elapsed().as_secs_f32() / 60.0
    );

    // --- Serialize ---
    eprintln!("\nWriting output to {}...", args.output_dir.display());
    fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");
    serialize::save(&trainer, &args.output_dir).expect("Failed to write tokenizer files");
    eprintln!("  tokenizer.json  (canonical — load with load.py)");
    eprintln!("  vocab.json      (human inspection)");
    eprintln!("  merges.txt      (human inspection)");

    let elapsed = t_start.elapsed();
    eprintln!("\n================================================================");
    eprintln!(
        "Done in {:.1}s ({:.1} min)",
        elapsed.as_secs_f32(),
        elapsed.as_secs_f32() / 60.0
    );
    eprintln!("================================================================");
}

// ---------------------------------------------------------------------------
// Input reader with progress
// ---------------------------------------------------------------------------

/// Read from file or stdin, printing bytes received to stderr every 50MB.
fn read_input_with_progress(path: Option<&Path>) -> io::Result<String> {
    const CHUNK: usize = 1 << 20; // 1MB per read
    const REPORT_EVERY: usize = 50 * 1024 * 1024; // report every 50MB

    let mut buf = vec![0u8; CHUNK];
    let mut bytes: Vec<u8> = Vec::with_capacity(1 << 30); // pre-alloc 1GB

    let mut source: Box<dyn Read> = match path {
        Some(p) => Box::new(fs::File::open(p)?),
        None    => Box::new(io::stdin()),
    };

    let mut last_report = 0usize;

    loop {
        let n = source.read(&mut buf)?;
        if n == 0 { break; }
        bytes.extend_from_slice(&buf[..n]);

        if bytes.len() - last_report >= REPORT_EVERY {
            last_report = bytes.len();
            eprint!("\r  {:.3}B chars received...", bytes.len() as f64 / 1e9);
            // flush stderr so progress shows immediately
            use std::io::Write;
            let _ = io::stderr().flush();
        }
    }

    if last_report > 0 {
        eprintln!(); // newline after \r progress line
    }

    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
