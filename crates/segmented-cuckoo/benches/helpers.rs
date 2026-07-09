//! Shared bench helpers for the `segmented-cuckoo` benches.
//!
//! # Purpose
//!
//! Two utilities every bench in this crate needs: a writer for the
//! `results/*.csv` files this crate's benches emit, and a `Stats` /
//! `compute_stats` pair for summarising trial samples.
//!
//! # Related files
//!
//! - `benches/insert_throughput.rs` and
//!   `benches/kv_store_insert_throughput.rs` — sole consumers.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Create a CSV writer at `${IKPIR_RESULTS_DIR:-results}/{path}` and emit
/// `header` as the first line (truncating any existing file).
///
/// # Arguments
///
/// - `path`   — file name relative to the results directory (parent dirs are
///   created if missing).
/// - `header` — comma-separated column header line.
///
/// # Returns
///
/// A `BufWriter<File>` positioned just after the header line; the caller
/// appends one data row per `writeln!` call. `scripts/bench.sh` points
/// `IKPIR_RESULTS_DIR` at `results/segmented-cuckoo`; a bare `cargo bench`
/// falls back to the crate-local `results/` directory.
pub fn csv_writer(path: &str, header: &str) -> BufWriter<fs::File> {
    let base = std::env::var("IKPIR_RESULTS_DIR").unwrap_or_else(|_| "results".to_string());
    let full_path = format!("{}/{}", base, path);
    if let Some(parent) = Path::new(&full_path).parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let file = fs::File::create(&full_path).unwrap();
    let mut w = BufWriter::new(file);
    writeln!(w, "{}", header).unwrap();
    w
}

/// Aggregated statistics over a sample of `f64` values.
///
/// Returned by [`compute_stats`]; the four fields match the CSV columns
/// the benches emit (`mean`, `min`, `max`, `stddev`).
#[allow(dead_code)]
pub struct Stats {
    /// Arithmetic mean of the sample.
    pub mean: f64,
    /// Smallest observed value.
    pub min: f64,
    /// Largest observed value.
    pub max: f64,
    /// Population standard deviation (`n` divisor, not `n-1`).
    pub stddev: f64,
}

/// Compute `(mean, min, max, population stddev)` over `values`.
///
/// # Arguments
///
/// - `values` — non-empty slice of trial samples.
///
/// # Constraints
///
/// Panics if `values` is empty.
///
/// # Returns
///
/// A [`Stats`] with the four aggregates. Population stddev (divisor `n`)
/// is used so a single trial yields `stddev = 0` rather than `NaN`.
#[allow(dead_code)]
pub fn compute_stats(values: &[f64]) -> Stats {
    assert!(!values.is_empty(), "compute_stats: empty slice");
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    Stats {
        mean,
        min,
        max,
        stddev: variance.sqrt(),
    }
}

/// `true` unless started by `cargo bench` (the only invocation where cargo
/// passes `--bench`). Bench `main()`s early-return on this so
/// `cargo test --all-targets` builds them without running the full sweep.
#[allow(dead_code)]
pub fn skip_when_cargo_test() -> bool {
    !std::env::args().any(|a| a == "--bench")
}
