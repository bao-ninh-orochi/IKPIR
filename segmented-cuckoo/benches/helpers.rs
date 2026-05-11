use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Create a CSV writer at `results/{path}`, ensuring parent directories exist.
/// Writes the header as the first line.
pub fn csv_writer(path: &str, header: &str) -> BufWriter<fs::File> {
    let full_path = format!("results/{}", path);
    if let Some(parent) = Path::new(&full_path).parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let file = fs::File::create(&full_path).unwrap();
    let mut w = BufWriter::new(file);
    writeln!(w, "{}", header).unwrap();
    w
}

/// Aggregated statistics over a sample of f64 values.
#[allow(dead_code)]
pub struct Stats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

/// Compute mean, min, max, and population stddev over `values`.
///
/// # Panics
/// Panics if `values` is empty.
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
