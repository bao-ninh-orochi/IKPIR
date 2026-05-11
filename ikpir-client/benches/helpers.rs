//! Shared bench helpers — same shape as `segmented-cuckoo/benches/helpers.rs`.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Create a CSV writer at `results/{path}`, ensuring parent directories exist.
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

#[allow(dead_code)]
pub struct Stats {
    pub mean:   f64,
    pub min:    f64,
    pub max:    f64,
    pub stddev: f64,
}

#[allow(dead_code)]
pub fn compute_stats(values: &[f64]) -> Stats {
    assert!(!values.is_empty(), "compute_stats: empty slice");
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    Stats { mean, min, max, stddev: variance.sqrt() }
}

/// Parse a clap-derive `Cli` from `std::env::args()`, filtering out the
/// `--bench` flag that `cargo bench` injects automatically.
#[allow(dead_code)]
pub fn parse_cli<C: clap::Parser>() -> C {
    C::parse_from(std::env::args().filter(|a| a != "--bench"))
}

/// Generic constructor bridge — `CuckooKVStore::new` has per-scheme impls,
/// not a generic one, so bench bodies that are generic over `S` can't call
/// `CuckooKVStore::<S>::new(..)` directly. This trait routes back to the
/// concrete per-scheme constructor.
#[allow(dead_code)]
pub trait MakeStore: segmented_cuckoo::IndexScheme + segmented_cuckoo::SchemeMeta + Sized + 'static {
    fn make_store(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>;
}

impl MakeStore for segmented_cuckoo::Segmented2aryScheme {
    fn make_store(num_buckets: u32, bucket_size: u32, fingerprint_bits: u32, value_bits: u32, plaintext_bits: u32) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError> {
        segmented_cuckoo::CuckooKVStore::<segmented_cuckoo::Segmented2aryScheme>::new(num_buckets, bucket_size, fingerprint_bits, value_bits, plaintext_bits)
    }
}
impl MakeStore for segmented_cuckoo::Segmented3aryScheme {
    fn make_store(num_buckets: u32, bucket_size: u32, fingerprint_bits: u32, value_bits: u32, plaintext_bits: u32) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError> {
        segmented_cuckoo::CuckooKVStore::<segmented_cuckoo::Segmented3aryScheme>::new(num_buckets, bucket_size, fingerprint_bits, value_bits, plaintext_bits)
    }
}
impl MakeStore for segmented_cuckoo::Segmented4aryScheme {
    fn make_store(num_buckets: u32, bucket_size: u32, fingerprint_bits: u32, value_bits: u32, plaintext_bits: u32) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError> {
        segmented_cuckoo::CuckooKVStore::<segmented_cuckoo::Segmented4aryScheme>::new(num_buckets, bucket_size, fingerprint_bits, value_bits, plaintext_bits)
    }
}
