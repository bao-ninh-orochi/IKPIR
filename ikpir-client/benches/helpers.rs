//! Shared bench helpers for ikpir-client benches.
//! duplicated deliberately — see CLAUDE.md (same content lives in ikpir-server/benches/helpers.rs)

use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use criterion::{Criterion, Throughput};

// ── CSV writer (append-aware) ───────────────────────────────────────────────

pub fn csv_writer(path: &str, header: &str) -> BufWriter<fs::File> {
    let full_path = Path::new("results").join(path);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let needs_header = !full_path.exists()
        || fs::metadata(&full_path).map(|m| m.len() == 0).unwrap_or(true);
    let mut w = BufWriter::new(
        OpenOptions::new().create(true).append(true).open(&full_path).unwrap(),
    );
    if needs_header {
        writeln!(w, "{header}").unwrap();
    }
    w
}

// ── Statistics ──────────────────────────────────────────────────────────────

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

// ── CLI parsing ─────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub fn parse_cli<C: clap::Parser>() -> C {
    let args: Vec<String> = std::env::args().filter(|a| a != "--bench").collect();
    let m = C::command().ignore_errors(true).get_matches_from(&args);
    C::from_arg_matches(&m).expect("parse CLI")
}

#[allow(dead_code)]
pub fn parse_cli_with_matches<C: clap::Parser>() -> (C, clap::ArgMatches) {
    let args: Vec<String> = std::env::args().filter(|a| a != "--bench").collect();
    let m = C::command().ignore_errors(true).get_matches_from(&args);
    let cli = C::from_arg_matches(&m).expect("parse CLI");
    (cli, m)
}

// ── Backend selection ───────────────────────────────────────────────────────

/// Index-PIR backend selector for benches.
///
/// `frodo` → [`ikpir_common::FrodoPirBackend`] (default; FrodoPIR-style
/// tall-skinny per-segment matrix, ternary errors, default `lwe_dim`
/// 1774). `simple` → [`ikpir_common::SimplePirBackend`] (SimplePIR-style
/// square reshape, discrete-Gaussian errors, default `lwe_dim` 1024).
#[allow(dead_code)]
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// FrodoPIR backend (tall-skinny matrix, ternary LWE).
    Frodo,
    /// SimplePIR backend (square reshape, discrete-Gaussian LWE).
    Simple,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Frodo  => write!(f, "frodo"),
            Backend::Simple => write!(f, "simple"),
        }
    }
}

/// Backend-appropriate default LWE dimension. FrodoPIR → 1774 (Table 5
/// 128-bit security). SimplePIR → 1024 (paper §4.2). Used by every
/// bench's `Cli` when `--lwe-dim` is omitted.
#[allow(dead_code)]
pub fn backend_default_lwe_dim(b: Backend) -> u32 {
    match b {
        Backend::Frodo  => 1774,
        Backend::Simple => 1024,
    }
}

// ── Default num_buckets per arity ────────────────────────────────────────────

#[allow(dead_code)]
pub fn default_num_buckets_for_arity(arity: u32) -> u32 {
    match arity {
        2 => 16_384,
        3 => 24_576,
        4 => 16_384,
        a => panic!("unsupported arity {a}; expected 2, 3, or 4"),
    }
}

// ── Preamble types and printer ───────────────────────────────────────────────

pub struct Knob {
    pub name: &'static str,
    pub value: String,
    pub is_default: bool,
}

pub struct StoreState {
    pub capacity: u64,
    pub populated: u64,
    pub load_pct: f64,
    pub cells_per_slot: u32,
    pub row_width: u32,
    pub segment_rows: u32,
}

pub struct Geometry {
    pub hint_per_seg_bytes: usize,
    pub setup_bundle_bytes: usize,
    pub query_bytes: usize,
    pub response_bytes: usize,
    pub hint_delta_typical_bytes: Option<usize>,
}

#[allow(dead_code)]
pub fn fmt_bytes(b: usize) -> String {
    if b >= 1_000_000 { format!("{:.1} MB", b as f64 / 1_000_000.0) }
    else if b >= 1_000 { format!("{:.0} KB", b as f64 / 1_000.0) }
    else { format!("{b} B") }
}

#[allow(dead_code)]
pub fn print_preamble(bench_name: &str, knobs: &[Knob], store: &StoreState, geom: &Geometry) {
    println!("=== {bench_name} ===");

    let prefix = "Parameters: ";
    let indent = " ".repeat(prefix.len());
    let mut line = prefix.to_string();
    let mut first = true;
    for k in knobs {
        let part = if k.is_default {
            format!("{}={} (default)", k.name, k.value)
        } else {
            format!("{}={}", k.name, k.value)
        };
        let sep = if first { "" } else { ", " };
        if !first && line.len() + sep.len() + part.len() > 100 {
            println!("{line},");
            line = format!("{indent}{part}");
        } else {
            line.push_str(sep);
            line.push_str(&part);
            first = false;
        }
    }
    println!("{line}");

    println!(
        "KV store:   capacity={} slots, populated={} keys ({:.1}% load),",
        store.capacity, store.populated, store.load_pct,
    );
    println!(
        "            cells_per_slot={}, row_width={}, segment_rows={}",
        store.cells_per_slot, store.row_width, store.segment_rows,
    );

    let mut geo = format!(
        "Geometry:   hint={}/seg, setup_bundle={}, query={}, response={}",
        fmt_bytes(geom.hint_per_seg_bytes),
        fmt_bytes(geom.setup_bundle_bytes),
        fmt_bytes(geom.query_bytes),
        fmt_bytes(geom.response_bytes),
    );
    if let Some(d) = geom.hint_delta_typical_bytes {
        geo.push_str(&format!(", hint_delta={}", fmt_bytes(d)));
    }
    println!("{geo}");
}

// ── Store population helpers ─────────────────────────────────────────────────

#[allow(dead_code)]
pub fn populate_until_full<S: MakeStore>(
    num_buckets: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
    value_bits: u32,
    plaintext_bits: u32,
) -> (segmented_cuckoo::CuckooKVStore<S>, u64) {
    let mut store = S::make_store(num_buckets, bucket_size, fingerprint_bits, value_bits, plaintext_bits)
        .expect("make_store");
    store.set_max_kicks(2_500);
    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut n_inserted = 0u64;
    let cap2 = (num_buckets as u64) * (bucket_size as u64) * 2;
    for k in 0u64..cap2 {
        let k32 = k as u32;
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k32.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k32.to_le_bytes(), &value) {
            Ok(()) => n_inserted += 1,
            Err(segmented_cuckoo::CuckooError::TableFull) => break,
            Err(e) => panic!("populate_until_full: {e:?}"),
        }
    }
    (store, n_inserted)
}

#[allow(dead_code)]
pub fn populate_to_load<S: MakeStore>(
    load_factor: f64,
    num_buckets: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
    value_bits: u32,
    plaintext_bits: u32,
) -> (segmented_cuckoo::CuckooKVStore<S>, u64) {
    let capacity = (num_buckets as u64) * (bucket_size as u64);
    let target = (capacity as f64 * load_factor).floor() as u64;
    let mut store = S::make_store(num_buckets, bucket_size, fingerprint_bits, value_bits, plaintext_bits)
        .expect("make_store");
    store.set_max_kicks(2_500);
    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut n_inserted = 0u64;
    for k in 0u64.. {
        if n_inserted >= target { break; }
        let k32 = k as u32;
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k32.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k32.to_le_bytes(), &value) {
            Ok(()) => n_inserted += 1,
            Err(segmented_cuckoo::CuckooError::TableFull) =>
                panic!("populate_to_load: TableFull before reaching {load_factor:.2} load"),
            Err(e) => panic!("populate_to_load: {e:?}"),
        }
    }
    (store, n_inserted)
}

// ── Criterion throughput helper ──────────────────────────────────────────────

#[allow(dead_code)]
pub struct CriterionThroughputStats {
    pub mean_ops_per_s: f64,
    pub min_ops_per_s: f64,
    pub max_ops_per_s: f64,
    pub stddev_ops_per_s: f64,
}

#[allow(dead_code)]
pub fn run_criterion_throughput<F>(
    bench_label: &str,
    elements_per_iter: u64,
    mut body: F,
) -> CriterionThroughputStats
where
    F: FnMut(),
{
    let samples: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_inner = samples.clone();
    {
        let mut c = Criterion::default();
        let mut group = c.benchmark_group(bench_label);
        group.throughput(Throughput::Elements(elements_per_iter));
        group.bench_function(bench_label, |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    body();
                }
                let elapsed = start.elapsed();
                let ns_per_iter = elapsed.as_nanos() as f64 / iters as f64;
                samples_inner.lock().unwrap().push(ns_per_iter);
                elapsed
            });
        });
        group.finish();
    }
    let raw: Vec<f64> = std::mem::take(&mut *samples.lock().unwrap());
    if raw.is_empty() {
        return CriterionThroughputStats { mean_ops_per_s: 0.0, min_ops_per_s: 0.0,
                                          max_ops_per_s: 0.0, stddev_ops_per_s: 0.0 };
    }
    let ops: Vec<f64> = raw.iter().map(|&ns| 1e9 / ns).collect();
    let s = compute_stats(&ops);
    CriterionThroughputStats {
        mean_ops_per_s: s.mean,
        min_ops_per_s: s.min,
        max_ops_per_s: s.max,
        stddev_ops_per_s: s.stddev,
    }
}

/// Like `run_criterion_throughput`, but the body receives a freshly-prepared
/// input per call. Mirrors `criterion::Bencher::iter_batched_ref` but uses
/// `iter_custom` under the hood so setup time is excluded from the timing
/// bracket (the per-call `Instant` brackets only `routine`).
///
/// `setup` runs once per `routine` call; both are reused across criterion's
/// sampling loop. `elements_per_iter` feeds `Throughput::Elements` for
/// criterion's native HTML/JSON report.
#[allow(dead_code)]
pub fn run_criterion_throughput_batched<S, I, R>(
    bench_label: &str,
    elements_per_iter: u64,
    mut setup: S,
    mut routine: R,
) -> CriterionThroughputStats
where
    S: FnMut() -> I,
    R: FnMut(&mut I),
{
    let samples: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_inner = samples.clone();
    {
        let mut c = Criterion::default();
        let mut group = c.benchmark_group(bench_label);
        group.throughput(Throughput::Elements(elements_per_iter));
        group.bench_function(bench_label, |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut input = setup();
                    let t = Instant::now();
                    routine(&mut input);
                    total += t.elapsed();
                }
                let ns_per_iter = total.as_nanos() as f64 / iters as f64;
                samples_inner.lock().unwrap().push(ns_per_iter);
                total
            });
        });
        group.finish();
    }
    let raw: Vec<f64> = std::mem::take(&mut *samples.lock().unwrap());
    if raw.is_empty() {
        return CriterionThroughputStats { mean_ops_per_s: 0.0, min_ops_per_s: 0.0,
                                          max_ops_per_s: 0.0, stddev_ops_per_s: 0.0 };
    }
    let ops: Vec<f64> = raw.iter().map(|&ns| 1e9 / ns).collect();
    let s = compute_stats(&ops);
    CriterionThroughputStats {
        mean_ops_per_s: s.mean,
        min_ops_per_s: s.min,
        max_ops_per_s: s.max,
        stddev_ops_per_s: s.stddev,
    }
}

// ── Generic store construction ────────────────────────────────────────────────

#[allow(dead_code)]
pub trait MakeStore:
    segmented_cuckoo::IndexScheme + segmented_cuckoo::SchemeMeta + Sized + 'static
{
    fn make_store(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>;
}

impl MakeStore for segmented_cuckoo::Segmented2aryScheme {
    fn make_store(n: u32, bs: u32, fp: u32, vb: u32, pb: u32)
        -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>
    {
        segmented_cuckoo::CuckooKVStore::<segmented_cuckoo::Segmented2aryScheme>::new(n, bs, fp, vb, pb)
    }
}
impl MakeStore for segmented_cuckoo::Segmented3aryScheme {
    fn make_store(n: u32, bs: u32, fp: u32, vb: u32, pb: u32)
        -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>
    {
        segmented_cuckoo::CuckooKVStore::<segmented_cuckoo::Segmented3aryScheme>::new(n, bs, fp, vb, pb)
    }
}
impl MakeStore for segmented_cuckoo::Segmented4aryScheme {
    fn make_store(n: u32, bs: u32, fp: u32, vb: u32, pb: u32)
        -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>
    {
        segmented_cuckoo::CuckooKVStore::<segmented_cuckoo::Segmented4aryScheme>::new(n, bs, fp, vb, pb)
    }
}

#[allow(dead_code)]
pub trait CloneStore: MakeStore {
    fn clone_from_cells(
        cells: Vec<u32>,
        params: segmented_cuckoo::CuckooParams,
        num_items: u64,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>;
}
impl CloneStore for segmented_cuckoo::Segmented2aryScheme {
    fn clone_from_cells(c: Vec<u32>, p: segmented_cuckoo::CuckooParams, n: u64)
        -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>
    {
        segmented_cuckoo::CuckooKVStore::<segmented_cuckoo::Segmented2aryScheme>::from_cells(c, p, n)
    }
}
impl CloneStore for segmented_cuckoo::Segmented3aryScheme {
    fn clone_from_cells(c: Vec<u32>, p: segmented_cuckoo::CuckooParams, n: u64)
        -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>
    {
        segmented_cuckoo::CuckooKVStore::<segmented_cuckoo::Segmented3aryScheme>::from_cells(c, p, n)
    }
}
impl CloneStore for segmented_cuckoo::Segmented4aryScheme {
    fn clone_from_cells(c: Vec<u32>, p: segmented_cuckoo::CuckooParams, n: u64)
        -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>
    {
        segmented_cuckoo::CuckooKVStore::<segmented_cuckoo::Segmented4aryScheme>::from_cells(c, p, n)
    }
}
