//! Shared bench helpers for ikpir-server benches.
//! duplicated deliberately — see CLAUDE.md (same content lives in ikpir-client/benches/helpers.rs)

// significant_drop_tightening: clippy's inline-`Criterion` fix borrows a temporary dropped while `BenchmarkGroup` holds it (won't compile).
#![allow(clippy::significant_drop_tightening)]

use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use criterion::{Criterion, Throughput};

// ── CSV writer (append-aware) ───────────────────────────────────────────────

/// Open a CSV writer at `${IKPIR_RESULTS_DIR:-results}/{path}` in append mode.
/// Writes the header only if the file is empty or does not yet exist.
/// `scripts/bench.sh` points `IKPIR_RESULTS_DIR` at `results/ikpir-server`;
/// a bare `cargo bench` falls back to the crate-local `results/` directory.
/// One invocation = one appended row; delete the CSV first for a clean file.
pub fn csv_writer(path: &str, header: &str) -> BufWriter<fs::File> {
    let base = std::env::var("IKPIR_RESULTS_DIR").unwrap_or_else(|_| "results".to_string());
    let full_path = Path::new(&base).join(path);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let needs_header = !full_path.exists()
        || fs::metadata(&full_path)
            .map(|m| m.len() == 0)
            .unwrap_or(true);
    let mut w = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&full_path)
            .unwrap(),
    );
    if needs_header {
        writeln!(w, "{header}").unwrap();
    }
    w
}

// ── Statistics ──────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct Stats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
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
    Stats {
        mean,
        min,
        max,
        stddev: variance.sqrt(),
    }
}

// ── CLI parsing ─────────────────────────────────────────────────────────────

/// Parse a clap-derive `Cli` from `std::env::args()`, filtering `--bench`
/// and ignoring unknown flags so criterion CLI args pass through silently.
#[allow(dead_code)]
pub fn parse_cli<C: clap::Parser>() -> C {
    let args: Vec<String> = std::env::args().filter(|a| a != "--bench").collect();
    let m = C::command().ignore_errors(true).get_matches_from(&args);
    C::from_arg_matches(&m).expect("parse CLI")
}

/// Like `parse_cli`, but also returns the raw `ArgMatches` so callers can
/// check which flags were set via `clap::parser::ValueSource::CommandLine`.
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
/// 1566). `simple` → [`ikpir_common::SimplePirBackend`] (SimplePIR-style
/// square reshape, discrete-Gaussian errors, default `lwe_dim` 1275).
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
            Self::Frodo => write!(f, "frodo"),
            Self::Simple => write!(f, "simple"),
        }
    }
}

/// Backend-appropriate default LWE dimension for 128-bit security,
/// estimated via the lattice estimator under the ADPS16 cost model.
/// FrodoPIR → 1566. SimplePIR → 1275. Used by every bench's `Cli` when
/// `--lwe-dim` is omitted.
#[allow(dead_code)]
pub const fn backend_default_lwe_dim(b: Backend) -> u32 {
    match b {
        Backend::Frodo => 1566,
        Backend::Simple => 1275,
    }
}

// ── Hint-patch mode selection ───────────────────────────────────────────────

/// Hint-patch realization selector for the mutation benches.
///
/// `entry` → [`ikpir_common::HintPatchMode::EntryLevel`] (the library
/// default; iSimplePIR per-cell patch, `Θ(n)` per touched cell).
/// `row` → [`ikpir_common::HintPatchMode::RowLevel`] (the SimplePIR
/// dense per-row baseline, `Θ(n·ω)` per touched row). Both realizations
/// produce identical state and identical delta wire bytes; the mutation
/// benches sweep them to isolate the patch-granularity cost — the two
/// mutation-phase columns of the paper's asymptotic table.
#[allow(dead_code)]
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchMode {
    /// Entry-level realization (iSimplePIR): patch only touched columns.
    Entry,
    /// Row-level realization (SimplePIR): dense rank-one update per row.
    Row,
}

impl std::fmt::Display for PatchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Entry => write!(f, "entry"),
            Self::Row => write!(f, "row"),
        }
    }
}

impl PatchMode {
    /// Library-level [`ikpir_common::HintPatchMode`] equivalent.
    #[allow(dead_code)]
    pub const fn to_hint_patch_mode(self) -> ikpir_common::HintPatchMode {
        match self {
            Self::Entry => ikpir_common::HintPatchMode::EntryLevel,
            Self::Row => ikpir_common::HintPatchMode::RowLevel,
        }
    }
}

/// Render a `--patch-mode` list for the bench preamble (e.g. `entry,row`).
#[allow(dead_code)]
pub fn patch_modes_label(modes: &[PatchMode]) -> String {
    modes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// Per-segment database matrix `D` shape `(rows, row_width)` for the
/// given backend, given the original `(n_rows_per_seg, row_width)`.
///
/// # Rationale
///
/// The head-to-head benches' OOM guard (and the `db_rows`/`db_cols` shape
/// columns every bench reports) need the per-segment `D` shape because every
/// other LWE-PIR buffer derives its dimensions from it:
/// - `A` is `rows × lwe_dim` — shares `D`'s row count, so `rows` drives
///   the `A` allocation and the per-slot `b`-vector length
///   (`b = A·s + e + Δ·u`, length `rows`).
/// - `H = Aᵀ · D` is `lwe_dim × row_width` — shares `D`'s row width,
///   so `row_width` drives the hint allocation and the prepared-slot
///   `c`-vector length (`c = sᵀ · H`, length `row_width`).
///
/// FrodoPIR keeps the original layout; SimplePIR reshapes into a
/// near-square `(reshape_rows × reshape_row_width)` matrix with
/// `k = max(1, round(√(n_rows_per_seg / row_width)))`. The formula
/// mirrors `reshape_dims` in `ikpir-common/src/backend/simple/backend.rs`.
///
/// # Returns
///
/// `(d_rows_per_seg, d_row_width_per_seg)`:
/// - FrodoPIR:  `(n_rows_per_seg, row_width)`
/// - SimplePIR: `(⌈n_rows_per_seg / k⌉, k · row_width)`
#[allow(dead_code)]
pub fn backend_shape_estimate(backend: Backend, n_rows_per_seg: u64, row_width: u64) -> (u64, u64) {
    match backend {
        Backend::Frodo => (n_rows_per_seg, row_width),
        Backend::Simple => {
            let k = ((n_rows_per_seg as f64 / row_width as f64)
                .sqrt()
                .round()
                .max(1.0)) as u64;
            let reshape_rows = n_rows_per_seg.div_ceil(k);
            let reshape_row_width = k * row_width;
            (reshape_rows, reshape_row_width)
        }
    }
}

// ── Default num_buckets per arity ────────────────────────────────────────────

/// Default `num_buckets` for a one-off run: **dev scale** (≈2^16 slots),
/// sub-second per bench.
/// 2-ary: 2^14 buckets × bucket_size=4 → 65536 slots.
/// 3-ary: 3·2^13 buckets × bucket_size=4 → 98304 slots.
/// 4-ary: 2^14 buckets × bucket_size=4 → 65536 slots.
///
/// This is **not** the paper's geometry, which is ~2^20 slots and lives in
/// `scripts/lib.sh::paper_num_buckets` — the `table{3,4,5}.sh` sweeps pass it
/// explicitly. Mirrors `default_num_buckets` in that file.
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
    if b >= 1_000_000 {
        format!("{:.1} MB", b as f64 / 1_000_000.0)
    } else if b >= 1_000 {
        format!("{:.0} KB", b as f64 / 1_000.0)
    } else {
        format!("{b} B")
    }
}

#[allow(dead_code)]
pub fn print_preamble(bench_name: &str, knobs: &[Knob], store: &StoreState, geom: &Geometry) {
    println!("=== {bench_name} ===");

    // Parameters: wrap at ~100 chars with fixed indent
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

/// Populate a fresh store by inserting keys `0,1,2,…` (4-byte LE, value salt 17)
/// until the first `TableFull`. Returns (store, n_inserted).
#[allow(dead_code)]
pub fn populate_until_full<S: MakeStore>(
    num_buckets: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
    value_bits: u32,
    plaintext_bits: u32,
) -> (segmented_cuckoo::CuckooKVStore<S>, u64) {
    let mut store = S::make_store(
        num_buckets,
        bucket_size,
        fingerprint_bits,
        value_bits,
        plaintext_bits,
    )
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

/// Populate a fresh store with **exactly** `target_n` items, inserting keys
/// `0,1,2,…` until the count is reached. Panics if `target_n` exceeds capacity
/// or if cuckoo eviction fails before the target is reached. Used by the
/// head-to-head bench (fixed N, schemes report their own DB size).
#[allow(dead_code)]
pub fn populate_exact_n_keys<S: MakeStore>(
    num_buckets: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
    value_bits: u32,
    plaintext_bits: u32,
    target_n: u64,
) -> (segmented_cuckoo::CuckooKVStore<S>, u64) {
    let capacity = (num_buckets as u64) * (bucket_size as u64);
    assert!(
        target_n <= capacity,
        "populate_exact_n_keys: target_n={target_n} > capacity={capacity} \
         (num_buckets={num_buckets}, bucket_size={bucket_size})"
    );
    let mut store = S::make_store(
        num_buckets,
        bucket_size,
        fingerprint_bits,
        value_bits,
        plaintext_bits,
    )
    .expect("make_store");
    store.set_max_kicks(2_500);
    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut n_inserted = 0u64;
    let cap2 = capacity.saturating_mul(2);
    for k in 0u64..cap2 {
        if n_inserted >= target_n {
            break;
        }
        let k32 = k as u32;
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k32.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k32.to_le_bytes(), &value) {
            Ok(()) => n_inserted += 1,
            Err(segmented_cuckoo::CuckooError::TableFull) => panic!(
                "populate_exact_n_keys: TableFull at {n_inserted}/{target_n} \
                        (capacity={capacity}); raise num_buckets or bucket_size"
            ),
            Err(e) => panic!("populate_exact_n_keys: {e:?} at {n_inserted}/{target_n}"),
        }
    }
    assert!(
        n_inserted == target_n,
        "populate_exact_n_keys: inserted {n_inserted}/{target_n} (cuckoo eviction limit)"
    );
    (store, n_inserted)
}

/// Populate a fresh store to `floor(capacity * load_factor)` items.
/// Panics if cuckoo kicks exhaust before reaching the target load.
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
    let mut store = S::make_store(
        num_buckets,
        bucket_size,
        fingerprint_bits,
        value_bits,
        plaintext_bits,
    )
    .expect("make_store");
    store.set_max_kicks(2_500);
    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut n_inserted = 0u64;
    for k in 0u64.. {
        if n_inserted >= target {
            break;
        }
        let k32 = k as u32;
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k32.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k32.to_le_bytes(), &value) {
            Ok(()) => n_inserted += 1,
            Err(segmented_cuckoo::CuckooError::TableFull) => {
                panic!("populate_to_load: TableFull before reaching {load_factor:.2} load")
            }
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

/// Run one criterion throughput benchmark, capturing per-sample timing.
///
/// Uses `iter_custom` so the body can cycle over pre-built data.
/// Criterion's native HTML/JSON report lands in `target/criterion/<bench_label>/`.
///
/// `elements_per_iter` only labels criterion's `Throughput::Elements` for its
/// HTML/JSON report. It does **not** scale the returned
/// `CriterionThroughputStats`, which always counts one body call per sample
/// (`mean_ops_per_s = 1e9 / mean_ns_per_body_call`). Pass `1` when each body
/// call executes one operation.
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
                samples.lock().unwrap().push(ns_per_iter);
                elapsed
            });
        });
        group.finish();
    }
    let raw: Vec<f64> = std::mem::take(&mut *samples.lock().unwrap());
    if raw.is_empty() {
        return CriterionThroughputStats {
            mean_ops_per_s: 0.0,
            min_ops_per_s: 0.0,
            max_ops_per_s: 0.0,
            stddev_ops_per_s: 0.0,
        };
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
/// sampling loop. `elements_per_iter` only labels criterion's
/// `Throughput::Elements` for its HTML/JSON report; it does **not** scale the
/// returned `CriterionThroughputStats` (always `1e9 / mean_ns_per_routine_call`).
/// Pass `1` when each `routine` call executes one operation.
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
                samples.lock().unwrap().push(ns_per_iter);
                total
            });
        });
        group.finish();
    }
    let raw: Vec<f64> = std::mem::take(&mut *samples.lock().unwrap());
    if raw.is_empty() {
        return CriterionThroughputStats {
            mean_ops_per_s: 0.0,
            min_ops_per_s: 0.0,
            max_ops_per_s: 0.0,
            stddev_ops_per_s: 0.0,
        };
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
    fn make_store(
        n: u32,
        bs: u32,
        fp: u32,
        vb: u32,
        pb: u32,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError> {
        segmented_cuckoo::CuckooKVStore::<Self>::new(n, bs, fp, vb, pb)
    }
}
impl MakeStore for segmented_cuckoo::Segmented3aryScheme {
    fn make_store(
        n: u32,
        bs: u32,
        fp: u32,
        vb: u32,
        pb: u32,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError> {
        segmented_cuckoo::CuckooKVStore::<Self>::new(n, bs, fp, vb, pb)
    }
}
impl MakeStore for segmented_cuckoo::Segmented4aryScheme {
    fn make_store(
        n: u32,
        bs: u32,
        fp: u32,
        vb: u32,
        pb: u32,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError> {
        segmented_cuckoo::CuckooKVStore::<Self>::new(n, bs, fp, vb, pb)
    }
}

/// Generic `from_cells` bridge (per-scheme associated fn, not a blanket impl).
#[allow(dead_code)]
pub trait CloneStore: MakeStore {
    fn clone_from_cells(
        cells: Vec<u32>,
        params: segmented_cuckoo::CuckooParams,
        num_items: u64,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError>;
}
impl CloneStore for segmented_cuckoo::Segmented2aryScheme {
    fn clone_from_cells(
        c: Vec<u32>,
        p: segmented_cuckoo::CuckooParams,
        n: u64,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError> {
        segmented_cuckoo::CuckooKVStore::<Self>::from_cells(c, p, n)
    }
}
impl CloneStore for segmented_cuckoo::Segmented3aryScheme {
    fn clone_from_cells(
        c: Vec<u32>,
        p: segmented_cuckoo::CuckooParams,
        n: u64,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError> {
        segmented_cuckoo::CuckooKVStore::<Self>::from_cells(c, p, n)
    }
}
impl CloneStore for segmented_cuckoo::Segmented4aryScheme {
    fn clone_from_cells(
        c: Vec<u32>,
        p: segmented_cuckoo::CuckooParams,
        n: u64,
    ) -> Result<segmented_cuckoo::CuckooKVStore<Self>, segmented_cuckoo::CuckooError> {
        segmented_cuckoo::CuckooKVStore::<Self>::from_cells(c, p, n)
    }
}

/// `true` unless started by `cargo bench` (the only invocation where cargo
/// passes `--bench`). Bench `main()`s early-return on this so
/// `cargo test --all-targets` builds them without running the full sweep.
#[allow(dead_code)]
pub fn skip_when_cargo_test() -> bool {
    !std::env::args().any(|a| a == "--bench")
}
