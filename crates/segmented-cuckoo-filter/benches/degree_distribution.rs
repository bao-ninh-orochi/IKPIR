//! **Intent:** Analyse how many items map to each bucket ("bucket degree") as the filter fills,
//! revealing load balance and any spatial structure introduced by segmented indexing.
//!
//! **Method:** Before each `filter.add()`, call `scheme.hash_item()` directly to retrieve the
//! k candidate bucket indices. On successful insert, increment degree counters for those
//! buckets. Two outputs in one run:
//!
//! - **Degree-index (Part 1):** Per-bucket degree vs. bucket index for a configurable
//!   (arity, n). Configured by `DI_ARITY` and `DI_N` constants (default: arity=2, n=65536).
//!   Shows whether segmented schemes create degree "bands" at segment boundaries.
//!
//! - **Degree histogram (Part 2):** Count of buckets with each degree value, for all 6
//!   schemes at fixed n. Shows the shape of the degree distribution (Poisson-like for
//!   standard; segmented may differ).
//!
//! **Rationale:** Bucket degree is a proxy for hash balance. A scheme where some buckets
//! reach very high degree will exhaust those first and force more evictions. The segmented
//! constraint (each index lives in its own segment) may produce more uniform degree within
//! segments compared to uniform hashing.
//!
//! **Parameters (degree-index):** DI_ARITY=2 (default, change to 3 or 4), DI_N=65536.
//! b ∈ {1, 2, 3, 4}.
//! **Parameters (histogram):** All arities. DH_N=65536 (seg3: 49152). b ∈ {1, 2, 3, 4}.
//!
//! **Output:**
//! - `results/degree_per_bucket.csv` — columns: scheme, arity, n, b, bucket_index, degree
//! - `results/degree_distribution.csv` — columns: scheme, arity, n, b, degree, count

mod helpers;

use segmented_cuckoo_filter::{
    CuckooError, IndexScheme, Segmented3aryCuckooFilter, Segmented3aryScheme,
    Segmented4aryCuckooFilter, Segmented4aryScheme, SegmentedCuckooFilter, SegmentedScheme,
    Standard3aryCuckooFilter, Standard3aryScheme, Standard4aryCuckooFilter, Standard4aryScheme,
    StandardCuckooFilter, StandardScheme,
};
use std::io::Write;

const MAX_KICKS: u32 = 500;
const FP_BITS: u32 = 12;

// ── Degree-index parameters ─────────────────────────────────────────────────
// Change DI_ARITY to 3 or 4 to experiment with other arities.
const DI_ARITY: usize = 2;
const DI_N: u32 = 1 << 16; // 65536 (used for 2-ary and 4-ary segmented)
const DI_N_POW3: u32 = 3u32.pow(10); // 59049 — power of 3 for standard 3-ary
const DI_N_POW4: u32 = 4u32.pow(8); // 65536 — power of 4 for standard 4-ary (same as DI_N)
const DI_B_VALUES: &[u32] = &[1, 2, 3, 4];

// ── Degree histogram parameters ─────────────────────────────────────────────
const DH_N: u32 = 1 << 16; // 65536
const DH_N_POW3: u32 = 3u32.pow(10); // 59049 — power of 3 for standard 3-ary
const DH_N_POW4: u32 = 4u32.pow(8); // 65536 — power of 4 for standard 4-ary
const DH_SEG3_N: u32 = 3 * (1 << 14); // 49152 — closest valid n for segmented 3-ary
const DH_B_VALUES: &[u32] = &[1, 2, 3, 4];

// ─── Shared helpers ─────────────────────────────────────────────────────────

fn collect_degrees<S, F>(
    scheme: &S,
    mut add: F,
    n: usize,
    arity: usize,
    max_items: u64,
) -> (Vec<u64>, u64)
where
    S: IndexScheme,
    F: FnMut(&[u8; 8]) -> Result<(), CuckooError>,
{
    let mut degree = vec![0u64; n];
    let mut total = 0u64;
    for i in 0u64..max_items {
        let item = i.to_le_bytes();
        let (_, indices) = scheme.hash_item(&item, FP_BITS);
        match add(&item) {
            Ok(()) => {
                for k in 0..arity {
                    degree[indices[k] as usize] += 1;
                }
                total += 1;
            }
            Err(CuckooError::TableFull) => break,
            Err(e) => panic!("unexpected error: {}", e),
        }
    }
    (degree, total)
}

fn degree_histogram(degree: &[u64]) -> Vec<(u64, u64)> {
    if degree.is_empty() {
        return vec![];
    }
    let max_deg = *degree.iter().max().unwrap();
    let mut hist = vec![0u64; (max_deg + 1) as usize];
    for &d in degree {
        hist[d as usize] += 1;
    }
    hist.iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(d, &c)| (d as u64, c))
        .collect()
}

fn write_histogram(
    csv: &mut impl Write,
    scheme: &str,
    arity: usize,
    n: u32,
    b: u32,
    hist: &[(u64, u64)],
) {
    for &(deg, count) in hist {
        writeln!(csv, "{},{},{},{},{},{}", scheme, arity, n, b, deg, count).unwrap();
    }
}

fn write_bucket_degrees(
    csv: &mut impl Write,
    scheme: &str,
    arity: usize,
    n: u32,
    b: u32,
    degree: &[u64],
) {
    for (idx, &deg) in degree.iter().enumerate() {
        writeln!(csv, "{},{},{},{},{},{}", scheme, arity, n, b, idx, deg).unwrap();
    }
}

// ─── Degree-index macro (writes per-bucket data) ───────────────────────────

macro_rules! run_degree_index {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme_expr:expr,
     $scheme_str:expr, $arity:expr, $n:expr, $b:expr) => {{
        let n: u32 = $n;
        let b: u32 = $b;
        let arity: usize = $arity;
        match <$filter_ty>::new(n, b, FP_BITS) {
            Ok(mut filter) => {
                filter.set_max_kicks(MAX_KICKS);
                let scheme = $scheme_expr;
                let max_items = n as u64 * b as u64 * 2;
                let (degree, inserted) = collect_degrees(
                    &scheme,
                    |item| filter.add(*item),
                    n as usize,
                    arity,
                    max_items,
                );
                let mean_deg = degree.iter().sum::<u64>() as f64 / n as f64;
                let max_deg = *degree.iter().max().unwrap_or(&0);
                println!(
                    "  {:<22} n={:<8} b={} | inserted={:<8} mean_deg={:.2} max_deg={}",
                    $label, n, b, inserted, mean_deg, max_deg
                );
                write_bucket_degrees($csv, $scheme_str, arity, n, b, &degree);
            }
            Err(e) => eprintln!("  Skip {} n={} b={}: {}", $label, n, b, e),
        }
    }};
}

// ─── Degree histogram macro (writes histogram data) ────────────────────────

macro_rules! run_degree_hist {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme_expr:expr,
     $scheme_str:expr, $arity:expr, $n:expr, $b:expr) => {{
        let n: u32 = $n;
        let b: u32 = $b;
        let arity: usize = $arity;
        match <$filter_ty>::new(n, b, FP_BITS) {
            Ok(mut filter) => {
                filter.set_max_kicks(MAX_KICKS);
                let scheme = $scheme_expr;
                let max_items = n as u64 * b as u64 * 2;
                let (degree, inserted) = collect_degrees(
                    &scheme,
                    |item| filter.add(*item),
                    n as usize,
                    arity,
                    max_items,
                );
                let hist = degree_histogram(&degree);
                let mean_deg = degree.iter().sum::<u64>() as f64 / n as f64;
                let max_deg = hist.last().map_or(0, |&(d, _)| d);
                println!(
                    "  {:<22} n={:<8} b={} | inserted={:<8} mean_deg={:.2} max_deg={}",
                    $label, n, b, inserted, mean_deg, max_deg
                );
                write_histogram($csv, $scheme_str, arity, n, b, &hist);
            }
            Err(e) => eprintln!("  Skip {} n={} b={}: {}", $label, n, b, e),
        }
    }};
}

fn main() {
    let mut csv_bucket = helpers::csv_writer(
        "degree_per_bucket.csv",
        "scheme,arity,n,b,bucket_index,degree",
    );
    let mut csv_hist =
        helpers::csv_writer("degree_distribution.csv", "scheme,arity,n,b,degree,count");

    // ── Part 1: Degree-index (per-bucket data) ─────────────────────────────
    println!(
        "=== Degree-Index (per-bucket, arity={}, n={}) ===",
        DI_ARITY, DI_N
    );

    match DI_ARITY {
        2 => {
            for &b in DI_B_VALUES {
                println!(" b={}", b);
                run_degree_index!(
                    &mut csv_bucket,
                    "Standard 2-ary",
                    StandardCuckooFilter,
                    StandardScheme { num_buckets: DI_N },
                    "standard",
                    2,
                    DI_N,
                    b
                );
                run_degree_index!(
                    &mut csv_bucket,
                    "Segmented 2-ary",
                    SegmentedCuckooFilter,
                    SegmentedScheme { half: DI_N / 2 },
                    "segmented",
                    2,
                    DI_N,
                    b
                );
            }
        }
        3 => {
            let seg3_n = 3 * (1u32 << (DI_N / 3).ilog2());
            for &b in DI_B_VALUES {
                println!(" b={}", b);
                run_degree_index!(
                    &mut csv_bucket,
                    "Standard 3-ary",
                    Standard3aryCuckooFilter,
                    Standard3aryScheme {
                        num_buckets: DI_N_POW3
                    },
                    "standard",
                    3,
                    DI_N_POW3,
                    b
                );
                run_degree_index!(
                    &mut csv_bucket,
                    "Segmented 3-ary",
                    Segmented3aryCuckooFilter,
                    Segmented3aryScheme {
                        segment_size: seg3_n / 3
                    },
                    "segmented",
                    3,
                    seg3_n,
                    b
                );
            }
        }
        4 => {
            for &b in DI_B_VALUES {
                println!(" b={}", b);
                run_degree_index!(
                    &mut csv_bucket,
                    "Standard 4-ary",
                    Standard4aryCuckooFilter,
                    Standard4aryScheme {
                        num_buckets: DI_N_POW4
                    },
                    "standard",
                    4,
                    DI_N_POW4,
                    b
                );
                run_degree_index!(
                    &mut csv_bucket,
                    "Segmented 4-ary",
                    Segmented4aryCuckooFilter,
                    Segmented4aryScheme {
                        segment_size: DI_N / 4
                    },
                    "segmented",
                    4,
                    DI_N,
                    b
                );
            }
        }
        _ => panic!("unsupported DI_ARITY={}", DI_ARITY),
    }

    // ── Part 2: Degree histogram (all arities) ─────────────────────────────
    println!(
        "\n=== Degree Histogram (n={}, seg3_n={}) ===",
        DH_N, DH_SEG3_N
    );

    for &b in DH_B_VALUES {
        println!("\n--- b={} ---", b);

        // 2-ary
        run_degree_hist!(
            &mut csv_hist,
            "Standard 2-ary",
            StandardCuckooFilter,
            StandardScheme { num_buckets: DH_N },
            "standard",
            2,
            DH_N,
            b
        );
        run_degree_hist!(
            &mut csv_hist,
            "Segmented 2-ary",
            SegmentedCuckooFilter,
            SegmentedScheme { half: DH_N / 2 },
            "segmented",
            2,
            DH_N,
            b
        );

        // 3-ary
        run_degree_hist!(
            &mut csv_hist,
            "Standard 3-ary",
            Standard3aryCuckooFilter,
            Standard3aryScheme {
                num_buckets: DH_N_POW3
            },
            "standard",
            3,
            DH_N_POW3,
            b
        );
        run_degree_hist!(
            &mut csv_hist,
            "Segmented 3-ary",
            Segmented3aryCuckooFilter,
            Segmented3aryScheme {
                segment_size: DH_SEG3_N / 3
            },
            "segmented",
            3,
            DH_SEG3_N,
            b
        );

        // 4-ary
        run_degree_hist!(
            &mut csv_hist,
            "Standard 4-ary",
            Standard4aryCuckooFilter,
            Standard4aryScheme {
                num_buckets: DH_N_POW4
            },
            "standard",
            4,
            DH_N_POW4,
            b
        );
        run_degree_hist!(
            &mut csv_hist,
            "Segmented 4-ary",
            Segmented4aryCuckooFilter,
            Segmented4aryScheme {
                segment_size: DH_N / 4
            },
            "segmented",
            4,
            DH_N,
            b
        );
    }

    println!("\nResults written to:");
    println!("  results/degree_per_bucket.csv");
    println!("  results/degree_distribution.csv");
}
