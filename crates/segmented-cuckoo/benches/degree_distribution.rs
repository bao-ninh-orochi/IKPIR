//! **Intent:** Analyse how many items map to each bucket ("bucket degree") as the filter fills,
//! revealing load balance and any spatial structure introduced by segmented indexing.
//!
//! **Method:** Before each `filter.insert()`, call `scheme.hash_item()` directly to retrieve the
//! candidate bucket indices. On successful insert, increment degree counters for those
//! buckets. Two outputs in one run:
//!
//! - **Degree-index (Part 1):** Per-bucket degree vs. bucket index for a configurable
//!   (arity, num_buckets). Configured by `DI_ARITY` and `DI_NUM_BUCKETS` constants (default:
//!   arity=2, num_buckets=65536). Shows whether segmented schemes create degree "bands" at
//!   segment boundaries.
//!
//! - **Degree histogram (Part 2):** Count of buckets with each degree value, for all 6
//!   schemes at fixed num_buckets. Shows the shape of the degree distribution (Poisson-like for
//!   standard; segmented may differ).
//!
//! **Rationale:** Bucket degree is a proxy for hash balance. A scheme where some buckets
//! reach very high degree will exhaust those first and force more evictions. The segmented
//! constraint (each index lives in its own segment) may produce more uniform degree within
//! segments compared to uniform hashing.
//!
//!
//! **Parameters (degree-index):** DI_ARITY=2 (default; change to 3 or 4),
//! DI_NUM_BUCKETS=65536 (used for arity=2 and arity=4 segmented; power of 2 ≥ 4),
//! DI_NUM_BUCKETS=49152 (used for arity=3 segmented; 3·2^14),
//! DI_NUM_BUCKETS_POW3=3^10=59049 (standard 3-ary; power of 3),
//! DI_NUM_BUCKETS_POW4=4^8=65536 (standard 4-ary; power of 4).
//! bucket_size ∈ {1, 2, 3, 4}.
//! **Parameters (histogram):** All arities. DH_NUM_BUCKETS=65536, DH_NUM_BUCKETS_POW3=59049,
//! DH_NUM_BUCKETS_POW4=65536, DH_SEG3_NUM_BUCKETS=3·2^14=49152. bucket_size ∈ {1, 2, 3, 4}.
//!
//! **Output:**
//! - `crates/segmented-cuckoo/results/degree_per_bucket.csv` — columns: scheme, arity, num_buckets, bucket_size, bucket_index, degree
//! - `crates/segmented-cuckoo/results/degree_distribution.csv` — columns: scheme, arity, num_buckets, bucket_size, degree, count

mod helpers;

use segmented_cuckoo::{
    CuckooError, IndexScheme, Segmented2aryCuckooFilter, Segmented2aryScheme,
    Segmented3aryCuckooFilter, Segmented3aryScheme, Segmented4aryCuckooFilter, Segmented4aryScheme,
    Standard2aryCuckooFilter, Standard2aryScheme, Standard3aryCuckooFilter, Standard3aryScheme,
    Standard4aryCuckooFilter, Standard4aryScheme,
};
use std::io::Write;

const MAX_KICKS: u32 = 500;
const FINGERPRINT_BITS: u32 = 12;

// ── Degree-index parameters ─────────────────────────────────────────────────
// Change DI_ARITY to 3 or 4 to experiment with other arities.
const DI_ARITY: usize = 2;
const DI_NUM_BUCKETS: u32 = 1 << 16; // 65536; valid for segmented 2-ary, segmented 4-ary, standard 2-ary
const DI_NUM_BUCKETS_POW3: u32 = 3u32.pow(10); // 59049; power of 3 for standard 3-ary
const DI_NUM_BUCKETS_POW4: u32 = 4u32.pow(8); // 65536; power of 4 for standard 4-ary
const DI_BUCKET_SIZE_VALUES: &[u32] = &[1, 2, 3, 4];

// ── Degree histogram parameters ─────────────────────────────────────────────
const DH_NUM_BUCKETS: u32 = 1 << 16; // 65536; valid for segmented 2-ary, segmented 4-ary, standard 2-ary
const DH_NUM_BUCKETS_POW3: u32 = 3u32.pow(10); // 59049; power of 3 for standard 3-ary
const DH_NUM_BUCKETS_POW4: u32 = 4u32.pow(8); // 65536; power of 4 for standard 4-ary
const DH_SEG3_NUM_BUCKETS: u32 = 3 * (1 << 14); // 49152; closest 3·2^t value for segmented 3-ary
const DH_BUCKET_SIZE_VALUES: &[u32] = &[1, 2, 3, 4];

// ─── Shared helpers ─────────────────────────────────────────────────────────

fn collect_degrees<S, F>(
    scheme: &S,
    mut add: F,
    num_buckets: usize,
    arity: usize,
    max_items: u64,
) -> (Vec<u64>, u64)
where
    S: IndexScheme,
    F: FnMut(&[u8; 8]) -> Result<(), CuckooError>,
{
    let mut degree = vec![0u64; num_buckets];
    let mut total = 0u64;
    for i in 0u64..max_items {
        let item = i.to_le_bytes();
        let (_, indices) = scheme.hash_item(&item, FINGERPRINT_BITS);
        match add(&item) {
            Ok(()) => {
                for a in 0..arity {
                    degree[indices[a] as usize] += 1;
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
    num_buckets: u32,
    bucket_size: u32,
    hist: &[(u64, u64)],
) {
    for &(deg, count) in hist {
        writeln!(
            csv,
            "{},{},{},{},{},{}",
            scheme, arity, num_buckets, bucket_size, deg, count
        )
        .unwrap();
    }
}

fn write_bucket_degrees(
    csv: &mut impl Write,
    scheme: &str,
    arity: usize,
    num_buckets: u32,
    bucket_size: u32,
    degree: &[u64],
) {
    for (idx, &deg) in degree.iter().enumerate() {
        writeln!(
            csv,
            "{},{},{},{},{},{}",
            scheme, arity, num_buckets, bucket_size, idx, deg
        )
        .unwrap();
    }
}

// ─── Degree-index macro (writes per-bucket data) ───────────────────────────

macro_rules! run_degree_index {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme_expr:expr,
     $scheme_str:expr, $arity:expr, $num_buckets:expr, $bucket_size:expr) => {{
        let num_buckets: u32 = $num_buckets;
        let bucket_size: u32 = $bucket_size;
        let arity: usize = $arity;
        match <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS) {
            Ok(mut filter) => {
                filter.set_max_kicks(MAX_KICKS);
                let scheme = $scheme_expr;
                let max_items = num_buckets as u64 * bucket_size as u64 * 2;
                let (degree, inserted) = collect_degrees(
                    &scheme,
                    |item| filter.insert(*item),
                    num_buckets as usize,
                    arity,
                    max_items,
                );
                let mean_deg = degree.iter().sum::<u64>() as f64 / num_buckets as f64;
                let max_deg = *degree.iter().max().unwrap_or(&0);
                println!(
                    "  {:<22} num_buckets={:<8} bucket_size={} | inserted={:<8} mean_deg={:.2} max_deg={}",
                    $label, num_buckets, bucket_size, inserted, mean_deg, max_deg
                );
                write_bucket_degrees($csv, $scheme_str, arity, num_buckets, bucket_size, &degree);
            }
            Err(e) => eprintln!("  Skip {} num_buckets={} bucket_size={}: {}", $label, num_buckets, bucket_size, e),
        }
    }};
}

// ─── Degree histogram macro (writes histogram data) ────────────────────────

macro_rules! run_degree_hist {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme_expr:expr,
     $scheme_str:expr, $arity:expr, $num_buckets:expr, $bucket_size:expr) => {{
        let num_buckets: u32 = $num_buckets;
        let bucket_size: u32 = $bucket_size;
        let arity: usize = $arity;
        match <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS) {
            Ok(mut filter) => {
                filter.set_max_kicks(MAX_KICKS);
                let scheme = $scheme_expr;
                let max_items = num_buckets as u64 * bucket_size as u64 * 2;
                let (degree, inserted) = collect_degrees(
                    &scheme,
                    |item| filter.insert(*item),
                    num_buckets as usize,
                    arity,
                    max_items,
                );
                let hist = degree_histogram(&degree);
                let mean_deg = degree.iter().sum::<u64>() as f64 / num_buckets as f64;
                let max_deg = hist.last().map_or(0, |&(d, _)| d);
                println!(
                    "  {:<22} num_buckets={:<8} bucket_size={} | inserted={:<8} mean_deg={:.2} max_deg={}",
                    $label, num_buckets, bucket_size, inserted, mean_deg, max_deg
                );
                write_histogram($csv, $scheme_str, arity, num_buckets, bucket_size, &hist);
            }
            Err(e) => eprintln!("  Skip {} num_buckets={} bucket_size={}: {}", $label, num_buckets, bucket_size, e),
        }
    }};
}

fn main() {
    if helpers::skip_when_cargo_test() {
        return;
    }
    let mut csv_bucket = helpers::csv_writer(
        "degree_per_bucket.csv",
        "scheme,arity,num_buckets,bucket_size,bucket_index,degree",
    );
    let mut csv_hist = helpers::csv_writer(
        "degree_distribution.csv",
        "scheme,arity,num_buckets,bucket_size,degree,count",
    );

    // ── Part 1: Degree-index (per-bucket data) ─────────────────────────────
    println!(
        "=== Degree-Index (per-bucket, arity={}, num_buckets={}) ===",
        DI_ARITY, DI_NUM_BUCKETS
    );

    match DI_ARITY {
        2 => {
            for &bucket_size in DI_BUCKET_SIZE_VALUES {
                println!(" bucket_size={}", bucket_size);
                run_degree_index!(
                    &mut csv_bucket,
                    "Segmented 2-ary",
                    Segmented2aryCuckooFilter,
                    Segmented2aryScheme {
                        segment_size: DI_NUM_BUCKETS / 2
                    },
                    "segmented",
                    2,
                    DI_NUM_BUCKETS,
                    bucket_size
                );
                run_degree_index!(
                    &mut csv_bucket,
                    "Standard 2-ary",
                    Standard2aryCuckooFilter,
                    Standard2aryScheme {
                        num_buckets: DI_NUM_BUCKETS
                    },
                    "standard",
                    2,
                    DI_NUM_BUCKETS,
                    bucket_size
                );
            }
        }
        3 => {
            // Segmented 3-ary requires num_buckets = 3·2^t; pick the largest valid value
            // that fits inside the DI_NUM_BUCKETS window.
            let seg3_num_buckets = 3 * (1u32 << (DI_NUM_BUCKETS / 3).ilog2());
            for &bucket_size in DI_BUCKET_SIZE_VALUES {
                println!(" bucket_size={}", bucket_size);
                run_degree_index!(
                    &mut csv_bucket,
                    "Segmented 3-ary",
                    Segmented3aryCuckooFilter,
                    Segmented3aryScheme {
                        segment_size: seg3_num_buckets / 3
                    },
                    "segmented",
                    3,
                    seg3_num_buckets,
                    bucket_size
                );
                run_degree_index!(
                    &mut csv_bucket,
                    "Standard 3-ary",
                    Standard3aryCuckooFilter,
                    Standard3aryScheme {
                        num_buckets: DI_NUM_BUCKETS_POW3
                    },
                    "standard",
                    3,
                    DI_NUM_BUCKETS_POW3,
                    bucket_size
                );
            }
        }
        4 => {
            for &bucket_size in DI_BUCKET_SIZE_VALUES {
                println!(" bucket_size={}", bucket_size);
                run_degree_index!(
                    &mut csv_bucket,
                    "Segmented 4-ary",
                    Segmented4aryCuckooFilter,
                    Segmented4aryScheme {
                        segment_size: DI_NUM_BUCKETS / 4
                    },
                    "segmented",
                    4,
                    DI_NUM_BUCKETS,
                    bucket_size
                );
                run_degree_index!(
                    &mut csv_bucket,
                    "Standard 4-ary",
                    Standard4aryCuckooFilter,
                    Standard4aryScheme {
                        num_buckets: DI_NUM_BUCKETS_POW4
                    },
                    "standard",
                    4,
                    DI_NUM_BUCKETS_POW4,
                    bucket_size
                );
            }
        }
        _ => panic!("unsupported DI_ARITY={}", DI_ARITY),
    }

    // ── Part 2: Degree histogram (all arities) ─────────────────────────────
    println!(
        "\n=== Degree Histogram (num_buckets={}, seg3_num_buckets={}) ===",
        DH_NUM_BUCKETS, DH_SEG3_NUM_BUCKETS
    );

    for &bucket_size in DH_BUCKET_SIZE_VALUES {
        println!("\n--- bucket_size={} ---", bucket_size);

        // 2-ary
        run_degree_hist!(
            &mut csv_hist,
            "Segmented 2-ary",
            Segmented2aryCuckooFilter,
            Segmented2aryScheme {
                segment_size: DH_NUM_BUCKETS / 2
            },
            "segmented",
            2,
            DH_NUM_BUCKETS,
            bucket_size
        );
        run_degree_hist!(
            &mut csv_hist,
            "Standard 2-ary",
            Standard2aryCuckooFilter,
            Standard2aryScheme {
                num_buckets: DH_NUM_BUCKETS
            },
            "standard",
            2,
            DH_NUM_BUCKETS,
            bucket_size
        );

        // 3-ary
        run_degree_hist!(
            &mut csv_hist,
            "Segmented 3-ary",
            Segmented3aryCuckooFilter,
            Segmented3aryScheme {
                segment_size: DH_SEG3_NUM_BUCKETS / 3
            },
            "segmented",
            3,
            DH_SEG3_NUM_BUCKETS,
            bucket_size
        );
        run_degree_hist!(
            &mut csv_hist,
            "Standard 3-ary",
            Standard3aryCuckooFilter,
            Standard3aryScheme {
                num_buckets: DH_NUM_BUCKETS_POW3
            },
            "standard",
            3,
            DH_NUM_BUCKETS_POW3,
            bucket_size
        );

        // 4-ary
        run_degree_hist!(
            &mut csv_hist,
            "Segmented 4-ary",
            Segmented4aryCuckooFilter,
            Segmented4aryScheme {
                segment_size: DH_NUM_BUCKETS / 4
            },
            "segmented",
            4,
            DH_NUM_BUCKETS,
            bucket_size
        );
        run_degree_hist!(
            &mut csv_hist,
            "Standard 4-ary",
            Standard4aryCuckooFilter,
            Standard4aryScheme {
                num_buckets: DH_NUM_BUCKETS_POW4
            },
            "standard",
            4,
            DH_NUM_BUCKETS_POW4,
            bucket_size
        );
    }

    println!("\nResults written to:");
    println!("  results/degree_per_bucket.csv");
    println!("  results/degree_distribution.csv");
}
