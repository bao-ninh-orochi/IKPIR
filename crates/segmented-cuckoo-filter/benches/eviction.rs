//! **Intent:** Measure the eviction chain length distribution — how often an insert is a
//! direct placement vs. requiring many displacements.
//!
//! **Method:** Use `add_with_stats` (not `add`) to capture per-insertion kick counts. Insert
//! until full, accumulating kicks into 5 histogram buckets: 0, 1–10, 11–50, 51–100, 101–500.
//!
//! **Rationale:** `add_with_stats` is a separate code path that returns `InsertStats { kicks }`
//! without touching the hot `add` path. Collecting over the full fill trajectory captures the
//! distribution from empty (cheap inserts) to near-full (expensive eviction chains). b=1 is
//! excluded because single-slot buckets trigger kicking on every collision, making the
//! histogram less informative at the low load factors b=1 achieves.
//!
//! **Parameters:** n ∈ {2^14, 2^16, 2^18, 2^20}. b ∈ {2, 3, 4}. fp_bits = 12.
//! Single trial per config.
//!
//! **Output:** `results/eviction.csv`
//! Columns: scheme, arity, n, b, fp_bits, total_inserts, total_kicks, direct_placements,
//!          max_kicks, mean_kicks, hist_0, hist_1_10, hist_11_50, hist_51_100, hist_101_500

mod helpers;

use segmented_cuckoo_filter::{
    CuckooError, InsertStats, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter,
    SegmentedCuckooFilter, Standard3aryCuckooFilter, Standard4aryCuckooFilter,
    StandardCuckooFilter,
};
use std::io::Write;

const MAX_KICKS: u32 = 500;
const FP_BITS: u32 = 12;
const N_VALUES: &[u32] = &[1 << 14, 1 << 16, 1 << 18, 1 << 20];
const B_VALUES: &[u32] = &[2, 3, 4];

struct EvictionStats {
    total_inserts: u64,
    total_kicks: u64,
    direct_placements: u64,
    max_kicks: u32,
    hist_0: u64,
    hist_1_10: u64,
    hist_11_50: u64,
    hist_51_100: u64,
    hist_101_500: u64,
}

fn collect_eviction_stats<F>(mut insert: F, max_items: u64) -> EvictionStats
where
    F: FnMut(u64) -> Result<InsertStats, CuckooError>,
{
    let mut s = EvictionStats {
        total_inserts: 0,
        total_kicks: 0,
        direct_placements: 0,
        max_kicks: 0,
        hist_0: 0,
        hist_1_10: 0,
        hist_11_50: 0,
        hist_51_100: 0,
        hist_101_500: 0,
    };
    for i in 0u64..max_items {
        match insert(i) {
            Ok(stats) => {
                s.total_inserts += 1;
                s.total_kicks += stats.kicks as u64;
                if stats.kicks > s.max_kicks {
                    s.max_kicks = stats.kicks;
                }
                match stats.kicks {
                    0 => {
                        s.direct_placements += 1;
                        s.hist_0 += 1;
                    }
                    1..=10 => s.hist_1_10 += 1,
                    11..=50 => s.hist_11_50 += 1,
                    51..=100 => s.hist_51_100 += 1,
                    _ => s.hist_101_500 += 1,
                }
            }
            Err(CuckooError::TableFull) => break,
            Err(e) => panic!("unexpected error: {}", e),
        }
    }
    s
}

macro_rules! run_config {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity:expr, $n:expr, $b:expr) => {{
        match <$filter_ty>::new($n, $b, FP_BITS) {
            Ok(mut filter) => {
                filter.set_max_kicks(MAX_KICKS);
                let max_items = ($n as u64) * ($b as u64) * 2;
                let s =
                    collect_eviction_stats(|i| filter.add_with_stats(i.to_le_bytes()), max_items);
                let direct_pct = if s.total_inserts > 0 {
                    s.direct_placements as f64 / s.total_inserts as f64 * 100.0
                } else {
                    0.0
                };
                let mean_kicks = if s.total_inserts > 0 {
                    s.total_kicks as f64 / s.total_inserts as f64
                } else {
                    0.0
                };
                println!(
                    "  {:<20} n={:<8} b={} | inserted={:<8} direct={:.1}% mean_kicks={:.3} max={}",
                    $label, $n, $b, s.total_inserts, direct_pct, mean_kicks, s.max_kicks
                );
                writeln!(
                    $csv,
                    "{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{},{}",
                    $scheme,
                    $arity,
                    $n,
                    $b,
                    FP_BITS,
                    s.total_inserts,
                    s.total_kicks,
                    s.direct_placements,
                    s.max_kicks,
                    mean_kicks,
                    s.hist_0,
                    s.hist_1_10,
                    s.hist_11_50,
                    s.hist_51_100,
                    s.hist_101_500,
                )
                .unwrap();
            }
            Err(e) => {
                eprintln!("  Skipping {} n={} b={}: {}", $label, $n, $b, e);
            }
        }
    }};
}

fn main() {
    let mut csv = helpers::csv_writer(
        "eviction.csv",
        "scheme,arity,n,b,fp_bits,total_inserts,total_kicks,direct_placements,\
         max_kicks,mean_kicks,hist_0,hist_1_10,hist_11_50,hist_51_100,hist_101_500",
    );

    println!("=== Eviction Chain Statistics (insert until full) ===");
    println!("fp_bits={}, max_kicks={}", FP_BITS, MAX_KICKS);

    let pow3_values: &[u32] = &[3u32.pow(8), 3u32.pow(9), 3u32.pow(10), 3u32.pow(11)];
    let pow4_values: &[u32] = &[4u32.pow(7), 4u32.pow(8), 4u32.pow(9), 4u32.pow(10)];

    for (idx, &n) in N_VALUES.iter().enumerate() {
        for &b in B_VALUES {
            println!("\n--- n={}, b={} ---", n, b);

            let seg3_n = 3 * (1u32 << (n / 3).ilog2());
            let std3_n = pow3_values[idx];
            let std4_n = pow4_values[idx];

            // 2-ary
            run_config!(
                csv,
                "Standard 2-ary",
                StandardCuckooFilter,
                "standard",
                2,
                n,
                b
            );
            run_config!(
                csv,
                "Segmented 2-ary",
                SegmentedCuckooFilter,
                "segmented",
                2,
                n,
                b
            );

            // 3-ary
            run_config!(
                csv,
                "Standard 3-ary",
                Standard3aryCuckooFilter,
                "standard",
                3,
                std3_n,
                b
            );
            run_config!(
                csv,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                3,
                seg3_n,
                b
            );

            // 4-ary
            run_config!(
                csv,
                "Standard 4-ary",
                Standard4aryCuckooFilter,
                "standard",
                4,
                std4_n,
                b
            );
            run_config!(
                csv,
                "Segmented 4-ary",
                Segmented4aryCuckooFilter,
                "segmented",
                4,
                n,
                b
            );
        }
    }

    println!("\nResults written to results/eviction.csv");
}
