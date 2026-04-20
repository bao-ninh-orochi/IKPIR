//! **Intent:** Measure pure delete speed on a filter that is already at capacity.
//!
//! **Method:** Determine capacity with a single fill run, then for each trial: fill with that
//! exact count (untimed), time only the deletion loop over all inserted items. 3 warmup +
//! 10 timed trials.
//!
//! **Rationale:** Timing only the delete loop (not fill + delete) isolates the hash-and-probe
//! delete path from insert behaviour. Using a fixed `count` (determined once) ensures every
//! trial deletes the same number of items, keeping throughput numbers comparable. A small
//! fraction of deletes may return `NotFound` due to fingerprint collisions — this is expected
//! and does not invalidate the measurement; the full delete code path is still exercised.
//!
//! **Parameters:** n ∈ {2^16, 2^18, 2^20}. b ∈ {1, 2, 3, 4}. fp_bits = 12.
//! 3 warmup + 10 measured trials.
//!
//! **Output:** `results/delete_throughput.csv`
//! Columns: scheme, arity, n, b, deleted, mean_lf, mean_duration_ns, mean_mops, min_mops, max_mops, stddev_mops

mod helpers;

use segmented_cuckoo_filter::{
    CuckooError, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter, SegmentedCuckooFilter,
    Standard3aryCuckooFilter, Standard4aryCuckooFilter, StandardCuckooFilter,
};
use std::io::Write;
use std::time::Instant;

const MAX_KICKS: u32 = 500;
const FP_BITS: u32 = 12;
const WARMUP_TRIALS: usize = 3;
const MEASURE_TRIALS: usize = 10;

const N_VALUES: &[u32] = &[1 << 16, 1 << 18, 1 << 20];
const B_VALUES: &[u32] = &[1, 2, 3, 4];

/// Fill a filter of type `$filter_ty` with sequential u64 keys until `TableFull`,
/// then measure how fast we can delete all inserted items in bulk.
///
/// The deletion loop timing deliberately excludes the fill phase so that the CSV
/// captures pure delete throughput, not insert + delete throughput.
///
/// CSV columns: scheme, arity, n, b, deleted, mean_lf, mean_duration_ns, mean_mops, min_mops, max_mops, stddev_mops
macro_rules! bench_delete {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity:expr, $n:expr, $b:expr) => {{
        let n: u32 = $n;
        let b: u32 = $b;

        match <$filter_ty>::new(n, b, FP_BITS) {
            Err(e) => {
                eprintln!("  Skip {} n={} b={}: {}", $label, n, b, e);
            }
            Ok(_) => {
                // ── Determine how many items fit ───────────────────────────
                // Use a single fill to find `count` so that every trial starts
                // with the same number of deletions, making throughput numbers
                // comparable across trials.
                let count: u64 = {
                    let mut probe = <$filter_ty>::new(n, b, FP_BITS).unwrap();
                    probe.set_max_kicks(MAX_KICKS);
                    let mut c = 0u64;
                    loop {
                        match probe.add(c.to_le_bytes()) {
                            Ok(()) => c += 1,
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("{}", e),
                        }
                    }
                    c
                };

                if count == 0 {
                    eprintln!("  Skip {} n={} b={}: filter holds 0 items", $label, n, b);
                } else {
                    // ── Warmup ─────────────────────────────────────────────
                    for _ in 0..WARMUP_TRIALS {
                        let mut filter = <$filter_ty>::new(n, b, FP_BITS).unwrap();
                        filter.set_max_kicks(MAX_KICKS);
                        for i in 0u64..count {
                            let _ = filter.add(i.to_le_bytes());
                        }
                        for i in 0u64..count {
                            let _ = filter.delete(i.to_le_bytes());
                        }
                    }

                    // ── Measure ────────────────────────────────────────────
                    let mut throughputs = Vec::with_capacity(MEASURE_TRIALS);
                    let mut lf_vals = Vec::with_capacity(MEASURE_TRIALS);
                    let mut duration_vals = Vec::with_capacity(MEASURE_TRIALS);

                    for _trial in 0..MEASURE_TRIALS {
                        // Fill the filter — not timed.
                        let mut filter = <$filter_ty>::new(n, b, FP_BITS).unwrap();
                        filter.set_max_kicks(MAX_KICKS);
                        for i in 0u64..count {
                            let _ = filter.add(i.to_le_bytes());
                        }
                        lf_vals.push(filter.load_factor());

                        // Time only the deletion loop.
                        let start = Instant::now();
                        for i in 0u64..count {
                            // Ignore the Result — items may occasionally resolve as
                            // NotFound when two distinct items share the same fingerprint
                            // and candidate buckets (false-positive collision). The
                            // throughput measurement is still valid: we are exercising the
                            // full delete path including the hash and bucket probes.
                            let _ = filter.delete(i.to_le_bytes());
                        }
                        let elapsed_ns = start.elapsed().as_nanos() as f64;
                        let mops = count as f64 / elapsed_ns * 1000.0;
                        throughputs.push(mops);
                        duration_vals.push(elapsed_ns);
                    }

                    let mops_stats = helpers::compute_stats(&throughputs);
                    let mean_lf = lf_vals.iter().sum::<f64>() / MEASURE_TRIALS as f64;
                    let mean_dur = duration_vals.iter().sum::<f64>() / MEASURE_TRIALS as f64;
                    writeln!(
                        $csv,
                        "{},{},{},{},{},{:.6},{:.0},{:.4},{:.4},{:.4},{:.4}",
                        $scheme, $arity, n, b, count, mean_lf, mean_dur,
                        mops_stats.mean, mops_stats.min, mops_stats.max, mops_stats.stddev
                    )
                    .unwrap();
                    println!(
                        "  {:<20} n={:<10} b={:<3} | mean={:<8.3} std={:<8.3} Mops",
                        $label, n, b, mops_stats.mean, mops_stats.stddev
                    );
                }
            }
        }
    }};
}

fn main() {
    let mut csv = helpers::csv_writer(
        "delete_throughput.csv",
        "scheme,arity,n,b,deleted,mean_lf,mean_duration_ns,mean_mops,min_mops,max_mops,stddev_mops",
    );

    println!("=== Delete Throughput (delete all items from a full filter) ===");
    println!(
        "Config: fp_bits={}, max_kicks={}, warmup={}, trials={}",
        FP_BITS, MAX_KICKS, WARMUP_TRIALS, MEASURE_TRIALS
    );

    let pow3_values: &[u32] = &[3u32.pow(9), 3u32.pow(10), 3u32.pow(11)];
    let pow4_values: &[u32] = &[4u32.pow(8), 4u32.pow(9), 4u32.pow(10)];

    for (idx, &n) in N_VALUES.iter().enumerate() {
        for &b in B_VALUES {
            // Segmented 3-ary requires n = 3 * 2^m; compute the largest valid n ≤ the
            // requested n so comparisons with other schemes at the same n are approximate.
            let seg3_n = 3 * (1u32 << (n / 3).ilog2());
            let std3_n = pow3_values[idx];
            let std4_n = pow4_values[idx];

            println!("\n--- n={}, b={} ---", n, b);

            bench_delete!(
                csv,
                "Standard 2-ary",
                StandardCuckooFilter,
                "standard",
                2,
                n,
                b
            );
            bench_delete!(
                csv,
                "Segmented 2-ary",
                SegmentedCuckooFilter,
                "segmented",
                2,
                n,
                b
            );
            bench_delete!(
                csv,
                "Standard 3-ary",
                Standard3aryCuckooFilter,
                "standard",
                3,
                std3_n,
                b
            );
            bench_delete!(
                csv,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                3,
                seg3_n,
                b
            );
            bench_delete!(
                csv,
                "Standard 4-ary",
                Standard4aryCuckooFilter,
                "standard",
                4,
                std4_n,
                b
            );
            bench_delete!(
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

    println!("\nResults written to results/delete_throughput.csv");
}
