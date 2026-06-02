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
//! **Parameters:** num_buckets ∈ {2^16, 2^18, 2^20}. bucket_size ∈ {1, 2, 3, 4}. fingerprint_bits = 12.
//! 3 warmup + 10 measured trials.
//!
//! **Output:** `results/delete_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, deleted, mean_lf, mean_duration_ns, mean_mops, min_mops, max_mops, stddev_mops

mod helpers;

use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooFilter, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter,
    Standard2aryCuckooFilter, Standard3aryCuckooFilter, Standard4aryCuckooFilter,
};
use std::io::Write;
use std::time::Instant;

const MAX_KICKS: u32 = 2500;
const FINGERPRINT_BITS: u32 = 32;
const WARMUP_TRIALS: usize = 3;
const MEASURE_TRIALS: usize = 10;

const NUM_BUCKETS_VALUES: &[u32] = &[1 << 20];
const BUCKET_SIZE_VALUES: &[u32] = &[1, 2, 3, 4];

/// Fill a filter of type `$filter_ty` with sequential u64 keys until `TableFull`,
/// then measure how fast we can delete all inserted items in bulk.
///
/// The deletion loop timing deliberately excludes the fill phase so that the CSV
/// captures pure delete throughput, not insert + delete throughput.
///
/// CSV columns: scheme, arity, num_buckets, bucket_size, deleted, mean_lf, mean_duration_ns, mean_mops, min_mops, max_mops, stddev_mops
macro_rules! bench_delete {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity:expr, $num_buckets:expr, $bucket_size:expr) => {{
        let num_buckets: u32 = $num_buckets;
        let bucket_size: u32 = $bucket_size;

        match <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS) {
            Err(e) => {
                eprintln!("  Skip {} num_buckets={} bucket_size={}: {}", $label, num_buckets, bucket_size, e);
            }
            Ok(_) => {
                // ── Determine how many items fit ───────────────────────────
                // Use a single fill to find `count` so that every trial starts
                // with the same number of deletions, making throughput numbers
                // comparable across trials.
                let count: u64 = {
                    let mut probe = <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS).unwrap();
                    probe.set_max_kicks(MAX_KICKS);
                    let mut c = 0u64;
                    loop {
                        match probe.insert(c.to_le_bytes()) {
                            Ok(()) => c += 1,
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("{}", e),
                        }
                    }
                    c
                };

                if count == 0 {
                    eprintln!("  Skip {} num_buckets={} bucket_size={}: filter holds 0 items", $label, num_buckets, bucket_size);
                } else {
                    // ── Warmup ─────────────────────────────────────────────
                    for _ in 0..WARMUP_TRIALS {
                        let mut filter = <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS).unwrap();
                        filter.set_max_kicks(MAX_KICKS);
                        for i in 0u64..count {
                            let _ = filter.insert(i.to_le_bytes());
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
                        let mut filter = <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS).unwrap();
                        filter.set_max_kicks(MAX_KICKS);
                        for i in 0u64..count {
                            let _ = filter.insert(i.to_le_bytes());
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
                        $scheme, $arity, num_buckets, bucket_size, count, mean_lf, mean_dur,
                        mops_stats.mean, mops_stats.min, mops_stats.max, mops_stats.stddev
                    )
                    .unwrap();
                    println!(
                        "  {:<20} num_buckets={:<10} bucket_size={:<3} | mean={:<8.3} std={:<8.3} Mops",
                        $label, num_buckets, bucket_size, mops_stats.mean, mops_stats.stddev
                    );
                }
            }
        }
    }};
}

#[allow(clippy::cognitive_complexity)] // bench main(): linear CLI-parse + dispatch plumbing
fn main() {
    if helpers::skip_when_cargo_test() {
        return;
    }
    let mut csv = helpers::csv_writer(
        "delete_throughput.csv",
        "scheme,arity,num_buckets,bucket_size,deleted,mean_lf,mean_duration_ns,mean_mops,min_mops,max_mops,stddev_mops",
    );

    println!("=== Delete Throughput (delete all items from a full filter) ===");
    println!(
        "Config: fingerprint_bits={}, max_kicks={}, warmup={}, trials={}",
        FINGERPRINT_BITS, MAX_KICKS, WARMUP_TRIALS, MEASURE_TRIALS
    );

    // Standard 3-ary num_buckets (power of 3) sized comparably to NUM_BUCKETS_VALUES.
    let pow3_num_buckets: &[u32] = &[3u32.pow(13)];
    // Standard 4-ary num_buckets (power of 4) sized comparably to NUM_BUCKETS_VALUES.
    let pow4_num_buckets: &[u32] = &[4u32.pow(10)];

    for (idx, &num_buckets) in NUM_BUCKETS_VALUES.iter().enumerate() {
        for &bucket_size in BUCKET_SIZE_VALUES {
            // Segmented 3-ary requires num_buckets = 3·2^t; pick the largest valid value
            // that fits inside the 2^k window so the comparison stays at a similar size.
            let seg3_num_buckets = 3 * (1u32 << 19);
            let std3_num_buckets = pow3_num_buckets[idx];
            let std4_num_buckets = pow4_num_buckets[idx];

            println!(
                "\n--- num_buckets={}, bucket_size={} ---",
                num_buckets, bucket_size
            );

            bench_delete!(
                csv,
                "Segmented 2-ary",
                Segmented2aryCuckooFilter,
                "segmented",
                2,
                num_buckets,
                bucket_size
            );
            bench_delete!(
                csv,
                "Standard 2-ary",
                Standard2aryCuckooFilter,
                "standard",
                2,
                num_buckets,
                bucket_size
            );
            bench_delete!(
                csv,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                3,
                seg3_num_buckets,
                bucket_size
            );
            bench_delete!(
                csv,
                "Standard 3-ary",
                Standard3aryCuckooFilter,
                "standard",
                3,
                std3_num_buckets,
                bucket_size
            );
            bench_delete!(
                csv,
                "Segmented 4-ary",
                Segmented4aryCuckooFilter,
                "segmented",
                4,
                num_buckets,
                bucket_size
            );
            bench_delete!(
                csv,
                "Standard 4-ary",
                Standard4aryCuckooFilter,
                "standard",
                4,
                std4_num_buckets,
                bucket_size
            );
        }
    }

    println!("\nResults written to results/delete_throughput.csv");
}
