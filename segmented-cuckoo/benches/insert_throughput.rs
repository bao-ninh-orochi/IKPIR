//! **Intent:** Measure raw insert speed for each scheme at its natural
//! load factor.
//!
//! **Method:** Insert sequential keys until `TableFull`, timing the full
//! loop. Divide total insertions by elapsed ns to get Mops/sec. 3 warmup
//! trials, then 10 timed trials.
//!
//! **Arguments (CLI):** none — sweeps a built-in cross-product (see
//! `NUM_BUCKETS_VALUES` / `BUCKET_SIZE_VALUES` constants below). One CSV
//! row per `(scheme, arity, num_buckets, bucket_size)`.
//!
//! **Design rationale:** "Insert until full" is the only fair comparison
//! method. Inserting a fixed number of items (e.g. 1 M) would confound
//! throughput with load factor: the same 1 M items might fill standard
//! 3-ary to ~90% (near full, expensive kicking) but segmented 3-ary to
//! only ~70% (sparse, cheap kicking), making the standard variant appear
//! artificially slow. Measuring over the full fill trajectory puts all
//! schemes under comparable conditions.
//!
//! **Parameters:** `num_buckets ∈ {2^16, 2^18, 2^20}`,
//! `bucket_size ∈ {1, 2, 3, 4}`, `fingerprint_bits = 12`, `max_kicks =
//! 2500`. 3 warmup + 10 measured trials per cell. Standard 3-ary uses
//! the matching power-of-3 size; standard 4-ary uses the matching
//! power-of-4 size.
//!
//! **Output:** `results/insert_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, mean_inserted,
//! mean_lf, mean_duration_ns, mean_mops, min_mops, max_mops, stddev_mops

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

macro_rules! bench_insert {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity:expr, $num_buckets:expr, $bucket_size:expr) => {{
        let num_buckets: u32 = $num_buckets;
        let bucket_size: u32 = $bucket_size;

        match <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS) {
            Err(e) => {
                eprintln!(
                    "  Skip {} num_buckets={} bucket_size={}: {}",
                    $label, num_buckets, bucket_size, e
                );
            }
            Ok(_) => {
                // Warmup
                for _ in 0..WARMUP_TRIALS {
                    let mut filter =
                        <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS).unwrap();
                    filter.set_max_kicks(MAX_KICKS);
                    let mut i = 0u64;
                    loop {
                        match filter.insert(i.to_le_bytes()) {
                            Ok(()) => i += 1,
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("{}", e),
                        }
                    }
                }

                // Measure
                let mut throughputs = Vec::with_capacity(MEASURE_TRIALS);
                let mut inserted_vals = Vec::with_capacity(MEASURE_TRIALS);
                let mut lf_vals = Vec::with_capacity(MEASURE_TRIALS);
                let mut duration_vals = Vec::with_capacity(MEASURE_TRIALS);
                for _trial in 0..MEASURE_TRIALS {
                    let mut filter =
                        <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS).unwrap();
                    filter.set_max_kicks(MAX_KICKS);
                    let start = Instant::now();
                    let mut i = 0u64;
                    loop {
                        match filter.insert(i.to_le_bytes()) {
                            Ok(()) => i += 1,
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("{}", e),
                        }
                    }
                    let elapsed_ns = start.elapsed().as_nanos() as f64;
                    let inserted = i;
                    let lf = filter.load_factor();
                    let mops = inserted as f64 / elapsed_ns * 1000.0;
                    throughputs.push(mops);
                    inserted_vals.push(inserted as f64);
                    lf_vals.push(lf);
                    duration_vals.push(elapsed_ns);
                }
                let mops_stats = helpers::compute_stats(&throughputs);
                let mean_inserted = inserted_vals.iter().sum::<f64>() / MEASURE_TRIALS as f64;
                let mean_lf = lf_vals.iter().sum::<f64>() / MEASURE_TRIALS as f64;
                let mean_dur = duration_vals.iter().sum::<f64>() / MEASURE_TRIALS as f64;
                writeln!(
                    $csv,
                    "{},{},{},{},{:.0},{:.6},{:.0},{:.4},{:.4},{:.4},{:.4}",
                    $scheme,
                    $arity,
                    num_buckets,
                    bucket_size,
                    mean_inserted,
                    mean_lf,
                    mean_dur,
                    mops_stats.mean,
                    mops_stats.min,
                    mops_stats.max,
                    mops_stats.stddev
                )
                .unwrap();
                println!(
                    "  {:<20} num_buckets={:<10} bucket_size={:<3} | mean={:<8.3} std={:<8.3} Mops",
                    $label, num_buckets, bucket_size, mops_stats.mean, mops_stats.stddev
                );
            }
        }
    }};
}

#[allow(clippy::cognitive_complexity)] // bench main(): linear CLI-parse + dispatch plumbing
fn main() {
    let mut csv = helpers::csv_writer(
        "insert_throughput.csv",
        "scheme,arity,num_buckets,bucket_size,mean_inserted,mean_lf,mean_duration_ns,mean_mops,min_mops,max_mops,stddev_mops",
    );

    println!("=== Insert Throughput (insert until full) ===");
    println!(
        "Config: fingerprint_bits={}, max_kicks={}, warmup={}, trials={}",
        FINGERPRINT_BITS, MAX_KICKS, WARMUP_TRIALS, MEASURE_TRIALS
    );

    // Standard 3-ary num_buckets (power of 3) sized comparably to NUM_BUCKETS_VALUES.
    let pow3_num_buckets: &[u32] = &[3u32.pow(13)]; // 19683, 59049, 177147
                                                    // Standard 4-ary num_buckets (power of 4) sized comparably to NUM_BUCKETS_VALUES.
    let pow4_num_buckets: &[u32] = &[4u32.pow(10)]; // 65536, 262144, 1048576

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

            bench_insert!(
                csv,
                "Segmented 2-ary",
                Segmented2aryCuckooFilter,
                "segmented",
                2,
                num_buckets,
                bucket_size
            );
            bench_insert!(
                csv,
                "Standard 2-ary",
                Standard2aryCuckooFilter,
                "standard",
                2,
                num_buckets,
                bucket_size
            );
            bench_insert!(
                csv,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                3,
                seg3_num_buckets,
                bucket_size
            );
            bench_insert!(
                csv,
                "Standard 3-ary",
                Standard3aryCuckooFilter,
                "standard",
                3,
                std3_num_buckets,
                bucket_size
            );
            bench_insert!(
                csv,
                "Segmented 4-ary",
                Segmented4aryCuckooFilter,
                "segmented",
                4,
                num_buckets,
                bucket_size
            );
            bench_insert!(
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

    println!("\nResults written to results/insert_throughput.csv");
}
