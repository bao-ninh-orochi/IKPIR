//! **Intent:** Measure query speed at varying hit rates on a full filter.
//!
//! **Method:** Fill until full, then for each hit rate in {0, 25, 50, 75, 100}%: build a
//! query set of q = num_buckets*bucket_size/2 keys mixing inserted items (hits) and never-inserted items (misses)
//! in the given ratio. Time q `contain` calls. 3 warmup + 10 timed trials per hit rate.
//!
//! **Rationale:** Hit rate matters because a lookup that finds a matching fingerprint can
//! terminate early (first matching bucket), while a miss must check all arity buckets. The
//! 5-point sweep reveals whether this early-exit benefit is significant. Testing at full load
//! ensures realistic cache pressure and FPR conditions. q = num_buckets*bucket_size/2 keeps runtime predictable
//! without being trivially small.
//!
//! **Parameters:** num_buckets ∈ {2^16, 2^18}. bucket_size ∈ {2, 4}. fingerprint_bits = 12.
//! 3 warmup + 10 measured trials per hit rate.
//!
//! **Output:** `results/lookup_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, hit_rate_pct, load_factor, num_queries, mean_mops, min_mops, max_mops, stddev_mops

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
const HIT_RATES: &[u64] = &[50];

fn build_queries(inserted: u64, num_queries: u64, hit_pct: u64) -> Vec<u64> {
    let num_hits = num_queries * hit_pct / 100;
    let num_misses = num_queries - num_hits;
    let mut queries: Vec<u64> = (0..num_hits).map(|i| i % inserted).collect();
    queries.extend(inserted..inserted + num_misses);
    queries
}

macro_rules! bench_lookup {
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
                // Fill the filter until full
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
                let inserted = i;
                let lf = filter.load_factor();
                let num_queries = (num_buckets as u64 * bucket_size as u64) / 2;

                println!(
                    "  {:<20} num_buckets={:<8} bucket_size={} | inserted={} lf={:.4} q={}",
                    $label, num_buckets, bucket_size, inserted, lf, num_queries
                );

                for &hit_rate in HIT_RATES {
                    let queries = build_queries(inserted, num_queries, hit_rate);

                    // Warmup
                    for _ in 0..WARMUP_TRIALS {
                        for &key in &queries {
                            std::hint::black_box(filter.contain(key.to_le_bytes()));
                        }
                    }

                    // Measure
                    let mut throughputs = Vec::with_capacity(MEASURE_TRIALS);
                    for _trial in 0..MEASURE_TRIALS {
                        let start = Instant::now();
                        for &key in &queries {
                            std::hint::black_box(filter.contain(key.to_le_bytes()));
                        }
                        let elapsed_ns = start.elapsed().as_nanos() as f64;
                        let mops = num_queries as f64 / elapsed_ns * 1000.0;
                        throughputs.push(mops);
                    }
                    let mops_stats = helpers::compute_stats(&throughputs);
                    writeln!(
                        $csv,
                        "{},{},{},{},{},{:.6},{},{:.4},{:.4},{:.4},{:.4}",
                        $scheme,
                        $arity,
                        num_buckets,
                        bucket_size,
                        hit_rate,
                        lf,
                        num_queries,
                        mops_stats.mean,
                        mops_stats.min,
                        mops_stats.max,
                        mops_stats.stddev
                    )
                    .unwrap();
                    println!(
                        "    hit={}% | mean={:.3} std={:.3} Mops",
                        hit_rate, mops_stats.mean, mops_stats.stddev
                    );
                }
            }
        }
    }};
}

#[allow(clippy::cognitive_complexity)] // bench main(): linear CLI-parse + dispatch plumbing
fn main() {
    let mut csv = helpers::csv_writer(
        "lookup_throughput.csv",
        "scheme,arity,num_buckets,bucket_size,hit_rate_pct,load_factor,num_queries,mean_mops,min_mops,max_mops,stddev_mops",
    );

    println!("=== Lookup Throughput (insert until full, q = num_buckets*bucket_size/2) ===");
    println!(
        "Config: fingerprint_bits={}, max_kicks={}, warmup={}, trials={}",
        FINGERPRINT_BITS, MAX_KICKS, WARMUP_TRIALS, MEASURE_TRIALS
    );

    // Standard 3-ary num_buckets (power of 3) sized comparably to NUM_BUCKETS_VALUES.
    let pow3_num_buckets: &[u32] = &[3u32.pow(13)]; // 19683, 59049
                                                    // Standard 4-ary num_buckets (power of 4) sized comparably to NUM_BUCKETS_VALUES.
    let pow4_num_buckets: &[u32] = &[4u32.pow(10)]; // 65536, 262144

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

            bench_lookup!(
                csv,
                "Segmented 2-ary",
                Segmented2aryCuckooFilter,
                "segmented",
                2,
                num_buckets,
                bucket_size
            );
            bench_lookup!(
                csv,
                "Standard 2-ary",
                Standard2aryCuckooFilter,
                "standard",
                2,
                num_buckets,
                bucket_size
            );
            bench_lookup!(
                csv,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                3,
                seg3_num_buckets,
                bucket_size
            );
            bench_lookup!(
                csv,
                "Standard 3-ary",
                Standard3aryCuckooFilter,
                "standard",
                3,
                std3_num_buckets,
                bucket_size
            );
            bench_lookup!(
                csv,
                "Segmented 4-ary",
                Segmented4aryCuckooFilter,
                "segmented",
                4,
                num_buckets,
                bucket_size
            );
            bench_lookup!(
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

    println!("\nResults written to results/lookup_throughput.csv");
}
