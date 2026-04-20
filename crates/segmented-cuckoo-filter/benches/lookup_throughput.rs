//! **Intent:** Measure query speed at varying hit rates on a full filter.
//!
//! **Method:** Fill until full, then for each hit rate in {0, 25, 50, 75, 100}%: build a
//! query set of q = n*b/2 keys mixing inserted items (hits) and never-inserted items (misses)
//! in the given ratio. Time q `contain` calls. 3 warmup + 10 timed trials per hit rate.
//!
//! **Rationale:** Hit rate matters because a lookup that finds a matching tag can terminate
//! early (first matching bucket), while a miss must check all k buckets. The 5-point sweep
//! reveals whether this early-exit benefit is significant. Testing at full load ensures
//! realistic cache pressure and FPR conditions. q = n*b/2 keeps runtime predictable without
//! being trivially small.
//!
//! **Parameters:** n ∈ {2^16, 2^18}. b ∈ {2, 4}. fp_bits = 12.
//! 3 warmup + 10 measured trials per hit rate.
//!
//! **Output:** `results/lookup_throughput.csv`
//! Columns: scheme, arity, n, b, hit_rate_pct, load_factor, num_queries, mean_mops, min_mops, max_mops, stddev_mops

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

const N_VALUES: &[u32] = &[1 << 16, 1 << 18];
const B_VALUES: &[u32] = &[2, 4];
const HIT_RATES: &[u64] = &[0, 25, 50, 75, 100];

fn build_queries(inserted: u64, num_queries: u64, hit_pct: u64) -> Vec<u64> {
    let num_hits = num_queries * hit_pct / 100;
    let num_misses = num_queries - num_hits;
    let mut queries: Vec<u64> = (0..num_hits).map(|i| i % inserted).collect();
    queries.extend(inserted..inserted + num_misses);
    queries
}

macro_rules! bench_lookup {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity:expr, $n:expr, $b:expr) => {{
        let n: u32 = $n;
        let b: u32 = $b;

        match <$filter_ty>::new(n, b, FP_BITS) {
            Err(e) => {
                eprintln!("  Skip {} n={} b={}: {}", $label, n, b, e);
            }
            Ok(_) => {
                // Fill the filter until full
                let mut filter = <$filter_ty>::new(n, b, FP_BITS).unwrap();
                filter.set_max_kicks(MAX_KICKS);
                let mut i = 0u64;
                loop {
                    match filter.add(i.to_le_bytes()) {
                        Ok(()) => i += 1,
                        Err(CuckooError::TableFull) => break,
                        Err(e) => panic!("{}", e),
                    }
                }
                let inserted = i;
                let lf = filter.load_factor();
                let num_queries = (n as u64 * b as u64) / 2;

                println!(
                    "  {:<20} n={:<8} b={} | inserted={} lf={:.4} q={}",
                    $label, n, b, inserted, lf, num_queries
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
                        n,
                        b,
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

fn main() {
    let mut csv = helpers::csv_writer(
        "lookup_throughput.csv",
        "scheme,arity,n,b,hit_rate_pct,load_factor,num_queries,mean_mops,min_mops,max_mops,stddev_mops",
    );

    println!("=== Lookup Throughput (insert until full, q = n*b/2) ===");
    println!(
        "Config: fp_bits={}, max_kicks={}, warmup={}, trials={}",
        FP_BITS, MAX_KICKS, WARMUP_TRIALS, MEASURE_TRIALS
    );

    let pow3_values: &[u32] = &[3u32.pow(9), 3u32.pow(10)]; // ~19683, 59049
    let pow4_values: &[u32] = &[4u32.pow(8), 4u32.pow(9)]; // 65536, 262144

    for (idx, &n) in N_VALUES.iter().enumerate() {
        for &b in B_VALUES {
            let seg3_n = 3 * (1u32 << (n / 3).ilog2());
            let std3_n = pow3_values[idx];
            let std4_n = pow4_values[idx];

            println!("\n--- n={}, b={} ---", n, b);

            bench_lookup!(
                csv,
                "Standard 2-ary",
                StandardCuckooFilter,
                "standard",
                2,
                n,
                b
            );
            bench_lookup!(
                csv,
                "Segmented 2-ary",
                SegmentedCuckooFilter,
                "segmented",
                2,
                n,
                b
            );
            bench_lookup!(
                csv,
                "Standard 3-ary",
                Standard3aryCuckooFilter,
                "standard",
                3,
                std3_n,
                b
            );
            bench_lookup!(
                csv,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                3,
                seg3_n,
                b
            );
            bench_lookup!(
                csv,
                "Standard 4-ary",
                Standard4aryCuckooFilter,
                "standard",
                4,
                std4_n,
                b
            );
            bench_lookup!(
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

    println!("\nResults written to results/lookup_throughput.csv");
}
