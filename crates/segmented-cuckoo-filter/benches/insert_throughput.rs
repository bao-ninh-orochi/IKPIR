//! **Intent:** Measure raw insert speed for each scheme at its natural load factor.
//!
//! **Method:** Insert sequential keys until `TableFull`, timing the full loop. Divide total
//! insertions by elapsed ns to get Mops/sec. 3 warmup trials, then 10 timed trials.
//!
//! **Rationale:** "Insert until full" is the only fair comparison method. Inserting a fixed
//! number of items (e.g. 1 M) would confound throughput with load factor: the same 1 M items
//! might fill standard 3-ary to ~90% (near full, expensive kicking) but segmented 3-ary to
//! only ~70% (sparse, cheap kicking), making the standard variant appear artificially slow.
//! Measuring over the full fill trajectory puts all schemes under comparable conditions.
//!
//! **Parameters:** n ∈ {2^16, 2^18, 2^20}. b ∈ {1, 2, 3, 4}. fp_bits = 12.
//! 3 warmup + 10 measured trials.
//!
//! **Output:** `results/insert_throughput.csv`
//! Columns: scheme, arity, n, b, mean_inserted, mean_lf, mean_duration_ns, mean_mops, min_mops, max_mops, stddev_mops

mod helpers;

use segmented_cuckoo_filter::{
    CuckooError, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter, SegmentedCuckooFilter,
    Standard3aryCuckooFilter, Standard4aryCuckooFilter, StandardCuckooFilter,
};
use std::io::Write;
use std::time::Instant;

const MAX_KICKS: u32 = 2500;
const FP_BITS: u32 = 12;
const WARMUP_TRIALS: usize = 3;
const MEASURE_TRIALS: usize = 10;

const N_VALUES: &[u32] = &[1 << 16, 1 << 18, 1 << 20];
const B_VALUES: &[u32] = &[1, 2, 3, 4];

macro_rules! bench_insert {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity:expr, $n:expr, $b:expr) => {{
        let n: u32 = $n;
        let b: u32 = $b;

        match <$filter_ty>::new(n, b, FP_BITS) {
            Err(e) => {
                eprintln!("  Skip {} n={} b={}: {}", $label, n, b, e);
            }
            Ok(_) => {
                // Warmup
                for _ in 0..WARMUP_TRIALS {
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
                }

                // Measure
                let mut throughputs = Vec::with_capacity(MEASURE_TRIALS);
                let mut inserted_vals = Vec::with_capacity(MEASURE_TRIALS);
                let mut lf_vals = Vec::with_capacity(MEASURE_TRIALS);
                let mut duration_vals = Vec::with_capacity(MEASURE_TRIALS);
                for _trial in 0..MEASURE_TRIALS {
                    let mut filter = <$filter_ty>::new(n, b, FP_BITS).unwrap();
                    filter.set_max_kicks(MAX_KICKS);
                    let start = Instant::now();
                    let mut i = 0u64;
                    loop {
                        match filter.add(i.to_le_bytes()) {
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
                    n,
                    b,
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
                    "  {:<20} n={:<10} b={:<3} | mean={:<8.3} std={:<8.3} Mops",
                    $label, n, b, mops_stats.mean, mops_stats.stddev
                );
            }
        }
    }};
}

fn main() {
    let mut csv = helpers::csv_writer(
        "insert_throughput.csv",
        "scheme,arity,n,b,mean_inserted,mean_lf,mean_duration_ns,mean_mops,min_mops,max_mops,stddev_mops",
    );

    println!("=== Insert Throughput (insert until full) ===");
    println!(
        "Config: fp_bits={}, max_kicks={}, warmup={}, trials={}",
        FP_BITS, MAX_KICKS, WARMUP_TRIALS, MEASURE_TRIALS
    );

    // pow3 n values comparable in size to N_VALUES
    let pow3_values: &[u32] = &[3u32.pow(9), 3u32.pow(10), 3u32.pow(11)]; // ~19683, 59049, 177147
                                                                          // pow4 n values comparable in size to N_VALUES
    let pow4_values: &[u32] = &[4u32.pow(8), 4u32.pow(9), 4u32.pow(10)]; // 65536, 262144, 1048576

    for (idx, &n) in N_VALUES.iter().enumerate() {
        for &b in B_VALUES {
            let seg3_n = 3 * (1u32 << (n / 3).ilog2());
            let std3_n = pow3_values[idx];
            let std4_n = pow4_values[idx];

            println!("\n--- n={}, b={} ---", n, b);

            bench_insert!(
                csv,
                "Standard 2-ary",
                StandardCuckooFilter,
                "standard",
                2,
                n,
                b
            );
            bench_insert!(
                csv,
                "Segmented 2-ary",
                SegmentedCuckooFilter,
                "segmented",
                2,
                n,
                b
            );
            bench_insert!(
                csv,
                "Standard 3-ary",
                Standard3aryCuckooFilter,
                "standard",
                3,
                std3_n,
                b
            );
            bench_insert!(
                csv,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                3,
                seg3_n,
                b
            );
            bench_insert!(
                csv,
                "Standard 4-ary",
                Standard4aryCuckooFilter,
                "standard",
                4,
                std4_n,
                b
            );
            bench_insert!(
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

    println!("\nResults written to results/insert_throughput.csv");
}
