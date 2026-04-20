//! **Intent:** Measure maximum load factor — how full each scheme can get before rejecting insertions.
//!
//! **Method:** Insert sequential u64 keys (LE bytes) until `add` returns `TableFull`. Record
//! the load factor. Repeat `TRIALS` trials per config to capture variance from the randomised
//! eviction chain.
//!
//! **Sweep:** MAX_KICKS is swept over `KICKS_VALUES` (500, 1000, …, 5000) to show how the
//! kick budget affects achievable load factor.
//!
//! **Parameters:** n ∈ {2^14, 2^16, 2^18, 2^20} (pow2 schemes);
//! n ∈ {3^8 .. 3^11} (standard 3-ary).
//! n ∈ {3·2^12 .. 3·2^18} (segmented 3-ary).
//! b ∈ {1, 2, 3, 4}. fp_bits = 12.
//!
//! **Output:** `results/load_factor.csv`
//! Columns: scheme, arity, n, b, fp_bits, max_kicks, mean_lf, min_lf, max_lf, stddev_lf

mod helpers;

use segmented_cuckoo_filter::{
    CuckooError, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter, SegmentedCuckooFilter,
    Standard3aryCuckooFilter, Standard4aryCuckooFilter, StandardCuckooFilter,
};
use std::io::Write;

const FP_BITS: u32 = 12;
const TRIALS: usize = 20;
const KICKS_VALUES: &[u32] = &[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000];

fn header(title: &str) {
    println!("=== Maximum Load Factor — {} ===", title);
    println!(
        "{:<14} {:<6} {:<10} {:<8} {:<8} {:<10} {:<10} {:<10}",
        "n", "b", "fp_bits", "kicks", "trials", "min_lf", "mean_lf", "max_lf"
    );
    println!("{}", "-".repeat(86));
}

macro_rules! bench_load_factor {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity:expr, $sizes:expr) => {{
        header($label);
        for &(n, n_label) in $sizes {
            for &b in &[1u32, 2, 3, 4] {
                for &max_kicks in KICKS_VALUES {
                    let mut lfs = Vec::with_capacity(TRIALS);
                    for _trial in 0..TRIALS {
                        let mut filter = <$filter_ty>::new(n, b, FP_BITS).unwrap();
                        filter.set_max_kicks(max_kicks);
                        let mut i = 0u64;
                        loop {
                            match filter.add(i.to_le_bytes()) {
                                Ok(()) => i += 1,
                                Err(CuckooError::TableFull) => break,
                                Err(e) => panic!("unexpected error: {}", e),
                            }
                        }
                        lfs.push(filter.load_factor());
                    }
                    let stats = helpers::compute_stats(&lfs);
                    writeln!(
                        $csv,
                        "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
                        $scheme,
                        $arity,
                        n,
                        b,
                        FP_BITS,
                        max_kicks,
                        stats.mean,
                        stats.min,
                        stats.max,
                        stats.stddev
                    )
                    .unwrap();
                    println!(
                        "{:<14} {:<6} {:<10} {:<8} {:<8} {:<10.4} {:<10.4} {:<10.4}",
                        n_label, b, FP_BITS, max_kicks, TRIALS, stats.min, stats.mean, stats.max
                    );
                }
            }
        }
        println!();
    }};
}

fn main() {
    let mut csv = helpers::csv_writer(
        "load_factor.csv",
        "scheme,arity,n,b,fp_bits,max_kicks,mean_lf,min_lf,max_lf,stddev_lf",
    );

    let pow2: Vec<(u32, &str)> = vec![
        (1 << 14, "2^14"), // 16384
        (1 << 16, "2^16"), // 65536
        (1 << 18, "2^18"), // 262144
        (1 << 20, "2^20"), // 1048576
    ];
    let seg3: Vec<(u32, &str)> = vec![
        (3 * (1 << 12), "3\u{b7}2^12"), // 12288
        (3 * (1 << 14), "3\u{b7}2^14"), // 49152
        (3 * (1 << 16), "3\u{b7}2^16"), // 196608
        (3 * (1 << 18), "3\u{b7}2^18"), // 786432
    ];
    // Standard 3-ary: n must be power of 3 (3^k)
    let pow3: Vec<(u32, &str)> = vec![
        (3u32.pow(9), "3^9"),   // 19683
        (3u32.pow(10), "3^10"), // 59049
        (3u32.pow(11), "3^11"), // 177147
        (3u32.pow(12), "3^12"), // 531441
    ];
    // Standard 4-ary: n must be power of 4 (4^k)
    let pow4: Vec<(u32, &str)> = vec![
        (4u32.pow(7), "4^7"),   // 16384
        (4u32.pow(8), "4^8"),   // 65536
        (4u32.pow(9), "4^9"),   // 262144
        (4u32.pow(10), "4^10"), // 1048576
    ];

    bench_load_factor!(
        csv,
        "Standard 2-ary",
        StandardCuckooFilter,
        "standard",
        2,
        &pow2
    );
    bench_load_factor!(
        csv,
        "Segmented 2-ary",
        SegmentedCuckooFilter,
        "segmented",
        2,
        &pow2
    );
    bench_load_factor!(
        csv,
        "Standard 3-ary",
        Standard3aryCuckooFilter,
        "standard",
        3,
        &pow3
    );
    bench_load_factor!(
        csv,
        "Segmented 3-ary",
        Segmented3aryCuckooFilter,
        "segmented",
        3,
        &seg3
    );
    bench_load_factor!(
        csv,
        "Standard 4-ary",
        Standard4aryCuckooFilter,
        "standard",
        4,
        &pow4
    );
    bench_load_factor!(
        csv,
        "Segmented 4-ary",
        Segmented4aryCuckooFilter,
        "segmented",
        4,
        &pow2
    );

    println!("Results written to results/load_factor.csv");
}
