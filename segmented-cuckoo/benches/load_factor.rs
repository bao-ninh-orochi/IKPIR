//! **Intent:** Measure maximum load factor — how full each scheme can get before rejecting insertions.
//!
//! **Method:** Insert sequential u64 keys (LE bytes) until `add` returns `TableFull`. Record
//! the load factor. Repeat `TRIALS` trials per config to capture variance from the randomised
//! eviction chain.
//!
//! **Sweep:** MAX_KICKS is swept over `KICKS_VALUES` (500, 1000, …, 5000) to show how the
//! kick budget affects achievable load factor.
//!
//! **Parameters:**
//! num_buckets ∈ {3·2^12, 3·2^14, 3·2^16, 3·2^18} (segmented 3-ary, num_buckets/3 power of 2).
//! num_buckets ∈ {2^14, 2^16, 2^18, 2^20} (segmented 2-ary, segmented 4-ary, standard 2-ary; power of 2).
//! num_buckets ∈ {3^9, 3^10, 3^11, 3^12} (standard 3-ary, power of 3).
//! num_buckets ∈ {4^7, 4^8, 4^9, 4^10} (standard 4-ary, power of 4).
//! bucket_size ∈ {1, 2, 3, 4}. fingerprint_bits = 12.
//!
//! **Output:** `results/load_factor.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, fingerprint_bits, max_kicks, mean_lf, min_lf, max_lf, stddev_lf

mod helpers;

use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooFilter, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter,
    Standard2aryCuckooFilter, Standard3aryCuckooFilter, Standard4aryCuckooFilter,
};
use std::io::Write;

const FINGERPRINT_BITS: u32 = 32;
const TRIALS: usize = 20;
const KICKS_VALUES: &[u32] = &[2500];

fn header(title: &str) {
    println!("=== Maximum Load Factor — {} ===", title);
    println!(
        "{:<14} {:<12} {:<16} {:<8} {:<8} {:<10} {:<10} {:<10}",
        "num_buckets",
        "bucket_size",
        "fingerprint_bits",
        "kicks",
        "trials",
        "min_lf",
        "mean_lf",
        "max_lf"
    );
    println!("{}", "-".repeat(96));
}

macro_rules! bench_load_factor {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity:expr, $sizes:expr) => {{
        header($label);
        for &(num_buckets, num_buckets_label) in $sizes {
            for &bucket_size in &[1u32, 2, 3, 4] {
                for &max_kicks in KICKS_VALUES {
                    let mut lfs = Vec::with_capacity(TRIALS);
                    for _trial in 0..TRIALS {
                        let mut filter =
                            <$filter_ty>::new(num_buckets, bucket_size, FINGERPRINT_BITS).unwrap();
                        filter.set_max_kicks(max_kicks);
                        let mut i = 0u64;
                        loop {
                            match filter.insert(i.to_le_bytes()) {
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
                        num_buckets,
                        bucket_size,
                        FINGERPRINT_BITS,
                        max_kicks,
                        stats.mean,
                        stats.min,
                        stats.max,
                        stats.stddev
                    )
                    .unwrap();
                    println!(
                        "{:<14} {:<12} {:<16} {:<8} {:<8} {:<10.4} {:<10.4} {:<10.4}",
                        num_buckets_label,
                        bucket_size,
                        FINGERPRINT_BITS,
                        max_kicks,
                        TRIALS,
                        stats.min,
                        stats.mean,
                        stats.max
                    );
                }
            }
        }
        println!();
    }};
}

#[allow(clippy::cognitive_complexity)] // bench main(): linear CLI-parse + dispatch plumbing
fn main() {
    let mut csv = helpers::csv_writer(
        "load_factor.csv",
        "scheme,arity,num_buckets,bucket_size,fingerprint_bits,max_kicks,mean_lf,min_lf,max_lf,stddev_lf",
    );

    // Segmented 2-ary, segmented 4-ary, standard 2-ary: num_buckets must be a power of 2.
    // (Segmented 2-ary requires ≥ 2; segmented 4-ary requires ≥ 4; standard 2-ary requires ≥ 1.)
    let pow2: Vec<(u32, &str)> = vec![
        (1 << 16, "2^16"), // 65536
        (1 << 18, "2^18"), // 262144
        (1 << 20, "2^20"), // 1048576
    ];
    // Segmented 3-ary: num_buckets must equal 3·2^t (num_buckets/3 a power of 2), ≥ 3.
    let seg3: Vec<(u32, &str)> = vec![
        (3 * (1 << 15), "3\u{b7}2^15"), // 49152
        (3 * (1 << 17), "3\u{b7}2^17"), // 196608
        (3 * (1 << 19), "3\u{b7}2^19"), // 786432
    ];
    // Standard 3-ary: num_buckets must be a power of 3 (3^t).
    let pow3: Vec<(u32, &str)> = vec![
        (3u32.pow(10), "3^11"), // 59049
        (3u32.pow(11), "3^12"), // 177147
        (3u32.pow(13), "3^13"), // 531441
    ];
    // Standard 4-ary: num_buckets must be a power of 4 (4^t).
    let pow4: Vec<(u32, &str)> = vec![
        (4u32.pow(8), "4^8"),   // 65536
        (4u32.pow(9), "4^9"),   // 262144
        (4u32.pow(10), "4^10"), // 1048576
    ];

    bench_load_factor!(
        csv,
        "Segmented 2-ary",
        Segmented2aryCuckooFilter,
        "segmented",
        2,
        &pow2
    );
    bench_load_factor!(
        csv,
        "Standard 2-ary",
        Standard2aryCuckooFilter,
        "standard",
        2,
        &pow2
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
        "Standard 3-ary",
        Standard3aryCuckooFilter,
        "standard",
        3,
        &pow3
    );
    bench_load_factor!(
        csv,
        "Segmented 4-ary",
        Segmented4aryCuckooFilter,
        "segmented",
        4,
        &pow2
    );
    bench_load_factor!(
        csv,
        "Standard 4-ary",
        Standard4aryCuckooFilter,
        "standard",
        4,
        &pow4
    );

    println!("Results written to results/load_factor.csv");
}
