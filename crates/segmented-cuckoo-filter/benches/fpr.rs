//! **Intent:** Measure actual false positive rate (FPR) as fingerprint bit width varies, and
//! compare it against the theoretical bound `2b / 2^fp_bits`.
//!
//! **Method:** For each (arity, n, b), sweep fp_bits from `min_fp_bits(b)` to 32. At each
//! fp_bits: insert until full, query q = 10·n·b never-inserted items, count false positives.
//!
//! **Rationale:** The standard theoretical FPR is `2b / 2^f`. This bench verifies that the
//! segmented variant achieves the same FPR — segmentation changes index placement but not
//! fingerprint collision probability, so FPR should be identical. Using q = 10·n·b gives
//! ≥10× the filter capacity in miss lookups, yielding enough expected false positives for a
//! statistically meaningful rate even at low FPR. A single run per fp_bits config is
//! sufficient because FPR variance is low with q this large.
//!
//! **Parameters:** n = 2^18. b ∈ {1, 2, 3, 4}. fp_bits swept from min_fp_bits(b) to 32.
//! max_kicks = 500.
//!
//! **Output:** `results/fpr/arity{a}_n{n}_b{b}.csv` (one file per (arity, n, b))
//! Columns: fp_bits, scheme, n, load_factor, num_inserted, num_queries, false_positives, fpr_pct, theoretical_pct

mod helpers;

use segmented_cuckoo_filter::{
    CuckooError, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter, SegmentedCuckooFilter,
    Standard3aryCuckooFilter, Standard4aryCuckooFilter, StandardCuckooFilter,
};
use std::io::Write;

const MAX_KICKS: u32 = 500;

/// n values to test (power of 2).
const N_VALUES: &[u32] = &[1 << 18];
/// Bucket sizes to test.
const B_VALUES: &[u32] = &[1, 2, 3, 4];

/// Minimum valid fingerprint bits for a given b: floor(log2(2b)) + 1.
fn min_fp_bits(b: u32) -> u32 {
    (2 * b).ilog2() + 1
}

macro_rules! run_fpr {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity_val:expr, $n:expr, $b:expr, $fp_bits:expr) => {{
        let n: u32 = $n;
        let b: u32 = $b;
        let fp_bits: u32 = $fp_bits;

        match <$filter_ty>::new(n, b, fp_bits) {
            Err(e) => {
                eprintln!("    Skip {} n={} b={} fp={}: {}", $label, n, b, fp_bits, e);
            }
            Ok(mut filter) => {
                filter.set_max_kicks(MAX_KICKS);

                // Insert until full
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

                // Query with non-inserted items
                let q = 10 * n as u64 * b as u64;
                let mut fp_count = 0u64;
                for j in inserted..inserted + q {
                    if filter.contain(j.to_le_bytes()) {
                        fp_count += 1;
                    }
                }
                let fpr_pct = fp_count as f64 / q as f64 * 100.0;
                let arity: u32 = $arity_val;
                let theoretical_pct = arity as f64 * b as f64 / (1u64 << fp_bits) as f64 * 100.0;

                writeln!(
                    $csv,
                    "{},{},{},{:.6},{},{},{},{:.6},{:.6}",
                    fp_bits, $scheme, n, lf, inserted, q, fp_count, fpr_pct, theoretical_pct
                )
                .unwrap();

                println!(
                    "    fp={:<3} {:<12} lf={:.4} fpr={:.4}% theo={:.4}%",
                    fp_bits, $label, lf, fpr_pct, theoretical_pct
                );
            }
        }
    }};
}

fn main() {
    println!("=== False Positive Rate (insert until full, sweep fp_bits) ===");
    println!("Config: max_kicks={}", MAX_KICKS);

    // pow3 n comparable to N_VALUES (2^18 = 262144 -> 3^11 = 177147)
    let std3_n = 3u32.pow(11); // 177147
                               // pow4 n comparable to N_VALUES (2^18 = 262144 -> 4^9 = 262144)
    let std4_n = 4u32.pow(9); // 262144

    for &n in N_VALUES {
        for &b in B_VALUES {
            let min_fp = min_fp_bits(b);
            let seg3_n = 3 * (1u32 << (n / 3).ilog2());

            // ── 2-ary ──
            {
                let arity = 2;
                let filename = format!("fpr/arity{}_n{}_b{}.csv", arity, n, b);
                let mut csv = helpers::csv_writer(
                    &filename,
                    "fp_bits,scheme,n,load_factor,num_inserted,num_queries,\
                     false_positives,fpr_pct,theoretical_pct",
                );
                println!(
                    "\n--- arity={}, n={}, b={} (fp_bits={}..32) ---",
                    arity, n, b, min_fp
                );
                for fp_bits in min_fp..=32 {
                    run_fpr!(
                        csv,
                        "standard",
                        StandardCuckooFilter,
                        "standard",
                        2,
                        n,
                        b,
                        fp_bits
                    );
                    run_fpr!(
                        csv,
                        "segmented",
                        SegmentedCuckooFilter,
                        "segmented",
                        2,
                        n,
                        b,
                        fp_bits
                    );
                }
            }

            // ── 3-ary ──
            {
                let arity = 3;
                let filename = format!("fpr/arity{}_n{}_b{}.csv", arity, n, b);
                let mut csv = helpers::csv_writer(
                    &filename,
                    "fp_bits,scheme,n,load_factor,num_inserted,num_queries,\
                     false_positives,fpr_pct,theoretical_pct",
                );
                println!(
                    "\n--- arity={}, n={} (seg3_n={}, std3_n={}), b={} (fp_bits={}..32) ---",
                    arity, n, seg3_n, std3_n, b, min_fp
                );
                for fp_bits in min_fp..=32 {
                    run_fpr!(
                        csv,
                        "standard",
                        Standard3aryCuckooFilter,
                        "standard",
                        3,
                        std3_n,
                        b,
                        fp_bits
                    );
                    run_fpr!(
                        csv,
                        "segmented",
                        Segmented3aryCuckooFilter,
                        "segmented",
                        3,
                        seg3_n,
                        b,
                        fp_bits
                    );
                }
            }

            // ── 4-ary ──
            {
                let arity = 4;
                let filename = format!("fpr/arity{}_n{}_b{}.csv", arity, n, b);
                let mut csv = helpers::csv_writer(
                    &filename,
                    "fp_bits,scheme,n,load_factor,num_inserted,num_queries,\
                     false_positives,fpr_pct,theoretical_pct",
                );
                println!(
                    "\n--- arity={}, n={} (std4_n={}), b={} (fp_bits={}..32) ---",
                    arity, n, std4_n, b, min_fp
                );
                for fp_bits in min_fp..=32 {
                    run_fpr!(
                        csv,
                        "standard",
                        Standard4aryCuckooFilter,
                        "standard",
                        4,
                        std4_n,
                        b,
                        fp_bits
                    );
                    run_fpr!(
                        csv,
                        "segmented",
                        Segmented4aryCuckooFilter,
                        "segmented",
                        4,
                        n,
                        b,
                        fp_bits
                    );
                }
            }
        }
    }

    println!("\nResults written to results/fpr/");
}
