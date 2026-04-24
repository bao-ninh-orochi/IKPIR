//! **Intent:** Measure actual false positive rate (FPR) as fingerprint bit width varies, and
//! compare it against the theoretical bound `arity · bucket_size / 2^fingerprint_bits`.
//!
//! **Method:** For each (arity, num_buckets, bucket_size), sweep fingerprint_bits from
//! `min_fingerprint_bits(arity, bucket_size)` to 32. At each fingerprint_bits: insert until full,
//! query q = 10·num_buckets·bucket_size never-inserted items, count false positives.
//!
//! **Rationale:** The theoretical FPR is `arity · bucket_size / 2^fingerprint_bits`. This bench
//! verifies that the segmented variant achieves the same FPR — segmentation changes index
//! placement but not fingerprint collision probability, so FPR should be identical. Using
//! q = 10·num_buckets·bucket_size gives ≥10× the filter capacity in miss lookups, yielding enough
//! expected false positives for a statistically meaningful rate even at low FPR. A single run per
//! fingerprint_bits config is sufficient because FPR variance is low with q this large.
//!
//! **Parameters:** num_buckets = 2^18 (segmented 2-ary, segmented 4-ary, standard 2-ary);
//! seg3 num_buckets = 3·2^16 = 196608; standard 3-ary num_buckets = 3^11 = 177147;
//! standard 4-ary num_buckets = 4^9 = 262144. bucket_size ∈ {1, 2, 3, 4}.
//! fingerprint_bits swept from `min_fingerprint_bits(arity, bucket_size)` to 32. max_kicks = 500.
//!
//! **Output:** `results/fpr/arity{arity}_num_buckets{num_buckets}_bucket_size{bucket_size}.csv`
//! Columns: fingerprint_bits, scheme, num_buckets, load_factor, num_inserted, num_queries, false_positives, fpr_pct, theoretical_pct

mod helpers;

use segmented_cuckoo_filter::{
    CuckooError, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter, Segmented2aryCuckooFilter,
    Standard3aryCuckooFilter, Standard4aryCuckooFilter, Standard2aryCuckooFilter,
};
use std::io::Write;

const MAX_KICKS: u32 = 500;

/// num_buckets values to test (power of 2; valid for segmented 2-ary, segmented 4-ary, standard 2-ary).
const NUM_BUCKETS_VALUES: &[u32] = &[1 << 18];
/// Bucket sizes to test.
const BUCKET_SIZE_VALUES: &[u32] = &[1, 2, 3, 4];

/// Minimum valid fingerprint bits: `floor(log2(arity * bucket_size)) + 1` (matches `validate_common_params` in /src).
fn min_fingerprint_bits(arity: u32, bucket_size: u32) -> u32 {
    (arity * bucket_size).ilog2() + 1
}

macro_rules! run_fpr {
    ($csv:expr, $label:expr, $filter_ty:ty, $scheme:expr, $arity_val:expr, $num_buckets:expr, $bucket_size:expr, $fingerprint_bits:expr) => {{
        let num_buckets: u32 = $num_buckets;
        let bucket_size: u32 = $bucket_size;
        let fingerprint_bits: u32 = $fingerprint_bits;

        match <$filter_ty>::new(num_buckets, bucket_size, fingerprint_bits) {
            Err(e) => {
                eprintln!("    Skip {} num_buckets={} bucket_size={} fp={}: {}", $label, num_buckets, bucket_size, fingerprint_bits, e);
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
                let q = 10 * num_buckets as u64 * bucket_size as u64;
                let mut fp_count = 0u64;
                for j in inserted..inserted + q {
                    if filter.contain(j.to_le_bytes()) {
                        fp_count += 1;
                    }
                }
                let fpr_pct = fp_count as f64 / q as f64 * 100.0;
                let arity: u32 = $arity_val;
                let theoretical_pct =
                    arity as f64 * bucket_size as f64 / (1u64 << fingerprint_bits) as f64 * 100.0;

                writeln!(
                    $csv,
                    "{},{},{},{:.6},{},{},{},{:.6},{:.6}",
                    fingerprint_bits, $scheme, num_buckets, lf, inserted, q, fp_count, fpr_pct,
                    theoretical_pct
                )
                .unwrap();

                println!(
                    "    fingerprint_bits={:<3} {:<12} lf={:.4} fpr={:.4}% theo={:.4}%",
                    fingerprint_bits, $label, lf, fpr_pct, theoretical_pct
                );
            }
        }
    }};
}

fn main() {
    println!("=== False Positive Rate (insert until full, sweep fingerprint_bits) ===");
    println!("Config: max_kicks={}", MAX_KICKS);

    // Standard 3-ary num_buckets (power of 3) sized comparably to NUM_BUCKETS_VALUES
    // (2^18 = 262144 -> 3^11 = 177147).
    let std3_num_buckets = 3u32.pow(11); // 177147
    // Standard 4-ary num_buckets (power of 4) sized comparably to NUM_BUCKETS_VALUES
    // (2^18 = 262144 -> 4^9 = 262144).
    let std4_num_buckets = 4u32.pow(9); // 262144

    for &num_buckets in NUM_BUCKETS_VALUES {
        for &bucket_size in BUCKET_SIZE_VALUES {
            // Segmented 3-ary requires num_buckets = 3·2^t; pick the largest valid value
            // that fits inside the 2^k window so the comparison stays at a similar size.
            let seg3_num_buckets = 3 * (1u32 << (num_buckets / 3).ilog2());

            // ── 2-ary ──
            {
                let arity = 2;
                let min_fingerprint_bits = min_fingerprint_bits(arity, bucket_size);
                let filename = format!(
                    "fpr/arity{}_num_buckets{}_bucket_size{}.csv",
                    arity, num_buckets, bucket_size
                );
                let mut csv = helpers::csv_writer(
                    &filename,
                    "fingerprint_bits,scheme,num_buckets,load_factor,num_inserted,num_queries,\
                     false_positives,fpr_pct,theoretical_pct",
                );
                println!(
                    "\n--- arity={}, num_buckets={}, bucket_size={} (fingerprint_bits={}..32) ---",
                    arity, num_buckets, bucket_size, min_fingerprint_bits
                );
                for fingerprint_bits in min_fingerprint_bits..=32 {
                    run_fpr!(
                        csv,
                        "segmented",
                        Segmented2aryCuckooFilter,
                        "segmented",
                        2,
                        num_buckets,
                        bucket_size,
                        fingerprint_bits
                    );
                    run_fpr!(
                        csv,
                        "standard",
                        Standard2aryCuckooFilter,
                        "standard",
                        2,
                        num_buckets,
                        bucket_size,
                        fingerprint_bits
                    );
                }
            }

            // ── 3-ary ──
            {
                let arity = 3;
                let min_fingerprint_bits = min_fingerprint_bits(arity, bucket_size);
                let filename = format!(
                    "fpr/arity{}_num_buckets{}_bucket_size{}.csv",
                    arity, num_buckets, bucket_size
                );
                let mut csv = helpers::csv_writer(
                    &filename,
                    "fingerprint_bits,scheme,num_buckets,load_factor,num_inserted,num_queries,\
                     false_positives,fpr_pct,theoretical_pct",
                );
                println!(
                    "\n--- arity={}, num_buckets={} (seg3_num_buckets={}, std3_num_buckets={}), bucket_size={} (fingerprint_bits={}..32) ---",
                    arity, num_buckets, seg3_num_buckets, std3_num_buckets, bucket_size, min_fingerprint_bits
                );
                for fingerprint_bits in min_fingerprint_bits..=32 {
                    run_fpr!(
                        csv,
                        "segmented",
                        Segmented3aryCuckooFilter,
                        "segmented",
                        3,
                        seg3_num_buckets,
                        bucket_size,
                        fingerprint_bits
                    );
                    run_fpr!(
                        csv,
                        "standard",
                        Standard3aryCuckooFilter,
                        "standard",
                        3,
                        std3_num_buckets,
                        bucket_size,
                        fingerprint_bits
                    );
                }
            }

            // ── 4-ary ──
            {
                let arity = 4;
                let min_fingerprint_bits = min_fingerprint_bits(arity, bucket_size);
                let filename = format!(
                    "fpr/arity{}_num_buckets{}_bucket_size{}.csv",
                    arity, num_buckets, bucket_size
                );
                let mut csv = helpers::csv_writer(
                    &filename,
                    "fingerprint_bits,scheme,num_buckets,load_factor,num_inserted,num_queries,\
                     false_positives,fpr_pct,theoretical_pct",
                );
                println!(
                    "\n--- arity={}, num_buckets={} (std4_num_buckets={}), bucket_size={} (fingerprint_bits={}..32) ---",
                    arity, num_buckets, std4_num_buckets, bucket_size, min_fingerprint_bits
                );
                for fingerprint_bits in min_fingerprint_bits..=32 {
                    run_fpr!(
                        csv,
                        "segmented",
                        Segmented4aryCuckooFilter,
                        "segmented",
                        4,
                        num_buckets,
                        bucket_size,
                        fingerprint_bits
                    );
                    run_fpr!(
                        csv,
                        "standard",
                        Standard4aryCuckooFilter,
                        "standard",
                        4,
                        std4_num_buckets,
                        bucket_size,
                        fingerprint_bits
                    );
                }
            }
        }
    }

    println!("\nResults written to results/fpr/");
}
