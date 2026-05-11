//! **Intent:** Measure raw KV insert speed for each segmented variant across
//! `arity`, `bucket_size`, and `value_bits`.
//!
//! **Method:** Insert sequential keys with a deterministic value until `TableFull`,
//! timing the full loop. 3 warmup + 10 timed trials.
//!
//! **Parameters:** arity ∈ {2,3,4}, bucket_size ∈ {2,4}, value_bits ∈ {8, 64, 256, 1024}.
//! fingerprint_bits = 12. num_buckets sized via from_num_items at a target capacity.
//!
//! **Output:** `results/kv_store_insert_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, value_bits, mean_inserted, mean_lf, mean_mops

mod helpers;

use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooKVStore, Segmented3aryCuckooKVStore,
    Segmented4aryCuckooKVStore,
};
use std::io::Write;
use std::time::Instant;

const MAX_KICKS: u32 = 2500;
const FINGERPRINT_BITS: u32 = 12;
const PLAINTEXT_BITS: u32 = 8;
const WARMUP_TRIALS: usize = 3;
const MEASURE_TRIALS: usize = 10;

const TARGET_ITEMS: u64 = 1 << 16;
const BUCKET_SIZE_VALUES: &[u32] = &[2, 4];
const VALUE_BITS_VALUES: &[u32] = &[8, 64, 256, 1024];

macro_rules! bench_kv_insert {
    ($csv:expr, $label:expr, $store_ty:ty, $scheme:expr, $arity:expr, $bucket_size:expr, $value_bits:expr) => {{
        let bucket_size: u32 = $bucket_size;
        let value_bits: u32 = $value_bits;
        let vsize = (value_bits as usize).div_ceil(8);
        let value: Vec<u8> = (0..vsize).map(|i| (i as u8).wrapping_mul(37).wrapping_add(7)).collect();

        match <$store_ty>::from_num_items(TARGET_ITEMS, bucket_size, FINGERPRINT_BITS, value_bits, PLAINTEXT_BITS) {
            Err(e) => {
                eprintln!("  Skip {} bucket_size={} value_bits={}: {}", $label, bucket_size, value_bits, e);
            }
            Ok(template) => {
                let num_buckets = (template.size_in_bytes() as f64
                    / ((FINGERPRINT_BITS + value_bits) as f64 / 8.0 * bucket_size as f64))
                    .ceil() as u32;

                // Warmup
                for _ in 0..WARMUP_TRIALS {
                    let mut store = <$store_ty>::from_num_items(
                        TARGET_ITEMS, bucket_size, FINGERPRINT_BITS, value_bits, PLAINTEXT_BITS,
                    ).unwrap();
                    store.set_max_kicks(MAX_KICKS);
                    let mut i = 0u64;
                    loop {
                        match store.insert(i.to_le_bytes(), &value) {
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
                for _trial in 0..MEASURE_TRIALS {
                    let mut store = <$store_ty>::from_num_items(
                        TARGET_ITEMS, bucket_size, FINGERPRINT_BITS, value_bits, PLAINTEXT_BITS,
                    ).unwrap();
                    store.set_max_kicks(MAX_KICKS);
                    let start = Instant::now();
                    let mut i = 0u64;
                    loop {
                        match store.insert(i.to_le_bytes(), &value) {
                            Ok(()) => i += 1,
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("{}", e),
                        }
                    }
                    let elapsed_ns = start.elapsed().as_nanos() as f64;
                    let inserted = i;
                    let lf = store.load_factor();
                    let mops = inserted as f64 / elapsed_ns * 1000.0;
                    throughputs.push(mops);
                    inserted_vals.push(inserted as f64);
                    lf_vals.push(lf);
                }
                let mops_stats = helpers::compute_stats(&throughputs);
                let mean_inserted = inserted_vals.iter().sum::<f64>() / MEASURE_TRIALS as f64;
                let mean_lf = lf_vals.iter().sum::<f64>() / MEASURE_TRIALS as f64;
                writeln!(
                    $csv,
                    "{},{},{},{},{},{:.0},{:.6},{:.4}",
                    $scheme, $arity, num_buckets, bucket_size, value_bits,
                    mean_inserted, mean_lf, mops_stats.mean,
                )
                .unwrap();
                println!(
                    "  {:<22} bs={:<2} vb={:<5} | mean={:<8.3} lf={:.4} Mops",
                    $label, bucket_size, value_bits, mops_stats.mean, mean_lf
                );
            }
        }
    }};
}

fn main() {
    let mut csv = helpers::csv_writer(
        "kv_store_insert_throughput.csv",
        "scheme,arity,num_buckets,bucket_size,value_bits,mean_inserted,mean_lf,mean_mops",
    );

    println!("=== KV Store Insert Throughput (insert until full) ===");
    println!(
        "Config: fingerprint_bits={}, max_kicks={}, target_items={}, warmup={}, trials={}",
        FINGERPRINT_BITS, MAX_KICKS, TARGET_ITEMS, WARMUP_TRIALS, MEASURE_TRIALS
    );

    for &bucket_size in BUCKET_SIZE_VALUES {
        for &value_bits in VALUE_BITS_VALUES {
            println!("\n--- bucket_size={}, value_bits={} ---", bucket_size, value_bits);
            bench_kv_insert!(csv, "Segmented 2-ary", Segmented2aryCuckooKVStore, "segmented", 2, bucket_size, value_bits);
            bench_kv_insert!(csv, "Segmented 3-ary", Segmented3aryCuckooKVStore, "segmented", 3, bucket_size, value_bits);
            bench_kv_insert!(csv, "Segmented 4-ary", Segmented4aryCuckooKVStore, "segmented", 4, bucket_size, value_bits);
        }
    }

    println!("\nResults written to results/kv_store_insert_throughput.csv");
}
