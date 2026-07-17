//! **Intent:** Measure KV-store `get_into` throughput on a full store across the
//! paper's `(arity, bucket_size)` configs and a sweep of `value_bits`.
//!
//! **Method:** Fill until `TableFull`, then issue `num_buckets · bucket_size / 2`
//! `get_into` calls at a 50/50 hit/miss mix into a caller-owned buffer.
//! `--warmup` untimed passes, then `--trials` timed ones.
//!
//! **Design rationale:** `get_into` rather than `get` because it is the
//! zero-allocation read path the IKPIR server uses; timing `get` would fold a
//! per-call `Vec` allocation into the number. The 50/50 mix is the balanced
//! midpoint between an all-hit read (stops at the first matching fingerprint)
//! and an all-miss one (probes every candidate bucket).
//!
//! **Relation to the paper.** Measures the KV-SCF primitive layer, not one of
//! the paper's tables; see `kv_store_insert_throughput` for why this bench sizes
//! from `--target-items` rather than Table 2's ~10^6 buckets.
//!
//! **Arguments (CLI):** all optional; with none, runs the paper's five configs.
//! `--arity`, `--bucket-size`, `--fingerprint-bits`, `--max-kicks`, `--warmup`,
//! `--trials` (see `benches/configs.rs`), plus `--value-bits` (comma-separated,
//! default `8,64,256,1024`), `--plaintext-bits` (default 8), `--target-items`
//! (default 65536), and `--num-buckets` (overrides `--target-items` sizing).
//!
//! **Output:** `results/segmented-cuckoo/kv_store_lookup_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, fingerprint_bits,
//! value_bits, plaintext_bits, num_inserted, load_factor, num_queries,
//! mean_mops, min_mops, max_mops, stddev_mops

mod configs;
mod helpers;

use configs::{ConfigCli, FilterConfig};
use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooKVStore, Segmented3aryCuckooKVStore, Segmented4aryCuckooKVStore,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "scheme,arity,num_buckets,bucket_size,fingerprint_bits,value_bits,\
                      plaintext_bits,num_inserted,load_factor,num_queries,mean_mops,min_mops,\
                      max_mops,stddev_mops";

#[derive(clap::Parser)]
#[command(about = "Lookup throughput of the segmented cuckoo KV store (IKPIR primitive layer).")]
struct Cli {
    #[command(flatten)]
    config: ConfigCli,

    /// Value widths to sweep, comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "8,64,256,1024")]
    value_bits: Vec<u32>,

    /// PIR plaintext cell width (1–32). 8 keeps byte↔cell a no-op.
    #[arg(long, default_value_t = 8)]
    plaintext_bits: u32,

    /// Target item count used to size the table when `--num-buckets` is absent.
    #[arg(long, default_value_t = 1 << 16)]
    target_items: u64,
}

/// Build a 50/50 hit/miss query set: half inside the inserted range, half past
/// it.
fn build_queries(inserted: u64, num_queries: u64) -> Vec<u64> {
    let half = num_queries / 2;
    let mut q: Vec<u64> = (0..half).map(|i| i % inserted.max(1)).collect();
    q.extend(inserted..inserted + (num_queries - half));
    q
}

/// Fill one KV store type to capacity, then time `get_into`; one CSV row.
macro_rules! bench_kv_lookup {
    ($csv:expr, $cli:expr, $label:expr, $store_ty:ty, $scheme:expr, $cfg:expr, $value_bits:expr) => {{
        let cfg: FilterConfig = $cfg;
        let value_bits: u32 = $value_bits;
        let c = &$cli.config;
        let fp_bits = c
            .fingerprint_bits
            .unwrap_or(configs::DEFAULT_FINGERPRINT_BITS);
        let trials = c.trials.unwrap_or(configs::DEFAULT_MEASURE_TRIALS);
        let pb = $cli.plaintext_bits;

        let value: Vec<u8> = (0..value_bits.div_ceil(8) as usize)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(7))
            .collect();

        let built = match c.num_buckets {
            Some(nb) => <$store_ty>::new(nb, cfg.bucket_size, fp_bits, value_bits, pb),
            None => <$store_ty>::from_num_items(
                $cli.target_items,
                cfg.bucket_size,
                fp_bits,
                value_bits,
                pb,
            ),
        };

        match built {
            Err(e) => eprintln!(
                "  Skip {} bucket_size={} value_bits={}: {}",
                $label, cfg.bucket_size, value_bits, e
            ),
            Ok(mut store) => {
                store.set_max_kicks(c.max_kicks);
                let mut i = 0u64;
                loop {
                    match store.insert(i.to_le_bytes(), &value) {
                        Ok(()) => i += 1,
                        Err(CuckooError::TableFull) => break,
                        Err(e) => panic!("{}", e),
                    }
                }
                let inserted = i;
                let lf = store.load_factor();
                let num_buckets = store.params().num_buckets;
                let num_queries = (num_buckets as u64 * cfg.bucket_size as u64) / 2;

                let queries = build_queries(inserted, num_queries);
                let mut buf = vec![0u8; store.value_size_in_bytes()];

                for _ in 0..c.warmup {
                    for &k in &queries {
                        std::hint::black_box(store.get_into(k.to_le_bytes(), &mut buf));
                    }
                }

                let mut mops = Vec::with_capacity(trials);
                for _ in 0..trials {
                    let start = Instant::now();
                    for &k in &queries {
                        std::hint::black_box(store.get_into(k.to_le_bytes(), &mut buf));
                    }
                    let ns = start.elapsed().as_nanos() as f64;
                    mops.push(num_queries as f64 / ns * 1000.0);
                }

                let s = helpers::compute_stats(&mops);
                writeln!(
                    $csv,
                    "{},{},{},{},{},{},{},{},{:.6},{},{:.4},{:.4},{:.4},{:.4}",
                    $scheme,
                    cfg.arity,
                    num_buckets,
                    cfg.bucket_size,
                    fp_bits,
                    value_bits,
                    pb,
                    inserted,
                    lf,
                    num_queries,
                    s.mean,
                    s.min,
                    s.max,
                    s.stddev
                )
                .unwrap();
                println!(
                    "  {:<16} nb={:<8} b={} vb={:<5} | mean={:>7.3}  std={:>6.3} Mops  (lf={:.4}%)",
                    $label,
                    num_buckets,
                    cfg.bucket_size,
                    value_bits,
                    s.mean,
                    s.stddev,
                    lf * 100.0
                );
            }
        }
    }};
}

/// Run one `(arity, bucket_size)` config across every requested `value_bits`.
fn run_config(csv: &mut std::io::BufWriter<std::fs::File>, cli: &Cli, cfg: FilterConfig) {
    for &value_bits in &cli.value_bits {
        println!(
            "\n--- arity={} bucket_size={} value_bits={} ---",
            cfg.arity, cfg.bucket_size, value_bits
        );
        match cfg.arity {
            2 => bench_kv_lookup!(
                csv,
                cli,
                "Segmented 2-ary",
                Segmented2aryCuckooKVStore,
                "segmented",
                cfg,
                value_bits
            ),
            3 => bench_kv_lookup!(
                csv,
                cli,
                "Segmented 3-ary",
                Segmented3aryCuckooKVStore,
                "segmented",
                cfg,
                value_bits
            ),
            4 => bench_kv_lookup!(
                csv,
                cli,
                "Segmented 4-ary",
                Segmented4aryCuckooKVStore,
                "segmented",
                cfg,
                value_bits
            ),
            a => panic!("arity must be 2, 3, or 4 (got {a})"),
        }
    }
}

fn main() {
    if helpers::skip_when_cargo_test() {
        return;
    }
    let cli: Cli = configs::parse();
    let cfgs = cli.config.configs();

    println!("=== KV store — lookup throughput (full store, 50/50 hit/miss, get_into) ===");
    println!(
        "{}",
        cli.config.describe(
            &cfgs,
            Some(
                cli.config
                    .fingerprint_bits
                    .unwrap_or(configs::DEFAULT_FINGERPRINT_BITS)
            )
        )
    );
    println!(
        "warmup={}, trials={}",
        cli.config.warmup,
        cli.config.trials.unwrap_or(configs::DEFAULT_MEASURE_TRIALS)
    );
    println!(
        "value_bits={:?}, plaintext_bits={}, target_items={}",
        cli.value_bits, cli.plaintext_bits, cli.target_items
    );

    let mut csv = helpers::csv_writer("kv_store_lookup_throughput.csv", HEADER);
    for cfg in cfgs {
        run_config(&mut csv, &cli, cfg);
    }
    println!("\nResults written to kv_store_lookup_throughput.csv");
}
