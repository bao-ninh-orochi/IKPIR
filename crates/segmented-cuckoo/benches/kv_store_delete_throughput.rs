//! **Intent:** Measure KV-store `delete` throughput on a full store across the
//! paper's `(arity, bucket_size)` configs and a sweep of `value_bits`.
//!
//! **Method:** Probe capacity once, then per trial: refill with exactly that
//! many items (untimed) and time only the loop that deletes them all.
//! `--warmup` untimed fill/delete rounds, then `--trials` timed ones.
//!
//! **Design rationale:** Timing only the delete loop isolates the delete path
//! from insert cost, and probing capacity once keeps every trial deleting the
//! same count so the per-trial numbers stay comparable — the same reasoning as
//! `cuckoo_filter_delete_throughput`. A delete may return `NotFound` when two
//! keys collide on fingerprint and candidate buckets; that is expected and does
//! not invalidate the measurement, since the full delete path still runs.
//!
//! **Relation to the paper.** Measures the KV-SCF primitive layer, not one of
//! the paper's tables; see `kv_store_insert_throughput` for why this bench sizes
//! from `--target-items` rather than Table 2's ~10^6 buckets.
//!
//! **Arguments (CLI):** all optional; with none, runs the paper's six configs.
//! `--arity`, `--bucket-size`, `--fingerprint-bits`, `--max-kicks`, `--warmup`,
//! `--trials` (see `benches/configs.rs`), plus `--value-bits` (comma-separated,
//! default `8,64,256,1024`), `--plaintext-bits` (default 8), `--target-items`
//! (default 65536), and `--num-buckets` (overrides `--target-items` sizing).
//!
//! **Output:** `results/segmented-cuckoo/kv_store_delete_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, fingerprint_bits,
//! value_bits, plaintext_bits, deleted, mean_lf, mean_mops, min_mops, max_mops,
//! stddev_mops

mod configs;
mod helpers;

use configs::{ConfigCli, FilterConfig};
use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooKVStore, Segmented3aryCuckooKVStore, Segmented4aryCuckooKVStore,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "scheme,arity,num_buckets,bucket_size,fingerprint_bits,value_bits,\
                      plaintext_bits,deleted,mean_lf,mean_mops,min_mops,max_mops,stddev_mops";

#[derive(clap::Parser)]
#[command(about = "Delete throughput of the segmented cuckoo KV store (IKPIR primitive layer).")]
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

/// Probe capacity, then time repeated delete-all loops of one KV store type;
/// one CSV row.
macro_rules! bench_kv_delete {
    ($csv:expr, $cli:expr, $label:expr, $store_ty:ty, $scheme:expr, $cfg:expr, $value_bits:expr) => {{
        let cfg: FilterConfig = $cfg;
        let value_bits: u32 = $value_bits;
        let c = &$cli.config;
        let fp_bits = c.fingerprint_bits.unwrap_or(configs::DEFAULT_FINGERPRINT_BITS);
        let trials = c.trials.unwrap_or(configs::DEFAULT_MEASURE_TRIALS);
        let pb = $cli.plaintext_bits;

        let value: Vec<u8> = (0..value_bits.div_ceil(8) as usize)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(7))
            .collect();

        let build = || match c.num_buckets {
            Some(nb) => <$store_ty>::new(nb, cfg.bucket_size, fp_bits, value_bits, pb),
            None => <$store_ty>::from_num_items(
                $cli.target_items,
                cfg.bucket_size,
                fp_bits,
                value_bits,
                pb,
            ),
        };

        match build() {
            Err(e) => eprintln!(
                "  Skip {} bucket_size={} value_bits={}: {}",
                $label, cfg.bucket_size, value_bits, e
            ),
            Ok(template) => {
                let num_buckets = template.params().num_buckets;

                // Probe capacity once so every trial deletes the same count.
                let count: u64 = {
                    let mut probe = build().unwrap();
                    probe.set_max_kicks(c.max_kicks);
                    let mut n = 0u64;
                    loop {
                        match probe.insert(n.to_le_bytes(), &value) {
                            Ok(()) => n += 1,
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("{}", e),
                        }
                    }
                    n
                };

                if count == 0 {
                    eprintln!(
                        "  Skip {} bucket_size={} value_bits={}: store holds 0 items",
                        $label, cfg.bucket_size, value_bits
                    );
                } else {
                    // Refill to `count` (untimed) → a store ready to delete from.
                    let fill = || {
                        let mut store = build().unwrap();
                        store.set_max_kicks(c.max_kicks);
                        for i in 0u64..count {
                            let _ = store.insert(i.to_le_bytes(), &value);
                        }
                        store
                    };

                    for _ in 0..c.warmup {
                        let mut store = fill();
                        for i in 0u64..count {
                            let _ = store.delete(i.to_le_bytes());
                        }
                    }

                    let mut mops = Vec::with_capacity(trials);
                    let mut lf_vals = Vec::with_capacity(trials);
                    for _ in 0..trials {
                        let mut store = fill();
                        lf_vals.push(store.load_factor());

                        let start = Instant::now();
                        for i in 0u64..count {
                            // Result ignored: a fingerprint+bucket collision can
                            // resolve as NotFound. The delete path still runs.
                            let _ = store.delete(i.to_le_bytes());
                        }
                        let ns = start.elapsed().as_nanos() as f64;
                        mops.push(count as f64 / ns * 1000.0);
                    }

                    let s = helpers::compute_stats(&mops);
                    let mean_lf = lf_vals.iter().sum::<f64>() / trials as f64;
                    writeln!(
                        $csv,
                        "{},{},{},{},{},{},{},{},{:.6},{:.4},{:.4},{:.4},{:.4}",
                        $scheme,
                        cfg.arity,
                        num_buckets,
                        cfg.bucket_size,
                        fp_bits,
                        value_bits,
                        pb,
                        count,
                        mean_lf,
                        s.mean,
                        s.min,
                        s.max,
                        s.stddev
                    )
                    .unwrap();
                    println!(
                        "  {:<16} nb={:<8} b={} vb={:<5} | mean={:>7.3}  std={:>6.3} Mops  ({} deleted)",
                        $label, num_buckets, cfg.bucket_size, value_bits, s.mean, s.stddev, count
                    );
                }
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
            2 => bench_kv_delete!(
                csv,
                cli,
                "Segmented 2-ary",
                Segmented2aryCuckooKVStore,
                "segmented",
                cfg,
                value_bits
            ),
            3 => bench_kv_delete!(
                csv,
                cli,
                "Segmented 3-ary",
                Segmented3aryCuckooKVStore,
                "segmented",
                cfg,
                value_bits
            ),
            4 => bench_kv_delete!(
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

    println!("=== KV store — delete throughput (delete all items from a full store) ===");
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

    let mut csv = helpers::csv_writer("kv_store_delete_throughput.csv", HEADER);
    for cfg in cfgs {
        run_config(&mut csv, &cli, cfg);
    }
    println!("\nResults written to kv_store_delete_throughput.csv");
}
