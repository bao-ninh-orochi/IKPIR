//! **Intent:** Measure KV-store insert throughput across the paper's
//! `(arity, bucket_size)` configs and a sweep of `value_bits`.
//!
//! **Method:** Insert sequential keys carrying a fixed value until `TableFull`,
//! timing the whole loop. `--warmup` untimed fills, then `--trials` timed ones.
//!
//! **Design rationale:** "Insert until full" is the fair comparison for the same
//! reason as `cuckoo_filter_insert_throughput`: it puts every config through its
//! natural load-factor trajectory, so the number reflects the steady-state cost
//! the IKPIR server actually pays rather than a cherry-picked sparse-table
//! regime. The `value_bits` sweep is what distinguishes this bench from the
//! filter one — it exposes cell-packing cost, which at `value_bits = 1024`
//! dominates the insert path.
//!
//! **Relation to the paper.** This measures the KV-SCF, the primitive layer
//! under RisePIR, and is *not* one of the paper's tables. It borrows Table 2's
//! six `(arity, bucket_size)` pairs so the geometry lines up with the filter
//! benches, but sizes the table from `--target-items` rather than Table 2's ~10^6
//! buckets: a KV slot carries `fp ‖ value`, so at 10^6 buckets and
//! `value_bits = 1024` the table alone would run to gigabytes. Pass
//! `--num-buckets` to size it explicitly instead.
//!
//! **Arguments (CLI):** all optional; with none, runs the paper's six configs.
//! `--arity`, `--bucket-size`, `--fingerprint-bits`, `--max-kicks`, `--warmup`,
//! `--trials` (see `benches/configs.rs`), plus `--value-bits` (comma-separated,
//! default `8,64,256,1024`), `--plaintext-bits` (default 8), `--target-items`
//! (default 65536), and `--num-buckets` (overrides `--target-items` sizing).
//!
//! **Output:** `results/segmented-cuckoo/kv_store_insert_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, fingerprint_bits,
//! value_bits, plaintext_bits, mean_inserted, mean_lf, mean_mops, min_mops,
//! max_mops, stddev_mops

mod configs;
mod helpers;

use configs::{ConfigCli, FilterConfig};
use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooKVStore, Segmented3aryCuckooKVStore, Segmented4aryCuckooKVStore,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "scheme,arity,num_buckets,bucket_size,fingerprint_bits,value_bits,\
                      plaintext_bits,mean_inserted,mean_lf,mean_mops,min_mops,max_mops,\
                      stddev_mops";

#[derive(clap::Parser)]
#[command(about = "Insert throughput of the segmented cuckoo KV store (IKPIR primitive layer).")]
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

/// Time repeated fill-to-`TableFull` runs of one KV store type; one CSV row.
macro_rules! bench_kv_insert {
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

        // `--num-buckets`, when given, sizes the table directly; otherwise size
        // from the target item count (see the module docs on why this bench does
        // not default to Table 2's ~10^6 buckets).
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

                // One fill to TableFull → (inserted, elapsed_ns, load_factor).
                let fill = || {
                    let mut store = build().unwrap();
                    store.set_max_kicks(c.max_kicks);
                    let start = Instant::now();
                    let mut i = 0u64;
                    loop {
                        match store.insert(i.to_le_bytes(), &value) {
                            Ok(()) => i += 1,
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("{}", e),
                        }
                    }
                    let ns = start.elapsed().as_nanos() as f64;
                    (i, ns, store.load_factor())
                };

                for _ in 0..c.warmup {
                    std::hint::black_box(fill());
                }

                let mut mops = Vec::with_capacity(trials);
                let mut inserted_vals = Vec::with_capacity(trials);
                let mut lf_vals = Vec::with_capacity(trials);
                for _ in 0..trials {
                    let (inserted, ns, lf) = fill();
                    mops.push(inserted as f64 / ns * 1000.0);
                    inserted_vals.push(inserted as f64);
                    lf_vals.push(lf);
                }

                let s = helpers::compute_stats(&mops);
                let mean_inserted = inserted_vals.iter().sum::<f64>() / trials as f64;
                let mean_lf = lf_vals.iter().sum::<f64>() / trials as f64;
                writeln!(
                    $csv,
                    "{},{},{},{},{},{},{},{:.0},{:.6},{:.4},{:.4},{:.4},{:.4}",
                    $scheme,
                    cfg.arity,
                    num_buckets,
                    cfg.bucket_size,
                    fp_bits,
                    value_bits,
                    pb,
                    mean_inserted,
                    mean_lf,
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
                    mean_lf * 100.0
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
            2 => bench_kv_insert!(
                csv,
                cli,
                "Segmented 2-ary",
                Segmented2aryCuckooKVStore,
                "segmented",
                cfg,
                value_bits
            ),
            3 => bench_kv_insert!(
                csv,
                cli,
                "Segmented 3-ary",
                Segmented3aryCuckooKVStore,
                "segmented",
                cfg,
                value_bits
            ),
            4 => bench_kv_insert!(
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

    println!("=== KV store — insert throughput (insert until full) ===");
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

    let mut csv = helpers::csv_writer("kv_store_insert_throughput.csv", HEADER);
    for cfg in cfgs {
        run_config(&mut csv, &cli, cfg);
    }
    println!("\nResults written to kv_store_insert_throughput.csv");
}
