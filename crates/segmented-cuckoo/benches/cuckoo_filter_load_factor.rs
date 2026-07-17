//! **Intent:** Measure the maximum load factor each filter reaches — how full
//! it gets before an insert reports `TableFull`. Feeds the "Load factor (%) —
//! Achieved" column of Table 2 in the CANS 2026 paper.
//!
//! **Method:** Insert sequential `u64` keys (LE bytes) until `insert` returns
//! `TableFull`, then record `load_factor()`. Repeated over `--trials` trials to
//! capture the variance of the randomised eviction chain, which is this bench's
//! only source of noise.
//!
//! **Design rationale:** "Insert until full" is the definition of the quantity
//! being measured, so there is no warmup phase: every trial is an untimed fill
//! from an empty table and the reported statistic is a load factor, not a
//! duration. `--trials` defaults to 20 here rather than the 10 the throughput
//! benches use, because this bench reports one headline number per config and a
//! tighter error bar is worth the extra fills.
//!
//! **Arguments (CLI):** all optional; with none, runs the paper matrix.
//! `--arity`, `--bucket-size`, `--num-buckets`, `--fingerprint-bits`,
//! `--max-kicks`, `--trials`. See `benches/configs.rs` for the defaults and for
//! how a partial selector resolves.
//!
//! **Output:** `results/segmented-cuckoo/cuckoo_filter_load_factor.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, fingerprint_bits,
//! max_kicks, trials, mean_lf, min_lf, max_lf, stddev_lf

mod configs;
mod helpers;

use configs::{ConfigCli, FilterConfig, DEFAULT_LOAD_FACTOR_TRIALS};
use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooFilter, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter,
    Standard2aryCuckooFilter, Standard3aryCuckooFilter, Standard4aryCuckooFilter,
};
use std::io::Write;

const HEADER: &str = "scheme,arity,num_buckets,bucket_size,fingerprint_bits,max_kicks,trials,\
                      mean_lf,min_lf,max_lf,stddev_lf";

#[derive(clap::Parser)]
#[command(
    about = "Maximum load factor of the segmented vs standard cuckoo filter (paper Table 2)."
)]
struct Cli {
    #[command(flatten)]
    config: ConfigCli,
}

/// Fill one filter type to `TableFull` over `trials` trials; write one CSV row.
macro_rules! bench_load_factor {
    ($csv:expr, $cli:expr, $label:expr, $filter_ty:ty, $scheme:expr, $cfg:expr, $num_buckets:expr) => {{
        let cfg: FilterConfig = $cfg;
        let num_buckets: u32 = $num_buckets;
        let c = &$cli.config;
        let fp_bits = c
            .fingerprint_bits
            .unwrap_or(configs::DEFAULT_FINGERPRINT_BITS);
        // This bench reports one headline number per config, so it defaults to
        // more trials than the throughput benches.
        let trials = c.trials.unwrap_or(DEFAULT_LOAD_FACTOR_TRIALS);

        match <$filter_ty>::new(num_buckets, cfg.bucket_size, fp_bits) {
            Err(e) => eprintln!(
                "  Skip {} num_buckets={} bucket_size={}: {}",
                $label, num_buckets, cfg.bucket_size, e
            ),
            Ok(_) => {
                let mut lfs = Vec::with_capacity(trials);
                for _ in 0..trials {
                    let mut filter =
                        <$filter_ty>::new(num_buckets, cfg.bucket_size, fp_bits).unwrap();
                    filter.set_max_kicks(c.max_kicks);
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
                let s = helpers::compute_stats(&lfs);
                writeln!(
                    $csv,
                    "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
                    $scheme,
                    cfg.arity,
                    num_buckets,
                    cfg.bucket_size,
                    fp_bits,
                    c.max_kicks,
                    trials,
                    s.mean,
                    s.min,
                    s.max,
                    s.stddev
                )
                .unwrap();
                println!(
                    "  {:<16} nb={:<9} b={} | mean={:>7.4}%  min={:>7.4}%  max={:>7.4}%",
                    $label,
                    num_buckets,
                    cfg.bucket_size,
                    s.mean * 100.0,
                    s.min * 100.0,
                    s.max * 100.0
                );
            }
        }
    }};
}

/// Run both schemes at one `(arity, bucket_size)` config.
fn run_config(csv: &mut std::io::BufWriter<std::fs::File>, cli: &Cli, cfg: FilterConfig) {
    println!(
        "\n--- arity={} bucket_size={} ---",
        cfg.arity, cfg.bucket_size
    );
    let (seg, std) = (cfg.segmented_num_buckets, cfg.standard_num_buckets);
    match cfg.arity {
        2 => {
            bench_load_factor!(
                csv,
                cli,
                "Segmented 2-ary",
                Segmented2aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_load_factor!(
                csv,
                cli,
                "Standard 2-ary",
                Standard2aryCuckooFilter,
                "standard",
                cfg,
                std
            );
        }
        3 => {
            bench_load_factor!(
                csv,
                cli,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_load_factor!(
                csv,
                cli,
                "Standard 3-ary",
                Standard3aryCuckooFilter,
                "standard",
                cfg,
                std
            );
        }
        4 => {
            bench_load_factor!(
                csv,
                cli,
                "Segmented 4-ary",
                Segmented4aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_load_factor!(
                csv,
                cli,
                "Standard 4-ary",
                Standard4aryCuckooFilter,
                "standard",
                cfg,
                std
            );
        }
        a => panic!("arity must be 2, 3, or 4 (got {a})"),
    }
}

fn main() {
    if helpers::skip_when_cargo_test() {
        return;
    }
    let cli: Cli = configs::parse();
    let cfgs = cli.config.configs();

    println!("=== Cuckoo filter — maximum load factor (insert until full) ===");
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
        "trials={} (no warmup: every trial is an untimed fill)",
        cli.config.trials.unwrap_or(DEFAULT_LOAD_FACTOR_TRIALS)
    );

    let mut csv = helpers::csv_writer("cuckoo_filter_load_factor.csv", HEADER);
    for cfg in cfgs {
        run_config(&mut csv, &cli, cfg);
    }
    println!("\nResults written to cuckoo_filter_load_factor.csv");
}
