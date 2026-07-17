//! **Intent:** Measure lookup throughput on a filter at capacity. Feeds the
//! "Throughput (Mops) — Lookup" column of Table 2 in the CANS 2026 paper.
//!
//! **Method:** Fill until `TableFull`, then build a query set of
//! `num_buckets · bucket_size / 2` keys mixing inserted keys (hits) with
//! never-inserted ones (misses) at each requested hit rate, and time the
//! `contain` calls. `--warmup` untimed passes, then `--trials` timed ones.
//!
//! **Design rationale:** Hit rate matters because a lookup that finds a
//! matching fingerprint can stop at the first matching bucket, while a miss must
//! probe all `arity` of them — so the hit/miss mix sets how much of the probe
//! path each call walks. The default 50% is the balanced midpoint and the rate
//! Table 2 reports; pass `--hit-rate 0,50,100` to see the spread between the
//! all-miss and all-hit extremes. Measuring at full load keeps cache pressure
//! and eviction-chain layout realistic, and a query set of half the table's
//! capacity keeps runtime predictable without being trivially small.
//!
//! **Arguments (CLI):** all optional; with none, runs the paper matrix.
//! `--arity`, `--bucket-size`, `--num-buckets`, `--fingerprint-bits`,
//! `--max-kicks`, `--warmup`, `--trials`, plus `--hit-rate` (comma-separated
//! percentages, default `50`). See `benches/configs.rs`.
//!
//! **Output:** `results/segmented-cuckoo/cuckoo_filter_lookup_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, fingerprint_bits,
//! hit_rate_pct, load_factor, num_queries, mean_mops, min_mops, max_mops,
//! stddev_mops

mod configs;
mod helpers;

use configs::{ConfigCli, FilterConfig};
use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooFilter, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter,
    Standard2aryCuckooFilter, Standard3aryCuckooFilter, Standard4aryCuckooFilter,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "scheme,arity,num_buckets,bucket_size,fingerprint_bits,hit_rate_pct,\
                      load_factor,num_queries,mean_mops,min_mops,max_mops,stddev_mops";

#[derive(clap::Parser)]
#[command(about = "Lookup throughput of the segmented vs standard cuckoo filter (paper Table 2).")]
struct Cli {
    #[command(flatten)]
    config: ConfigCli,

    /// Hit rates to measure, as comma-separated percentages (e.g. `0,50,100`).
    #[arg(long, value_delimiter = ',', default_value = "50")]
    hit_rate: Vec<u64>,
}

/// Build a query set of `num_queries` keys with `hit_pct`% drawn from the
/// inserted range and the rest from keys never inserted.
fn build_queries(inserted: u64, num_queries: u64, hit_pct: u64) -> Vec<u64> {
    let num_hits = num_queries * hit_pct / 100;
    let num_misses = num_queries - num_hits;
    let mut queries: Vec<u64> = (0..num_hits).map(|i| i % inserted.max(1)).collect();
    queries.extend(inserted..inserted + num_misses);
    queries
}

/// Fill one filter type to capacity, then time `contain` at each hit rate;
/// write one CSV row per hit rate.
macro_rules! bench_lookup {
    ($csv:expr, $cli:expr, $label:expr, $filter_ty:ty, $scheme:expr, $cfg:expr, $num_buckets:expr) => {{
        let cfg: FilterConfig = $cfg;
        let num_buckets: u32 = $num_buckets;
        let c = &$cli.config;
        let fp_bits = c
            .fingerprint_bits
            .unwrap_or(configs::DEFAULT_FINGERPRINT_BITS);
        let trials = c.trials.unwrap_or(configs::DEFAULT_MEASURE_TRIALS);

        match <$filter_ty>::new(num_buckets, cfg.bucket_size, fp_bits) {
            Err(e) => eprintln!(
                "  Skip {} num_buckets={} bucket_size={}: {}",
                $label, num_buckets, cfg.bucket_size, e
            ),
            Ok(_) => {
                let mut filter = <$filter_ty>::new(num_buckets, cfg.bucket_size, fp_bits).unwrap();
                filter.set_max_kicks(c.max_kicks);
                let mut i = 0u64;
                loop {
                    match filter.insert(i.to_le_bytes()) {
                        Ok(()) => i += 1,
                        Err(CuckooError::TableFull) => break,
                        Err(e) => panic!("{}", e),
                    }
                }
                let inserted = i;
                let lf = filter.load_factor();
                let num_queries = (num_buckets as u64 * cfg.bucket_size as u64) / 2;

                println!(
                    "  {:<16} nb={:<9} b={} | inserted={} lf={:.4}% q={}",
                    $label,
                    num_buckets,
                    cfg.bucket_size,
                    inserted,
                    lf * 100.0,
                    num_queries
                );

                for &hit_rate in &$cli.hit_rate {
                    let queries = build_queries(inserted, num_queries, hit_rate);

                    for _ in 0..c.warmup {
                        for &key in &queries {
                            std::hint::black_box(filter.contain(key.to_le_bytes()));
                        }
                    }

                    let mut mops = Vec::with_capacity(trials);
                    for _ in 0..trials {
                        let start = Instant::now();
                        for &key in &queries {
                            std::hint::black_box(filter.contain(key.to_le_bytes()));
                        }
                        let ns = start.elapsed().as_nanos() as f64;
                        mops.push(num_queries as f64 / ns * 1000.0);
                    }

                    let s = helpers::compute_stats(&mops);
                    writeln!(
                        $csv,
                        "{},{},{},{},{},{},{:.6},{},{:.4},{:.4},{:.4},{:.4}",
                        $scheme,
                        cfg.arity,
                        num_buckets,
                        cfg.bucket_size,
                        fp_bits,
                        hit_rate,
                        lf,
                        num_queries,
                        s.mean,
                        s.min,
                        s.max,
                        s.stddev
                    )
                    .unwrap();
                    println!(
                        "    hit={:>3}% | mean={:>7.3}  std={:>6.3} Mops",
                        hit_rate, s.mean, s.stddev
                    );
                }
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
            bench_lookup!(
                csv,
                cli,
                "Segmented 2-ary",
                Segmented2aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_lookup!(
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
            bench_lookup!(
                csv,
                cli,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_lookup!(
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
            bench_lookup!(
                csv,
                cli,
                "Segmented 4-ary",
                Segmented4aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_lookup!(
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

    println!("=== Cuckoo filter — lookup throughput (full filter, q = capacity/2) ===");
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
    println!("Hit rates: {:?}%", cli.hit_rate);

    let mut csv = helpers::csv_writer("cuckoo_filter_lookup_throughput.csv", HEADER);
    for cfg in cfgs {
        run_config(&mut csv, &cli, cfg);
    }
    println!("\nResults written to cuckoo_filter_lookup_throughput.csv");
}
