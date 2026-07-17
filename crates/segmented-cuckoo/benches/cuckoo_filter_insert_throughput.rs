//! **Intent:** Measure insert throughput for each filter at its natural load
//! factor. Feeds the "Throughput (Mops) — Insert" column of Table 2 in the
//! CANS 2026 paper.
//!
//! **Method:** Insert sequential `u64` keys until `TableFull`, timing the whole
//! loop, and divide insertions by elapsed time to get Mops/s. `--warmup`
//! untimed fills, then `--trials` timed ones.
//!
//! **Design rationale:** "Insert until full" is the only fair comparison here.
//! Inserting a fixed number of items would confound throughput with load
//! factor: the same 1 M items fill one scheme near saturation (long, expensive
//! eviction chains) and another only partway (sparse, cheap kicking), making
//! the first look artificially slow. Measuring over the full fill trajectory
//! puts every scheme under comparable conditions.
//!
//! **Arguments (CLI):** all optional; with none, runs the paper matrix.
//! `--arity`, `--bucket-size`, `--num-buckets`, `--fingerprint-bits`,
//! `--max-kicks`, `--warmup`, `--trials`. See `benches/configs.rs`.
//!
//! **Output:** `results/segmented-cuckoo/cuckoo_filter_insert_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, fingerprint_bits,
//! max_kicks, mean_inserted, mean_lf, mean_duration_ns, mean_mops, min_mops,
//! max_mops, stddev_mops

mod configs;
mod helpers;

use configs::{ConfigCli, FilterConfig};
use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooFilter, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter,
    Standard2aryCuckooFilter, Standard3aryCuckooFilter, Standard4aryCuckooFilter,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "scheme,arity,num_buckets,bucket_size,fingerprint_bits,max_kicks,\
                      mean_inserted,mean_lf,mean_duration_ns,mean_mops,min_mops,max_mops,\
                      stddev_mops";

#[derive(clap::Parser)]
#[command(about = "Insert throughput of the segmented vs standard cuckoo filter (paper Table 2).")]
struct Cli {
    #[command(flatten)]
    config: ConfigCli,
}

/// Time repeated fill-to-`TableFull` runs of one filter type; write one CSV row.
macro_rules! bench_insert {
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
                // One fill to TableFull → (inserted, elapsed_ns, load_factor).
                let fill = || {
                    let mut filter =
                        <$filter_ty>::new(num_buckets, cfg.bucket_size, fp_bits).unwrap();
                    filter.set_max_kicks(c.max_kicks);
                    let start = Instant::now();
                    let mut i = 0u64;
                    loop {
                        match filter.insert(i.to_le_bytes()) {
                            Ok(()) => i += 1,
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("{}", e),
                        }
                    }
                    let ns = start.elapsed().as_nanos() as f64;
                    (i, ns, filter.load_factor())
                };

                for _ in 0..c.warmup {
                    std::hint::black_box(fill());
                }

                let mut mops = Vec::with_capacity(trials);
                let mut inserted_vals = Vec::with_capacity(trials);
                let mut lf_vals = Vec::with_capacity(trials);
                let mut dur_vals = Vec::with_capacity(trials);
                for _ in 0..trials {
                    let (inserted, ns, lf) = fill();
                    mops.push(inserted as f64 / ns * 1000.0);
                    inserted_vals.push(inserted as f64);
                    lf_vals.push(lf);
                    dur_vals.push(ns);
                }

                let s = helpers::compute_stats(&mops);
                let mean_inserted = inserted_vals.iter().sum::<f64>() / trials as f64;
                let mean_lf = lf_vals.iter().sum::<f64>() / trials as f64;
                let mean_dur = dur_vals.iter().sum::<f64>() / trials as f64;
                writeln!(
                    $csv,
                    "{},{},{},{},{},{},{:.0},{:.6},{:.0},{:.4},{:.4},{:.4},{:.4}",
                    $scheme,
                    cfg.arity,
                    num_buckets,
                    cfg.bucket_size,
                    fp_bits,
                    c.max_kicks,
                    mean_inserted,
                    mean_lf,
                    mean_dur,
                    s.mean,
                    s.min,
                    s.max,
                    s.stddev
                )
                .unwrap();
                println!(
                    "  {:<16} nb={:<9} b={} | mean={:>7.3}  std={:>6.3} Mops  (lf={:.4}%)",
                    $label,
                    num_buckets,
                    cfg.bucket_size,
                    s.mean,
                    s.stddev,
                    mean_lf * 100.0
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
            bench_insert!(
                csv,
                cli,
                "Segmented 2-ary",
                Segmented2aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_insert!(
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
            bench_insert!(
                csv,
                cli,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_insert!(
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
            bench_insert!(
                csv,
                cli,
                "Segmented 4-ary",
                Segmented4aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_insert!(
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

    println!("=== Cuckoo filter — insert throughput (insert until full) ===");
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

    let mut csv = helpers::csv_writer("cuckoo_filter_insert_throughput.csv", HEADER);
    for cfg in cfgs {
        run_config(&mut csv, &cli, cfg);
    }
    println!("\nResults written to cuckoo_filter_insert_throughput.csv");
}
