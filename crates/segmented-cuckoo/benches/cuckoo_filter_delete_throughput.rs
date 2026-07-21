//! **Intent:** Measure delete throughput on a filter at capacity. Feeds the
//! "Throughput (Mops) — Delete" column of Table 2 in the CANS 2026 paper.
//!
//! **Method:** Probe capacity once with a single fill, then per trial: refill
//! with exactly that many items (untimed) and time only the loop that deletes
//! them all. `--warmup` untimed fill/delete rounds, then `--trials` timed ones.
//!
//! **Design rationale:** Timing only the delete loop isolates the hash-and-probe
//! delete path from insert cost; timing a fill+delete round would report a blend
//! of the two. Capacity is probed once and reused so every trial deletes the same
//! number of items, keeping the per-trial numbers comparable — a capacity
//! re-probe per trial would vary with the randomised eviction chain and widen the
//! error bar for no gain. A small fraction of deletes may return `NotFound` when
//! two items collide on both fingerprint and candidate buckets; that is expected
//! and does not invalidate the measurement, since the full delete path (hash,
//! probe, compare) still runs.
//!
//! **Arguments (CLI):** all optional; with none, runs the paper matrix.
//! `--arity`, `--bucket-size`, `--num-buckets`, `--fingerprint-bits`,
//! `--max-kicks`, `--warmup`, `--trials`. See `benches/configs.rs`.
//!
//! **Output:** `results/segmented-cuckoo/cuckoo_filter_delete_throughput.csv`
//! Columns: scheme, arity, num_buckets, bucket_size, fingerprint_bits,
//! max_kicks, deleted, mean_lf, mean_duration_ns, mean_mops, min_mops,
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

const HEADER: &str = "scheme,arity,num_buckets,bucket_size,fingerprint_bits,max_kicks,deleted,\
                      mean_lf,mean_duration_ns,mean_mops,min_mops,max_mops,stddev_mops";

#[derive(clap::Parser)]
#[command(about = "Delete throughput of the segmented vs standard cuckoo filter (paper Table 2).")]
struct Cli {
    #[command(flatten)]
    config: ConfigCli,
}

/// Probe capacity, then time repeated delete-all loops of one filter type;
/// write one CSV row.
macro_rules! bench_delete {
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
                // Probe capacity once so every trial deletes the same count.
                let count: u64 = {
                    let mut probe =
                        <$filter_ty>::new(num_buckets, cfg.bucket_size, fp_bits).unwrap();
                    probe.set_max_kicks(c.max_kicks);
                    let mut n = 0u64;
                    loop {
                        match probe.insert(n.to_le_bytes()) {
                            Ok(()) => n += 1,
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("{}", e),
                        }
                    }
                    n
                };

                if count == 0 {
                    eprintln!(
                        "  Skip {} num_buckets={} bucket_size={}: filter holds 0 items",
                        $label, num_buckets, cfg.bucket_size
                    );
                } else {
                    // Refill to `count` (untimed) → a filter ready to delete from.
                    let fill = || {
                        let mut filter =
                            <$filter_ty>::new(num_buckets, cfg.bucket_size, fp_bits).unwrap();
                        filter.set_max_kicks(c.max_kicks);
                        for i in 0u64..count {
                            let _ = filter.insert(i.to_le_bytes());
                        }
                        filter
                    };

                    for _ in 0..c.warmup {
                        let mut filter = fill();
                        for i in 0u64..count {
                            let _ = filter.delete(i.to_le_bytes());
                        }
                    }

                    let mut mops = Vec::with_capacity(trials);
                    let mut lf_vals = Vec::with_capacity(trials);
                    let mut dur_vals = Vec::with_capacity(trials);
                    for _ in 0..trials {
                        let mut filter = fill();
                        lf_vals.push(filter.load_factor());

                        let start = Instant::now();
                        for i in 0u64..count {
                            // Result ignored: a fingerprint+bucket collision can
                            // resolve as NotFound. The delete path still runs.
                            let _ = filter.delete(i.to_le_bytes());
                        }
                        let ns = start.elapsed().as_nanos() as f64;
                        mops.push(count as f64 / ns * 1000.0);
                        dur_vals.push(ns);
                    }

                    let s = helpers::compute_stats(&mops);
                    let mean_lf = lf_vals.iter().sum::<f64>() / trials as f64;
                    let mean_dur = dur_vals.iter().sum::<f64>() / trials as f64;
                    writeln!(
                        $csv,
                        "{},{},{},{},{},{},{},{:.6},{:.0},{:.4},{:.4},{:.4},{:.4}",
                        $scheme,
                        cfg.arity,
                        num_buckets,
                        cfg.bucket_size,
                        fp_bits,
                        c.max_kicks,
                        count,
                        mean_lf,
                        mean_dur,
                        s.mean,
                        s.min,
                        s.max,
                        s.stddev
                    )
                    .unwrap();
                    println!(
                        "  {:<16} nb={:<9} b={} | mean={:>7.3}  std={:>6.3} Mops  ({} deleted)",
                        $label, num_buckets, cfg.bucket_size, s.mean, s.stddev, count
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
            bench_delete!(
                csv,
                cli,
                "Segmented 2-ary",
                Segmented2aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_delete!(
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
            bench_delete!(
                csv,
                cli,
                "Segmented 3-ary",
                Segmented3aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_delete!(
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
            bench_delete!(
                csv,
                cli,
                "Segmented 4-ary",
                Segmented4aryCuckooFilter,
                "segmented",
                cfg,
                seg
            );
            bench_delete!(
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

    println!("=== Cuckoo filter — delete throughput (delete all items from a full filter) ===");
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

    let mut csv = helpers::csv_writer("cuckoo_filter_delete_throughput.csv", HEADER);
    for cfg in cfgs {
        run_config(&mut csv, &cli, cfg);
    }
    println!("\nResults written to cuckoo_filter_delete_throughput.csv");
}
