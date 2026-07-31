//! **Intent:** Measure the actual false-positive rate (FPR) as fingerprint width
//! varies, and compare it against the theoretical bound
//! `arity · bucket_size / 2^fingerprint_bits`.
//!
//! **Method:** For each config, sweep `fingerprint_bits` from the minimum the
//! geometry admits up to 40 by default; wider widths (up to 64) run only when
//! passed explicitly, since past ~40 bits the expected false-positive count
//! `k·b·q / 2^f` sits far below one for any feasible query budget and every
//! extra row would measure an exact zero. At each width: insert until
//! `TableFull`, then query `--num-queries` never-inserted keys and count how
//! many the filter reports as present.
//!
//! **Design rationale:** Segmentation changes where a candidate index lands, not
//! the probability that two keys share a fingerprint, so the segmented and
//! standard filters should track the same FPR curve — this bench is what makes
//! that claim falsifiable rather than assumed. The query count defaults to ten
//! times the table's capacity so that even at wide fingerprints the expected
//! false-positive count stays large enough for the measured rate to mean
//! something; with `q` that large the variance is low enough that one run per
//! width suffices, which is why this bench has no `--trials` loop.
//!
//! Unlike the other filter benches, this one *sweeps* `--fingerprint-bits`
//! rather than fixing it at the default 64 — that axis is the independent
//! variable here. Passing `--fingerprint-bits N` narrows the sweep to the single
//! width `N`.
//!
//! **Arguments (CLI):** all optional; with none, runs the paper matrix.
//! `--arity`, `--bucket-size`, `--num-buckets`, `--max-kicks`, plus
//! `--fingerprint-bits` (measure only this width) and `--num-queries` (default:
//! 10× capacity). See `benches/configs.rs`.
//!
//! **Output:** `results/segmented-cuckoo/cuckoo_filter_false_positive_rate/arity{a}_bucket_size{b}.csv`
//! Columns: fingerprint_bits, scheme, arity, num_buckets, bucket_size,
//! load_factor, num_inserted, num_queries, false_positives, fpr_pct,
//! theoretical_pct

mod configs;
mod helpers;

use configs::{ConfigCli, FilterConfig};
use segmented_cuckoo::{
    CuckooError, Segmented2aryCuckooFilter, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter,
    Standard2aryCuckooFilter, Standard3aryCuckooFilter, Standard4aryCuckooFilter,
};
use std::io::Write;

const HEADER: &str = "fingerprint_bits,scheme,arity,num_buckets,bucket_size,load_factor,\
                      num_inserted,num_queries,false_positives,fpr_pct,theoretical_pct";

#[derive(clap::Parser)]
#[command(about = "False-positive rate vs fingerprint width, segmented vs standard cuckoo filter.")]
struct Cli {
    #[command(flatten)]
    config: ConfigCli,

    /// Miss lookups per measurement. Defaults to 10× the table's capacity,
    /// which keeps the expected false-positive count meaningful at wide
    /// fingerprints.
    #[arg(long)]
    num_queries: Option<u64>,
}

/// Smallest fingerprint width the geometry admits: `⌊log2(arity·bucket_size)⌋+1`
/// (mirrors `validate_common_params` in `src/`).
const fn min_fingerprint_bits(arity: u32, bucket_size: u32) -> u32 {
    (arity * bucket_size).ilog2() + 1
}

/// Fill one filter type at one fingerprint width, count false positives on
/// `num_queries` misses, and write one CSV row.
macro_rules! run_fpr {
    ($csv:expr, $cli:expr, $label:expr, $filter_ty:ty, $scheme:expr, $cfg:expr, $num_buckets:expr, $fp_bits:expr) => {{
        let cfg: FilterConfig = $cfg;
        let num_buckets: u32 = $num_buckets;
        let fp_bits: u32 = $fp_bits;
        let c = &$cli.config;

        match <$filter_ty>::new(num_buckets, cfg.bucket_size, fp_bits) {
            Err(e) => eprintln!(
                "    Skip {} num_buckets={} bucket_size={} fp={}: {}",
                $label, num_buckets, cfg.bucket_size, fp_bits, e
            ),
            Ok(mut filter) => {
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

                // Query keys past the inserted range — guaranteed misses, so
                // every positive answer is a false positive.
                let q = $cli
                    .num_queries
                    .unwrap_or(10 * num_buckets as u64 * cfg.bucket_size as u64);
                let mut false_positives = 0u64;
                for j in inserted..inserted + q {
                    if filter.contain(j.to_le_bytes()) {
                        false_positives += 1;
                    }
                }

                let fpr_pct = false_positives as f64 / q as f64 * 100.0;
                // `2f64.powi` rather than `1u64 << fp_bits`: the sweep now reaches
                // fp_bits = 64, where the integer shift would overflow.
                let theoretical_pct =
                    cfg.arity as f64 * cfg.bucket_size as f64 / 2f64.powi(fp_bits as i32) * 100.0;

                writeln!(
                    $csv,
                    "{},{},{},{},{},{:.6},{},{},{},{:.6},{:.6}",
                    fp_bits,
                    $scheme,
                    cfg.arity,
                    num_buckets,
                    cfg.bucket_size,
                    lf,
                    inserted,
                    q,
                    false_positives,
                    fpr_pct,
                    theoretical_pct
                )
                .unwrap();
                println!(
                    "    fp_bits={:<3} {:<10} lf={:.4}% fpr={:.6}% theo={:.6}%",
                    fp_bits,
                    $scheme,
                    lf * 100.0,
                    fpr_pct,
                    theoretical_pct
                );
            }
        }
    }};
}

/// Sweep fingerprint widths for both schemes at one `(arity, bucket_size)`
/// config, into a config-specific CSV.
fn run_config(cli: &Cli, cfg: FilterConfig) {
    let widths: Vec<u32> = match cli.config.fingerprint_bits {
        Some(fb) => vec![fb],
        // Default sweep caps at 40: beyond that the expected false-positive
        // count is far below one at any feasible --num-queries, so every
        // extra row would report an exact zero against a rate the query
        // budget cannot resolve. `--fingerprint-bits 64` still measures the
        // widened path explicitly.
        None => (min_fingerprint_bits(cfg.arity, cfg.bucket_size)..=40).collect(),
    };

    let path = format!(
        "cuckoo_filter_false_positive_rate/arity{}_bucket_size{}.csv",
        cfg.arity, cfg.bucket_size
    );
    let mut csv = helpers::csv_writer(&path, HEADER);

    println!(
        "\n--- arity={} bucket_size={} (seg nb={}, std nb={}), fingerprint_bits {}..{} ---",
        cfg.arity,
        cfg.bucket_size,
        cfg.segmented_num_buckets,
        cfg.standard_num_buckets,
        widths.first().copied().unwrap_or(0),
        widths.last().copied().unwrap_or(0),
    );

    let (seg, std) = (cfg.segmented_num_buckets, cfg.standard_num_buckets);
    for fp_bits in widths {
        match cfg.arity {
            2 => {
                run_fpr!(
                    csv,
                    cli,
                    "Segmented 2-ary",
                    Segmented2aryCuckooFilter,
                    "segmented",
                    cfg,
                    seg,
                    fp_bits
                );
                run_fpr!(
                    csv,
                    cli,
                    "Standard 2-ary",
                    Standard2aryCuckooFilter,
                    "standard",
                    cfg,
                    std,
                    fp_bits
                );
            }
            3 => {
                run_fpr!(
                    csv,
                    cli,
                    "Segmented 3-ary",
                    Segmented3aryCuckooFilter,
                    "segmented",
                    cfg,
                    seg,
                    fp_bits
                );
                run_fpr!(
                    csv,
                    cli,
                    "Standard 3-ary",
                    Standard3aryCuckooFilter,
                    "standard",
                    cfg,
                    std,
                    fp_bits
                );
            }
            4 => {
                run_fpr!(
                    csv,
                    cli,
                    "Segmented 4-ary",
                    Segmented4aryCuckooFilter,
                    "segmented",
                    cfg,
                    seg,
                    fp_bits
                );
                run_fpr!(
                    csv,
                    cli,
                    "Standard 4-ary",
                    Standard4aryCuckooFilter,
                    "standard",
                    cfg,
                    std,
                    fp_bits
                );
            }
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

    println!(
        "=== Cuckoo filter — false-positive rate (insert until full, sweep fingerprint_bits) ==="
    );
    println!(
        "{}",
        cli.config.describe(&cfgs, cli.config.fingerprint_bits)
    );

    for cfg in cfgs {
        run_config(&cli, cfg);
    }
    println!("\nResults written under cuckoo_filter_false_positive_rate/");
}
