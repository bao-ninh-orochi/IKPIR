//! Shared bench configuration for `segmented-cuckoo`: the paper's config
//! matrix, the default filter tunables, and the CLI that overrides them.
//!
//! # Purpose
//!
//! Single source of truth for *what configuration a bench runs at*. Every
//! bench in this crate defaults to the five `(arity, bucket_size)` pairs of
//! Table 2 in the CANS 2026 paper and accepts CLI flags to narrow or override
//! them. Keeping the matrix here means the benches agree on geometry by
//! construction, and a paper revision is a one-file edit.
//!
//! # Design / architecture
//!
//! - [`PAPER_CONFIGS`] — the five rows of Table 2, as [`FilterConfig`] values.
//! - [`ConfigCli`] — the shared flag set, `#[command(flatten)]`ed into each
//!   bench's own `Cli` so bench-specific flags (`--value-bits`, `--hit-rate`,
//!   …) live with the bench that uses them.
//! - [`ConfigCli::configs`] — resolves flags to the list of configs to run:
//!   no selector runs the full matrix, a selector narrows or synthesizes.
//!
//! # Related files
//!
//! - `benches/helpers.rs` — CSV writing and trial statistics.
//! - `benches/cuckoo_filter_*.rs` and `benches/kv_store_*.rs` — consumers.
//! - `../../scripts/table2.sh` — runs the benches that populate Table 2.

// Each bench includes this module with `mod configs;`, and no single bench uses
// every item — that is the point of a shared module, not dead code.
#![allow(dead_code)]

// ─── Default tunables ────────────────────────────────────────────────────────

/// Fingerprint width, in bits.
///
/// The paper's filter evaluation uses a 32-bit fingerprint (Section 6.2), the
/// widest the slot layout admits. It drives the false-positive rate to
/// `arity·bucket_size / 2^32` — low enough that a false positive never
/// perturbs a load-factor or throughput measurement, so those numbers isolate
/// the indexing scheme rather than the fingerprint width. The
/// `cuckoo_filter_false_positive_rate` bench sweeps this axis deliberately and
/// so ignores this default.
pub const DEFAULT_FINGERPRINT_BITS: u32 = 32;

/// Cuckoo kick budget: relocations attempted before an insert reports
/// `TableFull`.
///
/// At ~10^6 buckets a small budget caps the achievable load factor well below
/// the scheme's true threshold, because the eviction chains near saturation are
/// longer than the budget allows to follow. 2500 is high enough that the
/// measured load factor reflects the scheme rather than the budget (the paper's
/// configs land within 0.7 percentage points of the d-partite threshold), while
/// staying low enough that a failed insert is detected promptly.
pub const DEFAULT_MAX_KICKS: u32 = 2500;

/// Untimed trials run before measurement, to warm caches and branch predictors.
pub const DEFAULT_WARMUP_TRIALS: usize = 3;

/// Timed trials per config; the CSV reports mean/min/max/stddev over these.
pub const DEFAULT_MEASURE_TRIALS: usize = 10;

/// Trials for `cuckoo_filter_load_factor`, which needs a tighter error bar than
/// the throughput benches because it reports a single headline number per
/// config and the randomized eviction chain is its only source of variance.
pub const DEFAULT_LOAD_FACTOR_TRIALS: usize = 20;

/// Arity used when `--bucket-size` is given without `--arity`.
pub const DEFAULT_ARITY: u32 = 4;

/// Bucket size used when `--arity` is given without `--bucket-size` and no
/// paper row matches that arity.
pub const DEFAULT_BUCKET_SIZE: u32 = 4;

// ─── Bucket counts ───────────────────────────────────────────────────────────
//
// Table 2 fixes ~10^6 buckets and compares the two schemes at equal table size.
// The two schemes admit different sizes, so "equal" is as close as their
// divisibility constraints allow:
//
//   scheme            constraint on nb      chosen value
//   ────────────────  ────────────────────  ───────────────────────
//   Segmented 2-ary   power of 2, ≥ 2       2^20     = 1 048 576
//   Standard  2-ary   power of 2            2^20     = 1 048 576
//   Segmented 3-ary   3·2^m                 3·2^19   = 1 572 864
//   Standard  3-ary   power of 3            3^13     = 1 594 323
//   Segmented 4-ary   power of 2, ≥ 4       2^20     = 1 048 576
//   Standard  4-ary   power of 4            4^10     = 1 048 576
//
// At arities 2 and 4 both ladders hit 2^20 exactly. At arity 3 they cannot
// coincide — one is 3·2^m, the other 3^k — so the two land 1.4% apart, the
// closest pair above 10^6. Every value exceeds 2^20 buckets, so each filter
// holds ≥ 10^6 slots even at `bucket_size = 1`.
//
// Note these depend only on arity, not on bucket_size: Table 2's five rows carry
// three distinct (segmented, standard) pairs between them. They are computed
// from arity below rather than repeated per row, so the five rows cannot drift.

/// Bucket count for a segmented filter of the given arity, per Table 2.
///
/// # Constraints
///
/// Panics if `arity` is not 2, 3, or 4.
pub const fn segmented_num_buckets(arity: u32) -> u32 {
    match arity {
        2 | 4 => 1 << 20,   // 1 048 576
        3 => 3 * (1 << 19), // 1 572 864
        _ => panic!("arity must be 2, 3, or 4"),
    }
}

/// Bucket count for a standard filter of the given arity, per Table 2.
///
/// # Constraints
///
/// Panics if `arity` is not 2, 3, or 4.
pub const fn standard_num_buckets(arity: u32) -> u32 {
    match arity {
        2 => 1 << 20,   // 1 048 576 = 2^20
        3 => 1_594_323, // 3^13 — the power of 3 nearest the segmented 3-ary size
        4 => 1 << 20,   // 1 048 576 = 4^10
        _ => panic!("arity must be 2, 3, or 4"),
    }
}

// ─── Config matrix ───────────────────────────────────────────────────────────

/// One `(arity, bucket_size)` cell of the config matrix, carrying the bucket
/// count each scheme runs at.
///
/// The two counts differ only at arity 3 (see the table above); a bench reads
/// whichever field matches the scheme it is about to construct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilterConfig {
    /// Number of candidate buckets per item (2, 3, or 4).
    pub arity: u32,
    /// Slots per bucket.
    pub bucket_size: u32,
    /// Bucket count for the segmented filter / KV store.
    pub segmented_num_buckets: u32,
    /// Bucket count for the standard filter.
    pub standard_num_buckets: u32,
}

impl FilterConfig {
    /// Build a config at the paper's bucket counts for `arity`.
    ///
    /// # Constraints
    ///
    /// Panics if `arity` is not 2, 3, or 4.
    pub const fn new(arity: u32, bucket_size: u32) -> Self {
        Self {
            arity,
            bucket_size,
            segmented_num_buckets: segmented_num_buckets(arity),
            standard_num_buckets: standard_num_buckets(arity),
        }
    }

    /// Total slot count for the segmented table (`num_buckets · bucket_size`).
    pub const fn segmented_capacity(&self) -> u64 {
        self.segmented_num_buckets as u64 * self.bucket_size as u64
    }

    /// Total slot count for the standard table.
    pub const fn standard_capacity(&self) -> u64 {
        self.standard_num_buckets as u64 * self.bucket_size as u64
    }
}

/// The five `(arity, bucket_size)` configurations of Table 2 — the ones RisePIR
/// is instantiated at, and the default matrix for every bench in this crate.
///
/// Only these five are reported: they are the configurations whose load factor
/// makes them worth instantiating RisePIR at, and each one carries a matching
/// row in the keyword-PIR tables. Lower bucket sizes at a given arity waste
/// table space (2-ary at `b = 1` peaks near 50%), and the omitted cells add no
/// information about the segmented-vs-standard comparison. Pass `--arity` /
/// `--bucket-size` to run a cell outside this set.
///
/// The set is mirrored on the PIR side by `PAPER_PIR_CONFIGS` in
/// `../../scripts/lib.sh`, which pairs each cell with the bucket count the
/// keyword-PIR benches run it at. Adding or dropping a cell means editing both.
pub const PAPER_CONFIGS: &[FilterConfig] = &[
    FilterConfig::new(2, 4),
    FilterConfig::new(3, 2),
    FilterConfig::new(3, 3),
    FilterConfig::new(4, 1),
    FilterConfig::new(4, 2),
];

// ─── CLI ─────────────────────────────────────────────────────────────────────

/// Flags shared by every `segmented-cuckoo` bench.
///
/// Flattened into each bench's own `Cli` with `#[command(flatten)]`. Every
/// field has a default, so a bare `cargo bench` runs the full paper matrix at
/// the paper's tunables; each flag overrides one axis of that.
#[derive(clap::Args, Clone, Debug)]
pub struct ConfigCli {
    /// Filter arity. Omit to run every arity in the paper matrix.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))]
    pub arity: Option<u32>,

    /// Slots per bucket. Omit to run every bucket size in the paper matrix.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=4))]
    pub bucket_size: Option<u32>,

    /// Bucket count, overriding the paper's per-arity value for *both* schemes.
    ///
    /// Each scheme constrains this differently (2-ary: power of 2; standard
    /// 3-ary: power of 3; standard 4-ary: power of 4; segmented 3-ary: 3·2^m),
    /// so a value legal for one scheme may be rejected by another — the bench
    /// prints a skip line for the schemes that reject it and runs the rest.
    #[arg(long)]
    pub num_buckets: Option<u32>,

    /// Fingerprint width in bits.
    ///
    /// Left as `Option` rather than defaulted here because the benches disagree
    /// on what "unset" means: most want [`DEFAULT_FINGERPRINT_BITS`], while
    /// `cuckoo_filter_false_positive_rate` sweeps the whole legal range and
    /// treats an explicit value as "measure only this width". Each bench
    /// resolves it with an `unwrap_or` at the point of use.
    #[arg(long)]
    pub fingerprint_bits: Option<u32>,

    /// Cuckoo kick budget before an insert reports `TableFull`.
    #[arg(long, default_value_t = DEFAULT_MAX_KICKS)]
    pub max_kicks: u32,

    /// Untimed warmup trials per config.
    #[arg(long, default_value_t = DEFAULT_WARMUP_TRIALS)]
    pub warmup: usize,

    /// Timed trials per config.
    ///
    /// `Option` for the same reason as `fingerprint_bits`:
    /// `cuckoo_filter_load_factor` defaults to
    /// [`DEFAULT_LOAD_FACTOR_TRIALS`], the rest to [`DEFAULT_MEASURE_TRIALS`].
    #[arg(long)]
    pub trials: Option<usize>,
}

impl ConfigCli {
    /// Resolve the flags to the list of configs to run.
    ///
    /// # Returns
    ///
    /// - Neither `--arity` nor `--bucket-size`: the full [`PAPER_CONFIGS`]
    ///   matrix.
    /// - Both: exactly that one config, whether or not it is a paper row.
    /// - One of the two: the paper rows matching it; if none do, one config at
    ///   that value and the default for the other axis.
    ///
    /// `--num-buckets`, when given, overrides both bucket counts on every
    /// resolved config.
    pub fn configs(&self) -> Vec<FilterConfig> {
        let mut out: Vec<FilterConfig> = match (self.arity, self.bucket_size) {
            (None, None) => PAPER_CONFIGS.to_vec(),
            (Some(a), Some(b)) => vec![FilterConfig::new(a, b)],
            (Some(a), None) => {
                let rows: Vec<_> = PAPER_CONFIGS
                    .iter()
                    .copied()
                    .filter(|c| c.arity == a)
                    .collect();
                if rows.is_empty() {
                    vec![FilterConfig::new(a, DEFAULT_BUCKET_SIZE)]
                } else {
                    rows
                }
            }
            (None, Some(b)) => {
                let rows: Vec<_> = PAPER_CONFIGS
                    .iter()
                    .copied()
                    .filter(|c| c.bucket_size == b)
                    .collect();
                if rows.is_empty() {
                    vec![FilterConfig::new(DEFAULT_ARITY, b)]
                } else {
                    rows
                }
            }
        };
        if let Some(nb) = self.num_buckets {
            for c in &mut out {
                c.segmented_num_buckets = nb;
                c.standard_num_buckets = nb;
            }
        }
        out
    }

    /// One line describing the resolved run, for the bench's stdout preamble.
    ///
    /// Reports only what every bench shares: the config matrix, the fingerprint
    /// width, and the kick budget. Trial counts are deliberately *not* included
    /// — they mean different things per bench (`cuckoo_filter_load_factor` has
    /// no warmup, `cuckoo_filter_false_positive_rate` has no trial loop at all),
    /// so each bench appends its own rather than have this print a number that
    /// does not apply.
    ///
    /// # Arguments
    ///
    /// - `configs` — the output of [`ConfigCli::configs`].
    /// - `fingerprint_bits` — how this bench resolved the flag; `None` prints
    ///   "swept", for `cuckoo_filter_false_positive_rate`.
    pub fn describe(&self, configs: &[FilterConfig], fingerprint_bits: Option<u32>) -> String {
        let matrix = if self.arity.is_none() && self.bucket_size.is_none() {
            "paper matrix (Table 2)".to_string()
        } else {
            format!("{} config(s)", configs.len())
        };
        let fp = match fingerprint_bits {
            Some(f) => f.to_string(),
            None => "swept".to_string(),
        };
        format!(
            "{matrix}: fingerprint_bits={fp}, max_kicks={}",
            self.max_kicks
        )
    }
}

/// Parse a bench CLI from `std::env::args`, tolerating cargo's own arguments.
///
/// # Purpose
///
/// `cargo bench` passes `--bench` to the harness binary, which clap would
/// reject as unknown. This drops it and enables `ignore_errors` so any other
/// cargo-injected argument is tolerated rather than aborting the run.
///
/// # Returns
///
/// The parsed CLI. Mirrors `parse_cli_with_matches` in the `ikpir-server` /
/// `ikpir-client` bench helpers.
pub fn parse<C: clap::Parser>() -> C {
    let args: Vec<String> = std::env::args().filter(|a| a != "--bench").collect();
    let m = C::command().ignore_errors(true).get_matches_from(&args);
    C::from_arg_matches(&m).expect("parse CLI")
}
