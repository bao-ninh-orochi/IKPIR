//! **Intent:** Measure server-side setup cost (deriving matrix A from seed
//! and computing the hint matrix) across both backends (FrodoPIR / SimplePIR).
//!
//! **Method:** Two interchangeable timing paths producing the *same*
//! single-threaded, non-SIMD setup number (the paper's reported regime):
//!
//! - **Full (default):** populate a store to `--load-factor`, wrap it in
//!   `IkpirServer::new`, and time the wall-clock cost of that call — the
//!   per-segment hint precompute `H_j = A_jᵀ·D_j` run sequentially over all
//!   `arity` segments. This is the ground truth.
//! - **Per-segment estimate (`--estimate`):** `IkpirServer::new` computes the
//!   `arity` hints in an independent sequential loop, and every segment has
//!   the *same* shape (`segment_rows × row_width`) and uniform per-row work.
//!   So timing the public per-segment primitive `B::server_setup` over **one**
//!   full segment and multiplying by `arity` reproduces the full
//!   single-threaded time exactly:
//!   `full = arity × time(server_setup over one segment)`.
//!   This computes `1/arity` of the hint work, so it is ~`arity×` faster while
//!   never leaving one thread / non-SIMD. The store is still built so the
//!   timed segment sees a real per-segment `D` (the database contents never
//!   affect setup timing — both backends' `compute_hint` multiply
//!   unconditionally — but the real build keeps the estimate honest).
//!
//! Repeats for `trials` trials after `warmup` warmup rounds.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (defaults to backend recommendation), `--load-factor`
//! (default 0.90 — matches `server_mutation`, the bench this one is the
//! static baseline for; setup time itself is fill-independent), `--trials`,
//! `--warmup`, `--estimate` (time one segment × arity instead of the full
//! `IkpirServer::new`; default off = full ground truth).
//!
//! **Output:** `results/ikpir_server_setup.csv`
//! Columns: backend, arity, num_buckets, bucket_size, value_bits, plaintext_bits,
//! lwe_dim, mean_setup_ms, min_setup_ms, max_setup_ms, stddev_setup_ms,
//! setup_bundle_bytes, hint_bytes_per_segment, server_params_bytes_per_segment,
//! cells_per_slot, row_width, segment_rows, db_rows, db_cols, load_factor,
//! setup_mode, measured_ms
//!
//! `mean_setup_ms` (and the min/max/stddev siblings) is the **full
//! single-threaded time** in every row — measured directly on the full path,
//! and the `arity ×` one-segment estimate on the `--estimate` path — so
//! existing analyses read the right number unchanged. The two trailing
//! columns are for traceability: `setup_mode` is `full|per_segment`, and
//! `measured_ms` is the raw per-call time *before* the `arity ×` scaling
//! (equals `mean_setup_ms` on the full path; the one-segment time on the
//! estimate path).
//!
//! `db_rows` / `db_cols` report the per-segment PIR matrix shape **after** any
//! backend-specific reshape. For FrodoPIR this is `(segment_rows, row_width)`;
//! for SimplePIR this is the post-reshape `(⌈segment_rows/k⌉, k·row_width)`.

mod helpers;

use helpers::{Backend, CloneStore};
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer, IncrementalPirBackend,
    IndexPirBackend, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "backend,arity,num_buckets,bucket_size,value_bits,plaintext_bits,lwe_dim,\
    mean_setup_ms,min_setup_ms,max_setup_ms,stddev_setup_ms,\
    setup_bundle_bytes,hint_bytes_per_segment,server_params_bytes_per_segment,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor,\
    setup_mode,measured_ms";

#[derive(clap::Parser)]
#[command(
    about = "Measure ikpir-server setup wall-clock cost (full IkpirServer::new over all \
    segments, or a one-segment × arity estimate)."
)]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    #[arg(long, default_value_t = 16_384)]
    num_buckets: u32,
    #[arg(long, default_value_t = 4)]
    bucket_size: u32,
    /// Value width in bits. The paper reports 2048 (256 B) and 8192 (1 kB).
    #[arg(long, default_value_t = 2048)]
    value_bits: u32,
    #[arg(long, default_value_t = 32)]
    fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]
    plaintext_bits: u32,
    /// LWE dimension. Defaults to 1566 (Frodo) or 1275 (Simple) when omitted.
    #[arg(long)]
    lwe_dim: Option<u32>,
    /// Fill to seed the store to before timing setup.
    ///
    /// Setup time does not depend on it — both backends' `compute_hint`
    /// multiply every cell unconditionally, so only the matrix shape matters —
    /// but the paper reports Setup as the rebuild a static scheme pays on every
    /// mutation, against `server_mutation` on the same store. Sharing that
    /// bench's fill keeps the two tables describing one store, and the CSV's
    /// `load_factor` column honest about which.
    #[arg(long, default_value_t = 0.90)]
    load_factor: f64,
    /// Untimed warmup runs before measurement. One by default: the first
    /// `IkpirServer::new` on a cold process pays page-fault and allocator
    /// warmup that later runs do not, so discarding it tightens the mean.
    #[arg(long, default_value_t = 1)]
    warmup: u32,
    /// Timed runs; the CSV reports mean/min/max/stddev over them. Three by
    /// default so the error bar is real rather than a single-sample `±0.000`.
    /// Setup is a deterministic linear pass with low variance, so three is
    /// plenty — raise it only if a machine is noisy. Each run is a full
    /// `Θ(nNw)` hint rebuild (seconds to minutes at paper scale), so this
    /// trades wall time for the error bar; pass `--estimate` to cut each run
    /// to one segment × arity.
    #[arg(long, default_value_t = 3)]
    trials: u32,
    /// Estimate the full single-threaded setup time by timing the per-segment
    /// hint precompute (`B::server_setup`) over ONE full segment and
    /// multiplying by `arity`, instead of timing the full `IkpirServer::new`
    /// over all `arity` segments. The `arity` segments are identical in shape
    /// and computed in an independent loop, so `full = arity × one-segment`
    /// is exact and ~`arity×` faster. The store is still built (real `D`).
    #[arg(long, default_value_t = false)]
    estimate: bool,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim
        .unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn run_one<S, B>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    make_config: impl Fn() -> B::Config,
) where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();
    let nb_is_default = matches.value_source("num_buckets") != Some(ValueSource::CommandLine);

    let lwe_dim_eff = effective_lwe_dim(cli);
    let pb = cli.plaintext_bits;

    let setup_mode_str = if cli.estimate { "per_segment" } else { "full" };

    // Both paths build the full store: the full path feeds it to
    // `IkpirServer::new`; the estimate path slices segment 0's real `D` out of
    // it. Build + snapshot once, then drop the store before timing.
    let (seed_store, n_inserted) = helpers::populate_to_load::<S>(
        cli.load_factor,
        num_buckets,
        cli.bucket_size,
        cli.fingerprint_bits,
        cli.value_bits,
        pb,
    );
    let params = seed_store.params();
    let segment_rows = params.segment_size();
    let cps = params.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let cells = seed_store.snapshot_cells();
    drop(seed_store);

    // Filled by whichever timing path runs below.
    let s: helpers::Stats; // full single-threaded time (measured, or arity×-scaled)
    let measured_ms: f64; // raw per-call time, pre-scale
    let (mut bundle_bytes, mut hint_bytes, mut sp_bytes) = (0usize, 0usize, 0usize);

    if !cli.estimate {
        // ── Full ground truth: time IkpirServer::new over all `arity` segments ──
        let mut samples = Vec::with_capacity((cli.warmup + cli.trials) as usize);
        for trial in 0..(cli.warmup + cli.trials) {
            let store = S::clone_from_cells(cells.clone(), params, n_inserted).expect("from_cells");
            let cfg = make_config();
            let t = Instant::now();
            let server: IkpirServer<S, B> = IkpirServer::new(store, cfg);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if trial == cli.warmup {
                let bundle = server.setup();
                bundle_bytes = bundle.wire_byte_size();
                hint_bytes = B::hint_byte_size(&bundle.hints[0]);
                sp_bytes = B::server_params_byte_size(&bundle.backend_params[0]);
            }
            if trial >= cli.warmup {
                samples.push(ms);
            }
        }
        s = helpers::compute_stats(&samples);
        measured_ms = s.mean;
    } else {
        // ── Per-segment estimate: time B::server_setup over ONE full segment,
        // then scale by `arity`. Single-threaded; computes 1/arity of the work ──
        let seg_db = &cells[..(segment_rows as usize) * (row_width as usize)];

        let mut samples = Vec::with_capacity((cli.warmup + cli.trials) as usize);
        for trial in 0..(cli.warmup + cli.trials) {
            let cfg = make_config();
            let t = Instant::now();
            let (sp, _mat, hint) = B::server_setup(&cfg, seg_db, segment_rows, row_width, pb);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if trial == cli.warmup {
                hint_bytes = B::hint_byte_size(&hint);
                sp_bytes = B::server_params_byte_size(&sp);
                // Mirror ServerSetupBundle::wire_byte_size for `arity` identical
                // segments: CuckooParams (1 + 5·u32 = 21) + epoch (8) + two Vec
                // length prefixes (4 + 4) + arity × (server_params + hint).
                // Exact for both backends: the estimate times one *full* segment,
                // so its hint shape equals the real per-segment hint.
                bundle_bytes = 21 + 8 + 4 + 4 + (arity as usize) * (sp_bytes + hint_bytes);
            }
            if trial >= cli.warmup {
                samples.push(ms);
            }
        }
        let raw = helpers::compute_stats(&samples);
        let scale = arity as f64;
        s = helpers::Stats {
            mean: raw.mean * scale,
            min: raw.min * scale,
            max: raw.max * scale,
            stddev: raw.stddev * scale,
        };
        measured_ms = raw.mean;
    }

    // ── Shared tail: derived geometry, CSV row, preamble ──
    let (db_rows, db_cols) =
        helpers::backend_shape_estimate(cli.backend, segment_rows as u64, row_width as u64);
    let load_factor = n_inserted as f64 / (num_buckets as u64 * cli.bucket_size as u64) as f64;

    writeln!(
        csv,
        "{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{:.3},{:.3},{:.3},{:.3},\
         {bundle_bytes},{hint_bytes},{sp_bytes},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4},\
         {setup_mode_str},{:.3}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits,
        s.mean, s.min, s.max, s.stddev, load_factor, measured_ms,
    ).unwrap();

    let store_state = helpers::StoreState {
        capacity: (num_buckets as u64) * (cli.bucket_size as u64),
        populated: n_inserted,
        load_pct: 100.0 * load_factor,
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes: hint_bytes,
        setup_bundle_bytes: bundle_bytes,
        query_bytes: 0,
        response_bytes: 0,
        hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob {
            name: "backend",
            value: cli.backend.to_string(),
            is_default: matches.value_source("backend") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "arity",
            value: arity.to_string(),
            is_default: matches.value_source("arity") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "num_buckets",
            value: num_buckets.to_string(),
            is_default: nb_is_default,
        },
        helpers::Knob {
            name: "bucket_size",
            value: cli.bucket_size.to_string(),
            is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "fingerprint_bits",
            value: cli.fingerprint_bits.to_string(),
            is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "value_bits",
            value: cli.value_bits.to_string(),
            is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "plaintext_bits",
            value: cli.plaintext_bits.to_string(),
            is_default: matches.value_source("plaintext_bits") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "lwe_dim",
            value: lwe_dim_eff.to_string(),
            is_default: cli.lwe_dim.is_none(),
        },
        helpers::Knob {
            name: "load_factor",
            value: format!("{:.2}", cli.load_factor),
            is_default: matches.value_source("load_factor") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "setup_mode",
            value: setup_mode_str.to_string(),
            is_default: !cli.estimate,
        },
    ];
    helpers::print_preamble("server_setup", &knobs, &store_state, &geom);

    let est_note = if cli.estimate {
        format!(" [est: 1 seg × arity={arity}, raw={measured_ms:.3} ms/seg]")
    } else {
        String::new()
    };
    println!(
        "  backend={} arity={arity} nb={num_buckets:<7} bs={} vb={:<4} | \
         mean={:.3} ms (±{:.3}){} | bundle={}B hint/seg={}B sp/seg={}B",
        cli.backend,
        cli.bucket_size,
        cli.value_bits,
        s.mean,
        s.stddev,
        est_note,
        bundle_bytes,
        hint_bytes,
        sp_bytes,
    );
}

fn dispatch_backend<S: CloneStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    let lwe_dim = effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo => run_one::<S, FrodoPirBackend>(csv, cli, arity, num_buckets, || {
            FrodoConfig::with_lwe_dim(lwe_dim)
        }),
        Backend::Simple => run_one::<S, SimplePirBackend>(csv, cli, arity, num_buckets, || {
            SimpleConfig::with_lwe_dim(lwe_dim)
        }),
    }
}

fn main() {
    if helpers::skip_when_cargo_test() {
        return;
    }
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets =
        if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
            cli.num_buckets
        } else {
            helpers::default_num_buckets_for_arity(cli.arity)
        };

    let mut csv = helpers::csv_writer("ikpir_server_setup.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_setup.csv");
}
