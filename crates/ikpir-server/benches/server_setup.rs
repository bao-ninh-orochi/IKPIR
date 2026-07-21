//! **Intent:** Measure server-side setup cost (deriving matrix A from seed
//! and computing the hint matrix) across both backends (FrodoPIR / SimplePIR).
//!
//! **This is the one bench that times setup.** On `main` it is therefore the
//! one bench that must run setup on the *reference* (single-threaded,
//! non-SIMD) implementation — the paper's reported regime — while every other
//! PIR bench builds its server and client through `IkpirServer::new_parallel`
//! / `IkpirClient::from_setup_parallel`, byte-identical state across all
//! cores, setup as preamble rather than measurement.
//!
//! **On `perf/optimized` that reference no longer exists by default:** the
//! `parallel` feature (default-on) makes `IkpirServer::new` itself run the
//! rayon kernels, so `--setup-impl reference` and `--setup-impl parallel` are
//! the same code path. The CSV's `setup_mode` column reports `full_parallel`
//! for both, and **no row from a default build is a paper number**. To
//! measure the paper's regime on this branch, force the reference schedule
//! with `IKPIR_SETUP_THREADS=1` (or build `--no-default-features`); then
//! `--setup-impl reference` reports `full` again and is comparable to
//! `main`.
//!
//! **Method:** populate a store to `--load-factor`, wrap it in
//! `IkpirServer::new`, and time the wall-clock cost of that call — the
//! per-segment hint precompute `H_j = A_jᵀ·D_j` run sequentially over all
//! `arity` segments. Every reported number is that call, measured end to
//! end: nothing is timed on a subset and scaled up. Repeats for `trials`
//! trials after `warmup` warmup rounds.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (defaults to backend recommendation), `--load-factor`
//! (default 0.90 — matches `server_mutation`, the bench this one is the
//! static baseline for; setup time itself is fill-independent), `--trials`,
//! `--warmup`, `--setup-impl` (`reference` | `parallel`, default
//! `reference`).
//!
//! **Output:** `results/ikpir_server_setup.csv`
//! Columns: backend, arity, num_buckets, bucket_size, value_bits, plaintext_bits,
//! lwe_dim, mean_setup_ms, min_setup_ms, max_setup_ms, stddev_setup_ms,
//! setup_bundle_bytes, hint_bytes_per_segment, server_params_bytes_per_segment,
//! cells_per_slot, row_width, segment_rows, db_rows, db_cols, load_factor,
//! setup_mode
//!
//! `mean_setup_ms` (and the min/max/stddev siblings) is the full
//! `IkpirServer::new` wall-clock. `setup_mode` (`full` | `full_parallel`)
//! records which setup implementation produced the row.
//!
//! A `full_parallel` row is **not** a paper number: the geometry columns still
//! describe the same configuration, but `mean_setup_ms` is then wall-clock on
//! `IKPIR_SETUP_THREADS` (default: all) cores, not the single-threaded cost.
//! On this branch that is what a default build always produces.
//!
//! `db_rows` / `db_cols` report the per-segment PIR matrix shape **after** any
//! backend-specific reshape. For FrodoPIR this is `(segment_rows, row_width)`;
//! for SimplePIR this is the post-reshape `(⌈segment_rows/k⌉, k·row_width)`.

mod helpers;

use helpers::{Backend, CloneStore};
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer, IncrementalPirBackend,
    IndexPirBackend, ParallelSetupBackend, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "backend,arity,num_buckets,bucket_size,value_bits,plaintext_bits,lwe_dim,\
    mean_setup_ms,min_setup_ms,max_setup_ms,stddev_setup_ms,\
    setup_bundle_bytes,hint_bytes_per_segment,server_params_bytes_per_segment,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor,\
    setup_mode";

/// Which realization of the setup phase to time.
///
/// The two produce byte-identical state (see `ParallelSetupBackend`);
/// only the wall-clock differs. Keeping the choice explicit — and
/// recording it in the CSV's `setup_mode` column — is what stops an
/// accidentally-parallel run from being read as a paper number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum SetupImpl {
    /// `IndexPirBackend::server_setup` — single-threaded, non-SIMD.
    /// **The paper's regime**, and the default.
    Reference,
    /// `ParallelSetupBackend::server_setup_parallel` — same output,
    /// fanned out over `IKPIR_SETUP_THREADS` (default: all) cores.
    /// Diagnostic only: quantifies what the other benches' preambles
    /// save.
    Parallel,
}

#[derive(clap::Parser)]
#[command(
    about = "Measure ikpir-server setup wall-clock cost: the full IkpirServer::new over all \
    segments."
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
    /// trades wall time for the error bar.
    #[arg(long, default_value_t = 3)]
    trials: u32,
    /// Which setup implementation to time. `reference` (default) is the
    /// single-threaded, non-SIMD path the paper reports. `parallel` times
    /// the byte-identical multi-threaded twin that every *other* bench
    /// uses in its (untimed) preamble — useful to quantify that saving,
    /// never a paper number. The CSV's `setup_mode` column records the
    /// choice as a `_parallel` suffix.
    #[arg(long, value_enum, default_value_t = SetupImpl::Reference)]
    setup_impl: SetupImpl,
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
    B: IndexPirBackend + ParallelSetupBackend + IncrementalPirBackend + BackendWireSize,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();
    let nb_is_default = matches.value_source("num_buckets") != Some(ValueSource::CommandLine);

    let lwe_dim_eff = effective_lwe_dim(cli);
    let pb = cli.plaintext_bits;

    let parallel = cli.setup_impl == SetupImpl::Parallel;
    // perf/optimized: with the default `parallel` feature the *reference*
    // entry points already run the rayon kernels, so `--setup-impl
    // reference` is not the paper's single-threaded regime and must not be
    // labelled as if it were. `IKPIR_SETUP_THREADS=1` (or
    // `--no-default-features`) does restore it, and the label follows.
    let threaded =
        cfg!(feature = "parallel") && ikpir_common::backend::parallel::setup_threads() > 1;
    let setup_mode_str = if parallel || threaded {
        "full_parallel"
    } else {
        "full"
    };

    // Build + snapshot the store once, then drop it: each trial re-derives its
    // own store from the snapshot so every timed `IkpirServer::new` sees the
    // same input.
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

    // Time IkpirServer::new over all `arity` segments; the measurement-round
    // trial also harvests the wire sizes and the backend's own matrix shape.
    let (mut bundle_bytes, mut hint_bytes, mut sp_bytes) = (0usize, 0usize, 0usize);
    let mut db_shape = (0u32, 0u32);

    let mut samples = Vec::with_capacity((cli.warmup + cli.trials) as usize);
    for trial in 0..(cli.warmup + cli.trials) {
        let store = S::clone_from_cells(cells.clone(), params, n_inserted).expect("from_cells");
        let cfg = make_config();
        let t = Instant::now();
        let server: IkpirServer<S, B> = if parallel {
            IkpirServer::new_parallel(store, cfg)
        } else {
            IkpirServer::new(store, cfg)
        };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if trial == cli.warmup {
            db_shape = B::db_matrix_shape(&server.backend_params()[0]);
            let bundle = server.setup();
            bundle_bytes = bundle.wire_byte_size();
            hint_bytes = B::hint_byte_size(&bundle.hints[0]);
            sp_bytes = B::server_params_byte_size(&bundle.backend_params[0]);
        }
        if trial >= cli.warmup {
            samples.push(ms);
        }
    }
    let s = helpers::compute_stats(&samples);

    // ── Derived geometry, CSV row, preamble ──
    let (db_rows, db_cols) = db_shape;
    let load_factor = n_inserted as f64 / (num_buckets as u64 * cli.bucket_size as u64) as f64;

    writeln!(
        csv,
        "{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{:.3},{:.3},{:.3},{:.3},\
         {bundle_bytes},{hint_bytes},{sp_bytes},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4},\
         {setup_mode_str}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits,
        s.mean, s.min, s.max, s.stddev, load_factor,
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
            is_default: !parallel,
        },
    ];
    helpers::print_preamble("server_setup", &knobs, &store_state, &geom);

    println!(
        "  backend={} arity={arity} nb={num_buckets:<7} bs={} vb={:<4} | \
         mean={:.3} ms (±{:.3}) | bundle={}B hint/seg={}B sp/seg={}B",
        cli.backend,
        cli.bucket_size,
        cli.value_bits,
        s.mean,
        s.stddev,
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
