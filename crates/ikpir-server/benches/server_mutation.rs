//! **Intent:** Measure server-side insert / update / delete throughput and
//! total delta wire cost for N mutations per kind, across both backends
//! (FrodoPIR and SimplePIR) and both hint-patch realizations
//! (entry-level and row-level).
//!
//! **Method:** Populate to `--load-factor` (default 0.90), snapshot the cell
//! array, then for each (patch mode, mutation kind) pair clone the snapshot
//! and time N consecutive mutations with wall-clock. Reports n_succeeded,
//! total_ms, ops_per_s = n_succeeded / total_ms × 1000, and the total delta
//! bytes produced by those mutations (identical across patch modes — the
//! wire format does not depend on the realization).
//!
//! Wall-clock batch timing is used (not Criterion) because store state
//! changes between mutations, making cycling meaningless.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--patch-mode` (entry|row, comma-separated list,
//! default entry), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (backend-dependent default), `--n-mutations` (default 1024),
//! `--load-factor` (default 0.90).
//!
//! **Output:** `results/ikpir_server_mutation.csv`
//! Columns: backend, mutation_kind, patch_mode, arity, num_buckets,
//! bucket_size, value_bits, plaintext_bits, lwe_dim, n_mutations,
//! load_factor, n_attempted, n_succeeded, total_ms, ops_per_s,
//! delta_bytes_total, cells_per_slot, row_width, segment_rows, db_rows,
//! db_cols
//!
//! `db_rows` / `db_cols` report the per-segment PIR matrix shape **after** any
//! backend-specific reshape. For FrodoPIR this is `(segment_rows, row_width)`;
//! for SimplePIR this is the post-reshape `(⌈segment_rows/k⌉, k·row_width)`.

mod helpers;

use helpers::{Backend, CloneStore, PatchMode};
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer, IncrementalPirBackend,
    IndexPirBackend, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{
    CuckooParams, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str =
    "backend,mutation_kind,patch_mode,arity,num_buckets,bucket_size,value_bits,plaintext_bits,\
    lwe_dim,n_mutations,load_factor,n_attempted,n_succeeded,total_ms,ops_per_s,delta_bytes_total,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols";

#[derive(Clone, Copy, Debug)]
enum MutationKind {
    Insert,
    Update,
    Delete,
}
impl MutationKind {
    const fn as_csv(self) -> &'static str {
        match self {
            Self::Insert => "insert",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
    const fn all() -> &'static [Self] {
        &[Self::Insert, Self::Update, Self::Delete]
    }
}

#[derive(clap::Parser)]
#[command(
    about = "Measure server insert/update/delete throughput and delta bytes for N mutations."
)]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    /// Hint-patch realization(s) to sweep, comma-separated (entry|row).
    /// One CSV row per (kind, mode) pair.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "entry")]
    patch_mode: Vec<PatchMode>,
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
    #[arg(long, default_value_t = 1024)]
    n_mutations: u32,
    #[arg(long, default_value_t = 0.90)]
    load_factor: f64,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim
        .unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn fill_value(value: &mut [u8], key: u32, salt: u32) {
    for (i, b) in value.iter_mut().enumerate() {
        *b = (key.wrapping_mul(salt).wrapping_add(i as u32) & 0xFF) as u8;
    }
}

struct KindResult {
    n_attempted: u32,
    n_succeeded: u32,
    total_ms: f64,
    delta_bytes_total: usize,
}

fn run_kind<S, B>(
    cli: &Cli,
    cells: &[u32],
    params: CuckooParams,
    n_seed: u64,
    kind: MutationKind,
    mode: PatchMode,
    make_config: &impl Fn() -> B::Config,
) -> KindResult
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
{
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];

    let mut store = S::clone_from_cells(cells.to_vec(), params, n_seed).expect("from_cells");
    // `from_cells` resets `max_kicks` to `MAX_KICKS_DEFAULT` (500); restore the
    // 2_500 budget the populate helper used so the timed insert loop runs with
    // the same cuckoo-eviction headroom as the populate phase.
    store.set_max_kicks(2_500);
    let mut server: IkpirServer<S, B> = IkpirServer::new(store, make_config());
    server.set_hint_patch_mode(mode.to_hint_patch_mode());

    let mut delta_bytes_total = 0usize;
    let mut n_succeeded = 0u32;
    let n_attempted = cli.n_mutations;

    let t = Instant::now();
    for i in 0..cli.n_mutations {
        let res = match kind {
            MutationKind::Insert => {
                let k = n_seed as u32 + i;
                fill_value(&mut value, k, 31);
                server.insert(&k.to_le_bytes(), &value)
            }
            MutationKind::Update => {
                let k = (n_seed as u32 - 1) - (i % n_seed as u32);
                fill_value(&mut value, k, 47);
                server.update(&k.to_le_bytes(), &value)
            }
            MutationKind::Delete => {
                let k = (n_seed as u32 - 1) - (i % n_seed as u32);
                server.delete(&k.to_le_bytes())
            }
        };
        match res {
            Ok(bundle) => {
                delta_bytes_total += bundle.wire_byte_size();
                n_succeeded += 1;
            }
            Err(IkpirError::TableFull) => {}
            Err(e) => panic!("server_mutation kind={}: {e:?}", kind.as_csv()),
        }
    }
    let total_ms = t.elapsed().as_secs_f64() * 1e3;

    KindResult {
        n_attempted,
        n_succeeded,
        total_ms,
        delta_bytes_total,
    }
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
    let lwe_dim_eff = effective_lwe_dim(cli);

    let (seed_store, n_seed) = helpers::populate_to_load::<S>(
        cli.load_factor,
        num_buckets,
        cli.bucket_size,
        cli.fingerprint_bits,
        cli.value_bits,
        cli.plaintext_bits,
    );

    if n_seed < cli.n_mutations as u64 {
        eprintln!("  Skip: n_seed={n_seed} < n_mutations={}", cli.n_mutations);
        return;
    }

    let cells = seed_store.snapshot_cells();
    let params = seed_store.params();

    let cps = params.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let segment_rows = params.segment_size();
    let (db_rows, db_cols) =
        helpers::backend_shape_estimate(cli.backend, segment_rows as u64, row_width as u64);
    let store_state = helpers::StoreState {
        capacity: (num_buckets as u64) * (cli.bucket_size as u64),
        populated: n_seed,
        load_pct: 100.0 * n_seed as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes: 0,
        setup_bundle_bytes: 0,
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
            name: "patch_mode",
            value: helpers::patch_modes_label(&cli.patch_mode),
            is_default: matches.value_source("patch_mode") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "arity",
            value: arity.to_string(),
            is_default: matches.value_source("arity") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "num_buckets",
            value: num_buckets.to_string(),
            is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "bucket_size",
            value: cli.bucket_size.to_string(),
            is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine),
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
            name: "n_mutations",
            value: cli.n_mutations.to_string(),
            is_default: matches.value_source("n_mutations") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "load_factor",
            value: format!("{:.2}", cli.load_factor),
            is_default: matches.value_source("load_factor") != Some(ValueSource::CommandLine),
        },
    ];
    helpers::print_preamble("server_mutation", &knobs, &store_state, &geom);

    let mut modes = cli.patch_mode.clone();
    modes.dedup();
    for &mode in &modes {
        for &kind in MutationKind::all() {
            let r = run_kind::<S, B>(cli, &cells, params, n_seed, kind, mode, &make_config);
            let ops_per_s = if r.total_ms > 0.0 {
                r.n_succeeded as f64 / r.total_ms * 1e3
            } else {
                0.0
            };
            writeln!(
                csv,
                "{},{},{mode},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{},{:.2},{},{},{:.3},{:.2},{},{cps},{row_width},{segment_rows},{db_rows},{db_cols}",
                cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits, cli.plaintext_bits,
                cli.n_mutations, cli.load_factor,
                r.n_attempted, r.n_succeeded, r.total_ms, ops_per_s, r.delta_bytes_total,
            ).unwrap();
            println!(
                "  backend={} kind={:<6} mode={mode:<5} arity={arity} nb={num_buckets:<7} N={:<4} | \
                 succ={}/{} total={:.1}ms ops/s={:.0} delta={}B",
                cli.backend,
                kind.as_csv(),
                cli.n_mutations,
                r.n_succeeded,
                r.n_attempted,
                r.total_ms,
                ops_per_s,
                r.delta_bytes_total,
            );
        }
    }
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

    let mut csv = helpers::csv_writer("ikpir_server_mutation.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_mutation.csv");
}
