//! **Intent:** Measure client-side `apply_delta` throughput for N mutations
//! per kind (insert / update / delete), in **empty-queue mode** (no
//! precomputed slots), across both backends. The measurement isolates the
//! cost of patching the hint `H` from any warm-bc queue-maintenance work,
//! giving a clean number for the "compute new hint" cost reported in the
//! paper.
//!
//! **Method:** Populate to `--load-factor`, build a fresh client with no
//! precompute (empty prepared-query queue), collect N deltas per kind from
//! a fresh server clone, then time the full sequence of N apply_delta calls
//! with wall-clock Instant. The timed loop runs exactly once (state
//! advances with each mutation, so criterion cycling is not meaningful).
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (backend-dependent default), `--n-mutations` (default 1024),
//! `--load-factor` (default 0.90).
//!
//! **Output:** `results/ikpir_client_mutation.csv`
//! Columns: backend, mutation_kind, arity, num_buckets, bucket_size,
//! value_bits, plaintext_bits, lwe_dim, n_mutations, load_factor, n_succeeded,
//! total_ms, ops_per_s, cells_per_slot, row_width, segment_rows, db_rows, db_cols
//!
//! `db_rows` / `db_cols` report the per-segment PIR matrix shape **after** any
//! backend-specific reshape. For FrodoPIR this is `(segment_rows, row_width)`;
//! for SimplePIR this is the post-reshape `(⌈segment_rows/k⌉, k·row_width)`.

mod helpers;

use helpers::{Backend, CloneStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, HintDeltaBundle, IkpirClient,
    IncrementalPirBackend, IndexPirBackend, SimpleConfig, SimplePirBackend,
};
use ikpir_server::{IkpirError, IkpirServer};
use segmented_cuckoo::{
    CuckooParams, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str =
    "backend,mutation_kind,arity,num_buckets,bucket_size,value_bits,plaintext_bits,lwe_dim,\
    n_mutations,load_factor,n_succeeded,total_ms,ops_per_s,\
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

#[derive(Clone, clap::Parser)]
#[command(
    about = "Measure client apply_delta throughput for N mutations per kind (empty queue, hint-patch only)."
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
    #[arg(long, default_value_t = 256)]
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

fn collect_deltas_for_kind<S, B>(
    cells: &[u32],
    params: CuckooParams,
    n_seed: u64,
    cli: &Cli,
    kind: MutationKind,
    make_config: &impl Fn() -> B::Config,
) -> (Vec<HintDeltaBundle<B>>, u32)
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend,
{
    let store = S::clone_from_cells(cells.to_vec(), params, n_seed).expect("from_cells");
    let mut server: IkpirServer<S, B> = IkpirServer::new(store, make_config());
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut deltas = Vec::with_capacity(cli.n_mutations as usize);
    let mut n_succeeded = 0u32;

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
                deltas.push(bundle);
                n_succeeded += 1;
            }
            Err(IkpirError::TableFull) => break,
            Err(e) => panic!("collect_deltas kind={}: {e:?}", kind.as_csv()),
        }
    }
    (deltas, n_succeeded)
}

fn run_one<S, B>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    make_config: impl Fn() -> B::Config,
) where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize + Clone,
    B::Query: Clone,
    B::Response: Clone,
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
    if n_seed < 2 {
        eprintln!("  Skip: seed too small");
        return;
    }

    let cells = seed_store.snapshot_cells();
    let params = seed_store.params();

    let seed_store2 = S::clone_from_cells(cells.clone(), params, n_seed).expect("from_cells");
    let seed_server: IkpirServer<S, B> = IkpirServer::new(seed_store2, make_config());
    let bundle = seed_server.setup();

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
        setup_bundle_bytes: bundle.wire_byte_size(),
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
    helpers::print_preamble("client_mutation", &knobs, &store_state, &geom);

    for &kind in MutationKind::all() {
        let (deltas, n_succeeded) =
            collect_deltas_for_kind::<S, B>(&cells, params, n_seed, cli, kind, &make_config);
        if deltas.is_empty() {
            eprintln!("  Skip kind={}: no deltas collected", kind.as_csv());
            continue;
        }

        // Build a fresh client with an empty prepared-query queue. No
        // precompute_queries / precompute_decodes — every apply_delta
        // patches only the hint H (the queue iteration in
        // `client_patch_state` is a no-op when the queue is empty), so
        // the timing reflects the "compute new hint" cost in isolation.
        let mut client = IkpirClient::<B>::from_setup(bundle.clone());

        // Wall-clock time the full N apply_delta sequence.
        let t = Instant::now();
        for d in deltas {
            client.apply_delta(d).expect("apply_delta");
        }
        let total_ms = t.elapsed().as_secs_f64() * 1e3;
        let ops_per_s = n_succeeded as f64 / total_ms * 1e3;

        writeln!(
            csv,
            "{},{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{},{:.2},{},{:.3},{:.2},{cps},{row_width},{segment_rows},{db_rows},{db_cols}",
            cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits, cli.plaintext_bits,
            cli.n_mutations, cli.load_factor, n_succeeded, total_ms, ops_per_s,
        ).unwrap();
        println!(
            "  backend={} kind={:<6} arity={arity} nb={num_buckets:<7} N={:<4} | \
             {:.2} apply_delta/s (total={:.1}ms)",
            cli.backend,
            kind.as_csv(),
            cli.n_mutations,
            ops_per_s,
            total_ms,
        );
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
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets =
        if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
            cli.num_buckets
        } else {
            helpers::default_num_buckets_for_arity(cli.arity)
        };

    let mut csv = helpers::csv_writer("ikpir_client_mutation.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_mutation.csv");
}
