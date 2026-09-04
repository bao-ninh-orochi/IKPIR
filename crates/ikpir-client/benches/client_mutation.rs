//! **Intent:** Measure client-side per-batch **maintenance** throughput for N
//! mutations per kind (insert / update / delete), in **empty-queue mode** (no
//! precomputed slots), across both backends, both update strategies
//! (`--update-mode patch,rewind`), and — for `patch` — both hint-patch
//! realizations (entry-level and row-level). `patch` times the bench-only
//! `HintPatchClient::apply_delta` (gated behind the `hint-patch-bench`
//! feature; recompute the hint, `Θ(n·τ·ω)`) — the classical baseline
//! response-rewind replaced in production; `rewind` times the production
//! `IkpirClient::accumulate_delta` (roll up the published `ΔD`, `Θ(τ·ω)` —
//! the paper's factor-`n` cheaper client maintenance,
//! `docs/rewind-client-mode.md`). The measurement isolates that per-batch
//! maintenance cost from any warm-bc queue-maintenance work.
//!
//! **Method:** Populate to `--load-factor`, snapshot the cell array, and
//! build **one** server per config with `IkpirServer::new_parallel`; its
//! epoch-0 `setup()` bundle is both the client bootstrap and the hints
//! every replay rewinds to. For each kind, rewind that server with
//! `IkpirServer::reset_for_replay` (a fresh store from the snapshot cells
//! plus a clone of the epoch-0 hints — the seed-derived `A` is kept, so the
//! post-reset state *is* the epoch-0 state) and collect N deltas (deltas
//! are identical under either patch mode). Then per patch mode build a
//! fresh `HintPatchClient` from the epoch-0 bundle with no precompute (empty
//! prepared-query queue), set its `HintPatchMode`, and time the full
//! sequence of N `apply_delta` calls with wall-clock Instant; per update
//! mode build a fresh production `IkpirClient` and time
//! `accumulate_delta` instead. The timed loop runs exactly once per (kind,
//! mode) (state advances with each mutation, so criterion cycling is not
//! meaningful). Previously every kind built its own server, four
//! `Θ(d · ρ · n · ω)` setups per config; the timed region is unchanged.
//! `tests/replay_equivalence.rs` pins that a replay yields the same deltas
//! as a fresh setup.
//!
//! **Update values:** salt 47 with offset 1, i.e. `(47k + i + 1) & 0xFF`
//! against the seeded `(17k + i) & 0xFF`, so every byte differs by
//! `30k + 1` — odd, never 0 mod 256. Before this change (offset 0) keys
//! `k ≡ 0 (mod 128)` re-wrote their own value: wire-level no-ops that did no
//! hint-patch work yet counted in `n_succeeded` (~0.8 % of updates).
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--update-mode` (patch|rewind, comma-separated, default
//! `patch,rewind`), `--patch-mode` (entry|row, comma-separated, default entry;
//! applies only to `patch`), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (backend-dependent default), `--n-mutations` (default 1024),
//! `--load-factor` (default 0.90). One CSV row per (kind, update_mode
//! [, patch_mode]).
//!
//! **Output:** `results/ikpir_client_mutation.csv`
//! Columns: backend, mutation_kind, update_mode, patch_mode, arity,
//! num_buckets, bucket_size, value_bits, plaintext_bits, lwe_dim, n_mutations,
//! load_factor, n_succeeded, total_ms, ops_per_s, pending_cells,
//! cells_per_slot, row_width, segment_rows, db_rows, db_cols
//! (`pending_cells` = final |ΔD| nonzero cells; 0 for `patch`, the Θ(τ·ω) set
//! `S` for `rewind`. `patch_mode` = `-` for `rewind` rows.)
//!
//! `db_rows` / `db_cols` report the per-segment PIR matrix shape **after** any
//! backend-specific reshape. For FrodoPIR this is `(segment_rows, row_width)`;
//! for SimplePIR this is the post-reshape `(⌈segment_rows/k⌉, k·row_width)`.

mod helpers;

use helpers::{Backend, CloneStore, PatchMode, UpdateMode};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, HintDeltaBundle, HintPatchClient, IkpirClient,
    IncrementalPirBackend, IndexPirBackend, ParallelSetupBackend, SimpleConfig, SimplePirBackend,
};
use ikpir_server::{IkpirError, IkpirServer};
use segmented_cuckoo::{
    CuckooParams, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str =
    "backend,mutation_kind,update_mode,patch_mode,arity,num_buckets,bucket_size,value_bits,\
    plaintext_bits,lwe_dim,n_mutations,load_factor,n_succeeded,total_ms,ops_per_s,pending_cells,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols";

/// Cuckoo kick budget for every replayed store. `from_cells` resets
/// `max_kicks` to `MAX_KICKS_DEFAULT` (500); the populate helper used 2_500,
/// and the insert deltas must come from the same eviction regime that
/// `server_mutation` times.
const MAX_KICKS: u32 = 2_500;

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
    about = "Measure client per-batch maintenance throughput for N mutations per kind (empty queue): \
             apply_delta (patch) vs accumulate_delta (rewind)."
)]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    /// Hint-patch realization(s) to sweep, comma-separated (entry|row).
    /// Applies only to `--update-mode patch`.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "entry")]
    patch_mode: Vec<PatchMode>,
    /// Client update strategy(ies) to sweep, comma-separated (patch|rewind).
    /// `patch` times `apply_delta` (once per `--patch-mode`); `rewind` times
    /// `accumulate_delta` (patch-mode-independent). One CSV row per
    /// (kind, update_mode[, patch_mode]).
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "patch,rewind"
    )]
    update_mode: Vec<UpdateMode>,
    #[arg(long, default_value_t = 16_384)]
    num_buckets: u32,
    #[arg(long, default_value_t = 4)]
    bucket_size: u32,
    /// Value width in bits. The paper reports 2048 (256 B) and 8192 (1 kB).
    #[arg(long, default_value_t = 2048)]
    value_bits: u32,
    #[arg(long, default_value_t = 64)]
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

/// `value[i] = (key · salt + i + offset) & 0xFF`. The seed pattern written by
/// `helpers::populate_to_load` is `salt = 17, offset = 0`.
fn fill_value(value: &mut [u8], key: u32, salt: u32, offset: u32) {
    for (i, b) in value.iter_mut().enumerate() {
        *b = (key
            .wrapping_mul(salt)
            .wrapping_add(i as u32)
            .wrapping_add(offset)
            & 0xFF) as u8;
    }
}

/// The epoch-0 store, as `from_cells` needs it: every replay rebuilds a
/// fresh store from these.
struct Snapshot<'a> {
    cells: &'a [u32],
    params: CuckooParams,
    n_seed: u64,
}

/// Rewind `server` to epoch 0 and collect the deltas of N mutations of
/// `kind`. Stops at the first `TableFull` (inserts only).
fn collect_deltas_for_kind<S, B>(
    server: &mut IkpirServer<S, B>,
    hints0: &[B::Hint],
    snap: &Snapshot<'_>,
    cli: &Cli,
    kind: MutationKind,
) -> (Vec<HintDeltaBundle<B>>, u32)
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend,
{
    let mut store =
        S::clone_from_cells(snap.cells.to_vec(), snap.params, snap.n_seed).expect("from_cells");
    store.set_max_kicks(MAX_KICKS);
    server.reset_for_replay(store, hints0.to_vec());

    let n_seed = snap.n_seed;
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut deltas = Vec::with_capacity(cli.n_mutations as usize);
    let mut n_succeeded = 0u32;

    for i in 0..cli.n_mutations {
        let res = match kind {
            MutationKind::Insert => {
                let k = n_seed as u32 + i;
                fill_value(&mut value, k, 31, 0);
                server.insert(&k.to_le_bytes(), &value)
            }
            MutationKind::Update => {
                let k = (n_seed as u32 - 1) - (i % n_seed as u32);
                // Differs from the seeded value in every byte: 30k + 1 is odd.
                fill_value(&mut value, k, 47, 1);
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
    B: IndexPirBackend + ParallelSetupBackend + IncrementalPirBackend + BackendWireSize + Clone,
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
    let snap = Snapshot {
        cells: &cells,
        params,
        n_seed,
    };

    // The one setup per config. This first clone is never mutated: every
    // kind's delta collection starts by swapping in a fresh clone through
    // `reset_for_replay`, so its kick budget is irrelevant.
    let store0 = S::clone_from_cells(cells.clone(), params, n_seed).expect("from_cells");
    let mut server: IkpirServer<S, B> = IkpirServer::new_parallel(store0, make_config());
    // Epoch 0: the client bootstrap bundle *and* the hints every replay
    // rewinds to.
    let bundle = server.setup();
    let (db_rows, db_cols) = B::db_matrix_shape(&server.backend_params()[0]);

    let cps = params.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let segment_rows = params.segment_size();
    let store_state = helpers::StoreState {
        capacity: (num_buckets as u64) * (cli.bucket_size as u64),
        populated: n_seed,
        load_pct: 100.0 * n_seed as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes: bundle.hints.first().map_or(0, B::hint_byte_size),
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
            name: "patch_mode",
            value: helpers::patch_modes_label(&cli.patch_mode),
            is_default: matches.value_source("patch_mode") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "update_mode",
            value: helpers::update_modes_label(&cli.update_mode),
            is_default: matches.value_source("update_mode") != Some(ValueSource::CommandLine),
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

    let mut patch_modes = cli.patch_mode.clone();
    patch_modes.dedup();
    let mut update_modes = cli.update_mode.clone();
    update_modes.dedup();
    for &kind in MutationKind::all() {
        let (deltas, n_succeeded) =
            collect_deltas_for_kind::<S, B>(&mut server, &bundle.hints, &snap, cli, kind);
        if deltas.is_empty() {
            eprintln!("  Skip kind={}: no deltas collected", kind.as_csv());
            continue;
        }

        // The deltas are identical under either mode (the wire format does not
        // depend on the realization), so they are collected once per kind and
        // replayed per (update_mode, patch_mode). `patch` sweeps the hint-patch
        // realization via the bench-only `HintPatchClient`; `rewind` is
        // patch-mode-independent, so it runs once (patch_mode column `-`)
        // against the production `IkpirClient`.
        for &umode in &update_modes {
            match umode {
                UpdateMode::Patch => {
                    for &pmode in &patch_modes {
                        // Fresh comparator client, no precompute: times only the
                        // hint patch — the `Θ(n·τ·ω)` client-maintenance cost the
                        // production rewind client replaced.
                        let mut client = HintPatchClient::<B>::from_setup_parallel(bundle.clone());
                        client.set_hint_patch_mode(pmode.to_hint_patch_mode());

                        let replay = deltas.clone();
                        let t = Instant::now();
                        for d in replay {
                            client.apply_delta(d).expect("apply_delta");
                        }
                        let total_ms = t.elapsed().as_secs_f64() * 1e3;
                        let ops_per_s = n_succeeded as f64 / total_ms * 1e3;
                        // Always 0: HintPatchClient patches the hint directly, no ΔD.
                        let pending_cells = 0usize;
                        let pmode_str = pmode.to_string();

                        writeln!(
                            csv,
                            "{},{},{umode},{pmode_str},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{},{:.2},{},{:.3},{:.2},{pending_cells},{cps},{row_width},{segment_rows},{db_rows},{db_cols}",
                            cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits, cli.plaintext_bits,
                            cli.n_mutations, cli.load_factor, n_succeeded, total_ms, ops_per_s,
                        ).unwrap();
                        println!(
                            "  backend={} kind={:<6} update={umode:<6} patch={pmode_str:<5} arity={arity} nb={num_buckets:<7} N={:<4} | \
                             {:.2} ops/s (total={:.1}ms, |ΔD|={pending_cells})",
                            cli.backend,
                            kind.as_csv(),
                            cli.n_mutations,
                            ops_per_s,
                            total_ms,
                        );
                    }
                }
                UpdateMode::Rewind => {
                    // Fresh client, empty prepared-query queue (no precompute):
                    // times only the ΔD accumulate — the `Θ(τ·ω)` client
                    // maintenance the paper reports, a factor-`n` cheaper than
                    // the hint-patch comparator above.
                    let mut client = IkpirClient::<B>::from_setup_parallel(bundle.clone());

                    let replay = deltas.clone();
                    let t = Instant::now();
                    for d in replay {
                        client.accumulate_delta(d).expect("accumulate_delta");
                    }
                    let total_ms = t.elapsed().as_secs_f64() * 1e3;
                    let ops_per_s = n_succeeded as f64 / total_ms * 1e3;
                    // The accumulated nonzero-cell count — the Θ(τ·ω) set S.
                    let pending_cells = client.pending_cells();
                    let pmode_str = "-".to_string();

                    writeln!(
                        csv,
                        "{},{},{umode},{pmode_str},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{},{:.2},{},{:.3},{:.2},{pending_cells},{cps},{row_width},{segment_rows},{db_rows},{db_cols}",
                        cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits, cli.plaintext_bits,
                        cli.n_mutations, cli.load_factor, n_succeeded, total_ms, ops_per_s,
                    ).unwrap();
                    println!(
                        "  backend={} kind={:<6} update={umode:<6} patch={pmode_str:<5} arity={arity} nb={num_buckets:<7} N={:<4} | \
                         {:.2} ops/s (total={:.1}ms, |ΔD|={pending_cells})",
                        cli.backend,
                        kind.as_csv(),
                        cli.n_mutations,
                        ops_per_s,
                        total_ms,
                    );
                }
            }
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

    let mut csv = helpers::csv_writer("ikpir_client_mutation.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_mutation.csv");
}
