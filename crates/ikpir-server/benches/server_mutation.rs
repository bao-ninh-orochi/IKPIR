//! **Intent:** Measure server-side insert / update / delete throughput and
//! the delta-transcript wire cost of N mutations per kind, across both
//! backends (FrodoPIR and SimplePIR) and both hint-patch realizations
//! (entry-level and row-level) — next to the fresh-hint download that
//! transcript competes with.
//!
//! **Method:** Populate to `--load-factor` (default 0.90) and snapshot the
//! cell array. Build **one** server per config with
//! `IkpirServer::new_parallel` — the `Θ(d · ρ · n · ω)` setup that takes
//! minutes at paper scale — and capture its epoch-0 hints and setup bundle
//! before any mutation. Then, for each (patch mode, mutation kind) pair,
//! rewind that same server with `IkpirServer::reset_for_replay` (a fresh
//! store from the snapshot cells plus a clone of the epoch-0 hints; the
//! seed-derived `A` is kept, so the post-reset state *is* the epoch-0
//! state) and apply N consecutive mutations. Six replays per config, one
//! setup — previously every pair paid its own `new_parallel`.
//! `ikpir-client/tests/replay_equivalence.rs` pins that a replay yields the
//! same deltas and reaches the same hints as a fresh setup.
//!
//! **Timing:** each `server.insert` / `update` / `delete` call is bracketed
//! on its own with `Instant`, and `total_ms` is the sum of those brackets;
//! `ops_per_s = n_succeeded / total_ms × 1000`. Key selection, the `Result`
//! `match`, and the per-op accounting all sit outside the bracket. One
//! bracket used to wrap the whole loop, which was harmless while the
//! accounting was a closed-form estimate; it is now real work — `encode()`
//! and `wire_stats()` walk every delta cell, `O(cells)` per op — so it must
//! not be charged to the server. Wall-clock is used rather than Criterion
//! because store state advances with every mutation, so cycling one call
//! is meaningless.
//!
//! **Update values:** salt 47 with offset 1, i.e. `(47k + i + 1) & 0xFF`
//! against the seeded `(17k + i) & 0xFF`, so every byte differs by
//! `30k + 1` — odd, never 0 mod 256. Before this change (offset 0) keys
//! `k ≡ 0 (mod 128)` re-wrote their own value: wire-level no-ops that did no
//! hint-patch work yet counted in `n_succeeded` (~0.8 % of updates).
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
//! delta_bytes_total, delta_rows_total, delta_runs_total,
//! delta_cells_total, delta_nonzero_cells_total, setup_bundle_bytes,
//! hint_bytes_total, delta_encoding, cells_per_slot, row_width,
//! segment_rows, db_rows, db_cols
//!
//! The transcript columns sum over the row's `n_succeeded` deltas; every
//! count is what `HintDeltaBundle::wire_stats` reports for the v1 encoding
//! (`docs/hint-delta-wire-format.md`):
//!
//! - `delta_bytes_total` — `Σ encode().len()`: the exact v1 transcript bytes
//!   (§6), i.e. what a client `n_succeeded` epochs behind downloads to
//!   catch up incrementally. Identical across patch modes — the wire format
//!   does not depend on the realization.
//! - `delta_rows_total` — touched rows, summed over segments.
//! - `delta_runs_total` — emitted runs (§5 rule 2: a new run starts at a
//!   gap wider than `G`).
//! - `delta_cells_total` — delta literals: run lengths summed, **including**
//!   the interior zeros bridged inside a run (§6 `cells`).
//! - `delta_nonzero_cells_total` — nonzero cells only: the size of the
//!   sparse edit set `S`, the `Θ(τ·w)` quantity the paper's asymptotics
//!   count.
//! - `setup_bundle_bytes` — `ServerSetupBundle::wire_byte_size()` at epoch
//!   0: the fresh-hint download a lagging client would fetch *instead* of
//!   the transcript (§9). `delta_bytes_total / setup_bundle_bytes` is the
//!   transcript-to-hint ratio the console line prints.
//! - `hint_bytes_total` — `Σ_j B::hint_byte_size(hints[j])` over the
//!   segments: the hint part of `setup_bundle_bytes` (the remainder is
//!   params, seeds, and length prefixes).
//! - `delta_encoding` — the literal `v1`, so a row can never be misread
//!   against a future encoding.
//!
//! `db_rows` / `db_cols` report the per-segment PIR matrix shape **after** any
//! backend-specific reshape. For FrodoPIR this is `(segment_rows, row_width)`;
//! for SimplePIR this is the post-reshape `(⌈segment_rows/k⌉, k·row_width)`.

mod helpers;

use helpers::{Backend, CloneStore, PatchMode};
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer, IncrementalPirBackend,
    IndexPirBackend, ParallelSetupBackend, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{
    CuckooParams, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::{Duration, Instant};

const HEADER: &str =
    "backend,mutation_kind,patch_mode,arity,num_buckets,bucket_size,value_bits,plaintext_bits,\
    lwe_dim,n_mutations,load_factor,n_attempted,n_succeeded,total_ms,ops_per_s,delta_bytes_total,\
    delta_rows_total,delta_runs_total,delta_cells_total,delta_nonzero_cells_total,\
    setup_bundle_bytes,hint_bytes_total,delta_encoding,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols";

/// Wire-encoding version the transcript columns are measured under — the
/// `delta_encoding` CSV column (`docs/hint-delta-wire-format.md`).
const DELTA_ENCODING: &str = "v1";

/// Cuckoo kick budget for every replayed store. `from_cells` resets
/// `max_kicks` to `MAX_KICKS_DEFAULT` (500); the populate helper used 2_500,
/// and the insert loop must run with the same eviction headroom.
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

#[derive(clap::Parser)]
#[command(
    about = "Measure server insert/update/delete throughput and delta transcript bytes for N mutations."
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

struct KindResult {
    n_attempted: u32,
    n_succeeded: u32,
    /// Sum of the per-op brackets around the `server.*` call alone.
    total_ms: f64,
    delta_bytes_total: usize,
    delta_rows_total: u64,
    delta_runs_total: u64,
    delta_cells_total: u64,
    delta_nonzero_cells_total: u64,
}

/// Run `f` and add its wall-clock duration to `acc`. The bracket is exactly
/// the call: nothing else in the mutation loop is timed.
fn timed<T>(acc: &mut Duration, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let out = f();
    *acc += t.elapsed();
    out
}

fn run_kind<S, B>(
    cli: &Cli,
    server: &mut IkpirServer<S, B>,
    snap: &Snapshot<'_>,
    hints0: &[B::Hint],
    kind: MutationKind,
    mode: PatchMode,
) -> KindResult
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
{
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];

    // Rewind to epoch 0: a fresh store from the snapshot cells plus a clone
    // of the epoch-0 hints. The seed-derived `A` stays, so this is the
    // state `new_parallel` produced, without paying for it again.
    let mut store =
        S::clone_from_cells(snap.cells.to_vec(), snap.params, snap.n_seed).expect("from_cells");
    store.set_max_kicks(MAX_KICKS);
    server.reset_for_replay(store, hints0.to_vec());
    server.set_hint_patch_mode(mode.to_hint_patch_mode());

    let n_attempted = cli.n_mutations;
    let mut n_succeeded = 0u32;
    let mut total = Duration::ZERO;
    let mut delta_bytes_total = 0usize;
    let mut delta_rows_total = 0u64;
    let mut delta_runs_total = 0u64;
    let mut delta_cells_total = 0u64;
    let mut delta_nonzero_cells_total = 0u64;

    for i in 0..cli.n_mutations {
        // Key selection, outside the bracket: inserts take fresh keys past
        // the seed range; updates and deletes walk the seed keys downward.
        let k = match kind {
            MutationKind::Insert => snap.n_seed as u32 + i,
            MutationKind::Update | MutationKind::Delete => {
                (snap.n_seed as u32 - 1) - (i % snap.n_seed as u32)
            }
        };
        let key = k.to_le_bytes();
        let res = match kind {
            MutationKind::Insert => {
                fill_value(&mut value, k, 31, 0);
                timed(&mut total, || server.insert(&key, &value))
            }
            MutationKind::Update => {
                // Differs from the seeded value in every byte: 30k + 1 is odd.
                fill_value(&mut value, k, 47, 1);
                timed(&mut total, || server.update(&key, &value))
            }
            MutationKind::Delete => timed(&mut total, || server.delete(&key)),
        };
        match res {
            Ok(bundle) => {
                // Accounting, outside the bracket: the real v1 bytes and
                // the row / run / cell counts behind them.
                let bytes = bundle.encode();
                let st = bundle.wire_stats();
                assert_eq!(bytes.len(), st.bytes, "encode/wire_byte_size invariant");
                delta_bytes_total += bytes.len();
                delta_rows_total += st.rows;
                delta_runs_total += st.runs;
                delta_cells_total += st.cells;
                delta_nonzero_cells_total += st.nonzero_cells;
                n_succeeded += 1;
            }
            Err(IkpirError::TableFull) => {}
            Err(e) => panic!("server_mutation kind={}: {e:?}", kind.as_csv()),
        }
    }

    KindResult {
        n_attempted,
        n_succeeded,
        total_ms: total.as_secs_f64() * 1e3,
        delta_bytes_total,
        delta_rows_total,
        delta_runs_total,
        delta_cells_total,
        delta_nonzero_cells_total,
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
    B: IndexPirBackend + ParallelSetupBackend + IncrementalPirBackend + BackendWireSize,
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
    let snap = Snapshot {
        cells: &cells,
        params,
        n_seed,
    };

    // The one setup per config. This first clone is never mutated: every
    // (mode, kind) replay starts by swapping in a fresh clone through
    // `reset_for_replay`, so its kick budget is irrelevant.
    let store0 = S::clone_from_cells(cells.clone(), params, n_seed).expect("from_cells");
    let mut server: IkpirServer<S, B> = IkpirServer::new_parallel(store0, make_config());
    let (db_rows, db_cols) = B::db_matrix_shape(&server.backend_params()[0]);

    // Epoch-0 hints, captured before any mutation. `setup()` deep-clones
    // every hint, so call it once and take both sizes and the replay hints
    // from the same bundle.
    let bundle0 = server.setup();
    let setup_bundle_bytes = bundle0.wire_byte_size();
    let hint_bytes_total: usize = bundle0.hints.iter().map(B::hint_byte_size).sum();
    let hints0 = bundle0.hints;

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
        hint_per_seg_bytes: hints0.first().map_or(0, B::hint_byte_size),
        setup_bundle_bytes,
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
            let r = run_kind::<S, B>(cli, &mut server, &snap, &hints0, kind, mode);
            let ops_per_s = if r.total_ms > 0.0 {
                r.n_succeeded as f64 / r.total_ms * 1e3
            } else {
                0.0
            };
            let transcript_to_hint = r.delta_bytes_total as f64 / setup_bundle_bytes as f64;
            writeln!(
                csv,
                "{},{},{mode},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{},{:.2},{},{},{:.3},{:.2},{},{},{},{},{},{setup_bundle_bytes},{hint_bytes_total},{DELTA_ENCODING},{cps},{row_width},{segment_rows},{db_rows},{db_cols}",
                cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits, cli.plaintext_bits,
                cli.n_mutations, cli.load_factor,
                r.n_attempted, r.n_succeeded, r.total_ms, ops_per_s, r.delta_bytes_total,
                r.delta_rows_total, r.delta_runs_total, r.delta_cells_total,
                r.delta_nonzero_cells_total,
            ).unwrap();
            println!(
                "  backend={} kind={:<6} mode={mode:<5} arity={arity} nb={num_buckets:<7} N={:<4} | \
                 succ={}/{} total={:.1}ms ops/s={:.0} | delta={}B rows={} runs={} cells={} \
                 nonzero={} transcript/hint={:.5} (setup_bundle={}B)",
                cli.backend,
                kind.as_csv(),
                cli.n_mutations,
                r.n_succeeded,
                r.n_attempted,
                r.total_ms,
                ops_per_s,
                r.delta_bytes_total,
                r.delta_rows_total,
                r.delta_runs_total,
                r.delta_cells_total,
                r.delta_nonzero_cells_total,
                transcript_to_hint,
                setup_bundle_bytes,
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
