//! Shared generic measurement body for the per-flow mutation benches
//! (`client_hint_patch_mutation.rs`, `client_rewind_mutation.rs`), included
//! via `mod flow_mutation_body;` the same way `mod helpers;` is. Each thin
//! binary supplies its own flow's client type at the two backend-dispatch
//! call sites; everything else — the generic `Cli<E>` (flow-specific `E`),
//! preamble, setup, delta collection, the timed maintenance loop, and CSV
//! writing — lives here, once.
//!
//! **Intent:** Measure one flow's client-side per-batch **maintenance**
//! throughput for N mutations per kind (insert / update / delete), in
//! **empty-queue mode** (no precomputed slots), across both backends. For
//! client-hint-patch this additionally sweeps `--patch-mode` (both hint-patch
//! realizations, entry-level and row-level): it times the
//! client-hint-patch flow's `HintPatchClient::apply_delta` (recompute the
//! hint, `Θ(n·τ·ω)`) once per patch mode. For client-rewind (no
//! `--patch-mode` flag — patch-mode-independent) it times the client-rewind
//! flow's `RewindClient::accumulate_delta` (roll up the published `ΔD`,
//! `Θ(τ·ω)` — the paper's factor-`n` cheaper client maintenance,
//! `docs/rewind-client-mode.md`). The measurement isolates that per-batch
//! maintenance cost from any warm-bc queue-maintenance work.
//!
//! **Method:** Populate to `--load-factor`, snapshot the cell array, and
//! build **one** server per config with `IkpirServer::new_parallel`; its
//! epoch-0 `setup()` bundle is both the client bootstrap and the hints
//! every replay rewinds to. For each kind, rewind that server with
//! `IkpirServer::reset_for_replay` (a fresh store from the snapshot cells
//! plus a clone of the epoch-0 hints — the seed-derived `A` is kept, so the
//! post-reset state *is* the epoch-0 state) and collect N deltas — shared
//! across every variant of that kind (identical under either patch mode).
//! Then, per **variant** (one per `--patch-mode` for client-hint-patch,
//! exactly one for client-rewind — `MutationFlow::variants` /
//! `MutationFlow::make_client`), build a fresh client from the epoch-0
//! bundle with no precompute (empty prepared-query queue) and time the full
//! sequence of N `sync_delta` calls with wall-clock Instant. The timed loop
//! runs exactly once per (kind, variant) (state advances with each
//! mutation, so criterion cycling is not meaningful).
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
//! default frodo), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (backend-dependent default), `--n-mutations` (default 1024),
//! `--load-factor` (default 0.90); client-hint-patch additionally takes
//! `--patch-mode` (entry|row, comma-separated, default entry). One CSV row
//! per (kind[, variant]).
//!
//! **Output:** `results/ikpir_client_hint_patch_mutation.csv` /
//! `results/ikpir_client_rewind_mutation.csv` (never merged — the CSV file
//! name is derived from the flow's `ClientFlow::FLOW`).
//!
//! Columns (client-hint-patch): flow, backend, mutation_kind, patch_mode,
//! arity, num_buckets, bucket_size, value_bits, plaintext_bits, lwe_dim,
//! n_mutations, load_factor, n_succeeded, total_ms, ops_per_s,
//! cells_per_slot, row_width, segment_rows, db_rows, db_cols.
//!
//! Columns (client-rewind): flow, backend, mutation_kind, arity, num_buckets,
//! bucket_size, value_bits, plaintext_bits, lwe_dim, n_mutations,
//! load_factor, n_succeeded, total_ms, ops_per_s, pending_cells,
//! cells_per_slot, row_width, segment_rows, db_rows, db_cols
//! (`pending_cells` = final |ΔD| nonzero cells, the Θ(τ·ω) set `S`).
//!
//! `db_rows` / `db_cols` report the per-segment PIR matrix shape **after** any
//! backend-specific reshape. For FrodoPIR this is `(segment_rows, row_width)`;
//! for SimplePIR this is the post-reshape `(⌈segment_rows/k⌉, k·row_width)`.

use crate::helpers::{self, ClientFlow, CloneStore, PatchMode};
use ikpir_client::{
    BackendWireSize, HintDeltaBundle, HintPatchClient, IncrementalPirBackend, IndexPirBackend,
    ParallelSetupBackend, PrecomputingPirBackend, ResponseRewind, RewindClient, ServerSetupBundle,
};
use ikpir_server::{IkpirError, IkpirServer};
use segmented_cuckoo::CuckooParams;
use std::io::Write;
use std::time::Instant;

/// Cuckoo kick budget for every replayed store. `from_cells` resets
/// `max_kicks` to `MAX_KICKS_DEFAULT` (500); the populate helper used 2_500,
/// and the insert deltas must come from the same eviction regime that
/// `server_mutation` times.
const MAX_KICKS: u32 = 2_500;

// ── Flow-specific CLI extras ─────────────────────────────────────────────────

/// `--patch-mode` — client-hint-patch only.
#[derive(Clone, clap::Args)]
pub struct PatchModeArgs {
    /// Hint-patch realization(s) to sweep, comma-separated (entry|row).
    #[arg(long, value_enum, value_delimiter = ',', default_value = "entry")]
    pub patch_mode: Vec<PatchMode>,
}

/// No extra flags — client-rewind (patch-mode-independent).
#[derive(Clone, clap::Args)]
pub struct NoExtraArgs {}

#[derive(Clone, clap::Parser)]
#[command(
    about = "Measure one client flow's per-batch maintenance throughput for N mutations per kind (empty queue)."
)]
pub struct Cli<E: clap::Args + Clone> {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    pub arity: u32,
    #[arg(long, value_enum, default_value_t = helpers::Backend::Frodo)]
    pub backend: helpers::Backend,
    #[arg(long, default_value_t = 16_384)]
    pub num_buckets: u32,
    #[arg(long, default_value_t = 4)]
    pub bucket_size: u32,
    /// Value width in bits. The paper reports 2048 (256 B) and 8192 (1 kB).
    #[arg(long, default_value_t = 2048)]
    pub value_bits: u32,
    #[arg(long, default_value_t = 64)]
    pub fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]
    pub plaintext_bits: u32,
    /// LWE dimension. Defaults to 1566 (Frodo) or 1275 (Simple) when omitted.
    #[arg(long)]
    pub lwe_dim: Option<u32>,
    #[arg(long, default_value_t = 1024)]
    pub n_mutations: u32,
    #[arg(long, default_value_t = 0.90)]
    pub load_factor: f64,
    #[command(flatten)]
    pub extra: E,
}

pub fn effective_lwe_dim<E: clap::Args + Clone>(cli: &Cli<E>) -> u32 {
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

/// The epoch-0 store, as `from_cells` needs it: every replay rebuilds a
/// fresh store from these.
struct Snapshot<'a> {
    cells: &'a [u32],
    params: CuckooParams,
    n_seed: u64,
}

/// Rewind `server` to epoch 0 and collect the deltas of N mutations of
/// `kind`. Stops at the first `TableFull` (inserts only). Shared, unchanged
/// across variants of a kind (the wire format does not depend on the patch
/// realization).
fn collect_deltas_for_kind<S, B>(
    server: &mut IkpirServer<S, B>,
    hints0: &[B::Hint],
    snap: &Snapshot<'_>,
    value_bits: u32,
    n_mutations: u32,
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
    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut deltas = Vec::with_capacity(n_mutations as usize);
    let mut n_succeeded = 0u32;

    for i in 0..n_mutations {
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

/// Fields shared by both flows' CSV rows — everything except the one
/// column that differs (`patch_mode` for client-hint-patch, `pending_cells`
/// for client-rewind, produced by `MutationFlow::format_row` instead).
pub struct MutationRowCommon {
    pub backend: String,
    pub kind: &'static str,
    pub arity: u32,
    pub num_buckets: u32,
    pub bucket_size: u32,
    pub value_bits: u32,
    pub plaintext_bits: u32,
    pub lwe_dim: u32,
    pub n_mutations: u32,
    pub load_factor: f64,
    pub n_succeeded: u32,
    pub total_ms: f64,
    pub ops_per_s: f64,
    pub cells_per_slot: u32,
    pub row_width: u32,
    pub segment_rows: u32,
    pub db_rows: u32,
    pub db_cols: u32,
}

/// Small per-flow extension of [`ClientFlow`] for the mutation bench's
/// variant sweep: client-hint-patch times one client per `--patch-mode`
/// (`Self::Variant = PatchMode`); client-rewind, which has no patch-mode
/// flag, times exactly one (`Self::Variant = ()`). Keeps the delta
/// collection (`collect_deltas_for_kind`, above) and the timed loop
/// (`run_one`, below) free of dynamic dispatch and free of per-flow
/// duplication — only the CSV shape (`HEADER` / `format_row`) and the
/// variant construction differ per flow.
pub trait MutationFlow<B>: ClientFlow<B>
where
    B: IndexPirBackend + ParallelSetupBackend + IncrementalPirBackend + PrecomputingPirBackend,
    B::Query: Clone,
    B::Response: Clone,
{
    /// Flow-specific CLI flags flattened into `Cli<Self::CliExtra>`:
    /// `PatchModeArgs` (client-hint-patch) or `NoExtraArgs` (client-rewind).
    type CliExtra: clap::Args + Clone;
    /// One value per client built and timed for a given kind: `PatchMode`
    /// for client-hint-patch, `()` for client-rewind.
    type Variant: Clone;
    /// CSV header for this flow's mutation CSV — differs from the other
    /// flow's only in whether the extra column is `patch_mode` or
    /// `pending_cells` (see the module docs).
    const HEADER: &'static str;

    /// Variants to sweep for one kind, derived from the parsed CLI extra.
    fn variants(extra: &Self::CliExtra) -> Vec<Self::Variant>;
    /// Extra preamble knobs beyond the shared set (the `patch_mode` sweep
    /// list for client-hint-patch; none for client-rewind).
    fn cli_knobs(extra: &Self::CliExtra, matches: &clap::ArgMatches) -> Vec<helpers::Knob>;
    /// Build a fresh, empty-queue client for `variant` from the epoch-0
    /// bundle (client-hint-patch additionally calls `set_hint_patch_mode`).
    fn make_client(bundle: ServerSetupBundle<B>, variant: &Self::Variant) -> Self;
    /// Render one CSV row (sans trailing newline) for `variant`'s completed
    /// timed run.
    fn format_row(&self, common: &MutationRowCommon, variant: &Self::Variant) -> String;
}

impl<B> MutationFlow<B> for HintPatchClient<B>
where
    B: IndexPirBackend + ParallelSetupBackend + IncrementalPirBackend + PrecomputingPirBackend,
    B::Query: Clone,
    B::Response: Clone,
{
    type CliExtra = PatchModeArgs;
    type Variant = PatchMode;
    const HEADER: &'static str =
        "flow,backend,mutation_kind,patch_mode,arity,num_buckets,bucket_size,value_bits,\
        plaintext_bits,lwe_dim,n_mutations,load_factor,n_succeeded,total_ms,ops_per_s,\
        cells_per_slot,row_width,segment_rows,db_rows,db_cols";

    fn variants(extra: &PatchModeArgs) -> Vec<PatchMode> {
        let mut modes = extra.patch_mode.clone();
        modes.dedup();
        modes
    }
    fn cli_knobs(extra: &PatchModeArgs, matches: &clap::ArgMatches) -> Vec<helpers::Knob> {
        use clap::parser::ValueSource;
        vec![helpers::Knob {
            name: "patch_mode",
            value: helpers::patch_modes_label(&extra.patch_mode),
            is_default: matches.value_source("patch_mode") != Some(ValueSource::CommandLine),
        }]
    }
    fn make_client(bundle: ServerSetupBundle<B>, variant: &PatchMode) -> Self {
        let mut client = Self::from_setup_parallel(bundle);
        client.set_hint_patch_mode(variant.to_hint_patch_mode());
        client
    }
    fn format_row(&self, c: &MutationRowCommon, variant: &PatchMode) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{:.2},{},{:.3},{:.2},{},{},{},{},{}",
            <Self as ClientFlow<B>>::FLOW,
            c.backend,
            c.kind,
            variant,
            c.arity,
            c.num_buckets,
            c.bucket_size,
            c.value_bits,
            c.plaintext_bits,
            c.lwe_dim,
            c.n_mutations,
            c.load_factor,
            c.n_succeeded,
            c.total_ms,
            c.ops_per_s,
            c.cells_per_slot,
            c.row_width,
            c.segment_rows,
            c.db_rows,
            c.db_cols,
        )
    }
}

impl<B> MutationFlow<B> for RewindClient<B>
where
    B: IndexPirBackend
        + ParallelSetupBackend
        + IncrementalPirBackend
        + PrecomputingPirBackend
        + ResponseRewind,
    B::Query: Clone,
    B::Response: Clone,
{
    type CliExtra = NoExtraArgs;
    type Variant = ();
    const HEADER: &'static str =
        "flow,backend,mutation_kind,arity,num_buckets,bucket_size,value_bits,\
        plaintext_bits,lwe_dim,n_mutations,load_factor,n_succeeded,total_ms,ops_per_s,\
        pending_cells,cells_per_slot,row_width,segment_rows,db_rows,db_cols";

    fn variants(_extra: &NoExtraArgs) -> Vec<()> {
        vec![()]
    }
    fn cli_knobs(_extra: &NoExtraArgs, _matches: &clap::ArgMatches) -> Vec<helpers::Knob> {
        Vec::new()
    }
    fn make_client(bundle: ServerSetupBundle<B>, (): &()) -> Self {
        Self::from_setup_parallel(bundle)
    }
    fn format_row(&self, c: &MutationRowCommon, (): &()) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{:.2},{},{:.3},{:.2},{},{},{},{},{},{}",
            <Self as ClientFlow<B>>::FLOW,
            c.backend,
            c.kind,
            c.arity,
            c.num_buckets,
            c.bucket_size,
            c.value_bits,
            c.plaintext_bits,
            c.lwe_dim,
            c.n_mutations,
            c.load_factor,
            c.n_succeeded,
            c.total_ms,
            c.ops_per_s,
            <Self as ClientFlow<B>>::pending_cells(self),
            c.cells_per_slot,
            c.row_width,
            c.segment_rows,
            c.db_rows,
            c.db_cols,
        )
    }
}

pub fn run_one<S, B, C>(
    cli: &Cli<C::CliExtra>,
    arity: u32,
    num_buckets: u32,
    make_config: impl Fn() -> B::Config,
) where
    S: CloneStore,
    B: IndexPirBackend
        + ParallelSetupBackend
        + IncrementalPirBackend
        + PrecomputingPirBackend
        + BackendWireSize
        + Clone,
    B::Query: Clone,
    B::Response: Clone,
    C: MutationFlow<B>,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli<C::CliExtra>>();
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
    let flow_slug = C::FLOW.replace('-', "_");
    let bench_name = format!("{flow_slug}_mutation");
    let mut knobs = vec![
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
    knobs.extend(C::cli_knobs(&cli.extra, &matches));
    helpers::print_preamble(&bench_name, &knobs, &store_state, &geom);

    let csv_name = format!("ikpir_{flow_slug}_mutation.csv");
    let mut csv = helpers::csv_writer(&csv_name, C::HEADER);

    for &kind in MutationKind::all() {
        let (deltas, n_succeeded) = collect_deltas_for_kind::<S, B>(
            &mut server,
            &bundle.hints,
            &snap,
            cli.value_bits,
            cli.n_mutations,
            kind,
        );
        if deltas.is_empty() {
            eprintln!("  Skip kind={}: no deltas collected", kind.as_csv());
            continue;
        }

        for variant in C::variants(&cli.extra) {
            // Fresh client, no precompute: times only the sync — the
            // per-batch client-maintenance cost in isolation.
            let mut client = C::make_client(bundle.clone(), &variant);

            let replay = deltas.clone();
            let t = Instant::now();
            for d in replay {
                client.sync_delta(d).expect("sync_delta");
            }
            let total_ms = t.elapsed().as_secs_f64() * 1e3;
            let ops_per_s = n_succeeded as f64 / total_ms * 1e3;

            let common = MutationRowCommon {
                backend: cli.backend.to_string(),
                kind: kind.as_csv(),
                arity,
                num_buckets,
                bucket_size: cli.bucket_size,
                value_bits: cli.value_bits,
                plaintext_bits: cli.plaintext_bits,
                lwe_dim: lwe_dim_eff,
                n_mutations: cli.n_mutations,
                load_factor: cli.load_factor,
                n_succeeded,
                total_ms,
                ops_per_s,
                cells_per_slot: cps,
                row_width,
                segment_rows,
                db_rows,
                db_cols,
            };
            let row = client.format_row(&common, &variant);
            writeln!(csv, "{row}").unwrap();
            println!(
                "  flow={} backend={} kind={:<6} arity={arity} nb={num_buckets:<7} N={:<4} | \
                 {:.2} ops/s (total={:.1}ms)",
                C::FLOW,
                cli.backend,
                kind.as_csv(),
                cli.n_mutations,
                ops_per_s,
                total_ms,
            );
        }
    }
    println!("\nResults written to results/{csv_name}");
}
