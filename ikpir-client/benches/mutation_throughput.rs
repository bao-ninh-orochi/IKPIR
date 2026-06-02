//! **Intent:** Measure server-side `insert / update / delete` throughput
//! *and* client-side `apply_delta` throughput in a single bench process so
//! the expensive `populate_to_load` and `IkpirServer::new` are paid once
//! per (config, backend) and shared across both sides and all three kinds.
//!
//! **Motivation:** Today `server_mutation` and `client_mutation` each
//! independently populate the KV store and construct a fresh
//! `IkpirServer` per mutation kind. At paper-scale configs that's 7
//! expensive `IkpirServer::new` calls per (config, backend) — 3 in the
//! server bench, 4 in the client bench. The hint-matrix preprocessing
//! (`H = Aᵀ·D`) that fires inside `new` dominates wall-clock cost
//! (30–120 s per call). This bench fuses both measurements **and**
//! collapses setup to a single `IkpirServer::new` per (config, backend):
//! all three kinds replay against the same epoch-0 snapshot, so they
//! share one `(seed, A, H)` triple. We build the server once and return
//! it to epoch 0 between kinds via `IkpirServer::reset_for_replay`
//! (which swaps in a fresh snapshot store and restores the epoch-0 hint
//! without recomputing `A` or `H`). Total expensive setups per config
//! drops 7 → 1.
//!
//! **Method:** Populate to `--load-factor` once and snapshot the cells.
//! Build *one* `IkpirServer` (one random seed → one `A`, one hint
//! `H = Aᵀ·D`) and capture its epoch-0 `ServerSetupBundle` once (small —
//! params + hint clones; no `A` on the wire after the HintMaterial
//! refactor). That single bundle is reused for every kind's client: the
//! shared `(seed, A, H)` triple means each kind's post-`apply_delta`
//! client hint equals that kind's post-mutation server hint
//! bit-for-bit. Then in two phases:
//!
//! *Phase 1 (server, one resident `A`):* for each kind (insert / update
//! / delete), return the server to epoch 0 (`reset_for_replay`, skipped
//! for the first kind) and wall-clock time the N-mutation loop,
//! collecting each returned `HintDeltaBundle`. `A` stays resident across
//! the loop (no `drop_hint_material`), so all N mutations reuse the
//! original allocation and no re-expansion bias creeps into
//! `server_total_ms`. Keeping one server alive across kinds holds peak
//! coexisting `A` copies at 1 for the whole phase.
//!
//! *Phase 2 (client, one resident `A` at a time):* drop the server
//! (freeing its `A`) and the snapshot cells, then for each kind build a
//! fresh client from the shared bundle with an **empty prepared-query
//! queue** (no `precompute_queries` / `precompute_decodes`) and
//! wall-clock time the N `apply_delta` calls over that kind's deltas.
//! Each `apply_delta` patches only the hint `H`; the queue-iteration
//! inside `client_patch_state` is a no-op when the queue is empty, so
//! the timing isolates the "compute new hint" cost from warm-bc
//! queue-maintenance overhead. Because the server is dropped before any
//! client is built, peak coexisting `A` copies is 1 here too.
//!
//! Wall-clock batch timing is used (not Criterion) for both sides,
//! because each mutation advances the server epoch / mutates state, so
//! Criterion's cycling pattern is not meaningful.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (backend-dependent default), `--n-mutations` (default
//! 1024), `--load-factor` (default 0.90, matches
//! `MUTATION_LOAD_FACTOR` in `scripts/configs.sh`),
//! `--max-mem-gb` (default 85.0; same OOM guard as
//! `classical_throughput`).
//!
//! **Output:** Two CSV files with the same schemas as the individual
//! mutation benches:
//!   `results/ikpir_server_mutation.csv`
//!   `results/ikpir_client_mutation.csv`
//!
//! Each row carries `db_rows` / `db_cols` reporting the per-segment PIR
//! matrix shape **after** any backend-specific reshape. For FrodoPIR
//! this is `(segment_rows, row_width)`; for SimplePIR this is the
//! post-reshape `(⌈segment_rows/k⌉, k·row_width)`.
//!
//! Use `scripts/run_mutation.sh` to sweep the full mutation config
//! matrix. Do **not** also run the individual `server_mutation` /
//! `client_mutation` scripts for the same configs — the CSV files are
//! shared and would accumulate duplicate rows.

mod helpers;

use helpers::{Backend, CloneStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, HintDeltaBundle, IkpirClient,
    IncrementalPirBackend, IndexPirBackend, SimpleConfig, SimplePirBackend,
};
use ikpir_server::{IkpirError, IkpirServer};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::time::Instant;

const HEADER_SERVER: &str =
    "backend,mutation_kind,arity,num_buckets,bucket_size,value_bits,plaintext_bits,lwe_dim,\
     n_mutations,load_factor,n_attempted,n_succeeded,total_ms,ops_per_s,delta_bytes_total,\
     cells_per_slot,row_width,segment_rows,db_rows,db_cols";

const HEADER_CLIENT: &str =
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

#[derive(clap::Parser)]
#[command(about = "Measure server insert/update/delete + client apply_delta sharing one populate.")]
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
    /// Skip configs whose estimated peak memory exceeds this limit. The
    /// dominant term is the LWE public matrix `A` (segment_rows × lwe_dim
    /// × 4 B per segment), held by `B::HintMaterial`. After the
    /// HintMaterial refactor the captured setup bundle no longer carries
    /// `A`. One server is built per config and kept resident across all
    /// three kinds' mutation loops (Phase 1; no `drop_hint_material` — that
    /// would re-expand `A` on the first mutation and bias the timing),
    /// then dropped before any client is built (Phase 2). The client runs
    /// in empty-queue mode (no `precompute_queries` / `precompute_decodes`),
    /// so no large prepared-query queue coexists with `A`. Peak coexisting
    /// `A` copies is therefore 1 (the single server during all mutation
    /// loops, then one client at a time during apply_delta). Default 85.0
    /// is tuned for a ~96 GB server; lower `--max-mem-gb` on smaller
    /// machines.
    #[arg(long, default_value_t = 85.0)]
    max_mem_gb: f64,
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

struct KindResult<B: IndexPirBackend> {
    /// Deltas produced by the timed server-side loop, in insertion order.
    deltas: Vec<HintDeltaBundle<B>>,
    n_succeeded: u32,
    server_total_ms: f64,
    delta_bytes_total: usize,
}

/// Run N mutations of `kind` on a server already sitting at a fresh
/// epoch-0 snapshot, timing only the mutation loop. Returns the collected
/// deltas plus throughput counters.
///
/// # Rationale
///
/// The caller builds the server **once** per config (paying
/// `IkpirServer::new`'s `H = Aᵀ·D` matmul a single time) and returns it to
/// epoch 0 before each kind via [`IkpirServer::reset_for_replay`]. All
/// three kinds therefore share one `(seed, A, epoch-0 hint)` triple, so the
/// single `ServerSetupBundle` captured by the caller matches every kind's
/// server step-for-step — apply_delta on a kind's deltas yields a
/// self-consistent client whose post-mutation hint equals the server's.
///
/// `A` stays resident throughout the loop (the caller does not
/// `drop_hint_material`), so all N mutations reuse the original allocation
/// and no re-expansion bias creeps into `server_total_ms`.
fn run_kind_loop<S, B>(
    server: &mut IkpirServer<S, B>,
    cli: &Cli,
    n_seed: u64,
    kind: MutationKind,
) -> KindResult<B>
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
{
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];

    let mut deltas: Vec<HintDeltaBundle<B>> = Vec::with_capacity(cli.n_mutations as usize);
    let mut delta_bytes_total = 0usize;
    let mut n_succeeded = 0u32;

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
                deltas.push(bundle);
                n_succeeded += 1;
            }
            Err(IkpirError::TableFull) => break,
            Err(e) => panic!("mutation_throughput kind={}: {e:?}", kind.as_csv()),
        }
    }
    let server_total_ms = t.elapsed().as_secs_f64() * 1e3;
    KindResult {
        deltas,
        n_succeeded,
        server_total_ms,
        delta_bytes_total,
    }
}

fn run_one<S, B>(
    csv_server: &mut std::io::BufWriter<std::fs::File>,
    csv_client: &mut std::io::BufWriter<std::fs::File>,
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

    // ── 0. Memory guard ─────────────────────────────────────────────────────────
    // Dominant allocation is the per-segment LWE public matrix `A` (in
    // `B::HintMaterial`). After the HintMaterial refactor `setup()` no
    // longer ships `A` over the wire. One server is built per config and
    // kept resident across all three kinds' mutation loops (Phase 1; no
    // `drop_hint_material` call — otherwise the first `commit_mutations`
    // would silently re-expand `A` and bias the timing), then dropped
    // before Phase 2 builds any client. So the client built inside
    // `measure_and_write` is the only `A` holder during apply_delta: peak
    // coexisting `A` copies = 1. The client runs in empty-queue mode (no
    // `precompute_queries` / `precompute_decodes`), so no large
    // prepared-query queue coexists with `A`.
    //
    // Per-segment `A` shape depends on the backend (same dispatch as
    // `classical_throughput`):
    //   FrodoPIR:  a_rows_per_seg = n_rows
    //   SimplePIR: a_rows_per_seg = reshape_rows
    //
    // The collected `Vec<HintDeltaBundle>` is sparse (~100 MB at paper
    // scale) and sits well below the noise of `A`.
    let lwe_dim_est = lwe_dim_eff as u64;
    let cells_per_slot_est =
        (cli.fingerprint_bits + cli.value_bits).div_ceil(cli.plaintext_bits) as u64;
    let row_width_est = cli.bucket_size as u64 * cells_per_slot_est;
    let n_rows_per_seg = num_buckets as u64 / arity as u64;
    let table_bytes = num_buckets as u64 * cli.bucket_size as u64 * cells_per_slot_est * 4;
    let (a_rows_per_seg, _c_len_per_seg) =
        helpers::backend_shape_estimate(cli.backend, n_rows_per_seg, row_width_est);
    let a_bytes_per_copy = arity as u64 * a_rows_per_seg * lwe_dim_est * 4;
    let estimated_bytes = table_bytes + a_bytes_per_copy;
    let estimated_gb = estimated_bytes as f64 / 1e9;
    if estimated_gb > cli.max_mem_gb {
        eprintln!(
            "  Skip (OOM guard): estimated peak {:.1} GB > --max-mem-gb {:.1} \
             (nb={num_buckets} bs={} vb={} lwe_dim={lwe_dim_est} backend={}, \
             A={:.2} GB, table={:.2} GB). \
             Raise --max-mem-gb on machines with more RAM.",
            estimated_gb,
            cli.max_mem_gb,
            cli.bucket_size,
            cli.value_bits,
            cli.backend,
            a_bytes_per_copy as f64 / 1e9,
            table_bytes as f64 / 1e9,
        );
        return;
    }

    // ── 1. Populate once ────────────────────────────────────────────────────────
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
    drop(seed_store); // we only need the cells + params from here on

    // ── 2. Build ONE server — the single expensive setup ───────────────────────
    // All three kinds replay against the SAME epoch-0 snapshot, so they
    // share one `(seed, A, hint)` triple. We pay `IkpirServer::new`'s
    // `H = Aᵀ·D` matmul exactly once here; `reset_for_replay` returns the
    // server to this epoch-0 state between kinds without recomputing `A`
    // or `H`. The bundle is captured once and reused for every kind's
    // client — it matches each kind's server step-for-step because the
    // triple is identical across kinds.
    let mut store0 = S::clone_from_cells(cells.clone(), params, n_seed).expect("from_cells");
    // `from_cells` resets `max_kicks` to `MAX_KICKS_DEFAULT` (500); restore the
    // 2_500 budget the populate helper used so the timed insert loop runs with
    // the same cuckoo-eviction headroom as the populate phase. Without this,
    // inserts at near-full load hit `TableFull` ~5× sooner than they would in
    // a consistent deployment, biasing the throughput downward.
    store0.set_max_kicks(2_500);
    let mut server: IkpirServer<S, B> = IkpirServer::new(store0, make_config());
    let bundle = server.setup();
    let hints0 = bundle.hints.clone(); // epoch-0 hints for the per-kind reset

    // Preamble once — geometry depends on the backend config, not the kind.
    {
        let setup_bundle_bytes = bundle.wire_byte_size();
        let hint_per_seg_bytes = B::hint_byte_size(&bundle.hints[0]);
        let cps = params.cells_per_slot();
        let store_state = helpers::StoreState {
            capacity: (num_buckets as u64) * (cli.bucket_size as u64),
            populated: n_seed,
            load_pct: 100.0 * n_seed as f64 / (num_buckets as f64 * cli.bucket_size as f64),
            cells_per_slot: cps,
            row_width: cli.bucket_size * cps,
            segment_rows: params.segment_size(),
        };
        let geom = helpers::Geometry {
            hint_per_seg_bytes,
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
                is_default: matches.value_source("plaintext_bits")
                    != Some(ValueSource::CommandLine),
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
        helpers::print_preamble("mutation_throughput", &knobs, &store_state, &geom);
    }

    // ── 3. Phase 1: server mutation loops (one resident `A`) ───────────────────
    // Run every kind's mutation loop on the single server, resetting to
    // epoch 0 between kinds. Keeping one server alive (rather than building
    // a fresh one per kind) holds peak coexisting `A` copies at 1 for the
    // whole server phase. Deltas are collected per kind for the client
    // phase; with the `--n-mutations` cap they total well under 1 GB.
    let kinds = MutationKind::all();
    let mut results: Vec<KindResult<B>> = Vec::with_capacity(kinds.len());
    for (idx, &kind) in kinds.iter().enumerate() {
        if idx != 0 {
            // Return the server to epoch 0 with a fresh snapshot store and
            // the captured epoch-0 hints — no `A` / `H` recompute.
            let mut store = S::clone_from_cells(cells.clone(), params, n_seed).expect("from_cells");
            store.set_max_kicks(2_500);
            server.reset_for_replay(store, hints0.clone());
        }
        results.push(run_kind_loop::<S, B>(&mut server, cli, n_seed, kind));
    }
    // Free the server (its `A`), the snapshot cells, and the epoch-0 hint
    // clone before the client phase so the client's freshly re-expanded
    // `A` is the only large copy resident during apply_delta.
    drop(server);
    drop(cells);
    drop(hints0);

    // ── 4. Phase 2: client apply_delta (one resident `A` at a time) ────────────
    // Each kind builds a fresh empty-queue client from the shared epoch-0
    // bundle and times apply_delta over that kind's collected deltas. The
    // server-side CSV row (timed in Phase 1) is also written here via
    // `measure_and_write`, alongside the client-side row.
    for (&kind, result) in kinds.iter().zip(results) {
        measure_and_write::<B>(
            csv_server,
            csv_client,
            cli,
            arity,
            num_buckets,
            lwe_dim_eff,
            kind,
            result,
            &bundle,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_and_write<B>(
    csv_server: &mut std::io::BufWriter<std::fs::File>,
    csv_client: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    lwe_dim_eff: u32,
    kind: MutationKind,
    result: KindResult<B>,
    bundle: &ikpir_client::ServerSetupBundle<B>,
) where
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    let KindResult {
        deltas,
        n_succeeded,
        server_total_ms,
        delta_bytes_total,
    } = result;

    let server_ops_per_s = if server_total_ms > 0.0 {
        n_succeeded as f64 / server_total_ms * 1e3
    } else {
        0.0
    };

    // Per-segment geometry derived from the bundle's CuckooParams.
    let cps = bundle.params.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let segment_rows = bundle.params.segment_size();
    let (db_rows, db_cols) =
        helpers::backend_shape_estimate(cli.backend, segment_rows as u64, row_width as u64);

    // Server-side row.
    writeln!(
        csv_server,
        "{},{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{},{:.2},{},{},{:.3},{:.2},{},{cps},{row_width},{segment_rows},{db_rows},{db_cols}",
        cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits, cli.plaintext_bits,
        cli.n_mutations, cli.load_factor,
        cli.n_mutations, n_succeeded, server_total_ms, server_ops_per_s, delta_bytes_total,
    ).unwrap();

    if deltas.is_empty() {
        eprintln!(
            "  Skip client side kind={}: no deltas collected",
            kind.as_csv()
        );
        println!(
            "  backend={} kind={:<6} arity={arity} nb={num_buckets:<7} N={:<4} | \
             server: succ={}/{} total={:.1}ms ops/s={:.0} delta={}B | client: SKIPPED",
            cli.backend,
            kind.as_csv(),
            cli.n_mutations,
            n_succeeded,
            cli.n_mutations,
            server_total_ms,
            server_ops_per_s,
            delta_bytes_total,
        );
        return;
    }

    // Build a fresh client from the captured bundle, with an empty
    // prepared-query queue (no precompute_queries / precompute_decodes).
    // Each apply_delta then patches only the hint H; the queue iteration
    // inside `client_patch_state` is a no-op when the queue is empty, so
    // the timing reflects the "compute new hint" cost in isolation.
    let mut client = IkpirClient::<B>::from_setup(bundle.clone());

    let t = Instant::now();
    for d in deltas {
        client.apply_delta(d).expect("apply_delta");
    }
    let client_total_ms = t.elapsed().as_secs_f64() * 1e3;
    let client_ops_per_s = if client_total_ms > 0.0 {
        n_succeeded as f64 / client_total_ms * 1e3
    } else {
        0.0
    };

    writeln!(
        csv_client,
        "{},{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{},{:.2},{},{:.3},{:.2},{cps},{row_width},{segment_rows},{db_rows},{db_cols}",
        cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits, cli.plaintext_bits,
        cli.n_mutations, cli.load_factor, n_succeeded, client_total_ms, client_ops_per_s,
    ).unwrap();

    println!(
        "  backend={} kind={:<6} arity={arity} nb={num_buckets:<7} N={:<4} | \
         server: succ={}/{} total={:.1}ms ops/s={:.0} delta={}B | \
         client: total={:.1}ms ops/s={:.0}",
        cli.backend,
        kind.as_csv(),
        cli.n_mutations,
        n_succeeded,
        cli.n_mutations,
        server_total_ms,
        server_ops_per_s,
        delta_bytes_total,
        client_total_ms,
        client_ops_per_s,
    );
}

fn dispatch_backend<S: CloneStore>(
    csv_server: &mut std::io::BufWriter<std::fs::File>,
    csv_client: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    let lwe_dim = effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo => {
            run_one::<S, FrodoPirBackend>(csv_server, csv_client, cli, arity, num_buckets, || {
                FrodoConfig::with_lwe_dim(lwe_dim)
            })
        }
        Backend::Simple => {
            run_one::<S, SimplePirBackend>(csv_server, csv_client, cli, arity, num_buckets, || {
                SimpleConfig::with_lwe_dim(lwe_dim)
            })
        }
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

    let mut csv_server = helpers::csv_writer("ikpir_server_mutation.csv", HEADER_SERVER);
    let mut csv_client = helpers::csv_writer("ikpir_client_mutation.csv", HEADER_CLIENT);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(
            &mut csv_server,
            &mut csv_client,
            &cli,
            2,
            num_buckets,
        ),
        3 => dispatch_backend::<Segmented3aryScheme>(
            &mut csv_server,
            &mut csv_client,
            &cli,
            3,
            num_buckets,
        ),
        4 => dispatch_backend::<Segmented4aryScheme>(
            &mut csv_server,
            &mut csv_client,
            &cli,
            4,
            num_buckets,
        ),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_mutation.csv, ikpir_client_mutation.csv");
}
