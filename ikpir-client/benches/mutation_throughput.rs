//! **Intent:** Measure server-side `insert / update / delete` throughput
//! *and* client-side `apply_delta` throughput in a single bench process so
//! the expensive `populate_to_load` and `IkpirServer::new` are paid once
//! per kind instead of once for each side.
//!
//! **Motivation:** Today `server_mutation` and `client_mutation` each
//! independently populate the KV store and construct a fresh
//! `IkpirServer` per mutation kind. At paper-scale configs that's 7
//! expensive `IkpirServer::new` calls per (config, backend) — 3 in the
//! server bench, 4 in the client bench. The hint-matrix preprocessing
//! that fires inside `new` dominates wall-clock cost (30–120 s per
//! call). This bench fuses both measurements: for each kind we
//! construct *one* fresh server, time the mutation loop while
//! collecting the deltas, then build a warm-bc client from a captured
//! setup bundle and time `apply_delta` over the very same deltas. Total
//! expensive setups per config drops 7 → 3.
//!
//! **Method:** Populate to `--load-factor` once and snapshot the cells.
//! For each kind (insert / update / delete):
//!   1. Clone the cells, build a fresh `IkpirServer` (fresh random seed
//!      → fresh `A`, fresh hint `H = Aᵀ·D`).
//!   2. Call `server.setup()` to capture this kind's epoch-0
//!      `ServerSetupBundle` (small — params + hint clones; no `A` on
//!      the wire after the HintMaterial refactor). The bundle is
//!      captured **per kind**, not shared across kinds, so the client
//!      built in step 4 tracks this kind's server's exact `(seed, A, H)`
//!      and the post-`apply_delta` client hint equals the server's
//!      post-mutation hint bit-for-bit.
//!   3. Wall-clock time the N-mutation loop on the server, collecting
//!      each returned `HintDeltaBundle`.
//!   4. Build a fresh warm-bc client from the captured bundle,
//!      `precompute_queries` + `precompute_decodes`.
//!   5. Wall-clock time the N `apply_delta` calls on those collected
//!      deltas.
//!   6. Drop client + server before the next kind so peak live
//!      A-matrix copies stay at the 1-copy budget the memory guard
//!      accounts for. Each `run_server_kind` calls
//!      `server.drop_hint_material()` immediately after `setup()`, so
//!      the first `commit_mutations` re-expands `A` from the seed
//!      (a one-time cost; subsequent mutations reuse the materialised
//!      copy).
//!
//! Wall-clock batch timing is used (not Criterion) for both sides,
//! because each mutation advances the server epoch / mutates state, so
//! Criterion's cycling pattern would either be meaningless or require
//! expensive re-precomputes per sample.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (backend-dependent default), `--n-mutations` (default
//! 1024), `--load-factor` (default 0.90, matches
//! `MUTATION_LOAD_FACTOR` in `scripts/configs.sh`),
//! `--max-mem-gb` (default 12.0; same OOM guard as
//! `classical_throughput`).
//!
//! **Output:** Two CSV files with the same schemas as the individual
//! mutation benches:
//!   `results/ikpir_server_mutation.csv`
//!   `results/ikpir_client_mutation.csv`
//!
//! Use `scripts/run_mutation.sh` to sweep the full mutation config
//! matrix. Do **not** also run the individual `server_mutation` /
//! `client_mutation` scripts for the same configs — the CSV files are
//! shared and would accumulate duplicate rows.

mod helpers;

use helpers::{Backend, CloneStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, HintDeltaBundle, IkpirClient,
    IncrementalPirBackend, IndexPirBackend, PrecomputingPirBackend, SimpleConfig, SimplePirBackend,
};
use ikpir_server::{IkpirError, IkpirServer};
use segmented_cuckoo::{
    CuckooParams, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const HEADER_SERVER: &str =
    "backend,mutation_kind,arity,num_buckets,bucket_size,value_bits,lwe_dim,\
     n_mutations,load_factor,n_attempted,n_succeeded,total_ms,ops_per_s,delta_bytes_total,\
     cells_per_slot,row_width,segment_rows";

const HEADER_CLIENT: &str =
    "backend,mutation_kind,arity,num_buckets,bucket_size,value_bits,lwe_dim,\
     n_mutations,load_factor,n_succeeded,total_ms,ops_per_s,\
     cells_per_slot,row_width,segment_rows";

// Precomputed-query queue size for the warm-bc client.
// Each apply_delta patches c = s^T·H for every slot still in this queue,
// so keep it modest at large configs.
const QUEUE_HEADROOM: u32 = 1 << 10;

#[derive(Clone, Copy, Debug)]
enum MutationKind { Insert, Update, Delete }
impl MutationKind {
    fn as_csv(self) -> &'static str {
        match self { Self::Insert => "insert", Self::Update => "update", Self::Delete => "delete" }
    }
    fn all() -> &'static [Self] { &[Self::Insert, Self::Update, Self::Delete] }
}

#[derive(clap::Parser)]
#[command(about = "Measure server insert/update/delete + client apply_delta sharing one populate.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    /// LWE dimension. Defaults to 1566 (Frodo) or 1275 (Simple) when omitted.
    #[arg(long)]                           lwe_dim: Option<u32>,
    #[arg(long, default_value_t = 1024)]   n_mutations: u32,
    #[arg(long, default_value_t = 0.90)]   load_factor: f64,
    /// Skip configs whose estimated peak memory exceeds this limit. The
    /// dominant term is the LWE public matrix `A` (segment_rows × lwe_dim
    /// × 4 B per segment), held by `B::HintMaterial`. After the
    /// HintMaterial refactor the captured setup bundle no longer carries
    /// `A` and the server drops its copy immediately after `setup()`, so
    /// the peak coexisting `A` count is 1 (only the active client's
    /// copy, since each `run_server_kind` is followed by a client phase
    /// with the server already dropped). Same formula and rationale as
    /// `classical_throughput`. Raise on machines with ≥ 32 GB RAM.
    #[arg(long, default_value_t = 12.0)]   max_mem_gb: f64,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim.unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn fill_value(value: &mut [u8], key: u32, salt: u32) {
    for (i, b) in value.iter_mut().enumerate() {
        *b = (key.wrapping_mul(salt).wrapping_add(i as u32) & 0xFF) as u8;
    }
}

struct KindResult<B: IndexPirBackend> {
    /// Deltas produced by the timed server-side loop, in insertion order.
    deltas:            Vec<HintDeltaBundle<B>>,
    n_succeeded:       u32,
    server_total_ms:   f64,
    delta_bytes_total: usize,
}

/// Run N mutations of `kind` on a fresh server cloned from `cells`,
/// timing only the mutation loop. Returns the collected deltas plus the
/// epoch-0 setup bundle from this kind's own server so the caller can
/// feed them into a matching client-side measurement.
///
/// # Rationale
///
/// The bundle is captured per-kind (not once-then-reused) so the client
/// built from it shares this kind's server's `(seed, A, hint)` triple
/// exactly. apply_delta on the kind's deltas then yields a
/// self-consistent client state — same hint as the server would reach
/// after applying the same mutations. The cost is one extra `server.setup()`
/// per kind (a cheap clone of params/hints, NOT a re-derivation of `A`).
fn run_server_kind<S, B>(
    cli:         &Cli,
    cells:       &[u32],
    params:      CuckooParams,
    n_seed:      u64,
    kind:        MutationKind,
    make_config: &impl Fn() -> B::Config,
) -> (KindResult<B>, ikpir_client::ServerSetupBundle<B>)
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
{
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];

    let store = S::clone_from_cells(cells.to_vec(), params, n_seed).expect("from_cells");
    let mut server: IkpirServer<S, B> = IkpirServer::new(store, make_config());

    // Capture the epoch-0 bundle BEFORE any mutation. This kind's deltas
    // (collected below) will be applied to a client built from this
    // bundle, so the client tracks this kind's server step-for-step.
    let bundle = server.setup();

    // Free the server's seed-derived `A` matrix now. The first
    // `commit_mutations` call below will silently re-expand it from the
    // seed (one-time cost), then subsequent calls reuse the materialised
    // copy. Brings peak live A copies inside this function to 1.
    server.drop_hint_material();

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

    (
        KindResult { deltas, n_succeeded, server_total_ms, delta_bytes_total },
        bundle,
    )
}

fn run_one<S, B>(
    csv_server: &mut std::io::BufWriter<std::fs::File>,
    csv_client: &mut std::io::BufWriter<std::fs::File>,
    cli:         &Cli,
    arity:       u32,
    num_buckets: u32,
    make_config: impl Fn() -> B::Config,
)
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + PrecomputingPirBackend + BackendWireSize + Clone,
    B::Query:    Clone,
    B::Response: Clone,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();
    let lwe_dim_eff = effective_lwe_dim(cli);

    // ── 0. Memory guard ─────────────────────────────────────────────────────────
    // Two coexisting large allocations during `measure_and_write`:
    //
    //   1. The per-segment LWE public matrix `A` (in `B::HintMaterial`).
    //      After the HintMaterial refactor `setup()` no longer ships `A`
    //      and the server drops its copy immediately. Each
    //      `run_server_kind` returns before the client is built, so peak
    //      coexisting `A` copies = 1 (server during the timed mutation
    //      loop, then the warm-bc client during apply_delta). The first
    //      mutation per kind pays a one-time re-expansion cost.
    //
    //   2. The warm-bc prepared-query queue (filled by
    //      `precompute_queries(QUEUE_HEADROOM)` + `precompute_decodes`):
    //      `arity × QUEUE_HEADROOM` slots, each carrying `secret`
    //      (lwe_dim u32) + `b` (a_rows_per_seg u32) + `c` (c_len_per_seg
    //      u32). For paper-scale Frodo (n_rows ≈ 2M) this rivals `A`
    //      itself (~16 GB at QUEUE_HEADROOM=1024).
    //
    // Per-segment shapes depend on the backend (same dispatch as
    // `classical_throughput`):
    //   FrodoPIR:  a_rows_per_seg = n_rows,        c_len_per_seg = row_width
    //   SimplePIR: a_rows_per_seg = reshape_rows,  c_len_per_seg = reshape_row_width
    //
    // The collected `Vec<HintDeltaBundle>` is sparse (~100 MB at paper
    // scale) and sits well below the noise of `A` and the queue.
    let lwe_dim_est        = lwe_dim_eff as u64;
    let cells_per_slot_est = (cli.fingerprint_bits + cli.value_bits).div_ceil(cli.plaintext_bits) as u64;
    let row_width_est      = cli.bucket_size as u64 * cells_per_slot_est;
    let n_rows_per_seg     = num_buckets as u64 / arity as u64;
    let table_bytes        = num_buckets as u64 * cli.bucket_size as u64 * cells_per_slot_est * 4;
    let (a_rows_per_seg, c_len_per_seg) =
        helpers::backend_shape_estimate(cli.backend, n_rows_per_seg, row_width_est);
    let a_bytes_per_copy   = arity as u64 * a_rows_per_seg * lwe_dim_est * 4;
    let queue_bytes        = arity as u64 * QUEUE_HEADROOM as u64
                             * (lwe_dim_est + a_rows_per_seg + c_len_per_seg) * 4;
    let estimated_bytes    = table_bytes + a_bytes_per_copy + queue_bytes;
    let estimated_gb       = estimated_bytes as f64 / 1e9;
    if estimated_gb > cli.max_mem_gb {
        eprintln!(
            "  Skip (OOM guard): estimated peak {:.1} GB > --max-mem-gb {:.1} \
             (nb={num_buckets} bs={} vb={} lwe_dim={lwe_dim_est} backend={}, \
             A={:.2} GB, queue={:.2} GB, table={:.2} GB). \
             Raise --max-mem-gb on machines with more RAM.",
            estimated_gb, cli.max_mem_gb, cli.bucket_size, cli.value_bits, cli.backend,
            a_bytes_per_copy as f64 / 1e9,
            queue_bytes     as f64 / 1e9,
            table_bytes     as f64 / 1e9,
        );
        return;
    }

    // ── 1. Populate once ────────────────────────────────────────────────────────
    let (seed_store, n_seed) = helpers::populate_to_load::<S>(
        cli.load_factor, num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    if n_seed < cli.n_mutations as u64 {
        eprintln!("  Skip: n_seed={n_seed} < n_mutations={}", cli.n_mutations);
        return;
    }

    let cells  = seed_store.snapshot_cells();
    let params = seed_store.params();
    drop(seed_store); // we only need the cells + params from here on

    // ── 2. Per-kind: server mutations + client apply_delta ─────────────────────
    // Each kind builds its own server (fresh seed → fresh A/hint) and
    // captures its own bundle from that server. The client constructed
    // inside `measure_and_write` is therefore matched to *this kind's*
    // server: same A, same starting hint, same epoch trajectory. The
    // resulting apply_delta produces a self-consistent client whose
    // post-mutation hint exactly equals the post-mutation server hint.
    //
    // Preamble (`=== mutation_throughput ===` banner) is printed once
    // on the first kind's bundle; the geometry numbers depend on the
    // backend config, not on which kind is running.
    for (idx, &kind) in MutationKind::all().iter().enumerate() {
        let (result, bundle) =
            run_server_kind::<S, B>(cli, &cells, params, n_seed, kind, &make_config);

        if idx == 0 {
            let setup_bundle_bytes = bundle.wire_byte_size();
            let hint_per_seg_bytes = B::hint_byte_size(&bundle.hints[0]);
            let cps = params.cells_per_slot();
            let store_state = helpers::StoreState {
                capacity:       (num_buckets as u64) * (cli.bucket_size as u64),
                populated:      n_seed,
                load_pct:       100.0 * n_seed as f64 / (num_buckets as f64 * cli.bucket_size as f64),
                cells_per_slot: cps,
                row_width:      cli.bucket_size * cps,
                segment_rows:   params.segment_size(),
            };
            let geom = helpers::Geometry {
                hint_per_seg_bytes,
                setup_bundle_bytes,
                query_bytes: 0,
                response_bytes: 0,
                hint_delta_typical_bytes: None,
            };
            let knobs = [
                helpers::Knob { name: "backend",     value: cli.backend.to_string(),           is_default: matches.value_source("backend") != Some(ValueSource::CommandLine) },
                helpers::Knob { name: "arity",       value: arity.to_string(),                 is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
                helpers::Knob { name: "num_buckets", value: num_buckets.to_string(),           is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
                helpers::Knob { name: "bucket_size", value: cli.bucket_size.to_string(),       is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
                helpers::Knob { name: "value_bits",  value: cli.value_bits.to_string(),        is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
                helpers::Knob { name: "lwe_dim",     value: lwe_dim_eff.to_string(),           is_default: cli.lwe_dim.is_none() },
                helpers::Knob { name: "n_mutations", value: cli.n_mutations.to_string(),       is_default: matches.value_source("n_mutations") != Some(ValueSource::CommandLine) },
                helpers::Knob { name: "load_factor", value: format!("{:.2}", cli.load_factor), is_default: matches.value_source("load_factor") != Some(ValueSource::CommandLine) },
            ];
            helpers::print_preamble("mutation_throughput", &knobs, &store_state, &geom);
        }

        measure_and_write::<B>(
            csv_server, csv_client, cli, arity, num_buckets, lwe_dim_eff,
            kind, result, &bundle,
        );
        // `bundle` drops at end of iteration → its hint/params clones are
        // freed before the next kind's server allocates `A` anew.
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_and_write<B>(
    csv_server: &mut std::io::BufWriter<std::fs::File>,
    csv_client: &mut std::io::BufWriter<std::fs::File>,
    cli:         &Cli,
    arity:       u32,
    num_buckets: u32,
    lwe_dim_eff: u32,
    kind:        MutationKind,
    result:      KindResult<B>,
    bundle:      &ikpir_client::ServerSetupBundle<B>,
)
where
    B: IndexPirBackend + IncrementalPirBackend + PrecomputingPirBackend + BackendWireSize + Clone,
    B::Query:    Clone,
    B::Response: Clone,
{
    let KindResult { deltas, n_succeeded, server_total_ms, delta_bytes_total } = result;

    let server_ops_per_s = if server_total_ms > 0.0 {
        n_succeeded as f64 / server_total_ms * 1e3
    } else { 0.0 };

    // Per-segment geometry derived from the bundle's CuckooParams.
    let cps          = bundle.params.cells_per_slot();
    let row_width    = cli.bucket_size * cps;
    let segment_rows = bundle.params.segment_size();

    // Server-side row.
    writeln!(
        csv_server,
        "{},{},{arity},{num_buckets},{},{},{lwe_dim_eff},{},{:.2},{},{},{:.3},{:.2},{},{cps},{row_width},{segment_rows}",
        cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits,
        cli.n_mutations, cli.load_factor,
        cli.n_mutations, n_succeeded, server_total_ms, server_ops_per_s, delta_bytes_total,
    ).unwrap();

    if deltas.is_empty() {
        eprintln!("  Skip client side kind={}: no deltas collected", kind.as_csv());
        println!(
            "  backend={} kind={:<6} arity={arity} nb={num_buckets:<7} N={:<4} | \
             server: succ={}/{} total={:.1}ms ops/s={:.0} delta={}B | client: SKIPPED",
            cli.backend, kind.as_csv(), cli.n_mutations,
            n_succeeded, cli.n_mutations, server_total_ms, server_ops_per_s, delta_bytes_total,
        );
        return;
    }

    // Build a fresh warm-bc client from the captured bundle.
    let mut client = IkpirClient::<B>::from_setup(bundle.clone());
    client.precompute_queries(QUEUE_HEADROOM);
    client.precompute_decodes();

    let t = Instant::now();
    for d in deltas {
        client.apply_delta(d).expect("apply_delta");
    }
    let client_total_ms = t.elapsed().as_secs_f64() * 1e3;
    let client_ops_per_s = if client_total_ms > 0.0 {
        n_succeeded as f64 / client_total_ms * 1e3
    } else { 0.0 };

    writeln!(
        csv_client,
        "{},{},{arity},{num_buckets},{},{},{lwe_dim_eff},{},{:.2},{},{:.3},{:.2},{cps},{row_width},{segment_rows}",
        cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits,
        cli.n_mutations, cli.load_factor, n_succeeded, client_total_ms, client_ops_per_s,
    ).unwrap();

    println!(
        "  backend={} kind={:<6} arity={arity} nb={num_buckets:<7} N={:<4} | \
         server: succ={}/{} total={:.1}ms ops/s={:.0} delta={}B | \
         client: total={:.1}ms ops/s={:.0}",
        cli.backend, kind.as_csv(), cli.n_mutations,
        n_succeeded, cli.n_mutations, server_total_ms, server_ops_per_s, delta_bytes_total,
        client_total_ms, client_ops_per_s,
    );
}

fn dispatch_backend<S: CloneStore>(
    csv_server: &mut std::io::BufWriter<std::fs::File>,
    csv_client: &mut std::io::BufWriter<std::fs::File>,
    cli:         &Cli,
    arity:       u32,
    num_buckets: u32,
) {
    let lwe_dim = effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo  => run_one::<S, FrodoPirBackend>(
            csv_server, csv_client, cli, arity, num_buckets,
            || FrodoConfig::with_lwe_dim(lwe_dim),
        ),
        Backend::Simple => run_one::<S, SimplePirBackend>(
            csv_server, csv_client, cli, arity, num_buckets,
            || SimpleConfig::with_lwe_dim(lwe_dim),
        ),
    }
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv_server = helpers::csv_writer("ikpir_server_mutation.csv", HEADER_SERVER);
    let mut csv_client = helpers::csv_writer("ikpir_client_mutation.csv", HEADER_CLIENT);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv_server, &mut csv_client, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv_server, &mut csv_client, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv_server, &mut csv_client, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_mutation.csv, ikpir_client_mutation.csv");
}
