//! **Intent:** Measure `server_answer`, `client_query`, and `client_decode`
//! throughput in a single bench process so the expensive `populate_until_full`
//! and `IkpirServer::new` setup is paid once per config, not three times.
//!
//! **Motivation:** Each individual bench (`server_answer`, `client_query`,
//! `client_decode`) independently runs `populate_until_full` + `IkpirServer::new`
//! before its criterion loop. At paper-scale configs (4 M+ slots, lwe_dim ≥ 1024)
//! those two steps can each take 30–120 s, making them the dominant cost of a
//! full sweep. This bench merges all three into one process so fill + hint-matrix
//! computation is shared, cutting per-config overhead from 3× to 1×.
//!
//! **Method:** Populate to `TableFull`, build `IkpirServer`, and call
//! `server.setup()` once. Then run three successive criterion benchmarks:
//!   1. `server_answer` — cycle through `batch` pre-built queries.
//!   2. `client_query` (warm-bc) — `precompute_queries` refilled per criterion
//!      sample so the timed call always pops from a warm queue.
//!   3. `client_decode` (warm-bc) — `precompute_queries` + `precompute_decodes`
//!      refilled per sample; `server.answer` runs outside the timing bracket.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple, default
//! frodo), `--num-buckets`, `--bucket-size`, `--value-bits`, `--lwe-dim`
//! (defaults to backend recommendation), `--batch` (shared key-pool / query-pool
//! size; default 64).
//!
//! **Output:** Three CSV files with the same schema as the individual benches:
//!   `results/ikpir_server_answer.csv`
//!   `results/ikpir_client_query.csv`
//!   `results/ikpir_client_decode.csv`
//!
//! Each row carries `db_rows` / `db_cols` reporting the per-segment PIR matrix
//! shape **after** any backend-specific reshape. For FrodoPIR this is
//! `(segment_rows, row_width)`; for SimplePIR this is the post-reshape
//! `(⌈segment_rows/k⌉, k·row_width)`.
//!
//! Use `scripts/run_classical.sh` to sweep the full config matrix. Do **not**
//! also run the individual `server_answer` / `client_query` / `client_decode`
//! scripts for the same configs — the CSV files are shared and would accumulate
//! duplicate rows.

mod helpers;

use criterion::{Criterion, Throughput};
use helpers::{Backend, MakeStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirClient, IncrementalPirBackend,
    IndexPirBackend, PrecomputingPirBackend, PirQueryBundle, SimpleConfig, SimplePirBackend,
};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::mem;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const HEADER_ANSWER: &str =
    "backend,arity,num_buckets,bucket_size,value_bits,plaintext_bits,lwe_dim,batch,\
    mean_qps,min_qps,max_qps,stddev_qps,query_bytes,response_bytes,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

const HEADER_QUERY: &str =
    "backend,arity,num_buckets,bucket_size,value_bits,plaintext_bits,lwe_dim,batch,\
    mean_qps,min_qps,max_qps,stddev_qps,query_bytes,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

const HEADER_DECODE: &str =
    "backend,arity,num_buckets,bucket_size,value_bits,plaintext_bits,lwe_dim,batch,\
    mean_dps,min_dps,max_dps,stddev_dps,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

#[derive(clap::Parser)]
#[command(about = "Measure server_answer + client_query + client_decode sharing one fill + setup.")]
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
    /// Shared key-pool / query-pool size for all three measurements.
    #[arg(long, default_value_t = 64)]     batch: u32,
    /// Skip configs whose estimated peak memory exceeds this limit. The
    /// dominant term is the FrodoPIR A matrix (segment_rows × lwe_dim × 4 B
    /// per segment), held simultaneously by the server and by each client state
    /// created from the setup bundle. Protects against SIGKILL from the OS OOM
    /// killer. Default 85.0 is tuned for a ~96 GB server; lower
    /// `--max-mem-gb` on smaller machines.
    #[arg(long, default_value_t = 85.0)]   max_mem_gb: f64,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim.unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn ops_per_sec_stats(raw_ns: Vec<f64>) -> helpers::CriterionThroughputStats {
    if raw_ns.is_empty() {
        return helpers::CriterionThroughputStats {
            mean_ops_per_s: 0.0, min_ops_per_s: 0.0,
            max_ops_per_s:  0.0, stddev_ops_per_s: 0.0,
        };
    }
    let ops: Vec<f64> = raw_ns.iter().map(|&ns| 1e9 / ns).collect();
    let s = helpers::compute_stats(&ops);
    helpers::CriterionThroughputStats {
        mean_ops_per_s: s.mean, min_ops_per_s: s.min,
        max_ops_per_s:  s.max,  stddev_ops_per_s: s.stddev,
    }
}

fn run_one<S, B>(
    csv_answer: &mut std::io::BufWriter<std::fs::File>,
    csv_query:  &mut std::io::BufWriter<std::fs::File>,
    csv_decode: &mut std::io::BufWriter<std::fs::File>,
    cli:            &Cli,
    arity:          u32,
    num_buckets:    u32,
    backend_config: B::Config,
)
where
    S: MakeStore,
    B: IndexPirBackend + IncrementalPirBackend + PrecomputingPirBackend + BackendWireSize + Clone,
    B::Query:    Clone,
    B::Response: Clone,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    // ── 0. Memory guard ─────────────────────────────────────────────────────────
    // Dominant cost is the per-segment LWE public matrix `A`, held in
    // `B::HintMaterial`. After the HintMaterial refactor `setup()` no
    // longer ships `A` over the wire and the server drops its copy
    // immediately after setup; peak coexisting copies = 1 (only the
    // active client). Each `IkpirClient::from_setup` re-expands `A`
    // from the seed.
    //
    // Per-segment `A` shape is backend-aware (see
    // `helpers::backend_shape_estimate`):
    //   FrodoPIR:  (n_rows,       lwe_dim)   n_rows = num_buckets / arity
    //   SimplePIR: (reshape_rows, lwe_dim)   reshape_rows ≈ √(n_rows × row_width)
    //                                        via the √N reshape (~100× smaller).
    //
    // The pre-built `queries` Vec is held across all three measurements
    // (sections 3–5):
    //   batch × arity × b_len × 4 bytes, where b_len = a_rows_per_seg.
    // For paper-scale Frodo + batch=64 this is ~1 GB; for SimplePIR it
    // is negligible.
    //
    // Other secondary terms — server/client hint copies, criterion-driven
    // refill queues (bounded by `iters`, typically << 1024 per sample),
    // per-bundle wire payloads — are dwarfed by `A` and the queries Vec
    // and not modelled here.
    //
    // Exceeding available RAM causes a silent SIGKILL from the OS — fail
    // fast with a clear message instead.
    let lwe_dim_est        = effective_lwe_dim(cli) as u64;
    let cells_per_slot_est = (cli.fingerprint_bits + cli.value_bits).div_ceil(cli.plaintext_bits) as u64;
    let row_width_est      = cli.bucket_size as u64 * cells_per_slot_est;
    let n_rows_per_seg     = num_buckets as u64 / arity as u64;
    let table_bytes        = num_buckets as u64 * cli.bucket_size as u64 * cells_per_slot_est * 4;
    let (a_rows_per_seg, _c_len_per_seg) =
        helpers::backend_shape_estimate(cli.backend, n_rows_per_seg, row_width_est);
    let a_bytes_per_copy   = arity as u64 * a_rows_per_seg * lwe_dim_est * 4;
    let queries_bytes      = cli.batch as u64 * arity as u64 * a_rows_per_seg * 4;
    let estimated_bytes    = table_bytes + a_bytes_per_copy + queries_bytes;
    let estimated_gb       = estimated_bytes as f64 / 1e9;
    if estimated_gb > cli.max_mem_gb {
        eprintln!(
            "  Skip (OOM guard): estimated peak {:.1} GB > --max-mem-gb {:.1} \
             (nb={num_buckets} bs={} vb={} lwe_dim={lwe_dim_est} backend={}, \
             A={:.2} GB, queries={:.2} GB, table={:.2} GB). \
             Raise --max-mem-gb on machines with more RAM.",
            estimated_gb, cli.max_mem_gb, cli.bucket_size, cli.value_bits, cli.backend,
            a_bytes_per_copy as f64 / 1e9,
            queries_bytes  as f64 / 1e9,
            table_bytes    as f64 / 1e9,
        );
        return;
    }

    // ── 1. Populate once ────────────────────────────────────────────────────────
    let (store, n_inserted) = helpers::populate_until_full::<S>(
        num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    if n_inserted == 0 { eprintln!("  Skip: empty store"); return; }

    // ── 2. Build server + hint matrix once ─────────────────────────────────────
    let mut server: IkpirServer<S, B> = IkpirServer::new(store, backend_config);
    let bundle = server.setup();
    // Read-only bench: free the server's seed-derived `A` matrix immediately.
    // `server.answer` doesn't touch `HintMaterial`, so no re-expansion is
    // triggered for the rest of the bench. This brings peak coexisting A
    // copies down to 1 (only the active client's copy).
    server.drop_hint_material();

    let params_store = server.params();
    let cps          = params_store.cells_per_slot();
    let row_width    = cli.bucket_size * cps;
    let segment_rows = params_store.segment_size();
    let (db_rows, db_cols) =
        helpers::backend_shape_estimate(cli.backend, segment_rows as u64, row_width as u64);
    let load_factor  = n_inserted as f64 / (num_buckets as f64 * cli.bucket_size as f64);
    let lwe_dim_eff  = effective_lwe_dim(cli);
    let n            = n_inserted as u32;
    let batch        = cli.batch;

    let store_state = helpers::StoreState {
        capacity:       (num_buckets as u64) * (cli.bucket_size as u64),
        populated:      n_inserted,
        load_pct:       100.0 * load_factor,
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };

    // Key pools — same keys reused across all three measurements.
    let keys4: Vec<[u8; 4]>  = (0..batch).map(|i| (i % n).to_le_bytes()).collect();
    let keys_v: Vec<Vec<u8>> = (0..batch).map(|i| (i % n).to_le_bytes().to_vec()).collect();

    // ── 3. Measure server_answer ────────────────────────────────────────────────
    // Build queries from a temporary client then drop it immediately — its A copy
    // is no longer needed once the query vectors are materialised.
    let (queries, query_bytes, response_bytes) = {
        let mut qclient = IkpirClient::<B>::from_setup(bundle.clone());
        let qs: Vec<PirQueryBundle<B>> = (0..batch)
            .map(|i| qclient.build_query(&(i % n).to_le_bytes()))
            .collect();
        let qb = qs[0].wire_byte_size();
        let rb = server.answer(&qs[0]).expect("answer ok").wire_byte_size();
        (qs, qb, rb)
        // qclient dropped here — frees one A copy
    };

    let mut ans_idx = 0usize;
    let crit_answer = helpers::run_criterion_throughput("server_answer", 1, || {
        let _ = server.answer(&queries[ans_idx]).expect("answer ok");
        ans_idx = (ans_idx + 1) % queries.len();
    });

    // ── 4. Measure client_query (warm-bc) ───────────────────────────────────────
    // client_q is scoped so its A copy is freed before client_d is created.
    let crit_query = {
    let mut client_q = IkpirClient::<B>::from_setup(bundle.clone());
    let samples_q: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let sq_inner = samples_q.clone();
    let mut q_idx = 0usize;
    {
        let mut c = Criterion::default();
        let mut group = c.benchmark_group("client_query");
        group.throughput(Throughput::Elements(1));
        group.bench_function("client_query", |b| {
            b.iter_custom(|iters| {
                client_q.precompute_queries(iters as u32);
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let k = keys4[q_idx % keys4.len()];
                    q_idx = q_idx.wrapping_add(1);
                    let t = Instant::now();
                    let _ = client_q.build_query(&k);
                    total += t.elapsed();
                }
                sq_inner.lock().unwrap().push(total.as_nanos() as f64 / iters as f64);
                total
            });
        });
        group.finish();
    }
    let crit_q = ops_per_sec_stats(mem::take(&mut *samples_q.lock().unwrap()));
    // client_q dropped here — frees one A copy before client_d is created
    crit_q
    };

    // ── 5. Measure client_decode (warm-bc) ──────────────────────────────────────
    // Extract wire sizes before moving bundle — cannot borrow after move.
    let hint_per_seg_bytes = B::hint_byte_size(&bundle.hints[0]);
    let setup_bundle_bytes = bundle.wire_byte_size();
    // Move bundle (not clone) into the last client. The bundle no longer
    // carries `A` after the HintMaterial refactor, so this is mostly
    // ergonomic; the client still re-expands its own `A` from the seed
    // during `from_setup`.
    let mut client_d = IkpirClient::<B>::from_setup(bundle);

    // Once-per-config decode sanity check: run one untimed query/answer/decode
    // for a known key and verify the recovered bytes equal the value the
    // populate helper wrote (`fill_value(k, salt=17)`). Catches packing,
    // cells_per_slot, and hint-mismatch regressions that would otherwise
    // produce silent garbage decodes with valid `Ok(Some(_))` shape.
    helpers::verify_decode::<B, _>(&mut client_d, &mut server, 1u32, cli.value_bits);

    let samples_d: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let sd_inner = samples_d.clone();
    let mut d_idx = 0usize;
    {
        let mut c = Criterion::default();
        let mut group = c.benchmark_group("client_decode");
        group.throughput(Throughput::Elements(1));
        group.bench_function("client_decode", |b| {
            b.iter_custom(|iters| {
                client_d.precompute_queries(iters as u32);
                client_d.precompute_decodes();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let k = keys_v[d_idx % keys_v.len()].clone();
                    d_idx = d_idx.wrapping_add(1);
                    let q = client_d.build_query(&k);
                    let r = server.answer(&q).expect("answer ok");
                    let t = Instant::now();
                    let _ = client_d.decode(&k, &r).expect("decode ok");
                    total += t.elapsed();
                }
                sd_inner.lock().unwrap().push(total.as_nanos() as f64 / iters as f64);
                total
            });
        });
        group.finish();
    }
    let crit_decode = ops_per_sec_stats(mem::take(&mut *samples_d.lock().unwrap()));

    // ── 6. Preamble ─────────────────────────────────────────────────────────────
    let geom = helpers::Geometry {
        hint_per_seg_bytes,
        setup_bundle_bytes,
        query_bytes,
        response_bytes,
        hint_delta_typical_bytes: None,
    };
    let nb_is_default = matches.value_source("num_buckets") != Some(ValueSource::CommandLine);
    let knobs = [
        helpers::Knob { name: "backend",          value: cli.backend.to_string(),          is_default: matches.value_source("backend") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "arity",            value: arity.to_string(),                is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",      value: num_buckets.to_string(),          is_default: nb_is_default },
        helpers::Knob { name: "bucket_size",      value: cli.bucket_size.to_string(),      is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "fingerprint_bits", value: cli.fingerprint_bits.to_string(), is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",       value: cli.value_bits.to_string(),       is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "plaintext_bits",   value: cli.plaintext_bits.to_string(),   is_default: matches.value_source("plaintext_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",          value: lwe_dim_eff.to_string(),          is_default: cli.lwe_dim.is_none() },
        helpers::Knob { name: "batch",            value: batch.to_string(),                is_default: matches.value_source("batch") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("classical_throughput", &knobs, &store_state, &geom);
    println!(
        "  answer={:.2} qps (±{:.2}) | query={:.2} qps (±{:.2}) | decode={:.2} dps (±{:.2})",
        crit_answer.mean_ops_per_s, crit_answer.stddev_ops_per_s,
        crit_query.mean_ops_per_s,  crit_query.stddev_ops_per_s,
        crit_decode.mean_ops_per_s, crit_decode.stddev_ops_per_s,
    );

    // ── 7. Write CSVs ───────────────────────────────────────────────────────────
    writeln!(
        csv_answer,
        "{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{batch},{:.2},{:.2},{:.2},{:.2},{query_bytes},{response_bytes},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits,
        crit_answer.mean_ops_per_s, crit_answer.min_ops_per_s,
        crit_answer.max_ops_per_s,  crit_answer.stddev_ops_per_s,
        load_factor,
    ).unwrap();
    writeln!(
        csv_query,
        "{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{batch},{:.2},{:.2},{:.2},{:.2},{query_bytes},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits,
        crit_query.mean_ops_per_s, crit_query.min_ops_per_s,
        crit_query.max_ops_per_s,  crit_query.stddev_ops_per_s,
        load_factor,
    ).unwrap();
    writeln!(
        csv_decode,
        "{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{batch},{:.2},{:.2},{:.2},{:.2},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits,
        crit_decode.mean_ops_per_s, crit_decode.min_ops_per_s,
        crit_decode.max_ops_per_s,  crit_decode.stddev_ops_per_s,
        load_factor,
    ).unwrap();
}

fn dispatch_backend<S: MakeStore>(
    csv_answer: &mut std::io::BufWriter<std::fs::File>,
    csv_query:  &mut std::io::BufWriter<std::fs::File>,
    csv_decode: &mut std::io::BufWriter<std::fs::File>,
    cli:         &Cli,
    arity:       u32,
    num_buckets: u32,
) {
    let lwe_dim = effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo  => run_one::<S, FrodoPirBackend>(
            csv_answer, csv_query, csv_decode, cli, arity, num_buckets,
            FrodoConfig::with_lwe_dim(lwe_dim),
        ),
        Backend::Simple => run_one::<S, SimplePirBackend>(
            csv_answer, csv_query, csv_decode, cli, arity, num_buckets,
            SimpleConfig::with_lwe_dim(lwe_dim),
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

    let mut csv_answer = helpers::csv_writer("ikpir_server_answer.csv", HEADER_ANSWER);
    let mut csv_query  = helpers::csv_writer("ikpir_client_query.csv",  HEADER_QUERY);
    let mut csv_decode = helpers::csv_writer("ikpir_client_decode.csv", HEADER_DECODE);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv_answer, &mut csv_query, &mut csv_decode, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv_answer, &mut csv_query, &mut csv_decode, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv_answer, &mut csv_query, &mut csv_decode, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_answer.csv, ikpir_client_query.csv, ikpir_client_decode.csv");
}
