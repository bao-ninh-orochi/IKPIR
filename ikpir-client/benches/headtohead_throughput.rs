//! **Intent:** Measure `server_answer`, `client_query`, and `client_decode`
//! throughput at a **fixed keyword count** (1 M / 1.5 M / 3 M / 4 M), in
//! support of head-to-head comparisons with ChalametPIR and Hao et al. 2025.
//!
//! **Motivation:** The `classical_throughput` bench fixes `num_buckets` (and
//! thus DB size) and populates `until_full`, so different schemes end up
//! storing different keyword counts in the same-size DB — a setting where
//! IKPIR's slot/cell layout wins by construction. For a fair head-to-head, we
//! flip the protocol: fix `num_keys` and let each scheme report the DB size
//! it needed to absorb that load. The expansion rate `db_size /
//! num_keys` (or in bytes, `num_keys / db_size` ↔ load factor) can then
//! be computed directly from the CSV.
//!
//! **Method:** Identical fill+setup+three-measurement pipeline as
//! `classical_throughput`, except `populate_exact_n_keys(target_n=num_keys)`
//! replaces `populate_until_full`. All three measurements (`server_answer`,
//! `client_query`, `client_decode`) share one fill+setup per config.
//!
//! **Arguments (CLI):** Same as `classical_throughput`, plus `--num-keys`
//! (required, the keyword count to populate).
//!
//! **Output:** Three CSV files with the same shape as
//! `classical_throughput`'s output **plus** two extra columns,
//! `num_keys` and `db_size` (= `num_buckets × bucket_size`):
//!   `results/ikpir_headtohead_server_answer.csv`
//!   `results/ikpir_headtohead_client_query.csv`
//!   `results/ikpir_headtohead_client_decode.csv`
//!
//! Use `scripts/run_headtohead.sh` to sweep the head-to-head matrix
//! defined in `scripts/configs.sh::HEADTOHEAD_CONFIGS`.

// significant_drop_tightening: clippy's inline-`Criterion` fix borrows a temporary dropped while `BenchmarkGroup` holds it (won't compile).
#![allow(clippy::significant_drop_tightening)]

mod helpers;

use criterion::{Criterion, Throughput};
use helpers::{Backend, MakeStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirClient, IncrementalPirBackend,
    IndexPirBackend, PirQueryBundle, PrecomputingPirBackend, SimpleConfig, SimplePirBackend,
};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::mem;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const HEADER_ANSWER: &str =
    "backend,arity,num_buckets,bucket_size,num_keys,db_size,value_bits,plaintext_bits,fingerprint_bits,lwe_dim,batch,\
    mean_qps,min_qps,max_qps,stddev_qps,query_bytes,response_bytes,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

const HEADER_QUERY: &str =
    "backend,arity,num_buckets,bucket_size,num_keys,db_size,value_bits,plaintext_bits,fingerprint_bits,lwe_dim,batch,\
    mean_qps,min_qps,max_qps,stddev_qps,query_bytes,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

const HEADER_DECODE: &str =
    "backend,arity,num_buckets,bucket_size,num_keys,db_size,value_bits,plaintext_bits,fingerprint_bits,lwe_dim,batch,\
    mean_dps,min_dps,max_dps,stddev_dps,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

#[derive(clap::Parser)]
#[command(
    about = "Head-to-head bench: fixes num_keys, reports DB size, otherwise mirrors classical_throughput."
)]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    /// Required: number of keys to populate. The DB size (= `num_buckets ×
    /// bucket_size`) is fixed by `--num-buckets` / `--bucket-size`; the
    /// caller is responsible for picking those so capacity ≥ `num_keys`
    /// at a reasonable load factor (~95% for the head-to-head matrix).
    #[arg(long)]
    num_keys: u64,
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
    /// Shared key-pool / query-pool size for all three measurements.
    #[arg(long, default_value_t = 64)]
    batch: u32,
    /// Skip configs whose estimated peak memory exceeds this limit. See
    /// `classical_throughput.rs` for the dominant terms; this guard mirrors it.
    /// Default 85.0 is tuned for a ~96 GB server; lower
    /// `--max-mem-gb` on smaller machines.
    #[arg(long, default_value_t = 85.0)]
    max_mem_gb: f64,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim
        .unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn ops_per_sec_stats(raw_ns: Vec<f64>) -> helpers::CriterionThroughputStats {
    if raw_ns.is_empty() {
        return helpers::CriterionThroughputStats {
            mean_ops_per_s: 0.0,
            min_ops_per_s: 0.0,
            max_ops_per_s: 0.0,
            stddev_ops_per_s: 0.0,
        };
    }
    let ops: Vec<f64> = raw_ns.iter().map(|&ns| 1e9 / ns).collect();
    let s = helpers::compute_stats(&ops);
    helpers::CriterionThroughputStats {
        mean_ops_per_s: s.mean,
        min_ops_per_s: s.min,
        max_ops_per_s: s.max,
        stddev_ops_per_s: s.stddev,
    }
}

fn run_one<S, B>(
    csv_answer: &mut std::io::BufWriter<std::fs::File>,
    csv_query: &mut std::io::BufWriter<std::fs::File>,
    csv_decode: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    backend_config: B::Config,
) where
    S: MakeStore,
    B: IndexPirBackend + IncrementalPirBackend + PrecomputingPirBackend + BackendWireSize + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    // ── 0. Memory guard ─────────────────────────────────────────────────────────
    // Same estimator as classical_throughput. The dominant term (per-segment
    // LWE matrix `A`, held in B::HintMaterial) is independent of how many
    // keys we end up inserting, so the formula is unchanged.
    let lwe_dim_est = effective_lwe_dim(cli) as u64;
    let cells_per_slot_est =
        (cli.fingerprint_bits + cli.value_bits).div_ceil(cli.plaintext_bits) as u64;
    let row_width_est = cli.bucket_size as u64 * cells_per_slot_est;
    let n_rows_per_seg = num_buckets as u64 / arity as u64;
    let table_bytes = num_buckets as u64 * cli.bucket_size as u64 * cells_per_slot_est * 4;
    let (a_rows_per_seg, _c_len_per_seg) =
        helpers::backend_shape_estimate(cli.backend, n_rows_per_seg, row_width_est);
    let a_bytes_per_copy = arity as u64 * a_rows_per_seg * lwe_dim_est * 4;
    let queries_bytes = cli.batch as u64 * arity as u64 * a_rows_per_seg * 4;
    let estimated_bytes = table_bytes + a_bytes_per_copy + queries_bytes;
    let estimated_gb = estimated_bytes as f64 / 1e9;
    if estimated_gb > cli.max_mem_gb {
        eprintln!(
            "  Skip (OOM guard): estimated peak {:.1} GB > --max-mem-gb {:.1} \
             (nb={num_buckets} bs={} vb={} lwe_dim={lwe_dim_est} backend={}, \
             A={:.2} GB, queries={:.2} GB, table={:.2} GB). \
             Raise --max-mem-gb on machines with more RAM.",
            estimated_gb,
            cli.max_mem_gb,
            cli.bucket_size,
            cli.value_bits,
            cli.backend,
            a_bytes_per_copy as f64 / 1e9,
            queries_bytes as f64 / 1e9,
            table_bytes as f64 / 1e9,
        );
        return;
    }

    // ── 1. Populate exactly num_keys ────────────────────────────────────────────
    // Fixed-N populate; panics if capacity is insufficient or cuckoo eviction
    // exhausts before the target is reached. The orchestrator picks
    // num_buckets / bucket_size to keep load at ~95%.
    let (store, n_inserted) = helpers::populate_exact_n_keys::<S>(
        num_buckets,
        cli.bucket_size,
        cli.fingerprint_bits,
        cli.value_bits,
        cli.plaintext_bits,
        cli.num_keys,
    );
    assert_eq!(
        n_inserted, cli.num_keys,
        "populate_exact_n_keys must return target_n"
    );

    // ── 2. Build server + hint matrix once ─────────────────────────────────────
    let mut server: IkpirServer<S, B> = IkpirServer::new(store, backend_config);
    let bundle = server.setup();
    // Read-only bench: drop the server's seed-derived `A` immediately.
    server.drop_hint_material();

    let params_store = server.params();
    let cps = params_store.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let segment_rows = params_store.segment_size();
    let (db_rows, db_cols) =
        helpers::backend_shape_estimate(cli.backend, segment_rows as u64, row_width as u64);
    let db_size = (num_buckets as u64) * (cli.bucket_size as u64);
    let load_factor = n_inserted as f64 / db_size as f64;
    let lwe_dim_eff = effective_lwe_dim(cli);
    let n = n_inserted as u32;
    let batch = cli.batch;

    let store_state = helpers::StoreState {
        capacity: db_size,
        populated: n_inserted,
        load_pct: 100.0 * load_factor,
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };

    // Key pools — same keys reused across all three measurements.
    let keys4: Vec<[u8; 4]> = (0..batch).map(|i| (i % n).to_le_bytes()).collect();
    let keys_v: Vec<Vec<u8>> = (0..batch).map(|i| (i % n).to_le_bytes().to_vec()).collect();

    // ── 3. Measure server_answer ────────────────────────────────────────────────
    let (queries, query_bytes, response_bytes) = {
        let mut qclient = IkpirClient::<B>::from_setup(bundle.clone());
        let qs: Vec<PirQueryBundle<B>> = (0..batch)
            .map(|i| qclient.build_query(&(i % n).to_le_bytes()))
            .collect();
        let qb = qs[0].wire_byte_size();
        let rb = server.answer(&qs[0]).expect("answer ok").wire_byte_size();
        (qs, qb, rb)
    };

    let mut ans_idx = 0usize;
    let crit_answer = helpers::run_criterion_throughput("server_answer", 1, || {
        let _ = server.answer(&queries[ans_idx]).expect("answer ok");
        ans_idx = (ans_idx + 1) % queries.len();
    });

    // ── 4. Measure client_query (warm-bc) ───────────────────────────────────────
    let crit_query = {
        let mut client_q = IkpirClient::<B>::from_setup(bundle.clone());
        let samples_q: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
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
                    samples_q
                        .lock()
                        .unwrap()
                        .push(total.as_nanos() as f64 / iters as f64);
                    total
                });
            });
            group.finish();
        }
        let crit_q = ops_per_sec_stats(mem::take(&mut *samples_q.lock().unwrap()));
        crit_q
    };

    // ── 5. Measure client_decode (warm-bc) ──────────────────────────────────────
    let hint_per_seg_bytes = B::hint_byte_size(&bundle.hints[0]);
    let setup_bundle_bytes = bundle.wire_byte_size();
    let mut client_d = IkpirClient::<B>::from_setup(bundle);

    // Once-per-config decode sanity check: same shape as classical_throughput.
    // Catches silent decode regressions (wrong cells_per_slot, packing bug,
    // hint mismatch, bad plaintext_bits operating point) before the
    // long-running timed loops.
    helpers::verify_decode::<B, _>(&mut client_d, &server, 1u32, 16, cli.value_bits);

    let samples_d: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
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
                samples_d
                    .lock()
                    .unwrap()
                    .push(total.as_nanos() as f64 / iters as f64);
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
            name: "num_keys",
            value: cli.num_keys.to_string(),
            is_default: false,
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
            name: "batch",
            value: batch.to_string(),
            is_default: matches.value_source("batch") != Some(ValueSource::CommandLine),
        },
    ];
    helpers::print_preamble("headtohead_throughput", &knobs, &store_state, &geom);
    println!(
        "  answer={:.2} qps (±{:.2}) | query={:.2} qps (±{:.2}) | decode={:.2} dps (±{:.2})",
        crit_answer.mean_ops_per_s,
        crit_answer.stddev_ops_per_s,
        crit_query.mean_ops_per_s,
        crit_query.stddev_ops_per_s,
        crit_decode.mean_ops_per_s,
        crit_decode.stddev_ops_per_s,
    );

    // ── 7. Write CSVs ───────────────────────────────────────────────────────────
    let num_keys = cli.num_keys;
    writeln!(
        csv_answer,
        "{},{arity},{num_buckets},{},{num_keys},{db_size},{},{},{},{lwe_dim_eff},{batch},{:.2},{:.2},{:.2},{:.2},{query_bytes},{response_bytes},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits, cli.fingerprint_bits,
        crit_answer.mean_ops_per_s, crit_answer.min_ops_per_s,
        crit_answer.max_ops_per_s,  crit_answer.stddev_ops_per_s,
        load_factor,
    ).unwrap();
    writeln!(
        csv_query,
        "{},{arity},{num_buckets},{},{num_keys},{db_size},{},{},{},{lwe_dim_eff},{batch},{:.2},{:.2},{:.2},{:.2},{query_bytes},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits, cli.fingerprint_bits,
        crit_query.mean_ops_per_s, crit_query.min_ops_per_s,
        crit_query.max_ops_per_s,  crit_query.stddev_ops_per_s,
        load_factor,
    ).unwrap();
    writeln!(
        csv_decode,
        "{},{arity},{num_buckets},{},{num_keys},{db_size},{},{},{},{lwe_dim_eff},{batch},{:.2},{:.2},{:.2},{:.2},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits, cli.fingerprint_bits,
        crit_decode.mean_ops_per_s, crit_decode.min_ops_per_s,
        crit_decode.max_ops_per_s,  crit_decode.stddev_ops_per_s,
        load_factor,
    ).unwrap();
}

fn dispatch_backend<S: MakeStore>(
    csv_answer: &mut std::io::BufWriter<std::fs::File>,
    csv_query: &mut std::io::BufWriter<std::fs::File>,
    csv_decode: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    let lwe_dim = effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo => run_one::<S, FrodoPirBackend>(
            csv_answer,
            csv_query,
            csv_decode,
            cli,
            arity,
            num_buckets,
            FrodoConfig::with_lwe_dim(lwe_dim),
        ),
        Backend::Simple => run_one::<S, SimplePirBackend>(
            csv_answer,
            csv_query,
            csv_decode,
            cli,
            arity,
            num_buckets,
            SimpleConfig::with_lwe_dim(lwe_dim),
        ),
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

    let mut csv_answer = helpers::csv_writer("ikpir_headtohead_server_answer.csv", HEADER_ANSWER);
    let mut csv_query = helpers::csv_writer("ikpir_headtohead_client_query.csv", HEADER_QUERY);
    let mut csv_decode = helpers::csv_writer("ikpir_headtohead_client_decode.csv", HEADER_DECODE);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(
            &mut csv_answer,
            &mut csv_query,
            &mut csv_decode,
            &cli,
            2,
            num_buckets,
        ),
        3 => dispatch_backend::<Segmented3aryScheme>(
            &mut csv_answer,
            &mut csv_query,
            &mut csv_decode,
            &cli,
            3,
            num_buckets,
        ),
        4 => dispatch_backend::<Segmented4aryScheme>(
            &mut csv_answer,
            &mut csv_query,
            &mut csv_decode,
            &cli,
            4,
            num_buckets,
        ),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_headtohead_server_answer.csv, ikpir_headtohead_client_query.csv, ikpir_headtohead_client_decode.csv");
}
