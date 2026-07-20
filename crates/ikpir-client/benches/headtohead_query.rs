//! **Intent:** Head-to-head counterpart of `client_query` — measure
//! client-side `build_query` throughput at a **fixed keyword count**, for the
//! fair comparison against ChalametPIR and Hao et al. 2025. See
//! `headtohead_answer` for the motivation behind fixing `num_keys` instead of
//! DB size, and for the two key-count regimes `scripts/table3.sh` sweeps.
//!
//! **Method:** Identical warm-bc pipeline as `client_query` except
//! `populate_exact_n_keys(target_n = num_keys)` replaces `populate_until_full`,
//! and the CSV carries the extra `num_keys` / `db_size` / `fingerprint_bits`
//! columns.
//!
//! **Arguments (CLI):** Same as `client_query`, plus `--num-keys` (required)
//! and `--max-mem-gb` (OOM guard).
//!
//! **Output:** `results/ikpir_headtohead_client_query.csv`

// significant_drop_tightening: clippy's inline-`Criterion` fix borrows a temporary dropped while `BenchmarkGroup` holds it (won't compile).
#![allow(clippy::significant_drop_tightening)]

mod helpers;

use criterion::Throughput;
use helpers::{Backend, MakeStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirClient, IncrementalPirBackend,
    IndexPirBackend, ParallelSetupBackend, PrecomputingPirBackend, SimpleConfig, SimplePirBackend,
};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const HEADER: &str =
    "backend,arity,num_buckets,bucket_size,num_keys,db_size,value_bits,plaintext_bits,fingerprint_bits,lwe_dim,batch,\
    mean_qps,min_qps,max_qps,stddev_qps,query_bytes,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

#[derive(Clone, clap::Parser)]
#[command(
    about = "Head-to-head client build_query bench: fixes num_keys, reports DB size, otherwise mirrors client_query."
)]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    /// Required: number of keys to populate. The DB size (= `num_buckets ×
    /// bucket_size`) is fixed by `--num-buckets` / `--bucket-size`; pick those
    /// so capacity ≥ `num_keys` at a reasonable load factor.
    #[arg(long)]
    num_keys: u64,
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
    /// Key-pool size: the bench rotates through this many distinct keys so
    /// repeated iterations do not reuse hot CPU-cache state from the previous call.
    #[arg(long, default_value_t = 64)]
    batch: u32,
    /// Skip configs whose estimated peak memory exceeds this limit. Default
    /// 85.0 is tuned for a ~96 GB server; lower it on smaller machines.
    #[arg(long, default_value_t = 85.0)]
    max_mem_gb: f64,
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
    backend_config: B::Config,
) where
    S: MakeStore,
    B: IndexPirBackend
        + ParallelSetupBackend
        + IncrementalPirBackend
        + PrecomputingPirBackend
        + BackendWireSize
        + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    // ── 0. Memory guard ─────────────────────────────────────────────────────────
    let lwe_dim_est = effective_lwe_dim(cli) as u64;
    let cells_per_slot_est =
        (cli.fingerprint_bits + cli.value_bits).div_ceil(cli.plaintext_bits) as u64;
    let row_width_est = cli.bucket_size as u64 * cells_per_slot_est;
    let n_rows_per_seg = num_buckets as u64 / arity as u64;
    let table_bytes = num_buckets as u64 * cli.bucket_size as u64 * cells_per_slot_est * 4;
    let (a_rows_per_seg, _c_len_per_seg) =
        helpers::backend_shape_estimate(cli.backend, n_rows_per_seg, row_width_est);
    let a_bytes = arity as u64 * a_rows_per_seg * lwe_dim_est * 4;
    let estimated_gb = (table_bytes + a_bytes) as f64 / 1e9;
    if estimated_gb > cli.max_mem_gb {
        eprintln!(
            "  Skip (OOM guard): estimated peak {:.1} GB > --max-mem-gb {:.1} \
             (nb={num_buckets} bs={} vb={} lwe_dim={lwe_dim_est} backend={}). \
             Raise --max-mem-gb on machines with more RAM.",
            estimated_gb, cli.max_mem_gb, cli.bucket_size, cli.value_bits, cli.backend,
        );
        return;
    }

    // ── 1. Populate exactly num_keys ────────────────────────────────────────────
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

    // ── 2. Server setup; the query path never touches the server's `A`, so
    //       free it right after taking the bundle to keep peak RAM to the
    //       single client-side copy. ────────────────────────────────────────────
    let mut server: IkpirServer<S, B> = IkpirServer::new_parallel(store, backend_config);
    let bundle = server.setup();
    server.drop_hint_material();

    let query_bytes = {
        let mut probe: IkpirClient<B> = IkpirClient::from_setup_parallel(bundle.clone());
        probe.build_query(&0u32.to_le_bytes()).wire_byte_size()
    };

    // ── 3. Geometry ─────────────────────────────────────────────────────────────
    let params_store = server.params();
    let cps = params_store.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let segment_rows = params_store.segment_size();
    let (db_rows, db_cols) =
        helpers::backend_shape_estimate(cli.backend, segment_rows as u64, row_width as u64);
    let db_size = (num_buckets as u64) * (cli.bucket_size as u64);
    let load_factor = n_inserted as f64 / db_size as f64;
    let lwe_dim_eff = effective_lwe_dim(cli);
    let store_state = helpers::StoreState {
        capacity: db_size,
        populated: n_inserted,
        load_pct: 100.0 * load_factor,
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes: B::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes: bundle.wire_byte_size(),
        query_bytes,
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
            name: "num_keys",
            value: cli.num_keys.to_string(),
            is_default: false,
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
            value: cli.batch.to_string(),
            is_default: matches.value_source("batch") != Some(ValueSource::CommandLine),
        },
    ];
    helpers::print_preamble("headtohead_query", &knobs, &store_state, &geom);

    // ── 4. Measure build_query (warm-bc) ─────────────────────────────────────────
    let n = n_inserted as u32;
    let keys: Vec<[u8; 4]> = (0..cli.batch).map(|i| (i % n).to_le_bytes()).collect();
    let mut client: IkpirClient<B> = IkpirClient::from_setup_parallel(bundle);

    let samples: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let mut idx = 0usize;
    {
        let mut c = helpers::configured_criterion();
        let mut group = c.benchmark_group("headtohead_query");
        group.throughput(Throughput::Elements(1));
        group.bench_function("headtohead_query", |b| {
            b.iter_custom(|iters| {
                client.precompute_queries(iters as u32);
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let k = keys[idx % keys.len()];
                    idx = idx.wrapping_add(1);
                    let t = Instant::now();
                    let _ = client.build_query(&k);
                    total += t.elapsed();
                }
                samples
                    .lock()
                    .unwrap()
                    .push(total.as_nanos() as f64 / iters as f64);
                total
            });
        });
        group.finish();
    }
    let raw = std::mem::take(&mut *samples.lock().unwrap());
    let crit = if raw.is_empty() {
        helpers::CriterionThroughputStats {
            mean_ops_per_s: 0.0,
            min_ops_per_s: 0.0,
            max_ops_per_s: 0.0,
            stddev_ops_per_s: 0.0,
        }
    } else {
        let ops: Vec<f64> = raw.iter().map(|&ns| 1e9 / ns).collect();
        let s = helpers::compute_stats(&ops);
        helpers::CriterionThroughputStats {
            mean_ops_per_s: s.mean,
            min_ops_per_s: s.min,
            max_ops_per_s: s.max,
            stddev_ops_per_s: s.stddev,
        }
    };

    // ── 5. Write CSV ─────────────────────────────────────────────────────────────
    let num_keys = cli.num_keys;
    writeln!(
        csv,
        "{},{arity},{num_buckets},{},{num_keys},{db_size},{},{},{},{lwe_dim_eff},{},{:.2},{:.2},{:.2},{:.2},{query_bytes},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits, cli.fingerprint_bits, cli.batch,
        crit.mean_ops_per_s, crit.min_ops_per_s, crit.max_ops_per_s, crit.stddev_ops_per_s,
        load_factor,
    ).unwrap();
    println!(
        "  backend={} arity={arity} nb={num_buckets:<7} N={num_keys} vb={:<4} | \
         mean={:.2} qps (±{:.2}) q={query_bytes}B",
        cli.backend, cli.value_bits, crit.mean_ops_per_s, crit.stddev_ops_per_s,
    );
}

fn dispatch_backend<S: MakeStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    let lwe_dim = effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo => run_one::<S, FrodoPirBackend>(
            csv,
            cli,
            arity,
            num_buckets,
            FrodoConfig::with_lwe_dim(lwe_dim),
        ),
        Backend::Simple => run_one::<S, SimplePirBackend>(
            csv,
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

    let mut csv = helpers::csv_writer("ikpir_headtohead_client_query.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to ikpir_headtohead_client_query.csv");
}
