//! **Intent:** Mixed insert/query workload measuring steady-state per-op
//! latency under a configurable query-to-mutation interleaving. This is the
//! metric that matters for a live IKPIR deployment — the other benches are
//! single-operation microbenchmarks.
//!
//! **Method:** Populate to `--load-factor` (default 0.50), snapshot the cell
//! array, then per trial clone via `from_cells` and run `n_inserts + n_queries`
//! operations interleaved: every `query_ratio`-th op is a query, the rest are
//! inserts. For inserts, `client.apply_delta` is also timed (client must stay
//! in sync). For queries, `build_query → answer → decode` is timed end-to-end.
//!
//! On `IkpirError::TableFull` inserts stop gracefully; the row reflects the
//! partial workload.
//!
//! **Output:** `results/ikpir_steady_state_workload.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! n_inserts, n_queries, query_ratio, total_ms, mean_ms_per_op,
//! mean_insert_ms, mean_query_ms

mod helpers;

use helpers::CloneStore;
use ikpir_client::IkpirClient;
use ikpir_server::{BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer};
use segmented_cuckoo::{
    CuckooParams,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::{Duration, Instant};

type Client = IkpirClient<FrodoPirBackend>;

const HEADER: &str = "arity,num_buckets,bucket_size,value_bits,lwe_dim,\
    n_inserts,n_queries,query_ratio,total_ms,mean_ms_per_op,mean_insert_ms,mean_query_ms";

#[derive(Clone, clap::Parser)]
#[command(about = "Mixed insert/query workload: amortised per-op latency.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)]   lwe_dim: u32,
    /// Inserts to attempt in the workload.
    #[arg(long, default_value_t = 4096)]   n_inserts: u32,
    /// Queries to attempt in the workload.
    #[arg(long, default_value_t = 410)]    n_queries: u32,
    /// Operation interleaving: every `query_ratio`-th op is a query.
    #[arg(long, default_value_t = 10)]     query_ratio: u32,
    #[arg(long, default_value_t = 1)]      warmup: u32,
    #[arg(long, default_value_t = 3)]      trials: u32,
    /// Load factor before the workload starts.
    #[arg(long, default_value_t = 0.50)]   load_factor: f64,
}

struct TrialResult {
    total_ms:     f64,
    mean_per_op:  f64,
    mean_insert:  f64,
    mean_query:   f64,
    done_inserts: u32,
    done_queries: u32,
}

fn run_trial<S: CloneStore>(
    cli:      &Cli,
    cells:    &[u32],
    params:   CuckooParams,
    n_seed:   u64,
) -> TrialResult {
    let store = S::clone_from_cells(cells.to_vec(), params, n_seed).expect("from_cells");
    let mut server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let mut client: Client = Client::from_setup(server.setup());

    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let n_ops = cli.n_inserts + cli.n_queries;
    let mut next_key = n_seed as u32;
    let mut insert_total = Duration::ZERO;
    let mut query_total  = Duration::ZERO;
    let mut done_i = 0u32;
    let mut done_q = 0u32;
    let mut stop_inserts = false;

    let start = Instant::now();
    for op in 0..n_ops {
        let do_query = (op % cli.query_ratio == 0 && done_q < cli.n_queries)
                    || done_i >= cli.n_inserts
                    || stop_inserts;
        if do_query && done_q < cli.n_queries {
            let key_u = if n_seed > 0 { (op as u64 % n_seed) as u32 } else { 0 };
            let key = key_u.to_le_bytes();
            let t = Instant::now();
            let q = client.build_query(&key);
            let r = server.answer(&q).expect("answer ok");
            let _ = client.decode(&key, &r).expect("decode ok");
            query_total += t.elapsed();
            done_q += 1;
        } else if done_i < cli.n_inserts && !stop_inserts {
            for (i, b) in value.iter_mut().enumerate() {
                *b = (next_key.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
            }
            let key = next_key.to_le_bytes();
            let t = Instant::now();
            match server.insert(&key, &value) {
                Ok(delta) => {
                    client.apply_delta(delta).expect("apply_delta ok");
                    insert_total += t.elapsed();
                    done_i += 1;
                    next_key += 1;
                }
                Err(IkpirError::TableFull) => { stop_inserts = true; }
                Err(e) => panic!("insert: {e:?}"),
            }
        }
        if done_i >= cli.n_inserts && done_q >= cli.n_queries { break; }
    }
    let total_ms = start.elapsed().as_secs_f64() * 1e3;
    let done_ops = (done_i + done_q).max(1) as f64;
    let mean_per_op = total_ms / done_ops;
    let mean_insert = if done_i > 0 { insert_total.as_secs_f64() * 1e3 / done_i as f64 } else { 0.0 };
    let mean_query  = if done_q > 0 { query_total.as_secs_f64()  * 1e3 / done_q as f64 } else { 0.0 };
    TrialResult { total_ms, mean_per_op, mean_insert, mean_query, done_inserts: done_i, done_queries: done_q }
}

fn run_one<S: CloneStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    let (seed_store, n_seed) = helpers::populate_to_load::<S>(
        cli.load_factor, num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    if n_seed < 2 { eprintln!("  Skip: seed too small"); return; }

    let cells  = seed_store.snapshot_cells();
    let params = seed_store.params();

    // Geometry (build once for preamble).
    let cps = params.cells_per_slot();
    let tmp_store = S::clone_from_cells(cells.clone(), params, n_seed).expect("from_cells");
    let tmp_server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(tmp_store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let bundle = tmp_server.setup();
    let store_state = helpers::StoreState {
        capacity:       (num_buckets as u64) * (cli.bucket_size as u64),
        populated:      n_seed,
        load_pct:       100.0 * n_seed as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width:      cli.bucket_size * cps,
        segment_rows:   params.segment_size(),
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       FrodoPirBackend::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:       bundle.wire_byte_size(),
        query_bytes:              0,
        response_bytes:           0,
        hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob { name: "arity",        value: arity.to_string(),                  is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",   value: num_buckets.to_string(),            is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",   value: cli.bucket_size.to_string(),        is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",    value: cli.value_bits.to_string(),         is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",       value: cli.lwe_dim.to_string(),            is_default: matches.value_source("lwe_dim") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "load_factor",   value: format!("{:.2}", cli.load_factor),  is_default: matches.value_source("load_factor") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "n_inserts",     value: cli.n_inserts.to_string(),          is_default: matches.value_source("n_inserts") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "n_queries",     value: cli.n_queries.to_string(),          is_default: matches.value_source("n_queries") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "query_ratio",   value: cli.query_ratio.to_string(),        is_default: matches.value_source("query_ratio") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("steady_state_workload", &knobs, &store_state, &geom);

    let mut samples: Vec<TrialResult> = Vec::with_capacity(cli.trials as usize);
    for trial in 0..(cli.warmup + cli.trials) {
        let r = run_trial::<S>(cli, &cells, params, n_seed);
        if trial >= cli.warmup { samples.push(r); }
    }
    let n = samples.len() as f64;
    let mean = |sel: fn(&TrialResult) -> f64| samples.iter().map(sel).sum::<f64>() / n;
    let m_total  = mean(|r| r.total_ms);
    let m_per_op = mean(|r| r.mean_per_op);
    let m_ins    = mean(|r| r.mean_insert);
    let m_q      = mean(|r| r.mean_query);
    let last = samples.last().unwrap();
    writeln!(
        csv,
        "{arity},{num_buckets},{},{},{},{},{},{},{:.3},{:.6},{:.6},{:.6}",
        cli.bucket_size, cli.value_bits, cli.lwe_dim,
        last.done_inserts, last.done_queries, cli.query_ratio,
        m_total, m_per_op, m_ins, m_q,
    ).unwrap();
    println!(
        "  arity={arity} nb={num_buckets:<6} bs={} vb={:<4} | \
         {}I + {}Q in {m_total:.3}ms | per_op={m_per_op:.3}ms ins={m_ins:.3}ms q={m_q:.3}ms",
        cli.bucket_size, cli.value_bits, last.done_inserts, last.done_queries,
    );
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_steady_state_workload.csv", HEADER);

    match cli.arity {
        2 => run_one::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run_one::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run_one::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_steady_state_workload.csv");
}
