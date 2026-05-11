//! **Intent:** Mixed insert/query workload measuring steady-state per-op
//! latency under a configurable query-to-mutation interleaving. This is the
//! metric that matters for a live IKPIR deployment — the existing throughput
//! benches are all single-operation microbenchmarks.
//!
//! **Method:** Pre-populate a server to ~40% load (not timed). Then run
//! `n_inserts + n_queries` operations interleaved: every `query_ratio`-th op
//! is a query, the rest are inserts. For inserts, also `client.apply_delta`
//! is timed (since the client cannot otherwise stay in sync). For queries,
//! `client.build_query → server.answer → client.decode` is timed end-to-end.
//!
//! On `IkpirError::TableFull` we stop further inserts gracefully and report
//! what completed; the row reflects the partial workload.
//!
//! **Invocation:** `cargo bench --bench steady_state_workload` runs a single
//! default config. Pass `--sweep` for the full matrix; pass
//! `--n-inserts <N> --n-queries <N> --query-ratio <K>` to pin the workload.
//!
//! **Output:** `results/ikpir_steady_state_workload.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! n_inserts, n_queries, query_ratio, total_ms, mean_ms_per_op,
//! mean_insert_ms, mean_query_ms

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer};
use segmented_cuckoo::{
    CuckooError,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::{Duration, Instant};

type Client = IkpirClient<FrodoPirBackend>;
const MAX_KICKS: u32 = 2_500;

#[derive(clap::Parser)]
#[command(about = "Mixed insert/query workload: amortised per-op latency.")]
struct Cli {
    #[arg(long)] sweep: bool,
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))] arity: Option<u32>,
    #[arg(long, default_value_t = 256)]  num_buckets: u32,
    #[arg(long, default_value_t = 4)]    bucket_size: u32,
    #[arg(long, default_value_t = 64)]   value_bits: u32,
    #[arg(long, default_value_t = 12)]   fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]    plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)] lwe_dim: u32,
    /// Inserts to attempt in the workload.
    #[arg(long, default_value_t = 200)]  n_inserts: u32,
    /// Queries to attempt in the workload.
    #[arg(long, default_value_t = 20)]   n_queries: u32,
    /// Operation interleaving: every `query_ratio`-th op is a query, the
    /// rest are inserts (until either budget is exhausted).
    #[arg(long, default_value_t = 10)]   query_ratio: u32,
    #[arg(long, default_value_t = 1)]    warmup: u32,
    #[arg(long, default_value_t = 3)]    trials: u32,
}

fn build_seeded<S>(
    cli: &Cli,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
) -> Option<(IkpirServer<S, FrodoPirBackend>, Client, u32)>
where S: MakeStore {
    let mut store = S::make_store(
        num_buckets, bucket_size, cli.fingerprint_bits, value_bits, cli.plaintext_bits,
    ).ok()?;
    store.set_max_kicks(MAX_KICKS);

    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let target_load: u32 = ((num_buckets * bucket_size) as f64 * 0.40) as u32;
    let mut last_seeded: u32 = 0;
    for k in 0u32..target_load {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k.to_le_bytes(), &value) {
            Ok(())                       => last_seeded = k,
            Err(CuckooError::TableFull)  => break,
            Err(e)                       => panic!("build_seeded: {e:?}"),
        }
    }
    if last_seeded < 2 { return None; }
    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let client: Client = Client::from_setup(server.setup());
    Some((server, client, last_seeded + 1))
}

/// Returns (total_ms, mean_ms_per_op, mean_insert_ms, mean_query_ms,
/// done_inserts, done_queries).
fn time_workload<S>(
    cli: &Cli,
    server: &mut IkpirServer<S, FrodoPirBackend>,
    client: &mut Client,
    seed_count: u32,
    value_bits: u32,
) -> (f64, f64, f64, f64, u32, u32)
where S: MakeStore {
    let n_inserts = cli.n_inserts;
    let n_queries = cli.n_queries;
    let n_ops     = n_inserts + n_queries;
    let vsize     = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];

    let mut next_key   = seed_count;
    let mut insert_total = Duration::ZERO;
    let mut query_total  = Duration::ZERO;
    let mut done_i: u32 = 0;
    let mut done_q: u32 = 0;
    let mut stop_inserts = false;

    let start = Instant::now();
    for op in 0..n_ops {
        // Query slot when (op % query_ratio == 0) AND query budget remains;
        // otherwise insert. If inserts are exhausted or have stalled, the
        // remaining ops are queries.
        let do_query = (op % cli.query_ratio == 0 && done_q < n_queries)
                    || done_i >= n_inserts || stop_inserts;
        if do_query && done_q < n_queries {
            // Bias the queried key into the seeded range so we always exercise
            // a hit path (steady-state user queries are usually hits).
            let key_u = if seed_count > 0 { op % seed_count } else { 0 };
            let key = key_u.to_le_bytes();
            let t = Instant::now();
            let q = client.build_query(&key);
            let r = server.answer(&q).expect("answer ok");
            let _ = client.decode(&key, &r).expect("decode ok");
            query_total += t.elapsed();
            done_q += 1;
        } else if done_i < n_inserts && !stop_inserts {
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
                Err(IkpirError::TableFull) => {
                    stop_inserts = true;
                    // Drop the elapsed time of the failed attempt; it is the
                    // rollback path, already measured in failure_modes.rs.
                }
                Err(e) => panic!("insert: {e:?}"),
            }
        }
        if done_i >= n_inserts && done_q >= n_queries { break; }
    }
    let total = start.elapsed();

    let done_ops = (done_i + done_q).max(1) as f64;
    let mean_insert = if done_i > 0 { insert_total.as_secs_f64() * 1e3 / done_i as f64 } else { 0.0 };
    let mean_query  = if done_q > 0 { query_total.as_secs_f64()  * 1e3 / done_q as f64 } else { 0.0 };
    let total_ms    = total.as_secs_f64() * 1e3;
    let mean_per_op = total_ms / done_ops;
    (total_ms, mean_per_op, mean_insert, mean_query, done_i, done_q)
}

fn run_one<S>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    bucket_size: u32,
    value_bits:  u32,
)
where S: MakeStore {
    let mut totals       = Vec::with_capacity(cli.trials as usize);
    let mut per_ops      = Vec::with_capacity(cli.trials as usize);
    let mut ins_means    = Vec::with_capacity(cli.trials as usize);
    let mut q_means      = Vec::with_capacity(cli.trials as usize);
    let mut last_done_i  = 0u32;
    let mut last_done_q  = 0u32;

    for trial in 0..(cli.warmup + cli.trials) {
        let (mut server, mut client, seed_count) =
            match build_seeded::<S>(cli, num_buckets, bucket_size, value_bits) {
                Some(t) => t,
                None    => {
                    eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}");
                    return;
                }
            };
        let (total, per_op, mean_i, mean_q, done_i, done_q) =
            time_workload::<S>(cli, &mut server, &mut client, seed_count, value_bits);
        if trial >= cli.warmup {
            totals.push(total);
            per_ops.push(per_op);
            ins_means.push(mean_i);
            q_means.push(mean_q);
            last_done_i = done_i;
            last_done_q = done_q;
        }
    }
    let n = totals.len() as f64;
    let mean_total  = totals.iter().sum::<f64>() / n;
    let mean_per_op = per_ops.iter().sum::<f64>() / n;
    let mean_ins    = ins_means.iter().sum::<f64>() / n;
    let mean_qry    = q_means.iter().sum::<f64>() / n;

    writeln!(
        csv,
        "{arity},{num_buckets},{bucket_size},{value_bits},{},{last_done_i},{last_done_q},{},{:.3},{:.6},{:.6},{:.6}",
        cli.lwe_dim, cli.query_ratio, mean_total, mean_per_op, mean_ins, mean_qry,
    )
    .unwrap();
    println!(
        "  arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} | \
         {last_done_i}I + {last_done_q}Q in {mean_total:.3}ms | \
         per_op={mean_per_op:.6}ms ins={mean_ins:.6}ms q={mean_qry:.6}ms"
    );
}

fn dispatch(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
) {
    match arity {
        2 => run_one::<Segmented2aryScheme>(csv, cli, 2, num_buckets, bucket_size, value_bits),
        3 => run_one::<Segmented3aryScheme>(csv, cli, 3, num_buckets, bucket_size, value_bits),
        4 => run_one::<Segmented4aryScheme>(csv, cli, 4, num_buckets, bucket_size, value_bits),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
}

fn sweep_buckets(arity: u32) -> &'static [u32] {
    match arity {
        2 | 4 => &[64, 256, 1024],
        3     => &[96, 384, 1536],
        _     => unreachable!(),
    }
}

fn resolve_num_buckets(cli: &Cli, arity: u32) -> u32 {
    if cli.num_buckets == 256 && arity == 3 { 384 } else { cli.num_buckets }
}

fn main() {
    let cli: Cli = helpers::parse_cli();
    let mut csv = helpers::csv_writer(
        "ikpir_steady_state_workload.csv",
        "arity,num_buckets,bucket_size,value_bits,lwe_dim,n_inserts,n_queries,query_ratio,total_ms,mean_ms_per_op,mean_insert_ms,mean_query_ms",
    );

    println!("=== ikpir steady-state workload (FrodoPirBackend) ===");
    println!(
        "Config: fingerprint_bits={}, plaintext_bits={}, lwe_dim={}, n_inserts={}, n_queries={}, query_ratio={}, warmup={}, trials={}",
        cli.fingerprint_bits, cli.plaintext_bits, cli.lwe_dim,
        cli.n_inserts, cli.n_queries, cli.query_ratio, cli.warmup, cli.trials,
    );

    let arities: Vec<u32> = match (cli.sweep, cli.arity) {
        (true,  None)    => vec![2, 3, 4],
        (true,  Some(a)) => vec![a],
        (false, opt)     => vec![opt.unwrap_or(2)],
    };

    for &arity in &arities {
        if cli.sweep {
            for &nb in sweep_buckets(arity) {
                for &bs in &[2u32, 4] {
                    for &vb in &[8u32, 64, 256] {
                        dispatch(&mut csv, &cli, arity, nb, bs, vb);
                    }
                }
            }
        } else {
            let nb = resolve_num_buckets(&cli, arity);
            dispatch(&mut csv, &cli, arity, nb, cli.bucket_size, cli.value_bits);
        }
    }
    println!("\nResults written to results/ikpir_steady_state_workload.csv");
}
