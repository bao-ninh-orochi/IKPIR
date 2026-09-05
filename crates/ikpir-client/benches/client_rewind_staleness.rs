//! **Intent:** Measure the response-rewind client's **per-query correction
//! cost as a function of staleness** `|ΔD|` — the price paid for the factor-`n`
//! cheaper maintenance the `client_rewind_mutation` bench reports. A rewind
//! client that never garbage-collects accumulates a growing `ΔD`; `decode`
//! subtracts `qᵀ·ΔD` per query (`Θ(|ΔD|)` on top of the constant
//! `client_rewind_decode`), so its
//! per-query latency rises with the number of unpatched mutations, until a
//! `collect_garbage` folds `ΔD` into the hint and returns it to baseline.
//!
//! **Method:** Populate to `--load-factor`, build **one** rewind client from the
//! epoch-0 setup bundle, and measure a baseline `decode` latency at
//! `|ΔD| = 0`. Then, for `--staleness-steps` steps, apply a batch of
//! `--batch-size` updates to the server, `accumulate_delta` every delta into the
//! client (its hint is never patched — `pin_epoch` stays 0), and re-measure the
//! per-query `decode` latency at the grown `|ΔD|`. Finally
//! `collect_garbage` and re-measure — the correction is reclaimed. Each
//! measurement times **only** `decode` (`build_query` + `answer` happen
//! outside the timed bracket), over `--queries` present keys round-robin, and
//! asserts the value is found so a mistimed no-op cannot masquerade as fast.
//!
//! **Single-threaded, non-SIMD** timed path (`decode` → `client_decode`),
//! the paper's regime.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple),
//! `--num-buckets`, `--bucket-size`, `--value-bits`, `--fingerprint-bits`,
//! `--plaintext-bits`, `--lwe-dim`, `--load-factor` (default 0.90),
//! `--batch-size` (updates accumulated per step, default 512), `--staleness-steps`
//! (default 8), `--queries` (decode calls per measurement, default 200).
//!
//! **Output:** `results/ikpir_client_rewind_staleness.csv`
//! Columns: backend, arity, num_buckets, bucket_size, value_bits, plaintext_bits,
//! lwe_dim, phase, step, pending_cells, queries, decode_us_mean,
//! cells_per_slot, row_width, segment_rows, db_rows, db_cols

mod helpers;

use helpers::{Backend, CloneStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IncrementalPirBackend, IndexPirBackend,
    ParallelSetupBackend, ResponseRewind, RewindClient, SimpleConfig, SimplePirBackend,
};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::time::{Duration, Instant};

const HEADER: &str = "backend,arity,num_buckets,bucket_size,value_bits,plaintext_bits,lwe_dim,\
    phase,step,pending_cells,queries,decode_us_mean,cells_per_slot,row_width,segment_rows,\
    db_rows,db_cols";

#[derive(Clone, clap::Parser)]
#[command(about = "Measure rewind decode per-query latency vs staleness |ΔD| (single client).")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    #[arg(long, default_value_t = 16_384)]
    num_buckets: u32,
    #[arg(long, default_value_t = 4)]
    bucket_size: u32,
    #[arg(long, default_value_t = 2048)]
    value_bits: u32,
    #[arg(long, default_value_t = 64)]
    fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]
    plaintext_bits: u32,
    #[arg(long)]
    lwe_dim: Option<u32>,
    #[arg(long, default_value_t = 0.90)]
    load_factor: f64,
    /// Updates accumulated into ΔD per staleness step.
    #[arg(long, default_value_t = 512)]
    batch_size: u32,
    /// Number of staleness steps (each adds `--batch-size` to |ΔD|).
    #[arg(long, default_value_t = 8)]
    staleness_steps: u32,
    /// `decode` calls timed per measurement.
    #[arg(long, default_value_t = 200)]
    queries: u32,
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

/// Time only `decode` over `m` present keys round-robin; assert every
/// query is found. Returns mean microseconds per query.
fn measure_decode<S, B>(
    client: &mut RewindClient<B>,
    server: &IkpirServer<S, B>,
    keys: &[u32],
    m: u32,
) -> f64
where
    S: CloneStore,
    B: IncrementalPirBackend + ResponseRewind,
    B::Query: Clone,
    B::Response: Clone,
{
    let mut total = Duration::ZERO;
    for i in 0..m {
        let key = keys[(i as usize) % keys.len()];
        let kb = key.to_le_bytes();
        let q = client.build_query(&kb);
        let r = server.answer(&q).expect("answer");
        let t = Instant::now();
        let v = client.decode(&kb, &q, &r).expect("decode");
        total += t.elapsed();
        assert!(v.is_some(), "staleness bench queried a present key {key}");
    }
    total.as_secs_f64() * 1e6 / f64::from(m)
}

fn run_one<S, B>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    make_config: impl Fn() -> B::Config,
) where
    S: CloneStore,
    B: IndexPirBackend
        + ParallelSetupBackend
        + IncrementalPirBackend
        + ResponseRewind
        + BackendWireSize,
    B::Query: Clone,
    B::Response: Clone,
{
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

    // One server, one rewind client bootstrapped from its epoch-0 bundle.
    let store0 = S::clone_from_cells(cells.clone(), params, n_seed).expect("from_cells");
    let mut server: IkpirServer<S, B> = IkpirServer::new_parallel(store0, make_config());
    let bundle = server.setup();
    let (db_rows, db_cols) = B::db_matrix_shape(&server.backend_params()[0]);
    let cps = params.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let segment_rows = params.segment_size();

    let mut client = RewindClient::<B>::from_setup_parallel(bundle);

    // A pool of present keys to query round-robin (also the update targets).
    let pool: Vec<u32> = (0..(n_seed as u32).min(4096)).collect();
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];

    let emit = |csv: &mut std::io::BufWriter<std::fs::File>,
                phase: &str,
                step: u32,
                pending: usize,
                us: f64| {
        writeln!(
            csv,
            "{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{phase},{step},{pending},{},{us:.3},{cps},{row_width},{segment_rows},{db_rows},{db_cols}",
            cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits, cli.queries,
        )
        .unwrap();
    };

    // Baseline at |ΔD| = 0.
    let base = measure_decode(&mut client, &server, &pool, cli.queries);
    emit(csv, "baseline", 0, client.pending_cells(), base);
    println!(
        "  {} arity={arity} baseline |ΔD|=0 -> {base:.2} us/query",
        cli.backend
    );

    // Grow ΔD in steps; never garbage-collect. Each step updates a fresh,
    // disjoint window of keys (so new cells enter ΔD and |ΔD| rises); when the
    // steps exhaust the seed they wrap and re-touch (ΔD then plateaus at the
    // touched-cell footprint, which is itself a meaningful data point).
    for step in 1..=cli.staleness_steps {
        for i in 0..cli.batch_size {
            let key = ((u64::from((step - 1) * cli.batch_size + i)) % n_seed) as u32;
            // salt varies per step so every update actually changes the value.
            fill_value(&mut value, key, 40 + step);
            let delta = server.update(&key.to_le_bytes(), &value).expect("update");
            client.accumulate_delta(delta).expect("accumulate_delta");
        }
        let pending = client.pending_cells();
        let us = measure_decode(&mut client, &server, &pool, cli.queries);
        emit(csv, "stale", step, pending, us);
        println!(
            "  {} arity={arity} step={step} |ΔD|={pending} -> {us:.2} us/query",
            cli.backend
        );
    }

    // Garbage-collect: fold ΔD into the hint, then re-measure at |ΔD| = 0.
    client.collect_garbage().expect("collect_garbage");
    let after = measure_decode(&mut client, &server, &pool, cli.queries);
    emit(
        csv,
        "post_gc",
        cli.staleness_steps,
        client.pending_cells(),
        after,
    );
    println!(
        "  {} arity={arity} post-GC |ΔD|={} -> {after:.2} us/query",
        cli.backend,
        client.pending_cells()
    );
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

    let mut csv = helpers::csv_writer("ikpir_client_rewind_staleness.csv", HEADER);
    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_rewind_staleness.csv");
}
