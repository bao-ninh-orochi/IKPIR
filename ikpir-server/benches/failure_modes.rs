//! **Intent:** Measure rejection-path latency for `StaleEpoch` and
//! `TableFull`, across both shipped backends (FrodoPIR and SimplePIR).
//!
//! **Method:**
//!
//! - `stale_epoch`: build a populated server, build a client query at
//!   epoch 0, mutate the server once (epoch → 1), then call `answer`
//!   with the stale query repeatedly.
//! - `table_full`: fill a tiny table to capacity, then repeatedly
//!   attempt to insert one more key (each attempt exhausts the kick
//!   budget deterministically).
//!
//! **Arguments (CLI):** `--arity`, `--backend` (frodo|simple, default
//! frodo), `--num-buckets`, `--bucket-size`, `--n-trials`, `--lwe-dim`
//! (backend-dependent default). See `helpers::parse_cli` for the rest.
//!
//! **Design rationale:** Both error paths are on the latency tail of a
//! well-behaved deployment — a stale query during a mutation, a
//! resize-needed warning at saturation. Backend-generic so we can
//! confirm the rejection paths are O(1) for both backends.
//!
//! **Output:** `results/ikpir_failure_modes.csv`
//! Columns: backend, failure_kind, arity, num_buckets, bucket_size,
//! n_trials, mean_us, min_us, max_us, stddev_us

mod helpers;

use helpers::{Backend, MakeStore};
use ikpir_client::IkpirClient;
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer,
    IncrementalPirBackend, IndexPirBackend, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{
    CuckooError, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const MAX_KICKS: u32 = 500;

const HEADER: &str =
    "backend,failure_kind,arity,num_buckets,bucket_size,n_trials,mean_us,min_us,max_us,stddev_us";

#[derive(clap::Parser)]
#[command(about = "Measure StaleEpoch and TableFull rejection-path latency.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    #[arg(long, default_value_t = 2_000)] num_trials: u32,
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    #[arg(long, default_value_t = 16)]     full_num_buckets: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    /// LWE dimension. Defaults to 1774 (Frodo) or 1024 (Simple) when omitted.
    #[arg(long)]                           lwe_dim: Option<u32>,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim.unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn time_stale_epoch<S, B>(
    cli: &Cli, _arity: u32, num_buckets: u32, make_config: &impl Fn() -> B::Config,
) -> helpers::Stats
where
    S: MakeStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
    B::Query:    Clone,
    B::Response: Clone,
{
    let (mut store, n_seeded) = helpers::populate_to_load::<S>(
        0.50, num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    store.set_max_kicks(MAX_KICKS);
    let mut server: IkpirServer<S, B> = IkpirServer::new(store, make_config());
    let mut client: IkpirClient<B> = IkpirClient::from_setup(server.setup());
    let stale_q = client.build_query(&0u32.to_le_bytes());

    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let new_k = n_seeded as u32;
    for (i, b) in value.iter_mut().enumerate() {
        *b = (new_k.wrapping_mul(31).wrapping_add(i as u32) & 0xFF) as u8;
    }
    let _ = server.insert(&new_k.to_le_bytes(), &value).expect("epoch advance insert");
    assert_eq!(server.epoch(), 1);

    let mut samples = Vec::with_capacity(cli.num_trials as usize);
    for _ in 0..cli.num_trials {
        let t = Instant::now();
        let res = server.answer(&stale_q);
        let us = t.elapsed().as_secs_f64() * 1e6;
        match res {
            Err(IkpirError::StaleEpoch { .. }) => {}
            Ok(_) => panic!("expected StaleEpoch, got Ok"),
            Err(e) => panic!("expected StaleEpoch, got {e:?}"),
        }
        samples.push(us);
    }
    helpers::compute_stats(&samples)
}

fn time_table_full<S, B>(
    cli: &Cli, make_config: &impl Fn() -> B::Config,
) -> helpers::Stats
where
    S: MakeStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
{
    let mut store = S::make_store(
        cli.full_num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    ).expect("make_store");
    store.set_max_kicks(MAX_KICKS);
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut k = 0u32;
    loop {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k.to_le_bytes(), &value) {
            Ok(()) => k += 1,
            Err(CuckooError::TableFull) => break,
            Err(e) => panic!("seed: {e:?}"),
        }
    }
    let mut server: IkpirServer<S, B> = IkpirServer::new(store, make_config());

    for (i, b) in value.iter_mut().enumerate() {
        *b = (k.wrapping_mul(31).wrapping_add(i as u32) & 0xFF) as u8;
    }
    let failing_key = k.to_le_bytes();

    let mut samples = Vec::with_capacity(cli.num_trials as usize);
    for _ in 0..cli.num_trials {
        let t = Instant::now();
        let res = server.insert(&failing_key, &value);
        let us = t.elapsed().as_secs_f64() * 1e6;
        match res {
            Err(IkpirError::TableFull) => {}
            Ok(_) => panic!("expected TableFull, got Ok"),
            Err(e) => panic!("expected TableFull, got {e:?}"),
        }
        samples.push(us);
    }
    helpers::compute_stats(&samples)
}

fn run<S, B>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    make_config: impl Fn() -> B::Config,
)
where
    S: MakeStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
    B::Query:    Clone,
    B::Response: Clone,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    println!("=== failure_modes ===");
    println!(
        "Parameters: backend={}{}, arity={arity}{}, num_buckets={num_buckets}{}, full_num_buckets={}, \
         bucket_size={}, fingerprint_bits={}, value_bits={}, num_trials={}, lwe_dim={}",
        cli.backend,
        if matches.value_source("backend") != Some(ValueSource::CommandLine) { " (default)" } else { "" },
        if matches.value_source("arity") != Some(ValueSource::CommandLine) { " (default)" } else { "" },
        if matches.value_source("num_buckets") != Some(ValueSource::CommandLine) { " (default)" } else { "" },
        cli.full_num_buckets,
        cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.num_trials,
        effective_lwe_dim(cli),
    );

    let s_se = time_stale_epoch::<S, B>(cli, arity, num_buckets, &make_config);
    writeln!(csv, "{},stale_epoch,{arity},{num_buckets},{},{},{:.3},{:.3},{:.3},{:.3}",
             cli.backend, cli.bucket_size, cli.num_trials, s_se.mean, s_se.min, s_se.max, s_se.stddev).unwrap();
    println!("  backend={} stale_epoch arity={arity} nb={num_buckets} | mean={:.3} us  stddev={:.3}",
             cli.backend, s_se.mean, s_se.stddev);

    let s_tf = time_table_full::<S, B>(cli, &make_config);
    writeln!(csv, "{},table_full,{arity},{},{},{},{:.3},{:.3},{:.3},{:.3}",
             cli.backend, cli.full_num_buckets, cli.bucket_size, cli.num_trials,
             s_tf.mean, s_tf.min, s_tf.max, s_tf.stddev).unwrap();
    println!("  backend={} table_full  arity={arity} nb={} | mean={:.3} us  stddev={:.3}",
             cli.backend, cli.full_num_buckets, s_tf.mean, s_tf.stddev);
}

fn dispatch_backend<S: MakeStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    let lwe_dim = effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo  => run::<S, FrodoPirBackend>(csv, cli, arity, num_buckets, || FrodoConfig::with_lwe_dim(lwe_dim)),
        Backend::Simple => run::<S, SimplePirBackend>(csv, cli, arity, num_buckets, || SimpleConfig::with_lwe_dim(lwe_dim)),
    }
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_failure_modes.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_failure_modes.csv");
}
