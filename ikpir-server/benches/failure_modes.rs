//! **Intent:** Measure rejection-path latency for `StaleEpoch` and `TableFull`.
//!
//! `stale_epoch`: build a populated server, build a client query at epoch 0,
//! mutate the server once (epoch → 1), then call `answer` with the stale
//! query repeatedly. `table_full`: fill a tiny table to capacity, then
//! repeatedly attempt to insert one more key (each attempt exhausts the kick
//! budget deterministically).
//!
//! **Output:** `results/ikpir_failure_modes.csv`
//! Columns: failure_kind, arity, num_buckets, bucket_size, n_trials,
//! mean_us, min_us, max_us, stddev_us

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer};
use segmented_cuckoo::{
    CuckooError, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const MAX_KICKS: u32 = 500;

const HEADER: &str =
    "failure_kind,arity,num_buckets,bucket_size,n_trials,mean_us,min_us,max_us,stddev_us";

#[derive(clap::Parser)]
#[command(about = "Measure StaleEpoch and TableFull rejection-path latency.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    /// Per-failure-kind trial count.
    #[arg(long, default_value_t = 2_000)] num_trials: u32,
    /// Stale-epoch bench: num_buckets for the test server.
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    /// Table-full bench: tiny table that fills fast.
    #[arg(long, default_value_t = 16)]     full_num_buckets: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)]   lwe_dim: u32,
}

fn time_stale_epoch<S: MakeStore>(cli: &Cli, arity: u32, num_buckets: u32) -> helpers::Stats {
    let (mut store, n_seeded) = helpers::populate_to_load::<S>(
        0.50, num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    store.set_max_kicks(MAX_KICKS);
    let mut server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let mut client: IkpirClient<FrodoPirBackend> = IkpirClient::from_setup(server.setup());
    let stale_q = client.build_query(&0u32.to_le_bytes());

    // Advance epoch by 1.
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let new_k = n_seeded as u32;
    for (i, b) in value.iter_mut().enumerate() {
        *b = (new_k.wrapping_mul(31).wrapping_add(i as u32) & 0xFF) as u8;
    }
    let _ = server.insert(&new_k.to_le_bytes(), &value).expect("epoch advance insert");
    assert_eq!(server.epoch(), 1);
    let _ = arity; // used in dispatch only

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

fn time_table_full<S: MakeStore>(cli: &Cli) -> helpers::Stats {
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
    let mut server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));

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

fn run<S: MakeStore>(csv: &mut std::io::BufWriter<std::fs::File>, cli: &Cli, arity: u32, num_buckets: u32) {
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    // Print a minimal preamble (no store populate needed for preamble).
    println!("=== failure_modes ===");
    println!(
        "Parameters: arity={arity}{}, num_buckets={num_buckets}{}, full_num_buckets={}, \
         bucket_size={}, fingerprint_bits={}, value_bits={}, num_trials={}",
        if matches.value_source("arity") != Some(ValueSource::CommandLine) { " (default)" } else { "" },
        if matches.value_source("num_buckets") != Some(ValueSource::CommandLine) { " (default)" } else { "" },
        cli.full_num_buckets,
        cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.num_trials,
    );

    let s_se = time_stale_epoch::<S>(cli, arity, num_buckets);
    writeln!(csv, "stale_epoch,{arity},{num_buckets},{},{},{:.3},{:.3},{:.3},{:.3}",
             cli.bucket_size, cli.num_trials, s_se.mean, s_se.min, s_se.max, s_se.stddev).unwrap();
    println!("  stale_epoch arity={arity} nb={num_buckets} | mean={:.3} us  stddev={:.3}",
             s_se.mean, s_se.stddev);

    let s_tf = time_table_full::<S>(cli);
    writeln!(csv, "table_full,{arity},{},{},{},{:.3},{:.3},{:.3},{:.3}",
             cli.full_num_buckets, cli.bucket_size, cli.num_trials,
             s_tf.mean, s_tf.min, s_tf.max, s_tf.stddev).unwrap();
    println!("  table_full  arity={arity} nb={} | mean={:.3} us  stddev={:.3}",
             cli.full_num_buckets, s_tf.mean, s_tf.stddev);
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
        2 => run::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_failure_modes.csv");
}
