//! **Intent:** Measure rejection-path latency for the two adversarial-
//! relevant failure modes:
//!
//! - **`stale_epoch`** — `IkpirServer::answer` rejects a query whose
//!   announced epoch doesn't match the server's current epoch. This is
//!   the first line of defence against replay / cross-epoch queries; the
//!   rejection must be cheaper than answering, otherwise a stale-epoch
//!   flood is a viable DoS.
//!
//! - **`table_full`** — `IkpirServer::insert` rolls back the entire
//!   cuckoo kick chain on `TableFull` and leaves the store unchanged.
//!   The rollback cost is observable; this bench reports it as a lower
//!   bound for an attacker spamming inserts against a near-saturated
//!   table.
//!
//! **Method:**
//!   - `stale_epoch`: build a populated server, snapshot its setup,
//!     mutate it once (epoch → 1), then repeatedly call `answer` with a
//!     query built at epoch 0. Each call must error with `StaleEpoch`.
//!   - `table_full`: build a small store, fill it to capacity, then
//!     repeatedly attempt to insert one more key. Each attempt must error
//!     with `TableFull` after exhausting the kick budget.
//!
//! **Invocation:** `cargo bench --bench failure_modes` runs both kinds at
//! a default config. `--num-trials` controls per-kind iteration count.
//!
//! **Output:** `results/ikpir_failure_modes.csv`
//! Columns: `failure_kind, arity, num_buckets, bucket_size, n_trials,
//! mean_us, min_us, max_us, stddev_us`

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer};
use segmented_cuckoo::{
    CuckooError, Segmented2aryScheme,
};
use std::io::Write;
use std::time::Instant;

const MAX_KICKS: u32 = 500;

#[derive(clap::Parser)]
#[command(about = "Measure latency of StaleEpoch and TableFull rejection paths.")]
struct Cli {
    /// Per-failure-kind trial count (each trial is one rejection call).
    #[arg(long, default_value_t = 10_000)] num_trials: u32,
    /// Stale-epoch bench: number of buckets in the test server.
    #[arg(long, default_value_t = 256)]    num_buckets: u32,
    /// Both benches: bucket size.
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    /// Table-full bench: tiny table that fills fast (saturated quickly).
    #[arg(long, default_value_t = 16)]     full_num_buckets: u32,
    #[arg(long, default_value_t = 12)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 64)]     value_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)]   lwe_dim: u32,
}

fn time_stale_epoch(cli: &Cli) -> (u32, u32, helpers::Stats) {
    let arity = 2u32;
    // Populated server.
    let mut store = Segmented2aryScheme::make_store(
        cli.num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    ).expect("make_store");
    store.set_max_kicks(MAX_KICKS);
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let target_load = ((cli.num_buckets * cli.bucket_size) as f64 * 0.50) as u32;
    let mut seeded = 0u32;
    for k in 0..target_load {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        if store.insert(k.to_le_bytes(), &value).is_ok() {
            seeded = k + 1;
        }
    }
    let mut server: IkpirServer<Segmented2aryScheme, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    // Client built at epoch 0.
    let mut client: IkpirClient<FrodoPirBackend> = IkpirClient::from_setup(server.setup());
    // Build the query at the current epoch (0).
    let stale_q = client.build_query(&0u32.to_le_bytes());
    // Mutate the server so its epoch advances to 1.
    for (i, b) in value.iter_mut().enumerate() {
        *b = (seeded.wrapping_mul(31).wrapping_add(i as u32) & 0xFF) as u8;
    }
    let _ = server.insert(&seeded.to_le_bytes(), &value).expect("insert mutation");
    assert_eq!(server.epoch(), 1);

    let mut samples = Vec::with_capacity(cli.num_trials as usize);
    for _ in 0..cli.num_trials {
        let start = Instant::now();
        let res = server.answer(&stale_q);
        let us = start.elapsed().as_secs_f64() * 1e6;
        match res {
            Err(IkpirError::StaleEpoch { .. }) => {}
            Err(other) => panic!("expected StaleEpoch, got {other:?}"),
            Ok(_)      => panic!("expected StaleEpoch, got Ok"),
        }
        samples.push(us);
    }
    (arity, cli.num_buckets, helpers::compute_stats(&samples))
}

fn time_table_full(cli: &Cli) -> (u32, u32, helpers::Stats) {
    let arity = 2u32;
    let mut store = Segmented2aryScheme::make_store(
        cli.full_num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    ).expect("make_store");
    store.set_max_kicks(MAX_KICKS);
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    // Fill until first TableFull, then stop — we want a saturated table
    // where every subsequent insert deterministically rejects.
    let mut k = 0u32;
    loop {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k.to_le_bytes(), &value) {
            Ok(())                       => k += 1,
            Err(CuckooError::TableFull)  => break,
            Err(e)                       => panic!("seed: {e:?}"),
        }
    }
    let mut server: IkpirServer<Segmented2aryScheme, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));

    // Re-use the *same* failing key/value across all trials. The cuckoo
    // kick chain is deterministic in placement order, so the rolled-back
    // insert reproduces the same kick budget exhaustion each time — that
    // is the measurement we want. Picking a *new* probe per trial would
    // sometimes succeed (other keys may still fit), which is a different
    // experiment.
    for (i, b) in value.iter_mut().enumerate() {
        *b = (k.wrapping_mul(31).wrapping_add(i as u32) & 0xFF) as u8;
    }
    let failing_key = k.to_le_bytes();

    let mut samples = Vec::with_capacity(cli.num_trials as usize);
    for _ in 0..cli.num_trials {
        let start = Instant::now();
        let res = server.insert(&failing_key, &value);
        let us = start.elapsed().as_secs_f64() * 1e6;
        match res {
            Err(IkpirError::TableFull) => {}
            Err(other) => panic!("expected TableFull, got {other:?}"),
            Ok(_)      => panic!("expected TableFull, got Ok"),
        }
        samples.push(us);
    }
    (arity, cli.full_num_buckets, helpers::compute_stats(&samples))
}

fn main() {
    let cli: Cli = helpers::parse_cli();
    let mut csv = helpers::csv_writer(
        "ikpir_failure_modes.csv",
        "failure_kind,arity,num_buckets,bucket_size,n_trials,mean_us,min_us,max_us,stddev_us",
    );

    println!("=== ikpir failure-mode rejection latency (FrodoPirBackend) ===");
    println!(
        "Config: bucket_size={}, fingerprint_bits={}, value_bits={}, plaintext_bits={}, lwe_dim={}, num_trials={}",
        cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits, cli.lwe_dim, cli.num_trials,
    );

    let (arity, nb, s) = time_stale_epoch(&cli);
    writeln!(csv, "stale_epoch,{arity},{nb},{},{},{:.3},{:.3},{:.3},{:.3}",
             cli.bucket_size, cli.num_trials, s.mean, s.min, s.max, s.stddev).unwrap();
    println!("  stale_epoch arity={arity} nb={nb} | mean={:.3} us  min={:.3}  max={:.3}  stddev={:.3}",
             s.mean, s.min, s.max, s.stddev);

    let (arity, nb, s) = time_table_full(&cli);
    writeln!(csv, "table_full,{arity},{nb},{},{},{:.3},{:.3},{:.3},{:.3}",
             cli.bucket_size, cli.num_trials, s.mean, s.min, s.max, s.stddev).unwrap();
    println!("  table_full  arity={arity} nb={nb} | mean={:.3} us  min={:.3}  max={:.3}  stddev={:.3}",
             s.mean, s.min, s.max, s.stddev);

    println!("\nResults written to results/ikpir_failure_modes.csv");
}
