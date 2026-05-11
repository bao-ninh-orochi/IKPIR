//! **Intent:** Measure server-side setup cost on the FrodoPIR backend.
//!
//! **Method:** For each `(arity, num_buckets, bucket_size, value_bits)` config,
//! build and populate a Segmented{2,3,4}-ary KV store, wrap it in
//! [`IkpirServer::new`], and time the wall-clock cost. The `IkpirServer::new`
//! call internally runs `B::server_setup` once per segment (computing
//! `H = Aᵀ · D`), so the timing isolates setup preprocessing.
//!
//! **Invocation:** `cargo bench --bench setup_throughput` runs a single
//! sensible default config. Pass `--sweep` for the full matrix or
//! `--num-buckets <N>` (etc.) for a specific config. Use `--arity <N>` to
//! pick 2, 3, or 4; with `--sweep` and no explicit `--arity`, all three
//! arities are swept.
//!
//! **Output:** `results/ikpir_server_setup_throughput.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! mean_setup_ms, min_setup_ms, max_setup_ms, stddev_setup_ms,
//! setup_bundle_bytes, hint_bytes_per_segment, server_params_bytes_per_segment
//!
//! Wire-size columns are deterministic given the params; sampled once per
//! config row.

mod helpers;

use helpers::MakeStore;
use ikpir_server::{BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer};
use segmented_cuckoo::{
    CuckooError, CuckooKVStore,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const MAX_KICKS: u32 = 2_500;

#[derive(clap::Parser)]
#[command(about = "Measure ikpir-server setup wall-clock cost.")]
struct Cli {
    /// Run the full hardcoded matrix
    /// (num_buckets per arity × bucket_size ∈ {2,4} × value_bits ∈ {8,64,256}).
    #[arg(long)]
    sweep: bool,

    /// Cuckoo arity (2, 3, or 4). With --sweep and no --arity, sweep all three.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))]
    arity: Option<u32>,

    #[arg(long, default_value_t = 256)]  num_buckets: u32,
    #[arg(long, default_value_t = 4)]    bucket_size: u32,
    #[arg(long, default_value_t = 64)]   value_bits: u32,
    #[arg(long, default_value_t = 12)]   fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]    plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)] lwe_dim: u32,
    #[arg(long, default_value_t = 2)]    warmup: u32,
    #[arg(long, default_value_t = 5)]    trials: u32,
}

fn populate_store<S>(
    cli:         &Cli,
    num_buckets: u32,
    bucket_size: u32,
    value_bits:  u32,
) -> Option<CuckooKVStore<S>>
where S: MakeStore {
    let mut store = match S::make_store(
        num_buckets, bucket_size, cli.fingerprint_bits, value_bits, cli.plaintext_bits,
    ) {
        Ok(s)  => s,
        Err(_) => return None,
    };
    store.set_max_kicks(MAX_KICKS);

    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];

    let target_load: u32 = ((num_buckets * bucket_size) as f64 * 0.85) as u32;
    for k in 0u32..target_load {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k.to_le_bytes(), &value) {
            Ok(())                       => {}
            Err(CuckooError::TableFull)  => break,
            Err(e)                       => panic!("populate_store: {e:?}"),
        }
    }
    Some(store)
}

fn time_setup<S>(
    cli: &Cli,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
) -> Option<(f64, usize, usize, usize)>
where S: MakeStore {
    let store = populate_store::<S>(cli, num_buckets, bucket_size, value_bits)?;
    let start = Instant::now();
    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let bundle       = server.setup();
    let bundle_bytes = bundle.wire_byte_size();
    let hint_bytes   = FrodoPirBackend::hint_byte_size(&bundle.hints[0]);
    let sp_bytes     = FrodoPirBackend::server_params_byte_size(&bundle.backend_params[0]);
    Some((ms, bundle_bytes, hint_bytes, sp_bytes))
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
    let mut warmup_ok = true;
    for _ in 0..cli.warmup {
        if time_setup::<S>(cli, num_buckets, bucket_size, value_bits).is_none() {
            warmup_ok = false;
            break;
        }
    }
    if !warmup_ok {
        eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}");
        return;
    }

    let mut samples = Vec::with_capacity(cli.trials as usize);
    let (mut bundle_bytes, mut hint_bytes, mut sp_bytes) = (0usize, 0usize, 0usize);
    for _ in 0..cli.trials {
        let (t, bb, hb, spb) = time_setup::<S>(cli, num_buckets, bucket_size, value_bits)
            .expect("setup should succeed after warmup");
        samples.push(t);
        bundle_bytes = bb;
        hint_bytes   = hb;
        sp_bytes     = spb;
    }
    let s = helpers::compute_stats(&samples);
    writeln!(
        csv,
        "{arity},{num_buckets},{bucket_size},{value_bits},{},{:.3},{:.3},{:.3},{:.3},{bundle_bytes},{hint_bytes},{sp_bytes}",
        cli.lwe_dim, s.mean, s.min, s.max, s.stddev,
    )
    .unwrap();
    println!(
        "  arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} | \
         mean={:.3} ms (±{:.3}) | bundle={bundle_bytes}B hint/seg={hint_bytes}B sp/seg={sp_bytes}B",
        s.mean, s.stddev,
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

/// Sweep matrix of `num_buckets` per arity (constraints: 2-ary `2^t`,
/// 3-ary `3·2^t`, 4-ary `2^t ≥ 4`). Sets are scaled to roughly comparable
/// segment sizes across arities.
fn sweep_buckets(arity: u32) -> &'static [u32] {
    match arity {
        2 | 4 => &[64, 256, 1024, 4096],
        3     => &[96, 384, 1536, 6144],
        _     => unreachable!(),
    }
}

/// For single-config mode: if user did not override `--num-buckets` (it's
/// at the clap default 256) and chose arity 3, substitute the per-arity
/// default 384 (which is `3·128`).
fn resolve_num_buckets(cli: &Cli, arity: u32) -> u32 {
    if cli.num_buckets == 256 && arity == 3 { 384 } else { cli.num_buckets }
}

fn main() {
    let cli: Cli = helpers::parse_cli();
    let mut csv = helpers::csv_writer(
        "ikpir_server_setup_throughput.csv",
        "arity,num_buckets,bucket_size,value_bits,lwe_dim,mean_setup_ms,min_setup_ms,max_setup_ms,stddev_setup_ms,setup_bundle_bytes,hint_bytes_per_segment,server_params_bytes_per_segment",
    );

    println!("=== ikpir-server setup throughput (FrodoPirBackend) ===");
    println!(
        "Config: fingerprint_bits={}, plaintext_bits={}, lwe_dim={}, warmup={}, trials={}",
        cli.fingerprint_bits, cli.plaintext_bits, cli.lwe_dim, cli.warmup, cli.trials,
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
    println!("\nResults written to results/ikpir_server_setup_throughput.csv");
}
