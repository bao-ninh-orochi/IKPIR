//! **Intent:** Measure client-side `IkpirClient::from_setup` wall-clock cost
//! on the FrodoPIR backend.
//!
//! **Method:** For each `(arity, num_buckets, bucket_size, value_bits)`
//! config, build and populate a server (excluded from timing) and snapshot
//! a `ServerSetupBundle`. Then time `IkpirClient::from_setup(bundle.clone())`
//! over `trials` repetitions. With `--with-precompute`, additionally times
//! `precompute_queries(batch)` and `precompute_decodes()` as separate
//! columns (Phase B / Phase C latency on a freshly-built client).
//!
//! **Invocation:** `cargo bench --bench client_setup_latency` runs a single
//! default config. Pass `--sweep` for the full matrix, `--with-precompute`
//! to also measure Phase B/C, or specific flags to pin a config.
//!
//! **Output:** `results/ikpir_client_setup_latency.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! with_precompute, batch, mean_setup_ms, min_setup_ms, max_setup_ms,
//! stddev_setup_ms, mean_precompute_b_ms, mean_precompute_c_ms

mod helpers;

use helpers::MakeStore;
use ikpir_client::{FrodoConfig, FrodoPirBackend, IkpirClient};
use ikpir_server::{IkpirServer, ServerSetupBundle};
use segmented_cuckoo::{
    CuckooError,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

type Client = IkpirClient<FrodoPirBackend>;
type Bundle = ServerSetupBundle<FrodoPirBackend>;

const MAX_KICKS: u32 = 2_500;

#[derive(clap::Parser)]
#[command(about = "Measure ikpir-client from_setup wall-clock cost.")]
struct Cli {
    /// Run the full hardcoded matrix.
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
    #[arg(long, default_value_t = 64)]   batch: u32,
    #[arg(long, default_value_t = 2)]    warmup: u32,
    #[arg(long, default_value_t = 5)]    trials: u32,

    /// Also time `precompute_queries(batch)` and `precompute_decodes()`
    /// immediately after `from_setup`. Columns mean_precompute_b_ms and
    /// mean_precompute_c_ms are 0 when this flag is absent.
    #[arg(long)]
    with_precompute: bool,
}

fn build_bundle<S>(
    cli:         &Cli,
    num_buckets: u32,
    bucket_size: u32,
    value_bits:  u32,
) -> Option<Bundle>
where S: MakeStore {
    let mut store = S::make_store(
        num_buckets, bucket_size, cli.fingerprint_bits, value_bits, cli.plaintext_bits,
    ).ok()?;
    store.set_max_kicks(MAX_KICKS);

    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let target_load: u32 = ((num_buckets * bucket_size) as f64 * 0.80) as u32;
    for k in 0u32..target_load {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k.to_le_bytes(), &value) {
            Ok(())                       => {}
            Err(CuckooError::TableFull)  => break,
            Err(e)                       => panic!("build_bundle: {e:?}"),
        }
    }
    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    Some(server.setup())
}

fn time_one(bundle: &Bundle, cli: &Cli) -> (f64, f64, f64) {
    // Clone the bundle so the timed `from_setup` matches a fresh-on-wire path.
    let local = bundle.clone();
    let t0 = Instant::now();
    let mut client: Client = Client::from_setup(local);
    let setup_ms = t0.elapsed().as_secs_f64() * 1e3;

    let (pb_ms, pc_ms) = if cli.with_precompute {
        let t1 = Instant::now();
        client.precompute_queries(cli.batch);
        let pb = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = Instant::now();
        client.precompute_decodes();
        let pc = t2.elapsed().as_secs_f64() * 1e3;
        (pb, pc)
    } else {
        (0.0, 0.0)
    };
    (setup_ms, pb_ms, pc_ms)
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
    let bundle = match build_bundle::<S>(cli, num_buckets, bucket_size, value_bits) {
        Some(b) => b,
        None => {
            eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}");
            return;
        }
    };

    for _ in 0..cli.warmup { let _ = time_one(&bundle, cli); }

    let mut setup_samples = Vec::with_capacity(cli.trials as usize);
    let mut pb_samples    = Vec::with_capacity(cli.trials as usize);
    let mut pc_samples    = Vec::with_capacity(cli.trials as usize);
    for _ in 0..cli.trials {
        let (s, pb, pc) = time_one(&bundle, cli);
        setup_samples.push(s);
        pb_samples.push(pb);
        pc_samples.push(pc);
    }
    let s  = helpers::compute_stats(&setup_samples);
    let pb_mean = pb_samples.iter().sum::<f64>() / pb_samples.len() as f64;
    let pc_mean = pc_samples.iter().sum::<f64>() / pc_samples.len() as f64;

    let with_pre = if cli.with_precompute { 1 } else { 0 };
    writeln!(
        csv,
        "{arity},{num_buckets},{bucket_size},{value_bits},{},{with_pre},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        cli.lwe_dim, cli.batch, s.mean, s.min, s.max, s.stddev, pb_mean, pc_mean,
    )
    .unwrap();
    println!(
        "  arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} | \
         setup={:.3}ms (±{:.3}) pb={pb_mean:.3}ms pc={pc_mean:.3}ms",
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
        "ikpir_client_setup_latency.csv",
        "arity,num_buckets,bucket_size,value_bits,lwe_dim,with_precompute,batch,mean_setup_ms,min_setup_ms,max_setup_ms,stddev_setup_ms,mean_precompute_b_ms,mean_precompute_c_ms",
    );

    println!("=== ikpir-client setup latency (FrodoPirBackend) ===");
    println!(
        "Config: fingerprint_bits={}, plaintext_bits={}, lwe_dim={}, with_precompute={}, batch={}, warmup={}, trials={}",
        cli.fingerprint_bits, cli.plaintext_bits, cli.lwe_dim, cli.with_precompute, cli.batch, cli.warmup, cli.trials,
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
    println!("\nResults written to results/ikpir_client_setup_latency.csv");
}
