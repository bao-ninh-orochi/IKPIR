//! **Intent:** Measure client-side preprocessing throughput on the FrodoPIR
//! backend, separated into the two amortisation phases:
//!
//! - **Phase B** (`precompute_queries`): sample fresh `(s, e)` per slot and
//!   compute `b = A·s + e`. Independent of the database; the only material
//!   that stays valid across server-side mutations.
//! - **Phase C** (`precompute_decodes`): for every prepared slot, compute
//!   `c = sᵀ·H` against the current per-segment hint. Invalidated by
//!   server-side mutations unless `apply_delta` is allowed to re-patch the
//!   queued `c` values.
//!
//! In a static-database benchmark the paper reports both phases. In a
//! dynamic (incremental) benchmark we typically only run Phase B and let
//! `client_decode` materialise `c` on the fly per query, since Phase C
//! would otherwise be re-run after every mutation. The two columns here
//! make either choice quantifiable.
//!
//! **Method:** Build a server, populate it to ~80% load, materialise a
//! fresh client, then time `precompute_queries(batch)` followed by
//! `precompute_decodes()`. Each call is repeated `warmup + trials` times
//! with a fresh client per trial (precomputation mutates state).
//!
//! **Invocation:** `cargo bench --bench preprocess_throughput` runs a
//! single sensible default config. Pass `--sweep` for the full matrix or
//! `--num-buckets <N>` (etc.) for a specific config. Use `--arity <N>` to
//! pick 2, 3, or 4; with `--sweep` and no explicit `--arity`, all three
//! arities are swept.
//!
//! **Output:** `results/ikpir_client_preprocess_throughput.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, batch,
//! mean_phase_b_sps, stddev_phase_b_sps, mean_phase_c_sps, stddev_phase_c_sps
//! (sps = slots/sec)

mod helpers;

use helpers::MakeStore;
use ikpir_client::{FrodoConfig, FrodoPirBackend, IkpirClient};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{
    CuckooError,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

type Client = IkpirClient<FrodoPirBackend>;

const MAX_KICKS: u32 = 2_500;

#[derive(clap::Parser)]
#[command(about = "Measure ikpir-client preprocessing throughput (Phase B + Phase C, slots/sec).")]
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
    #[arg(long, default_value_t = 64)]   batch: u32,
    #[arg(long, default_value_t = 2)]    warmup: u32,
    #[arg(long, default_value_t = 5)]    trials: u32,
}

fn build<S>(cli: &Cli, num_buckets: u32, bucket_size: u32, value_bits: u32) -> Option<IkpirServer<S, FrodoPirBackend>>
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
            Err(e)                       => panic!("build: {e:?}"),
        }
    }
    Some(IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim)))
}

/// Returns (phase_b_sps, phase_c_sps) — slots/sec for each phase.
fn time_one_trial<S>(server: &IkpirServer<S, FrodoPirBackend>, batch: u32) -> (f64, f64)
where S: MakeStore {
    let mut client: Client = Client::from_setup(server.setup());

    let t_b = Instant::now();
    client.precompute_queries(batch);
    let phase_b_sps = batch as f64 / t_b.elapsed().as_secs_f64();

    let t_c = Instant::now();
    client.precompute_decodes();
    let phase_c_sps = batch as f64 / t_c.elapsed().as_secs_f64();

    (phase_b_sps, phase_c_sps)
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
    let server = match build::<S>(cli, num_buckets, bucket_size, value_bits) {
        Some(s) => s,
        None    => {
            eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}");
            return;
        }
    };

    for _ in 0..cli.warmup { let _ = time_one_trial(&server, cli.batch); }

    let mut b_samples = Vec::with_capacity(cli.trials as usize);
    let mut c_samples = Vec::with_capacity(cli.trials as usize);
    for _ in 0..cli.trials {
        let (b, c) = time_one_trial(&server, cli.batch);
        b_samples.push(b);
        c_samples.push(c);
    }
    let bs = helpers::compute_stats(&b_samples);
    let cs = helpers::compute_stats(&c_samples);

    writeln!(
        csv,
        "{arity},{num_buckets},{bucket_size},{value_bits},{},{:.2},{:.2},{:.2},{:.2}",
        cli.batch, bs.mean, bs.stddev, cs.mean, cs.stddev,
    ).unwrap();
    println!(
        "  arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} | \
         B={:.2} sps (±{:.2})  C={:.2} sps (±{:.2})",
        bs.mean, bs.stddev, cs.mean, cs.stddev,
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
        "ikpir_client_preprocess_throughput.csv",
        "arity,num_buckets,bucket_size,value_bits,batch,mean_phase_b_sps,stddev_phase_b_sps,mean_phase_c_sps,stddev_phase_c_sps",
    );

    println!("=== ikpir-client preprocessing throughput (FrodoPirBackend) ===");
    println!(
        "Config: fingerprint_bits={}, plaintext_bits={}, lwe_dim={}, warmup={}, trials={}, batch={}",
        cli.fingerprint_bits, cli.plaintext_bits, cli.lwe_dim, cli.warmup, cli.trials, cli.batch,
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
    println!("\nResults written to results/ikpir_client_preprocess_throughput.csv");
}
