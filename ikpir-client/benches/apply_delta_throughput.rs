//! **Intent:** Measure client-side `apply_delta` throughput on the FrodoPIR
//! backend, with and without a populated precomputation queue.
//!
//! Each delta unconditionally patches the per-segment hint matrix
//! (`O(arity × lwe_dim × touched_cells)`). When the client has prepared
//! slots whose `c = sᵀ·H` was filled by `precompute_decodes`, every such
//! slot also gets its `c` patched in lock-step
//! (`O(precomputed_slots × arity × (lwe_dim + touched_cells))`).
//!
//! **Method:** Build a populated server. For each config, materialise a
//! fresh client per trial, optionally call
//! `precompute_queries(precomputed_slots) + precompute_decodes()`, perform
//! `batch` sequential server inserts to collect `HintDeltaBundle`s, then
//! time the cost of folding all deltas into the client state.
//!
//! **Invocation:** `cargo bench --bench apply_delta_throughput` runs a
//! single sensible default config (`--precomputed-slots 0` matches the
//! pre-precomputation cost profile). Pass `--precomputed-slots N` to
//! measure the load-bearing case for the c-patch math. Pass `--sweep` for
//! the full matrix or `--num-buckets <N>` (etc.) for a specific config.
//! Use `--arity <N>` to pick 2, 3, or 4; with `--sweep` and no explicit
//! `--arity`, all three arities are swept.
//!
//! **Output:** `results/ikpir_client_apply_delta_throughput.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, batch,
//! precomputed_slots, mean_dps, min_dps, max_dps, stddev_dps  (deltas/sec)

mod helpers;

use helpers::MakeStore;
use ikpir_client::{FrodoConfig, FrodoPirBackend, HintDeltaBundle, IkpirClient};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{
    CuckooError,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

type Client = IkpirClient<FrodoPirBackend>;
type Delta  = HintDeltaBundle<FrodoPirBackend>;

const MAX_KICKS: u32 = 2_500;

#[derive(clap::Parser)]
#[command(about = "Measure ikpir-client apply_delta throughput (deltas/sec).")]
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

    /// Per-segment count of precomputed (s, b, c) slots present at the
    /// moment apply_delta runs. `0` (default) measures the hint-only patch
    /// cost — equivalent to the pre-preprocessing behaviour. Larger values
    /// surface the per-slot `c` patching overhead.
    #[arg(long, default_value_t = 0)]    precomputed_slots: u32,
}

fn build<S>(cli: &Cli, num_buckets: u32, bucket_size: u32, value_bits: u32) -> Option<IkpirServer<S, FrodoPirBackend>>
where S: MakeStore {
    let mut store = S::make_store(
        num_buckets, bucket_size, cli.fingerprint_bits, value_bits, cli.plaintext_bits,
    ).ok()?;
    store.set_max_kicks(MAX_KICKS);
    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];

    let n_seed: u32 = (num_buckets * bucket_size) / 2;
    for k in 0u32..n_seed {
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

fn collect_deltas<S>(server: &mut IkpirServer<S, FrodoPirBackend>, n_seed: u32, value_bits: u32, batch: u32) -> Vec<Delta>
where S: MakeStore {
    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut deltas = Vec::with_capacity(batch as usize);
    for k in n_seed..n_seed + batch {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(31).wrapping_add(i as u32) & 0xFF) as u8;
        }
        let d = server.insert(&k.to_le_bytes(), &value).expect("insert");
        deltas.push(d);
    }
    deltas
}

fn time_apply_batch(client: &mut Client, deltas: &[Delta]) -> f64 {
    let start = Instant::now();
    for d in deltas {
        client.apply_delta(d.clone()).expect("apply");
    }
    let elapsed_s = start.elapsed().as_secs_f64();
    deltas.len() as f64 / elapsed_s
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
    let n_seed: u32 = (num_buckets * bucket_size) / 2;
    let mut samples = Vec::with_capacity(cli.trials as usize);
    for trial in 0..(cli.warmup + cli.trials) {
        let mut server = match build::<S>(cli, num_buckets, bucket_size, value_bits) {
            Some(s) => s,
            None    => break,
        };
        let mut client: Client = Client::from_setup(server.setup());
        // Populate the precomputation queue *before* collecting deltas, so
        // each apply_delta call patches `precomputed_slots` queued c values
        // in addition to the hint matrix.
        if cli.precomputed_slots > 0 {
            client.precompute_queries(cli.precomputed_slots);
            client.precompute_decodes();
        }
        let deltas = collect_deltas(&mut server, n_seed, value_bits, cli.batch);
        let dps = time_apply_batch(&mut client, &deltas);
        if trial >= cli.warmup { samples.push(dps); }
    }
    if samples.is_empty() {
        eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}");
        return;
    }
    let s = helpers::compute_stats(&samples);
    let pre = cli.precomputed_slots;
    writeln!(
        csv,
        "{arity},{num_buckets},{bucket_size},{value_bits},{},{pre},{:.2},{:.2},{:.2},{:.2}",
        cli.batch, s.mean, s.min, s.max, s.stddev,
    )
    .unwrap();
    println!(
        "  arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} pre={pre:<4} | \
         mean={:.2} dps (±{:.2})",
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
        2 | 4 => &[256, 1024],
        3     => &[384, 1536],
        _     => unreachable!(),
    }
}

fn resolve_num_buckets(cli: &Cli, arity: u32) -> u32 {
    if cli.num_buckets == 256 && arity == 3 { 384 } else { cli.num_buckets }
}

fn main() {
    let cli: Cli = helpers::parse_cli();
    let mut csv = helpers::csv_writer(
        "ikpir_client_apply_delta_throughput.csv",
        "arity,num_buckets,bucket_size,value_bits,batch,precomputed_slots,mean_dps,min_dps,max_dps,stddev_dps",
    );

    println!("=== ikpir-client apply_delta throughput (FrodoPirBackend) ===");
    println!(
        "Config: fingerprint_bits={}, plaintext_bits={}, lwe_dim={}, warmup={}, trials={}, batch={}, precomputed_slots={}",
        cli.fingerprint_bits, cli.plaintext_bits, cli.lwe_dim, cli.warmup, cli.trials, cli.batch, cli.precomputed_slots,
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
    println!("\nResults written to results/ikpir_client_apply_delta_throughput.csv");
}
