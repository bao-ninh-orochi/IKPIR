//! **Intent:** Measure client-side `build_query` throughput on the FrodoPIR
//! backend, in three preprocessing modes:
//!
//! - **`cold`** — no preprocessing; each `build_query` samples `(s, e)` and
//!   computes `b = A·s + e` inline. Cost ∝ `arity × n_rows × lwe_dim`.
//! - **`warm-b`** — `precompute_queries(batch)` runs once outside the
//!   timed loop; each `build_query` consumes one prepared slot per segment.
//!   Cost collapses to one vector add per segment (the `Δ·u_row` step).
//! - **`warm-bc`** — same as `warm-b` plus `precompute_decodes()`.
//!   `build_query` cost is identical to `warm-b` (Phase C only affects
//!   decode), but reported separately for parity with the decode bench.
//!
//! **Method:** Build a server, materialise a fresh client per trial,
//! optionally preprocess, then time `batch` `build_query` calls.
//!
//! **Invocation:** `cargo bench --bench query_throughput` runs a single
//! sensible default config in `cold` mode. Pass `--mode warm-b` or
//! `--mode warm-bc` for the precomputed paths, `--sweep` for the full
//! matrix, or `--num-buckets <N>` (etc.) for a specific config. Use
//! `--arity <N>` to pick 2, 3, or 4; with `--sweep` and no explicit
//! `--arity`, all three arities are swept.
//!
//! **Output:** `results/ikpir_client_query_throughput.csv`
//! Columns: mode, arity, num_buckets, bucket_size, value_bits, batch,
//! mean_qps, min_qps, max_qps, stddev_qps, query_bytes
//!
//! **Sweep semantics:** `--sweep` (or `--sweep --mode cold`) iterates all
//! three modes across the config matrix. `--sweep --mode warm-b` or
//! `--sweep --mode warm-bc` pins to a single mode while sweeping configs.
//! `query_bytes` is deterministic given the params; sampled once per row.

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

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Mode {
    /// No preprocessing — `build_query` samples `(s, e)` and computes b inline.
    Cold,
    /// Phase B warm — `precompute_queries(batch)` outside the timed loop.
    WarmB,
    /// Phase B + Phase C warm — also `precompute_decodes()` outside the loop.
    WarmBc,
}
impl Mode {
    fn as_csv(self) -> &'static str {
        match self { Mode::Cold => "cold", Mode::WarmB => "warm-b", Mode::WarmBc => "warm-bc" }
    }
}

#[derive(Clone, clap::Parser)]
#[command(about = "Measure ikpir-client build_query throughput (queries/sec).")]
struct Cli {
    /// Run the full hardcoded matrix
    /// (num_buckets per arity × bucket_size ∈ {2,4} × value_bits ∈ {8,64,256}).
    #[arg(long)]
    sweep: bool,

    /// Cuckoo arity (2, 3, or 4). With --sweep and no --arity, sweep all three.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))]
    arity: Option<u32>,

    /// Preprocessing mode: cold (default), warm-b, or warm-bc.
    #[arg(long, value_enum, default_value_t = Mode::Cold)]
    mode: Mode,

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

fn build_client<S>(cli: &Cli, num_buckets: u32, bucket_size: u32, value_bits: u32) -> Option<(Client, u32)>
where S: MakeStore {
    let mut store = S::make_store(
        num_buckets, bucket_size, cli.fingerprint_bits, value_bits, cli.plaintext_bits,
    ).ok()?;
    store.set_max_kicks(MAX_KICKS);

    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let target_load: u32 = ((num_buckets * bucket_size) as f64 * 0.80) as u32;
    let mut inserted = 0u32;
    for k in 0u32..target_load {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k.to_le_bytes(), &value) {
            Ok(())                       => inserted = k + 1,
            Err(CuckooError::TableFull)  => break,
            Err(e)                       => panic!("build_client: {e:?}"),
        }
    }
    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let client: Client = Client::from_setup(server.setup());
    Some((client, inserted))
}

fn time_query_batch(client: &mut Client, n_inserted: u32, batch: u32) -> f64 {
    let start = Instant::now();
    for i in 0..batch {
        let _ = client.build_query(&(i % n_inserted).to_le_bytes());
    }
    let elapsed_s = start.elapsed().as_secs_f64();
    batch as f64 / elapsed_s
}

/// Apply the configured preprocessing mode to a fresh client. `warm-b` and
/// `warm-bc` need enough prepared slots to cover both the timed batch and
/// any subsequent `cli.warmup` re-runs in this trial — overshoot generously.
fn warm_client(client: &mut Client, mode: Mode, batch: u32) {
    match mode {
        Mode::Cold => {}
        Mode::WarmB => {
            client.precompute_queries(batch);
        }
        Mode::WarmBc => {
            client.precompute_queries(batch);
            client.precompute_decodes();
        }
    }
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
    let mode_label = cli.mode.as_csv();
    let mut samples = Vec::with_capacity(cli.trials as usize);
    let mut query_bytes: usize = 0;

    // Per-trial fresh client — preprocessing mutates state, and `warm-*`
    // modes need a queue refill once it drains. Warmup runs share the
    // trial-local client.
    for trial in 0..(cli.warmup + cli.trials) {
        let (mut client, n_inserted) = match build_client::<S>(cli, num_buckets, bucket_size, value_bits) {
            Some(t) => t,
            None => {
                eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}");
                return;
            }
        };
        if n_inserted == 0 { return; }
        warm_client(&mut client, cli.mode, cli.batch);
        if trial == cli.warmup {
            // Sample wire size once on a representative query (deterministic
            // given params; the warm-up rounds already populated the queue).
            query_bytes = client.build_query(&0u32.to_le_bytes()).wire_byte_size();
            warm_client(&mut client, cli.mode, cli.batch);
        }
        let qps = time_query_batch(&mut client, n_inserted, cli.batch);
        if trial >= cli.warmup { samples.push(qps); }
    }
    let s = helpers::compute_stats(&samples);

    writeln!(
        csv,
        "{mode_label},{arity},{num_buckets},{bucket_size},{value_bits},{},{:.2},{:.2},{:.2},{:.2},{query_bytes}",
        cli.batch, s.mean, s.min, s.max, s.stddev,
    )
    .unwrap();
    println!(
        "  mode={mode_label:<7} arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} | \
         mean={:.2} qps (±{:.2}) | q={query_bytes}B",
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
        "ikpir_client_query_throughput.csv",
        "mode,arity,num_buckets,bucket_size,value_bits,batch,mean_qps,min_qps,max_qps,stddev_qps,query_bytes",
    );

    println!("=== ikpir-client query throughput (FrodoPirBackend) ===");
    println!(
        "Config: mode={}, fingerprint_bits={}, plaintext_bits={}, lwe_dim={}, warmup={}, trials={}, batch={}",
        cli.mode.as_csv(), cli.fingerprint_bits, cli.plaintext_bits, cli.lwe_dim, cli.warmup, cli.trials, cli.batch,
    );

    let arities: Vec<u32> = match (cli.sweep, cli.arity) {
        (true,  None)    => vec![2, 3, 4],
        (true,  Some(a)) => vec![a],
        (false, opt)     => vec![opt.unwrap_or(2)],
    };

    // Pragmatic mode-sweep semantics: `--sweep` (or `--sweep --mode cold`)
    // iterates all three modes; `--sweep --mode warm-b` and
    // `--sweep --mode warm-bc` pin a single mode (the cold default is
    // indistinguishable from "user didn't pass --mode" at the clap layer).
    let modes: Vec<Mode> = match (cli.sweep, cli.mode) {
        (true,  Mode::Cold)   => vec![Mode::Cold, Mode::WarmB, Mode::WarmBc],
        (true,  Mode::WarmB)  => vec![Mode::WarmB],
        (true,  Mode::WarmBc) => vec![Mode::WarmBc],
        (false, m)            => vec![m],
    };

    for &mode in &modes {
        let cli_local = Cli { mode, ..cli.clone() };
        for &arity in &arities {
            if cli_local.sweep {
                for &nb in sweep_buckets(arity) {
                    for &bs in &[2u32, 4] {
                        for &vb in &[8u32, 64, 256] {
                            dispatch(&mut csv, &cli_local, arity, nb, bs, vb);
                        }
                    }
                }
            } else {
                let nb = resolve_num_buckets(&cli_local, arity);
                dispatch(&mut csv, &cli_local, arity, nb, cli_local.bucket_size, cli_local.value_bits);
            }
        }
    }
    println!("\nResults written to results/ikpir_client_query_throughput.csv");
}
