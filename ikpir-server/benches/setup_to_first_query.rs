//! **Intent:** Wall-clock latency from "no client state" to "first decoded
//! answer", split into per-phase columns. Captures the cold-start cost a real
//! deployment experiences when bringing up a new client.
//!
//! **Method:** Pre-populate a server once per config (NOT timed — it's the
//! "data already exists" precondition). Snapshot the populated cell array
//! once, then per trial clone it via `from_cells` and time these phases
//! independently:
//!
//! 1. `IkpirServer::new`            → server_setup_ms
//! 2. `IkpirClient::from_setup`     → client_from_setup_ms
//! 3. `client.precompute_queries(1)` (warm-b / warm-bc) → precompute_b_ms
//! 4. `client.precompute_decodes()` (warm-bc only)      → precompute_c_ms
//! 5. `client.build_query(key)`     → build_query_ms
//! 6. `server.answer(&q)`           → answer_ms
//! 7. `client.decode(key, &resp)`   → decode_ms
//!
//! `total_ms` = sum of phases that ran.
//!
//! **Invocation:** `cargo bench --bench setup_to_first_query` runs a single
//! default config in `cold` mode. Pass `--sweep` for the full matrix.
//!
//! **Sweep semantics:** `--sweep` (or `--sweep --mode cold`) iterates all
//! three modes across the config matrix; `--sweep --mode warm-b` or
//! `--sweep --mode warm-bc` pins to a single mode.
//!
//! **Output:** `results/ikpir_setup_to_first_query.csv`
//! Columns: mode, arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! server_setup_ms, client_from_setup_ms, precompute_b_ms, precompute_c_ms,
//! build_query_ms, answer_ms, decode_ms, total_ms

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirServer};
use segmented_cuckoo::{
    CuckooError, CuckooKVStore, CuckooParams,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

type Client = IkpirClient<FrodoPirBackend>;
const MAX_KICKS: u32 = 2_500;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Mode { Cold, WarmB, WarmBc }
impl Mode {
    fn as_csv(self) -> &'static str {
        match self { Mode::Cold => "cold", Mode::WarmB => "warm-b", Mode::WarmBc => "warm-bc" }
    }
}

#[derive(Clone, clap::Parser)]
#[command(about = "Measure end-to-end setup-to-first-query latency.")]
struct Cli {
    #[arg(long)] sweep: bool,
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))] arity: Option<u32>,
    #[arg(long, value_enum, default_value_t = Mode::Cold)] mode: Mode,
    #[arg(long, default_value_t = 256)]  num_buckets: u32,
    #[arg(long, default_value_t = 4)]    bucket_size: u32,
    #[arg(long, default_value_t = 64)]   value_bits: u32,
    #[arg(long, default_value_t = 12)]   fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]    plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)] lwe_dim: u32,
    #[arg(long, default_value_t = 2)]    warmup: u32,
    #[arg(long, default_value_t = 5)]    trials: u32,
}

/// Per-scheme bridge for snapshot-clone. `CuckooKVStore::from_cells` has a
/// per-scheme impl, not a generic one; this trait routes back to the right
/// constructor.
trait CloneStore: MakeStore {
    fn clone_from_cells(
        cells:     Vec<u32>,
        params:    CuckooParams,
        num_items: u64,
    ) -> Result<CuckooKVStore<Self>, CuckooError>;
}
impl CloneStore for Segmented2aryScheme {
    fn clone_from_cells(cells: Vec<u32>, params: CuckooParams, num_items: u64) -> Result<CuckooKVStore<Self>, CuckooError> {
        CuckooKVStore::<Segmented2aryScheme>::from_cells(cells, params, num_items)
    }
}
impl CloneStore for Segmented3aryScheme {
    fn clone_from_cells(cells: Vec<u32>, params: CuckooParams, num_items: u64) -> Result<CuckooKVStore<Self>, CuckooError> {
        CuckooKVStore::<Segmented3aryScheme>::from_cells(cells, params, num_items)
    }
}
impl CloneStore for Segmented4aryScheme {
    fn clone_from_cells(cells: Vec<u32>, params: CuckooParams, num_items: u64) -> Result<CuckooKVStore<Self>, CuckooError> {
        CuckooKVStore::<Segmented4aryScheme>::from_cells(cells, params, num_items)
    }
}

fn build_seed_store<S>(
    cli: &Cli,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
) -> Option<(CuckooKVStore<S>, u32)>
where S: CloneStore {
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
            Err(e)                       => panic!("build_seed_store: {e:?}"),
        }
    }
    Some((store, inserted))
}

/// Phases summed in milliseconds. Phases that did not run are 0.0.
#[derive(Default, Clone, Copy)]
struct Phases {
    server_setup_ms:      f64,
    client_from_setup_ms: f64,
    precompute_b_ms:      f64,
    precompute_c_ms:      f64,
    build_query_ms:       f64,
    answer_ms:            f64,
    decode_ms:            f64,
}
impl Phases {
    fn total(&self) -> f64 {
        self.server_setup_ms
            + self.client_from_setup_ms
            + self.precompute_b_ms
            + self.precompute_c_ms
            + self.build_query_ms
            + self.answer_ms
            + self.decode_ms
    }
}

fn time_one<S>(
    cli:    &Cli,
    cells:  &[u32],
    params: CuckooParams,
    n_inserted: u32,
    mode:   Mode,
) -> Phases
where S: CloneStore {
    let store = S::clone_from_cells(cells.to_vec(), params, n_inserted as u64)
        .expect("clone_from_cells ok");

    let t1 = Instant::now();
    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let server_setup_ms = t1.elapsed().as_secs_f64() * 1e3;

    let bundle = server.setup(); // sub-microsecond clone; not timed
    let t2 = Instant::now();
    let mut client: Client = Client::from_setup(bundle);
    let client_from_setup_ms = t2.elapsed().as_secs_f64() * 1e3;

    let (precompute_b_ms, precompute_c_ms) = match mode {
        Mode::Cold => (0.0, 0.0),
        Mode::WarmB => {
            let t = Instant::now();
            client.precompute_queries(1);
            (t.elapsed().as_secs_f64() * 1e3, 0.0)
        }
        Mode::WarmBc => {
            let t = Instant::now();
            client.precompute_queries(1);
            let b = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            client.precompute_decodes();
            (b, t.elapsed().as_secs_f64() * 1e3)
        }
    };

    let key = 0u32.to_le_bytes();
    let t = Instant::now();
    let q = client.build_query(&key);
    let build_query_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let r = server.answer(&q).expect("answer ok");
    let answer_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let _ = client.decode(&key, &r).expect("decode ok");
    let decode_ms = t.elapsed().as_secs_f64() * 1e3;

    Phases {
        server_setup_ms, client_from_setup_ms,
        precompute_b_ms, precompute_c_ms,
        build_query_ms, answer_ms, decode_ms,
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
where S: CloneStore {
    let (seed_store, n_inserted) = match build_seed_store::<S>(cli, num_buckets, bucket_size, value_bits) {
        Some(s) => s,
        None => {
            eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}");
            return;
        }
    };
    if n_inserted == 0 { return; }
    let cells:  Vec<u32>   = seed_store.snapshot_cells();
    let params: CuckooParams = seed_store.params();

    for _ in 0..cli.warmup { let _ = time_one::<S>(cli, &cells, params, n_inserted, cli.mode); }

    let mut samples: Vec<Phases> = Vec::with_capacity(cli.trials as usize);
    for _ in 0..cli.trials {
        samples.push(time_one::<S>(cli, &cells, params, n_inserted, cli.mode));
    }
    let n = samples.len() as f64;
    let mean = |sel: fn(&Phases) -> f64| samples.iter().map(sel).sum::<f64>() / n;
    let mode_label = cli.mode.as_csv();
    let m_server   = mean(|p| p.server_setup_ms);
    let m_client   = mean(|p| p.client_from_setup_ms);
    let m_pb       = mean(|p| p.precompute_b_ms);
    let m_pc       = mean(|p| p.precompute_c_ms);
    let m_q        = mean(|p| p.build_query_ms);
    let m_a        = mean(|p| p.answer_ms);
    let m_d        = mean(|p| p.decode_ms);
    let m_total    = mean(|p| p.total());

    writeln!(
        csv,
        "{mode_label},{arity},{num_buckets},{bucket_size},{value_bits},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        cli.lwe_dim, m_server, m_client, m_pb, m_pc, m_q, m_a, m_d, m_total,
    )
    .unwrap();
    println!(
        "  mode={mode_label:<7} arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} | \
         total={m_total:.3}ms (server={m_server:.3} client={m_client:.3} pb={m_pb:.3} pc={m_pc:.3} q={m_q:.3} a={m_a:.3} d={m_d:.3})"
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
        "ikpir_setup_to_first_query.csv",
        "mode,arity,num_buckets,bucket_size,value_bits,lwe_dim,server_setup_ms,client_from_setup_ms,precompute_b_ms,precompute_c_ms,build_query_ms,answer_ms,decode_ms,total_ms",
    );

    println!("=== ikpir setup-to-first-query latency (FrodoPirBackend) ===");
    println!(
        "Config: mode={}, fingerprint_bits={}, plaintext_bits={}, lwe_dim={}, warmup={}, trials={}",
        cli.mode.as_csv(), cli.fingerprint_bits, cli.plaintext_bits, cli.lwe_dim, cli.warmup, cli.trials,
    );

    let arities: Vec<u32> = match (cli.sweep, cli.arity) {
        (true,  None)    => vec![2, 3, 4],
        (true,  Some(a)) => vec![a],
        (false, opt)     => vec![opt.unwrap_or(2)],
    };

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
    println!("\nResults written to results/ikpir_setup_to_first_query.csv");
}
