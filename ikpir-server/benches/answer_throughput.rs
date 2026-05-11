//! **Intent:** Measure server-side `answer` throughput on the FrodoPIR backend.
//!
//! **Method:** Build and populate a server, materialise a same-process client,
//! pre-build a batch of queries (so query-construction is excluded), then time
//! `batch` answers in a single loop. Repeat `trials` times.
//!
//! Throughput is reported as queries / second.
//!
//! **Invocation:** `cargo bench --bench answer_throughput` runs a single
//! sensible default config (one CSV row). Pass `--sweep` to run the full
//! hardcoded matrix; pass `--num-buckets <N>` (etc.) to pin a specific config.
//! Use `--arity <N>` to pick 2, 3, or 4; with `--sweep` and no explicit
//! `--arity`, all three arities are swept.
//!
//! **Output:** `results/ikpir_server_answer_throughput.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, batch,
//! mean_qps, min_qps, max_qps, stddev_qps, query_bytes, response_bytes
//!
//! Wire-size columns are deterministic given the params; sampled once per
//! config row from the pre-built query / answer pair.

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirServer, PirQueryBundle};
use segmented_cuckoo::{
    CuckooError,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

type Client = IkpirClient<FrodoPirBackend>;
type Query  = PirQueryBundle<FrodoPirBackend>;

const MAX_KICKS: u32 = 2_500;

#[derive(clap::Parser)]
#[command(about = "Measure ikpir-server answer throughput (queries/sec).")]
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
    #[arg(long, default_value_t = 32)]   batch: u32,
    #[arg(long, default_value_t = 2)]    warmup: u32,
    #[arg(long, default_value_t = 5)]    trials: u32,
}

fn build<S>(
    cli: &Cli,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
) -> Option<(IkpirServer<S, FrodoPirBackend>, Client, u32)>
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
            Err(e)                       => panic!("build: {e:?}"),
        }
    }
    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let client: Client = Client::from_setup(server.setup());
    Some((server, client, inserted))
}

fn pre_build_queries(client: &mut Client, n_inserted: u32, batch: u32) -> Vec<Query> {
    (0..batch)
        .map(|i| client.build_query(&(i % n_inserted).to_le_bytes()))
        .collect()
}

fn time_answer_batch<S>(server: &IkpirServer<S, FrodoPirBackend>, queries: &[Query]) -> f64
where S: MakeStore {
    let start = Instant::now();
    for q in queries {
        let _ = server.answer(q).expect("answer ok");
    }
    let elapsed_s = start.elapsed().as_secs_f64();
    queries.len() as f64 / elapsed_s
}

fn run_one<S>(
    csv:  &mut std::io::BufWriter<std::fs::File>,
    cli:  &Cli,
    arity: u32,
    num_buckets: u32,
    bucket_size: u32,
    value_bits:  u32,
)
where S: MakeStore {
    let (server, mut client, n_inserted) = match build::<S>(cli, num_buckets, bucket_size, value_bits) {
        Some(t) => t,
        None => {
            eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}: setup failed");
            return;
        }
    };
    if n_inserted == 0 { return; }

    let queries        = pre_build_queries(&mut client, n_inserted, cli.batch);
    let query_bytes    = queries[0].wire_byte_size();
    let response_bytes = server.answer(&queries[0]).expect("answer ok").wire_byte_size();
    for _ in 0..cli.warmup { let _ = time_answer_batch(&server, &queries); }

    let mut samples = Vec::with_capacity(cli.trials as usize);
    for _ in 0..cli.trials {
        samples.push(time_answer_batch(&server, &queries));
    }
    let s = helpers::compute_stats(&samples);
    writeln!(
        csv,
        "{arity},{num_buckets},{bucket_size},{value_bits},{},{:.2},{:.2},{:.2},{:.2},{query_bytes},{response_bytes}",
        cli.batch, s.mean, s.min, s.max, s.stddev,
    )
    .unwrap();
    println!(
        "  arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} | \
         mean={:.2} qps (±{:.2}) | q={query_bytes}B r={response_bytes}B",
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
        "ikpir_server_answer_throughput.csv",
        "arity,num_buckets,bucket_size,value_bits,batch,mean_qps,min_qps,max_qps,stddev_qps,query_bytes,response_bytes",
    );

    println!("=== ikpir-server answer throughput (FrodoPirBackend) ===");
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
    println!("\nResults written to results/ikpir_server_answer_throughput.csv");
}
