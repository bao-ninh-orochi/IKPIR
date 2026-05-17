//! **Intent:** Measure client-side `IkpirClient::from_setup` wall-clock
//! cost across both shipped backends.
//!
//! **Arguments (CLI):** `--arity`, `--backend` (frodo|simple, default
//! frodo), `--num-buckets`, `--bucket-size`, `--value-bits`, `--lwe-dim`
//! (backend-dependent default), `--trials`, `--with-precompute`,
//! `--batch`. See `helpers::parse_cli` for the rest.
//!
//! **Output:** `results/ikpir_client_setup_latency.csv`
//! Columns: backend, arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! with_precompute, batch, mean_setup_ms, min_setup_ms, max_setup_ms,
//! stddev_setup_ms, mean_precompute_b_ms, mean_precompute_c_ms

mod helpers;

use helpers::{Backend, MakeStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirClient, IncrementalPirBackend,
    IndexPirBackend, PrecomputingPirBackend, ServerSetupBundle, SimpleConfig, SimplePirBackend,
};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "backend,arity,num_buckets,bucket_size,value_bits,lwe_dim,\
    with_precompute,batch,mean_setup_ms,min_setup_ms,max_setup_ms,stddev_setup_ms,\
    mean_precompute_b_ms,mean_precompute_c_ms";

#[derive(Clone, clap::Parser)]
#[command(about = "Measure ikpir-client from_setup wall-clock cost.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    /// LWE dimension. Defaults to 1774 (Frodo) or 1024 (Simple) when omitted.
    #[arg(long)]                           lwe_dim: Option<u32>,
    #[arg(long, default_value_t = 64)]     batch: u32,
    #[arg(long, default_value_t = 2)]      warmup: u32,
    #[arg(long, default_value_t = 5)]      trials: u32,
    #[arg(long)]                           with_precompute: bool,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim.unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn time_one<B>(bundle: &ServerSetupBundle<B>, cli: &Cli) -> (f64, f64, f64)
where
    B: IndexPirBackend + IncrementalPirBackend + PrecomputingPirBackend + Clone,
{
    let local = bundle.clone();
    let t0 = Instant::now();
    let mut client: IkpirClient<B> = IkpirClient::from_setup(local);
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

fn run_one<S, B>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    backend_config: B::Config,
)
where
    S: MakeStore,
    B: IndexPirBackend + IncrementalPirBackend + PrecomputingPirBackend + BackendWireSize + Clone,
    B::Query: Clone, B::Response: Clone,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    let (store, n_inserted) = helpers::populate_until_full::<S>(
        num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    if n_inserted == 0 { eprintln!("  Skip: empty store"); return; }

    let server: IkpirServer<S, B> = IkpirServer::new(store, backend_config);
    let bundle: ServerSetupBundle<B> = server.setup();

    let params_store = server.params();
    let cps = params_store.cells_per_slot();
    let lwe_dim_eff = effective_lwe_dim(cli);
    let store_state = helpers::StoreState {
        capacity:       (num_buckets as u64) * (cli.bucket_size as u64),
        populated:      n_inserted,
        load_pct:       100.0 * n_inserted as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width:      cli.bucket_size * cps,
        segment_rows:   params_store.segment_size(),
    };
    let mut probe_client: IkpirClient<B> = IkpirClient::from_setup(bundle.clone());
    let q0 = probe_client.build_query(&0u32.to_le_bytes());
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       B::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:       bundle.wire_byte_size(),
        query_bytes:              q0.wire_byte_size(),
        response_bytes:           server.answer(&q0).expect("answer ok").wire_byte_size(),
        hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob { name: "backend",           value: cli.backend.to_string(),         is_default: matches.value_source("backend") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "arity",             value: arity.to_string(),               is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",       value: num_buckets.to_string(),         is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",       value: cli.bucket_size.to_string(),     is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "fingerprint_bits",  value: cli.fingerprint_bits.to_string(),is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",        value: cli.value_bits.to_string(),      is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "plaintext_bits",    value: cli.plaintext_bits.to_string(),  is_default: matches.value_source("plaintext_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",           value: lwe_dim_eff.to_string(),         is_default: cli.lwe_dim.is_none() },
        helpers::Knob { name: "with_precompute",   value: cli.with_precompute.to_string(), is_default: matches.value_source("with_precompute") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "batch",             value: cli.batch.to_string(),           is_default: matches.value_source("batch") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("client_setup_latency", &knobs, &store_state, &geom);

    for _ in 0..cli.warmup { let _ = time_one::<B>(&bundle, cli); }

    let mut setup_samples = Vec::with_capacity(cli.trials as usize);
    let mut pb_samples    = Vec::with_capacity(cli.trials as usize);
    let mut pc_samples    = Vec::with_capacity(cli.trials as usize);
    for _ in 0..cli.trials {
        let (s, pb, pc) = time_one::<B>(&bundle, cli);
        setup_samples.push(s);
        pb_samples.push(pb);
        pc_samples.push(pc);
    }
    let s = helpers::compute_stats(&setup_samples);
    let pb_mean = pb_samples.iter().sum::<f64>() / pb_samples.len() as f64;
    let pc_mean = pc_samples.iter().sum::<f64>() / pc_samples.len() as f64;

    let with_pre = if cli.with_precompute { 1 } else { 0 };
    writeln!(
        csv,
        "{},{arity},{num_buckets},{},{},{},{with_pre},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        cli.backend, cli.bucket_size, cli.value_bits, lwe_dim_eff, cli.batch,
        s.mean, s.min, s.max, s.stddev, pb_mean, pc_mean,
    ).unwrap();
    println!(
        "  backend={} arity={arity} nb={num_buckets:<6} bs={} vb={:<4} | \
         setup={:.3}ms (±{:.3}) pb={pb_mean:.3}ms pc={pc_mean:.3}ms",
        cli.backend, cli.bucket_size, cli.value_bits, s.mean, s.stddev,
    );
}

fn dispatch_backend<S: MakeStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    let lwe_dim = effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo  => run_one::<S, FrodoPirBackend>(csv, cli, arity, num_buckets, FrodoConfig::with_lwe_dim(lwe_dim)),
        Backend::Simple => run_one::<S, SimplePirBackend>(csv, cli, arity, num_buckets, SimpleConfig::with_lwe_dim(lwe_dim)),
    }
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_client_setup_latency.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_setup_latency.csv");
}
