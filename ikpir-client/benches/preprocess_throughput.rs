//! **Intent:** Measure client-side preprocessing throughput across both
//! shipped backends, separated into the two amortisation phases (B and C).
//!
//! **Arguments (CLI):** `--arity`, `--backend` (frodo|simple, default
//! frodo), `--num-buckets`, `--bucket-size`, `--value-bits`, `--lwe-dim`
//! (backend-dependent default), `--batch`. See `helpers::parse_cli` for
//! the rest.
//!
//! **Output:** `results/ikpir_client_preprocess_throughput.csv`
//! Columns: backend, arity, num_buckets, bucket_size, value_bits, batch,
//! mean_phase_b_sps, stddev_phase_b_sps, mean_phase_c_sps, stddev_phase_c_sps

mod helpers;

use helpers::{Backend, MakeStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirClient, IncrementalPirBackend,
    IndexPirBackend, PrecomputingPirBackend, SimpleConfig, SimplePirBackend,
};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;

const HEADER: &str = "backend,arity,num_buckets,bucket_size,value_bits,batch,\
    mean_phase_b_sps,stddev_phase_b_sps,mean_phase_c_sps,stddev_phase_c_sps";

#[derive(Clone, clap::Parser)]
#[command(about = "Measure ikpir-client preprocessing throughput (Phase B + Phase C) via criterion.")]
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
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim.unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
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
    let bundle = server.setup();

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
        helpers::Knob { name: "batch",             value: cli.batch.to_string(),           is_default: matches.value_source("batch") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("preprocess_throughput", &knobs, &store_state, &geom);

    let batch = cli.batch;
    let phase_b = helpers::run_criterion_throughput_batched(
        "preprocess_phase_b",
        batch as u64,
        || IkpirClient::<B>::from_setup(bundle.clone()),
        |client| { client.precompute_queries(batch); },
    );

    let phase_c = helpers::run_criterion_throughput_batched(
        "preprocess_phase_c",
        batch as u64,
        || {
            let mut c: IkpirClient<B> = IkpirClient::from_setup(bundle.clone());
            c.precompute_queries(batch);
            c
        },
        |client| { client.precompute_decodes(); },
    );

    let b_mean = phase_b.mean_ops_per_s * batch as f64;
    let b_sd   = phase_b.stddev_ops_per_s * batch as f64;
    let c_mean = phase_c.mean_ops_per_s * batch as f64;
    let c_sd   = phase_c.stddev_ops_per_s * batch as f64;
    writeln!(
        csv,
        "{},{arity},{num_buckets},{},{},{batch},{:.2},{:.2},{:.2},{:.2}",
        cli.backend, cli.bucket_size, cli.value_bits,
        b_mean, b_sd, c_mean, c_sd,
    ).unwrap();
    println!(
        "  backend={} arity={arity} nb={num_buckets:<6} bs={} vb={:<4} | \
         B={:.2} sps (±{:.2})  C={:.2} sps (±{:.2})",
        cli.backend, cli.bucket_size, cli.value_bits, b_mean, b_sd, c_mean, c_sd,
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

    let mut csv = helpers::csv_writer("ikpir_client_preprocess_throughput.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_preprocess_throughput.csv");
}
