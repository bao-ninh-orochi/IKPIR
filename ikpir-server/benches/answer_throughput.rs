//! **Intent:** Measure server-side `answer` throughput on the FrodoPIR
//! backend.
//!
//! **Method:** Populate to `TableFull`, build a same-process client,
//! pre-build `batch` queries, then time individual `server.answer` calls
//! via criterion (cycling through the pre-built queries). The criterion
//! HTML/JSON report lands in `target/criterion/answer_throughput/`.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--num-buckets`,
//! `--bucket-size`, `--value-bits`, `--batch` (pre-built query batch
//! size). See `helpers::parse_cli` for defaults (academic-scale per
//! arity).
//!
//! **Design rationale:** `answer` is the per-query server hot path —
//! its matvec dominates IKPIR server CPU at steady state. Pre-building
//! the query batch lets criterion's `iter_custom` exclude
//! query-construction cost from the timed window, so we measure only
//! the `B::server_answer` matvec.
//!
//! **Output:** `results/ikpir_server_answer_throughput.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, batch,
//! mean_qps, min_qps, max_qps, stddev_qps, query_bytes, response_bytes

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer, PirQueryBundle};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;

type Client = IkpirClient<FrodoPirBackend>;
type Query  = PirQueryBundle<FrodoPirBackend>;

const HEADER: &str =
    "arity,num_buckets,bucket_size,value_bits,batch,mean_qps,min_qps,max_qps,stddev_qps,query_bytes,response_bytes";

#[derive(clap::Parser)]
#[command(about = "Measure ikpir-server answer throughput (queries/sec) via criterion.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)]   lwe_dim: u32,
    #[arg(long, default_value_t = 64)]     batch: u32,
}

fn run_one<S: MakeStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    let (store, n_inserted) = helpers::populate_until_full::<S>(
        num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let mut client: Client = Client::from_setup(server.setup());

    let queries: Vec<Query> = (0..cli.batch)
        .map(|i| client.build_query(&((i % n_inserted as u32).to_le_bytes())))
        .collect();

    let query_bytes    = queries[0].wire_byte_size();
    let response_bytes = server.answer(&queries[0]).expect("answer ok").wire_byte_size();

    // Geometry / preamble
    let params = server.params();
    let cps    = params.cells_per_slot();
    let bundle = server.setup();
    let store_state = helpers::StoreState {
        capacity:      (num_buckets as u64) * (cli.bucket_size as u64),
        populated:     n_inserted,
        load_pct:      100.0 * n_inserted as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width:     cli.bucket_size * cps,
        segment_rows:  params.segment_size(),
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       FrodoPirBackend::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:       bundle.wire_byte_size(),
        query_bytes,
        response_bytes,
        hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob { name: "arity",           value: arity.to_string(),            is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",      value: num_buckets.to_string(),      is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",      value: cli.bucket_size.to_string(),  is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "fingerprint_bits", value: cli.fingerprint_bits.to_string(), is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",       value: cli.value_bits.to_string(),   is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "plaintext_bits",   value: cli.plaintext_bits.to_string(), is_default: matches.value_source("plaintext_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",          value: cli.lwe_dim.to_string(),      is_default: matches.value_source("lwe_dim") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "batch",            value: cli.batch.to_string(),        is_default: matches.value_source("batch") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("answer_throughput", &knobs, &store_state, &geom);

    let mut idx = 0usize;
    let crit = helpers::run_criterion_throughput("answer_throughput", cli.batch as u64, || {
        let _ = server.answer(&queries[idx]).expect("answer ok");
        idx = (idx + 1) % queries.len();
    });

    writeln!(
        csv,
        "{arity},{num_buckets},{},{},{},{:.2},{:.2},{:.2},{:.2},{query_bytes},{response_bytes}",
        cli.bucket_size, cli.value_bits, cli.batch,
        crit.mean_ops_per_s, crit.min_ops_per_s, crit.max_ops_per_s, crit.stddev_ops_per_s,
    ).unwrap();
    println!(
        "  arity={arity} num_buckets={num_buckets:<6} bs={} vb={:<4} | \
         mean={:.2} qps (±{:.2}) | q={query_bytes}B r={response_bytes}B",
        cli.bucket_size, cli.value_bits, crit.mean_ops_per_s, crit.stddev_ops_per_s,
    );
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_server_answer_throughput.csv", HEADER);

    match cli.arity {
        2 => run_one::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run_one::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run_one::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_answer_throughput.csv");
}
