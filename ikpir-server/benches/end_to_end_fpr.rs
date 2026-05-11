//! **Intent:** Measure the end-to-end false-positive rate of IKPIR.
//!
//! **Method:** Populate to `TableFull`, build a client, query `n_queried`
//! absent keys end-to-end, and count decoder matches.
//!
//! At fp_bits=32 + 100k queries, expected FPs ≈ 2.3×10⁻⁵ (≈0). The bench's
//! value is when the orchestrator sweeps `--fingerprint-bits`; the curve
//! flattens to zero past ~24 bits.
//!
//! **Output:** `results/ikpir_end_to_end_fpr.csv`
//! Columns: arity, num_buckets, bucket_size, fingerprint_bits, value_bits,
//! n_inserted, n_queried, n_false_positives, fpr

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;

const HEADER: &str =
    "arity,num_buckets,bucket_size,fingerprint_bits,value_bits,n_inserted,n_queried,n_false_positives,fpr";

#[derive(clap::Parser)]
#[command(about = "Measure end-to-end IKPIR false-positive rate on the FrodoPIR backend.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)]   lwe_dim: u32,
    /// Number of absent keys to query for FPR estimation.
    #[arg(long, default_value_t = 100_000)] n_queried: u32,
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
    if n_inserted == 0 { eprintln!("  Skip: empty store"); return; }

    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let mut client: IkpirClient<FrodoPirBackend> = IkpirClient::from_setup(server.setup());

    let params = server.params();
    let cps = params.cells_per_slot();
    let bundle = server.setup();
    let store_state = helpers::StoreState {
        capacity:      (num_buckets as u64) * (cli.bucket_size as u64),
        populated:     n_inserted,
        load_pct:      100.0 * n_inserted as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width:     cli.bucket_size * cps,
        segment_rows:  params.segment_size(),
    };
    let q0 = client.build_query(&0u64.to_le_bytes());
    let geom = helpers::Geometry {
        hint_per_seg_bytes:   FrodoPirBackend::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:   bundle.wire_byte_size(),
        query_bytes:          q0.wire_byte_size(),
        response_bytes:       server.answer(&q0).expect("answer ok").wire_byte_size(),
        hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob { name: "arity",            value: arity.to_string(),               is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",       value: num_buckets.to_string(),         is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",       value: cli.bucket_size.to_string(),     is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "fingerprint_bits",  value: cli.fingerprint_bits.to_string(), is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",        value: cli.value_bits.to_string(),      is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "n_queried",         value: cli.n_queried.to_string(),       is_default: matches.value_source("n_queried") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("end_to_end_fpr", &knobs, &store_state, &geom);

    // Query keys disjoint from inserted range (8-byte LE).
    let mut n_false_positives = 0u32;
    for i in 0..cli.n_queried {
        let key = (u32::MAX as u64 - i as u64).to_le_bytes();
        let q = client.build_query(&key);
        let r = server.answer(&q).expect("answer");
        if client.decode(&key, &r).expect("decode").is_some() {
            n_false_positives += 1;
        }
    }
    let fpr = n_false_positives as f64 / cli.n_queried as f64;
    writeln!(
        csv,
        "{arity},{num_buckets},{},{},{},{n_inserted},{},{n_false_positives},{fpr:.6}",
        cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.n_queried,
    ).unwrap();
    println!(
        "  arity={arity} nb={num_buckets:<6} bs={} fb={:<2} vb={:<4} | \
         n_ins={n_inserted} n_q={} fp={n_false_positives} fpr={fpr:.6}",
        cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.n_queried,
    );
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_end_to_end_fpr.csv", HEADER);

    match cli.arity {
        2 => run_one::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run_one::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run_one::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_end_to_end_fpr.csv");
}
