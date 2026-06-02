//! **Intent:** Measure server-side `answer` throughput and wire sizes
//! (query bytes, response bytes) across both backends (FrodoPIR / SimplePIR).
//!
//! **Method:** Populate to `TableFull`, build a same-process client,
//! pre-build `batch` queries, then time individual `server.answer` calls
//! via criterion (cycling through the pre-built queries).
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (defaults to backend recommendation), `--batch`.
//!
//! **Output:** `results/ikpir_server_answer.csv`
//! Columns: backend, arity, num_buckets, bucket_size, value_bits, plaintext_bits,
//! lwe_dim, batch, mean_qps, min_qps, max_qps, stddev_qps, query_bytes,
//! response_bytes, cells_per_slot, row_width, segment_rows, db_rows, db_cols,
//! load_factor
//!
//! `db_rows` / `db_cols` report the per-segment PIR matrix shape **after** any
//! backend-specific reshape. For FrodoPIR this is `(segment_rows, row_width)`;
//! for SimplePIR this is the post-reshape `(⌈segment_rows/k⌉, k·row_width)`.

mod helpers;

use helpers::{Backend, MakeStore};
use ikpir_client::IkpirClient;
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer, IncrementalPirBackend,
    IndexPirBackend, PirQueryBundle, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;

const HEADER: &str =
    "backend,arity,num_buckets,bucket_size,value_bits,plaintext_bits,lwe_dim,batch,\
    mean_qps,min_qps,max_qps,stddev_qps,query_bytes,response_bytes,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

#[derive(clap::Parser)]
#[command(about = "Measure ikpir-server answer throughput and wire sizes via criterion.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    #[arg(long, default_value_t = 16_384)]
    num_buckets: u32,
    #[arg(long, default_value_t = 4)]
    bucket_size: u32,
    #[arg(long, default_value_t = 256)]
    value_bits: u32,
    #[arg(long, default_value_t = 32)]
    fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]
    plaintext_bits: u32,
    /// LWE dimension. Defaults to 1566 (Frodo) or 1275 (Simple) when omitted.
    #[arg(long)]
    lwe_dim: Option<u32>,
    #[arg(long, default_value_t = 64)]
    batch: u32,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim
        .unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn run_one<S, B>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    backend_config: B::Config,
) where
    S: MakeStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
    B::Query: Clone,
    B::Response: Clone,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    let (store, n_inserted) = helpers::populate_until_full::<S>(
        num_buckets,
        cli.bucket_size,
        cli.fingerprint_bits,
        cli.value_bits,
        cli.plaintext_bits,
    );
    let server: IkpirServer<S, B> = IkpirServer::new(store, backend_config);
    let mut client: IkpirClient<B> = IkpirClient::from_setup(server.setup());

    let queries: Vec<PirQueryBundle<B>> = (0..cli.batch)
        .map(|i| client.build_query(&((i % n_inserted as u32).to_le_bytes())))
        .collect();

    let query_bytes = queries[0].wire_byte_size();
    let response_bytes = server
        .answer(&queries[0])
        .expect("answer ok")
        .wire_byte_size();

    let params = server.params();
    let cps = params.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let segment_rows = params.segment_size();
    let (db_rows, db_cols) =
        helpers::backend_shape_estimate(cli.backend, segment_rows as u64, row_width as u64);
    let load_factor = n_inserted as f64 / (num_buckets as f64 * cli.bucket_size as f64);
    let bundle = server.setup();
    let lwe_dim_eff = effective_lwe_dim(cli);
    let store_state = helpers::StoreState {
        capacity: (num_buckets as u64) * (cli.bucket_size as u64),
        populated: n_inserted,
        load_pct: 100.0 * load_factor,
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes: B::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes: bundle.wire_byte_size(),
        query_bytes,
        response_bytes,
        hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob {
            name: "backend",
            value: cli.backend.to_string(),
            is_default: matches.value_source("backend") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "arity",
            value: arity.to_string(),
            is_default: matches.value_source("arity") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "num_buckets",
            value: num_buckets.to_string(),
            is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "bucket_size",
            value: cli.bucket_size.to_string(),
            is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "fingerprint_bits",
            value: cli.fingerprint_bits.to_string(),
            is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "value_bits",
            value: cli.value_bits.to_string(),
            is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "plaintext_bits",
            value: cli.plaintext_bits.to_string(),
            is_default: matches.value_source("plaintext_bits") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "lwe_dim",
            value: lwe_dim_eff.to_string(),
            is_default: cli.lwe_dim.is_none(),
        },
        helpers::Knob {
            name: "batch",
            value: cli.batch.to_string(),
            is_default: matches.value_source("batch") != Some(ValueSource::CommandLine),
        },
    ];
    helpers::print_preamble("server_answer", &knobs, &store_state, &geom);

    let mut idx = 0usize;
    let crit = helpers::run_criterion_throughput("server_answer", 1, || {
        let _ = server.answer(&queries[idx]).expect("answer ok");
        idx = (idx + 1) % queries.len();
    });

    writeln!(
        csv,
        "{},{arity},{num_buckets},{},{},{},{lwe_dim_eff},{},{:.2},{:.2},{:.2},{:.2},{query_bytes},{response_bytes},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits, cli.batch,
        crit.mean_ops_per_s, crit.min_ops_per_s, crit.max_ops_per_s, crit.stddev_ops_per_s,
        load_factor,
    ).unwrap();
    println!(
        "  backend={} arity={arity} nb={num_buckets:<7} bs={} vb={:<4} | \
         mean={:.2} qps (±{:.2}) | q={query_bytes}B r={response_bytes}B",
        cli.backend, cli.bucket_size, cli.value_bits, crit.mean_ops_per_s, crit.stddev_ops_per_s,
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
        Backend::Frodo => run_one::<S, FrodoPirBackend>(
            csv,
            cli,
            arity,
            num_buckets,
            FrodoConfig::with_lwe_dim(lwe_dim),
        ),
        Backend::Simple => run_one::<S, SimplePirBackend>(
            csv,
            cli,
            arity,
            num_buckets,
            SimpleConfig::with_lwe_dim(lwe_dim),
        ),
    }
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets =
        if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
            cli.num_buckets
        } else {
            helpers::default_num_buckets_for_arity(cli.arity)
        };

    let mut csv = helpers::csv_writer("ikpir_server_answer.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_answer.csv");
}
