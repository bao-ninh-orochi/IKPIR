//! **Intent:** Catalogue on-wire byte sizes of every IKPIR bundle shape
//! (no timing).
//!
//! **Method:** Populate to `--load-factor` (default `0.50`), snapshot
//! the setup bundle, sample one query + response, then issue one
//! `insert` / `update` / `delete` and record their delta sizes.
//!
//! **Arguments (CLI):** `--arity`, `--num-buckets`, `--bucket-size`,
//! `--value-bits`, `--lwe-dim`, `--load-factor`. See
//! `helpers::parse_cli` for defaults.
//!
//! **Design rationale:** Companion to `setup_latency` and
//! `incremental_vs_rebuild` — the timing benches report cost, this one
//! reports the wire footprint. Together they pin both halves of the
//! deployment trade-off.
//!
//! **Output:** `results/ikpir_wire_sizes.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! setup_bundle_bytes, query_bytes, response_bytes,
//! hint_delta_insert_bytes, hint_delta_update_bytes,
//! hint_delta_delete_bytes

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;

type Client = IkpirClient<FrodoPirBackend>;

const HEADER: &str = "arity,num_buckets,bucket_size,value_bits,lwe_dim,\
    setup_bundle_bytes,query_bytes,response_bytes,\
    hint_delta_insert_bytes,hint_delta_update_bytes,hint_delta_delete_bytes";

#[derive(clap::Parser)]
#[command(about = "Catalog IKPIR bundle wire sizes (no timing).")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)]   lwe_dim: u32,
    /// Load factor — needs headroom for insert and targets for update/delete.
    #[arg(long, default_value_t = 0.50)]   load_factor: f64,
}

fn run_one<S: MakeStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    let (store, n_seed) = helpers::populate_to_load::<S>(
        cli.load_factor, num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    if n_seed < 2 { eprintln!("  Skip: seed too small"); return; }

    let mut server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let mut client: Client = Client::from_setup(server.setup());

    let setup_bundle_bytes = server.setup().wire_byte_size();
    let q = client.build_query(&0u32.to_le_bytes());
    let query_bytes    = q.wire_byte_size();
    let r = server.answer(&q).expect("answer ok");
    let response_bytes = r.wire_byte_size();

    let vsize = (cli.value_bits as usize).div_ceil(8);
    let value = vec![0u8; vsize];
    let insert_key = (n_seed as u32).to_le_bytes();
    let ins = server.insert(&insert_key, &value).expect("insert ok");
    let hint_delta_insert_bytes = ins.wire_byte_size();

    let upd = server.update(&0u32.to_le_bytes(), &value).expect("update ok");
    let hint_delta_update_bytes = upd.wire_byte_size();

    let del = server.delete(&1u32.to_le_bytes()).expect("delete ok");
    let hint_delta_delete_bytes = del.wire_byte_size();

    let params = server.params();
    let cps = params.cells_per_slot();
    let bundle = server.setup();
    let store_state = helpers::StoreState {
        capacity:      (num_buckets as u64) * (cli.bucket_size as u64),
        populated:     n_seed,
        load_pct:      100.0 * n_seed as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width:     cli.bucket_size * cps,
        segment_rows:  params.segment_size(),
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       FrodoPirBackend::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes,
        query_bytes,
        response_bytes,
        hint_delta_typical_bytes: Some(hint_delta_insert_bytes),
    };
    let knobs = [
        helpers::Knob { name: "arity",       value: arity.to_string(),           is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",  value: num_buckets.to_string(),     is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",  value: cli.bucket_size.to_string(), is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",   value: cli.value_bits.to_string(),  is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",      value: cli.lwe_dim.to_string(),     is_default: matches.value_source("lwe_dim") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "load_factor",  value: format!("{:.2}", cli.load_factor), is_default: matches.value_source("load_factor") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("wire_sizes", &knobs, &store_state, &geom);

    writeln!(
        csv,
        "{arity},{num_buckets},{},{},{},{setup_bundle_bytes},{query_bytes},{response_bytes},{hint_delta_insert_bytes},{hint_delta_update_bytes},{hint_delta_delete_bytes}",
        cli.bucket_size, cli.value_bits, cli.lwe_dim,
    ).unwrap();
    println!(
        "  arity={arity} num_buckets={num_buckets:<6} bs={} vb={:<4} | \
         setup={setup_bundle_bytes}B q={query_bytes}B r={response_bytes}B \
         dI={hint_delta_insert_bytes}B dU={hint_delta_update_bytes}B dD={hint_delta_delete_bytes}B",
        cli.bucket_size, cli.value_bits,
    );
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_wire_sizes.csv", HEADER);

    match cli.arity {
        2 => run_one::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run_one::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run_one::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_wire_sizes.csv");
}
