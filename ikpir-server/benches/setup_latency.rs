//! **Intent:** Measure server-side setup cost on the FrodoPIR backend.
//!
//! **Method:** Populate a store to `TableFull`, wrap it in `IkpirServer::new`,
//! and time the wall-clock cost of that call. Repeats for `trials` trials.
//!
//! **Output:** `results/ikpir_server_setup_latency.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! mean_setup_ms, min_setup_ms, max_setup_ms, stddev_setup_ms,
//! setup_bundle_bytes, hint_bytes_per_segment, server_params_bytes_per_segment

mod helpers;

use helpers::CloneStore;
use ikpir_server::{BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer};
use segmented_cuckoo::{
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "arity,num_buckets,bucket_size,value_bits,lwe_dim,\
    mean_setup_ms,min_setup_ms,max_setup_ms,stddev_setup_ms,\
    setup_bundle_bytes,hint_bytes_per_segment,server_params_bytes_per_segment";

#[derive(clap::Parser)]
#[command(about = "Measure ikpir-server setup wall-clock cost.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)]   lwe_dim: u32,
    #[arg(long, default_value_t = 2)]      warmup: u32,
    #[arg(long, default_value_t = 5)]      trials: u32,
}

fn run_one<S: CloneStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();
    let nb_is_default = matches.value_source("num_buckets") != Some(ValueSource::CommandLine);

    // Populate once, snapshot, clone per trial to avoid re-populating.
    let (seed_store, n_inserted) = helpers::populate_until_full::<S>(
        num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    let cells  = seed_store.snapshot_cells();
    let params = seed_store.params();
    drop(seed_store);

    // Sample wire sizes from first trial.
    let (mut bundle_bytes, mut hint_bytes, mut sp_bytes) = (0usize, 0usize, 0usize);

    let mut samples = Vec::with_capacity((cli.warmup + cli.trials) as usize);
    for trial in 0..(cli.warmup + cli.trials) {
        let store = S::clone_from_cells(cells.clone(), params, n_inserted).expect("from_cells");
        let t = Instant::now();
        let server: IkpirServer<S, FrodoPirBackend> =
            IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if trial == cli.warmup {
            let bundle  = server.setup();
            bundle_bytes = bundle.wire_byte_size();
            hint_bytes   = FrodoPirBackend::hint_byte_size(&bundle.hints[0]);
            sp_bytes     = FrodoPirBackend::server_params_byte_size(&bundle.backend_params[0]);
        }
        if trial >= cli.warmup { samples.push(ms); }
    }

    let s = helpers::compute_stats(&samples);
    writeln!(
        csv,
        "{arity},{num_buckets},{},{},{},{:.3},{:.3},{:.3},{:.3},{bundle_bytes},{hint_bytes},{sp_bytes}",
        cli.bucket_size, cli.value_bits, cli.lwe_dim, s.mean, s.min, s.max, s.stddev,
    ).unwrap();

    // Preamble (printed after first run so geometry is populated).
    let cps  = params.cells_per_slot();
    let store_state = helpers::StoreState {
        capacity:     (num_buckets as u64) * (cli.bucket_size as u64),
        populated:    n_inserted,
        load_pct:     100.0 * n_inserted as f64 / ((num_buckets as u64 * cli.bucket_size as u64) as f64),
        cells_per_slot: cps,
        row_width:    cli.bucket_size * cps,
        segment_rows: params.segment_size(),
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes:   hint_bytes,
        setup_bundle_bytes:   bundle_bytes,
        query_bytes:          0,
        response_bytes:       0,
        hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob { name: "arity",            value: arity.to_string(),            is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",       value: num_buckets.to_string(),      is_default: nb_is_default },
        helpers::Knob { name: "bucket_size",       value: cli.bucket_size.to_string(),  is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "fingerprint_bits",  value: cli.fingerprint_bits.to_string(), is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",        value: cli.value_bits.to_string(),   is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",           value: cli.lwe_dim.to_string(),      is_default: matches.value_source("lwe_dim") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("setup_latency", &knobs, &store_state, &geom);

    println!(
        "  arity={arity} num_buckets={num_buckets:<6} bs={} vb={:<4} | \
         mean={:.3} ms (±{:.3}) | bundle={bundle_bytes}B hint/seg={hint_bytes}B sp/seg={sp_bytes}B",
        cli.bucket_size, cli.value_bits, s.mean, s.stddev,
    );
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_server_setup_latency.csv", HEADER);

    match cli.arity {
        2 => run_one::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run_one::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run_one::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_setup_latency.csv");
}
