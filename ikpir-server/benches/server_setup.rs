//! **Intent:** Measure server-side setup cost (deriving matrix A from seed
//! and computing the hint matrix) across both backends (FrodoPIR / SimplePIR).
//!
//! **Method:** Populate a store to `TableFull`, wrap it in
//! `IkpirServer::new`, and time the wall-clock cost of that call.
//! Repeats for `trials` trials after `warmup` warmup rounds.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (defaults to backend recommendation), `--trials`, `--warmup`.
//!
//! **Output:** `results/ikpir_server_setup.csv`
//! Columns: backend, arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! mean_setup_ms, min_setup_ms, max_setup_ms, stddev_setup_ms,
//! setup_bundle_bytes, hint_bytes_per_segment, server_params_bytes_per_segment,
//! cells_per_slot, row_width, segment_rows, load_factor

mod helpers;

use helpers::{Backend, CloneStore};
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer, IncrementalPirBackend,
    IndexPirBackend, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "backend,arity,num_buckets,bucket_size,value_bits,lwe_dim,\
    mean_setup_ms,min_setup_ms,max_setup_ms,stddev_setup_ms,\
    setup_bundle_bytes,hint_bytes_per_segment,server_params_bytes_per_segment,\
    cells_per_slot,row_width,segment_rows,load_factor";

#[derive(clap::Parser)]
#[command(about = "Measure ikpir-server setup wall-clock cost (populate-to-full then IkpirServer::new).")]
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
    /// LWE dimension. Defaults to 1566 (Frodo) or 1275 (Simple) when omitted.
    #[arg(long)]                           lwe_dim: Option<u32>,
    #[arg(long, default_value_t = 2)]      warmup: u32,
    #[arg(long, default_value_t = 5)]      trials: u32,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim.unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn run_one<S, B>(
    csv:         &mut std::io::BufWriter<std::fs::File>,
    cli:         &Cli,
    arity:       u32,
    num_buckets: u32,
    make_config: impl Fn() -> B::Config,
)
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();
    let nb_is_default = matches.value_source("num_buckets") != Some(ValueSource::CommandLine);

    let (seed_store, n_inserted) = helpers::populate_until_full::<S>(
        num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    let cells  = seed_store.snapshot_cells();
    let params = seed_store.params();
    drop(seed_store);

    let (mut bundle_bytes, mut hint_bytes, mut sp_bytes) = (0usize, 0usize, 0usize);
    let lwe_dim_eff = effective_lwe_dim(cli);

    let mut samples = Vec::with_capacity((cli.warmup + cli.trials) as usize);
    for trial in 0..(cli.warmup + cli.trials) {
        let store = S::clone_from_cells(cells.clone(), params, n_inserted).expect("from_cells");
        let t = Instant::now();
        let server: IkpirServer<S, B> = IkpirServer::new(store, make_config());
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if trial == cli.warmup {
            let bundle  = server.setup();
            bundle_bytes = bundle.wire_byte_size();
            hint_bytes   = B::hint_byte_size(&bundle.hints[0]);
            sp_bytes     = B::server_params_byte_size(&bundle.backend_params[0]);
        }
        if trial >= cli.warmup { samples.push(ms); }
    }

    let s = helpers::compute_stats(&samples);
    let cps          = params.cells_per_slot();
    let row_width    = cli.bucket_size * cps;
    let segment_rows = params.segment_size();
    let load_factor  = n_inserted as f64 / (num_buckets as u64 * cli.bucket_size as u64) as f64;
    writeln!(
        csv,
        "{},{arity},{num_buckets},{},{},{lwe_dim_eff},{:.3},{:.3},{:.3},{:.3},{bundle_bytes},{hint_bytes},{sp_bytes},{cps},{row_width},{segment_rows},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits,
        s.mean, s.min, s.max, s.stddev, load_factor,
    ).unwrap();

    let store_state = helpers::StoreState {
        capacity:       (num_buckets as u64) * (cli.bucket_size as u64),
        populated:      n_inserted,
        load_pct:       100.0 * load_factor,
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       hint_bytes,
        setup_bundle_bytes:       bundle_bytes,
        query_bytes:              0,
        response_bytes:           0,
        hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob { name: "backend",          value: cli.backend.to_string(),          is_default: matches.value_source("backend") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "arity",            value: arity.to_string(),                is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",      value: num_buckets.to_string(),          is_default: nb_is_default },
        helpers::Knob { name: "bucket_size",      value: cli.bucket_size.to_string(),      is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "fingerprint_bits", value: cli.fingerprint_bits.to_string(), is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",       value: cli.value_bits.to_string(),       is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",          value: lwe_dim_eff.to_string(),          is_default: cli.lwe_dim.is_none() },
    ];
    helpers::print_preamble("server_setup", &knobs, &store_state, &geom);
    println!(
        "  backend={} arity={arity} nb={num_buckets:<7} bs={} vb={:<4} | \
         mean={:.3} ms (±{:.3}) | bundle={}B hint/seg={}B sp/seg={}B",
        cli.backend, cli.bucket_size, cli.value_bits, s.mean, s.stddev,
        bundle_bytes, hint_bytes, sp_bytes,
    );
}

fn dispatch_backend<S: CloneStore>(
    csv:         &mut std::io::BufWriter<std::fs::File>,
    cli:         &Cli,
    arity:       u32,
    num_buckets: u32,
) {
    let lwe_dim = effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo  => run_one::<S, FrodoPirBackend>(csv, cli, arity, num_buckets, || FrodoConfig::with_lwe_dim(lwe_dim)),
        Backend::Simple => run_one::<S, SimplePirBackend>(csv, cli, arity, num_buckets, || SimpleConfig::with_lwe_dim(lwe_dim)),
    }
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_server_setup.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_setup.csv");
}
