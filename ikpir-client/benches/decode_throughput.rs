//! **Intent:** Measure client-side `decode` throughput on the FrodoPIR
//! backend, in three preprocessing modes:
//!
//! - **`cold`** — decode recomputes `c = sᵀ·H` on the fly.
//! - **`warm-b`** — Phase B only; queries cheap, decode still on-the-fly.
//! - **`warm-bc`** — Phase B + Phase C; decode is one vector subtract +
//!   rounding plus the slot scan.
//!
//! **Method:** Populate to `TableFull`, build the server. Each criterion
//! iter: setup (untimed) calls `build_query` + `server.answer` to produce a
//! matching `(key, response)` pair for the *same* client that will decode;
//! the routine times only `client.decode(key, response)`. The criterion
//! HTML/JSON report lands in `target/criterion/decode_throughput/`.
//!
//! **Warm-mode caveat:** the prebuilt queue holds `QUEUE_HEADROOM` slots
//! per segment. If criterion's measurement budget calls `build_query`
//! more than `QUEUE_HEADROOM` times, the bench panics on queue exhaustion
//! — bump `QUEUE_HEADROOM` if you raise criterion's budgets above default.
//!
//! **Output:** `results/ikpir_client_decode_throughput.csv`
//! Columns: mode, arity, num_buckets, bucket_size, value_bits, batch,
//! mean_qps, min_qps, max_qps, stddev_qps, response_bytes

mod helpers;

use helpers::MakeStore;
use ikpir_client::{FrodoConfig, FrodoPirBackend, IkpirClient, PirResponseBundle};
use ikpir_server::{BackendWireSize, IkpirServer};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::cell::RefCell;
use std::io::Write;

type Client   = IkpirClient<FrodoPirBackend>;
type Response = PirResponseBundle<FrodoPirBackend>;

const HEADER: &str = "mode,arity,num_buckets,bucket_size,value_bits,batch,\
    mean_qps,min_qps,max_qps,stddev_qps,response_bytes";

/// Cap on prepared slots for warm modes. Matches `query_throughput`.
const QUEUE_HEADROOM: u32 = 1 << 20;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Mode { Cold, WarmB, WarmBc }
impl Mode {
    fn as_csv(self) -> &'static str {
        match self { Mode::Cold => "cold", Mode::WarmB => "warm-b", Mode::WarmBc => "warm-bc" }
    }
}

#[derive(Clone, clap::Parser)]
#[command(about = "Measure ikpir-client decode throughput (queries/sec) via criterion.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Mode::Cold)] mode: Mode,
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
    if n_inserted == 0 { eprintln!("  Skip: empty store"); return; }

    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let bundle = server.setup();

    // Probe geometry — response_bytes is constant per (params, lwe_dim).
    let mut probe = Client::from_setup(bundle.clone());
    let probe_q = probe.build_query(&0u32.to_le_bytes());
    let response_bytes = server.answer(&probe_q).expect("answer ok").wire_byte_size();
    let query_bytes_for_geom = probe_q.wire_byte_size();

    let params_store = server.params();
    let cps = params_store.cells_per_slot();
    let store_state = helpers::StoreState {
        capacity:       (num_buckets as u64) * (cli.bucket_size as u64),
        populated:      n_inserted,
        load_pct:       100.0 * n_inserted as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width:      cli.bucket_size * cps,
        segment_rows:   params_store.segment_size(),
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       FrodoPirBackend::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:       bundle.wire_byte_size(),
        query_bytes:              query_bytes_for_geom,
        response_bytes,
        hint_delta_typical_bytes: None,
    };
    let mode_str = cli.mode.as_csv();
    let knobs = [
        helpers::Knob { name: "mode",        value: mode_str.to_string(),        is_default: matches.value_source("mode") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "arity",       value: arity.to_string(),           is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",  value: num_buckets.to_string(),     is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",  value: cli.bucket_size.to_string(), is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",   value: cli.value_bits.to_string(),  is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",      value: cli.lwe_dim.to_string(),     is_default: matches.value_source("lwe_dim") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "batch",        value: cli.batch.to_string(),       is_default: matches.value_source("batch") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("decode_throughput", &knobs, &store_state, &geom);

    // Fresh client for the timed bench. Optionally warm (queue + decodes).
    let client_cell = RefCell::new(Client::from_setup(server.setup()));
    {
        let mut c = client_cell.borrow_mut();
        match cli.mode {
            Mode::Cold => {}
            Mode::WarmB | Mode::WarmBc => { c.precompute_queries(QUEUE_HEADROOM); }
        }
        if matches!(cli.mode, Mode::WarmBc) { c.precompute_decodes(); }
    }

    let n = n_inserted as u32;
    let keys: Vec<Vec<u8>> = (0..cli.batch).map(|i| (i % n).to_le_bytes().to_vec()).collect();
    let mut idx = 0usize;
    let crit = helpers::run_criterion_throughput_batched(
        "decode_throughput",
        1,
        || {
            let key = keys[idx % keys.len()].clone();
            idx = idx.wrapping_add(1);
            let q = client_cell.borrow_mut().build_query(&key);
            let r = server.answer(&q).expect("answer ok");
            (key, r)
        },
        |slot: &mut (Vec<u8>, Response)| {
            let (key, r) = slot;
            let _ = client_cell.borrow_mut().decode(key, r).expect("decode");
        },
    );

    writeln!(
        csv,
        "{mode_str},{arity},{num_buckets},{},{},{},{:.2},{:.2},{:.2},{:.2},{response_bytes}",
        cli.bucket_size, cli.value_bits, cli.batch,
        crit.mean_ops_per_s, crit.min_ops_per_s, crit.max_ops_per_s, crit.stddev_ops_per_s,
    ).unwrap();
    println!(
        "  mode={mode_str:<7} arity={arity} nb={num_buckets:<6} | \
         mean={:.2} qps (±{:.2}) r={response_bytes}B",
        crit.mean_ops_per_s, crit.stddev_ops_per_s,
    );
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_client_decode_throughput.csv", HEADER);

    match cli.arity {
        2 => run_one::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run_one::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run_one::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_decode_throughput.csv");
}
