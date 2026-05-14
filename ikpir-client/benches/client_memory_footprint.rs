//! **Intent:** Report the heap footprint of an
//! `IkpirClient<FrodoPirBackend>` in each preprocessing state
//! (cold / warm-b / warm-bc), as a function of `(arity, num_buckets,
//! bucket_size, value_bits, lwe_dim, batch)`.
//!
//! **Method:** Closed-form accounting from public state introspection.
//! `FrodoClientState::PreparedSlot` is private to the FrodoPIR backend, so
//! the per-slot size is derived analytically from the public geometry:
//!
//! ```text
//!   per-segment baseline   = 2 × (n_rows × lwe_dim × 4)   // params.a + state.params.a (cloned)
//!                            + (lwe_dim × row_width × 4)  // hint.data
//!   per-slot (cold/warm-b) = (lwe_dim + n_rows) × 4       // secret + b
//!   per-slot (warm-bc)     = (lwe_dim + n_rows + row_width) × 4   // + precomputed c
//!   prepared count         = 0 (cold) | batch (warm-b/warm-bc)
//!   heap_bytes             = arity × (baseline + prepared count × per-slot)
//!   stack_bytes            = mem::size_of::<IkpirClient<FrodoPirBackend>>()
//! ```
//!
//! Geometry:
//!   - `n_rows`    = num_buckets / arity (segment_size)
//!   - `row_width` = bucket_size × cells_per_slot
//!   - `cells_per_slot` = (fingerprint_bits + value_bits).div_ceil(plaintext_bits)
//!
//! **Arguments (CLI):** `--arity`, `--num-buckets`, `--bucket-size`,
//! `--value-bits`, `--lwe-dim`, `--mode` (cold / warm-b / warm-bc),
//! `--batch` (queue depth when `mode != cold`). See
//! `helpers::parse_cli` for defaults.
//!
//! **Design rationale:** Memory is a deployment constraint independent
//! of CPU. The cold-vs-warm-bc spread shows how much RAM Phase-B and
//! Phase-C amortisation cost — useful for sizing an edge or mobile
//! client.
//!
//! **Output:** `results/ikpir_client_memory_footprint.csv`
//! Columns: mode, arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! batch, stack_bytes, heap_bytes, total_bytes

mod helpers;

use helpers::MakeStore;
use ikpir_client::{FrodoConfig, FrodoPirBackend, IkpirClient};
use ikpir_server::{BackendWireSize, IkpirServer};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::mem;

type Client = IkpirClient<FrodoPirBackend>;

const HEADER: &str = "mode,arity,num_buckets,bucket_size,value_bits,lwe_dim,\
    batch,stack_bytes,heap_bytes,total_bytes";

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Mode { Cold, WarmB, WarmBc }
impl Mode {
    fn as_csv(self) -> &'static str {
        match self { Mode::Cold => "cold", Mode::WarmB => "warm-b", Mode::WarmBc => "warm-bc" }
    }
}

#[derive(Clone, clap::Parser)]
#[command(about = "Closed-form heap accounting for IkpirClient<FrodoPirBackend>.")]
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

fn footprint(cli: &Cli, arity: u32, num_buckets: u32) -> (usize, usize) {
    let n_rows         = (num_buckets / arity) as usize;
    let cells_per_slot = (cli.fingerprint_bits + cli.value_bits).div_ceil(cli.plaintext_bits) as usize;
    let row_width      = cli.bucket_size as usize * cells_per_slot;
    let lwe_dim        = cli.lwe_dim as usize;

    let per_seg_baseline   = 2 * (n_rows * lwe_dim * 4) + (lwe_dim * row_width * 4);
    let per_slot_cold      = (lwe_dim + n_rows) * 4;
    let per_slot_warm_c    = (lwe_dim + n_rows + row_width) * 4;
    let prepared_count     = match cli.mode { Mode::Cold => 0, Mode::WarmB | Mode::WarmBc => cli.batch as usize };
    let per_seg_prepared   = match cli.mode {
        Mode::Cold | Mode::WarmB => prepared_count * per_slot_cold,
        Mode::WarmBc             => prepared_count * per_slot_warm_c,
    };
    let heap_bytes  = (arity as usize) * (per_seg_baseline + per_seg_prepared);
    let stack_bytes = mem::size_of::<Client>();
    (stack_bytes, heap_bytes)
}

fn run_one<S: MakeStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    // Build an empty server purely to extract Geometry for the preamble.
    // No populate, no timing — closed-form accounting follows.
    let store = S::make_store(
        num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    ).expect("make_store");
    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let bundle = server.setup();
    let params_store = server.params();
    let cps = params_store.cells_per_slot();
    let store_state = helpers::StoreState {
        capacity:       (num_buckets as u64) * (cli.bucket_size as u64),
        populated:      0,
        load_pct:       0.0,
        cells_per_slot: cps,
        row_width:      cli.bucket_size * cps,
        segment_rows:   params_store.segment_size(),
    };
    let mut probe_client = Client::from_setup(bundle.clone());
    let q0 = probe_client.build_query(&0u32.to_le_bytes());
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       FrodoPirBackend::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:       bundle.wire_byte_size(),
        query_bytes:              q0.wire_byte_size(),
        response_bytes:           server.answer(&q0).expect("answer ok").wire_byte_size(),
        hint_delta_typical_bytes: None,
    };
    let mode_str = cli.mode.as_csv();
    let knobs = [
        helpers::Knob { name: "arity",            value: arity.to_string(),               is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "mode",             value: mode_str.to_string(),            is_default: matches.value_source("mode") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",       value: num_buckets.to_string(),         is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",       value: cli.bucket_size.to_string(),     is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "fingerprint_bits",  value: cli.fingerprint_bits.to_string(), is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",        value: cli.value_bits.to_string(),      is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "plaintext_bits",    value: cli.plaintext_bits.to_string(),  is_default: matches.value_source("plaintext_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",           value: cli.lwe_dim.to_string(),         is_default: matches.value_source("lwe_dim") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "batch",             value: cli.batch.to_string(),           is_default: matches.value_source("batch") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("client_memory_footprint", &knobs, &store_state, &geom);

    let (stack_bytes, heap_bytes) = footprint(cli, arity, num_buckets);
    let total_bytes = stack_bytes + heap_bytes;
    writeln!(
        csv,
        "{mode_str},{arity},{num_buckets},{},{},{},{},{stack_bytes},{heap_bytes},{total_bytes}",
        cli.bucket_size, cli.value_bits, cli.lwe_dim, cli.batch,
    ).unwrap();
    println!(
        "  mode={mode_str:<7} arity={arity} nb={num_buckets:<6} bs={} vb={:<4} | \
         stack={stack_bytes}B heap={heap_bytes}B total={total_bytes}B",
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

    let mut csv = helpers::csv_writer("ikpir_client_memory_footprint.csv", HEADER);

    match cli.arity {
        2 => run_one::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run_one::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run_one::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_memory_footprint.csv");
}
