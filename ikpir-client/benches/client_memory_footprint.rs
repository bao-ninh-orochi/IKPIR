//! **Intent:** Report the heap footprint of an `IkpirClient<B>` in each
//! preprocessing state (cold / warm-b / warm-bc), across both shipped
//! backends.
//!
//! **Method:** Closed-form accounting from public state geometry. The
//! per-slot and per-segment-baseline formulas differ per backend because
//! FrodoPIR holds the tall-skinny `(n_rows × row_width)` layout while
//! SimplePIR uses the reshaped `(reshape_rows × reshape_row_width)`
//! layout. Both backends' `PreparedSlot` is `(secret, b, c?)` of the
//! corresponding lengths.
//!
//! ```text
//! FrodoPIR per segment:
//!   baseline       = 2 × (n_rows × lwe_dim × 4)        // params.a + state.params.a (cloned)
//!                  + (lwe_dim × row_width × 4)         // hint.data
//!   per-slot cold  = (lwe_dim + n_rows) × 4            // secret + b
//!   per-slot warmc = (lwe_dim + n_rows + row_width) × 4 // + precomputed c
//!
//! SimplePIR per segment (same shape, but with reshape dims):
//!   baseline       = 2 × (reshape_rows × lwe_dim × 4)
//!                  + (lwe_dim × reshape_row_width × 4)
//!   per-slot cold  = (lwe_dim + reshape_rows) × 4
//!   per-slot warmc = (lwe_dim + reshape_rows + reshape_row_width) × 4
//! ```
//!
//! **Arguments (CLI):** `--arity`, `--backend` (frodo|simple, default
//! frodo), `--num-buckets`, `--bucket-size`, `--value-bits`, `--lwe-dim`
//! (backend-dependent default), `--mode` (cold / warm-b / warm-bc),
//! `--batch`. See `helpers::parse_cli` for the rest.
//!
//! **Output:** `results/ikpir_client_memory_footprint.csv`
//! Columns: backend, mode, arity, num_buckets, bucket_size, value_bits,
//! lwe_dim, batch, stack_bytes, heap_bytes, total_bytes

mod helpers;

use helpers::{Backend, MakeStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirClient, IncrementalPirBackend,
    IndexPirBackend, PrecomputingPirBackend, SimpleConfig, SimplePirBackend,
};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;
use std::mem;

const HEADER: &str = "backend,mode,arity,num_buckets,bucket_size,value_bits,lwe_dim,\
    batch,stack_bytes,heap_bytes,total_bytes";

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Mode { Cold, WarmB, WarmBc }
impl Mode {
    fn as_csv(self) -> &'static str {
        match self { Mode::Cold => "cold", Mode::WarmB => "warm-b", Mode::WarmBc => "warm-bc" }
    }
}

#[derive(Clone, clap::Parser)]
#[command(about = "Closed-form heap accounting for IkpirClient<B>.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    #[arg(long, value_enum, default_value_t = Mode::Cold)] mode: Mode,
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

/// SimplePIR reshape: `k = max(1, round(√(n_rows / row_width)))`,
/// `R = ⌈n_rows / k⌉`, `C = k · row_width`. Mirrors the formula in
/// `ikpir-common/src/backend/simple/backend.rs::reshape_dims`.
fn simple_reshape_dims(n_rows: u32, row_width: u32) -> (u32, u32) {
    let ratio = (n_rows as f64) / (row_width as f64);
    let k = ratio.sqrt().round().max(1.0) as u32;
    let r = n_rows.div_ceil(k);
    let c = k * row_width;
    (r, c)
}

fn footprint<B>(cli: &Cli, arity: u32, num_buckets: u32) -> (usize, usize)
where
    B: IndexPirBackend + IncrementalPirBackend + PrecomputingPirBackend,
{
    let n_rows         = num_buckets / arity;
    let cells_per_slot = (cli.fingerprint_bits + cli.value_bits).div_ceil(cli.plaintext_bits);
    let row_width      = cli.bucket_size * cells_per_slot;
    let lwe_dim        = effective_lwe_dim(cli) as usize;

    // Backend-specific A and hint dims.
    let (matvec_rows, matvec_cols) = match cli.backend {
        Backend::Frodo  => (n_rows as usize, row_width as usize),
        Backend::Simple => {
            let (r, c) = simple_reshape_dims(n_rows, row_width);
            (r as usize, c as usize)
        }
    };

    let per_seg_baseline   = 2 * (matvec_rows * lwe_dim * 4) + (lwe_dim * matvec_cols * 4);
    let per_slot_cold      = (lwe_dim + matvec_rows) * 4;
    let per_slot_warm_c    = (lwe_dim + matvec_rows + matvec_cols) * 4;
    let prepared_count     = match cli.mode { Mode::Cold => 0, Mode::WarmB | Mode::WarmBc => cli.batch as usize };
    let per_seg_prepared   = match cli.mode {
        Mode::Cold | Mode::WarmB => prepared_count * per_slot_cold,
        Mode::WarmBc             => prepared_count * per_slot_warm_c,
    };
    let heap_bytes  = (arity as usize) * (per_seg_baseline + per_seg_prepared);
    let stack_bytes = mem::size_of::<IkpirClient<B>>();
    (stack_bytes, heap_bytes)
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

    let store = S::make_store(
        num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    ).expect("make_store");
    let server: IkpirServer<S, B> = IkpirServer::new(store, backend_config);
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
    let mut probe_client: IkpirClient<B> = IkpirClient::from_setup(bundle.clone());
    let q0 = probe_client.build_query(&0u32.to_le_bytes());
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       B::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:       bundle.wire_byte_size(),
        query_bytes:              q0.wire_byte_size(),
        response_bytes:           server.answer(&q0).expect("answer ok").wire_byte_size(),
        hint_delta_typical_bytes: None,
    };
    let mode_str = cli.mode.as_csv();
    let lwe_dim_eff = effective_lwe_dim(cli);
    let knobs = [
        helpers::Knob { name: "backend",          value: cli.backend.to_string(),         is_default: matches.value_source("backend") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "arity",            value: arity.to_string(),               is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "mode",             value: mode_str.to_string(),            is_default: matches.value_source("mode") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",      value: num_buckets.to_string(),         is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",      value: cli.bucket_size.to_string(),     is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "fingerprint_bits", value: cli.fingerprint_bits.to_string(),is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",       value: cli.value_bits.to_string(),      is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "plaintext_bits",   value: cli.plaintext_bits.to_string(),  is_default: matches.value_source("plaintext_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",          value: lwe_dim_eff.to_string(),         is_default: cli.lwe_dim.is_none() },
        helpers::Knob { name: "batch",            value: cli.batch.to_string(),           is_default: matches.value_source("batch") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("client_memory_footprint", &knobs, &store_state, &geom);

    let (stack_bytes, heap_bytes) = footprint::<B>(cli, arity, num_buckets);
    let total_bytes = stack_bytes + heap_bytes;
    writeln!(
        csv,
        "{},{mode_str},{arity},{num_buckets},{},{},{},{},{stack_bytes},{heap_bytes},{total_bytes}",
        cli.backend, cli.bucket_size, cli.value_bits, lwe_dim_eff, cli.batch,
    ).unwrap();
    println!(
        "  backend={} mode={mode_str:<7} arity={arity} nb={num_buckets:<6} bs={} vb={:<4} | \
         stack={stack_bytes}B heap={heap_bytes}B total={total_bytes}B",
        cli.backend, cli.bucket_size, cli.value_bits,
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

    let mut csv = helpers::csv_writer("ikpir_client_memory_footprint.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_memory_footprint.csv");
}
