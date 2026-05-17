//! **Intent:** Measure client-side `apply_delta` throughput across both
//! shipped backends, with and without a populated precomputation queue.
//!
//! **Arguments (CLI):** `--arity`, `--backend` (frodo|simple, default
//! frodo), `--num-buckets`, `--bucket-size`, `--value-bits`, `--lwe-dim`
//! (backend-dependent default), `--load-factor`, `--batch`,
//! `--precomputed-slots`. See `helpers::parse_cli` for the rest.
//!
//! **Output:** `results/ikpir_client_apply_delta_throughput.csv`
//! Columns: backend, arity, num_buckets, bucket_size, value_bits, batch,
//! precomputed_slots, mean_dps, min_dps, max_dps, stddev_dps

mod helpers;

use helpers::{Backend, CloneStore};
use ikpir_client::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, HintDeltaBundle, IkpirClient,
    IncrementalPirBackend, IndexPirBackend, PrecomputingPirBackend, SimpleConfig, SimplePirBackend,
};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{
    CuckooParams, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::cell::RefCell;
use std::io::Write;

const HEADER: &str = "backend,arity,num_buckets,bucket_size,value_bits,batch,\
    precomputed_slots,mean_dps,min_dps,max_dps,stddev_dps";

#[derive(Clone, clap::Parser)]
#[command(about = "Measure ikpir-client apply_delta throughput (deltas/sec) via criterion.")]
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
    #[arg(long, default_value_t = 10_000)] batch: u32,
    #[arg(long, default_value_t = 0)]      precomputed_slots: u32,
    #[arg(long, default_value_t = 0.50)]   load_factor: f64,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim.unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn collect_deltas<S, B>(
    cells:       &[u32],
    params:      CuckooParams,
    n_seed:      u64,
    cli:         &Cli,
    batch:       u32,
    make_config: &impl Fn() -> B::Config,
) -> Vec<HintDeltaBundle<B>>
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend,
{
    let store = S::clone_from_cells(cells.to_vec(), params, n_seed).expect("from_cells");
    let mut server: IkpirServer<S, B> = IkpirServer::new(store, make_config());
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let mut deltas = Vec::with_capacity(batch as usize);
    for i in 0..batch {
        let k = n_seed as u32 + i;
        for (j, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(31).wrapping_add(j as u32) & 0xFF) as u8;
        }
        match server.insert(&k.to_le_bytes(), &value) {
            Ok(d) => deltas.push(d),
            Err(ikpir_server::IkpirError::TableFull) => break,
            Err(e) => panic!("insert for delta collection: {e:?}"),
        }
    }
    deltas
}

fn run_one<S, B>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    make_config: impl Fn() -> B::Config,
)
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + PrecomputingPirBackend + BackendWireSize + Clone,
    B::Query: Clone, B::Response: Clone,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();
    let lwe_dim_eff = effective_lwe_dim(cli);

    let (seed_store, n_seed) = helpers::populate_to_load::<S>(
        cli.load_factor, num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    if n_seed < 2 { eprintln!("  Skip: seed too small"); return; }

    let cells  = seed_store.snapshot_cells();
    let params = seed_store.params();

    let seed_store2 = S::clone_from_cells(cells.clone(), params, n_seed).expect("from_cells");
    let seed_server: IkpirServer<S, B> = IkpirServer::new(seed_store2, make_config());
    let bundle = seed_server.setup();

    let deltas = collect_deltas::<S, B>(&cells, params, n_seed, cli, cli.batch, &make_config);

    let cps = params.cells_per_slot();
    let store_state = helpers::StoreState {
        capacity:       (num_buckets as u64) * (cli.bucket_size as u64),
        populated:      n_seed,
        load_pct:       100.0 * n_seed as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width:      cli.bucket_size * cps,
        segment_rows:   params.segment_size(),
    };
    let delta_bytes = deltas.first().map(|d| d.wire_byte_size()).unwrap_or(0);
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       B::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:       bundle.wire_byte_size(),
        query_bytes:              0,
        response_bytes:           0,
        hint_delta_typical_bytes: Some(delta_bytes),
    };
    let knobs = [
        helpers::Knob { name: "backend",           value: cli.backend.to_string(),            is_default: matches.value_source("backend") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "arity",             value: arity.to_string(),                  is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",       value: num_buckets.to_string(),            is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",       value: cli.bucket_size.to_string(),        is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",        value: cli.value_bits.to_string(),         is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",           value: lwe_dim_eff.to_string(),            is_default: cli.lwe_dim.is_none() },
        helpers::Knob { name: "load_factor",       value: format!("{:.2}", cli.load_factor),  is_default: matches.value_source("load_factor") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "batch",             value: cli.batch.to_string(),              is_default: matches.value_source("batch") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "precomputed_slots", value: cli.precomputed_slots.to_string(),  is_default: matches.value_source("precomputed_slots") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("apply_delta_throughput", &knobs, &store_state, &geom);

    let client_cell = RefCell::new(IkpirClient::<B>::from_setup(bundle.clone()));
    {
        let mut c = client_cell.borrow_mut();
        if cli.precomputed_slots > 0 {
            c.precompute_queries(cli.precomputed_slots);
            c.precompute_decodes();
        }
    }
    let mut pool_idx = 0usize;
    let crit = helpers::run_criterion_throughput_batched(
        "apply_delta_throughput",
        1,
        || -> Option<HintDeltaBundle<B>> {
            if pool_idx >= deltas.len() {
                let mut c = client_cell.borrow_mut();
                *c = IkpirClient::<B>::from_setup(bundle.clone());
                if cli.precomputed_slots > 0 {
                    c.precompute_queries(cli.precomputed_slots);
                    c.precompute_decodes();
                }
                pool_idx = 0;
            }
            let d = deltas[pool_idx].clone();
            pool_idx += 1;
            Some(d)
        },
        |slot: &mut Option<HintDeltaBundle<B>>| {
            let d = slot.take().expect("setup produced Some");
            client_cell.borrow_mut().apply_delta(d).expect("apply_delta");
        },
    );

    let pre = cli.precomputed_slots;
    writeln!(
        csv,
        "{},{arity},{num_buckets},{},{},{},{pre},{:.2},{:.2},{:.2},{:.2}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.batch,
        crit.mean_ops_per_s, crit.min_ops_per_s, crit.max_ops_per_s, crit.stddev_ops_per_s,
    ).unwrap();
    println!(
        "  backend={} arity={arity} nb={num_buckets:<6} bs={} vb={:<4} pre={pre:<4} | \
         mean={:.2} dps (±{:.2}) delta={delta_bytes}B",
        cli.backend, cli.bucket_size, cli.value_bits, crit.mean_ops_per_s, crit.stddev_ops_per_s,
    );
}

fn dispatch_backend<S: CloneStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
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

    let mut csv = helpers::csv_writer("ikpir_client_apply_delta_throughput.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_apply_delta_throughput.csv");
}
