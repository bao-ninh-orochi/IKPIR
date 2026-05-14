//! **Intent:** Measure client-side `apply_delta` throughput on the FrodoPIR
//! backend, with and without a populated precomputation queue.
//!
//! **Method:** Populate to `--load-factor` (default 0.50). Pre-collect a
//! large pool of `HintDeltaBundle`s (sequential inserts past the seed
//! point). The pool is exhausted by replaying through it once per client
//! reset; on exhaustion, recreate the client from the seed bundle (untimed
//! in setup). The criterion HTML/JSON report lands in
//! `target/criterion/apply_delta_throughput/`.
//!
//! **Arguments (CLI):** `--arity`, `--num-buckets`, `--bucket-size`,
//! `--value-bits`, `--load-factor`, `--batch` (deltas per timed
//! window), `--precomputed-slots` (Phase-B/C queue depth at window
//! start). See `helpers::parse_cli` for defaults.
//!
//! **Design rationale:** `apply_delta` is the client steady-state
//! cost — the throughput here determines how many mutations/sec a
//! client can absorb before falling behind. The `precomputed_slots`
//! axis exposes the Phase-C patching overhead: warm queues pay
//! `O(prepared · row_deltas · (lwe_dim + n_cells))` extra per delta.
//!
//! **Output:** `results/ikpir_client_apply_delta_throughput.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, batch,
//! precomputed_slots, mean_dps, min_dps, max_dps, stddev_dps

mod helpers;

use helpers::CloneStore;
use ikpir_client::{FrodoConfig, FrodoPirBackend, HintDeltaBundle, IkpirClient};
use ikpir_server::{BackendWireSize, IkpirServer};
use segmented_cuckoo::{
    CuckooParams, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::cell::RefCell;
use std::io::Write;

type Client = IkpirClient<FrodoPirBackend>;
type Delta  = HintDeltaBundle<FrodoPirBackend>;

const HEADER: &str = "arity,num_buckets,bucket_size,value_bits,batch,\
    precomputed_slots,mean_dps,min_dps,max_dps,stddev_dps";

#[derive(Clone, clap::Parser)]
#[command(about = "Measure ikpir-client apply_delta throughput (deltas/sec) via criterion.")]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, default_value_t = 16_384)] num_buckets: u32,
    #[arg(long, default_value_t = 4)]      bucket_size: u32,
    #[arg(long, default_value_t = 256)]    value_bits: u32,
    #[arg(long, default_value_t = 32)]     fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]      plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)]   lwe_dim: u32,
    /// Number of deltas to pre-collect; reported in CSV `batch` column.
    /// Set high enough to cover criterion's iteration budget without reset.
    #[arg(long, default_value_t = 10_000)] batch: u32,
    #[arg(long, default_value_t = 0)]      precomputed_slots: u32,
    #[arg(long, default_value_t = 0.50)]   load_factor: f64,
}

/// Collect up to `batch` deltas. Stops early on `TableFull` (returns
/// whatever was built so far) so small smoke configs don't panic.
fn collect_deltas<S: CloneStore>(
    cells:       &[u32],
    params:      CuckooParams,
    n_seed:      u64,
    cli:         &Cli,
    batch:       u32,
) -> Vec<Delta> {
    let store = S::clone_from_cells(cells.to_vec(), params, n_seed).expect("from_cells");
    let mut server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
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

fn run_one<S: CloneStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    let (seed_store, n_seed) = helpers::populate_to_load::<S>(
        cli.load_factor, num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    if n_seed < 2 { eprintln!("  Skip: seed too small"); return; }

    let cells  = seed_store.snapshot_cells();
    let params = seed_store.params();

    // Build seed server for setup bundle (used to create fresh clients).
    let seed_store2 = S::clone_from_cells(cells.clone(), params, n_seed).expect("from_cells");
    let seed_server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(seed_store2, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let bundle = seed_server.setup();

    // Pre-collect deltas once from a separate clone.
    let deltas = collect_deltas::<S>(&cells, params, n_seed, cli, cli.batch);

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
        hint_per_seg_bytes:       FrodoPirBackend::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:       bundle.wire_byte_size(),
        query_bytes:              0,
        response_bytes:           0,
        hint_delta_typical_bytes: Some(delta_bytes),
    };
    let knobs = [
        helpers::Knob { name: "arity",            value: arity.to_string(),                  is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",       value: num_buckets.to_string(),            is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",       value: cli.bucket_size.to_string(),        is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",        value: cli.value_bits.to_string(),         is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",           value: cli.lwe_dim.to_string(),            is_default: matches.value_source("lwe_dim") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "load_factor",       value: format!("{:.2}", cli.load_factor),  is_default: matches.value_source("load_factor") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "batch",             value: cli.batch.to_string(),              is_default: matches.value_source("batch") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "precomputed_slots", value: cli.precomputed_slots.to_string(),  is_default: matches.value_source("precomputed_slots") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("apply_delta_throughput", &knobs, &store_state, &geom);

    // Pool walk: setup clones the next delta out of the pool and resets the
    // client from the seed bundle when the pool is exhausted. Both happen
    // outside the timed bracket. Routine `take`s the owned delta from the
    // Option and feeds it straight into `apply_delta`. The reset uses the
    // monotone epoch sequence in the pool, so each replay round delivers a
    // fresh strictly-increasing-epoch run.
    let client_cell = RefCell::new(Client::from_setup(bundle.clone()));
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
        || -> Option<Delta> {
            if pool_idx >= deltas.len() {
                let mut c = client_cell.borrow_mut();
                *c = Client::from_setup(bundle.clone());
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
        |slot: &mut Option<Delta>| {
            let d = slot.take().expect("setup produced Some");
            client_cell.borrow_mut().apply_delta(d).expect("apply_delta");
        },
    );

    let pre = cli.precomputed_slots;
    writeln!(
        csv,
        "{arity},{num_buckets},{},{},{},{pre},{:.2},{:.2},{:.2},{:.2}",
        cli.bucket_size, cli.value_bits, cli.batch,
        crit.mean_ops_per_s, crit.min_ops_per_s, crit.max_ops_per_s, crit.stddev_ops_per_s,
    ).unwrap();
    println!(
        "  arity={arity} nb={num_buckets:<6} bs={} vb={:<4} pre={pre:<4} | \
         mean={:.2} dps (±{:.2}) delta={delta_bytes}B",
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

    let mut csv = helpers::csv_writer("ikpir_client_apply_delta_throughput.csv", HEADER);

    match cli.arity {
        2 => run_one::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run_one::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run_one::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_client_apply_delta_throughput.csv");
}
