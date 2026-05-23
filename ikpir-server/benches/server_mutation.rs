//! **Intent:** Measure server-side insert / update / delete throughput and
//! total delta wire cost for N mutations per kind, across both backends
//! (FrodoPIR and SimplePIR).
//!
//! **Method:** Populate to `--load-factor` (default 0.80), snapshot the cell
//! array, then for each mutation kind clone the snapshot and time N
//! consecutive mutations with wall-clock. Reports n_succeeded, total_ms,
//! ops_per_s = n_succeeded / total_ms × 1000, and the total delta bytes
//! produced by those mutations.
//!
//! Wall-clock batch timing is used (not Criterion) because store state
//! changes between mutations, making cycling meaningless.
//!
//! **Arguments (CLI):** `--arity` (2/3/4), `--backend` (frodo|simple,
//! default frodo), `--num-buckets`, `--bucket-size`, `--value-bits`,
//! `--lwe-dim` (backend-dependent default), `--n-mutations` (default 1024),
//! `--load-factor` (default 0.80).
//!
//! **Output:** `results/ikpir_server_mutation.csv`
//! Columns: backend, mutation_kind, arity, num_buckets, bucket_size,
//! value_bits, lwe_dim, n_mutations, load_factor, n_attempted, n_succeeded,
//! total_ms, ops_per_s, delta_bytes_total, cells_per_slot, row_width,
//! segment_rows

mod helpers;

use helpers::{Backend, CloneStore};
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer,
    IncrementalPirBackend, IndexPirBackend, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{
    CuckooParams,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const HEADER: &str = "backend,mutation_kind,arity,num_buckets,bucket_size,value_bits,lwe_dim,\
    n_mutations,load_factor,n_attempted,n_succeeded,total_ms,ops_per_s,delta_bytes_total,\
    cells_per_slot,row_width,segment_rows";

#[derive(Clone, Copy, Debug)]
enum MutationKind { Insert, Update, Delete }
impl MutationKind {
    fn as_csv(self) -> &'static str {
        match self { Self::Insert => "insert", Self::Update => "update", Self::Delete => "delete" }
    }
    fn all() -> &'static [Self] { &[Self::Insert, Self::Update, Self::Delete] }
}

#[derive(clap::Parser)]
#[command(about = "Measure server insert/update/delete throughput and delta bytes for N mutations.")]
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
    #[arg(long, default_value_t = 1024)]   n_mutations: u32,
    #[arg(long, default_value_t = 0.80)]   load_factor: f64,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim.unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn fill_value(value: &mut [u8], key: u32, salt: u32) {
    for (i, b) in value.iter_mut().enumerate() {
        *b = (key.wrapping_mul(salt).wrapping_add(i as u32) & 0xFF) as u8;
    }
}

struct KindResult {
    n_attempted:       u32,
    n_succeeded:       u32,
    total_ms:          f64,
    delta_bytes_total: usize,
}

fn run_kind<S, B>(
    cli:         &Cli,
    cells:       &[u32],
    params:      CuckooParams,
    n_seed:      u64,
    kind:        MutationKind,
    make_config: &impl Fn() -> B::Config,
) -> KindResult
where
    S: CloneStore,
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
{
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];

    let store = S::clone_from_cells(cells.to_vec(), params, n_seed).expect("from_cells");
    let mut server: IkpirServer<S, B> = IkpirServer::new(store, make_config());

    let mut delta_bytes_total = 0usize;
    let mut n_succeeded = 0u32;
    let n_attempted = cli.n_mutations;

    let t = Instant::now();
    for i in 0..cli.n_mutations {
        let res = match kind {
            MutationKind::Insert => {
                let k = n_seed as u32 + i;
                fill_value(&mut value, k, 31);
                server.insert(&k.to_le_bytes(), &value)
            }
            MutationKind::Update => {
                let k = (n_seed as u32 - 1) - (i % n_seed as u32);
                fill_value(&mut value, k, 47);
                server.update(&k.to_le_bytes(), &value)
            }
            MutationKind::Delete => {
                let k = (n_seed as u32 - 1) - (i % n_seed as u32);
                server.delete(&k.to_le_bytes())
            }
        };
        match res {
            Ok(bundle) => {
                delta_bytes_total += bundle.wire_byte_size();
                n_succeeded += 1;
            }
            Err(IkpirError::TableFull) => {}
            Err(e) => panic!("server_mutation kind={}: {e:?}", kind.as_csv()),
        }
    }
    let total_ms = t.elapsed().as_secs_f64() * 1e3;

    KindResult { n_attempted, n_succeeded, total_ms, delta_bytes_total }
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
    let lwe_dim_eff = effective_lwe_dim(cli);

    let (seed_store, n_seed) = helpers::populate_to_load::<S>(
        cli.load_factor, num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );

    if n_seed < cli.n_mutations as u64 {
        eprintln!("  Skip: n_seed={n_seed} < n_mutations={}", cli.n_mutations);
        return;
    }

    let cells  = seed_store.snapshot_cells();
    let params = seed_store.params();

    let cps          = params.cells_per_slot();
    let row_width    = cli.bucket_size * cps;
    let segment_rows = params.segment_size();
    let store_state = helpers::StoreState {
        capacity:      (num_buckets as u64) * (cli.bucket_size as u64),
        populated:     n_seed,
        load_pct:      100.0 * n_seed as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes: 0, setup_bundle_bytes: 0,
        query_bytes: 0, response_bytes: 0, hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob { name: "backend",     value: cli.backend.to_string(),           is_default: matches.value_source("backend") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "arity",       value: arity.to_string(),                 is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets", value: num_buckets.to_string(),           is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size", value: cli.bucket_size.to_string(),       is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",  value: cli.value_bits.to_string(),        is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",     value: lwe_dim_eff.to_string(),           is_default: cli.lwe_dim.is_none() },
        helpers::Knob { name: "n_mutations", value: cli.n_mutations.to_string(),       is_default: matches.value_source("n_mutations") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "load_factor", value: format!("{:.2}", cli.load_factor), is_default: matches.value_source("load_factor") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("server_mutation", &knobs, &store_state, &geom);

    for &kind in MutationKind::all() {
        let r = run_kind::<S, B>(cli, &cells, params, n_seed, kind, &make_config);
        let ops_per_s = if r.total_ms > 0.0 { r.n_succeeded as f64 / r.total_ms * 1e3 } else { 0.0 };
        writeln!(
            csv,
            "{},{},{arity},{num_buckets},{},{},{lwe_dim_eff},{},{:.2},{},{},{:.3},{:.2},{},{cps},{row_width},{segment_rows}",
            cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits,
            cli.n_mutations, cli.load_factor,
            r.n_attempted, r.n_succeeded, r.total_ms, ops_per_s, r.delta_bytes_total,
        ).unwrap();
        println!(
            "  backend={} kind={:<6} arity={arity} nb={num_buckets:<7} N={:<4} | \
             succ={}/{} total={:.1}ms ops/s={:.0} delta={}B",
            cli.backend, kind.as_csv(), cli.n_mutations,
            r.n_succeeded, r.n_attempted, r.total_ms, ops_per_s, r.delta_bytes_total,
        );
    }
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

    let mut csv = helpers::csv_writer("ikpir_server_mutation.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_mutation.csv");
}
