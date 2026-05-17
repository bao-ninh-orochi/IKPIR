//! **Intent:** Compare the wall-clock and wire cost of `N` incremental
//! hint patches against a single `full_rebuild` for the same total
//! mutation footprint, split by mutation kind (insert / update /
//! delete). Generic over both shipped backends (FrodoPIR / SimplePIR).
//!
//! **Method:** Populate once to `--load-factor` (default `0.80`),
//! snapshot the cell array, then for each `(mutation_kind, path)`
//! clone via `from_cells`. This avoids re-populating for each path.
//!
//! Insert path may fail at high load; tracks `n_attempted` and
//! `n_succeeded`. At `--load-factor full`, the insert row is emitted
//! with `n_succeeded=0`, `incremental_ms=NaN`,
//! `ratio_inc_over_rebuild=NaN`.
//!
//! **Arguments (CLI):** `--arity`, `--backend` (frodo|simple, default
//! frodo), `--num-buckets`, `--bucket-size`, `--value-bits`, `--lwe-dim`
//! (defaults to backend recommendation), `--load-factor` (`0.0..=1.0`
//! or `"full"`), `--n-mutations`. See `helpers::parse_cli` for the rest.
//!
//! **Design rationale:** Headline plot for the paper — the crossover
//! point where `N · incremental_patch_ms` overtakes a single
//! `full_rebuild_ms` is what determines when a deployment should drop
//! the patch stream and force a resync. The `delta_to_hint_ratio`
//! column quantifies the wire saving on the patch path. Backend
//! parameterisation lets us compare both backends against the same
//! mutation footprint.
//!
//! **Output:** `results/ikpir_server_incremental_vs_rebuild.csv`
//! Columns: backend, mutation_kind, arity, num_buckets, bucket_size,
//! value_bits, n_mutations, n_attempted, n_succeeded, incremental_ms,
//! incremental_per_op_ms, rebuild_ms, ratio_inc_over_rebuild,
//! delta_bytes_total, rebuild_hint_bytes, delta_to_hint_ratio

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

const HEADER: &str = "backend,mutation_kind,arity,num_buckets,bucket_size,value_bits,\
    n_mutations,n_attempted,n_succeeded,incremental_ms,incremental_per_op_ms,\
    rebuild_ms,ratio_inc_over_rebuild,delta_bytes_total,rebuild_hint_bytes,delta_to_hint_ratio";

#[derive(Clone, Debug)]
enum LoadFactor { Value(f64), Full }

fn parse_lf(s: &str) -> Result<LoadFactor, String> {
    if s == "full" { return Ok(LoadFactor::Full); }
    s.parse::<f64>().map(LoadFactor::Value).map_err(|e| e.to_string())
}

#[derive(Clone, Copy, Debug)]
enum MutationKind { Insert, Update, Delete }
impl MutationKind {
    fn as_csv(self) -> &'static str {
        match self { Self::Insert => "insert", Self::Update => "update", Self::Delete => "delete" }
    }
    fn all() -> &'static [Self] { &[Self::Insert, Self::Update, Self::Delete] }
}

#[derive(clap::Parser)]
#[command(about = "Compare incremental hint patching vs full rebuild for N mutations.")]
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
    #[arg(long, default_value_t = 1024)]   n_mutations: u32,
    #[arg(long, value_parser = parse_lf, default_value = "0.80")]
    load_factor: LoadFactor,
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
    n_attempted:        u32,
    n_succeeded:        u32,
    incremental_ms:     f64,
    rebuild_ms:         f64,
    delta_bytes_total:  usize,
    rebuild_hint_bytes: usize,
}

fn run_kind<S, B>(
    cli:          &Cli,
    cells:        &[u32],
    params:       CuckooParams,
    n_seed:       u64,
    n_mut:        u32,
    kind:         MutationKind,
    make_config:  &impl Fn() -> B::Config,
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
    let n_attempted = n_mut;
    let inc_start = Instant::now();
    for i in 0..n_mut {
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
            Err(IkpirError::TableFull) => { /* count as attempted, not succeeded */ }
            Err(e) => panic!("incremental path: {e:?}"),
        }
    }
    let incremental_ms = if n_succeeded > 0 {
        inc_start.elapsed().as_secs_f64() * 1e3
    } else {
        f64::NAN
    };

    let store2 = S::clone_from_cells(cells.to_vec(), params, n_seed).expect("from_cells");
    let mut server2: IkpirServer<S, B> = IkpirServer::new(store2, make_config());
    for i in 0..n_mut {
        let _ = match kind {
            MutationKind::Insert => {
                let k = n_seed as u32 + i;
                fill_value(&mut value, k, 31);
                server2.insert(&k.to_le_bytes(), &value).ok()
            }
            MutationKind::Update => {
                let k = (n_seed as u32 - 1) - (i % n_seed as u32);
                fill_value(&mut value, k, 47);
                server2.update(&k.to_le_bytes(), &value).ok()
            }
            MutationKind::Delete => {
                let k = (n_seed as u32 - 1) - (i % n_seed as u32);
                server2.delete(&k.to_le_bytes()).ok()
            }
        };
    }
    let reb_start = Instant::now();
    let rebuild_bundle = server2.full_rebuild();
    let rebuild_ms = reb_start.elapsed().as_secs_f64() * 1e3;
    let rebuild_hint_bytes: usize = rebuild_bundle.hints.iter()
        .map(B::hint_byte_size).sum();

    KindResult { n_attempted, n_succeeded, incremental_ms, rebuild_ms,
                 delta_bytes_total, rebuild_hint_bytes }
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
    B: IndexPirBackend + IncrementalPirBackend + BackendWireSize,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();
    let lwe_dim_eff = effective_lwe_dim(cli);

    let (seed_store, n_seed) = match &cli.load_factor {
        LoadFactor::Full => helpers::populate_until_full::<S>(
            num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
        ),
        LoadFactor::Value(lf) => helpers::populate_to_load::<S>(
            *lf, num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
        ),
    };
    let cells  = seed_store.snapshot_cells();
    let params = seed_store.params();

    let cps = params.cells_per_slot();
    let store_state = helpers::StoreState {
        capacity:      (num_buckets as u64) * (cli.bucket_size as u64),
        populated:     n_seed,
        load_pct:      100.0 * n_seed as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width:     cli.bucket_size * cps,
        segment_rows:  params.segment_size(),
    };
    let lf_str = match &cli.load_factor {
        LoadFactor::Full => "full".to_string(),
        LoadFactor::Value(v) => format!("{v:.2}"),
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes: 0, setup_bundle_bytes: 0,
        query_bytes: 0, response_bytes: 0, hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob { name: "backend",      value: cli.backend.to_string(),     is_default: matches.value_source("backend") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "arity",        value: arity.to_string(),           is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",  value: num_buckets.to_string(),     is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",  value: cli.bucket_size.to_string(), is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",   value: cli.value_bits.to_string(),  is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",      value: lwe_dim_eff.to_string(),     is_default: cli.lwe_dim.is_none() },
        helpers::Knob { name: "n_mutations",  value: cli.n_mutations.to_string(), is_default: matches.value_source("n_mutations") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "load_factor",  value: lf_str,                      is_default: matches.value_source("load_factor") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("incremental_vs_rebuild", &knobs, &store_state, &geom);

    if n_seed < cli.n_mutations as u64 {
        eprintln!("  Skip: n_seed={n_seed} < n_mutations={}", cli.n_mutations);
        return;
    }

    for &kind in MutationKind::all() {
        let r = run_kind::<S, B>(cli, &cells, params, n_seed, cli.n_mutations, kind, &make_config);
        let ratio_time = if r.rebuild_ms > 0.0 { r.incremental_ms / r.rebuild_ms } else { f64::NAN };
        let inc_per_op = if r.n_succeeded > 0 {
            r.incremental_ms / r.n_succeeded as f64
        } else { f64::NAN };
        let ratio_bytes = if r.rebuild_hint_bytes > 0 {
            r.delta_bytes_total as f64 / r.rebuild_hint_bytes as f64
        } else { f64::NAN };
        writeln!(
            csv,
            "{},{},{arity},{num_buckets},{},{},{},{},{},{:.3},{:.6},{:.3},{:.4},{},{},{:.6}",
            cli.backend, kind.as_csv(), cli.bucket_size, cli.value_bits, cli.n_mutations,
            r.n_attempted, r.n_succeeded,
            r.incremental_ms, inc_per_op, r.rebuild_ms, ratio_time,
            r.delta_bytes_total, r.rebuild_hint_bytes, ratio_bytes,
        ).unwrap();
        println!(
            "  backend={} kind={:<6} arity={arity} num_buckets={num_buckets:<6} N={} | \
             inc={:.3}ms ({:.4}ms/op)  rebuild={:.3}ms  ratio={:.3}  succ={}/{}  delta={}B  hint={}B",
            cli.backend, kind.as_csv(), cli.n_mutations,
            r.incremental_ms, inc_per_op, r.rebuild_ms, ratio_time,
            r.n_succeeded, r.n_attempted,
            r.delta_bytes_total, r.rebuild_hint_bytes,
        );
    }
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

    let mut csv = helpers::csv_writer("ikpir_server_incremental_vs_rebuild.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_server_incremental_vs_rebuild.csv");
}
