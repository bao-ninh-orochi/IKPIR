//! **Intent:** Head-to-head counterpart of `server_answer` — measure
//! server-side `answer` throughput and wire sizes at a **fixed keyword count**,
//! for the fair comparison against ChalametPIR and Hao et al. 2025.
//!
//! The paper reports it at `--num-keys` = 10^6, the count both baselines
//! publish at (`scripts/table3.sh`). The arity-3 KV-SCF shapes cannot hold 10^6
//! keys at a comparable fill and have no baseline to match, so that sweep runs
//! them at 90% fill instead — a different `--num-keys`, same bench. The CSV's
//! `num_keys` / `db_size` columns record which regime a row came from.
//!
//! **Motivation:** `server_answer` fixes `num_buckets` (and thus DB size) and
//! populates `until_full`, so different schemes store different keyword counts
//! in the same-size DB — a setting where IKPIR's slot/cell layout wins by
//! construction. For a fair head-to-head we flip the protocol: fix `num_keys`
//! and let each scheme report the DB size (= `num_buckets × bucket_size`) it
//! needed to absorb that load.
//!
//! **Method:** Identical to `server_answer` except `populate_exact_n_keys(
//! target_n = num_keys)` replaces `populate_until_full`, and the CSV carries
//! the extra `num_keys` / `db_size` / `fingerprint_bits` columns.
//!
//! **Arguments (CLI):** Same as `server_answer`, plus `--num-keys` (required,
//! the keyword count to populate) and `--max-mem-gb` (OOM guard).
//!
//! **Output:** `results/ikpir_headtohead_server_answer.csv`

// significant_drop_tightening: clippy's inline-`Criterion` fix borrows a temporary dropped while `BenchmarkGroup` holds it (won't compile).
#![allow(clippy::significant_drop_tightening)]

mod helpers;

use helpers::{Backend, MakeStore};
use ikpir_client::IkpirClient;
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer, IncrementalPirBackend,
    IndexPirBackend, ParallelSetupBackend, PirQueryBundle, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use std::io::Write;

const HEADER: &str =
    "backend,arity,num_buckets,bucket_size,num_keys,db_size,value_bits,plaintext_bits,fingerprint_bits,lwe_dim,batch,\
    mean_qps,min_qps,max_qps,stddev_qps,query_bytes,response_bytes,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

#[derive(clap::Parser)]
#[command(
    about = "Head-to-head server answer bench: fixes num_keys, reports DB size, otherwise mirrors server_answer."
)]
struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    backend: Backend,
    /// Required: number of keys to populate. The DB size (= `num_buckets ×
    /// bucket_size`) is fixed by `--num-buckets` / `--bucket-size`; the caller
    /// picks those so capacity ≥ `num_keys` at a reasonable load factor — 0.954
    /// for the paper's arity-2/4 shapes at 10^6 keys, 0.90 for its arity-3
    /// ones. Both stay under every achieved load factor of Table 2.
    #[arg(long)]
    num_keys: u64,
    #[arg(long, default_value_t = 16_384)]
    num_buckets: u32,
    #[arg(long, default_value_t = 4)]
    bucket_size: u32,
    /// Value width in bits. The paper reports 2048 (256 B) and 8192 (1 kB).
    #[arg(long, default_value_t = 2048)]
    value_bits: u32,
    #[arg(long, default_value_t = 32)]
    fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]
    plaintext_bits: u32,
    /// LWE dimension. Defaults to 1566 (Frodo) or 1275 (Simple) when omitted.
    #[arg(long)]
    lwe_dim: Option<u32>,
    #[arg(long, default_value_t = 64)]
    batch: u32,
    /// Skip configs whose estimated peak memory exceeds this limit. Default
    /// 85.0 is tuned for a ~96 GB server; lower it on smaller machines.
    #[arg(long, default_value_t = 85.0)]
    max_mem_gb: f64,
}

fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim
        .unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

fn run_one<S, B>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    backend_config: B::Config,
) where
    S: MakeStore,
    B: IndexPirBackend + ParallelSetupBackend + IncrementalPirBackend + BackendWireSize,
    B::Query: Clone,
    B::Response: Clone,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    // ── 0. Memory guard ─────────────────────────────────────────────────────────
    // The dominant term (per-segment LWE matrix `A`, held in B::HintMaterial)
    // is independent of how many keys we insert, so the formula is unchanged
    // from the fixed-DB benches.
    let lwe_dim_est = effective_lwe_dim(cli) as u64;
    let cells_per_slot_est =
        (cli.fingerprint_bits + cli.value_bits).div_ceil(cli.plaintext_bits) as u64;
    let row_width_est = cli.bucket_size as u64 * cells_per_slot_est;
    let n_rows_per_seg = num_buckets as u64 / arity as u64;
    let table_bytes = num_buckets as u64 * cli.bucket_size as u64 * cells_per_slot_est * 4;
    let (a_rows_per_seg, _c_len_per_seg) =
        helpers::backend_shape_estimate(cli.backend, n_rows_per_seg, row_width_est);
    let a_bytes = arity as u64 * a_rows_per_seg * lwe_dim_est * 4;
    let queries_bytes = cli.batch as u64 * arity as u64 * a_rows_per_seg * 4;
    let estimated_gb = (table_bytes + a_bytes + queries_bytes) as f64 / 1e9;
    if estimated_gb > cli.max_mem_gb {
        eprintln!(
            "  Skip (OOM guard): estimated peak {:.1} GB > --max-mem-gb {:.1} \
             (nb={num_buckets} bs={} vb={} lwe_dim={lwe_dim_est} backend={}). \
             Raise --max-mem-gb on machines with more RAM.",
            estimated_gb, cli.max_mem_gb, cli.bucket_size, cli.value_bits, cli.backend,
        );
        return;
    }

    // ── 1. Populate exactly num_keys ────────────────────────────────────────────
    let (store, n_inserted) = helpers::populate_exact_n_keys::<S>(
        num_buckets,
        cli.bucket_size,
        cli.fingerprint_bits,
        cli.value_bits,
        cli.plaintext_bits,
        cli.num_keys,
    );
    assert_eq!(
        n_inserted, cli.num_keys,
        "populate_exact_n_keys must return target_n"
    );

    // ── 2. Build server, queries, and drop the seed-derived `A` ────────────────
    // `answer` does not read the LWE matrix `A`, so a read-only bench frees it
    // right after building the queries to keep peak RAM to a single copy.
    let mut server: IkpirServer<S, B> = IkpirServer::new_parallel(store, backend_config);
    let n = n_inserted as u32;
    let queries: Vec<PirQueryBundle<B>> = {
        let mut client: IkpirClient<B> = IkpirClient::from_setup_parallel(server.setup());
        (0..cli.batch)
            .map(|i| client.build_query(&((i % n).to_le_bytes())))
            .collect()
    };
    server.drop_hint_material();

    let query_bytes = queries[0].wire_byte_size();
    let response_bytes = server
        .answer(&queries[0])
        .expect("answer ok")
        .wire_byte_size();

    // ── 3. Geometry ─────────────────────────────────────────────────────────────
    let params = server.params();
    let cps = params.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let segment_rows = params.segment_size();
    let (db_rows, db_cols) =
        helpers::backend_shape_estimate(cli.backend, segment_rows as u64, row_width as u64);
    let db_size = (num_buckets as u64) * (cli.bucket_size as u64);
    let load_factor = n_inserted as f64 / db_size as f64;
    let lwe_dim_eff = effective_lwe_dim(cli);
    let bundle = server.setup();
    let store_state = helpers::StoreState {
        capacity: db_size,
        populated: n_inserted,
        load_pct: 100.0 * load_factor,
        cells_per_slot: cps,
        row_width,
        segment_rows,
    };
    let geom = helpers::Geometry {
        hint_per_seg_bytes: B::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes: bundle.wire_byte_size(),
        query_bytes,
        response_bytes,
        hint_delta_typical_bytes: None,
    };
    let knobs = [
        helpers::Knob {
            name: "backend",
            value: cli.backend.to_string(),
            is_default: matches.value_source("backend") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "arity",
            value: arity.to_string(),
            is_default: matches.value_source("arity") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "num_keys",
            value: cli.num_keys.to_string(),
            is_default: false,
        },
        helpers::Knob {
            name: "num_buckets",
            value: num_buckets.to_string(),
            is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "bucket_size",
            value: cli.bucket_size.to_string(),
            is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "fingerprint_bits",
            value: cli.fingerprint_bits.to_string(),
            is_default: matches.value_source("fingerprint_bits") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "value_bits",
            value: cli.value_bits.to_string(),
            is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "plaintext_bits",
            value: cli.plaintext_bits.to_string(),
            is_default: matches.value_source("plaintext_bits") != Some(ValueSource::CommandLine),
        },
        helpers::Knob {
            name: "lwe_dim",
            value: lwe_dim_eff.to_string(),
            is_default: cli.lwe_dim.is_none(),
        },
        helpers::Knob {
            name: "batch",
            value: cli.batch.to_string(),
            is_default: matches.value_source("batch") != Some(ValueSource::CommandLine),
        },
    ];
    helpers::print_preamble("headtohead_answer", &knobs, &store_state, &geom);

    // ── 4. Measure server_answer ────────────────────────────────────────────────
    let mut idx = 0usize;
    let crit = helpers::run_criterion_throughput("headtohead_answer", 1, || {
        let _ = server.answer(&queries[idx]).expect("answer ok");
        idx = (idx + 1) % queries.len();
    });

    // ── 5. Write CSV ─────────────────────────────────────────────────────────────
    let num_keys = cli.num_keys;
    writeln!(
        csv,
        "{},{arity},{num_buckets},{},{num_keys},{db_size},{},{},{},{lwe_dim_eff},{},{:.2},{:.2},{:.2},{:.2},{query_bytes},{response_bytes},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits, cli.fingerprint_bits, cli.batch,
        crit.mean_ops_per_s, crit.min_ops_per_s, crit.max_ops_per_s, crit.stddev_ops_per_s,
        load_factor,
    ).unwrap();
    println!(
        "  backend={} arity={arity} nb={num_buckets:<7} N={num_keys} bs={} vb={:<4} | \
         mean={:.2} qps (±{:.2}) | q={query_bytes}B r={response_bytes}B",
        cli.backend, cli.bucket_size, cli.value_bits, crit.mean_ops_per_s, crit.stddev_ops_per_s,
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
        Backend::Frodo => run_one::<S, FrodoPirBackend>(
            csv,
            cli,
            arity,
            num_buckets,
            FrodoConfig::with_lwe_dim(lwe_dim),
        ),
        Backend::Simple => run_one::<S, SimplePirBackend>(
            csv,
            cli,
            arity,
            num_buckets,
            SimpleConfig::with_lwe_dim(lwe_dim),
        ),
    }
}

fn main() {
    if helpers::skip_when_cargo_test() {
        return;
    }
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets =
        if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
            cli.num_buckets
        } else {
            helpers::default_num_buckets_for_arity(cli.arity)
        };

    let mut csv = helpers::csv_writer("ikpir_headtohead_server_answer.csv", HEADER);

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to ikpir_headtohead_server_answer.csv");
}
