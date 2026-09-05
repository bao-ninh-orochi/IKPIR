//! Shared generic measurement body for the per-flow head-to-head decode
//! benches (`headtohead_hint_patch_decode.rs`, `headtohead_rewind_decode.rs`),
//! included via `mod flow_headtohead_decode_body;` the same way `mod
//! helpers;` is. Each thin binary supplies its own `C:
//! helpers::ClientFlow<B>` at the two backend-dispatch call sites;
//! everything else lives here, once.
//!
//! **Intent:** Head-to-head counterpart of `flow_decode_body.rs` — measure
//! one flow's `decode` throughput at a **fixed keyword count**, for the fair
//! comparison against ChalametPIR and Hao et al. 2025. See
//! `ikpir-server/benches/headtohead_answer.rs` for the motivation behind
//! fixing `num_keys` instead of DB size, and for the two key-count regimes
//! `scripts/table3.sh` sweeps.
//!
//! **Method:** Identical warm-bc pipeline as `flow_decode_body.rs` except
//! `populate_exact_n_keys(target_n = num_keys)` replaces `populate_until_full`,
//! and the CSV carries the extra `num_keys` / `db_size` / `fingerprint_bits`
//! columns. A once-per-config `verify_decode` guards against silent decode
//! regressions before the timed loop, flow-generic via `helpers::ClientFlow`.
//!
//! **Arguments (CLI):** Same as `flow_decode_body.rs`, plus `--num-keys`
//! (required).
//!
//! **Output:** `results/ikpir_headtohead_client_hint_patch_decode.csv` /
//! `results/ikpir_headtohead_client_rewind_decode.csv` (never merged — the
//! CSV file name is derived from `C::FLOW`).

// significant_drop_tightening: clippy's inline-`Criterion` fix borrows a temporary dropped while `BenchmarkGroup` holds it (won't compile).
#![allow(clippy::significant_drop_tightening)]

use crate::helpers::{self, Backend, ClientFlow, MakeStore};
use criterion::Throughput;
use ikpir_client::{
    BackendWireSize, IncrementalPirBackend, IndexPirBackend, ParallelSetupBackend,
    PrecomputingPirBackend,
};
use ikpir_server::IkpirServer;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const HEADER: &str =
    "flow,backend,arity,num_buckets,bucket_size,num_keys,db_size,value_bits,plaintext_bits,fingerprint_bits,lwe_dim,batch,\
    mean_dps,min_dps,max_dps,stddev_dps,\
    cells_per_slot,row_width,segment_rows,db_rows,db_cols,load_factor";

#[derive(Clone, clap::Parser)]
#[command(
    about = "Head-to-head client decode bench for one flow: fixes num_keys, reports DB size, otherwise mirrors flow_decode_body."
)]
pub struct Cli {
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    pub arity: u32,
    #[arg(long, value_enum, default_value_t = Backend::Frodo)]
    pub backend: Backend,
    /// Required: number of keys to populate. The DB size (= `num_buckets ×
    /// bucket_size`) is fixed by `--num-buckets` / `--bucket-size`; pick those
    /// so capacity ≥ `num_keys` at a reasonable load factor.
    #[arg(long)]
    pub num_keys: u64,
    #[arg(long, default_value_t = 16_384)]
    pub num_buckets: u32,
    #[arg(long, default_value_t = 4)]
    pub bucket_size: u32,
    /// Value width in bits. The paper reports 2048 (256 B) and 8192 (1 kB).
    #[arg(long, default_value_t = 2048)]
    pub value_bits: u32,
    #[arg(long, default_value_t = 64)]
    pub fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]
    pub plaintext_bits: u32,
    /// LWE dimension. Defaults to 1566 (Frodo) or 1275 (Simple) when omitted.
    #[arg(long)]
    pub lwe_dim: Option<u32>,
    /// Key-pool size: the bench rotates through this many distinct keys so
    /// repeated iterations do not reuse hot CPU-cache state from the previous call.
    #[arg(long, default_value_t = 64)]
    pub batch: u32,
}

pub fn effective_lwe_dim(cli: &Cli) -> u32 {
    cli.lwe_dim
        .unwrap_or_else(|| helpers::backend_default_lwe_dim(cli.backend))
}

pub fn run_one<S, B, C>(cli: &Cli, arity: u32, num_buckets: u32, backend_config: B::Config)
where
    S: MakeStore,
    B: IndexPirBackend
        + ParallelSetupBackend
        + IncrementalPirBackend
        + PrecomputingPirBackend
        + BackendWireSize
        + Clone,
    B::Query: Clone,
    B::Response: Clone,
    C: ClientFlow<B>,
{
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

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

    // ── 2. Server setup; `answer` never reads the server's `A`, so free it
    //       right after the bundle. The client materialises its own copy. ───────
    let mut server: IkpirServer<S, B> = IkpirServer::new_parallel(store, backend_config);
    let bundle = server.setup();
    server.drop_hint_material();

    let (query_bytes, response_bytes) = {
        let mut probe: C = C::from_setup_parallel(bundle.clone());
        let q = probe.build_query(&0u32.to_le_bytes());
        let rb = server.answer(&q).expect("answer ok").wire_byte_size();
        (q.wire_byte_size(), rb)
    };

    // ── 3. Geometry ─────────────────────────────────────────────────────────────
    let params_store = server.params();
    let cps = params_store.cells_per_slot();
    let row_width = cli.bucket_size * cps;
    let segment_rows = params_store.segment_size();
    let (db_rows, db_cols) = B::db_matrix_shape(&server.backend_params()[0]);
    let db_size = (num_buckets as u64) * (cli.bucket_size as u64);
    let load_factor = n_inserted as f64 / db_size as f64;
    let lwe_dim_eff = effective_lwe_dim(cli);
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
    let flow_slug = C::FLOW.replace('-', "_");
    let flow_suffix = flow_slug
        .strip_prefix("client_")
        .expect("ClientFlow::FLOW starts with \"client-\"");
    let bench_name = format!("headtohead_{flow_suffix}_decode");
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
    helpers::print_preamble(&bench_name, &knobs, &store_state, &geom);

    // ── 4. Measure decode (warm-bc) ──────────────────────────────────────────────
    let n = n_inserted as u32;
    let keys: Vec<Vec<u8>> = (0..cli.batch)
        .map(|i| (i % n).to_le_bytes().to_vec())
        .collect();
    let mut client: C = C::from_setup_parallel(bundle);

    // Once-per-config decode sanity check — catches silent decode regressions
    // (wrong cells_per_slot, packing bug, hint mismatch, bad plaintext_bits
    // operating point) before the long-running timed loop.
    helpers::verify_decode::<B, _, C>(&mut client, &server, 1u32, 16, cli.value_bits);

    let samples: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let mut idx = 0usize;
    {
        let mut c = helpers::configured_criterion();
        let mut group = c.benchmark_group(bench_name.clone());
        group.throughput(Throughput::Elements(1));
        group.bench_function(bench_name.clone(), |b| {
            b.iter_custom(|iters| {
                client.precompute_queries(iters as u32);
                client.precompute_decodes();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let k = keys[idx % keys.len()].clone();
                    idx = idx.wrapping_add(1);
                    let q = client.build_query(&k);
                    let r = server.answer(&q).expect("answer ok");
                    let t = Instant::now();
                    let _ = client.decode(&k, &q, &r).expect("decode ok");
                    total += t.elapsed();
                }
                samples
                    .lock()
                    .unwrap()
                    .push(total.as_nanos() as f64 / iters as f64);
                total
            });
        });
        group.finish();
    }
    let raw = std::mem::take(&mut *samples.lock().unwrap());
    let crit = if raw.is_empty() {
        helpers::CriterionThroughputStats {
            mean_ops_per_s: 0.0,
            min_ops_per_s: 0.0,
            max_ops_per_s: 0.0,
            stddev_ops_per_s: 0.0,
        }
    } else {
        let ops: Vec<f64> = raw.iter().map(|&ns| 1e9 / ns).collect();
        let s = helpers::compute_stats(&ops);
        helpers::CriterionThroughputStats {
            mean_ops_per_s: s.mean,
            min_ops_per_s: s.min,
            max_ops_per_s: s.max,
            stddev_ops_per_s: s.stddev,
        }
    };

    // ── 5. Write CSV ─────────────────────────────────────────────────────────────
    let num_keys = cli.num_keys;
    let csv_name = format!("ikpir_headtohead_{flow_slug}_decode.csv");
    let mut csv = helpers::csv_writer(&csv_name, HEADER);
    writeln!(
        csv,
        "{},{},{arity},{num_buckets},{},{num_keys},{db_size},{},{},{},{lwe_dim_eff},{},{:.2},{:.2},{:.2},{:.2},{cps},{row_width},{segment_rows},{db_rows},{db_cols},{:.4}",
        C::FLOW, cli.backend, cli.bucket_size, cli.value_bits, cli.plaintext_bits, cli.fingerprint_bits, cli.batch,
        crit.mean_ops_per_s, crit.min_ops_per_s, crit.max_ops_per_s, crit.stddev_ops_per_s,
        load_factor,
    ).unwrap();
    println!(
        "  flow={} backend={} arity={arity} nb={num_buckets:<7} N={num_keys} vb={:<4} | \
         mean={:.2} dps (±{:.2})",
        C::FLOW,
        cli.backend,
        cli.value_bits,
        crit.mean_ops_per_s,
        crit.stddev_ops_per_s,
    );
    println!("\nResults written to results/{csv_name}");
}
