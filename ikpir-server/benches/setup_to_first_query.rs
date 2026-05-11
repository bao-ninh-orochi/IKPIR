//! **Intent:** Wall-clock latency from "no client state" to "first decoded
//! answer", split into per-phase columns.
//!
//! **Method:** Populate to `TableFull`, snapshot the cell array, then per
//! trial clone via `from_cells` and time each phase independently.
//!
//! **Output:** `results/ikpir_setup_to_first_query.csv`
//! Columns: mode, arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! server_setup_ms, client_from_setup_ms, precompute_b_ms, precompute_c_ms,
//! build_query_ms, answer_ms, decode_ms, total_ms

mod helpers;

use helpers::CloneStore;
use ikpir_client::IkpirClient;
use ikpir_server::{BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirServer};
use segmented_cuckoo::{
    CuckooParams, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

type Client = IkpirClient<FrodoPirBackend>;

const HEADER: &str = "mode,arity,num_buckets,bucket_size,value_bits,lwe_dim,\
    server_setup_ms,client_from_setup_ms,precompute_b_ms,precompute_c_ms,\
    build_query_ms,answer_ms,decode_ms,total_ms";

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Mode { Cold, WarmB, WarmBc }
impl Mode {
    fn as_csv(self) -> &'static str {
        match self { Mode::Cold => "cold", Mode::WarmB => "warm-b", Mode::WarmBc => "warm-bc" }
    }
}

#[derive(Clone, clap::Parser)]
#[command(about = "Measure end-to-end setup-to-first-query latency.")]
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
    #[arg(long, default_value_t = 2)]      warmup: u32,
    #[arg(long, default_value_t = 5)]      trials: u32,
}

#[derive(Default, Clone, Copy)]
struct Phases {
    server_setup_ms: f64,
    client_from_setup_ms: f64,
    precompute_b_ms: f64,
    precompute_c_ms: f64,
    build_query_ms: f64,
    answer_ms: f64,
    decode_ms: f64,
}
impl Phases {
    fn total(&self) -> f64 {
        self.server_setup_ms + self.client_from_setup_ms + self.precompute_b_ms
            + self.precompute_c_ms + self.build_query_ms + self.answer_ms + self.decode_ms
    }
}

fn time_one<S: CloneStore>(
    cli: &Cli, cells: &[u32], params: CuckooParams, n_inserted: u64, mode: Mode,
) -> Phases {
    let store = S::clone_from_cells(cells.to_vec(), params, n_inserted).expect("from_cells");

    let t1 = Instant::now();
    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let server_setup_ms = t1.elapsed().as_secs_f64() * 1e3;

    let bundle = server.setup();
    let t2 = Instant::now();
    let mut client: Client = Client::from_setup(bundle);
    let client_from_setup_ms = t2.elapsed().as_secs_f64() * 1e3;

    let (precompute_b_ms, precompute_c_ms) = match mode {
        Mode::Cold => (0.0, 0.0),
        Mode::WarmB => {
            let t = Instant::now();
            client.precompute_queries(1);
            (t.elapsed().as_secs_f64() * 1e3, 0.0)
        }
        Mode::WarmBc => {
            let t = Instant::now();
            client.precompute_queries(1);
            let b = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            client.precompute_decodes();
            (b, t.elapsed().as_secs_f64() * 1e3)
        }
    };

    let key = 0u32.to_le_bytes();
    let t = Instant::now();
    let q = client.build_query(&key);
    let build_query_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let r = server.answer(&q).expect("answer ok");
    let answer_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let _ = client.decode(&key, &r).expect("decode ok");
    let decode_ms = t.elapsed().as_secs_f64() * 1e3;

    Phases { server_setup_ms, client_from_setup_ms, precompute_b_ms, precompute_c_ms,
             build_query_ms, answer_ms, decode_ms }
}

fn run_one<S: CloneStore>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
) {
    use clap::parser::ValueSource;
    let (_, matches) = helpers::parse_cli_with_matches::<Cli>();

    let (seed_store, n_inserted) = helpers::populate_until_full::<S>(
        num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    );
    if n_inserted == 0 { return; }
    let cells  = seed_store.snapshot_cells();
    let params = seed_store.params();

    // Geometry (build once for preamble).
    let cps = params.cells_per_slot();
    let tmp_store = S::clone_from_cells(cells.clone(), params, n_inserted).expect("from_cells");
    let tmp_server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(tmp_store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let bundle = tmp_server.setup();
    let mut tmp_client = Client::from_setup(bundle.clone());
    let q0 = tmp_client.build_query(&0u32.to_le_bytes());
    let geom = helpers::Geometry {
        hint_per_seg_bytes:       FrodoPirBackend::hint_byte_size(&bundle.hints[0]),
        setup_bundle_bytes:       bundle.wire_byte_size(),
        query_bytes:              q0.wire_byte_size(),
        response_bytes:           tmp_server.answer(&q0).expect("answer").wire_byte_size(),
        hint_delta_typical_bytes: None,
    };
    let store_state = helpers::StoreState {
        capacity:      (num_buckets as u64) * (cli.bucket_size as u64),
        populated:     n_inserted,
        load_pct:      100.0 * n_inserted as f64 / (num_buckets as f64 * cli.bucket_size as f64),
        cells_per_slot: cps,
        row_width:     cli.bucket_size * cps,
        segment_rows:  params.segment_size(),
    };
    let mode_str = cli.mode.as_csv();
    let knobs = [
        helpers::Knob { name: "mode",        value: mode_str.to_string(),         is_default: matches.value_source("mode") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "arity",       value: arity.to_string(),            is_default: matches.value_source("arity") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "num_buckets",  value: num_buckets.to_string(),      is_default: matches.value_source("num_buckets") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "bucket_size",  value: cli.bucket_size.to_string(),  is_default: matches.value_source("bucket_size") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "value_bits",   value: cli.value_bits.to_string(),   is_default: matches.value_source("value_bits") != Some(ValueSource::CommandLine) },
        helpers::Knob { name: "lwe_dim",      value: cli.lwe_dim.to_string(),      is_default: matches.value_source("lwe_dim") != Some(ValueSource::CommandLine) },
    ];
    helpers::print_preamble("setup_to_first_query", &knobs, &store_state, &geom);

    let mut samples: Vec<Phases> = Vec::with_capacity(cli.trials as usize);
    for trial in 0..(cli.warmup + cli.trials) {
        let p = time_one::<S>(cli, &cells, params, n_inserted, cli.mode);
        if trial >= cli.warmup { samples.push(p); }
    }
    let n = samples.len() as f64;
    let mean = |sel: fn(&Phases) -> f64| samples.iter().map(sel).sum::<f64>() / n;
    let (m_srv, m_cli, m_pb, m_pc, m_q, m_a, m_d, m_tot) = (
        mean(|p| p.server_setup_ms),
        mean(|p| p.client_from_setup_ms),
        mean(|p| p.precompute_b_ms),
        mean(|p| p.precompute_c_ms),
        mean(|p| p.build_query_ms),
        mean(|p| p.answer_ms),
        mean(|p| p.decode_ms),
        mean(|p| p.total()),
    );
    writeln!(
        csv,
        "{mode_str},{arity},{num_buckets},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        cli.bucket_size, cli.value_bits, cli.lwe_dim,
        m_srv, m_cli, m_pb, m_pc, m_q, m_a, m_d, m_tot,
    ).unwrap();
    println!(
        "  mode={mode_str:<7} arity={arity} nb={num_buckets:<6} | \
         total={m_tot:.3}ms (srv={m_srv:.3} cli={m_cli:.3} pb={m_pb:.3} pc={m_pc:.3} q={m_q:.3} a={m_a:.3} d={m_d:.3})"
    );
}

fn main() {
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets = if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
        cli.num_buckets
    } else {
        helpers::default_num_buckets_for_arity(cli.arity)
    };

    let mut csv = helpers::csv_writer("ikpir_setup_to_first_query.csv", HEADER);

    match cli.arity {
        2 => run_one::<Segmented2aryScheme>(&mut csv, &cli, 2, num_buckets),
        3 => run_one::<Segmented3aryScheme>(&mut csv, &cli, 3, num_buckets),
        4 => run_one::<Segmented4aryScheme>(&mut csv, &cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
    println!("\nResults written to results/ikpir_setup_to_first_query.csv");
}
