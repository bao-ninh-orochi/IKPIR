//! **Intent:** Catalog the minimum on-wire byte size of every IKPIR bundle
//! shape, sweeping `arity × num_buckets × bucket_size × value_bits × lwe_dim`.
//!
//! No timing — pure structural accounting via `BackendWireSize` /
//! `wire_byte_size` helpers. Three `HintDeltaBundle` flavours are sampled
//! per row: one insert, one update, one delete on disjoint keys.
//!
//! **Method:** For each config, build and populate a server to ~40% load,
//! snapshot the setup bundle. Build a same-process client and sample one
//! query + one response. Then issue one `insert` on a fresh key, one
//! `update` on a seeded key, and one `delete` on a different seeded key —
//! each returns a `HintDeltaBundle` with `wire_byte_size`.
//!
//! **Invocation:** `cargo bench --bench wire_sizes` runs a single default
//! config. Pass `--sweep` for the full matrix, or specific flags to pin a
//! config.
//!
//! **Output:** `results/ikpir_wire_sizes.csv`
//! Columns: arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! setup_bundle_bytes, query_bytes, response_bytes,
//! hint_delta_insert_bytes, hint_delta_update_bytes, hint_delta_delete_bytes

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirServer};
use segmented_cuckoo::{
    CuckooError,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;

type Client = IkpirClient<FrodoPirBackend>;
const MAX_KICKS: u32 = 2_500;

#[derive(clap::Parser)]
#[command(about = "Catalog IKPIR bundle wire sizes (no timing).")]
struct Cli {
    /// Run the full hardcoded matrix.
    #[arg(long)]
    sweep: bool,

    /// Cuckoo arity (2, 3, or 4). With --sweep and no --arity, sweep all three.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))]
    arity: Option<u32>,

    #[arg(long, default_value_t = 256)]  num_buckets: u32,
    #[arg(long, default_value_t = 4)]    bucket_size: u32,
    #[arg(long, default_value_t = 64)]   value_bits: u32,
    #[arg(long, default_value_t = 12)]   fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]    plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)] lwe_dim: u32,
}

fn run_one<S>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    bucket_size: u32,
    value_bits:  u32,
)
where S: MakeStore {
    let mut store = match S::make_store(
        num_buckets, bucket_size, cli.fingerprint_bits, value_bits, cli.plaintext_bits,
    ) {
        Ok(s)  => s,
        Err(_) => { eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}"); return; }
    };
    store.set_max_kicks(MAX_KICKS);

    let vsize = (value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    // Seed ~40% load so update/delete have targets and insert has headroom.
    let n_seed: u32 = ((num_buckets * bucket_size) as f64 * 0.40) as u32;
    let mut last_seeded: u32 = 0;
    for k in 0u32..n_seed {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k.to_le_bytes(), &value) {
            Ok(())                       => last_seeded = k,
            Err(CuckooError::TableFull)  => break,
            Err(e)                       => panic!("seed: {e:?}"),
        }
    }
    if last_seeded < 2 {
        eprintln!("  Skip arity={arity} num_buckets={num_buckets} bs={bucket_size} vb={value_bits}: seed too small");
        return;
    }

    let mut server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let mut client: Client = Client::from_setup(server.setup());

    let setup_bundle_bytes = server.setup().wire_byte_size();
    let q = client.build_query(&0u32.to_le_bytes());
    let query_bytes = q.wire_byte_size();
    let r = server.answer(&q).expect("answer ok");
    let response_bytes = r.wire_byte_size();

    // Disjoint keys: insert on a fresh slot, update on key 0, delete on key 1.
    let insert_key = (last_seeded + 1).to_le_bytes();
    let ins_delta = server.insert(&insert_key, &value).expect("insert ok");
    let hint_delta_insert_bytes = ins_delta.wire_byte_size();

    let upd_delta = server.update(&0u32.to_le_bytes(), &value).expect("update ok");
    let hint_delta_update_bytes = upd_delta.wire_byte_size();

    let del_delta = server.delete(&1u32.to_le_bytes()).expect("delete ok");
    let hint_delta_delete_bytes = del_delta.wire_byte_size();

    writeln!(
        csv,
        "{arity},{num_buckets},{bucket_size},{value_bits},{},{setup_bundle_bytes},{query_bytes},{response_bytes},{hint_delta_insert_bytes},{hint_delta_update_bytes},{hint_delta_delete_bytes}",
        cli.lwe_dim,
    )
    .unwrap();
    println!(
        "  arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} | \
         setup={setup_bundle_bytes}B q={query_bytes}B r={response_bytes}B \
         dI={hint_delta_insert_bytes}B dU={hint_delta_update_bytes}B dD={hint_delta_delete_bytes}B"
    );
}

fn dispatch(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
) {
    match arity {
        2 => run_one::<Segmented2aryScheme>(csv, cli, 2, num_buckets, bucket_size, value_bits),
        3 => run_one::<Segmented3aryScheme>(csv, cli, 3, num_buckets, bucket_size, value_bits),
        4 => run_one::<Segmented4aryScheme>(csv, cli, 4, num_buckets, bucket_size, value_bits),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
}

fn sweep_buckets(arity: u32) -> &'static [u32] {
    match arity {
        2 | 4 => &[64, 256, 1024],
        3     => &[96, 384, 1536],
        _     => unreachable!(),
    }
}

fn resolve_num_buckets(cli: &Cli, arity: u32) -> u32 {
    if cli.num_buckets == 256 && arity == 3 { 384 } else { cli.num_buckets }
}

fn main() {
    let cli: Cli = helpers::parse_cli();
    let mut csv = helpers::csv_writer(
        "ikpir_wire_sizes.csv",
        "arity,num_buckets,bucket_size,value_bits,lwe_dim,setup_bundle_bytes,query_bytes,response_bytes,hint_delta_insert_bytes,hint_delta_update_bytes,hint_delta_delete_bytes",
    );

    println!("=== ikpir wire sizes (FrodoPirBackend) ===");
    println!(
        "Config: fingerprint_bits={}, plaintext_bits={}, lwe_dim={}",
        cli.fingerprint_bits, cli.plaintext_bits, cli.lwe_dim,
    );

    let arities: Vec<u32> = match (cli.sweep, cli.arity) {
        (true,  None)    => vec![2, 3, 4],
        (true,  Some(a)) => vec![a],
        (false, opt)     => vec![opt.unwrap_or(2)],
    };

    for &arity in &arities {
        if cli.sweep {
            for &nb in sweep_buckets(arity) {
                for &bs in &[2u32, 4] {
                    for &vb in &[8u32, 64, 256] {
                        dispatch(&mut csv, &cli, arity, nb, bs, vb);
                    }
                }
            }
        } else {
            let nb = resolve_num_buckets(&cli, arity);
            dispatch(&mut csv, &cli, arity, nb, cli.bucket_size, cli.value_bits);
        }
    }
    println!("\nResults written to results/ikpir_wire_sizes.csv");
}
