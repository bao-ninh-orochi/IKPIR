//! **Intent:** Measure the **end-to-end false-positive rate** of IKPIR:
//! how often `IkpirClient::decode` returns `Some(value)` for a key that
//! was never inserted.
//!
//! The cuckoo-layer FPR
//! (`segmented-cuckoo/benches/fpr.rs`) measures the filter's "fingerprint
//! match in a candidate bucket" rate. The IKPIR-level FPR is what the
//! reader of the paper actually cares about and differs from the cuckoo
//! number in two ways:
//!
//! - **`cells_per_slot` re-packing.** Values are stored alongside
//!   fingerprints in a packed cell stream. A fingerprint match that
//!   happens to align with garbage value cells still surfaces as
//!   `Some(...)`. So IKPIR FPR ≈ cuckoo-layer FPR (the cell unpack does
//!   not add or remove matches; it just decodes garbage value bytes when
//!   a non-match slot is scanned by the constant-time decode path).
//!
//! - **Slot-scan widening.** Decode now scans **all** `arity ×
//!   bucket_size` slots in constant time, so a stray fingerprint in any
//!   candidate slot is counted; previous early-return decode could miss
//!   later slots. Differs only at the rare two-slot-collision case.
//!
//! **Method:** For each config:
//!   1. Build a populated server (target load 80%).
//!   2. Build a client from the setup bundle.
//!   3. Query a disjoint key range, end-to-end (build_query → answer →
//!      decode), and count how many decodes return `Some`.
//!   4. Report `fpr = n_false_positives / n_queried`.
//!
//! **Invocation:** `cargo bench --bench end_to_end_fpr` runs a single
//! sensible default config. `--sweep` runs the matrix over
//! `fingerprint_bits ∈ {8,10,12,14,16}` × `bucket_size ∈ {2,4}` ×
//! `arity ∈ {2,3,4}`.
//!
//! **Output:** `results/ikpir_end_to_end_fpr.csv`
//! Columns: `arity, num_buckets, bucket_size, fingerprint_bits,
//! value_bits, n_inserted, n_queried, n_false_positives, fpr`

mod helpers;

use helpers::MakeStore;
use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirServer};
use segmented_cuckoo::{
    CuckooError,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;

const MAX_KICKS: u32 = 2_500;

#[derive(clap::Parser)]
#[command(about = "Measure end-to-end IKPIR false-positive rate on the FrodoPIR backend.")]
struct Cli {
    /// Run the full hardcoded matrix
    /// (arity ∈ {2,3,4} × bucket_size ∈ {2,4} × fingerprint_bits ∈ {8,10,12,14,16}).
    #[arg(long)]
    sweep: bool,

    /// Cuckoo arity (2, 3, or 4). With --sweep, ignored.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4), default_value_t = 2)]
    arity: u32,

    #[arg(long, default_value_t = 256)]  num_buckets: u32,
    #[arg(long, default_value_t = 4)]    bucket_size: u32,
    #[arg(long, default_value_t = 12)]   fingerprint_bits: u32,
    #[arg(long, default_value_t = 64)]   value_bits: u32,
    #[arg(long, default_value_t = 8)]    plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)] lwe_dim: u32,
    /// Number of absent keys to query. Larger gives a tighter estimate at
    /// the cost of bench wall-clock time (each query is one full PIR
    /// round trip).
    #[arg(long, default_value_t = 2_000)] n_queried: u32,
}

fn fpr_one<S>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
)
where S: MakeStore {
    let mut store = match S::make_store(
        num_buckets, bucket_size, fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    ) {
        Ok(s) => s,
        Err(e) => { eprintln!("  Skip arity={arity} nb={num_buckets} bs={bucket_size} fb={fingerprint_bits}: {e:?}"); return; }
    };
    store.set_max_kicks(MAX_KICKS);

    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    let target_load: u32 = ((num_buckets * bucket_size) as f64 * 0.80) as u32;
    let mut n_inserted = 0u32;
    for k in 0u32..target_load {
        for (i, b) in value.iter_mut().enumerate() {
            *b = (k.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8;
        }
        match store.insert(k.to_le_bytes(), &value) {
            Ok(())                      => n_inserted = k + 1,
            Err(CuckooError::TableFull) => break,
            Err(e)                      => panic!("insert: {e:?}"),
        }
    }
    if n_inserted == 0 { eprintln!("  Skip: empty store"); return; }

    let server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let mut client: IkpirClient<FrodoPirBackend> = IkpirClient::from_setup(server.setup());

    // Query keys disjoint from the inserted range. Encode using a
    // distinct byte layout (8-byte LE) to keep them out of the inserted
    // key set (which used 4-byte LE).
    let mut n_false_positives = 0u32;
    for i in 0..cli.n_queried {
        let key = (u32::MAX as u64 - i as u64).to_le_bytes(); // 8-byte LE
        let q = client.build_query(&key);
        let r = server.answer(&q).expect("answer");
        if client.decode(&key, &r).expect("decode").is_some() {
            n_false_positives += 1;
        }
    }
    let fpr = n_false_positives as f64 / cli.n_queried as f64;
    writeln!(
        csv,
        "{arity},{num_buckets},{bucket_size},{fingerprint_bits},{},{n_inserted},{},{n_false_positives},{fpr:.6}",
        cli.value_bits, cli.n_queried,
    )
    .unwrap();
    println!(
        "  arity={arity} nb={num_buckets:<5} bs={bucket_size} fb={fingerprint_bits:<2} \
         vb={:<3} | n_inserted={n_inserted:<5} n_queried={:<5} fp={n_false_positives:<5} fpr={fpr:.6}",
        cli.value_bits, cli.n_queried,
    );
}

fn dispatch(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
) {
    match arity {
        2 => fpr_one::<Segmented2aryScheme>(csv, cli, 2, num_buckets, bucket_size, fingerprint_bits),
        3 => fpr_one::<Segmented3aryScheme>(csv, cli, 3, num_buckets, bucket_size, fingerprint_bits),
        4 => fpr_one::<Segmented4aryScheme>(csv, cli, 4, num_buckets, bucket_size, fingerprint_bits),
        _ => unreachable!(),
    }
}

fn num_buckets_for(arity: u32) -> u32 {
    // Larger tables give a tighter estimate of the per-bucket false-match
    // probability. 256-ish is the sweet spot for bench wall-clock at 80%
    // load with `n_queried = 2000`.
    match arity { 2 | 4 => 256, 3 => 384, _ => unreachable!() }
}

fn main() {
    let cli: Cli = helpers::parse_cli();
    let mut csv = helpers::csv_writer(
        "ikpir_end_to_end_fpr.csv",
        "arity,num_buckets,bucket_size,fingerprint_bits,value_bits,n_inserted,n_queried,n_false_positives,fpr",
    );

    println!("=== ikpir end-to-end false-positive rate (FrodoPirBackend) ===");
    println!(
        "Config: plaintext_bits={}, lwe_dim={}, n_queried={}",
        cli.plaintext_bits, cli.lwe_dim, cli.n_queried,
    );

    if cli.sweep {
        for &arity in &[2u32, 3, 4] {
            let nb = num_buckets_for(arity);
            for &bs in &[2u32, 4] {
                for &fb in &[8u32, 10, 12, 14, 16] {
                    dispatch(&mut csv, &cli, arity, nb, bs, fb);
                }
            }
        }
    } else {
        dispatch(&mut csv, &cli, cli.arity, cli.num_buckets, cli.bucket_size, cli.fingerprint_bits);
    }
    println!("\nResults written to results/ikpir_end_to_end_fpr.csv");
}
