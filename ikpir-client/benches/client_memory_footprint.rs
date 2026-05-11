//! **Intent:** Report the heap footprint of an
//! `IkpirClient<FrodoPirBackend>` in each preprocessing state
//! (cold / warm-b / warm-bc), as a function of `(arity, num_buckets,
//! bucket_size, value_bits, lwe_dim, batch)`.
//!
//! **Method:** Closed-form accounting from public state introspection.
//! `FrodoClientState::PreparedSlot` is private to the FrodoPIR backend, so
//! the per-slot size is derived analytically from the public geometry:
//!
//! ```text
//!   per-segment baseline   = 2 × (n_rows × lwe_dim × 4)   // params.a + state.params.a (cloned)
//!                            + (lwe_dim × row_width × 4)  // hint.data
//!   per-slot (cold/warm-b) = (lwe_dim + n_rows) × 4       // secret + b
//!   per-slot (warm-bc)     = (lwe_dim + n_rows + row_width) × 4   // + precomputed c
//!   prepared count         = 0 (cold) | batch (warm-b/warm-bc)
//!   heap_bytes             = arity × (baseline + prepared count × per-slot)
//!   stack_bytes            = mem::size_of::<IkpirClient<FrodoPirBackend>>()
//! ```
//!
//! Geometry:
//!   - `n_rows`    = num_buckets / arity (segment_size)
//!   - `row_width` = bucket_size × cells_per_slot
//!   - `cells_per_slot` = (fingerprint_bits + value_bits).div_ceil(plaintext_bits)
//!
//! **Invocation:** `cargo bench --bench client_memory_footprint` runs a
//! single default config. Pass `--sweep` to iterate the full matrix
//! (including all three modes — see sweep semantics below).
//!
//! **Sweep semantics:** `--sweep` (or `--sweep --mode cold`) iterates all
//! three modes; `--sweep --mode warm-b` / `--sweep --mode warm-bc` pins to
//! a single mode while sweeping configs.
//!
//! **Output:** `results/ikpir_client_memory_footprint.csv`
//! Columns: mode, arity, num_buckets, bucket_size, value_bits, lwe_dim,
//! batch, stack_bytes, heap_bytes, total_bytes

mod helpers;

use ikpir_client::{FrodoPirBackend, IkpirClient};
use std::io::Write;
use std::mem;

type Client = IkpirClient<FrodoPirBackend>;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Mode { Cold, WarmB, WarmBc }
impl Mode {
    fn as_csv(self) -> &'static str {
        match self { Mode::Cold => "cold", Mode::WarmB => "warm-b", Mode::WarmBc => "warm-bc" }
    }
}

#[derive(Clone, clap::Parser)]
#[command(about = "Closed-form heap accounting for IkpirClient<FrodoPirBackend>.")]
struct Cli {
    /// Run the full hardcoded matrix.
    #[arg(long)]
    sweep: bool,

    /// Cuckoo arity (2, 3, or 4). With --sweep and no --arity, sweep all three.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))]
    arity: Option<u32>,

    /// Preprocessing mode: cold (default), warm-b, or warm-bc.
    #[arg(long, value_enum, default_value_t = Mode::Cold)]
    mode: Mode,

    #[arg(long, default_value_t = 256)]  num_buckets: u32,
    #[arg(long, default_value_t = 4)]    bucket_size: u32,
    #[arg(long, default_value_t = 64)]   value_bits: u32,
    #[arg(long, default_value_t = 12)]   fingerprint_bits: u32,
    #[arg(long, default_value_t = 8)]    plaintext_bits: u32,
    #[arg(long, default_value_t = 1774)] lwe_dim: u32,
    #[arg(long, default_value_t = 64)]   batch: u32,
}

fn footprint(
    cli:         &Cli,
    arity:       u32,
    num_buckets: u32,
    bucket_size: u32,
    value_bits:  u32,
    mode:        Mode,
) -> (usize, usize) {
    let n_rows         = (num_buckets / arity) as usize;
    let cells_per_slot = (cli.fingerprint_bits + value_bits).div_ceil(cli.plaintext_bits) as usize;
    let row_width      = bucket_size as usize * cells_per_slot;
    let lwe_dim        = cli.lwe_dim as usize;

    let per_seg_baseline   = 2 * (n_rows * lwe_dim * 4) + (lwe_dim * row_width * 4);
    let per_slot_cold      = (lwe_dim + n_rows) * 4;
    let per_slot_warm_c    = (lwe_dim + n_rows + row_width) * 4;
    let prepared_count     = match mode { Mode::Cold => 0, Mode::WarmB | Mode::WarmBc => cli.batch as usize };
    let per_seg_prepared   = match mode {
        Mode::Cold | Mode::WarmB => prepared_count * per_slot_cold,
        Mode::WarmBc             => prepared_count * per_slot_warm_c,
    };
    let heap_bytes  = (arity as usize) * (per_seg_baseline + per_seg_prepared);
    let stack_bytes = mem::size_of::<Client>();
    (stack_bytes, heap_bytes)
}

fn run_one(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    bucket_size: u32,
    value_bits:  u32,
) {
    let (stack_bytes, heap_bytes) = footprint(cli, arity, num_buckets, bucket_size, value_bits, cli.mode);
    let total_bytes = stack_bytes + heap_bytes;
    let mode_label  = cli.mode.as_csv();
    writeln!(
        csv,
        "{mode_label},{arity},{num_buckets},{bucket_size},{value_bits},{},{},{stack_bytes},{heap_bytes},{total_bytes}",
        cli.lwe_dim, cli.batch,
    )
    .unwrap();
    println!(
        "  mode={mode_label:<7} arity={arity} num_buckets={num_buckets:<5} bs={bucket_size} vb={value_bits:<4} | \
         stack={stack_bytes}B heap={heap_bytes}B total={total_bytes}B"
    );
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
        "ikpir_client_memory_footprint.csv",
        "mode,arity,num_buckets,bucket_size,value_bits,lwe_dim,batch,stack_bytes,heap_bytes,total_bytes",
    );

    println!("=== ikpir-client memory footprint (FrodoPirBackend, closed-form) ===");
    println!(
        "Config: mode={}, fingerprint_bits={}, plaintext_bits={}, lwe_dim={}, batch={}",
        cli.mode.as_csv(), cli.fingerprint_bits, cli.plaintext_bits, cli.lwe_dim, cli.batch,
    );

    let arities: Vec<u32> = match (cli.sweep, cli.arity) {
        (true,  None)    => vec![2, 3, 4],
        (true,  Some(a)) => vec![a],
        (false, opt)     => vec![opt.unwrap_or(2)],
    };

    let modes: Vec<Mode> = match (cli.sweep, cli.mode) {
        (true,  Mode::Cold)   => vec![Mode::Cold, Mode::WarmB, Mode::WarmBc],
        (true,  Mode::WarmB)  => vec![Mode::WarmB],
        (true,  Mode::WarmBc) => vec![Mode::WarmBc],
        (false, m)            => vec![m],
    };

    for &mode in &modes {
        let cli_local = Cli { mode, ..cli.clone() };
        for &arity in &arities {
            if cli_local.sweep {
                for &nb in sweep_buckets(arity) {
                    for &bs in &[2u32, 4] {
                        for &vb in &[8u32, 64, 256] {
                            run_one(&mut csv, &cli_local, arity, nb, bs, vb);
                        }
                    }
                }
            } else {
                let nb = resolve_num_buckets(&cli_local, arity);
                run_one(&mut csv, &cli_local, arity, nb, cli_local.bucket_size, cli_local.value_bits);
            }
        }
    }
    println!("\nResults written to results/ikpir_client_memory_footprint.csv");
}
