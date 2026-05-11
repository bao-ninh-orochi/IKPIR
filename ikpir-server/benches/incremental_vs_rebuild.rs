//! **Intent:** Compare the wall-clock and *wire* cost of `N` incremental
//! hint patches against a single `full_rebuild` for the same total
//! mutation footprint, split by mutation kind (`insert` / `update` /
//! `delete`).
//!
//! This is the headline benefit of the incremental scheme: a
//! `full_rebuild` pays an `O(arity × n_rows × lwe_dim × row_width)`
//! matvec regardless of how few cells changed; the per-mutation patch is
//! `O(touched_cells × lwe_dim)`. There is some break-even point `N*` past
//! which incremental becomes strictly more expensive than rebuilding
//! once.
//!
//! Each mutation kind has a different per-call cost profile:
//!   - `insert` may trigger a cuckoo kick chain that touches up to
//!     `max_kicks` slots; the worst-case delta footprint is the largest
//!     of the three.
//!   - `update` always touches exactly one slot's value cells.
//!   - `delete` always zeroes exactly one slot (fingerprint + value).
//!
//! **Method:** For each `(arity, num_buckets, n_mutations)` and each
//! `mutation_kind ∈ {insert, update, delete}`, time:
//!   - **incremental path**: build a populated server, run `N` mutations
//!     via `server.{insert,update,delete}`, summing the
//!     [`HintDeltaBundle::wire_byte_size`] of each returned delta.
//!   - **rebuild path**: build a fresh store, run the same `N` mutations
//!     directly on the store (no hint patches), wrap in `IkpirServer`, and
//!     time one `full_rebuild` call. Sum `BackendWireSize::hint_byte_size`
//!     across the rebuilt hints.
//!
//! **Invocation:** `cargo bench --bench incremental_vs_rebuild` runs at
//! one `(arity, num_buckets, n_mutations)` point for all three mutation
//! kinds. Pass `--sweep` for the full curve. Use `--arity <N>` to pick 2,
//! 3, or 4; with `--sweep` and no explicit `--arity`, all three arities
//! are swept.
//!
//! **Output:** `results/ikpir_server_incremental_vs_rebuild.csv`
//! Columns: `mutation_kind, arity, num_buckets, bucket_size, value_bits,
//! n_mutations, incremental_ms, rebuild_ms, ratio_inc_over_rebuild,
//! delta_bytes_total, rebuild_hint_bytes, delta_to_hint_ratio`

mod helpers;

use helpers::MakeStore;
use ikpir_server::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer,
};
use segmented_cuckoo::{
    CuckooError, CuckooKVStore,
    Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme,
};
use std::io::Write;
use std::time::Instant;

const MAX_KICKS: u32 = 2_500;

#[derive(Clone, Copy, Debug)]
enum MutationKind { Insert, Update, Delete }
impl MutationKind {
    fn as_csv(self) -> &'static str {
        match self { Self::Insert => "insert", Self::Update => "update", Self::Delete => "delete" }
    }
    fn all() -> &'static [Self] {
        &[Self::Insert, Self::Update, Self::Delete]
    }
}

#[derive(clap::Parser)]
#[command(about = "Compare incremental hint patching vs full rebuild for N mutations.")]
struct Cli {
    /// Run the full hardcoded curve
    /// (num_buckets per arity × n_mutations ∈ {1,4,16,64,256,1024}).
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
    #[arg(long, default_value_t = 64)]   n_mutations: u32,
}

/// Build a seeded store at ~50% load with deterministic `(key, value)`
/// pairs for keys `0..n_seed`. Returns `None` if the store geometry is
/// invalid or seeding hits `TableFull` (the latter is unusual at 50% load
/// but possible for tiny tables).
fn build_seeded_store<S>(cli: &Cli, num_buckets: u32, n_seed: u32) -> Option<CuckooKVStore<S>>
where S: MakeStore {
    let mut store = S::make_store(
        num_buckets, cli.bucket_size, cli.fingerprint_bits, cli.value_bits, cli.plaintext_bits,
    ).ok()?;
    store.set_max_kicks(MAX_KICKS);
    let vsize = (cli.value_bits as usize).div_ceil(8);
    let mut value = vec![0u8; vsize];
    for k in 0u32..n_seed {
        fill_value(&mut value, k, 17);
        match store.insert(k.to_le_bytes(), &value) {
            Ok(())                       => {}
            Err(CuckooError::TableFull)  => return None,
            Err(e)                       => panic!("seed: {e:?}"),
        }
    }
    Some(store)
}

/// Deterministic value filler; `salt` lets two different mutation kinds
/// generate distinct payloads for the same key (e.g., seed vs update).
fn fill_value(value: &mut [u8], key: u32, salt: u32) {
    for (i, b) in value.iter_mut().enumerate() {
        *b = (key.wrapping_mul(salt).wrapping_add(i as u32) & 0xFF) as u8;
    }
}

#[derive(Clone, Copy)]
struct PerKindResult {
    incremental_ms:     f64,
    rebuild_ms:         f64,
    delta_bytes_total:  usize,
    rebuild_hint_bytes: usize,
}

/// Run the incremental + rebuild paths for a single mutation kind.
fn run_kind<S>(
    cli: &Cli,
    num_buckets: u32,
    n_seed: u32,
    n_mut: u32,
    kind: MutationKind,
) -> Result<PerKindResult, IkpirError>
where S: MakeStore {
    let vsize = (cli.value_bits as usize).div_ceil(8);

    // ── Incremental path ─────────────────────────────────────────────────
    let store = build_seeded_store::<S>(cli, num_buckets, n_seed).ok_or(IkpirError::TableFull)?;
    let mut server: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));

    let mut delta_bytes_total = 0usize;
    let mut value = vec![0u8; vsize];
    let inc_start = Instant::now();
    for i in 0..n_mut {
        let bundle = match kind {
            MutationKind::Insert => {
                let k = n_seed + i;
                fill_value(&mut value, k, 31);
                server.insert(&k.to_le_bytes(), &value)?
            }
            MutationKind::Update => {
                let k = (n_seed - 1) - i;
                fill_value(&mut value, k, 47); // different salt from seed
                server.update(&k.to_le_bytes(), &value)?
            }
            MutationKind::Delete => {
                let k = (n_seed - 1) - i;
                server.delete(&k.to_le_bytes())?
            }
        };
        delta_bytes_total += bundle.wire_byte_size();
    }
    let incremental_ms = inc_start.elapsed().as_secs_f64() * 1e3;

    // ── Rebuild path ─────────────────────────────────────────────────────
    // Recreate the store, apply the same mutations directly (no hint
    // patches), wrap in a fresh server, then time one full_rebuild call.
    let mut store = build_seeded_store::<S>(cli, num_buckets, n_seed).ok_or(IkpirError::TableFull)?;
    for i in 0..n_mut {
        match kind {
            MutationKind::Insert => {
                let k = n_seed + i;
                fill_value(&mut value, k, 31);
                if store.insert(k.to_le_bytes(), &value).is_err() { break; }
            }
            MutationKind::Update => {
                let k = (n_seed - 1) - i;
                fill_value(&mut value, k, 47);
                if store.update(k.to_le_bytes(), &value).is_err() { break; }
            }
            MutationKind::Delete => {
                let k = (n_seed - 1) - i;
                if store.delete(k.to_le_bytes()).is_err() { break; }
            }
        }
    }
    let mut server2: IkpirServer<S, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::with_lwe_dim(cli.lwe_dim));
    let reb_start = Instant::now();
    let rebuild_bundle = server2.full_rebuild();
    let rebuild_ms = reb_start.elapsed().as_secs_f64() * 1e3;
    let rebuild_hint_bytes: usize = rebuild_bundle.hints.iter()
        .map(FrodoPirBackend::hint_byte_size).sum();

    Ok(PerKindResult { incremental_ms, rebuild_ms, delta_bytes_total, rebuild_hint_bytes })
}

fn run_one<S>(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    n_mut:       u32,
)
where S: MakeStore {
    let cap    = num_buckets * cli.bucket_size;
    let n_seed = cap / 2;
    // Insert path needs headroom for n_seed + n_mut. Update/delete need
    // n_mut ≤ n_seed (so the keys exist). Use the same conservative bound
    // as the original bench for the insert headroom; update/delete are
    // trivially satisfied because n_mut ≤ n_seed = cap/2.
    if n_seed + n_mut > (cap as f64 * 0.85) as u32 {
        eprintln!("  Skip arity={arity} num_buckets={num_buckets} N={n_mut}: insufficient headroom");
        return;
    }
    if n_mut > n_seed {
        eprintln!("  Skip arity={arity} num_buckets={num_buckets} N={n_mut}: n_mut > n_seed");
        return;
    }

    for &kind in MutationKind::all() {
        let r = match run_kind::<S>(cli, num_buckets, n_seed, n_mut, kind) {
            Ok(r)  => r,
            Err(e) => {
                eprintln!("  Skip kind={} arity={arity} num_buckets={num_buckets} N={n_mut}: {e:?}",
                          kind.as_csv());
                continue;
            }
        };
        let ratio_time = r.incremental_ms / r.rebuild_ms;
        let ratio_bytes = if r.rebuild_hint_bytes == 0 {
            f64::NAN
        } else {
            r.delta_bytes_total as f64 / r.rebuild_hint_bytes as f64
        };
        writeln!(
            csv,
            "{},{arity},{num_buckets},{},{},{n_mut},{:.3},{:.3},{:.4},{},{},{:.6}",
            kind.as_csv(),
            cli.bucket_size, cli.value_bits,
            r.incremental_ms, r.rebuild_ms, ratio_time,
            r.delta_bytes_total, r.rebuild_hint_bytes, ratio_bytes,
        )
        .unwrap();
        println!(
            "  kind={:<6} arity={arity} num_buckets={num_buckets:<5} N={n_mut:<5} | \
             inc={:>8.3} ms  rebuild={:>8.3} ms  ratio={:.3}  delta={:>9} B  hint={:>10} B  ratio_b={:.4}",
            kind.as_csv(),
            r.incremental_ms, r.rebuild_ms, ratio_time,
            r.delta_bytes_total, r.rebuild_hint_bytes, ratio_bytes,
        );
    }
}

fn dispatch(
    csv: &mut std::io::BufWriter<std::fs::File>,
    cli: &Cli,
    arity: u32,
    num_buckets: u32,
    n_mut: u32,
) {
    match arity {
        2 => run_one::<Segmented2aryScheme>(csv, cli, 2, num_buckets, n_mut),
        3 => run_one::<Segmented3aryScheme>(csv, cli, 3, num_buckets, n_mut),
        4 => run_one::<Segmented4aryScheme>(csv, cli, 4, num_buckets, n_mut),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
}

fn sweep_buckets(arity: u32) -> &'static [u32] {
    match arity {
        2 | 4 => &[256, 1024],
        3     => &[384, 1536],
        _     => unreachable!(),
    }
}

fn resolve_num_buckets(cli: &Cli, arity: u32) -> u32 {
    if cli.num_buckets == 256 && arity == 3 { 384 } else { cli.num_buckets }
}

fn main() {
    let cli: Cli = helpers::parse_cli();
    let mut csv = helpers::csv_writer(
        "ikpir_server_incremental_vs_rebuild.csv",
        "mutation_kind,arity,num_buckets,bucket_size,value_bits,n_mutations,incremental_ms,rebuild_ms,ratio_inc_over_rebuild,delta_bytes_total,rebuild_hint_bytes,delta_to_hint_ratio",
    );

    println!("=== ikpir-server: incremental hint patch vs full_rebuild (FrodoPirBackend) ===");
    println!(
        "Config: fingerprint_bits={}, plaintext_bits={}, value_bits={}, bucket_size={}, lwe_dim={}",
        cli.fingerprint_bits, cli.plaintext_bits, cli.value_bits, cli.bucket_size, cli.lwe_dim,
    );

    let arities: Vec<u32> = match (cli.sweep, cli.arity) {
        (true,  None)    => vec![2, 3, 4],
        (true,  Some(a)) => vec![a],
        (false, opt)     => vec![opt.unwrap_or(2)],
    };

    for &arity in &arities {
        if cli.sweep {
            for &nb in sweep_buckets(arity) {
                for &n in &[1u32, 4, 16, 64, 256, 1024] {
                    dispatch(&mut csv, &cli, arity, nb, n);
                }
            }
        } else {
            let nb = resolve_num_buckets(&cli, arity);
            dispatch(&mut csv, &cli, arity, nb, cli.n_mutations);
        }
    }
    println!("\nResults written to results/ikpir_server_incremental_vs_rebuild.csv");
}
