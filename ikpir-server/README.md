# ikpir-server

Server-side crate for [Incremental Keyword PIR](../README.md). Wraps a
`Segmented{2,3,4}aryCuckooKVStore` in per-segment Index-PIR sub-databases
and exposes the full server protocol: `setup`, `answer`, `insert`,
`update`, `delete`, `full_rebuild`.

## Quick start

```rust
use ikpir_server::{FrodoConfig, FrodoPirBackend, Segmented2aryIkpirServer};
use segmented_cuckoo::Segmented2aryCuckooKVStore;

// Build and populate the SCF KV store.
let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
store.insert(b"alice", &[0x01u8]).unwrap();
store.insert(b"bob",   &[0x02u8]).unwrap();

// Wrap in a server; backend tunables (LWE dim) come from FrodoConfig.
// FrodoConfig::default() picks the FrodoPIR Table-5 dim (1774).
let mut server: Segmented2aryIkpirServer<FrodoPirBackend> =
    Segmented2aryIkpirServer::new(store, FrodoConfig::default());
let _bundle = server.setup();

// Override the LWE dimension:
// let server = Segmented2aryIkpirServer::<FrodoPirBackend>::new(
//     store, FrodoConfig::with_lwe_dim(1024),
// );
```

## Benches

Three `clap`-parsed CSV-emitting benches mirroring `segmented-cuckoo`'s
style. Output lands in `results/`. Each bench has three modes:

- **No args** → one sensible default config (1 CSV row, runs in seconds).
- **`--sweep`** → the full hardcoded parameter matrix.
- **Explicit flags** → a single user-specified config.

Pass arguments after `--`, e.g. `cargo bench --bench foo -- --sweep`.

### Bench overview

`num_buckets` constraints differ per arity: 2-ary `2^t`, 3-ary `3·2^t`,
4-ary `2^t ≥ 4`. Sweep matrices below list the values used for arities
2 and 4; the 3-ary sweep substitutes scale-comparable `3·2^t` values.

| Bench | What it measures | Sweep matrix (with `--sweep`) |
|---|---|---|
| `setup_throughput` | `IkpirServer::new` wall-clock cost (`B::server_setup` per segment) | arity ∈ {2,3,4} × num_buckets ∈ {64, 256, 1024, 4096} (3-ary: {96, 384, 1536, 6144}) × bucket_size ∈ {2, 4} × value_bits ∈ {8, 64, 256} |
| `answer_throughput` | PIR matvec answer rate (queries/sec) | arity ∈ {2,3,4} × num_buckets ∈ {64, 256, 1024} (3-ary: {96, 384, 1536}) × bucket_size ∈ {2, 4} × value_bits ∈ {8, 64, 256} |
| `incremental_vs_rebuild` | Incremental hint patch vs `full_rebuild` for N mutations (the headline plot) | arity ∈ {2,3,4} × num_buckets ∈ {256, 1024} (3-ary: {384, 1536}) × n_mutations ∈ {1, 4, 16, 64, 256, 1024} |

### Common flags

| Flag | Default | Meaning |
|---|---|---|
| `--sweep` | off | Run the full hardcoded matrix instead of a single config |
| `--arity <N>` | unset | Cuckoo arity (2, 3, or 4). With `--sweep` and unset, sweep all three. Without `--sweep`, defaults to 2 |
| `--num-buckets <N>` | `256` (auto-becomes `384` when `--arity 3` is selected without `--num-buckets`) | Buckets per segment. Per-arity constraint: 2-ary `2^t`, 3-ary `3·2^t`, 4-ary `2^t ≥ 4` |
| `--bucket-size <N>` | `4` | Slots per bucket (1–4) |
| `--value-bits <N>` | `64` | Value width per `(key, value)` entry |
| `--fingerprint-bits <N>` | `12` | Fingerprint width |
| `--plaintext-bits <N>` | `8` | PIR plaintext-cell bit-width (1–31) |
| `--lwe-dim <N>` | `1774` | FrodoPIR LWE dimension `n` |
| `--trials <N>` | `5` | Measured trials per config |

### Bench-specific flags

| Bench | Extra flag(s) | Default |
|---|---|---|
| `setup_throughput` | `--warmup <N>` | `2` |
| `answer_throughput` | `--warmup <N>`, `--batch <N>` | `2`, `32` |
| `incremental_vs_rebuild` | `--n-mutations <N>` | `64` |

### Examples

```bash
# 1) Default single config (one CSV row, ~seconds)
cargo bench -p ikpir-server --bench answer_throughput

# 2) Full sweep — produces the CSV that `scripts/plot.py` consumes
cargo bench -p ikpir-server --bench answer_throughput       -- --sweep
cargo bench -p ikpir-server --bench setup_throughput        -- --sweep
cargo bench -p ikpir-server --bench incremental_vs_rebuild  -- --sweep

# 3) Pin one specific config (one CSV row)
cargo bench -p ikpir-server --bench answer_throughput -- \
    --num-buckets 1024 --bucket-size 4 --value-bits 64 \
    --lwe-dim 1024 --batch 64 --trials 10

# 4) Headline incremental-vs-rebuild crossover at N = 256 mutations
cargo bench -p ikpir-server --bench incremental_vs_rebuild -- \
    --num-buckets 1024 --n-mutations 256

# 5) Compare arities at a fixed config
cargo bench -p ikpir-server --bench answer_throughput -- \
    --arity 3                                    # arity 3 single config (auto-selects 384 buckets)
cargo bench -p ikpir-server --bench answer_throughput -- \
    --arity 4 --num-buckets 1024                 # explicit arity 4

# 6) Full flag list
cargo bench -p ikpir-server --bench answer_throughput -- --help
```

### Plotting

Render plots from the CSVs with `scripts/plot.py` (matplotlib + pandas):

```bash
pip install -r scripts/requirements.txt
python scripts/plot.py            # all plots → results/plots/
python scripts/plot.py --list     # list individual plot functions
python scripts/plot.py incremental_vs_rebuild   # one specific plot
```

Override paths via `IKPIR_SERVER_RESULTS_DIR` / `IKPIR_SERVER_PLOTS_DIR`.

## Per-segment architecture

An arity-`k` SCF is partitioned into `k` independent Index-PIR
sub-databases (one per segment). A query for key `k` targets row
`indices[j] % segment_size` in segment `j` — `k` PIR queries suffice for
any arity.

```
key  ──candidate_buckets──▶  [b0, b1]
                               │      │
                           seg 0   seg 1
client.build_query  ──▶   Q[0]    Q[1]
server.answer       ──▶   R[0]    R[1]
client.decode  → fp match → value
```

## Implementing a new backend

Implement `IndexPirBackend` (mandatory) and optionally
`IncrementalPirBackend` (for incremental hint updates without a full
rebuild).

Minimal correctness contract:

```
client_decode(server_answer(client_query(state, row)))
    == db[row * row_width .. (row+1) * row_width]
```

See [`CLAUDE.md §6`](CLAUDE.md) for the full backend-author checklist.

## Status

`FrodoPirBackend` is the shipped backend. SimplePIR is a future track.
Bundle types are not versioned; serialisation is out of scope.
