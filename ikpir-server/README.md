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
// FrodoConfig::default() picks the lattice-estimator-recommended dim (1566).
let mut server: Segmented2aryIkpirServer<FrodoPirBackend> =
    Segmented2aryIkpirServer::new(store, FrodoConfig::default());
let _bundle = server.setup();

// Override the LWE dimension:
// let server = Segmented2aryIkpirServer::<FrodoPirBackend>::new(
//     store, FrodoConfig::with_lwe_dim(1024),
// );
```

## Benches

Three focused `clap`-parsed CSV-emitting benches under `benches/`. Output
lands in `results/`. Each invocation produces one CSV row (append-mode);
a sweep is the orchestrator's job — `rm` the CSV first, then loop.
Pass arguments after `--`, e.g. `cargo bench --bench server_setup -- --help`.

### Bench overview

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `server_setup` | `TableFull` | `IkpirServer::new` wall-clock (trials=5, warmup=2); setup_bundle_bytes, hint_bytes/seg | `ikpir_server_setup.csv` |
| `server_answer` | `TableFull` | PIR answer rate (queries/sec, criterion, batch=64); query_bytes, response_bytes | `ikpir_server_answer.csv` |
| `server_mutation` | `--load-factor` | Per-kind (insert/update/delete) ops/sec, wall-clock batch; delta_bytes_total | `ikpir_server_mutation.csv` |

`num_buckets` constraints differ per arity: 2-ary `2^t`, 3-ary `3·2^t`,
4-ary `2^t ≥ 4`.

### Common flags

| Flag | Default | Meaning |
|---|---|---|
| `--arity <N>` | `2` | Cuckoo arity (2, 3, 4) |
| `--backend frodo\|simple` | `frodo` | Index-PIR backend |
| `--num-buckets <N>` | per-arity | Buckets per segment |
| `--bucket-size <N>` | `4` | Slots per bucket |
| `--value-bits <N>` | `256` | Value width per entry |
| `--lwe-dim <N>` | 1566 (frodo) / 1275 (simple) | LWE dimension |

### Bench-specific flags

| Bench | Extra flags |
|---|---|
| `server_setup` | `--trials 5`, `--warmup 2` |
| `server_answer` | `--batch 64` |
| `server_mutation` | `--n-mutations 1024`, `--load-factor 0.80` |

### Examples

```bash
# One config, one CSV row.
cargo bench -p ikpir-server --bench server_setup -- \
    --arity 2 --num-buckets 65536 --bucket-size 4 --value-bits 256

# Answer throughput with SimplePIR backend.
cargo bench -p ikpir-server --bench server_answer -- \
    --backend simple --num-buckets 262144 --value-bits 2048 --batch 64

# Mutation throughput sweep: 64 mutations at 80 % load.
cargo bench -p ikpir-server --bench server_mutation -- \
    --arity 3 --num-buckets 393216 --bucket-size 2 --value-bits 256 \
    --n-mutations 64 --load-factor 0.80

# Flag list for any bench.
cargo bench -p ikpir-server --bench <name> -- --help
```

### Orchestrator sweep

`ikpir-server/scripts/run_benches.sh` sweeps the full paper config matrix
(12 configs × 3 value\_bits = 36 runs per bench; the mutation bench
reuses the same 12 configs × 3 value\_bits = 36 runs, with N\_mutations
derived per config as capacity / 100). The orchestrator removes the CSV
before each sweep and re-runs per backend set in `IKPIR_BENCH_BACKENDS`.

```bash
# Server benches only, FrodoPIR.
./ikpir-server/scripts/run_benches.sh

# One bench.
./ikpir-server/scripts/run_benches.sh server_answer

# Both backends.
IKPIR_BENCH_BACKENDS=frodo,simple ./ikpir-server/scripts/run_benches.sh
```

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

The backend trait family and the two shipped backends (`FrodoPirBackend`
and `SimplePirBackend`) live in [`ikpir-common`](../ikpir-common). This
crate re-exports them, so implementations land in the workspace either
inside `ikpir-common` itself or in a downstream crate that depends on
`ikpir-common`.

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

Two backends ship: `FrodoPirBackend` (ternary errors, tall-skinny matrix,
default `lwe_dim = 1566`) and `SimplePirBackend` (discrete-Gaussian
errors with σ = 6.4, `√N × √N` internal reshape, default `lwe_dim = 1275`).
Both defaults target 128-bit security, estimated via the lattice
estimator under the ADPS16 cost model.
Both implement all four traits (`IndexPirBackend` + the three optional
extensions) and are drop-in alternatives at the `B: IndexPirBackend`
type parameter on `IkpirServer`. Bundle types are not versioned;
serialisation is out of scope.

