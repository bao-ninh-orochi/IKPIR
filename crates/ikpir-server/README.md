# ikpir-server

Server-side crate for [Incremental Keyword PIR](../../README.md). Wraps a
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

Four focused `clap`-parsed CSV-emitting benches under `benches/`. The
recommended way to run one is the workspace runner
[`../../scripts/bench.sh`](../../scripts/bench.sh), which auto-derives the largest
correct `--plaintext-bits` and the backend `--lwe-dim`, and routes output to
`results/ikpir-server/`:

```bash
./scripts/bench.sh server_answer --arity 4 --num-buckets 65536 --value-bits 256
./scripts/bench.sh server_setup --backend simple
./scripts/bench.sh                              # -h: full flag + bench list
```

Each invocation is one config = one appended CSV row (`server_mutation` emits
one row per `(patch mode, kind)` pair). The root [README](../../README.md#benches)
has the paper config matrix; there is no full-matrix sweep script.

### Bench overview

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `server_setup` | `TableFull` | `IkpirServer::new` wall-clock (trials=1, warmup=0), or `--estimate` = one segment × arity; setup_bundle_bytes, hint_bytes/seg | `ikpir_server_setup.csv` |
| `server_answer` | `TableFull` | PIR answer rate (queries/sec, criterion, batch=64); query_bytes, response_bytes | `ikpir_server_answer.csv` |
| `server_mutation` | `--load-factor` (0.90) | Per-(patch mode, kind) ops/sec, wall-clock batch; delta_bytes_total | `ikpir_server_mutation.csv` |
| `headtohead_answer` | fixed `--num-keys` | answer rate at a fixed keyword count (fair comparison vs ChalametPIR / Hao 2025); +`num_keys`/`db_size` columns | `ikpir_headtohead_server_answer.csv` |

`num_buckets` constraints differ per arity: 2-ary `2^t`, 3-ary `3·2^t`,
4-ary `2^t ≥ 4`.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--arity <N>` | `2` | Cuckoo arity (2, 3, 4) |
| `--backend frodo\|simple` | `frodo` | Index-PIR backend |
| `--num-buckets <N>` | per-arity | Buckets per segment |
| `--bucket-size <N>` | `4` | Slots per bucket |
| `--value-bits <N>` | `256` | Value width per entry |
| `--plaintext-bits <N>` | `8` bench / max via `bench.sh` | PIR cell width |
| `--lwe-dim <N>` | 1566 (frodo) / 1275 (simple) | LWE dimension |

Bench-specific: `server_setup` takes `--estimate` / `--trials` / `--warmup`;
`server_answer` and `headtohead_answer` take `--batch`; `server_mutation` takes
`--patch-mode entry\|row` (comma list, default `entry`), `--n-mutations`,
`--load-factor`; `headtohead_answer` requires `--num-keys` and takes
`--max-mem-gb`.

### Low-level: `cargo bench`

`bench.sh` is a thin wrapper; the benches also run standalone — then
`--plaintext-bits` defaults to `8` and output lands in the crate-local
`results/` unless `IKPIR_RESULTS_DIR` is set:

```bash
cargo bench -p ikpir-server --bench server_answer -- --backend simple --plaintext-bits 10
cargo bench -p ikpir-server --bench server_mutation -- --patch-mode entry,row --n-mutations 64
cargo bench -p ikpir-server --bench <name> -- --help
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

