# ikpir-client

Client-side crate for [Incremental Keyword PIR](../../README.md). Holds
per-segment `ClientState` and translates `(key, value)` lookups into
wire-level Index-PIR query/response exchanges with `ikpir-server`. Ships
**two parallel, first-class client flows** over the same server-published
delta stream: **client-rewind** (`RewindClient`) pins its bootstrap hint and
accumulates the server's published deltas instead of patching the hint
directly; **client-hint-patch** (`HintPatchClient`) folds every delta into
its own hint immediately and decodes directly against it. Both are always
available, chosen at the type like the backend at `B`
(see [`../../docs/rewind-client-mode.md`](../../docs/rewind-client-mode.md)).

## Quick start

Client-rewind (`RewindClient`):

```rust
use ikpir_client::RewindClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, Segmented2aryIkpirServer};
use segmented_cuckoo::Segmented2aryCuckooKVStore;

// In a real deployment the bundle arrives over the network.
let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
store.insert(b"alice", &[0x01u8]).unwrap();

let mut server: Segmented2aryIkpirServer<FrodoPirBackend> =
    Segmented2aryIkpirServer::new(store, FrodoConfig::default());
let bundle = server.setup();

// Initialise client from the setup bundle.
let mut client: RewindClient<FrodoPirBackend> = RewindClient::from_setup(bundle);

// Round-trip a key. `decode` threads the query back so the response can be
// rewound to the client's pinned hint before decoding.
let q = client.build_query(b"alice");
let r = server.answer(&q).unwrap();
let v = client.decode(b"alice", &q, &r).unwrap().expect("key present");
assert_eq!(v, &[0x01u8]);
```

Client-hint-patch (`HintPatchClient`):

```rust
use ikpir_client::HintPatchClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, Segmented2aryIkpirServer};
use segmented_cuckoo::Segmented2aryCuckooKVStore;

let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
store.insert(b"alice", &[0x01u8]).unwrap();

let mut server: Segmented2aryIkpirServer<FrodoPirBackend> =
    Segmented2aryIkpirServer::new(store, FrodoConfig::default());
let mut client: HintPatchClient<FrodoPirBackend> =
    HintPatchClient::from_setup(server.setup());

// The hint is patched immediately on every delta (`apply_delta`), so
// `decode` needs only the response — no query threading.
let q = client.build_query(b"alice");
let r = server.answer(&q).unwrap();
let v = client.decode(b"alice", &r).unwrap().expect("key present");
assert_eq!(v, &[0x01u8]);
```

## Benches

Nine focused `clap`-parsed CSV-emitting benches under `benches/`. The client
flow is always a separate binary, never a runtime flag, and **benchmark data
of the two flows is always written to separate CSV files and never merged**;
`client_query` and `headtohead_query` are the only client-side benches
without a per-flow pair, because `build_query` is identical code on both
flows. The recommended way to run one is the workspace runner
[`../../scripts/bench.sh`](../../scripts/bench.sh), which auto-derives the largest
correct `--plaintext-bits` and the backend `--lwe-dim`, and routes output to
`results/ikpir-client/`:

```bash
./scripts/bench.sh client_hint_patch_decode --arity 4 --num-buckets 65536 --value-bits 256
./scripts/bench.sh client_rewind_decode --arity 4 --num-buckets 65536 --value-bits 256
./scripts/bench.sh client_hint_patch_mutation --patch-mode entry,row
./scripts/bench.sh client_rewind_mutation
./scripts/bench.sh                              # -h: full flag + bench list
```

Each invocation is one config = one appended CSV row (the mutation benches
emit one row per `(patch mode, kind)` for client-hint-patch, one per `kind`
for client-rewind). `client_query` / `client_{hint_patch,rewind}_decode` run
in **warm-bc** mode (precompute before the timed loop); `client_rewind_mutation`
runs in **empty-queue** mode so it times the client-rewind flow's
`accumulate_delta` (`ΔD` roll-forward) in isolation, and
`client_hint_patch_mutation` times the client-hint-patch flow's
`HintPatchClient::apply_delta` the same way. The root
[README](../../README.md#benches) has the paper config matrix.

### Bench overview

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `client_query` | `TableFull` | `build_query` rate (queries/sec, criterion, warm-bc); flow-independent | `ikpir_client_query.csv` |
| `client_hint_patch_decode` | `TableFull` | client-hint-patch's `decode` rate (queries/sec, criterion, warm-bc) | `ikpir_client_hint_patch_decode.csv` |
| `client_rewind_decode` | `TableFull` | client-rewind's `decode` rate at empty `ΔD` (queries/sec, criterion, warm-bc) | `ikpir_client_rewind_decode.csv` |
| `client_hint_patch_mutation` | `--load-factor` (0.90) | per-batch maintenance throughput per (kind, `--patch-mode` entry\|row); times `HintPatchClient::apply_delta` | `ikpir_client_hint_patch_mutation.csv` |
| `client_rewind_mutation` | `--load-factor` (0.90) | per-batch maintenance throughput per kind (no `--patch-mode`); times `RewindClient::accumulate_delta` | `ikpir_client_rewind_mutation.csv` |
| `client_rewind_staleness` | `--load-factor` (0.90) | `decode` per-query latency vs staleness \|ΔD\|, then post-`collect_garbage` | `ikpir_client_rewind_staleness.csv` |
| `headtohead_query` | fixed `--num-keys` | `build_query` rate at a fixed keyword count; +`num_keys`/`db_size` cols; flow-independent | `ikpir_headtohead_client_query.csv` |
| `headtohead_hint_patch_decode` | fixed `--num-keys` | client-hint-patch's `decode` rate at a fixed keyword count; +`num_keys`/`db_size` cols | `ikpir_headtohead_client_hint_patch_decode.csv` |
| `headtohead_rewind_decode` | fixed `--num-keys` | client-rewind's `decode` rate at a fixed keyword count; +`num_keys`/`db_size` cols | `ikpir_headtohead_client_rewind_decode.csv` |

`num_buckets` constraints differ per arity: 2-ary `2^t`, 3-ary `3·2^t`, 4-ary `2^t ≥ 4`.

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

Bench-specific: `client_query` / `client_{hint_patch,rewind}_decode` /
`headtohead_*` take `--batch` (key-pool size); `client_hint_patch_mutation`
takes `--patch-mode entry\|row` (comma list, default `entry`) — no such flag
on `client_rewind_mutation`, whose maintenance cost is patch-mode-independent
— plus `--n-mutations`, `--load-factor` on both mutation benches;
`client_rewind_staleness` takes `--batch-size`, `--staleness-steps`,
`--queries`; `headtohead_query` / `headtohead_{hint_patch,rewind}_decode`
require `--num-keys`.

### Low-level: `cargo bench`

`bench.sh` is a thin wrapper; the benches also run standalone — then
`--plaintext-bits` defaults to `8` and output lands in the crate-local
`results/` unless `IKPIR_RESULTS_DIR` is set:

```bash
cargo bench -p ikpir-client --bench client_hint_patch_decode -- --backend simple --plaintext-bits 10
cargo bench -p ikpir-client --bench client_rewind_mutation -- --n-mutations 64
cargo bench -p ikpir-client --bench <name> -- --help
```

## Lifecycle

Client-rewind (`RewindClient`):

```
from_setup(bundle)                  — initialise from server's setup bundle
  │
  └── loop:
        build_query(key)            — one B::Query per segment
        [send queries to server]
        server.answer(&q)           — server returns PirResponseBundle
        decode(key, &q, &resp)      — rewind the response, fp match → Option<Vec<u8>>
        accumulate_delta(delta)     — roll the published ΔD forward (epoch+1)
  │
  └── (optional) collect_garbage()  — fold ΔD into the hint, reclaim the
        per-query correction cost
  │
  └── on FutureDelta / after server full_rebuild:
        reset_from(new_bundle)      — replace all internal state
```

Client-hint-patch (`HintPatchClient`):

```
from_setup(bundle)                  — initialise from server's setup bundle
  │
  └── loop:
        build_query(key)            — one B::Query per segment
        [send queries to server]
        server.answer(&q)           — server returns PirResponseBundle
        decode(key, &resp)          — fp match → Option<Vec<u8>>, no query threading
        apply_delta(delta)          — fold the published delta into the hint (epoch+1)
  │
  └── on FutureDelta / after server full_rebuild:
        reset_from(new_bundle)      — replace all internal state
```

`from_setup` re-expands each segment's LWE matrix `A` from the wire-shipped
seed, which is the whole cost of bootstrapping — gigabytes of keystream at
paper scale, single-threaded. For any backend implementing
`ParallelSetupBackend` (both shipped ones do), `from_setup_parallel` and
`reset_from_parallel` build the identical client across all cores, on either
flow:

```rust
// Interchangeable — same queries, same decodes, same epoch.
let client = RewindClient::<FrodoPirBackend>::from_setup(bundle);
let client = RewindClient::<FrodoPirBackend>::from_setup_parallel(bundle);
```

Worker count comes from `IKPIR_SETUP_THREADS`, else the machine's available
parallelism. All nine benches use the parallel path — none of them reports
client-bootstrap cost.

See [`../../docs/rewind-client-mode.md`](../../docs/rewind-client-mode.md)
for how the two flows relate and how to pick between them.

## Status and wire-format stability

Two backends ship: `FrodoPirBackend` (default `lwe_dim = 1566`, ternary
errors, tall-skinny matrix) and `SimplePirBackend` (default `lwe_dim =
1275`, discrete-Gaussian errors with σ = 6.4, `√N × √N` internal
reshape). Both defaults target 128-bit security, estimated via the
lattice estimator under the ADPS16 cost model. Both are drop-in
alternatives at the `B: IndexPirBackend` type parameter on
`RewindClient` / `HintPatchClient`. Bundle types are not versioned; serialisation is out of
scope.
