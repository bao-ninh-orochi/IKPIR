# ikpir-client

Client-side crate for [Incremental Keyword PIR](../../README.md). Holds
per-segment `ClientState` and translates `(key, value)` lookups into
wire-level Index-PIR query/response exchanges with `ikpir-server`.

## Quick start

```rust
use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, Segmented2aryIkpirServer};
use segmented_cuckoo::Segmented2aryCuckooKVStore;

// In a real deployment the bundle arrives over the network.
let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
store.insert(b"alice", &[0x01u8]).unwrap();

let mut server: Segmented2aryIkpirServer<FrodoPirBackend> =
    Segmented2aryIkpirServer::new(store, FrodoConfig::default());
let bundle = server.setup();

// Initialise client from the setup bundle.
let mut client: IkpirClient<FrodoPirBackend> = IkpirClient::from_setup(bundle);

// Round-trip a key.
let q = client.build_query(b"alice");
let r = server.answer(&q).unwrap();
let v = client.decode(b"alice", &r).unwrap().expect("key present");
assert_eq!(v, &[0x01u8]);
```

## Benches

Five focused `clap`-parsed CSV-emitting benches under `benches/`. The
recommended way to run one is the workspace runner
[`../../scripts/bench.sh`](../../scripts/bench.sh), which auto-derives the largest
correct `--plaintext-bits` and the backend `--lwe-dim`, and routes output to
`results/ikpir-client/`:

```bash
./scripts/bench.sh client_decode --arity 4 --num-buckets 65536 --value-bits 256
./scripts/bench.sh client_mutation --patch-mode entry,row
./scripts/bench.sh                              # -h: full flag + bench list
```

Each invocation is one config = one appended CSV row (`client_mutation` emits
one row per `(patch mode, kind)` pair). `client_query` / `client_decode` run in
**warm-bc** mode (precompute before the timed loop); `client_mutation` runs in
**empty-queue** mode so `apply_delta` reports the hint-patch cost in isolation.
The root [README](../../README.md#benches) has the paper config matrix.

### Bench overview

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `client_query` | `TableFull` | `build_query` rate (queries/sec, criterion, warm-bc) | `ikpir_client_query.csv` |
| `client_decode` | `TableFull` | `decode` rate (queries/sec, criterion, warm-bc) | `ikpir_client_decode.csv` |
| `client_mutation` | `--load-factor` (0.90) | `apply_delta` throughput per (patch mode, kind), wall-clock, empty-queue | `ikpir_client_mutation.csv` |
| `headtohead_query` | fixed `--num-keys` | `build_query` rate at a fixed keyword count; +`num_keys`/`db_size` cols | `ikpir_headtohead_client_query.csv` |
| `headtohead_decode` | fixed `--num-keys` | `decode` rate at a fixed keyword count; +`num_keys`/`db_size` cols | `ikpir_headtohead_client_decode.csv` |

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

Bench-specific: `client_query` / `client_decode` / `headtohead_*` take `--batch`
(key-pool size); `client_mutation` takes `--patch-mode entry\|row` (comma list,
default `entry`), `--n-mutations`, `--load-factor`; `headtohead_query` /
`headtohead_decode` require `--num-keys`.

### Low-level: `cargo bench`

`bench.sh` is a thin wrapper; the benches also run standalone — then
`--plaintext-bits` defaults to `8` and output lands in the crate-local
`results/` unless `IKPIR_RESULTS_DIR` is set:

```bash
cargo bench -p ikpir-client --bench client_decode -- --backend simple --plaintext-bits 10
cargo bench -p ikpir-client --bench client_mutation -- --patch-mode entry,row --n-mutations 64
cargo bench -p ikpir-client --bench <name> -- --help
```

## Lifecycle

```
from_setup(bundle)                  — initialise from server's setup bundle
  │
  └── loop:
        build_query(key)            — one B::Query per segment
        [send queries to server]
        server.answer(&q)           — server returns PirResponseBundle
        decode(key, &resp)          — fp match → Option<Vec<u8>>
        apply_delta(delta)          — fold incremental hint update (epoch+1)
  │
  └── on FutureDelta / after server full_rebuild:
        reset_from(new_bundle)      — replace all internal state
```

`from_setup` re-expands each segment's LWE matrix `A` from the wire-shipped
seed, which is the whole cost of bootstrapping — gigabytes of keystream at
paper scale, single-threaded. For any backend implementing
`ParallelSetupBackend` (both shipped ones do), `from_setup_parallel` and
`reset_from_parallel` build the identical client across all cores:

```rust
// Interchangeable — same queries, same decodes, same epoch.
let client = IkpirClient::<FrodoPirBackend>::from_setup(bundle);
let client = IkpirClient::<FrodoPirBackend>::from_setup_parallel(bundle);
```

Worker count comes from `IKPIR_SETUP_THREADS`, else the machine's available
parallelism. All five benches use the parallel path — none of them reports
client-bootstrap cost.

On `perf/optimized` the two constructors are the same code: `from_setup`
already expands `A` across cores, because `sample_a` is the rayon ChaCha20
kernel. `IKPIR_SETUP_THREADS=1` or `--no-default-features` restores the
single-threaded bootstrap.

## Status and wire-format stability

Two backends ship: `FrodoPirBackend` (default `lwe_dim = 1566`, ternary
errors, tall-skinny matrix) and `SimplePirBackend` (default `lwe_dim =
1275`, discrete-Gaussian errors with σ = 6.4, `√N × √N` internal
reshape). Both defaults target 128-bit security, estimated via the
lattice estimator under the ADPS16 cost model. Both are drop-in
alternatives at the `B: IndexPirBackend` type parameter on
`IkpirClient`. Bundle types are not versioned; serialisation is out of
scope.
