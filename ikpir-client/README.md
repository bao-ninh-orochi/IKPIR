# ikpir-client

Client-side crate for [Incremental Keyword PIR](../README.md). Holds
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

Three focused `clap`-parsed CSV-emitting benches under `benches/`. Output
lands in `results/`. Each invocation produces one CSV row (append-mode);
a sweep is the orchestrator's job — `rm` the CSV first, then loop.
All benches run in **warm-bc** mode (precompute\_queries +
precompute\_decodes before the timed loop).

### Bench overview

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `client_query` | `TableFull` | `build_query` rate (queries/sec, criterion, warm-bc) | `ikpir_client_query.csv` |
| `client_decode` | `TableFull` | `decode` rate (queries/sec, criterion, warm-bc) | `ikpir_client_decode.csv` |
| `client_mutation` | `--load-factor` | `apply_delta` throughput per kind (insert/update/delete), wall-clock, warm-bc | `ikpir_client_mutation.csv` |

`num_buckets` constraints differ per arity: 2-ary `2^t`, 3-ary `3·2^t`, 4-ary `2^t ≥ 4`.

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

| Bench | Flag | Default | Meaning |
|---|---|---|---|
| `client_query`, `client_decode` | `--batch` | `64` | Key-pool size: the bench rotates through this many distinct keys so repeated iterations do not reuse hot CPU-cache state from the previous call. |
| `client_mutation` | `--n-mutations` | `1024` | Number of mutations per timed batch. |
| `client_mutation` | `--load-factor` | `0.80` | Initial store load fraction before timing starts. |

### Examples

```bash
# One config, one CSV row.
cargo bench -p ikpir-client --bench client_query -- \
    --arity 2 --num-buckets 65536 --bucket-size 4 --value-bits 256

# Decode throughput with SimplePIR.
cargo bench -p ikpir-client --bench client_decode -- \
    --backend simple --num-buckets 262144 --value-bits 2048 --batch 64

# apply_delta throughput at 80 % load, N=64 mutations per kind.
cargo bench -p ikpir-client --bench client_mutation -- \
    --arity 3 --num-buckets 393216 --bucket-size 2 --value-bits 256 \
    --n-mutations 64 --load-factor 0.80

# Flag list for any bench.
cargo bench -p ikpir-client --bench <name> -- --help
```

### Orchestrator sweep

`ikpir-client/scripts/run_benches.sh` sweeps the full paper config matrix
(20 configs × 3 value\_bits = 60 runs per bench; mutation bench × 7
N\_mutations = 420 runs). The orchestrator removes the CSV before each
sweep and re-runs per backend set in `IKPIR_BENCH_BACKENDS`.

```bash
# Client benches only, FrodoPIR.
./ikpir-client/scripts/run_benches.sh

# One bench.
./ikpir-client/scripts/run_benches.sh client_decode

# Both backends.
IKPIR_BENCH_BACKENDS=frodo,simple ./ikpir-client/scripts/run_benches.sh
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

## Status and wire-format stability

Two backends ship: `FrodoPirBackend` (default `lwe_dim = 1566`, ternary
errors, tall-skinny matrix) and `SimplePirBackend` (default `lwe_dim =
1275`, discrete-Gaussian errors with σ = 6.4, `√N × √N` internal
reshape). Both defaults target 128-bit security, estimated via the
lattice estimator under the ADPS16 cost model. Both are drop-in
alternatives at the `B: IndexPirBackend` type parameter on
`IkpirClient`. Bundle types are not versioned; serialisation is out of
scope.
