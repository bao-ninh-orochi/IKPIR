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

Three `clap`-parsed CSV-emitting benches under `benches/`. Output lands
in `results/`. Each bench has three modes:

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
| `query_throughput` | `client.build_query` rate (queries/sec) — one LWE secret + ternary error sample + matvec per segment | arity ∈ {2,3,4} × num_buckets ∈ {64, 256, 1024} (3-ary: {96, 384, 1536}) × bucket_size ∈ {2, 4} × value_bits ∈ {8, 64, 256} |
| `decode_throughput` | `client.decode` rate (queries/sec) — `lwe_dim × row_width` matvec + slot scan + fp match | arity ∈ {2,3,4} × num_buckets ∈ {64, 256, 1024} (3-ary: {96, 384, 1536}) × bucket_size ∈ {2, 4} × value_bits ∈ {8, 64, 256} |
| `apply_delta_throughput` | `client.apply_delta` rate (deltas/sec) — incremental hint patch fold | arity ∈ {2,3,4} × num_buckets ∈ {256, 1024} (3-ary: {384, 1536}) × bucket_size ∈ {2, 4} × value_bits ∈ {8, 64, 256} |

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
| `--batch <N>` | `64` | Operations timed per measurement run |
| `--warmup <N>` | `2` | Warmup trials before measurement |
| `--trials <N>` | `5` | Measured trials per config |

All three client benches share the same flag set; there are no
bench-specific flags.

### Examples

```bash
# 1) Default single config (one CSV row, ~seconds)
cargo bench -p ikpir-client --bench query_throughput

# 2) Full sweep — produces the CSV that `scripts/plot.py` consumes
cargo bench -p ikpir-client --bench query_throughput        -- --sweep
cargo bench -p ikpir-client --bench decode_throughput       -- --sweep
cargo bench -p ikpir-client --bench apply_delta_throughput  -- --sweep

# 3) Pin one specific config (one CSV row)
cargo bench -p ikpir-client --bench decode_throughput -- \
    --num-buckets 1024 --bucket-size 4 --value-bits 64 \
    --lwe-dim 1024 --batch 128 --trials 10

# 4) Probe apply_delta at a non-default delta count
cargo bench -p ikpir-client --bench apply_delta_throughput -- \
    --num-buckets 1024 --batch 256

# 5) Compare arities at a fixed config
cargo bench -p ikpir-client --bench query_throughput -- \
    --arity 3                                    # arity 3 single config (auto-selects 384 buckets)
cargo bench -p ikpir-client --bench decode_throughput -- \
    --arity 4 --num-buckets 1024                 # explicit arity 4

# 6) Full flag list
cargo bench -p ikpir-client --bench query_throughput -- --help
```

### Plotting

Render plots from the CSVs with `scripts/plot.py` (matplotlib + pandas):

```bash
pip install -r scripts/requirements.txt
python scripts/plot.py            # all plots → results/plots/
python scripts/plot.py --list     # list individual plot functions
python scripts/plot.py decode_throughput   # one specific plot
```

Override paths via `IKPIR_CLIENT_RESULTS_DIR` / `IKPIR_CLIENT_PLOTS_DIR`.

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

`FrodoPirBackend` is the shipped backend. SimplePIR is a future track.
Bundle types are not versioned; serialisation is out of scope.
