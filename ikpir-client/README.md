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

Six `clap`-parsed CSV-emitting benches under `benches/`. Output lands in
`results/`. Each invocation produces one CSV row (the writer is
append-aware); a sweep is the orchestrator's job — `rm` the CSV, then
loop `cargo bench` over the configs you care about. Pass arguments
after `--`, e.g. `cargo bench --bench query_throughput -- --num-buckets 32768`.

Four of the six benches are criterion-backed (`query_throughput`,
`decode_throughput`, `apply_delta_throughput`, `preprocess_throughput`)
and emit a browsable HTML/JSON report under
`target/criterion/<label>/`. The remaining two (`client_setup_latency`,
`client_memory_footprint`) use a manual `Instant`-based timing loop or
closed-form accounting respectively.

### Bench overview

| Bench | What it measures | CSV (under `results/`) | Variable knobs the orchestrator typically iterates |
|---|---|---|---|
| `query_throughput` | `client.build_query` rate (queries/sec, criterion); per cold/warm-b/warm-bc preprocessing mode | `ikpir_client_query_throughput.csv` | `--mode`, `--num-buckets`, `--value-bits` |
| `decode_throughput` | `client.decode` rate (queries/sec, criterion); `lwe_dim × row_width` matvec + slot scan + fp match, per mode | `ikpir_client_decode_throughput.csv` | `--mode`, `--num-buckets`, `--value-bits` |
| `apply_delta_throughput` | `client.apply_delta` rate (deltas/sec, criterion); incremental hint patch fold, optionally with precomputed `c` queue | `ikpir_client_apply_delta_throughput.csv` | `--load-factor`, `--num-buckets`, `--precomputed-slots` |
| `preprocess_throughput` | Phase B (`b = A·s + e`) and Phase C (`c = sᵀ·H`) precomputation rate (slots/sec, two criterion bench_functions) | `ikpir_client_preprocess_throughput.csv` | `--batch`, `--num-buckets`, `--value-bits` |
| `client_setup_latency` | `IkpirClient::from_setup` latency, optionally also `precompute_queries` + `precompute_decodes` | `ikpir_client_setup_latency.csv` | `--with-precompute`, `--num-buckets`, `--lwe-dim` |
| `client_memory_footprint` | Heap + stack footprint of `IkpirClient<FrodoPirBackend>` per preprocessing mode (closed-form, no timing) | `ikpir_client_memory_footprint.csv` | `--mode`, `--num-buckets`, `--lwe-dim` |

`num_buckets` constraints differ per arity: 2-ary `2^t`, 3-ary `3·2^t`,
4-ary `2^t ≥ 4`. The default `--num-buckets` is set per arity by
`benches/helpers.rs::default_num_buckets_for_arity` so that capacity is
≈2^16 slots at `bucket_size = 4`.

#### Preprocessing modes (`--mode cold` / `warm-b` / `warm-bc`)

The client can amortise per-query LWE work across batches via two
precomputation phases. The mode flag selects how much of that work has
already been done before the timed loop starts:

| Mode | What's primed | `build_query` cost | `decode` cost | Extra heap per prepared slot |
|---|---|---|---|---|
| `cold` | nothing | inline LWE sample (`s`, `b = A·s + e`) per segment | full `lwe_dim × row_width` matvec | 0 |
| `warm-b` | `precompute_queries(N)` has populated `N` query slots per segment | consume one prepared `(s, b)` off the queue (one vector add) | unchanged from cold | `(lwe_dim + n_rows) × 4 B` |
| `warm-bc` | also `precompute_decodes()` has filled `c = sᵀ·H` for every prepared / in-flight slot | (as `warm-b`) | one vector subtract + rounding (cheap path) | `(lwe_dim + n_rows + row_width) × 4 B` |

Prepared `(s, b)` material is database-independent, so it survives
mutations; `apply_delta` keeps the precomputed `c` values consistent with
the patched hint. `client_memory_footprint` reports the per-mode heap
cost; `query_throughput` / `decode_throughput` report the per-mode
speedup for the corresponding hot path.

### Common flags (all benches)

| Flag | Default | Meaning |
|---|---|---|
| `--arity <N>` | `2` | Cuckoo arity (2, 3, 4) |
| `--num-buckets <N>` | per-arity (see `default_num_buckets_for_arity`) | Buckets per segment |
| `--bucket-size <N>` | `4` | Slots per bucket (1–4) |
| `--value-bits <N>` | `256` | Value width per entry |
| `--fingerprint-bits <N>` | `32` | Fingerprint width |
| `--plaintext-bits <N>` | `8` | PIR plaintext-cell bit-width (1–31) |
| `--lwe-dim <N>` | `1774` | FrodoPIR LWE dimension `n` |
| `--batch <N>` | `64` | Operations per timed routine (criterion) or per `precompute_queries` chunk |

### Bench-specific flags

| Bench | Flags (with defaults) |
|---|---|
| `query_throughput`, `decode_throughput` | `--mode cold` (`cold`/`warm-b`/`warm-bc`) — criterion handles warmup/sample-count itself |
| `preprocess_throughput` | _common flags only_ — two criterion bench_functions (`preprocess_phase_b`, `preprocess_phase_c`) |
| `client_setup_latency` | `--warmup 2`, `--trials 5`, `--with-precompute` (also time `precompute_queries`/`precompute_decodes`) |
| `apply_delta_throughput` | `--batch 10000` (deltas pre-collected before timing), `--precomputed-slots 0`, `--load-factor 0.50` |
| `client_memory_footprint` | `--mode cold` (no warmup/trials — closed-form accounting) |

### Orchestrator-driven sweeps

Each `cargo bench` invocation appends one row to its CSV. To produce a
multi-row CSV that `scripts/plot.py` can consume, the orchestrator (shell
or Python) deletes the file first then loops:

```bash
rm -f results/ikpir_client_query_throughput.csv
for mode in cold warm-b warm-bc; do
    for nb in 4096 8192 16384 32768; do
        cargo bench -p ikpir-client --bench query_throughput -- \
            --mode $mode --num-buckets $nb
    done
done
python scripts/plot.py
```

For the four criterion benches, the same loop also accumulates HTML
reports under `target/criterion/<label>/` (criterion keeps the last
run per `(group, function)` label — clean it manually if you want a
fresh baseline).

### Examples

```bash
# Default single config (one CSV row).
cargo bench -p ikpir-client --bench query_throughput

# Pin one specific config.
cargo bench -p ikpir-client --bench decode_throughput -- \
    --num-buckets 32768 --bucket-size 4 --value-bits 64 --lwe-dim 1024

# apply_delta with a populated precomputation queue (load-bearing case).
cargo bench -p ikpir-client --bench apply_delta_throughput -- \
    --num-buckets 16384 --precomputed-slots 64

# Decode throughput in fully warm mode.
cargo bench -p ikpir-client --bench decode_throughput -- \
    --mode warm-bc --num-buckets 16384

# Setup latency including phase-B + phase-C precompute.
cargo bench -p ikpir-client --bench client_setup_latency -- \
    --with-precompute --num-buckets 16384 --trials 10

# Memory footprint at one mode.
cargo bench -p ikpir-client --bench client_memory_footprint -- --mode warm-bc

# Per-arity comparison.
cargo bench -p ikpir-client --bench query_throughput -- --arity 3
cargo bench -p ikpir-client --bench decode_throughput -- --arity 4 --num-buckets 16384

# Live flag list for any bench.
cargo bench -p ikpir-client --bench <name> -- --help
```

### Plotting

Render plots from the CSVs with `scripts/plot.py` (matplotlib + pandas):

```bash
pip install -r scripts/requirements.txt
python scripts/plot.py                          # all available plots → results/plots/
python scripts/plot.py --list                   # list plot functions
python scripts/plot.py decode_throughput        # one specific plot
```

Plot ↔ bench mapping:

| Plot function | Bench (CSV consumed) | Output PNG |
|---|---|---|
| `query_throughput` | `query_throughput` | `query_throughput.png` |
| `decode_throughput` | `decode_throughput` | `decode_throughput.png` |
| `apply_delta_throughput` | `apply_delta_throughput` | `apply_delta_throughput.png` |

Benches without a packaged plotter (`preprocess_throughput`,
`client_setup_latency`, `client_memory_footprint`) emit CSV under
`results/` for ad-hoc analysis.

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
