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

Eight `clap`-parsed CSV-emitting benches under `benches/`. Output lands in
`results/`. Each invocation produces one CSV row (the writer is
append-aware); a sweep is the orchestrator's job — `rm` the CSV, then
loop `cargo bench` over the configs you care about. Pass arguments after
`--`, e.g. `cargo bench --bench foo -- --num-buckets 32768`.

`answer_throughput` is criterion-backed: it also emits a browsable
HTML/JSON report under `target/criterion/answer_throughput/`. The other
seven benches use a manual `Instant`-based timing loop.

### Bench overview

`num_buckets` constraints differ per arity: 2-ary `2^t`, 3-ary `3·2^t`,
4-ary `2^t ≥ 4`. The default `--num-buckets` is set per arity by
`benches/helpers.rs::default_num_buckets_for_arity` so that capacity is
≈2^16 slots at `bucket_size = 4`.

| Bench | What it measures | CSV (under `results/`) | Variable knobs the orchestrator typically iterates |
|---|---|---|---|
| `setup_latency` | `IkpirServer::new` wall-clock — `B::server_setup` per segment | `ikpir_server_setup_latency.csv` | `--num-buckets`, `--lwe-dim` |
| `answer_throughput` | PIR matvec answer rate (queries/sec, criterion) | `ikpir_server_answer_throughput.csv` | `--num-buckets`, `--value-bits` |
| `incremental_vs_rebuild` | Incremental hint patch vs `full_rebuild` for N mutations | `ikpir_server_incremental_vs_rebuild.csv` | `--n-mutations`, `--load-factor` |
| `setup_to_first_query` | Cold-start latency to first decoded answer; per-phase breakdown | `ikpir_setup_to_first_query.csv` | `--mode`, `--num-buckets` |
| `steady_state_workload` | Mixed insert/query workload at a configurable query-to-mutation ratio | `ikpir_steady_state_workload.csv` | `--query-ratio`, `--n-inserts`, `--load-factor` |
| `wire_sizes` | Minimum on-wire byte sizes of every IKPIR bundle shape (no timing) | `ikpir_wire_sizes.csv` | `--num-buckets`, `--value-bits`, `--load-factor` |
| `failure_modes` | Rejection-path latency for `StaleEpoch` and `TableFull` | `ikpir_failure_modes.csv` | `--arity` |

### Common flags

Every bench accepts these config knobs (defaults are academic-paper scale):

| Flag | Default | Meaning |
|---|---|---|
| `--arity <N>` | `2` | Cuckoo arity (2, 3, 4) |
| `--num-buckets <N>` | per-arity (see `default_num_buckets_for_arity`) | Buckets per segment |
| `--bucket-size <N>` | `4` | Slots per bucket (1–4) |
| `--value-bits <N>` | `256` | Value width per `(key, value)` entry |
| `--fingerprint-bits <N>` | `32` | Fingerprint width |
| `--plaintext-bits <N>` | `8` | PIR plaintext-cell bit-width (1–31) |
| `--lwe-dim <N>` | `1774` | FrodoPIR LWE dimension `n` |

### Bench-specific flags

| Bench | Flags (with defaults) |
|---|---|
| `setup_latency` | `--warmup 2`, `--trials 5` |
| `answer_throughput` | `--batch 64` (criterion handles warmup/sample-count itself) |
| `setup_to_first_query` | `--warmup 2`, `--trials 5`, `--mode cold` (`cold`/`warm-b`/`warm-bc`) |
| `steady_state_workload` | `--warmup 1`, `--trials 3`, `--n-inserts 4096`, `--n-queries 410`, `--query-ratio 10`, `--load-factor 0.50` |
| `incremental_vs_rebuild` | `--n-mutations 1024`, `--load-factor 0.80` (numeric or the sentinel `full` → populate to `TableFull`) |
| `wire_sizes` | `--load-factor 0.50` |
| `failure_modes` | `--num-trials 2000` per failure kind, `--num-buckets 256` (stale-epoch), `--full-num-buckets 16` (table-full), `--arity 2` |

#### Preprocessing modes (`setup_to_first_query --mode cold|warm-b|warm-bc`)

`setup_to_first_query` measures end-to-end first-query latency assuming
the client has already done some amount of LWE precomputation before the
query arrives. The mode picks how warm the client is:

| Mode | Client state at query time | `build_query` path | `decode` path |
|---|---|---|---|
| `cold` | nothing precomputed | inline LWE sample (`s`, `b = A·s + e`) per segment | full `lwe_dim × row_width` matvec |
| `warm-b` | `precompute_queries(N)` has populated query slots per segment | consume one prepared `(s, b)` off the queue (one vector add) | unchanged from cold |
| `warm-bc` | also `precompute_decodes()` has filled `c = sᵀ·H` for every prepared / in-flight slot | (as `warm-b`) | one vector subtract + rounding (cheap path) |

`cold` reports the worst-case first-query latency for a freshly-restored
client; `warm-bc` reports the steady-state best case. The reported
`build_query_ms`, `decode_ms`, etc. columns isolate where the savings
land. See `ikpir-client`'s README for memory cost per mode.

### Orchestrator-driven sweeps

Each `cargo bench` invocation appends one row to its CSV. To produce a
multi-row CSV that `scripts/plot.py` can consume, the orchestrator (shell
or Python) deletes the file first then loops:

```bash
rm -f results/ikpir_server_answer_throughput.csv
for nb in 4096 8192 16384 32768; do
    for vb in 64 256; do
        cargo bench -p ikpir-server --bench answer_throughput -- \
            --num-buckets $nb --value-bits $vb
    done
done
python scripts/plot.py
```

For `answer_throughput`, the same loop also accumulates criterion HTML
reports under `target/criterion/answer_throughput/` (criterion keeps the
last run per `(group, function)` label — clean it manually if you want a
fresh baseline).

### Examples

```bash
# Default single config (one CSV row).
cargo bench -p ikpir-server --bench answer_throughput

# Pin one specific config.
cargo bench -p ikpir-server --bench answer_throughput -- \
    --num-buckets 32768 --bucket-size 4 --value-bits 256 \
    --lwe-dim 1024 --batch 64

# Headline incremental-vs-rebuild crossover at N = 1024 mutations.
cargo bench -p ikpir-server --bench incremental_vs_rebuild -- \
    --num-buckets 16384 --n-mutations 1024 --load-factor 0.50

# Cold-start latency in warm-b mode.
cargo bench -p ikpir-server --bench setup_to_first_query -- \
    --mode warm-b --num-buckets 16384 --trials 10

# Workload mix — query-heavy ratio.
cargo bench -p ikpir-server --bench steady_state_workload -- \
    --n-inserts 4096 --n-queries 410 --query-ratio 50

# Wire-size catalogue at one config.
cargo bench -p ikpir-server --bench wire_sizes -- --arity 3 --num-buckets 24576

# Rejection-path microbench.
cargo bench -p ikpir-server --bench failure_modes -- --num-trials 5000

# Per-arity comparison.
cargo bench -p ikpir-server --bench answer_throughput -- --arity 3
cargo bench -p ikpir-server --bench answer_throughput -- --arity 4 --num-buckets 16384

# Live flag list for any bench.
cargo bench -p ikpir-server --bench <name> -- --help
```

### Plotting

Render plots from the CSVs with `scripts/plot.py` (matplotlib + pandas):

```bash
pip install -r scripts/requirements.txt
python scripts/plot.py                          # all available plots → results/plots/
python scripts/plot.py --list                   # list plot functions
python scripts/plot.py incremental_vs_rebuild   # one specific plot
```

Plot ↔ bench mapping:

| Plot function | Bench (CSV consumed) | Output PNG |
|---|---|---|
| `setup_latency` | `setup_latency` | `setup_latency.png` |
| `answer_throughput` | `answer_throughput` | `answer_throughput.png` |
| `incremental_vs_rebuild` | `incremental_vs_rebuild` | `incremental_vs_rebuild.png` |

Benches without a packaged plotter (`setup_to_first_query`,
`steady_state_workload`, `wire_sizes`, `failure_modes`)
emit CSV under `results/` for ad-hoc analysis with pandas, gnuplot, etc.

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

## Usage

  ./scripts/run_all.sh                                    # default profile, ~30-45 min
  IKPIR_BENCH_PROFILE=quick ./scripts/run_all.sh          # ~5-10 min
  IKPIR_BENCH_PROFILE=full ./scripts/run_all.sh           # adds m=2^22 (~1+ hour)
  ./scripts/run_all.sh --plot-only                        # re-plot from existing CSVs
  ./scripts/run_all.sh --server-only                      # just the server side
  ./ikpir-server/scripts/run_benches.sh answer_throughput # one bench only