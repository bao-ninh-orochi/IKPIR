# CLAUDE.md — ikpir-client crate

## 1. Crate purpose

Client-side IKPIR: holds `CuckooParams` and per-segment `B::ClientState`
plus an epoch counter. Translates user-level `(key, query)` operations
into wire-level Index-PIR query/response bundles defined in
[`ikpir-common`](../ikpir-common/CLAUDE.md). The client **never** owns a
`CuckooKVStore`; its only persistent material from the server is the
setup bundle. Its sole update strategy is **response-rewind**
(`docs/rewind-client-mode.md`); client-side hint-patching survives only
as a benchmark comparator behind the `hint-patch-bench` Cargo feature.

## 2. File map

| File | Role |
|---|---|
| `src/lib.rs` | Declares `mod client; mod error; mod pending;` (+ `pub mod bench_comparator` behind `hint-patch-bench`) and re-exports `IkpirClient`, `DeltaApplyOutcome`, `IkpirClientError`, plus the shared protocol surface (`IndexPirBackend`, `ResponseRewind`, `FrodoConfig`, `SimpleConfig`, wire bundles, `IkpirError`) from `ikpir-common` |
| `src/client.rs` | `IkpirClient<B>` generic + its public methods — response-rewind, the sole production update strategy (see §3, §7) |
| `src/error.rs` | `IkpirClientError` enum (5 protocol variants + `Server(IkpirError)` forward; `IkpirError` is re-exported from `ikpir-common`) |
| `src/pending.rs` | `PendingDelta` — the rolling per-segment `ΔD` accumulator (crate-private) |
| `src/bench_comparator/mod.rs` | `HintPatchClient<B>` — the classical hint-patch client, kept only as a benchmark comparator; `#[cfg(feature = "hint-patch-bench")]`, disabled by default |

> Production code in this crate depends only on `ikpir-common` and
> `segmented-cuckoo`. `ikpir-server` is carried as a `[dev-dependency]`
> for `tests/client_e2e.rs`, the six benches, and the quick-start doctest.

## 3. Key design decisions (the WHY)

- **Params-only state** — the client stores `CuckooParams` and per-segment
  `ClientState`. Cells, mutation log, and the KV store are server-side.
  Lookup geometry (`candidate_buckets`) is public, not secret; the client
  re-derives it on every query without any privacy cost.

- **Client re-expands `A` locally from the wire-shipped seed** — the
  `ServerSetupBundle` does **not** carry the LWE public matrix `A`.
  During `from_setup`, the backend's `client_setup` calls
  `B::expand_hint_material(&params)`, which deterministically reproduces
  `A` from the 16-byte seed inside `B::ServerParams`. This is invisible
  to callers (no extra round trips, no protocol-visible difference); it
  just keeps the wire bundle small and centralises `A` ownership at the
  client. The materialised matrix lives inside `B::ClientState` and is
  used by both `client_query` (LWE matvec) and `client_patch_state`
  (the hint patch `collect_garbage` folds `ΔD` through).

- **Parallel per-segment queries** — `build_query` emits one `B::Query`
  per segment (j-th query targets row `indices[j] % segment_size` in
  segment j). The server processes each segment independently in `answer`.

- **`fp` re-derivation in `decode`** — `decode(key, query, resp)` re-runs
  `candidate_buckets(key)` to recover `fp` deterministically. No query
  IDs, no stashed state, no extra wire field. There is no privacy reason
  to hide `fp` from the client itself.

- **Dual-path recovery** — `accumulate_delta` for the steady state (strict
  monotone epoch+1 accumulate), `reset_from` after `full_rebuild` or after a
  `FutureDelta` gap that cannot be bridged incrementally.

- **Two bootstrap implementations, one result** — `from_setup` /
  `reset_from` re-expand each segment's `A` **single-threaded**, which is
  the entire cost of bootstrapping a client (`Θ(arity · n_rows · lwe_dim)`
  ChaCha20 words — gigabytes at paper scale). `from_setup_parallel` /
  `reset_from_parallel` (available whenever `B: ParallelSetupBackend`,
  which both shipped backends are) do the same across all cores and yield
  an observationally identical client: same queries, same decodes, same
  patch behaviour, same epoch. Both pairs share one body via a
  `PerSegmentClientSetup<B>` fn pointer.

  No bench reports client-bootstrap cost, so all six build their client
  with `from_setup_parallel`. The reference path stays the default so a
  future bootstrap-cost measurement has something honest to call.

- **Response-rewind is the sole production update strategy** — the client
  pins its bootstrap hint `H₀` at setup time and never patches it.
  `accumulate_delta` rolls the published `ΔD` forward (`Θ(τ·ω)` per batch
  — a factor-`n` cheaper client maintenance than patching the hint
  directly), `decode` corrects a head-answered response back to `H₀`'s
  epoch before decoding (via the per-backend `ResponseRewind` trait from
  `ikpir-common`), and `collect_garbage` reclaims the staleness-growing
  per-query correction by folding `ΔD` into the hint on demand (using
  `HintPatchMode::EntryLevel`, hardcoded — there is no client-selectable
  patch realization in production). Full mechanism and correctness:
  `docs/rewind-client-mode.md`.

- **Client-side hint-patching is a benchmark comparator only** —
  `HintPatchClient<B>` (`src/bench_comparator/`, gated behind the
  `hint-patch-bench` Cargo feature, disabled by default) mirrors the
  client's pre-pivot shape: `apply_delta` folds each delta into the hint
  immediately (`Θ(n·τ·ω)` per batch), and `decode` reads it directly, with
  a selectable `HintPatchMode` realization (`set_hint_patch_mode`). It
  exists solely so `benches/client_mutation.rs` can measure the two
  strategies head-to-head for the paper's §6.2 evaluation, and so
  `tests/rewind_equivalence.rs` can pin that they agree. A production
  `cargo build --release` (default features) never links this module in.

## 4. Epoch state machine

```
        accumulate_delta(delta.epoch == self.epoch + 1)
               ┌──────────────────────────────────────┐
               │                                      ▼
 [from_setup] epoch=E  ──────────────────────▶  epoch=E+1  ···
               │
               │  accumulate_delta(epoch ≤ self.epoch)   → StaleDelta
               │  accumulate_delta(epoch > self.epoch+1) → FutureDelta
               │                                        ↓ caller must:
               └──────────────────────────────▶  reset_from(new_bundle)
```

- `accumulate_delta` is strict-monotone: only `delta.epoch == self.epoch + 1`.
- `decode` requires `resp.epoch == self.epoch`; mismatch → `EpochMismatch`.
- `pin_epoch` (where the pinned hint `H₀` sits) trails `epoch` by the
  accumulated `ΔD` span; advances only on `collect_garbage` or `reset_from`.

## 5. Failure-mode table

| Variant | Source | Meaning |
|---|---|---|
| `StaleDelta` | `accumulate_delta` (or the bench comparator's `apply_delta`) | `delta.epoch ≤ self.epoch` |
| `FutureDelta` | `accumulate_delta` (or `apply_delta`) | `delta.epoch > self.epoch + 1` (gap) |
| `EpochMismatch` | `decode` | server moved between query and answer |
| `MalformedBundle` | `accumulate_delta` / `decode` (or `apply_delta`) | params mismatch, or wrong segment count / row width |
| `CellOutOfRange` | `decode` | a corrected cell escaped `[0, 2^plaintext_bits)` — corrupt/inconsistent delta or response, never a wrong value |
| `Server(IkpirError)` | forward | for synchronous in-process composition |

## 6. Entry points and test taxonomy

| Task | Where to look |
|---|---|
| Build a fresh client | `client.rs::IkpirClient::from_setup` |
| Bootstrap a client fast (untimed preamble) | `client.rs::IkpirClient::{from_setup_parallel, reset_from_parallel}` — identical client, all cores; contract in `ikpir-common::ParallelSetupBackend` |
| Issue a query | `client.rs::IkpirClient::build_query` |
| Decode a response | `client.rs::IkpirClient::decode` (threads the query bundle; corrects for accumulated `ΔD`) |
| Accumulate a delta | `client.rs::IkpirClient::accumulate_delta` |
| Reclaim staleness | `client.rs::IkpirClient::collect_garbage` |
| Recover from a gap | `client.rs::IkpirClient::reset_from` |
| Debug a fingerprint mismatch | `client.rs::IkpirClient::decode` — check `candidate_buckets` + `unpack_slot_cells` |
| Hint-patch bench comparator | `bench_comparator/mod.rs::HintPatchClient` (`hint-patch-bench` feature) — `from_setup(_parallel)`, `apply_delta`, `decode`, `{hint_patch_mode, set_hint_patch_mode}` |
| Integration tests | `tests/client_e2e.rs` + `tests/simple_client_e2e.rs` (mirror of `client_e2e.rs` for `SimplePirBackend`); `tests/replay_equivalence.rs` — the mutation benches' `reset_for_replay` harness measures what a fresh setup would (both backends, arities 2/3/4, plus a stale-hints negative control); `tests/rewind_equivalence.rs` (feature-gated, `hint-patch-bench`) — production rewind == bench-only hint-patch == fresh decode (both backends × arities 2/3/4), GC-then-query, post-pin insert |
| Benches | `benches/client_query.rs`, `benches/client_decode.rs`, `benches/client_mutation.rs` (needs `hint-patch-bench` for its `--update-mode patch` sweep), `benches/client_rewind_staleness.rs`, `benches/headtohead_query.rs`, `benches/headtohead_decode.rs`. All accept `--backend frodo\|simple`; run via `../../scripts/bench.sh <name>` |
| Backend enum (bench CLI) | `benches/helpers.rs::Backend` + `backend_default_lwe_dim` — duplicated in `ikpir-server/benches/helpers.rs` |

### Bench layer (under `benches/`)

Six focused benches covering classical and incremental client criteria for the paper:

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `client_query` | `TableFull` | `build_query` rate (queries/sec, criterion, warm-bc) | `ikpir_client_query.csv` |
| `client_decode` | `TableFull` | `decode` rate (queries/sec, criterion, warm-bc) | `ikpir_client_decode.csv` |
| `client_mutation` | `--load-factor` (0.90) | per-batch **maintenance** throughput per (kind, `--update-mode` patch\|rewind [, `--patch-mode` entry\|row]): `patch` times the bench-only `HintPatchClient::apply_delta` (needs `hint-patch-bench`), `rewind` times the production `IkpirClient::accumulate_delta`; wall-clock, empty queue; `pending_cells` = final \|ΔD\| (0 for `patch`); one setup per config, deltas collected from a server rewound per kind with `reset_for_replay` | `ikpir_client_mutation.csv` |
| `client_rewind_staleness` | `--load-factor` (0.90) | production `IkpirClient::decode` per-query latency vs staleness \|ΔD\| (one client accumulates over `--staleness-steps` × `--batch-size` updates, never GC'd, then a `collect_garbage` returns it to baseline); times only `decode` | `ikpir_client_rewind_staleness.csv` |
| `headtohead_query` | fixed `--num-keys` | `build_query` rate at a fixed keyword count (fair comparison vs ChalametPIR / Hao 2025); mirrors `client_query` + `num_keys`/`db_size` columns | `ikpir_headtohead_client_query.csv` |
| `headtohead_decode` | fixed `--num-keys` | `decode` rate at a fixed keyword count; mirrors `client_decode` + `num_keys`/`db_size` columns, with the once-per-config `verify_decode` sanity check | `ikpir_headtohead_client_decode.csv` |

`client_query` and `client_decode` use **warm-bc** mode (precompute the
prepared-query queue + decode material before the timed loop), so the
timed call hits the cheap amortised path.

`client_mutation` runs the
client in **empty-queue** mode (no `precompute_queries` /
`precompute_decodes`). Each `apply_delta` / `accumulate_delta` then patches
only the hint `H` / rolls `ΔD` forward — the queue-iteration inside
`client_patch_state` is a no-op when the queue is empty — so the timing
reports the "client maintenance" cost in isolation, without warm-bc
queue-maintenance overhead mixed in.

- Each bench is `harness = false` and parses CLI via `clap` (see helpers
  `parse_cli` / `parse_cli_with_matches`). Per-arity dispatch happens
  through `MakeStore` / `CloneStore`; the typed scheme is picked once in
  `main` based on `--arity`.
- **Backend dispatch.** Every bench exposes `--backend frodo|simple`
  (default `frodo`). A two-level match in `main` picks the typed
  `<S, B>` pair; `run_one` is generic over both. `--lwe-dim` defaults
  to the backend-appropriate value (1566 for Frodo, 1275 for Simple)
  via `helpers::backend_default_lwe_dim`.
- **Plaintext bits.** Every bench accepts `--plaintext-bits` (default
  `8` — a safe lower bound that works for every backend and every DB
  size). `scripts/bench.sh` overrides this per `(backend, geometry,
  value_bits)` via `scripts/lib.sh::backend_plaintext_bits` so each run
  uses the largest `pb` admitted by the backend's correctness bound at
  `q = 2^32`. The chosen value is written to every CSV row as the
  `plaintext_bits` column.
- **Patch modes.** `client_mutation` accepts `--patch-mode entry|row`
  (comma-separated list, default `entry`) and emits one CSV row per
  `(patch mode, kind)` pair; the `patch_mode` column records which
  `HintPatchMode` realization the bench-only `HintPatchClient` used for
  its timed `apply_delta` loop (`patch_mode` = `-` for `rewind` rows,
  which are patch-mode-independent). Deltas
  are collected once per kind (identical under either mode) and replayed
  per mode — from **one** server per config, rewound before each kind with
  `IkpirServer::reset_for_replay` (fresh store from the snapshot cells,
  clone of the epoch-0 hints), whose epoch-0 `setup()` bundle also
  bootstraps every timed client. `tests/replay_equivalence.rs` pins that a
  replay yields the same deltas as a fresh setup. `scripts/bench.sh`
  passes `entry,row` by default (and `--features hint-patch-bench`,
  needed only for this one bench).
- **Runner.** `scripts/bench.sh <bench> [flags]` maps the bench to its
  crate, auto-derives `--plaintext-bits` / `--lwe-dim`, and exports
  `IKPIR_RESULTS_DIR=results/ikpir-client` before `cargo bench`. One
  invocation = one CSV row (append-mode `csv_writer`, honoring
  `IKPIR_RESULTS_DIR`; default `results/`). Its geometry defaults are dev
  scale, not the paper's: the paper matrix lives in `scripts/lib.sh`
  (`PAPER_*`) and is swept by `scripts/table3.sh` (online, via
  `headtohead_{query,decode}`) and `table4.sh` (mutation, via
  `client_mutation`). `scripts/smoke.sh` runs every PIR bench tiny.
- Shared helpers in `benches/helpers.rs` (deliberately duplicated across
  crates — a common core is mirrored in `ikpir-server/benches/helpers.rs`,
  but this copy additionally carries `verify_decode`, which round-trips
  through both client and server — a dev-dep cycle on the server side —
  and backs the `client_decode` / `headtohead_decode` sanity checks):
    - `populate_until_full::<S>(…)` / `populate_to_load::<S>(load_factor, …)`
      — seed a `CuckooKVStore<S>` to `TableFull` or to a target load.
    - `print_preamble(name, knobs, store_state, geom)` — the standard
      `=== <bench> ===` / Parameters / KV store / Geometry banner.
    - `configured_criterion()` — the `Criterion` pinned to the shared
      Table 3 contract (100 samples, 3 s warm-up, 5 s measurement), which
      `client_query`, `client_decode`, `headtohead_query`, and
      `headtohead_decode` drive directly through `iter_custom`.
- `client_mutation` uses wall-clock `Instant` batch timing (not criterion)
  because `apply_delta` / `accumulate_delta` advances the client epoch with
  each call; criterion's cycling pattern is not meaningful when state
  changes between calls.

**Per-segment data flow (client annotations):**

```
                     ┌──────── arity-2 SCF ────────┐
key  ──candidate_buckets──▶  (fp, [b0, b1])
                                    │      │
                            seg 0   │  seg 1
                      row = b0%N ◀──┘  b1%N ──▶ row
                               │           │
build_query: B::client_query   ▼           ▼
                              Q[0]        Q[1]    ── PirQueryBundle
                               │           │
       (server processes)      ▼           ▼
                              R[0]        R[1]    ── PirResponseBundle
                               │           │
decode: B::client_decode       ▼           ▼
                          row_cells    row_cells
           slot scan: unpack_slot_cells → fp match? → value
```
