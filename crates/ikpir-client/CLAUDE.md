# CLAUDE.md — ikpir-client crate

## 1. Crate purpose

Client-side IKPIR: holds `CuckooParams` and per-segment `B::ClientState`
plus an epoch counter. Translates user-level `(key, query)` operations
into wire-level Index-PIR query/response bundles defined in
[`ikpir-common`](../ikpir-common/CLAUDE.md). The client **never** owns a
`CuckooKVStore`; its only persistent material from the server is the
setup bundle.

## 2. File map

| File | Role |
|---|---|
| `src/lib.rs` | Declares `mod client; mod error; mod pending;` and re-exports `IkpirClient`, `DeltaApplyOutcome`, `IkpirClientError`, plus the shared protocol surface (`IndexPirBackend`, `ClientUpdateMode`, `ResponseRewind`, `FrodoConfig`, `SimpleConfig`, wire bundles, `IkpirError`) from `ikpir-common` |
| `src/client.rs` | `IkpirClient<B>` generic + its public methods, in two update modes (hint-patch and response-rewind — see §3, §7) |
| `src/error.rs` | `IkpirClientError` enum (6 protocol variants + `Server(IkpirError)` forward; `IkpirError` is re-exported from `ikpir-common`) |
| `src/pending.rs` | `PendingDelta` — the rewind mode's rolling per-segment `ΔD` accumulator (crate-private) |

> Production code in this crate depends only on `ikpir-common` and
> `segmented-cuckoo`. `ikpir-server` is carried as a `[dev-dependency]`
> for `tests/client_e2e.rs`, the five benches, and the quick-start doctest.

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
  (sparse hint patch).

- **Parallel per-segment queries** — `build_query` emits one `B::Query`
  per segment (j-th query targets row `indices[j] % segment_size` in
  segment j). The server processes each segment independently in `answer`.

- **`fp` re-derivation in `decode`** — `decode(key, resp)` re-runs
  `candidate_buckets(key)` to recover `fp` deterministically. No query
  IDs, no stashed state, no extra wire field. There is no privacy reason
  to hide `fp` from the client itself.

- **Dual-path recovery** — `apply_delta` for the steady state (strict
  monotone epoch+1 patch), `reset_from` after `full_rebuild` or after a
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

  No bench reports client-bootstrap cost, so all five build their client
  with `from_setup_parallel`. The reference path stays the default so a
  future bootstrap-cost measurement has something honest to call.

- **Selectable hint-patch realization** — `apply_delta` realizes the
  patch at the client's `hint_patch_mode` (`set_hint_patch_mode`;
  default `HintPatchMode::EntryLevel`, the iSimplePIR per-cell patch;
  `RowLevel` is the SimplePIR dense per-row baseline the mutation
  benches compare against). Purely local: any mode combination between
  server and client yields bit-identical state from the same delta
  stream. The preference is client-side state, so it survives
  `reset_from`.

- **Selectable update strategy (rewind vs hint-patch)** — the
  `ClientUpdateMode` (`set_update_mode`; **default `Rewind`**) selects how
  the client tracks mutations. Hint-patch (`apply_delta` patches `H`;
  `decode` reads it directly) is the classical path; rewind pins the
  bootstrap hint `H₀`, accumulates the published `ΔD` (`accumulate_delta`,
  `Θ(τ·ω)` — a factor-`n` cheaper maintenance), corrects a head-answered
  response back to `H₀`'s epoch on decode (`decode_rewind`), and reclaims
  the staleness-growing correction by folding `ΔD` into the hint
  (`collect_garbage`). Both modes return the same decoded value (pinned by
  `tests/rewind_equivalence.rs`); the mode is a local choice preserved
  across `reset_from`, no Cargo feature (runtime enum, backend
  monomorphised on `B`). The correction is per-backend via the
  `ResponseRewind` trait (`ikpir-common`). Full mechanism and correctness:
  `docs/rewind-client-mode.md`.

## 4. Epoch state machine

```
           apply_delta(delta.epoch == self.epoch + 1)
               ┌──────────────────────────────────────┐
               │                                      ▼
 [from_setup] epoch=E  ──────────────────────▶  epoch=E+1  ···
               │
               │  apply_delta(epoch ≤ self.epoch)   → StaleDelta
               │  apply_delta(epoch > self.epoch+1) → FutureDelta
               │                                        ↓ caller must:
               └──────────────────────────────▶  reset_from(new_bundle)
```

- `apply_delta` is strict-monotone: only `delta.epoch == self.epoch + 1`.
- `decode` requires `resp.epoch == self.epoch`; mismatch → `EpochMismatch`.

## 5. Failure-mode table

| Variant | Source | Meaning |
|---|---|---|
| `StaleDelta` | `apply_delta` / `accumulate_delta` | `delta.epoch ≤ self.epoch` |
| `FutureDelta` | `apply_delta` / `accumulate_delta` | `delta.epoch > self.epoch + 1` (gap) |
| `EpochMismatch` | `decode` / `decode_rewind` | server moved between query and answer |
| `MalformedBundle` | `apply_delta` / `accumulate_delta` / `decode` / `decode_rewind` | params mismatch, or wrong segment count / row width |
| `WrongUpdateMode` | mode-gated methods | hint-patch method in rewind mode (or vice versa) — switch entry point or mode |
| `CellOutOfRange` | `decode_rewind` | a corrected cell escaped `[0, 2^plaintext_bits)` — corrupt/inconsistent delta or response, never a wrong value |
| `Server(IkpirError)` | forward | for synchronous in-process composition |

## 6. Entry points and test taxonomy

| Task | Where to look |
|---|---|
| Build a fresh client | `client.rs::IkpirClient::from_setup` |
| Bootstrap a client fast (untimed preamble) | `client.rs::IkpirClient::{from_setup_parallel, reset_from_parallel}` — identical client, all cores; contract in `ikpir-common::ParallelSetupBackend` |
| Issue a query | `client.rs::IkpirClient::build_query` |
| Decode a response (hint-patch) | `client.rs::IkpirClient::decode` |
| Decode a response (rewind) | `client.rs::IkpirClient::decode_rewind` |
| Apply an incremental delta (hint-patch) | `client.rs::IkpirClient::apply_delta` |
| Accumulate a delta (rewind) | `client.rs::IkpirClient::accumulate_delta` |
| Reclaim rewind staleness | `client.rs::IkpirClient::collect_garbage` |
| Update-strategy knob | `client.rs::IkpirClient::{update_mode, set_update_mode}` + `ikpir-common::ClientUpdateMode` (default `Rewind`) |
| Hint-patch realization knob | `client.rs::IkpirClient::{hint_patch_mode, set_hint_patch_mode}` + `ikpir-common::HintPatchMode` |
| Recover from a gap | `client.rs::IkpirClient::reset_from` |
| Debug a fingerprint mismatch | `client.rs::IkpirClient::decode` — check `candidate_buckets` + `unpack_slot_cells` |
| Integration tests | `tests/client_e2e.rs` + `tests/simple_client_e2e.rs` (mirror of `client_e2e.rs` for `SimplePirBackend`); `tests/replay_equivalence.rs` — the mutation benches' `reset_for_replay` harness measures what a fresh setup would (both backends, arities 2/3/4, plus a stale-hints negative control); `tests/rewind_equivalence.rs` — rewind == hint-patch == fresh decode (both backends × arities 2/3/4), GC-then-query, post-pin insert, mode/epoch guards |
| Benches | `benches/client_query.rs`, `benches/client_decode.rs`, `benches/client_mutation.rs`, `benches/headtohead_query.rs`, `benches/headtohead_decode.rs`. All accept `--backend frodo\|simple`; run via `../../scripts/bench.sh <name>` |
| Backend enum (bench CLI) | `benches/helpers.rs::Backend` + `backend_default_lwe_dim` — duplicated in `ikpir-server/benches/helpers.rs` |

### Bench layer (under `benches/`)

Five focused benches covering classical and incremental client criteria for the paper:

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `client_query` | `TableFull` | `build_query` rate (queries/sec, criterion, warm-bc) | `ikpir_client_query.csv` |
| `client_decode` | `TableFull` | `decode` rate (queries/sec, criterion, warm-bc) | `ikpir_client_decode.csv` |
| `client_mutation` | `--load-factor` (0.90) | `apply_delta` throughput per (kind, patch mode) pair (insert/update/delete × entry/row), wall-clock, empty queue (isolates hint-patch cost); one setup per config, deltas collected from a server rewound per kind with `reset_for_replay` | `ikpir_client_mutation.csv` |
| `headtohead_query` | fixed `--num-keys` | `build_query` rate at a fixed keyword count (fair comparison vs ChalametPIR / Hao 2025); mirrors `client_query` + `num_keys`/`db_size` columns | `ikpir_headtohead_client_query.csv` |
| `headtohead_decode` | fixed `--num-keys` | `decode` rate at a fixed keyword count; mirrors `client_decode` + `num_keys`/`db_size` columns, with the once-per-config `verify_decode` sanity check | `ikpir_headtohead_client_decode.csv` |

`client_query` and `client_decode` use **warm-bc** mode (precompute the
prepared-query queue + decode material before the timed loop), so the
timed call hits the cheap amortised path.

`client_mutation` runs the
client in **empty-queue** mode (no `precompute_queries` /
`precompute_decodes`). Each `apply_delta` then patches only the hint
`H` — the queue-iteration inside `client_patch_state` is a no-op when
the queue is empty — so the timing reports the "compute new hint" cost
in isolation, without warm-bc queue-maintenance overhead mixed in.

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
  `HintPatchMode` realization the timed `apply_delta` loop used. Deltas
  are collected once per kind (identical under either mode) and replayed
  per mode — from **one** server per config, rewound before each kind with
  `IkpirServer::reset_for_replay` (fresh store from the snapshot cells,
  clone of the epoch-0 hints), whose epoch-0 `setup()` bundle also
  bootstraps every timed client. `tests/replay_equivalence.rs` pins that a
  replay yields the same deltas as a fresh setup. `scripts/bench.sh`
  passes `entry,row` by default.
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
  because apply_delta advances the client epoch with each call; criterion's
  cycling pattern is not meaningful when state changes between calls.

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
