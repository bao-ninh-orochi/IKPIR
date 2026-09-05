# CLAUDE.md — ikpir-client crate

## 1. Crate purpose

Client-side IKPIR: holds `CuckooParams` and per-segment `B::ClientState`
plus an epoch counter. Translates user-level `(key, query)` operations
into wire-level Index-PIR query/response bundles defined in
[`ikpir-common`](../ikpir-common/CLAUDE.md). The client **never** owns a
`CuckooKVStore`; its only persistent material from the server is the
setup bundle. This crate ships **two parallel, first-class client
flows** over the same server-published `HintDeltaBundle` stream:
**client-hint-patch** (`HintPatchClient` — folds every delta into its
own hint immediately via `apply_delta`, `Θ(n·τ·ω)` per batch, decodes
directly with `decode(key, resp)`) and **client-rewind** (`RewindClient`,
alias `IkpirClient` — pins the bootstrap hint and accumulates the
published `ΔD` via `accumulate_delta`, `Θ(τ·ω)` per batch, corrects a
response at decode time with `decode(key, query, resp)`). Both are
always available, chosen at the type like the backend at `B`
(`docs/rewind-client-mode.md`).

## 2. File map

| File | Role |
|---|---|
| `src/lib.rs` | Declares `mod client_hint_patch; mod client_rewind; mod ct; mod error; mod outcome; mod pending;` and re-exports `HintPatchClient`, `RewindClient` (+ the `IkpirClient` alias of `RewindClient`), `DeltaApplyOutcome`, `IkpirClientError`, plus the shared protocol surface (`IndexPirBackend`, `ResponseRewind`, `FrodoConfig`, `SimpleConfig`, wire bundles, `IkpirError`) from `ikpir-common` |
| `src/client_hint_patch.rs` | `HintPatchClient<B>` generic + its public methods — the client-hint-patch flow: folds every delta into its own hint immediately (see §3, §7) |
| `src/client_rewind.rs` | `RewindClient<B>` generic + its public methods — the client-rewind flow: pins the bootstrap hint and accumulates `ΔD` (see §3, §7) |
| `src/outcome.rs` | `DeltaApplyOutcome` — shared result of `HintPatchClient::try_apply_delta_or_resync` / `RewindClient::try_accumulate_delta_or_resync` |
| `src/ct.rs` | `ct_eq_u64_mask` — shared branchless `u64` equality mask used by both flows' fingerprint scans |
| `src/error.rs` | `IkpirClientError` enum (5 protocol variants + `Server(IkpirError)` forward; `IkpirError` is re-exported from `ikpir-common`) |
| `src/pending.rs` | `PendingDelta` — the client-rewind flow's rolling per-segment `ΔD` accumulator (crate-private; rewind-only, `HintPatchClient` has no accumulator) |

> Production code in this crate depends only on `ikpir-common` and
> `segmented-cuckoo`. `ikpir-server` is carried as a `[dev-dependency]`
> for the crate's integration tests (§6), the six benches, and the two
> quick-start doctests (one per flow).

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
  (client-hint-patch's `apply_delta`; client-rewind's `collect_garbage`).

- **Parallel per-segment queries** — `build_query` emits one `B::Query`
  per segment (j-th query targets row `indices[j] % segment_size` in
  segment j). The server processes each segment independently in `answer`.

- **`fp` re-derivation in `decode`** — both flows' `decode` re-runs
  `candidate_buckets(key)` to recover `fp` deterministically. No query
  IDs, no stashed state, no extra wire field. There is no privacy reason
  to hide `fp` from the client itself.

- **Dual-path recovery** — the sync verb (`apply_delta` / `accumulate_delta`)
  for the steady state (strict monotone epoch+1 sync), `reset_from` after
  `full_rebuild` or after a `FutureDelta` gap that cannot be bridged
  incrementally. Same shape on both flows.

- **Two bootstrap implementations, one result** — `from_setup` /
  `reset_from` re-expand each segment's `A` **single-threaded**, which is
  the entire cost of bootstrapping a client (`Θ(arity · n_rows · lwe_dim)`
  ChaCha20 words — gigabytes at paper scale). `from_setup_parallel` /
  `reset_from_parallel` (available whenever `B: ParallelSetupBackend`,
  which both shipped backends are) do the same across all cores and yield
  an observationally identical client: same queries, same decodes, same
  patch behaviour, same epoch. Both pairs share one body via a
  `PerSegmentClientSetup<B>` fn pointer — one such pointer per flow
  (`HintPatchClient` and `RewindClient` each define their own).

  No bench reports client-bootstrap cost, so every bench builds its
  client with `from_setup_parallel`. The reference path stays the
  default so a future bootstrap-cost measurement has something honest
  to call.

- **client-hint-patch** — `HintPatchClient<B>` (`src/client_hint_patch.rs`)
  folds every delta into its own hint immediately: `apply_delta` computes
  `H ← H + Σ A[:,col]·δ`, `Θ(n·τ·ω)` per batch of `τ` mutations over row
  width `ω`, via a selectable `HintPatchMode` realization
  (`{hint_patch_mode, set_hint_patch_mode}`). `decode(key, resp)` then
  reads the patched hint directly — no query threading, no per-query
  correction. This is the flow the CANS 2026 camera-ready's client
  numbers were measured with.

- **client-rewind** — `RewindClient<B>` (`src/client_rewind.rs`, alias
  `IkpirClient<B>`) pins its bootstrap hint `H₀` at setup time and never
  patches it. `accumulate_delta` rolls the published `ΔD` forward
  (`Θ(τ·ω)` per batch — a factor-`n` cheaper client maintenance than
  patching the hint directly), `decode(key, query, resp)` corrects a
  head-answered response back to `H₀`'s epoch before decoding (via the
  per-backend `ResponseRewind` trait from `ikpir-common`), and
  `collect_garbage` reclaims the staleness-growing per-query correction
  by folding `ΔD` into the hint on demand (using
  `HintPatchMode::EntryLevel`, hardcoded — there is no client-selectable
  patch realization on this flow). This is the flow the extended (full)
  paper reports. Full mechanism and correctness:
  `docs/rewind-client-mode.md`.

- **Two types, not a mode enum** — the flow is chosen at the type
  parameter, the same way the backend is chosen at `B`, rather than a
  runtime `UpdateMode` field on one client struct: there is no way to
  call `apply_delta` on a `RewindClient` or `accumulate_delta` on a
  `HintPatchClient` (no runtime "wrong mode" error, because the compiler
  rejects it), and each flow's `decode` keeps its own natural signature
  (2-arg for client-hint-patch, 3-arg for client-rewind) instead of a
  lowest-common-denominator shape. The two never coexist on one
  instance. Parity — both flows decode identically and equal a fresh
  client at the head — is pinned by `tests/client_flow_parity.rs`.

## 4. Epoch state machine

The state machine is shared by both flows — only the sync verb's name
differs (`apply_delta` on `HintPatchClient`, `accumulate_delta` on
`RewindClient`):

```
        sync(delta.epoch == self.epoch + 1)
               ┌──────────────────────────────────────┐
               │                                      ▼
 [from_setup] epoch=E  ──────────────────────▶  epoch=E+1  ···
               │
               │  sync(epoch ≤ self.epoch)   → StaleDelta
               │  sync(epoch > self.epoch+1) → FutureDelta
               │                            ↓ caller must:
               └──────────────────────────────▶  reset_from(new_bundle)
```

- The sync verb is strict-monotone on both flows: only
  `delta.epoch == self.epoch + 1` is accepted.
- `decode` requires `resp.epoch == self.epoch` on both flows; mismatch →
  `EpochMismatch`.
- The pin differs between flows: `apply_delta` folds the delta into
  `HintPatchClient`'s hint directly, so it always tracks the head — there
  is no separate pin, `epoch` alone describes its state. `accumulate_delta`
  instead lets `RewindClient::pin_epoch` (where the pinned hint `H₀` sits)
  trail `epoch` by the accumulated `ΔD` span; it advances only on
  `collect_garbage` or `reset_from`.

## 5. Failure-mode table

| Variant | Source | Meaning |
|---|---|---|
| `StaleDelta` | `accumulate_delta` / `apply_delta` | `delta.epoch ≤ self.epoch` |
| `FutureDelta` | `accumulate_delta` / `apply_delta` | `delta.epoch > self.epoch + 1` (gap) |
| `EpochMismatch` | `decode` (either flow) | server moved between query and answer |
| `MalformedBundle` | `accumulate_delta` / `apply_delta` / `decode` (either flow) | params mismatch, or wrong segment count / row width |
| `CellOutOfRange` | `RewindClient::decode` only | a corrected cell escaped `[0, 2^plaintext_bits)` — corrupt/inconsistent delta or response, never a wrong value |
| `Server(IkpirError)` | forward | for synchronous in-process composition |

## 6. Entry points and test taxonomy

| Task | client-hint-patch (`HintPatchClient`) | client-rewind (`RewindClient`) |
|---|---|---|
| Build a fresh client | `client_hint_patch.rs::HintPatchClient::from_setup` | `client_rewind.rs::RewindClient::from_setup` |
| Bootstrap a client fast (untimed preamble) | `HintPatchClient::{from_setup_parallel, reset_from_parallel}` | `RewindClient::{from_setup_parallel, reset_from_parallel}` — identical client, all cores; contract in `ikpir-common::ParallelSetupBackend` |
| Issue a query | `HintPatchClient::build_query` | `RewindClient::build_query` |
| Decode a response | `HintPatchClient::decode(key, resp)` | `RewindClient::decode(key, query, resp)` — threads the query bundle; corrects for accumulated `ΔD` |
| Sync a delta | `HintPatchClient::apply_delta` — folds into the hint directly | `RewindClient::accumulate_delta` — rolls `ΔD` forward |
| Resync sugar | `HintPatchClient::try_apply_delta_or_resync` | `RewindClient::try_accumulate_delta_or_resync` |
| Reclaim staleness | — (hint always current, nothing to reclaim) | `RewindClient::collect_garbage` |
| Recover from a gap | `HintPatchClient::reset_from` | `RewindClient::reset_from` |
| Patch realization | `HintPatchClient::{hint_patch_mode, set_hint_patch_mode}` | — (`collect_garbage` hardcodes `HintPatchMode::EntryLevel`) |
| Debug a fingerprint mismatch | `HintPatchClient::decode` — check `candidate_buckets` + `unpack_slot_cells` | `RewindClient::decode` — same |

| Task | Where to look |
|---|---|
| Integration tests | `tests/client_hint_patch_e2e.rs` + `tests/client_hint_patch_simple_e2e.rs` — client-hint-patch, `FrodoPirBackend` / `SimplePirBackend`, arities 2/3/4; `tests/client_rewind_e2e.rs` + `tests/client_rewind_simple_e2e.rs` — client-rewind, same matrix; `tests/client_flow_parity.rs` — both flows decode identically and equal a fresh client at the head (fixed mixed traces × both backends × arities 2/3/4, GC-then-query, post-pin insert, plus a proptest over random insert/update/delete traces); `tests/replay_equivalence.rs` — the mutation benches' `reset_for_replay` harness measures what a fresh setup would (both backends, arities 2/3/4, plus a stale-hints negative control) |
| Benches | `benches/client_query.rs`, `benches/client_rewind_staleness.rs`, `benches/headtohead_query.rs` (flow-independent) plus a client-hint-patch / client-rewind pair each for decode (`client_hint_patch_decode.rs` / `client_rewind_decode.rs`), mutation (`client_hint_patch_mutation.rs` / `client_rewind_mutation.rs`), and head-to-head decode (`headtohead_hint_patch_decode.rs` / `headtohead_rewind_decode.rs`) — every pair a thin binary over a shared generic body (`benches/flow_decode_body.rs`, `benches/flow_headtohead_decode_body.rs`, `benches/flow_mutation_body.rs`). No Cargo feature needed — the flow is chosen at the type, monomorphised per binary. All accept `--backend frodo\|simple`; run via `../../scripts/bench.sh <name>` |
| Backend enum (bench CLI) | `benches/helpers.rs::Backend` + `backend_default_lwe_dim` — duplicated in `ikpir-server/benches/helpers.rs` |

### Bench layer (under `benches/`)

Nine focused benches covering classical and incremental client criteria for
the paper. The client flow is always a separate binary, never a runtime flag,
and **benchmark data of the two flows is always written to separate CSV
files and never merged**:

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `client_query` | `TableFull` | `build_query` rate (queries/sec, criterion, warm-bc); flow-independent — `build_query` is the same code in both flows | `ikpir_client_query.csv` |
| `client_hint_patch_decode` | `TableFull` | client-hint-patch's `HintPatchClient::decode(key, resp)` rate (queries/sec, criterion, warm-bc) — the flow the CANS 2026 camera-ready reports | `ikpir_client_hint_patch_decode.csv` |
| `client_rewind_decode` | `TableFull` | client-rewind's `RewindClient::decode(key, query, resp)` rate at empty `ΔD` (queries/sec, criterion, warm-bc) — the flow the extended (full) paper reports | `ikpir_client_rewind_decode.csv` |
| `client_hint_patch_mutation` | `--load-factor` (0.90) | per-batch **maintenance** throughput per (kind, `--patch-mode` entry\|row): times `HintPatchClient::apply_delta`; wall-clock, empty queue; one setup per config, deltas collected from a server rewound per kind with `reset_for_replay` | `ikpir_client_hint_patch_mutation.csv` |
| `client_rewind_mutation` | `--load-factor` (0.90) | per-batch **maintenance** throughput per kind (no `--patch-mode` — patch-mode-independent): times `RewindClient::accumulate_delta`; wall-clock, empty queue; `pending_cells` = final \|ΔD\| | `ikpir_client_rewind_mutation.csv` |
| `client_rewind_staleness` | `--load-factor` (0.90) | `RewindClient::decode` per-query latency vs staleness \|ΔD\| (one client accumulates over `--staleness-steps` × `--batch-size` updates, never GC'd, then a `collect_garbage` returns it to baseline); times only `decode` | `ikpir_client_rewind_staleness.csv` |
| `headtohead_query` | fixed `--num-keys` | `build_query` rate at a fixed keyword count (fair comparison vs ChalametPIR / Hao 2025); mirrors `client_query` + `num_keys`/`db_size` columns; flow-independent | `ikpir_headtohead_client_query.csv` |
| `headtohead_hint_patch_decode` | fixed `--num-keys` | client-hint-patch's `decode` rate at a fixed keyword count; mirrors `client_hint_patch_decode` + `num_keys`/`db_size` columns, with the once-per-config `verify_decode` sanity check | `ikpir_headtohead_client_hint_patch_decode.csv` |
| `headtohead_rewind_decode` | fixed `--num-keys` | client-rewind's `decode` rate at a fixed keyword count; mirrors `client_rewind_decode` + `num_keys`/`db_size` columns, with the once-per-config `verify_decode` sanity check | `ikpir_headtohead_client_rewind_decode.csv` |

`client_{hint_patch,rewind}_decode` use **warm-bc** mode (precompute the
prepared-query queue + decode material before the timed loop), so the
timed call hits the cheap amortised path. For client-hint-patch the timed
call is the real 2-arg `HintPatchClient::decode(key, resp)` — no query
threading, no response clone — even though the shared `ClientFlow::decode`
trait method used by the generic bench body takes a `query` parameter for
both flows uniformly; `HintPatchClient`'s implementation just ignores it
(a borrow, never cloned).

`client_hint_patch_mutation` / `client_rewind_mutation` run the
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
- **Flow dispatch.** The client-hint-patch / client-rewind pairs are thin
  binaries: `mod helpers; mod flow_{decode,headtohead_decode,mutation}_body;`
  plus a `dispatch_backend` that matches `--backend` and calls the shared
  body's generic `run_one::<S, B, C>` with `C = HintPatchClient<B>` or
  `RewindClient<B>` named explicitly per arm — the only place the flow's
  concrete type appears. Every other line (CLI struct, preamble, populate,
  setup, the criterion/wall-clock loop, CSV writing) lives once in the body
  file, generic over `C: helpers::ClientFlow<B>` (decode bodies) or
  `C: MutationFlow<B>` (the mutation body's small extension of it, in
  `flow_mutation_body.rs`) — never dynamic dispatch in the timed loop.
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
- **Patch modes.** `client_hint_patch_mutation` accepts `--patch-mode
  entry|row` (comma-separated list, default `entry`) and emits one CSV row
  per `(patch mode, kind)` pair; the `patch_mode` column records which
  `HintPatchMode` realization `HintPatchClient` used for its timed
  `apply_delta` loop. `client_rewind_mutation` has no `--patch-mode` flag —
  `RewindClient::accumulate_delta` is patch-mode-independent — and instead
  carries a `pending_cells` column (final \|ΔD\|). Deltas are collected once
  per kind (identical regardless of patch mode) and replayed per variant —
  from **one** server per config, rewound before each kind with
  `IkpirServer::reset_for_replay` (fresh store from the snapshot cells,
  clone of the epoch-0 hints), whose epoch-0 `setup()` bundle also
  bootstraps every timed client. `tests/replay_equivalence.rs` pins that a
  replay yields the same deltas as a fresh setup. `scripts/bench.sh`
  passes `entry,row` by default (only to `client_hint_patch_mutation` and
  `server_mutation`; forwarding it to `client_rewind_mutation` is a no-op
  with a warning).
- **Runner.** `scripts/bench.sh <bench> [flags]` maps the bench to its
  crate, auto-derives `--plaintext-bits` / `--lwe-dim`, and exports
  `IKPIR_RESULTS_DIR=results/ikpir-client` before `cargo bench`. One
  invocation = one CSV row (append-mode `csv_writer`, honoring
  `IKPIR_RESULTS_DIR`; default `results/`). Its geometry defaults are dev
  scale, not the paper's: the paper matrix lives in `scripts/lib.sh`
  (`PAPER_*`) and is swept by `scripts/table3.sh` (online, via
  `headtohead_{query,hint_patch_decode,rewind_decode}`) and `table4.sh`
  (mutation, via `client_{hint_patch,rewind}_mutation`) — both take
  `--flow client-hint-patch|client-rewind|all` to pick which flow's leg
  runs. `scripts/smoke.sh` runs every PIR bench tiny.
- Shared helpers in `benches/helpers.rs` (deliberately duplicated across
  crates — a common core is mirrored in `ikpir-server/benches/helpers.rs`,
  but this copy additionally carries `verify_decode` — flow-generic via
  `ClientFlow`, round-trips through both client and server — a dev-dep
  cycle on the server side — and backs the `headtohead_{hint_patch,rewind}_decode`
  sanity checks) plus the bench-local `ClientFlow<B>` trait: unifies
  `HintPatchClient<B>` and `RewindClient<B>` behind one interface
  (`FLOW`, `build_query`, `decode`, `sync_delta`, `precompute_{queries,decodes}`,
  `pending_cells`) so the shared body files stay generic instead of
  duplicating per-flow logic; not part of `ikpir-client`'s public API:
    - `populate_until_full::<S>(…)` / `populate_to_load::<S>(load_factor, …)`
      — seed a `CuckooKVStore<S>` to `TableFull` or to a target load.
    - `print_preamble(name, knobs, store_state, geom)` — the standard
      `=== <bench> ===` / Parameters / KV store / Geometry banner.
    - `configured_criterion()` — the `Criterion` pinned to the shared
      Table 3 contract (100 samples, 3 s warm-up, 5 s measurement), which
      `client_query`, `client_{hint_patch,rewind}_decode`, `headtohead_query`,
      and `headtohead_{hint_patch,rewind}_decode` drive directly through
      `iter_custom`.
- The mutation benches use wall-clock `Instant` batch timing (not criterion)
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
