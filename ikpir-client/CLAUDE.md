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
| `src/lib.rs` | Declares `mod client; mod error;` and re-exports `IkpirClient`, `DeltaApplyOutcome`, `IkpirClientError`, plus the shared protocol surface (`IndexPirBackend`, `FrodoConfig`, `SimpleConfig`, wire bundles, `IkpirError`) from `ikpir-common` |
| `src/client.rs` | `IkpirClient<B>` generic + 7 public methods |
| `src/error.rs` | `IkpirClientError` enum (4 protocol variants + `Server(IkpirError)` forward; `IkpirError` is re-exported from `ikpir-common`) |

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

- **Selectable hint-patch realization** — `apply_delta` realizes the
  patch at the client's `hint_patch_mode` (`set_hint_patch_mode`;
  default `HintPatchMode::EntryLevel`, the iSimplePIR per-cell patch;
  `RowLevel` is the SimplePIR dense per-row baseline the mutation
  benches compare against). Purely local: any mode combination between
  server and client yields bit-identical state from the same delta
  stream. The preference is client-side state, so it survives
  `reset_from`.

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
| `StaleDelta` | `apply_delta` | `delta.epoch ≤ self.epoch` |
| `FutureDelta` | `apply_delta` | `delta.epoch > self.epoch + 1` (gap) |
| `EpochMismatch` | `decode` | server moved between query and answer |
| `MalformedBundle` | `apply_delta` / `decode` | wrong segment count or row width |
| `Server(IkpirError)` | forward | for synchronous in-process composition |

## 6. Entry points and test taxonomy

| Task | Where to look |
|---|---|
| Build a fresh client | `client.rs::IkpirClient::from_setup` |
| Issue a query | `client.rs::IkpirClient::build_query` |
| Decode a response | `client.rs::IkpirClient::decode` |
| Apply an incremental delta | `client.rs::IkpirClient::apply_delta` |
| Hint-patch realization knob | `client.rs::IkpirClient::{hint_patch_mode, set_hint_patch_mode}` + `ikpir-common::HintPatchMode` |
| Recover from a gap | `client.rs::IkpirClient::reset_from` |
| Debug a fingerprint mismatch | `client.rs::IkpirClient::decode` — check `candidate_buckets` + `unpack_slot_cells` |
| Integration tests | `tests/client_e2e.rs` + `tests/simple_client_e2e.rs` (mirror of `client_e2e.rs` for `SimplePirBackend`) |
| Benches | `benches/client_query.rs`, `benches/client_decode.rs`, `benches/client_mutation.rs`. All accept `--backend frodo\|simple` |
| Backend enum (bench CLI) | `benches/helpers.rs::Backend` + `backend_default_lwe_dim` — duplicated in `ikpir-server/benches/helpers.rs` |

### Bench layer (under `benches/`)

Three focused benches covering classical and incremental client criteria for the paper:

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `client_query` | `TableFull` | `build_query` rate (queries/sec, criterion, warm-bc) | `ikpir_client_query.csv` |
| `client_decode` | `TableFull` | `decode` rate (queries/sec, criterion, warm-bc) | `ikpir_client_decode.csv` |
| `client_mutation` | `--load-factor` (0.90) | `apply_delta` throughput per (kind, patch mode) pair (insert/update/delete × entry/row), wall-clock, empty queue (isolates hint-patch cost) | `ikpir_client_mutation.csv` |

`client_query` and `client_decode` use **warm-bc** mode (precompute the
prepared-query queue + decode material before the timed loop), so the
timed call hits the cheap amortised path.

`client_mutation` (and the fused `mutation_throughput` bench) runs the
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
  size). The orchestrator scripts override this per `(backend, m_label)`
  via `scripts/configs.sh::backend_plaintext_bits` so each sweep runs
  at the largest `pb` admitted by the backend's correctness bound at
  `q = 2^32`. The chosen value is written to every CSV row as the
  `plaintext_bits` column.
- **Patch modes.** `client_mutation` and the fused `mutation_throughput`
  bench accept `--patch-mode entry|row` (comma-separated list, default
  `entry`) and emit one CSV row per `(patch mode, kind)` pair; the
  `patch_mode` column records which `HintPatchMode` realization the
  timed `apply_delta` loop used. Deltas are collected once per kind
  (they are identical under either mode) and replayed per mode. The
  orchestrators forward `IKPIR_BENCH_PATCH_MODES` (default `entry,row`).
- One invocation = one CSV row (append-mode writer); the orchestrator
  is responsible for `rm`-ing the CSV before sweeping. The orchestrator
  also reads `IKPIR_BENCH_BACKENDS` (default `frodo`) and re-runs every
  bench once per backend in that comma-separated list.
- Shared helpers in `benches/helpers.rs` (deliberately duplicated across
  crates — a common core is mirrored in `ikpir-server/benches/helpers.rs`,
  but this copy additionally carries `verify_decode` and the fused-bench
  plumbing, which would create a dev-dep cycle on the server side):
    - `populate_until_full::<S>(…)` / `populate_to_load::<S>(load_factor, …)`
      — seed a `CuckooKVStore<S>` to `TableFull` or to a target load.
    - `print_preamble(name, knobs, store_state, geom)` — the standard
      `=== <bench> ===` / Parameters / KV store / Geometry banner.
    - `run_criterion_throughput_batched(label, elems, setup, routine)` —
      criterion wrapper used by `client_query` and `client_decode`.
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
