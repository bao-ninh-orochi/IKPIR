# CLAUDE.md — ikpir-server crate

## 1. Crate purpose

Server-side IKPIR: wraps a `CuckooKVStore` in per-segment Index-PIR
sub-databases and exposes the full server protocol — setup, answer, insert,
update, delete, and full_rebuild. Mutation-log-driven incremental hint
patching keeps the client's preprocessing state in sync after each mutation
without a full rebuild.

## 2. File map

| File | Role |
|---|---|
| `src/lib.rs` | Re-exports (incl. `FrodoConfig`) + `IkpirError` enum (5 variants) |
| `src/server.rs` | `IkpirServer<S, B>` generic + 9 public methods + `Segmented{2,3,4}aryIkpirServer` type aliases |
| `src/wire.rs` | `ServerSetupBundle / PirQueryBundle / PirResponseBundle / HintDeltaBundle` |
| `src/backend/mod.rs` | `IndexPirBackend` trait (6 associated types incl. `Config` + 5 methods) and `IncrementalPirBackend` (2 extra methods) |
| `src/backend/frodo/params.rs` | `FrodoParams` (per-segment runtime values) + `FrodoConfig` (user-facing tunable knobs, default `lwe_dim = 1774`) |
| `src/hint_patch.rs` | `fold_mutations_into_row_deltas` — `SlotMutation` list → per-segment sparse row deltas |

## 3. Key design decisions (the WHY)

- **Per-segment partition** — an arity-k SCF splits into k independent
  Index-PIR sub-databases. A query for key `k` targets row
  `indices[j] % segment_size` in segment j. This is why k PIR queries
  (one per segment) suffice to recover the slot.

- **Generics over `<S: IndexScheme + SchemeMeta, B: IndexPirBackend>`**
  — avoids dynamic dispatch on the answer/decode hot path. A monomorphised
  `B::server_answer` call has no vtable overhead.

- **Backend tunables via `B::Config` associated type** — `IkpirServer::new`
  takes `(store, backend_config)` and persists the config so every
  `full_rebuild` re-emits hints with identical dimensions. For FrodoPIR,
  `FrodoConfig { lwe_dim }` controls the LWE dimension; the previous
  hardcoded `DEFAULT_LWE_DIM` constant became `FrodoConfig::default()`.
  This is the extension seam SimplePIR will plug into.

- **Mutation-log-driven incremental hint** — after each `insert` /
  `update` / `delete`, `drain_mutations` produces `SlotMutation` records.
  `fold_mutations_into_row_deltas` converts them into sparse cell deltas
  per segment; the backend applies them with `server_patch_hint` /
  `client_patch_state`. No full-matrix recompute; bandwidth scales with
  the mutation footprint, not DB size.

- **Sync API; no async, no `Arc`** — all calls are synchronous and
  single-threaded. Concurrency wrapping is the caller's responsibility.

- **`FrodoPirBackend` is the sole shipped backend** — LWE-based, post-quantum, with incremental hint patching. SimplePIR is a future track.

- **Sparse row-delta encoding** — `HintDeltaBundle` carries
  `Vec<(row, Vec<(cell_offset, Δ)>)>` per segment. Wire cost is
  proportional to touched cells, not total DB size.

- **`IndexPirBackend` vs `IncrementalPirBackend` split** — a backend
  without efficient incremental patching can still be used for
  `full_rebuild`-only deployments by implementing only the base trait.

- **Threat model** — Index-PIR hides the queried row within each segment.
  The SCF candidate-bucket set is public and deterministic; an observer
  who sees many queries learns which SCF buckets were touched, not the
  slot contents or fingerprint value. Constant-time decode is out of scope.

## 4. Protocol invariants

- Mutation log is always enabled after `IkpirServer::new`; disabling it
  breaks incremental correctness.
- **Strict-monotone epoch** — every `commit_mutations` call (i.e. every
  `insert` / `update` / `delete`) and every `full_rebuild` increments
  `self.epoch`. Epoch never decrements.
- `hints[j]` stays in lock-step with
  `store.as_cells()[j*seg_cells..(j+1)*seg_cells]` after every mutation.
  Verified by the proptest in `tests/incremental_correctness.rs`.
- `answer()` rejects queries with `epoch ≠ self.epoch` with
  `IkpirError::StaleEpoch` — never silently re-routes.

## 5. Failure-mode table

| Variant | Source | Meaning |
|---|---|---|
| `StaleEpoch` | `answer` | client query against an older server epoch |
| `MalformedQuery` | `answer` | wrong number of segments in `q.queries` |
| `TableFull` | `insert` | SCF cuckoo kicks exhausted |
| `NotFound` | `update` / `delete` | key absent |
| `InvalidInput` | `insert` / `update` | bad value width |

## 6. Entry points and backend-author checklist

| Task | Where to look |
|---|---|
| Setup + answer flow | `server.rs::IkpirServer::{new, setup, answer}` |
| Mutation + incremental hint | `server.rs::commit_mutations` → `hint_patch.rs::fold_mutations_into_row_deltas` |
| Backend trait contract | `backend/mod.rs::IndexPirBackend` + `IncrementalPirBackend` |
| FrodoPIR config knobs | `backend/frodo/params.rs::FrodoConfig` (`lwe_dim`) |
| Integration tests | `tests/setup_answer.rs`, `tests/incremental_correctness.rs` |
| Benches | `benches/setup_throughput.rs`, `benches/answer_throughput.rs`, `benches/incremental_vs_rebuild.rs` |

**Backend-author checklist** — a new `IndexPirBackend` impl must:

1. Define `Config: Clone + Default` holding the backend's tunable knobs
   (e.g. `FrodoConfig { lwe_dim }` or future `SimplePirConfig`).
2. `server_setup(config, db, n_rows, row_width, plaintext_bits)` returns a
   `(ServerParams, Hint)` from the DB slice and the supplied config.
3. `client_setup` returns `ClientState` from `(ServerParams, Hint)`.
4. `client_query(state, row)` + `server_answer(params, db, n_rows, row_width, query)` +
   `client_decode(state, response)` must satisfy:
   `client_decode(server_answer(client_query(state, row))) == db[row*row_width..(row+1)*row_width]`.
5. If implementing `IncrementalPirBackend`: `server_patch_hint` and
   `client_patch_state` must keep `Hint` and `ClientState` consistent
   with the updated DB for all future queries.

**Per-segment data flow:**

```
                     ┌──────── arity-2 SCF ────────┐
key  ──candidate_buckets──▶  indices = [b0, b1]
                               │           │
                               ▼           ▼
                       segment 0      segment 1
                      (rows 0..N)    (rows 0..N)
                               │           │
client.build_query             │           │
     ─per-segment query──▶  Q[0]         Q[1]
                               │           │
server.answer                  ▼           ▼
     ─per-segment ans ───▶  R[0]         R[1]
                               │           │
client.decode                  ▼           ▼
     slot scan + fp match  →  row[s0]?  row[s1]?  → fp match → value
```
