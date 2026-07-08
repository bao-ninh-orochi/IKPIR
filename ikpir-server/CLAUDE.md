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
| `src/lib.rs` | Declares `mod hint_patch; mod server;` and re-exports `IkpirServer`, the `Segmented{2,3,4}aryIkpirServer` aliases, and the full shared protocol surface (`IndexPirBackend`, `FrodoConfig`, `SimpleConfig`, wire bundles, `IkpirError`) from `ikpir-common` |
| `src/server.rs` | `IkpirServer<S, B>` generic + 9 public methods + `Segmented{2,3,4}aryIkpirServer` type aliases |
| `src/hint_patch.rs` | `fold_mutations_into_row_deltas` — `SlotMutation` list → per-segment sparse row deltas (consumes `ikpir_common::SegmentRowDeltas`) |
| `tests/frodo_compose.rs` | Smoke test for the `FrodoPirBackend` × `IkpirServer` composition |
| `tests/simple_compose.rs` | Smoke test for the `SimplePirBackend` × `IkpirServer` composition (mirror of `frodo_compose.rs`) |
| `tests/simple_setup_answer.rs` | SimplePIR mirror of `setup_answer.rs` — setup/answer/full_rebuild over arities 2/3/4 |
| `tests/simple_incremental_correctness.rs` | SimplePIR mirror of `incremental_correctness.rs` — mutation log + delta + warm-queue stress |

> **Trait family, wire bundles, both shipped backends (FrodoPIR +
> SimplePIR), and `IkpirError` live in
> [`ikpir-common`](../ikpir-common/CLAUDE.md).** This crate re-exports
> them so existing call sites (`use ikpir_server::IndexPirBackend`,
> `use ikpir_server::FrodoConfig`, `use ikpir_server::SimplePirBackend`,
> `use ikpir_server::ServerSetupBundle`, `use ikpir_server::IkpirError`,
> ...) keep resolving unchanged.

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
  `FrodoConfig { lwe_dim }` controls the LWE dimension (default 1566).
  For SimplePIR, `SimpleConfig { lwe_dim, sigma }` controls both knobs
  (default `{1275, 6.4}`). Both `lwe_dim` defaults target 128-bit
  security, estimated via the lattice estimator under the ADPS16 cost
  model. Both backends slot into this seam without any change to
  `IkpirServer` itself.

- **Mutation-log-driven incremental hint** — after each `insert` /
  `update` / `delete`, `drain_mutations` produces `SlotMutation` records.
  `fold_mutations_into_row_deltas` converts them into sparse cell deltas
  per segment; the backend applies them with `server_patch_hint` /
  `client_patch_state`. No full-matrix recompute; bandwidth scales with
  the mutation footprint, not DB size.

- **Selectable hint-patch realization** — `commit_mutations` realizes
  the patch at the granularity in `hint_patch_mode`
  (`set_hint_patch_mode`; default `HintPatchMode::EntryLevel`).
  Entry-level is the iSimplePIR per-cell patch (`Θ(n)` per touched
  cell); row-level is the SimplePIR dense per-row baseline (`Θ(n·ω)`
  per touched row) that the mutation benches compare against. The
  emitted `HintDeltaBundle` is byte-identical under either mode, and
  the server's mode never needs to match its clients'.

- **Droppable `HintMaterial`** — the per-segment LWE matrix `A` (or any
  analogous backend-local material) lives in `B::HintMaterial`, **not**
  in `ServerParams`. `IkpirServer` carries a
  `Vec<Option<B::HintMaterial>>` alongside `Vec<B::ServerParams>`.
  Read-only deployments and benches that finish setup before sampling
  queries can call [`IkpirServer::drop_hint_material`] to free `A`;
  `server.answer` does not touch it, so the read path is unaffected.
  The next `commit_mutations` call silently re-expands the affected
  segments from the seed inside `ServerParams` via
  `B::expand_hint_material`. Callers observe nothing different other
  than a one-time first-mutation re-expansion cost.

- **Sync API; no async, no `Arc`** — all calls are synchronous and
  single-threaded. Concurrency wrapping is the caller's responsibility.

- **Two shipped backends** — both LWE-based, post-quantum, with full incremental hint patching:
  `FrodoPirBackend` (ternary errors, tall-skinny `n_rows × row_width`
  matrix, default `lwe_dim = 1566`) and `SimplePirBackend`
  (discrete-Gaussian errors with σ = 6.4, square-ish `√N × √N` internal
  reshape, default `lwe_dim = 1275`). Both `lwe_dim` defaults target
  128-bit security, estimated via the lattice estimator under the
  ADPS16 cost model. Selectable per-bench via the `--backend
  frodo|simple` CLI flag (default `frodo`).

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
| Hint-patch realization knob | `server.rs::IkpirServer::{hint_patch_mode, set_hint_patch_mode}` + `ikpir-common::HintPatchMode` |
| Backend trait contract | `ikpir-common/src/backend/mod.rs::IndexPirBackend` + `IncrementalPirBackend` + `PrecomputingPirBackend` + `BackendWireSize` |
| FrodoPIR config knobs | `ikpir-common/src/backend/frodo/params.rs::FrodoConfig` (`lwe_dim`) |
| FrodoPIR backend implementation | `ikpir-common/src/backend/frodo/backend.rs` |
| SimplePIR config knobs | `ikpir-common/src/backend/simple/params.rs::SimpleConfig` (`lwe_dim`, `sigma`) |
| SimplePIR backend implementation | `ikpir-common/src/backend/simple/backend.rs` |
| Wire-bundle definitions | `ikpir-common/src/wire.rs` |
| `IkpirError` variants | `ikpir-common/src/error.rs` |
| Integration tests | `tests/setup_answer.rs`, `tests/incremental_correctness.rs`, `tests/frodo_compose.rs` + SimplePIR mirrors (`simple_*.rs`) |
| Benches | `benches/server_setup.rs`, `benches/server_answer.rs`, `benches/server_mutation.rs`, `benches/headtohead_answer.rs`. All accept `--backend frodo\|simple`; run via `../scripts/bench.sh <name>` |
| Backend enum (bench CLI) | `benches/helpers.rs::Backend` + `backend_default_lwe_dim` — duplicated in `ikpir-client/benches/helpers.rs` |

**Backend-author checklist** — a new `IndexPirBackend` impl must:

1. Define `Config: Clone + Default` holding the backend's tunable knobs
   (e.g. `FrodoConfig { lwe_dim }`, `SimpleConfig { lwe_dim, sigma }`).
2. Define `HintMaterial: Default + Send + 'static` for server-local
   working state (e.g. the LWE matrix `A`). Backends with no analogous
   material set `type HintMaterial = ()`.
3. `server_setup(config, db, n_rows, row_width, plaintext_bits)` returns
   `(ServerParams, HintMaterial, Hint)` from the DB slice and the
   supplied config.
4. `expand_hint_material(params)` re-derives the `HintMaterial`
   deterministically from `ServerParams` (the server may drop and
   re-expand mid-protocol; the client materialises its own copy
   independently during `client_setup`).
5. `client_setup` returns `ClientState` from `(ServerParams, Hint)`,
   internally calling `expand_hint_material(params)`.
6. `client_query(state, row)` + `server_answer(params, db, n_rows, row_width, query)` +
   `client_decode(state, response)` must satisfy:
   `client_decode(server_answer(client_query(state, row))) == db[row*row_width..(row+1)*row_width]`.
   `server_answer` is permitted **not** to read the `HintMaterial` — this
   is what makes read-only `drop_hint_material` deployments work.
7. If implementing `IncrementalPirBackend`: `server_patch_hint(params,
   material, hint, row_deltas, mode)` and `client_patch_state(state,
   row_deltas, mode)` must keep `Hint` and `ClientState` consistent with
   the updated DB for all future queries — and must produce the same
   post-patch state under every `HintPatchMode` (the mode may only
   change the arithmetic schedule, never the result).

### Bench layer (under `benches/`)

Four focused benches covering classical and incremental server criteria for the paper:

| Bench | Populate to | What it measures | CSV |
|---|---|---|---|
| `server_setup` | `TableFull` | setup wall-clock (default trials=1, warmup=0): full `IkpirServer::new`, or `--estimate` = time one segment's `B::server_setup` × `arity`; setup_bundle_bytes, hint_bytes/seg | `ikpir_server_setup.csv` |
| `server_answer` | `TableFull` | PIR matvec answer rate (queries/sec, criterion); query_bytes, response_bytes | `ikpir_server_answer.csv` |
| `server_mutation` | `--load-factor` (0.90) | Per-(kind, patch-mode) throughput (insert/update/delete × entry/row), wall-clock batch; delta_bytes_total | `ikpir_server_mutation.csv` |
| `headtohead_answer` | fixed `--num-keys` | answer rate at a fixed keyword count (fair comparison vs ChalametPIR / Hao 2025); mirrors `server_answer` + `num_keys`/`db_size` columns | `ikpir_headtohead_server_answer.csv` |

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
- **Patch modes.** `server_mutation` accepts `--patch-mode entry|row`
  (comma-separated list, default `entry`) and emits one CSV row per
  `(patch mode, kind)` pair — the `patch_mode` column records which
  `HintPatchMode` realization the timed loop used. `scripts/bench.sh`
  passes `entry,row` by default so a single run produces both
  mutation-phase columns of the paper's asymptotic table.
- **Runner.** `scripts/bench.sh <bench> [flags]` maps the bench to its
  crate, auto-derives `--plaintext-bits` / `--lwe-dim`, and exports
  `IKPIR_RESULTS_DIR=results/ikpir-server` before `cargo bench`. Pass
  `--backend simple` to switch backends. There is no full-matrix sweep
  script; `scripts/smoke.sh` runs every PIR bench tiny for correctness.
- One invocation = one CSV row (append-mode writer via
  `csv_writer`, honoring `IKPIR_RESULTS_DIR`; default `results/`).
- Shared helpers in `benches/helpers.rs` (deliberately duplicated across
  crates — a common core is mirrored in `ikpir-client/benches/helpers.rs`;
  the client copy additionally carries `verify_decode`, used by the
  `headtohead_decode` sanity check):
    - `populate_until_full::<S>(…)` / `populate_to_load::<S>(load_factor, …)`
      — seed a `CuckooKVStore<S>` to `TableFull` or to a target load.
    - `print_preamble(name, knobs, store_state, geom)` — the standard
      `=== <bench> ===` / Parameters / KV store / Geometry banner.
    - `run_criterion_throughput_batched(label, elems, setup, routine)` —
      criterion wrapper used by `server_answer`.
- `server_mutation` uses wall-clock `Instant` batch timing (not criterion)
  because store state changes between mutations; criterion cycling is not
  meaningful here.

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
