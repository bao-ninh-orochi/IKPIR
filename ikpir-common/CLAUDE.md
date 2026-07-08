# CLAUDE.md — ikpir-common crate

## 1. Crate purpose

Shared building blocks of the IKPIR protocol that both server and client
consume but neither owns: the **pluggable Index-PIR backend trait
family**, the two shipped **LWE backends (FrodoPIR and SimplePIR)**, the
**wire-format bundles** exchanged across the server/client boundary, and
the shared **`IkpirError`** enum.

`ikpir-server` and `ikpir-client` both depend on this crate and re-export
the items they expose in their own public signatures, so existing call
sites (`use ikpir_server::IndexPirBackend`, `use ikpir_client::FrodoConfig`,
`use ikpir_server::ServerSetupBundle`, ...) keep resolving unchanged.

## 2. File map

| File | Role |
|---|---|
| `src/lib.rs` | Top-level re-exports of `backend`, `wire`, and `error` |
| `src/error.rs` | `IkpirError` enum (5 variants) — returned by server methods, wrapped by `IkpirClientError::Server` on the client side |
| `src/wire.rs` | Wire-format bundles `ServerSetupBundle / PirQueryBundle / PirResponseBundle / HintDeltaBundle`, the `SegmentRowDeltas` type alias, and per-bundle `wire_byte_size` helpers |
| `src/backend/mod.rs` | Trait family: `IndexPirBackend` (6 associated types incl. `Config` + 5 methods), `IncrementalPirBackend` (+2 methods, both taking a `HintPatchMode`), `PrecomputingPirBackend` (+4 methods), `BackendWireSize` (+4 methods); `HintPatchMode` enum (`RowLevel` / `EntryLevel`, default `EntryLevel`) |
| `src/backend/frodo/mod.rs` | Re-exports the FrodoPIR backend's public surface |
| `src/backend/frodo/params.rs` | `FrodoParams` (per-segment runtime values) + `FrodoConfig` (user-facing tunable knobs, default `lwe_dim = 1566`) |
| `src/backend/frodo/backend.rs` | `FrodoPirBackend` impl of all four traits + `FrodoServerParams / FrodoHint / FrodoClientState / FrodoQuery / FrodoResponse` |
| `src/backend/frodo/arith.rs` | `round_p_to_q` / `round_q_to_p` — plaintext ↔ ciphertext modulus conversion |
| `src/backend/frodo/sampler.rs` | `sample_a` (LWE public matrix) + `sample_ternary_into` (LWE secret / error sampling) |
| `src/backend/simple/mod.rs` | Re-exports the SimplePIR backend's public surface + math summary |
| `src/backend/simple/params.rs` | `SimpleParams` + `SimpleConfig` (user-facing knobs, default `lwe_dim = 1275`, `sigma = 6.4`) |
| `src/backend/simple/backend.rs` | `SimplePirBackend` impl of all four traits + `SimpleServerParams / SimpleHint / SimpleClientState / SimpleQuery / SimpleResponse` + internal `reshape_dims` / `translate` helpers |
| `src/backend/simple/arith.rs` | Δ-scaling (duplicated from frodo per project rule) |
| `src/backend/simple/sampler.rs` | `sample_a` + `sample_uniform_zq_into` (secret) + `sample_discrete_gaussian_into` (Box–Muller error) |
| `src/pir_params.rs` | Operating-point selection: `frodo_max_plaintext_bits` / `simple_max_plaintext_bits` — the largest `plaintext_bits` each backend decodes correctly at `q = 2³²`, evaluated at the per-segment matrix shape (full derivation of both correctness bounds in the module docs) |
| `examples/max_plaintext_bits.rs` | Dependency-free CLI over `pir_params`, consumed by `scripts/lib.sh::backend_plaintext_bits` (sourced by `scripts/bench.sh`) |

## 3. Key design decisions (the WHY)

- **Sibling-crate placement** — backend traits, FrodoPIR, wire bundles,
  and the shared error live here rather than in `ikpir-server` so that
  `ikpir-client` is no longer a downstream consumer of the server crate.
  The two protocol crates now consume `ikpir-common` as siblings;
  `ikpir-client` depends on `ikpir-server` only via `[dev-dependencies]`
  for end-to-end tests, benches, and the doctest in its quick-start.

- **Re-export discipline** — `ikpir-server` and `ikpir-client` both
  re-export the common items they expose in their own signatures
  (`IndexPirBackend`, `FrodoConfig`, all four bundle types, `IkpirError`).
  Every shared item resolves under all three crate paths; callers pick
  whichever is most ergonomic for their import context.

- **Three-layer trait hierarchy** — `IndexPirBackend` is mandatory and
  defines the basic `setup/query/answer/decode` triple. `IncrementalPirBackend`
  is optional and adds `server_patch_hint / client_patch_state` for
  full-rebuild-free DB mutations. `PrecomputingPirBackend` is optional
  and adds Phase B / Phase C precomputation for amortising per-query
  LWE work (FrodoPIR Fig. 1). `BackendWireSize` is optional and reports
  per-type byte sizes for wire-size benches. A backend that doesn't
  implement an optional extension is simply unavailable on the
  corresponding code path.

- **Two hint-patch realizations, one wire format** — both
  `IncrementalPirBackend` methods take a `HintPatchMode` selecting the
  realization of the patch: `RowLevel` (the SimplePIR row-granular
  patch — densify each touched row's edits and apply a dense rank-one
  update, `Θ(n·ω)` per touched row) or `EntryLevel` (the iSimplePIR
  sharpening — patch only touched columns, `Θ(n)` per touched cell;
  the default). Either mode leaves the hint equal to `A·D mod 2³²` and
  consumes the same `HintDeltaBundle`, so the mode is a purely local
  compute choice: server and client may run different modes and never
  diverge. The mutation benches sweep both to isolate the granularity
  cost — the two mutation-phase columns of the paper's asymptotic
  table. `patch_slot_c` (Phase-C maintenance) is deliberately
  mode-independent: it is inherently sparse and identical either way.

- **No I/O, no serialisation in the wire types** — bundles are plain
  data crossing process boundaries by value within tests and examples.
  Production deployments layer their own serialiser on top;
  `wire_byte_size()` reports the *minimum* on-wire footprint under
  fixed-width little-endian encoding so configs can be compared without
  committing to a specific format.

- **`HintDeltaBundle::new` is `#[doc(hidden)] pub`** — the constructor
  must be reachable from `ikpir-server` (sibling crate) but is hidden
  from the public API. It skips the invariant checks that
  `IkpirClient::apply_delta` then trusts (epoch monotonicity,
  `per_segment_row_deltas.len() == arity`); only
  `IkpirServer::commit_mutations` is permitted to call it.

- **`SegmentRowDeltas` is `pub`** (was `pub(crate)` before the split) —
  the type appears in the public `HintDeltaBundle` field, and
  `ikpir-server::hint_patch::fold_mutations_into_row_deltas` (now in a
  sibling crate) must produce values of this type.

- **Threat model unchanged** — Index-PIR hides the queried row within
  each segment. The SCF candidate-bucket set is public and
  deterministic; an observer who sees many queries learns which SCF
  buckets were touched, not the slot contents or fingerprint value.
  On side channels: the LWE matvecs avoid data-dependent shortcuts on
  secret values (skips keyed only on the public matrix `A` are fine),
  and `IkpirClient::decode` scans fingerprints with a branchless
  compare as best-effort hardening — but a full constant-time-audited
  decode path is out of scope for this prototype.

## 4. Trait family

```text
IndexPirBackend (mandatory)
│   ServerParams / HintMaterial / Hint / ClientState / Query / Response / Config
│   server_setup(config, db, n_rows, row_width, plaintext_bits)
│       -> (ServerParams, HintMaterial, Hint)
│   expand_hint_material(params) -> HintMaterial
│   client_setup(params, hint) -> ClientState
│   client_query(state, row) -> Query
│   server_answer(params, db, n_rows, row_width, query) -> Response
│   client_decode(state, response) -> Vec<u32>
│
├── IncrementalPirBackend
│   server_patch_hint(params, material, hint, row_deltas, mode)
│   client_patch_state(state, row_deltas, mode)
│       mode: HintPatchMode = RowLevel | EntryLevel (default EntryLevel)
│
├── PrecomputingPirBackend
│   client_precompute_queries(state, count)        — Phase B
│   client_precompute_decodes(state)               — Phase C
│   prepared_slot_count(state) -> usize
│   in_flight_slot_count(state) -> usize
│
└── BackendWireSize
    query_byte_size(q) / response_byte_size(r) / hint_byte_size(h) / server_params_byte_size(p)
```

`HintMaterial` is server-local working state (e.g. the LWE public matrix
`A` expanded from a 16-byte seed inside `ServerParams`). It is **not**
shipped on the wire. `IkpirServer` holds a `Vec<Option<B::HintMaterial>>`
per segment and can free it via
[`drop_hint_material`](../ikpir-server/CLAUDE.md#section-3); the next
mutation transparently re-expands the affected segments via
`expand_hint_material`. The client always re-expands its own copy from
the seed during `client_setup` (the bundle does not carry `A`). The
determinism contract on `expand_hint_material` is load-bearing — the
server and client both rely on the same seed reproducing the same `A`
bit-for-bit.

Both `FrodoPirBackend` and `SimplePirBackend` implement all four traits.
They are drop-in alternatives at the `B: IndexPirBackend` type parameter
on `IkpirServer<S, B>` / `IkpirClient<B>`. The per-bench `--backend
frodo|simple` flag (default `frodo`) selects between them at runtime;
the `B::Config` associated type (`FrodoConfig` vs `SimpleConfig`)
carries the backend-specific tunables (`lwe_dim`, plus `sigma` for
SimplePIR).

## 5. Wire bundle taxonomy

| Bundle | Direction | Carries |
|---|---|---|
| `ServerSetupBundle<B>` | server → client | preprocessing material (`Hint`, `ServerParams`, `CuckooParams`, epoch). **Not** `HintMaterial` — the client re-expands `A` from the seed inside `ServerParams` via `expand_hint_material`. |
| `PirQueryBundle<B>` | client → server | one `B::Query` per segment + epoch |
| `PirResponseBundle<B>` | server → client | one `B::Response` per segment + epoch |
| `HintDeltaBundle<B>` | server → client | sparse per-segment row deltas + epoch (after one mutation) |

The exclusion of `HintMaterial` from `ServerSetupBundle` is what makes
the bundle small at paper scale. `setup_bundle_bytes` in the bench
preambles drops by ~`num_buckets × lwe_dim × 4` bytes (one full `A` per
setup) after this refactor.

`SegmentRowDeltas = Vec<(u32, Vec<(u16, i64)>)>` — per-segment list of
`(row_in_segment, [(cell_offset, signed_delta), …])`. Backend-agnostic:
contents are plain integers, no backend ciphertext.

## 6. Failure-mode table (`IkpirError`)

| Variant | Source (server method) | Meaning |
|---|---|---|
| `StaleEpoch` | `answer` | client query against an older server epoch |
| `MalformedQuery` | `answer` | wrong number of segments in `q.queries` |
| `TableFull` | `insert` | SCF cuckoo kicks exhausted |
| `NotFound` | `update` / `delete` | key absent |
| `InvalidInput` | `insert` / `update` | bad value width |

`ikpir-client` re-exports `IkpirError` and wraps it in
`IkpirClientError::Server(IkpirError)` for synchronous in-process
composition.

## 7. Entry points

| Task | Where to look |
|---|---|
| Backend trait contract | `backend/mod.rs::IndexPirBackend` + the three extension traits |
| Max safe `plaintext_bits` for a geometry | `pir_params.rs` (library) / `examples/max_plaintext_bits.rs` (CLI); empirical validation via the `#[ignore]`d `noise_margin` tests in both `backend/*/backend.rs` |
| FrodoPIR config knobs | `backend/frodo/params.rs::FrodoConfig` (`lwe_dim`) |
| SimplePIR config knobs | `backend/simple/params.rs::SimpleConfig` (`lwe_dim`, `sigma`) |
| Implement a new backend | mirror `backend/frodo/backend.rs` (tall-skinny) or `backend/simple/backend.rs` (square reshape); see backend-author checklist below |
| Wire-bundle layout | `wire.rs` module docs + each bundle's labelled-section block |
| Shared error variants | `error.rs::IkpirError` |
| Round-trip cell-modulus conversion | `backend/frodo/arith.rs` (also duplicated in `backend/simple/arith.rs`) |
| LWE sampling (FrodoPIR) | `backend/frodo/sampler.rs` (`sample_a`, `sample_ternary_into`) |
| LWE sampling (SimplePIR) | `backend/simple/sampler.rs` (`sample_a`, `sample_uniform_zq_into`, `sample_discrete_gaussian_into`) |
| Cross-crate integration tests | `ikpir-server/tests/frodo_compose.rs` + `simple_compose.rs` exercise each backend end-to-end against `IkpirServer` |

**Backend-author checklist** — a new `IndexPirBackend` impl must:

1. Define `Config: Clone + Default` holding the backend's tunable knobs
   (e.g. `FrodoConfig { lwe_dim }`, `SimpleConfig { lwe_dim, sigma }`).
2. Define `HintMaterial: Default + Send + 'static` for server-local
   working state (e.g. the LWE matrix `A`). Backends with no analogous
   material set `type HintMaterial = ()`. Don't derive `Clone` on a
   bulky `HintMaterial` — every "extra" copy should be an explicit
   `expand_hint_material` call.
3. `server_setup(config, db, n_rows, row_width, plaintext_bits)` returns
   `(ServerParams, HintMaterial, Hint)` from the DB slice and the
   supplied config.
4. `expand_hint_material(params)` re-derives the `HintMaterial` from
   `ServerParams`. **Determinism contract**: same seed/state inside
   `params` must yield bit-identical output, since the server may drop
   and re-expand mid-protocol and the client materialises its own copy
   independently from the wire seed.
5. `client_setup` returns `ClientState` from `(ServerParams, Hint)`,
   internally calling `expand_hint_material(params)` and stashing both
   `params` and the materialised state.
6. The triple `(client_query, server_answer, client_decode)` must satisfy:
   `client_decode(server_answer(client_query(state, row))) == db[row*row_width..(row+1)*row_width]`.
7. If implementing `IncrementalPirBackend`: `server_patch_hint(params,
   material, hint, row_deltas, mode)` and `client_patch_state(state,
   row_deltas, mode)` must keep `Hint` and `ClientState` consistent with
   the updated DB for **all** future queries. `client_patch_state` reads
   `HintMaterial` from the stashed `ClientState`. Every `HintPatchMode`
   must produce the same post-patch state — the mode may only change the
   arithmetic schedule (row-level dense pass vs entry-level per-cell
   pass), never the result.
8. If implementing `PrecomputingPirBackend`: prepared slots are consumed
   FIFO segment-locally; `client_patch_state` must also update
   already-prepared Phase-C material (see the trait's contract block).
9. If implementing `BackendWireSize`: return the *minimum* fixed-width
   little-endian byte size — no framing, no compression. **Do not
   include `HintMaterial`** in `server_params_byte_size`; it never
   travels on the wire.
