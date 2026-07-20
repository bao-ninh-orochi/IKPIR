# ikpir-common

Shared primitives for [Incremental Keyword PIR](../../README.md). Holds the
Index-PIR backend trait family, the two shipped LWE backends (FrodoPIR
and SimplePIR), the wire-format bundles exchanged between server and
client, and the shared `IkpirError` enum.

## Role in the workspace

```
                ┌── ikpir-server ──┐
ikpir-common ───┤                  ├── segmented-cuckoo
                └── ikpir-client ──┘
```

`ikpir-server` and `ikpir-client` are siblings: both depend on
`ikpir-common` and re-export the items they expose in their own public
signatures. Production callers typically import from one of the protocol
crates (e.g. `use ikpir_server::IndexPirBackend`, `use ikpir_client::FrodoConfig`)
rather than from `ikpir-common` directly — every shared item resolves
under all three paths.

## What's here

| Module | Items |
|---|---|
| `backend` | `IndexPirBackend`, `IncrementalPirBackend`, `PrecomputingPirBackend`, `ParallelSetupBackend`, `BackendWireSize` traits + the `HintPatchMode` realization selector |
| `backend::frodo` | `FrodoPirBackend`, `FrodoConfig`, `FrodoParams`, plus the associated `FrodoServerParams / FrodoHint / FrodoClientState / FrodoQuery / FrodoResponse` types (ternary LWE, tall-skinny `n_rows × row_width` matrix; default `lwe_dim = 1566`) |
| `backend::simple` | `SimplePirBackend`, `SimpleConfig`, `SimpleParams`, plus the associated `SimpleServerParams / SimpleHint / SimpleClientState / SimpleQuery / SimpleResponse` types (discrete-Gaussian LWE with σ = 6.4, internal `√N × √N` reshape; default `lwe_dim = 1275`) |
| `wire` | `ServerSetupBundle`, `PirQueryBundle`, `PirResponseBundle`, `HintDeltaBundle`, `SegmentRowDeltas` type alias |
| `pir_params` | `frodo_max_plaintext_bits` / `simple_max_plaintext_bits` — largest `plaintext_bits` each backend decodes correctly at `q = 2³²` for a given per-segment geometry (consumed by `scripts/lib.sh::backend_plaintext_bits`, which `scripts/bench.sh` sources, via the `max_plaintext_bits` example) |
| `error` | `IkpirError` enum (`StaleEpoch`, `MalformedQuery`, `TableFull`, `NotFound`, `InvalidInput`) |

## Trait family at a glance

```text
IndexPirBackend (mandatory)
│   server_setup / client_setup / client_query / server_answer / client_decode
│   expand_hint_material  — re-derive server-local `A` from the seed
│   db_matrix_shape       — the (rows, cols) the backend settled on at setup
│
├── IncrementalPirBackend           (sparse hint patching, no full recompute)
│   server_patch_hint / client_patch_state   — both take a HintPatchMode:
│   RowLevel  = SimplePIR dense rank-one update, Θ(n·ω) per touched row
│   EntryLevel = iSimplePIR per-cell patch, Θ(n) per touched cell (default)
│   (identical post-patch state and wire bytes either way)
│
├── PrecomputingPirBackend          (FrodoPIR Fig. 1 amortisation)
│   client_precompute_queries (Phase B: A·s + e)
│   client_precompute_decodes (Phase C: sᵀ·H)
│   prepared_slot_count / in_flight_slot_count
│
├── ParallelSetupBackend            (same setup results, across cores)
│   server_setup_parallel / expand_hint_material_parallel / client_setup_parallel
│   (bit-identical to the single-threaded twins for the same seed; the
│    base trait stays single-threaded because the paper reports that
│    regime and `benches/server_setup.rs` times it)
│
└── BackendWireSize                 (byte-size accounting)
    query_byte_size / response_byte_size / hint_byte_size / server_params_byte_size
```

Both `FrodoPirBackend` and `SimplePirBackend` implement all five traits.

## Implementing a new backend

Implement `IndexPirBackend` (mandatory) and optionally
`IncrementalPirBackend`, `PrecomputingPirBackend`, `ParallelSetupBackend`,
`BackendWireSize`. Minimal correctness contract:

```text
client_decode(server_answer(client_query(state, row)))
    == db[row * row_width .. (row+1) * row_width]
```

See [`CLAUDE.md`](CLAUDE.md) for the full backend-author checklist and the
key design decisions behind the trait split.

## Status

Bundle types are not versioned; serialisation is out of scope.
Two backends ship: `FrodoPirBackend` (ternary errors, tall-skinny matrix,
default `lwe_dim = 1566`) and `SimplePirBackend` (discrete-Gaussian
errors with σ = 6.4, `√N × √N` internal reshape, default `lwe_dim = 1275`).
Both defaults target 128-bit security, estimated via the lattice
estimator under the ADPS16 cost model. Both implement all four traits
and are drop-in alternatives at the `B: IndexPirBackend` type parameter
on the server / client.
