# ikpir-common

Shared primitives for [Incremental Keyword PIR](../README.md). Holds the
Index-PIR backend trait family, the shipped FrodoPIR backend, the
wire-format bundles exchanged between server and client, and the shared
`IkpirError` enum.

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
| `backend` | `IndexPirBackend`, `IncrementalPirBackend`, `PrecomputingPirBackend`, `BackendWireSize` traits |
| `backend::frodo` | `FrodoPirBackend`, `FrodoConfig`, `FrodoParams`, plus the associated `FrodoServerParams / FrodoHint / FrodoClientState / FrodoQuery / FrodoResponse` types |
| `wire` | `ServerSetupBundle`, `PirQueryBundle`, `PirResponseBundle`, `HintDeltaBundle`, `SegmentRowDeltas` type alias |
| `error` | `IkpirError` enum (`StaleEpoch`, `MalformedQuery`, `TableFull`, `NotFound`, `InvalidInput`) |

## Trait family at a glance

```text
IndexPirBackend (mandatory)
│   server_setup / client_setup / client_query / server_answer / client_decode
│
├── IncrementalPirBackend           (sparse hint patching, no full recompute)
│   server_patch_hint / client_patch_state
│
├── PrecomputingPirBackend          (FrodoPIR Fig. 1 amortisation)
│   client_precompute_queries (Phase B: A·s + e)
│   client_precompute_decodes (Phase C: sᵀ·H)
│   prepared_slot_count / in_flight_slot_count
│
└── BackendWireSize                 (byte-size accounting)
    query_byte_size / response_byte_size / hint_byte_size / server_params_byte_size
```

`FrodoPirBackend` implements all four traits.

## Implementing a new backend

Implement `IndexPirBackend` (mandatory) and optionally
`IncrementalPirBackend`, `PrecomputingPirBackend`, `BackendWireSize`.
Minimal correctness contract:

```text
client_decode(server_answer(client_query(state, row)))
    == db[row * row_width .. (row+1) * row_width]
```

See [`CLAUDE.md`](CLAUDE.md) for the full backend-author checklist and the
key design decisions behind the trait split.

## Status

Bundle types are not versioned; serialisation is out of scope.
`FrodoPirBackend` is the shipped backend. SimplePIR is a future track.
