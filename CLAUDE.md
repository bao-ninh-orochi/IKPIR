# CLAUDE.md — Incremental Keyword PIR

## Project overview

Research prototype of **Incremental Keyword PIR (IKPIR)**: a single-server keyword-PIR scheme that supports efficient `insert`, `update`, and `delete` on the server database while preserving the one-round, single-Index-PIR-query profile of ChalametPIR-style constructions.

The key novelty is the **Segmented Cuckoo Filter (SCF)**: a dynamic fingerprint-based filter that (a) supports incremental updates like a standard Cuckoo Filter, and (b) makes key lookups read a *deterministic, fixed* set of slots, so the client always needs exactly one Index-PIR query.

Target Index-PIR backends: **FrodoPIR** and **SimplePIR** (LWE-based, post-quantum).

## Workspace structure

```
Incremental-Keywork-PIR/          ← workspace root
├── Cargo.toml                    ← workspace manifest
├── CLAUDE.md
├── README.md
├── segmented-cuckoo/             ← filter + key-value store primitives
├── ikpir-server/                 ← server-side IKPIR logic
└── ikpir-client/                 ← client-side IKPIR logic
```

## Crates

### `segmented-cuckoo`

Implements the core data structure layer. Exposes a public `SegmentedCuckooKVStore` for use by the `ikpir-server` crate.

Responsibilities:
- Standard Cuckoo Filter — used as the comparison baseline.
- Segmented Cuckoo Filter (SCF) — the novel variant with deterministic, fixed lookup positions.
- SCF-backed key-value store (`SegmentedCuckooKVStore`) — upgrades SCF by storing `fp(k) ‖ v` in each slot, providing the array that the PIR scheme will operate on.

### `ikpir-server`

Implements all server-side logic of the IKPIR protocol.

Responsibilities:
- **Create database** — initialise a `SegmentedCuckooKVStore` and populate it with the initial set of `(key, value)` pairs.
- **Preprocessing PIR** — compute the PIR hint / preprocessing matrix from the current database state (compatible with FrodoPIR / SimplePIR preprocessing).
- **Transfer preprocessing matrix** — send the preprocessing material to the client at setup time.
- **Handle PIR request** — receive an encoded PIR query from the client.
- **Process PIR request** — evaluate the PIR query against the current database array.
- **Response PIR request** — send the PIR response back to the client.
- **Insert `(key, value)`** — insert a new entry into the key-value store.
- **Delete key** — remove a key (and its associated value) from the key-value store.
- **Update value** — change the value stored for a given key from `v` to `v'`.
- **Re-compute preprocessing** — incrementally update the PIR preprocessing material to reflect the latest database mutation without a full rebuild.

### `ikpir-client`

Implements all client-side logic of the IKPIR protocol.

Responsibilities:
- **Extract indices** — given a query keyword `k`, use the SCF public rule to derive the fixed set of database array indices that encode `fp(k) ‖ v`.
- **Preprocessing** — receive and store the preprocessing matrix sent by the server; perform client-side setup.
- **Create PIR request** — encode the target index set into a single Index-PIR query using the stored preprocessing material.
- **Post-process PIR response** — decode the server's response, verify the fingerprint `fp(k)`, and extract the value `v`.
- **Re-compute preprocessing** — apply incremental updates to the local preprocessing state when the server database mutates.

## Design principles

- Each crate has a single, well-defined responsibility; cross-crate dependencies flow in one direction: `ikpir-server` and `ikpir-client` depend on `segmented-cuckoo`, never the reverse.
- The PIR backend (FrodoPIR vs SimplePIR) should be selectable via Cargo features on the server and client crates.
- Avoid dynamic dispatch on the hot path; prefer generics.
- All cryptographic and PIR primitives must be constant-time where relevant to avoid side-channel leakage.
