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

Wraps a `CuckooKVStore` in per-segment Index-PIR sub-databases; exposes
setup, answer, insert, update, delete, and full_rebuild. Incremental
hint patching keeps the client in sync without a full rebuild. Backend
tunables are passed via the `IndexPirBackend::Config` associated type
(e.g. `FrodoConfig { lwe_dim }`); see
[`ikpir-server/CLAUDE.md`](ikpir-server/CLAUDE.md) for the full
per-segment architecture, protocol invariants, and backend-author checklist.

### `ikpir-client`

Holds `CuckooParams` and per-segment `ClientState`; translates keyword
lookups into PIR query/response bundles and applies incremental hint deltas.
See [`ikpir-client/CLAUDE.md`](ikpir-client/CLAUDE.md) for the epoch
state machine, failure-mode table, and entry-point map.

## Benches and visualization

Each crate ships `clap`-parsed benches that emit CSV under `results/`,
plus a `scripts/plot.py` (matplotlib + pandas) that turns those CSVs into
PNG charts under `results/plots/`. Each invocation = one config = one CSV
row (the writer is append-aware); sweeping across configs is the
orchestrator's job — `rm` the CSV first, then loop. Specific flags
(`--num-buckets`, `--bucket-size`, `--value-bits`, ...) pin a configuration.

The canonical sweep is `scripts/run_all.sh`, which iterates every bench
over a paper-derived config matrix (see `scripts/configs.sh` — anchors
ChalametPIR Tables 1/2 and Hao-2025 Table 1/Figure 10) and then renders
all plots. Per-crate orchestrators (`<crate>/scripts/run_benches.sh`) run
just that crate.

```bash
# Full sweep + plots (server then client).
./scripts/run_all.sh
IKPIR_BENCH_PROFILE=quick ./scripts/run_all.sh   # smaller matrix, ~minutes
IKPIR_BENCH_PROFILE=full  ./scripts/run_all.sh   # adds m=2^22 (very slow)

# One bench at one config (manual).
cargo bench -p ikpir-server --bench answer_throughput -- \
    --num-buckets 16384 --bucket-size 4 --value-bits 256

# Plots only (CSVs already populated).
cd ikpir-server && python scripts/plot.py
cd ikpir-client && python scripts/plot.py
```

## Design principles

- Each crate has a single, well-defined responsibility; cross-crate dependencies flow in one direction: `ikpir-server` and `ikpir-client` depend on `segmented-cuckoo`, never the reverse.
- The PIR backend (FrodoPIR vs SimplePIR) should be selectable via Cargo features on the server and client crates.
- Avoid dynamic dispatch on the hot path; prefer generics.
- All cryptographic and PIR primitives must be constant-time where relevant to avoid side-channel leakage.
