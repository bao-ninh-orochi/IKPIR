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
├── ikpir-common/                 ← shared: backend traits, FrodoPIR, wire bundles, IkpirError
├── ikpir-server/                 ← server-side IKPIR logic
└── ikpir-client/                 ← client-side IKPIR logic
```

Dependency direction:

```
                ┌── ikpir-server ──┐
ikpir-common ───┤                  ├── segmented-cuckoo
                └── ikpir-client ──┘
```

`ikpir-server` and `ikpir-client` are siblings that both depend on
`ikpir-common`. `ikpir-client` depends on `ikpir-server` only via
`[dev-dependencies]` for end-to-end integration tests, benches, and the
quick-start doctest.

## Crates

### `segmented-cuckoo`

Implements the core data structure layer. Exposes a public `SegmentedCuckooKVStore` for use by the `ikpir-server` crate.

Responsibilities:
- Standard Cuckoo Filter — used as the comparison baseline.
- Segmented Cuckoo Filter (SCF) — the novel variant with deterministic, fixed lookup positions.
- SCF-backed key-value store (`SegmentedCuckooKVStore`) — upgrades SCF by storing `fp(k) ‖ v` in each slot, providing the array that the PIR scheme will operate on.

### `ikpir-common`

Single source of truth for items both protocol crates consume but
neither owns: the `IndexPirBackend` trait family (base +
`IncrementalPirBackend` + `PrecomputingPirBackend` + `BackendWireSize`),
the shipped `FrodoPirBackend`, the wire-format bundles
(`ServerSetupBundle / PirQueryBundle / PirResponseBundle / HintDeltaBundle`),
and the `IkpirError` enum. `ikpir-server` and `ikpir-client` both
re-export the items they expose in their own public signatures, so
existing call sites (`use ikpir_server::IndexPirBackend`, etc.) keep
resolving unchanged. See [`ikpir-common/CLAUDE.md`](ikpir-common/CLAUDE.md)
for the trait family overview and backend-author checklist.

### `ikpir-server`

Wraps a `CuckooKVStore` in per-segment Index-PIR sub-databases; exposes
setup, answer, insert, update, delete, and full_rebuild. Incremental
hint patching keeps the client in sync without a full rebuild; the
patch is realized at a selectable `HintPatchMode` granularity
(row-level à la SimplePIR or entry-level à la iSimplePIR, default
entry-level — identical state and wire bytes either way). Backend
tunables are passed via the `IndexPirBackend::Config` associated type
(e.g. `FrodoConfig { lwe_dim }`, defined in `ikpir-common`); see
[`ikpir-server/CLAUDE.md`](ikpir-server/CLAUDE.md) for the full
per-segment architecture, protocol invariants, and backend-author checklist.

### `ikpir-client`

Holds `CuckooParams` and per-segment `ClientState`; translates keyword
lookups into PIR query/response bundles and applies incremental hint deltas.
See [`ikpir-client/CLAUDE.md`](ikpir-client/CLAUDE.md) for the epoch
state machine, failure-mode table, and entry-point map.

## Benches

Six focused `clap`-parsed benches (3 server + 3 client) plus three fused
orchestration benches in `ikpir-client` (`classical_throughput`,
`mutation_throughput`, `headtohead_throughput` — each shares one
expensive populate + setup across several measurements) emit CSV under
`results/`. Each invocation = one config = one CSV row (the fused and
mutation benches emit one row per measured pair); sweeping across
configs is the orchestrator's job — `rm` the CSV first, then loop.

The canonical sweep is `scripts/run_all.sh`, which chains the fused
sweeps (`run_classical.sh` → `run_mutation.sh` → `run_server_setup.sh`;
`--headtohead` appends the fixed-N matrix, `--skip-setup` /
`--*-only` select subsets) over the full paper config matrix
(12 configs × 3 value\_bits = 36 runs per sweep; the mutation sweep
reuses the same 12 `BENCH_CONFIGS` entries × 3 value\_bits with
`n_mutations = capacity/100` per config; the historical separate
`MUTATION_CONFIGS` array has been unified into `BENCH_CONFIGS`; see
`scripts/configs.sh`). Per-crate orchestrators
(`<crate>/scripts/run_benches.sh`) run the individual benches for just
that crate.

`plaintext_bits` is **not fixed across configs**. For each
`(backend, DB size)` pair, the orchestrator picks the maximum `pb` whose
correctness bound holds at `q = 2^32` — FrodoPIR uses paper Eq. 8
(`q ≥ 8·p²·√m`), SimplePIR uses paper Appendix C.2 Eq. 2
(`⌊q/p⌋ ≥ √2·σ·p·N^{1/4}·√ln(2/δ)`, σ = 6.4, δ = 2⁻⁴⁰). The lookup
lives in `scripts/configs.sh::backend_plaintext_bits` and is passed
to every bench as `--plaintext-bits`. Each CSV row carries its
`plaintext_bits` so analyses can match results to operating points.

The mutation benches (`server_mutation`, `client_mutation`,
`mutation_throughput`) additionally sweep the hint-patch realization via
`--patch-mode entry|row` (orchestrator env `IKPIR_BENCH_PATCH_MODES`,
default `entry,row`), emitting one CSV row per `(patch mode, kind)`
pair with a `patch_mode` column — the empirical counterpart of the
paper's row-level vs entry-level mutation columns.

```bash
# Full sweep (server then client), FrodoPIR only (default).
./scripts/run_all.sh
IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_all.sh  # both backends (~2× runtime)
IKPIR_BENCH_PATCH_MODES=entry ./scripts/run_mutation.sh # one hint-patch realization

# One bench at one config (manual). When omitted, --plaintext-bits
# defaults to 8 — safe for every backend / DB size combination but
# usually *not* the max that the backend supports. Pass it explicitly
# (or use the orchestrator scripts) to bench at the best operating point.
cargo bench -p ikpir-server --bench server_answer -- \
    --num-buckets 65536 --bucket-size 4 --value-bits 256 --plaintext-bits 10
cargo bench -p ikpir-client --bench client_query -- \
    --num-buckets 65536 --bucket-size 4 --value-bits 256 --plaintext-bits 10
cargo bench -p ikpir-client --bench client_mutation -- \
    --num-buckets 65536 --bucket-size 4 --value-bits 256 --plaintext-bits 10 \
    --patch-mode entry,row
```

## Design principles

- Each crate has a single, well-defined responsibility; cross-crate dependencies flow in one direction: `ikpir-server` and `ikpir-client` are siblings that both depend on `ikpir-common` and `segmented-cuckoo`. `ikpir-client` carries `ikpir-server` only as a `[dev-dependency]` for end-to-end tests / benches / doctest.
- The PIR backend (FrodoPIR vs SimplePIR) should be selectable via Cargo features on the server and client crates.
- Avoid dynamic dispatch on the hot path; prefer generics.
- All cryptographic and PIR primitives must be constant-time where relevant to avoid side-channel leakage.
