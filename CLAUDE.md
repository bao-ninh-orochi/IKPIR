# CLAUDE.md — Incremental Keyword PIR

## Project overview

Research prototype of **Incremental Keyword PIR (IKPIR)**: a single-server keyword-PIR scheme that supports efficient `insert`, `update`, and `delete` on the server database while preserving the one-round, single-Index-PIR-query profile of ChalametPIR-style constructions.

This is the implementation behind the CANS 2026 paper *"Incremental Keyword Private Information Retrieval from d-ary Segmented Cuckoo Filters"*: the paper constructs IKPIR generically from any **updatable index PIR (UIPIR** — the `IndexPirBackend` + `IncrementalPirBackend` trait family**)** and names the LWE instantiation **RisePIR** — **RisePIR-F** with the FrodoPIR backend, **RisePIR-S** with the SimplePIR backend. The root `README.md` carries the full paper ↔ code notation table.

The key novelty is the **Segmented Cuckoo Filter (SCF)**: a dynamic fingerprint-based filter that (a) supports incremental updates like a standard Cuckoo Filter, and (b) makes key lookups read a *deterministic, fixed* set of slots, so the client always needs exactly one Index-PIR query per segment.

Target Index-PIR backends: **FrodoPIR** and **SimplePIR** (LWE-based, post-quantum).

## Workspace structure

```
Incremental-Keyword-PIR/          ← workspace root
├── Cargo.toml                    ← workspace manifest (members = ["crates/*"])
├── CLAUDE.md
├── README.md
├── CONTRIBUTING.md               ← toolchain pin, local CI gates, PR conventions
├── SECURITY.md                   ← threat-model caveats, private vulnerability reporting
├── LICENSE-APACHE / LICENSE-MIT  ← dual license (MIT OR Apache-2.0)
├── .github/workflows/            ← ci.yml (fmt / clippy / test / bench compile check)
├── crates/
│   ├── segmented-cuckoo/         ← filter + key-value store primitives
│   ├── ikpir-common/             ← shared: backend traits, FrodoPIR, wire bundles, IkpirError
│   ├── ikpir-server/             ← server-side IKPIR logic
│   └── ikpir-client/             ← client-side IKPIR logic
├── docs/                         ← benchmark-machine spec (server-specs.txt)
└── scripts/                      ← bench runner (bench.sh), smoke.sh, shared lib.sh
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
resolving unchanged. See [`crates/ikpir-common/CLAUDE.md`](crates/ikpir-common/CLAUDE.md)
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
[`crates/ikpir-server/CLAUDE.md`](crates/ikpir-server/CLAUDE.md) for the full
per-segment architecture, protocol invariants, and backend-author checklist.

### `ikpir-client`

Holds `CuckooParams` and per-segment `ClientState`; translates keyword
lookups into PIR query/response bundles and applies incremental hint deltas.
See [`crates/ikpir-client/CLAUDE.md`](crates/ikpir-client/CLAUDE.md) for the epoch
state machine, failure-mode table, and entry-point map.

## Benches

Nine focused `clap`-parsed benches — four server (`server_setup`,
`server_answer`, `server_mutation`, `headtohead_answer`) and five client
(`client_query`, `client_decode`, `client_mutation`, `headtohead_query`,
`headtohead_decode`) — emit CSV under `results/<crate>/`. Each invocation =
one config = one CSV row (the mutation benches emit one row per
`(patch mode, kind)` pair; the `headtohead_*` benches fix `--num-keys` and add
`num_keys`/`db_size` columns for the fixed-N comparison vs ChalametPIR /
Hao 2025). `segmented-cuckoo` adds nine filter/KV-store micro-benches that run
a fixed internal config matrix (no CLI flags).

Run one bench at one config with **`scripts/bench.sh <name> [flags]`**, which
maps the bench to its crate, auto-derives `--plaintext-bits` and `--lwe-dim`,
and exports `IKPIR_RESULTS_DIR=results/<crate>` before `cargo bench`. There is
**no full-matrix sweep script** (the paper is complete); reproducing the whole
matrix means looping `bench.sh`. **`scripts/smoke.sh`** runs every PIR bench at
a tiny config on both backends in a couple of minutes (into a throwaway
`results/.smoke/`) — the fast correctness/property check. Shared derivation
(pb/lwe, the bench→crate map) lives in **`scripts/lib.sh`**.

`plaintext_bits` is **not fixed across configs**. For each
`(backend, SCF geometry, value_bits)` triple, `bench.sh` picks the maximum `pb`
whose correctness bound holds at `q = 2^32`, evaluated at the **per-segment**
matrix each backend actually multiplies — FrodoPIR paper Eq. 8 (`q ≥ 8·p²·√m`
with `m = num_buckets / arity`), SimplePIR paper Theorem C.1 adjusted for
uncentered cells and the near-square reshape (`q/p ≥ 2√2·σ·√ln(2/δ)·p·√R`,
σ = 6.4, δ = 2⁻⁴⁰, `R` = reshape row count, which depends on `value_bits`). The
single source of truth is `ikpir_common::pir_params` (full derivation in its
module docs); `scripts/lib.sh::backend_plaintext_bits` shells out to the
`max_plaintext_bits` example and passes the result as `--plaintext-bits`. Each
CSV row carries its `plaintext_bits`. The `#[ignore]`d `noise_margin` tests in
`ikpir-common` (`cargo test -p ikpir-common --release -- --ignored
noise_margin`) validate the selected operating points empirically.

The mutation benches (`server_mutation`, `client_mutation`) sweep the
hint-patch realization via `--patch-mode entry|row` (bench CLI default `entry`;
`bench.sh` passes `entry,row`), emitting one CSV row per `(patch mode, kind)`
pair with a `patch_mode` column — the empirical counterpart of the paper's
row-level vs entry-level mutation columns.

```bash
# One bench at one config (auto pb + lwe; results → results/<crate>/).
./scripts/bench.sh server_answer --arity 4 --num-buckets 65536 --value-bits 256
./scripts/bench.sh client_mutation --backend simple --patch-mode entry,row
./scripts/bench.sh headtohead_answer --arity 4 --num-buckets 262144 --num-keys 1000000

# Fast correctness/property smoke across all PIR benches, both backends.
./scripts/smoke.sh

# Low-level: cargo bench directly (--plaintext-bits defaults to 8; results land
# in the crate-local results/ unless IKPIR_RESULTS_DIR is set).
cargo bench -p ikpir-server --bench server_answer -- \
    --num-buckets 65536 --bucket-size 4 --value-bits 256 --plaintext-bits 10
```

## Design principles

- Each crate has a single, well-defined responsibility; cross-crate dependencies flow in one direction: `ikpir-server` and `ikpir-client` are siblings that both depend on `ikpir-common` and `segmented-cuckoo`. `ikpir-client` carries `ikpir-server` only as a `[dev-dependency]` for end-to-end tests / benches / doctest.
- The PIR backend (FrodoPIR vs SimplePIR) is selected at the `B: IndexPirBackend` type parameter on `IkpirServer<S, B>` / `IkpirClient<B>` (monomorphised, no Cargo features involved); the benches expose it as a runtime `--backend frodo|simple` flag.
- Avoid dynamic dispatch on the hot path; prefer generics.
- All cryptographic and PIR primitives must be constant-time where relevant to avoid side-channel leakage.
