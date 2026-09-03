# CLAUDE.md — Incremental Keyword PIR

## Project overview

Research prototype of **Incremental Keyword PIR (IKPIR)**: a single-server keyword-PIR scheme that supports efficient `insert`, `update`, and `delete` on the server database while preserving the one-round, single-Index-PIR-query profile of ChalametPIR-style constructions.

This is the implementation behind the CANS 2026 paper *"Incremental Keyword Private Information Retrieval from d-ary Segmented Cuckoo Filters"*: the paper constructs IKPIR generically from any **updatable index PIR (UIPIR** — the `IndexPirBackend` + `IncrementalPirBackend` trait family**)** and names the LWE instantiation **RisePIR** — **RisePIR-F** with the FrodoPIR backend, **RisePIR-S** with the SimplePIR backend. The root `README.md` carries the full paper ↔ code notation table.

The key novelty is the **Segmented Cuckoo Filter (SCF)**: a dynamic fingerprint-based filter that (a) supports incremental updates like a standard Cuckoo Filter, and (b) makes key lookups read a *deterministic, fixed* set of slots, so the client always needs exactly one Index-PIR query per segment.

Target Index-PIR backends: **FrodoPIR** and **SimplePIR** (LWE-based, post-quantum).

## Workspace structure

```
RisePIR/                          ← workspace root
├── Cargo.toml                    ← workspace manifest (members = ["crates/*"])
├── CLAUDE.md
├── README.md
├── CONTRIBUTING.md               ← toolchain pin, local CI gates, PR conventions
├── SECURITY.md                   ← threat-model caveats, private vulnerability reporting
├── LICENSE / NOTICE              ← Apache-2.0 only, plus the copyright notice
├── deny.toml                     ← cargo-deny allow-list: dependencies stay permissive
├── .github/workflows/            ← ci.yml (fmt / clippy / test / bench compile check / licenses)
├── crates/
│   ├── segmented-cuckoo/         ← filter + key-value store primitives
│   ├── ikpir-common/             ← shared: backend traits, FrodoPIR, wire bundles, IkpirError
│   ├── ikpir-server/             ← server-side IKPIR logic
│   └── ikpir-client/             ← client-side IKPIR logic
├── docs/                         ← benchmark-machine spec (server-specs.txt) and the
│                                    HintDeltaBundle wire-format spec (hint-delta-wire-format.md)
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

Nine focused `clap`-parsed PIR benches — four server (`server_setup`,
`server_answer`, `server_mutation`, `headtohead_answer`) and five client
(`client_query`, `client_decode`, `client_mutation`, `headtohead_query`,
`headtohead_decode`) — emit CSV under `results/<crate>/`. Each invocation =
one config = one CSV row (the mutation benches emit one row per
`(patch mode, kind)` pair; the `headtohead_*` benches fix `--num-keys` and add
`num_keys`/`db_size` columns for the fixed-N comparison vs ChalametPIR /
Hao 2025).

`segmented-cuckoo` adds eight filter/KV-store benches: five `cuckoo_filter_*`
(`load_factor`, `insert_throughput`, `lookup_throughput`, `delete_throughput`,
`false_positive_rate`) and three `kv_store_*`. They are `clap`-parsed too, but
every flag is optional: with none, each runs the paper's **Table 2** matrix — the
five `(arity, bucket_size)` pairs at `fingerprint_bits = 64`, `max_kicks = 2500`,
~10^6 buckets. That matrix and every default live in
`crates/segmented-cuckoo/benches/configs.rs`, the single source of truth;
`--arity` / `--bucket-size` narrow it, other flags override one axis.

Run one bench at one config with **`scripts/bench.sh <name> [flags]`**, which
maps the bench to its crate, auto-derives `--plaintext-bits` and `--lwe-dim` for
the PIR benches, and exports `IKPIR_RESULTS_DIR=results/<crate>` before
`cargo bench`. Its geometry defaults are **dev scale** (~2^16 slots), not the
paper's — it is the everyday one-config runner.

**One sweep script per paper table**, each looping `bench.sh` over the matrix
that table reports: **`table2.sh`** (filter), **`table3.sh`** (online),
**`table4.sh`** (mutation), **`table5.sh`** (setup). `table2.sh` forwards all
flags to each bench; `table{3,4,5}.sh` additionally take
`--arity` / `--bucket-size` / `--value-bits` / `--backend` to narrow, and
forward the rest.

**`scripts/smoke.sh`** runs every PIR bench at a tiny config on both backends in
a couple of minutes (into a throwaway `results/.smoke/`) — the fast
correctness/property check. Shared derivation (pb/lwe, the bench→crate map) and
**the paper's PIR config matrix** (`PAPER_*`, `paper_num_buckets`,
`paper_num_keys`) live in **`scripts/lib.sh`** — the PIR-side counterpart of
`configs.rs`, and the single source of truth for what `table{3,4,5}.sh` sweep.

`plaintext_bits` is **not fixed across configs**. For each
`(backend, SCF geometry, value_bits)` quadruple, `bench.sh` picks the maximum
`pb` whose correctness bound holds at `q = 2^32`, evaluated at the
**per-segment** matrix each backend actually multiplies. Both selectors target
an explicit **per-cell** decode-failure budget
`δ_cell ≤ 2⁻⁽ᵏᵃᵖᵖᵃ⁺¹⁾ / (arity · row_width)` — half the paper's overall
`κ = 40` correctness target (Lemma 2, corrected: `δ, ξ ≤ κ`'s index term gets
half the budget in log-space, union-bounded over the `row_width` cells each of
the `arity` per-segment reads returns; the filter term is negligible at the
shipped `fingerprint_bits = 64` and is not this bound's concern). FrodoPIR
evaluates its uniform-ternary error's exact Bernstein tail against `δ_cell`
(replacing the old paper Eq. 8, `q ≥ 8·p²·√m`, whose implicit "w.h.p." constant
was far too weak for a `κ = 40` target); SimplePIR retargets its Theorem C.1
bound — adjusted for uncentered cells and the near-square reshape,
`q/p ≥ 2√2·σ·√ln(2/δ_cell)·p·√R` (σ = 6.4, `R` = reshape row count) — from a
fixed `δ = 2⁻⁴⁰` to the same per-config `δ_cell`. Both backends now depend on
the full per-segment geometry, including `fingerprint_bits`, not only
`segment_rows` (FrodoPIR) or `(segment_rows, value_bits)` (SimplePIR). The
single source of truth is `ikpir_common::pir_params` (full derivation,
including the History note on why Eq. 8 was replaced, in its module docs);
`scripts/lib.sh::backend_plaintext_bits` shells out to the
`max_plaintext_bits` example and passes the result as `--plaintext-bits`. Each
CSV row carries its `plaintext_bits`. The `#[ignore]`d `noise_margin` tests in
`ikpir-common` (`cargo test -p ikpir-common --release -- --ignored
noise_margin`) validate the selected operating points empirically.

The mutation benches (`server_mutation`, `client_mutation`) sweep the
hint-patch realization via `--patch-mode entry|row` (bench CLI default `entry`;
`bench.sh` passes `entry,row`), emitting one CSV row per `(patch mode, kind)`
pair with a `patch_mode` column — the empirical counterpart of the paper's
row-level vs entry-level mutation columns.

### Setup in the benches: reference vs optimized

Setup is `Θ(arity · n_rows · lwe_dim · row_width)` — minutes to hours at
paper scale, dwarfing every online operation. But **only `server_setup`
reports it**; for the other eight PIR benches it is preamble that
contributes nothing to the measured number, and their results depend on
setup's *output*, never on how it was computed.

So the setup phase ships in two implementations with identical output:

| | Entry points | Used by |
|---|---|---|
| **Reference** — single-threaded, non-SIMD; the paper's regime | `B::server_setup`, `IkpirServer::{new, full_rebuild}`, `IkpirClient::{from_setup, reset_from}` | `benches/server_setup.rs` — the measurement |
| **Optimized** — same output, all cores | `B::server_setup_parallel`, `IkpirServer::{new_parallel, full_rebuild_parallel}`, `IkpirClient::{from_setup_parallel, reset_from_parallel}` | every other bench's preamble |

The optimized path is **bit-identical**, not merely decode-equivalent: it
partitions only the output — disjoint bands of the hint matrix `H`,
disjoint runs of the ChaCha20 keystream that expands `A` (via
`set_word_pos`) — so each cell accumulates the same terms in the same
order. A server built on one path interoperates with a client built on
the other; `ikpir-server/tests/parallel_setup_equivalence.rs` pins every
combination. Threading is `std::thread::scope` (`ikpir-common`'s
`backend/parallel.rs`); no new dependency, no cargo feature, nothing to
enable. Contract: `ikpir_common::ParallelSetupBackend`.

Worker count comes from `IKPIR_SETUP_THREADS`, else
`available_parallelism()`; setting it to `1` forces the reference
schedule everywhere, which is the first thing to try when bisecting a
result mismatch. Measured on 8 cores: **4.8× (FrodoPIR), 6.3×
(SimplePIR)**.

`server_setup --setup-impl parallel` times the optimized path instead, to
quantify what the other benches save. Such a row is not a paper number,
and says so: the CSV's `setup_mode` column reads `full_parallel` rather
than `full`.

Either way the number is a whole `IkpirServer::new`, timed end to end.
No bench times a fraction of an operation and scales the result up —
that shortcut existed while the geometry was `N = 2²²` and was dropped
once `N = 2²⁰` made single-threaded setup affordable to measure
outright.

## The paper's PIR config matrix

Five KV-SCF shapes, each at one size (`paper_num_buckets` in `scripts/lib.sh`),
× value widths `2048 / 8192` (256 B / 1 kB) × backends `frodo / simple`:

| `(d, b)` | `n_b` | slots `N` | online keys `m` | in tables |
|---|---|---|---|---|
| (2, 4) | 2¹⁸ | 2²⁰ | 10⁶ | 3, 4, 5 |
| (4, 1) | 2²⁰ | 2²⁰ | 10⁶ | 3, 4, 5 |
| (4, 2) | 2¹⁹ | 2²⁰ | 10⁶ | 3, 4, 5 |
| (3, 2) | 3·2¹⁸ | 3·2¹⁹ = 1 572 864 | 1 415 577 (fill 0.90) | 3, 4 |
| (3, 3) | 3·2¹⁷ | 9·2¹⁷ = 1 179 648 | 1 061 683 (fill 0.90) | 3, 4 |

Why these, and not others:

- **Arity 2/4 share `N = 2²⁰`**, differing only in shape, and fix `m = 10⁶` —
  the count ChalametPIR and KPIR^index publish at, which fills them to 0.954,
  under every achieved load factor of Table 2.
- **Arity 3 is a full-paper addition**, absent from the submitted version. It
  cannot reach `2²⁰` (a segmented 3-ary table needs `n_b = 3·2^t`), so it takes
  the smallest rung of its ladder still holding 10⁶ keys at fill 0.90, and
  reports at that fill rather than a fixed `m` — it has no baseline to match.
  No Table 5 row: it carries no static-rebuild comparison.
- **32 B values and `(4, 3)` are deliberately gone** from every default and
  matrix. Both still run if passed explicitly; neither is a paper config.
- Mutation and setup seed to fill 0.90; mutation applies one batch of
  τ = 1 % of the slots (10 485 at `N = 2²⁰`), the paper's batch rule, with no
  clamp — a clamp would only ever bind at paper scale.

```bash
# One bench at one config, DEV scale (auto pb + lwe; results → results/<crate>/).
./scripts/bench.sh server_answer --arity 4 --num-buckets 65536 --value-bits 8192
./scripts/bench.sh client_mutation --backend simple --patch-mode entry,row
./scripts/bench.sh headtohead_answer --arity 4 --num-buckets 262144 --num-keys 1000000

# Reproduce a paper table end to end (paper geometry, from the matrix above).
./scripts/table2.sh                    # filter: SCF vs standard, five configs
./scripts/table3.sh                    # online: query / response / answer
./scripts/table4.sh --arity 3          # mutation: the full-paper arity-3 cells
./scripts/table5.sh --backend frodo    # setup: RisePIR-F rows only

# One segmented-cuckoo bench: no flags = the full Table 2 matrix.
./scripts/bench.sh cuckoo_filter_insert_throughput --arity 4 --bucket-size 2

# Fast correctness/property smoke across all PIR benches, both backends.
./scripts/smoke.sh

# Low-level: cargo bench directly (--plaintext-bits defaults to 8; results land
# in the crate-local results/ unless IKPIR_RESULTS_DIR is set).
cargo bench -p ikpir-server --bench server_answer -- \
    --num-buckets 65536 --bucket-size 4 --value-bits 8192 --plaintext-bits 10
```

## Design principles

- Each crate has a single, well-defined responsibility; cross-crate dependencies flow in one direction: `ikpir-server` and `ikpir-client` are siblings that both depend on `ikpir-common` and `segmented-cuckoo`. `ikpir-client` carries `ikpir-server` only as a `[dev-dependency]` for end-to-end tests / benches / doctest.
- The PIR backend (FrodoPIR vs SimplePIR) is selected at the `B: IndexPirBackend` type parameter on `IkpirServer<S, B>` / `IkpirClient<B>` (monomorphised, no Cargo features involved); the benches expose it as a runtime `--backend frodo|simple` flag.
- Avoid dynamic dispatch on the hot path; prefer generics.
- All cryptographic and PIR primitives must be constant-time where relevant to avoid side-channel leakage.
- **Measured code stays single-threaded and non-SIMD.** The paper reports that regime, so every operation a bench times runs it. Parallelism is confined to the setup phase, offered as a separate, explicitly named entry point (`*_parallel`) with a bit-identical-output contract — never as a flag that could silently change what a timed path does. Adding an optimized twin of any other operation must follow the same shape.
