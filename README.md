# Incremental Keyword PIR (RisePIR)

A research prototype of **Incremental Keyword PIR (IKPIR)** — a single-server
keyword-PIR construction that supports efficient **insert / update / delete** on
the server's database, while preserving the one-round structure of
state-of-the-art schemes.

This is the open-source implementation accompanying the paper *"Incremental
Keyword Private Information Retrieval from d-ary Segmented Cuckoo Filters"*
(CANS 2026 submission). The paper builds IKPIR generically from any **updatable
index PIR (UIPIR)** and instantiates the construction over LWE as **RisePIR**,
in two variants:

- **RisePIR-F** — over FrodoPIR: `IkpirServer<S, FrodoPirBackend>` / `IkpirClient<FrodoPirBackend>`;
- **RisePIR-S** — over SimplePIR: `IkpirServer<S, SimplePirBackend>` / `IkpirClient<SimplePirBackend>`.

The crates keep the generic names (`ikpir-*`): the code implements the generic
IKPIR-from-UIPIR construction, and the RisePIR variants are what you get by
choosing a backend at the `B: IndexPirBackend` type parameter.

> **Status.** Research prototype. Interfaces, parameters, and internals are
> subject to change.

## Background

Following the framework popularised by *ChalametPIR*, a keyword-PIR scheme can
be built from any **fingerprint-based filter** in two stages:

1. **Fingerprint filter → key-value store.**
   A fingerprint-based filter (e.g. Binary Fuse Filter, Cuckoo Filter) stores,
   for each inserted key `k`, a short fingerprint `fp(k)` placed at filter
   positions determined by a public, key-derived rule. The filter is upgraded
   into a key-value store by replacing each stored fingerprint with the pair
   `fp(k) ‖ v`. On lookup, the client reconstructs `fp(k) ‖ v` from the
   filter slots dictated by the public rule, checks the fingerprint, and — on
   match — accepts `v` as the value.

2. **Key-value store → keyword PIR.**
   The server publishes the key-value store as an array; the client knows,
   from the public rule, exactly which array indices it must read to recover
   `fp(k) ‖ v`. Reading those indices privately is the job of a standard
   **single-server Index-based PIR**. Because the rule selects a small,
   fixed-size set of indices, a single Index-PIR query suffices.

Under this framework, the choice of fingerprint-based filter determines the
*functionality* of the resulting keyword PIR.

## Why incremental?

ChalametPIR instantiates the framework with a **Binary Fuse Filter (BFF)**,
which is *static*: the entire filter must be rebuilt to insert, update, or
delete a key. For real-world databases — which evolve continuously — this
makes the static instantiation impractical.

A natural alternative is the standard **Cuckoo Filter**, which is dynamic.
However, Cuckoo Filter lookups read a *variable* number of buckets (usually
two, but the client cannot tell in advance which one holds the key). Plugged
into the framework above, this forces the client to issue **multiple
Index-PIR queries**, eroding the round and bandwidth profile that makes the
ChalametPIR-style construction attractive in the first place.

## This repository

This repository introduces the **Segmented Cuckoo Filter (SCF)** — a Cuckoo
Filter variant designed specifically for use as the fingerprint-based filter
inside the keyword-PIR framework. SCF is engineered so that:

- it supports **incremental** `insert`, `update`, and `delete`, like a
  standard Cuckoo Filter, and
- a key lookup reads a **deterministic, fixed set of slots**, so the resulting
  keyword PIR retains the **single Index-PIR query** profile of the static
  BFF-based construction.

Combined with an efficient preprocessing-update technique, SCF yields an
**Incremental Keyword PIR** scheme suitable for evolving databases.

## Compatibility

The construction is compatible with **any single-server Index-based PIR**
that supports efficient in-place updates — the paper's **UIPIR** interface,
realised in code as the `IndexPirBackend` (+ `IncrementalPirBackend`) trait
family. This repository ships two such backends, **FrodoPIR** and
**SimplePIR** — LWE-based Index-PIR schemes that offer high server throughput
and well-studied post-quantum security — yielding the paper's RisePIR-F and
RisePIR-S.

## Implementation status

| Component | Status |
|---|---|
| Segmented Cuckoo Filter + KV store | Shipped (`segmented-cuckoo`) |
| Backend trait family + wire bundles + shared error | Shipped (`ikpir-common`) |
| Server protocol (setup / answer / insert / update / delete / full_rebuild) | Shipped (`ikpir-server`) |
| Client protocol (from_setup / build_query / decode / apply_delta / reset_from) | Shipped (`ikpir-client`) |
| FrodoPIR backend | Shipped (`ikpir-common`) — ternary errors, tall-skinny matrix, default `lwe_dim = 1566` |
| SimplePIR backend | Shipped (`ikpir-common`) — discrete-Gaussian errors (σ = 6.4), √N×√N reshape, default `lwe_dim = 1275` |
| Hint-patch realizations (`HintPatchMode`) | Shipped (`ikpir-common`) — entry-level (iSimplePIR, default) and row-level (SimplePIR baseline); identical state + wire bytes, selectable per side |

## Repository tour

| Resource | Purpose |
|---|---|
| [`crates/segmented-cuckoo/CLAUDE.md`](crates/segmented-cuckoo/CLAUDE.md) | Filter + KV store internals, file map, design decisions |
| [`crates/ikpir-common/CLAUDE.md`](crates/ikpir-common/CLAUDE.md) | Shared crate internals: backend trait family, FrodoPIR, wire bundles, `IkpirError` |
| [`crates/ikpir-common/README.md`](crates/ikpir-common/README.md) | Shared crate role in the workspace + backend-author quick reference |
| [`crates/ikpir-server/CLAUDE.md`](crates/ikpir-server/CLAUDE.md) | Server crate internals, per-segment architecture, backend-author checklist |
| [`crates/ikpir-server/README.md`](crates/ikpir-server/README.md) | Server quick start and backend implementation guide |
| [`crates/ikpir-client/CLAUDE.md`](crates/ikpir-client/CLAUDE.md) | Client crate internals, epoch state machine, failure modes |
| [`crates/ikpir-client/README.md`](crates/ikpir-client/README.md) | Client quick start and lifecycle overview |

## Paper ↔ code notation

The paper's notation maps onto the code as follows.

**Filter layer (SCF / KV-SCF, paper §4–§5.1).**

| Paper | Meaning | Code |
|---|---|---|
| `d` | arity: candidate buckets per key = number of segments | `arity`, `CuckooParams::arity()` |
| `b` | slots per bucket | `bucket_size` |
| `n_b` | total buckets | `num_buckets` |
| `s = n_b / d` | buckets per segment (a power of two) | `CuckooParams::segment_size()` |
| `f` | fingerprint bits (benches fix 32) | `fingerprint_bits` |
| `ℓ` | value bits | `value_bits` |
| `fp(k) ‖ v` slot payload | fingerprint-then-value cell packing | `pack_slot_cells` / `unpack_slot_cells` |
| `MaxKicks` | eviction-walk budget | `MAX_KICKS_DEFAULT` (= 500) |
| `Candidates(x)` | candidate buckets of a key | `CuckooParams::candidate_buckets()`, `IndexScheme::hash_item` |
| `AltBuckets(i, φ)` | rebuild candidates from one bucket + fingerprint | `IndexScheme::all_indices` |
| write log `W` | per-mutation slot-change records | `SlotMutation`, `CuckooKVStore::drain_mutations` |
| `KV-SCF` | SCF-backed key-value store | `CuckooKVStore<S>` (`Segmented{2,3,4}aryCuckooKVStore`) |

**PIR layer (UIPIR / LWE-PIR, paper §3.1 and Appendix A).**

| Paper | Meaning | Code |
|---|---|---|
| `UIPIR.Setup` | preprocess one segment | `IndexPirBackend::server_setup` |
| `UIPIR.Query / Answer / Recover` | online phase | `client_query` / `server_answer` / `client_decode` |
| `UIPIR.DBMutation + HintUpdate` | mutation phase | server `insert/update/delete` → `IncrementalPirBackend::server_patch_hint`; client `IkpirClient::apply_delta` → `client_patch_state` |
| `IKPIR.Setup(DB)` | offline phase, all `d` segments | `IkpirServer::new` + `IkpirServer::setup` → `ServerSetupBundle` |
| `IKPIR.Query / Answer / Recover` | keyword online phase | `IkpirClient::build_query` / `IkpirServer::answer` / `IkpirClient::decode` |
| transcript `trans = (S_j)` | sparse per-segment overwrites | `HintDeltaBundle`, `SegmentRowDeltas` |
| hint `H = A·D` | client preprocessing material | `B::Hint` (`FrodoHint` / `SimpleHint`) |
| `A` (expanded from seed `β`) | public LWE matrix, never on the wire | `B::HintMaterial`, `expand_hint_material` |
| `n` | LWE dimension | `lwe_dim` (1566 FrodoPIR / 1275 SimplePIR) |
| `q = 2³²` | ciphertext modulus | native `u32` wraparound |
| `p = 2^pb` | plaintext modulus | `plaintext_bits` |
| `Δ = q/p` | plaintext↔ciphertext scaling | `round_p_to_q` / `round_q_to_p` |
| `χ_s, χ_e` | secret / error distributions | `sample_ternary_into` (FrodoPIR); `sample_uniform_zq_into` + `sample_discrete_gaussian_into` (SimplePIR) |
| `(ρ, ω)` reshape | per-segment matrix shape | FrodoPIR: identity `(n_rows, row_width)`; SimplePIR: near-square via `reshape_dims` |
| row-level / entry-level patch (Fig. 7) | hint-patch realizations | `HintPatchMode::RowLevel` / `HintPatchMode::EntryLevel` (default) |

One caveat on the letter `δ`/`Δ`: the paper uses `Δ = q/p` for the LWE
scaling, `δ` for correctness-failure probability, and the code additionally
uses "delta" for sparse hint patches (`HintDeltaBundle`, `row_deltas`). The
table above is the disambiguation.

## Benches

The PIR evaluation is nine focused, `clap`-parsed, CSV-emitting benches — four
in `ikpir-server`, five in `ikpir-client` — each measuring one criterion at one
config and appending one row (the per-`(patch mode, kind)` mutation benches
append one row per pair):

| Crate | Benches |
|---|---|
| `ikpir-server` | `server_setup`, `server_answer`, `server_mutation`, `headtohead_answer` |
| `ikpir-client` | `client_query`, `client_decode`, `client_mutation`, `headtohead_query`, `headtohead_decode` |

The `headtohead_*` benches fix the **keyword count** (`--num-keys`) and report
the DB size each scheme needed — the fair-comparison setting vs ChalametPIR /
Hao et al. 2025; the others fix the DB geometry and populate to `TableFull`
(or to `--load-factor` for the mutation benches). The `segmented-cuckoo` crate
adds nine filter / KV-store micro-benches (`load_factor`, `insert_throughput`,
`fpr`, …) that run their own fixed internal config matrix.

### Run one bench at one config

`scripts/bench.sh` runs a single bench at a single config, auto-deriving
`--plaintext-bits` (the max the backend's noise budget admits at `q = 2^32`)
and `--lwe-dim`, and writing the CSV under `results/<crate>/`:

```bash
./scripts/bench.sh server_answer --arity 4 --num-buckets 65536 --bucket-size 4 --value-bits 256
./scripts/bench.sh client_decode --backend simple
./scripts/bench.sh server_mutation --patch-mode entry,row
./scripts/bench.sh headtohead_answer --arity 4 --num-buckets 262144 --num-keys 1000000
./scripts/bench.sh insert_throughput            # segmented-cuckoo (fixed matrix)
./scripts/bench.sh                              # -h: full flag + bench list
```

All flags are optional — omitted geometry falls back to a small default config.
There is intentionally **no full-matrix sweep script**: reproducing the whole
paper dataset means looping `bench.sh` over the config matrix below (hours),
which a reader rarely wants. Run the handful of points you care about instead.

### Quick smoke / correctness check

`scripts/smoke.sh` runs every PIR bench at a tiny config on both backends in a
couple of minutes, exercising the full setup → answer → query → decode →
mutation path (each decode bench self-checks with `verify_decode`) — the
"test the properties on small configs" path:

```bash
./scripts/smoke.sh                              # frodo + simple
IKPIR_SMOKE_BACKENDS=frodo ./scripts/smoke.sh   # one backend
cargo test -p segmented-cuckoo                  # filter / KV-store properties, fast
```

### Plaintext-bits and the paper config matrix

`bench.sh` sets `plaintext_bits` per `(backend, SCF geometry, value_bits)` from
the correctness bound each backend actually multiplies per segment — FrodoPIR
Eq. 8 `q ≥ 8·p²·√m` (`m = num_buckets / arity`), SimplePIR Theorem C.1 adjusted
for uncentered cells and the near-square reshape,
`q/p ≥ 2√2·σ·√ln(2/δ)·p·√R` (σ = 6.4, δ = 2⁻⁴⁰) — which makes the SimplePIR
operating point depend on `value_bits`. The single source of truth is
`ikpir_common::pir_params`, exposed by the `max_plaintext_bits` example that
`scripts/lib.sh` shells out to; the chosen `pb` appears as a CSV column. The
`#[ignore]`d `noise_margin` tests validate these operating points empirically:

```bash
cargo test -p ikpir-common --release -- --ignored noise_margin --nocapture
```

The paper evaluates 12 throughput configs (6 `(arity, bucket_size)` pairs × 2 DB
sizes, all > 1 M entries) × 3 value widths, plus a fixed-N head-to-head matrix:

| arity `d` | bucket_size `b` | num_buckets `n_b` | total entries |
|---|---|---|---|
| 2 | 4 | 262144 / 1048576 | 2²⁰ / 2²² |
| 4 | 1 | 1048576 / 4194304 | 2²⁰ / 2²² |
| 4 | 2 | 524288 / 2097152 | 2²⁰ / 2²² |
| 3 | 2 | 786432 / 1572864 | 3·2¹⁹ / 3·2²⁰ |
| 3 | 3 | 393216 / 1572864 | 9·2¹⁷ / 9·2¹⁹ |
| 4 | 3 | 524288 / 1048576 | 3·2¹⁹ / 3·2²⁰ |

value widths: `--value-bits 256 / 2048 / 8192` (32 B / 256 B / 1 kB);
head-to-head fixes `--num-keys` ∈ {1 M, 1.5 M, 3 M, 4 M} at ~95 % load.

### Low-level: `cargo bench` directly

`bench.sh` is a thin wrapper; the benches also run standalone (results land in
the crate-local `results/` unless `IKPIR_RESULTS_DIR` is set). `--plaintext-bits`
then defaults to `8` (safe everywhere, but below each backend's max):

```bash
cargo bench -p ikpir-server --bench server_answer -- --backend simple --plaintext-bits 10
cargo bench -p segmented-cuckoo --bench fpr
```

CSVs land under `results/<crate>/` (`results/ikpir-server/`,
`results/ikpir-client/`, `results/segmented-cuckoo/`); the directory is
git-ignored and regenerated on demand.
