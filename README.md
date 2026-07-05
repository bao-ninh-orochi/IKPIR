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
| [`segmented-cuckoo/CLAUDE.md`](segmented-cuckoo/CLAUDE.md) | Filter + KV store internals, file map, design decisions |
| [`ikpir-common/CLAUDE.md`](ikpir-common/CLAUDE.md) | Shared crate internals: backend trait family, FrodoPIR, wire bundles, `IkpirError` |
| [`ikpir-common/README.md`](ikpir-common/README.md) | Shared crate role in the workspace + backend-author quick reference |
| [`ikpir-server/CLAUDE.md`](ikpir-server/CLAUDE.md) | Server crate internals, per-segment architecture, backend-author checklist |
| [`ikpir-server/README.md`](ikpir-server/README.md) | Server quick start and backend implementation guide |
| [`ikpir-client/CLAUDE.md`](ikpir-client/CLAUDE.md) | Client crate internals, epoch state machine, failure modes |
| [`ikpir-client/README.md`](ikpir-client/README.md) | Client quick start and lifecycle overview |

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

Six focused `clap`-parsed CSV-emitting benches (3 server + 3 client) cover
the classical and incremental PIR criteria needed for the paper, plus three
fused orchestration benches in `ikpir-client`
(`classical_throughput`, `mutation_throughput`, `headtohead_throughput`)
that share one expensive populate + setup across several measurements. Each
invocation runs one config and appends its row(s); sweeping across configs
is the orchestrator's job.

The canonical sweep is **`./scripts/run_all.sh`**, which chains the fused
sweeps (`run_classical.sh` → `run_mutation.sh` → `run_server_setup.sh`)
over the full paper config matrix (12 configs × 3 value\_bits = 36 runs
per sweep; the mutation sweep reuses the same 12 configs × 3 value\_bits,
with N\_mutations derived per config as capacity / 100 and one row per
(patch mode, kind) pair):

```bash
./scripts/run_all.sh                                  # FrodoPIR only (default)
IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_all.sh  # both backends (~2× runtime)
IKPIR_BENCH_BACKENDS=simple ./scripts/run_all.sh      # SimplePIR only

./scripts/run_all.sh --skip-setup                     # classical + mutation only
./scripts/run_all.sh --mutation-only                  # one sweep only (also --classical-only, --setup-only)
./scripts/run_all.sh --headtohead                     # additionally run the fixed-N head-to-head matrix
./ikpir-server/scripts/run_benches.sh server_answer   # one individual bench, full matrix
```

**Plaintext bits per config.** The orchestrator selects the largest
`plaintext_bits` whose noise budget admits `q = 2^32`, evaluated at the
**per-segment matrix each backend actually multiplies** (one index-PIR
instance per SCF segment): FrodoPIR Eq. 8 `q ≥ 8·p²·√m` with
`m = num_buckets / arity`, and SimplePIR Theorem C.1 adjusted for this
implementation's uncentered cells and near-square reshape,
`q/p ≥ 2√2·σ·√ln(2/δ)·p·√R` with σ = 6.4, δ = 2⁻⁴⁰ — which makes the
SimplePIR operating point depend on `value_bits`. The single source of
truth is `ikpir_common::pir_params` (invoked by
`scripts/configs.sh::backend_plaintext_bits` via the `max_plaintext_bits`
example); the chosen `pb` appears as a column in every CSV. The
`#[ignore]`d `noise_margin` tests in `ikpir-common` validate the selected
operating points empirically:

```bash
cargo test -p ikpir-common --release -- --ignored noise_margin --nocapture
```

For one-off measurements, invoke `cargo bench` directly. The CLI default
for `--plaintext-bits` is `8` (safe everywhere, but usually below the max
each backend supports); pass `--plaintext-bits N` explicitly to bench at
the best operating point.

```bash
# Server: setup time, answer throughput, mutation throughput.
cargo bench -p ikpir-server --bench server_setup
cargo bench -p ikpir-server --bench server_answer
cargo bench -p ikpir-server --bench server_mutation -- --n-mutations 1024

# Client: query, decode, and apply_delta (all warm-bc).
cargo bench -p ikpir-client --bench client_query
cargo bench -p ikpir-client --bench client_decode
cargo bench -p ikpir-client --bench client_mutation -- --n-mutations 64

# Backend selection: every bench accepts --backend frodo|simple (default frodo).
cargo bench -p ikpir-server --bench server_answer  -- --backend simple
cargo bench -p ikpir-client --bench client_query   -- --backend simple

# Hint-patch realization: the mutation benches accept --patch-mode entry|row
# (comma-separated, default entry) and emit one CSV row per (mode, kind) pair.
cargo bench -p ikpir-server --bench server_mutation -- --patch-mode entry,row
cargo bench -p ikpir-client --bench client_mutation -- --patch-mode entry,row

# Override plaintext_bits explicitly (defaults to 8).
cargo bench -p ikpir-server --bench server_answer -- --plaintext-bits 10

# Filter / KV-store baselines (load_factor, fpr, throughputs).
cargo bench -p segmented-cuckoo
```

CSVs land under `ikpir-server/results/` and `ikpir-client/results/`.
