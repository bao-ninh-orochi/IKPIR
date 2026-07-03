# Incremental Keyword PIR

A research prototype of **Incremental Keyword PIR** — a single-server keyword-PIR
construction that supports efficient **insert / update / delete** on the server's
database, while preserving the one-round structure of state-of-the-art schemes.

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

The construction is compatible with **any single-server Index-based PIR**.
This repository targets in particular **FrodoPIR** and **SimplePIR**, two
LWE-based Index-PIR schemes that offer high server throughput and well-studied post-quantum security.

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

## Benches

Six focused `clap`-parsed CSV-emitting benches (3 server + 3 client) cover
the classical and incremental PIR criteria needed for the paper. Each
invocation runs one config and appends one row; sweeping across configs is
the orchestrator's job.

The canonical sweep is **`./scripts/run_all.sh`**, which iterates every
bench over the full paper config matrix (12 configs × 3 value\_bits = 36
runs per bench; mutation benches reuse the same 12 configs × 3 value\_bits
= 36 runs per bench, with N\_mutations derived per config as
capacity / 100):

```bash
./scripts/run_all.sh                                  # FrodoPIR only (default)
IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_all.sh  # both backends (~2× runtime)
IKPIR_BENCH_BACKENDS=simple ./scripts/run_all.sh      # SimplePIR only

./scripts/run_all.sh --server-only                    # server benches only
./scripts/run_all.sh --client-only                    # client benches only
./ikpir-server/scripts/run_benches.sh server_answer   # one bench only
```

**Plaintext bits per config.** The orchestrator selects the largest
`plaintext_bits` whose noise budget admits `q = 2^32` for each
`(backend, DB size)` pair (FrodoPIR Eq. 8 `q ≥ 8·p²·√m`; SimplePIR App. C.2
Eq. 2 `⌊q/p⌋ ≥ √2·σ·p·N^{1/4}·√ln(2/δ)` with σ = 6.4, δ = 2⁻⁴⁰).
The chosen `pb` lands in `scripts/configs.sh::backend_plaintext_bits` and
appears as a column in every CSV — see `scripts/configs.sh` for the table.

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
