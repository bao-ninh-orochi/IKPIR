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
| Server protocol (setup / answer / insert / update / delete / full_rebuild) | Shipped (`ikpir-server`) |
| Client protocol (from_setup / build_query / decode / apply_delta / reset_from) | Shipped (`ikpir-client`) |
| FrodoPIR backend | Shipped |
| SimplePIR backend | Planned |

## Repository tour

| Resource | Purpose |
|---|---|
| [`segmented-cuckoo/CLAUDE.md`](segmented-cuckoo/CLAUDE.md) | Filter + KV store internals, file map, design decisions |
| [`ikpir-server/CLAUDE.md`](ikpir-server/CLAUDE.md) | Server crate internals, per-segment architecture, backend-author checklist |
| [`ikpir-server/README.md`](ikpir-server/README.md) | Server quick start and backend implementation guide |
| [`ikpir-client/CLAUDE.md`](ikpir-client/CLAUDE.md) | Client crate internals, epoch state machine, failure modes |
| [`ikpir-client/README.md`](ikpir-client/README.md) | Client quick start and lifecycle overview |

## Benches and visualization

Each crate ships `clap`-parsed benches that emit CSV under `results/`,
mirroring `segmented-cuckoo`'s style. Each invocation runs a single config
and appends one row to its CSV (`csv_writer` is append-aware); sweeping
across configs is the orchestrator's job — a shell or Python wrapper that
`rm`s the CSV first, then loops over configs.

```bash
# Server: setup latency, answer rate, and the headline incremental crossover
cargo bench -p ikpir-server --bench setup_latency
cargo bench -p ikpir-server --bench answer_throughput
cargo bench -p ikpir-server --bench incremental_vs_rebuild  -- --n-mutations 1024

# Client: query construction, decode, and apply_delta
cargo bench -p ikpir-client --bench query_throughput
cargo bench -p ikpir-client --bench decode_throughput
cargo bench -p ikpir-client --bench apply_delta_throughput

# Plus the original segmented-cuckoo benches (load_factor, fpr, throughputs)
cargo bench -p segmented-cuckoo
```

Each crate also ships a `scripts/plot.py` (matplotlib + pandas) that
turns the CSV outputs into PNG charts under `results/plots/`:

```bash
cd ikpir-server && pip install -r scripts/requirements.txt && \
    python scripts/plot.py        # all plots
cd ikpir-client && pip install -r scripts/requirements.txt && \
    python scripts/plot.py --list # list individual functions
```
