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
