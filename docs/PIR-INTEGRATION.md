# PIR integration: replacing BFF with SCF

This document specifies the contract between the Segmented Cuckoo Filter
(SCF, `crates/segmented-cuckoo-filter`) and the PIR client/server in
`crates/ikpir-{common,client,server}`. It is the design target Phase B must
satisfy.

## Why swap BFF for SCF?

[ChalametPIR] uses a **Binary Fuse Filter (BFF)** as the static index between
keywords and row positions in the hint matrix. BFFs achieve very high load
factors (~0.95) and have single-probe lookup — excellent for a *static*
database. But they have two structural drawbacks for our setting:

1. **Construction is monolithic.** Building a BFF for `n` items is an offline
   algorithm; individual inserts and deletes require a full rebuild.
2. **No concept of candidate set.** BFF lookup touches one position per key.
   There's no natural "degree of freedom" for the server to hide an update
   inside.

A **Segmented Cuckoo Filter** (SCF), by contrast:

- supports insert/delete/update natively,
- exposes a small candidate set (2, 3, or 4 buckets depending on arity) that
  the client queries in a single PIR round,
- matches or beats BFF load factor for arity 4 (see
  [`results/paper/scf/load_factor.csv`](../results/paper/scf/load_factor.csv)),
- and has a segmented layout that naturally maps one candidate per segment.

[ChalametPIR]: https://eprint.iacr.org/2024/092

## Encoding contract (the SCF ↔ PIR bridge)

Implemented in `ikpir-common/src/encoding.rs`. Invariants:

1. For a given `(keyword, filter_params)`, `encode(keyword)` returns a
   deterministic, ordered set of at most `arity` candidate bucket indices
   plus a fingerprint.
2. If `keyword` is present in the server's database, exactly one of the
   candidate buckets stores its record. The fingerprint disambiguates which.
3. If `keyword` is absent, no candidate bucket stores a matching
   fingerprint — modulo the filter's probabilistic false-positive rate.

This contract must hold under **any** of the 6 SCF variants. Phase B picks
one as the concrete instantiation (proposed: `Segmented4ary`, for its 0.94+
load factor and four-way parallelism in the response).

## Query shape

The single-round PIR query is a vector of length `num_buckets` over `Z_q`
encoding:

- `1` (masked with LWE noise) at each of the `arity` candidate indices,
- `0` (masked) elsewhere.

The server's response is a linear combination of DB rows weighted by the
client's noisy vector, revealing only the XOR of the candidate rows after
the client unmasks.

## Single round — why segmented matters

With a segmented filter, each candidate lives in a **disjoint** segment.
That means the server can respond to all `arity` candidates in one pass over
the database — no per-round bandwidth blow-up. If we used a standard
(non-segmented) cuckoo filter, the `arity` candidates could overlap in
position, forcing the server to re-read the same row multiple times or
expand the query basis.

## Hint matrix derivation

`ikpir-server::setup` constructs the hint matrix `H` of shape
`(LWE_DIMENSION, num_buckets * bucket_size)` such that for every key
`k`, the XOR of the rows at `encode(k).candidates` equals
`LWE_mask(v_k) XOR record_of(k)`. The precise construction mirrors
[ChalametPIR §3] with two substitutions:

1. Replace BFF's `hash(k) -> position` with `SCF.lookup(k) -> candidates`.
2. Replace BFF's monolithic row XOR with a candidate-indexed XOR per
   arity-row.

The full algebra is proved out in the paper, §<TODO-SECTION>.

## Wire format

See `ikpir-common/src/serialization.rs`. Each message type has a fixed
header (version, param hash) followed by the payload:

```
FilterParams:    [arity:u8][num_buckets:u32][bucket_size:u8][fp_bits:u8]
HintMatrix:      [rows:u32][cols:u32][payload ...]
Query:           [param_hash:u32][lwe_masked_selector: vec<Zq>]
Response:        [lwe_masked_xor: vec<Zq>]
```

## What Phase B must deliver to satisfy this contract

1. Concrete LWE parameters in `ikpir-common/src/params.rs`.
2. `encode()` and `decode()` in `ikpir-common/src/encoding.rs`.
3. `Server::new` building both SCF and `H` in `ikpir-server/src/setup.rs`.
4. `Client::new`, `Client::query`, `Client::decrypt` in `ikpir-client/`.
5. End-to-end test `crates/ikpir-server/tests/e2e.rs`.
6. Wire-format round-trip test in `ikpir-common`.

## Non-goals (for Phase B)

- Streaming / progressive responses.
- Batching multiple queries in one round.
- Composable PIR (layer on top of another scheme).

All three are interesting future directions but orthogonal to the updateable-
PIR thesis.
