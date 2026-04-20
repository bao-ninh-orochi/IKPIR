# Incremental update protocol — the novel contribution

**Status:** Phase C (to implement). This document specifies the design so
the implementation in `crates/ikpir-server/src/update.rs` has a clear target.

## Problem statement

The ChalametPIR hint matrix `H` is derived from a Binary Fuse Filter layout.
When even a single key is added to or removed from the database, BFF requires
a full reconstruction — which in turn rebuilds `H`. On a database of 2^20
keys this takes seconds to minutes and rewrites megabytes of matrix data.

We want to support:

- `server.insert(k, v)` in `O(polylog n)` work,
- `server.delete(k)` in `O(polylog n)` work,
- `server.update(k, v')` in `O(polylog n)` work,

such that a client holding any version of `H` is notified of the delta and
can refresh its state cheaply (ideally the same `polylog n` order).

## SCF makes this tractable

The Segmented Cuckoo Filter supports insert/delete natively. When we
`insert(k)`, at most `max_kicks` buckets are touched by the kick chain.
That bounds the subset of `H`'s columns that need to be rewritten: at most
`max_kicks × bucket_size × arity` columns, which is `polylog n` with the
typical `max_kicks = 500`.

## Insert protocol

```
server.insert(k, v):
  1. Δfilter ← SCF.add(k)        # tracks (bucket, slot, tag) transitions
  2. Δmatrix ← apply_filter_delta(Δfilter, v)
        For each (bucket, old_tag, new_tag) in Δfilter:
            if new_tag is None:  # slot was vacated
                subtract H_row(bucket, slot) from the hint column
            if old_tag is None:  # slot was filled
                add LWE_mask(v) XOR record_of(new_tag) to the hint column
            if both non-None:    # slot was replaced (kicked-in)
                do a symmetric subtract+add
  3. publish Δmatrix as a sparse patch
```

## Delete protocol

```
server.delete(k):
  1. (bucket, slot) ← SCF.locate(k)
  2. Δfilter ← SCF.remove(k)
  3. Δmatrix ← undo_row_contribution(bucket, slot, record_of(k))
  4. publish Δmatrix
```

## Update protocol

`update(k, v')` is `delete(k)` followed by `insert(k, v')`, but merged so
only one `Δmatrix` message is emitted.

## Client refresh

The client maintains a version counter. When it receives a `Δmatrix` patch,
it XORs the listed (column, delta) pairs into its local copy. Queries
formed against pre-patch `H` will fail to decrypt and the client MUST
refuse the response.

Open question (for the paper): do we require clients to apply patches
**in order**, or can we tolerate eventual consistency with a Merkle-root
commitment? Phase C will explore the ordered-patch design first.

## Correctness argument (sketch)

Let `H_0` be the hint matrix after initial setup and `H_t` the matrix after
`t` updates. We show by induction on `t`:

1. `H_0` satisfies the ChalametPIR decryption relation by construction.
2. Assume `H_{t-1}` satisfies the relation. Let `Δ_t` be the patch for the
   t-th update. Then `H_t = H_{t-1} XOR Δ_t` also satisfies the relation,
   because the only rows whose SCF lookup-set changed are those listed in
   `Δfilter_t`, and the corresponding hint-column deltas in `Δmatrix_t`
   exactly cancel the old contribution and add the new one.

The full proof lives in the paper, §<TODO-SECTION>.

## Break-even with full rebuild

Empirically (`results/paper/pir/full_rebuild_vs_incremental.csv`) the
incremental protocol is cheaper than a full rebuild up to <TODO>% of the
database being rewritten per batch. For bulk operations (>50% churn),
the Justfile exposes `server.rebuild()` as an explicit escape hatch.

## Leakage

Incremental updates introduce three leakage surfaces analyzed in
`docs/THREAT-MODEL.md`:

1. **Patch size** reveals `|Δfilter|` modulo padding, which correlates with
   kick-chain length. Mitigation: pad every patch to a fixed bucket of
   sizes.
2. **Patch timing** distinguishes "cheap incremental" from "expensive
   rebuild" to a network observer. Mitigation: constant-time response
   schedule.
3. **Client-linkability** across patches — if a malicious server emits
   distinct patches per client, it can probe queries for side-channel
   information. Mitigation: patches are broadcast identically to all
   clients; clients verify a published commitment.

## Benchmarks

| Bench | Output | Populated |
|---|---|---|
| `insert_cost` | `results/paper/pir/insert_cost.csv` | Phase C |
| `delete_cost` | `results/paper/pir/delete_cost.csv` | Phase C |
| `full_rebuild_vs_incremental` | `results/paper/pir/full_rebuild_vs_incremental.csv` | Phase C |
