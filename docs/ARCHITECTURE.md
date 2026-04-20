# Architecture

One-page map of the Incremental-Keyword-PIR system and a guided tour of the
workspace.

## System overview

```
                 ┌────────────────────────────┐
                 │       DB owner / server     │
                 │                              │
  DB = { (k_i,   │   ┌──────────────────────┐   │
         v_i) }──┼──▶│  Segmented Cuckoo    │   │
                 │   │  Filter (SCF)        │   │   hint matrix H
                 │   └──────────┬───────────┘   │   (public, derived
                 │              │               │    from SCF layout)
                 │              ▼               │
                 │   ┌──────────────────────┐   │
                 │   │  Preprocessing /     │────┼──────────▶ Client
                 │   │  hint matrix H       │   │
                 │   └──────────────────────┘   │
                 └──────────────┬─────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │       Client                  │
                 │   query(k) = q (LWE)          │
                 └──────────────┬───────────────┘
                                │ q
                                ▼
                 ┌──────────────────────────────┐
                 │       Server                  │
                 │   response r = respond(q)     │
                 └──────────────┬───────────────┘
                                │ r
                                ▼
                 ┌──────────────────────────────┐
                 │       Client                  │
                 │   v = decrypt(r)              │
                 └──────────────────────────────┘
```

Incremental mutation (Phase C) is a second arrow from the DB owner back into
the server's SCF + hint matrix:

```
  insert(k,v) / delete(k) / update(k, v') ──▶ SCF.mutate() ──▶ hint matrix Δ
```

## Workspace layout

```
crates/
├── segmented-cuckoo-filter/   — Phase A: stand-alone filter library
├── ikpir-common/              — Phase B: shared PIR primitives
├── ikpir-client/              — Phase B: client
└── ikpir-server/              — Phase B (+ C): server with incremental updates
```

Dependencies flow strictly downstream:

```
   ikpir-server ─▶ ikpir-common ─▶ segmented-cuckoo-filter
   ikpir-client ─▶ ikpir-common ─▶ segmented-cuckoo-filter
   ikpir-server ─▶ ikpir-client (dev-only — for end-to-end tests)
```

## Module map

| Crate                   | Module             | Purpose |
|-------------------------|-------------------|---------|
| segmented-cuckoo-filter | `filter`           | `CuckooFilter<S>` generic over `IndexScheme` |
| segmented-cuckoo-filter | `scheme`           | 6 scheme impls (standard/segmented × 2/3/4-ary) |
| segmented-cuckoo-filter | `bucket`           | Bit-packed `TagTable` — arbitrary fingerprint widths |
| segmented-cuckoo-filter | `hash`             | xxHash3 + tag-hash variants + xor3/xor4 index cycling |
| segmented-cuckoo-filter | `util`             | `upper_power_of_2` / power-of-{3,4} helpers |
| ikpir-common            | `params`           | LWE params, `Arity` enum, `FilterParams` |
| ikpir-common            | `encoding`         | keyword → SCF position → PIR query row set |
| ikpir-common            | `matrix`           | matrix types + arithmetic over `Z_q` |
| ikpir-common            | `lwe`              | noise sampling, keygen, vector encrypt/decrypt |
| ikpir-common            | `hash`             | extra domain-separated hashing |
| ikpir-common            | `serialization`    | wire format for params, matrix, queries, responses |
| ikpir-client            | `setup`            | `Client::new(hint_matrix, params)` |
| ikpir-client            | `query`            | build query for keyword → `(QueryState, QueryBytes)` |
| ikpir-client            | `decrypt`          | response → record value |
| ikpir-server            | `setup`            | build SCF + derive hint matrix |
| ikpir-server            | `respond`          | single-round server response |
| ikpir-server            | `update`           | **Phase C — novel.** Incremental insert/delete/update |

## Data flow — single query

1. **Server setup (one-time).** Build SCF over the DB key set; for each row
   in the SCF, derive the corresponding hint-matrix row via the LWE
   preprocessing relation. The filter + hint matrix are published.
2. **Client setup.** Ingest `(filter_params, hint_matrix)`; store client state.
3. **Client query.** `client.query(k)` resolves `k` to a set of candidate
   bucket indices (the SCF candidates), masks them with LWE noise, and emits
   a single query message.
4. **Server respond.** XOR the DB rows at the requested candidate indices
   using the LWE-blinded selector.
5. **Client decrypt.** Undo the LWE mask and verify the fingerprint to
   identify which of the candidates actually holds `k`.

## Data flow — incremental update (Phase C)

1. **Insert.** Add the key to the SCF. If a kick chain moves row `i` to row
   `j`, update the hint-matrix columns that depended on row `i`.
2. **Delete.** Remove the fingerprint from the SCF bucket. Update the hint
   column that depended on that row so subsequent queries for that key
   decrypt to "not found" (per protocol).
3. **Update.** Combine delete(k) + insert(k, v') with a single
   hint-matrix pass.

See [`docs/INCREMENTAL-UPDATE.md`](INCREMENTAL-UPDATE.md) for the full algorithm
and [`docs/THREAT-MODEL.md`](THREAT-MODEL.md) for the leakage analysis.

## Where each paper claim is proved

| Claim                                                       | Location |
|-------------------------------------------------------------|----------|
| SCF achieves ≥94% load factor at arity 4                    | [`results/paper/scf/load_factor.csv`](../results/paper/scf/load_factor.csv) |
| Segmented vs. standard throughput comparison                | [`docs/BENCHMARKS.md`](BENCHMARKS.md) |
| PIR query is single-round                                   | [`docs/PIR-INTEGRATION.md`](PIR-INTEGRATION.md) §<TODO> |
| Incremental update is cheaper than full rebuild from n=<TODO> on | [`results/paper/pir/full_rebuild_vs_incremental.csv`](../results/paper/pir/) |
| Updateability preserves LWE-IND-CPA security                | [`docs/THREAT-MODEL.md`](THREAT-MODEL.md) |
