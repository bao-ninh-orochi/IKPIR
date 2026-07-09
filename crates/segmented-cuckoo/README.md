# Segmented Cuckoo Filter

A **cuckoo filter** is a space-efficient probabilistic structure for approximate
set membership. It stores short fingerprints of items and supports **insert**,
**lookup**, and **delete** with a tunable false-positive rate. Unlike Bloom
filters, it supports deletion and is more space-efficient at low FPR.

This crate extends the design along two axes and compares the variants:

1. **Arity** — generalise partial-key cuckoo hashing from 2 candidate buckets
   (the original) to **3-ary and 4-ary** using the xor3/xor4 operations from the
   D-Ary Cuckoo Filter (Liu et al. 2017).
2. **Indexing** — the **Segmented Cuckoo Filter** partitions the bucket array
   into k equal segments and confines each candidate index to its own segment,
   directly realising a k-partite (hyper)graph structure.

The combination yields **6 filter variants** (Standard/Segmented × 2/3/4-ary),
all sharing identical insert/lookup/delete logic and differing only in index
computation. A segmented filter additionally gives every item a *deterministic,
fixed* set of lookup positions — the property IKPIR relies on to keep keyword
lookups to a single Index-PIR query.

> **Notation.** This README uses `k` for the arity, following the k-ary
> cuckoo-hashing literature it compares against; the accompanying CANS 2026
> paper calls the same parameter `d` (the *d-ary Segmented Cuckoo Filter*,
> SCF) and the code calls it `arity`. `k`, `d`, and `arity` are the same
> number throughout.

## Contributions

- **k-ary partial-key cuckoo hashing.** Extends the XOR partial-key scheme of
  Fan et al. (2014) to 3 and 4 buckets via xor3/xor4. Their *cycling property*
  (applying xord d times returns to the start) reconstructs all k candidate
  indices from any one index plus the fingerprint, with no per-slot position
  storage, for all arities.
- **Segmented (k-partite) construction.** Confines each candidate index to a
  dedicated segment; chain position derives from the segment index, again with
  no per-slot position storage.
- **Empirical comparison.** Benchmarks all 6 variants on load factor,
  insert/lookup/delete throughput, false-positive rate, and degree distribution,
  against theoretical k-ary cuckoo-hashing thresholds.

## Key findings

| Finding | Detail |
|---|---|
| k-ary works as expected | 3/4-ary standard filters reach load factors 0.15–2.10% below theoretical thresholds, consistent with the partial-key penalty |
| Segmented matches or beats standard | For b ≥ 2, segmented is +0.04% to +0.23% higher load factor than standard at the same arity |
| Standard 3-ary throughput limited by modulo | Standard 3-ary (n = 3^k) needs modulo-based `fingerprint_hash_mod`, slower than the bitmask path available when n is a power of 2 |
| Load factor decreases with table size | At fixed `max_kicks = 500`, larger tables hit lower load factor (the kick budget is proportionally smaller) |

---

## Getting started

```bash
cargo test                          # unit + doc tests
cargo run --example basic_usage     # demo all 6 filter types
cargo run --example load_factor     # fill to capacity, print load factors
```

**Filter usage:**

```rust
use segmented_cuckoo::Segmented2aryCuckooFilter;

let mut f = Segmented2aryCuckooFilter::new(1024, 4, 12).unwrap();
f.insert("hello").unwrap();
assert!(f.contain("hello"));
f.delete("hello").unwrap();

// Auto-size from expected item count.
use segmented_cuckoo::Standard3aryCuckooFilter;
let mut f = Standard3aryCuckooFilter::from_num_items(100_000, 4, 12).unwrap();
```

**KV-store usage** — the constructor takes the four filter-shape parameters plus
`plaintext_bits` (PIR cell width, 1–32; use 8 for byte-aligned values):

```rust
use segmented_cuckoo::Segmented2aryCuckooKVStore;

// num_buckets=64, bucket_size=4, fingerprint_bits=12, value_bits=64, plaintext_bits=8.
let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 64, 8).unwrap();
store.insert("hello", &[0u8; 8]).unwrap();
assert_eq!(store.get("hello"), Some(vec![0u8; 8]));

// Zero-allocation read into a caller-owned buffer.
let mut buf = vec![0u8; store.value_size_in_bytes()];
assert!(store.get_into("hello", &mut buf));

store.update("hello", &[1u8; 8]).unwrap();
store.delete("hello").unwrap();
```

---

## Background

A cuckoo filter stores `n` *buckets* of `b` *slots*; each slot holds an `f`-bit
*fingerprint*.

- **Insert** — compute a fingerprint and k candidate indices; place it in the
  first candidate with a free slot. If all are full, evict a random fingerprint
  and relocate it to one of *its* alternates (cuckoo kicking), up to `max_kicks`
  times; on failure, roll back all mutations and report the table full.
- **Lookup** — check the candidate buckets. A match is "probably yes"
  (FPR ≤ `k·b / 2^f`); a miss is "definitely no".
- **Delete** — remove the first matching fingerprint from a candidate bucket.

**Partial-key cuckoo hashing (2-ary).** Fan et al. (2014): the alternate index is
computable from the fingerprint alone, without storing or re-hashing the item:

```
i1 = H(x) mod n                     (primary index)
i2 = i1 XOR h(fingerprint(x))       (alternate index)
```

Since XOR is self-inverse, either index recovers the other:
`i_other = index XOR h(fingerprint)`. Eviction needs only the stored fingerprint.

---

## Extension 1: k-ary partial-key cuckoo hashing

We extend partial-key hashing to 3 and 4 candidates using **xor3** (digit-wise
add mod 3 over base-3 digits) and **xor4** (digit-wise add mod 4 over 2-bit
groups, an O(1) bit-twiddle) from Liu et al. (2017). Both are *generalised XOR*:
applying xord d times returns to the start.

```
xor2(xor2(a,b),b) = a                          (2 applications)
xor3(xor3(xor3(a,b),b),b) = a                  (3 applications)
xor4(xor4(xor4(xor4(a,b),b),b),b) = a          (4 applications)
```

**Index chain** — k candidates are chained xord applications with a
fingerprint-derived offset; e.g. standard 4-ary (n a power of 4):

```
fingerprint_hash(fp) = (fp * 0x5bd1e995) & (n-1)
i1 = (H(x) >> 32) & (n-1)
i2 = xor4(i1, fingerprint_hash(fp))
i3 = xor4(i2, fingerprint_hash(fp))
i4 = xor4(i3, fingerprint_hash(fp))
```

Standard 3-ary uses `(fp * 0x5bd1e995) % n` because n = 3^k is not a power of 2.

**No position storage.** The same offset is applied at each step, so the cycling
property reconstructs all k candidates from any one of them — no per-slot chain
position is stored, even for standard k > 2 filters.

The item hash is xxHash3 (64-bit, non-cryptographic): fingerprint from the lower
32 bits, primary index from the upper 32 bits. A fingerprint of 0 is forbidden
(marks empty); a 0 hash is replaced with 1.

---

## Extension 2: segmented cuckoo filters

**Motivation.** Theoretical k-ary cuckoo-hashing thresholds are derived under the
**k-partite** model (Sanders & Walzer 2022, Dietzfelbinger et al. 2010): the
vertex set is partitioned into k equal parts and each candidate is constrained to
one part. The original cuckoo filter instead uses a *standard* construction where
all indices share `[0, n)`. The question: does enforcing the k-partite structure
— with fingerprint-based, not truly random, hashes — match or improve the
standard construction?

**Construction.** Divide the `n` buckets into `k` equal segments and confine each
candidate to its own segment. The partial-key XOR formulas operate *within* a
segment; writing `i_j = j*segment_size + i_j_local`, e.g. segmented 4-ary:

```
i1_local = (H(x) >> 32) & (segment_size - 1)
i2_local = i1_local XOR h1(fp)
i3_local = i2_local XOR h2(fp)
i4_local = i3_local XOR h3(fp)
i_j = j * segment_size + i_j_local           (j = 0..k-1)
```

`segment_size` must be a power of two so `& (segment_size-1)` replaces a modulo;
thus n = `k * 2^m`.

**No position storage.** Chain position is `index / segment_size` — derivable on
the fly during kicking, no per-slot storage. (Standard k > 2 achieves the same
via the xord cycling property.)

**Table-size constraints:**

| Scheme | Constraint on n |
|---|---|
| Standard 2-ary | Power of 2 (≥ 1) |
| Standard 3-ary | Power of 3 (3^k) |
| Standard 4-ary | Power of 4 (4^k) |
| Segmented 2-ary | Power of 2, ≥ 2 |
| Segmented 3-ary | n = 3·2^m |
| Segmented 4-ary | Power of 2, ≥ 4 |

---

## Experimental setup

Defaults: `fingerprint_bits = 12` (unless swept), `max_kicks = 500`, xxHash3,
bucket sizes b ∈ {1,2,3,4}. Table sizes: standard 2/4-ary and segmented 2/4-ary
at n ∈ {2^14..2^20}; standard 3-ary at n ∈ {3^8..3^11}; segmented 3-ary at
n ∈ {3·2^12..3·2^18}.

| Bench | Measures | CSV (under `results/segmented-cuckoo/`) |
|---|---|---|
| `load_factor` | Max load factor, sweeping `max_kicks` ∈ {500..5000} | `load_factor.csv` |
| `insert_throughput` | Insert MOps/s while filling | `insert_throughput.csv` |
| `lookup_throughput` | Lookup MOps/s at 5 hit rates | `lookup_throughput.csv` |
| `delete_throughput` | Delete MOps/s on a full filter | `delete_throughput.csv` |
| `fpr` | FPR vs `fingerprint_bits` | `fpr/*.csv` |
| `degree_distribution` | Per-bucket degree at saturation | `degree_*.csv` |
| `kv_store_{insert,lookup,delete}_throughput` | KV-store MOps/s (segmented only) | `kv_store_*.csv` |

The first six are the comparison study below; the `kv_store_*` benches measure
the IKPIR primitive layer (segmented `(fingerprint, value)` slots).

### Running benchmarks

Each bench runs its own hardcoded config matrix (no CLI flags) and appends CSV
under `results/segmented-cuckoo/`. Run one through the workspace runner
[`../../scripts/bench.sh`](../../scripts/bench.sh), which routes the output:

```bash
./scripts/bench.sh load_factor                # one bench
./scripts/bench.sh fpr

# or directly — writes to the crate-local results/ unless IKPIR_RESULTS_DIR is set:
cargo bench -p segmented-cuckoo --bench load_factor
```

The filter / KV-store properties also have fast unit-test coverage:
`cargo test -p segmented-cuckoo`.

---

## Results

### Load factor

Mean load factor over all tested table sizes, 20 trials per config. **Threshold**
is the theoretical limit for k-ary cuckoo hashing with truly random hashes
(Sanders & Walzer 2022 / Dietzfelbinger et al. 2010).

| Arity | b | Threshold | Standard | Segmented | Seg. vs Std | Std vs Thresh. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 1 | ~50.0% | 51.76% | 48.71% | −3.05%\* | +1.76% |
| 2 | 2 | ~89.0% | 87.25% | 87.38% | **+0.13%** | −1.75% |
| 2 | 3 | ~95.0% | 93.63% | 93.86% | **+0.23%** | −1.37% |
| 2 | 4 | ~98.0% | 96.20% | 96.41% | **+0.21%** | −1.80% |
| 3 | 1 | ~91.8% | 89.70% | 89.68% | −0.01% | −2.10% |
| 3 | 2 | ~97.7% | 97.47% | 97.64% | **+0.17%** | −0.23% |
| 3 | 3 | ~99.2% | 98.86% | 98.96% | **+0.09%** | −0.34% |
| 3 | 4 | ~99.7% | 99.31% | 99.40% | **+0.09%** | −0.39% |
| 4 | 1 | ~97.6% | 96.13% | 96.12% | −0.01% | −1.47% |
| 4 | 2 | ~99.3% | 99.07% | 99.18% | **+0.11%** | −0.23% |
| 4 | 3 | ~99.7% | 99.55% | 99.61% | **+0.06%** | −0.15% |
| 4 | 4 | ~99.9% | 99.71% | 99.75% | **+0.04%** | −0.19% |

\* The 2-ary b=1 case is anomalous (see Discussion): standard slightly *exceeds*
the threshold, segmented falls below it.

- For **b ≥ 2 across all arities**, segmented equals or slightly beats standard.
- Standard 2-ary b=4 hits 96.2%, consistent with Fan et al. (2014)'s ~95.5% (our
  larger kick budget explains the gap).
- Higher arity helps sharply: 4-ary b=4 reaches 99.71%, within 0.19% of maximum.
- The 0.15–2.10% gap to threshold is the partial-key penalty: the xord offset has
  only `2^f` (or `3^f`) possible values, not `n`.

### Insert throughput (MOps/s)

2/4-ary at n = 262,144 (2^18); standard 3-ary at n = 177,147 (3^11), segmented
3-ary at n = 196,608 (3·2^16) — the 3-ary rows are at different n.

| Arity | b | Standard | Segmented | Diff |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 1 | 25.52 | 27.22 | +6.7% |
| 2 | 2 | 13.36 | 13.96 | +4.5% |
| 2 | 3 | 12.17 | 12.45 | +2.3% |
| 2 | 4 | 11.24 | 10.22 | −9.0% |
| 3 | 1 | 4.62 | 9.33 | n/a† |
| 3 | 2 | 5.06 | 11.38 | n/a† |
| 3 | 3 | 5.82 | 12.29 | n/a† |
| 3 | 4 | 6.10 | 11.08 | n/a† |
| 4 | 1 | 10.00 | 9.59 | −4.1% |
| 4 | 2 | 12.94 | 11.36 | −12.2% |
| 4 | 3 | 13.86 | 10.49 | −24.3% |
| 4 | 4 | 11.65 | 11.42 | −1.9% |

† Standard 3-ary is slowed by the modulo-based `fingerprint_hash_mod` (n = 3^k);
the n values also differ, so the raw numbers aren't directly comparable.

For 2-ary, segmented wins 3 of 4 configs; for 4-ary, standard wins 3 of 4. The
locality advantage of segmented (chains stay within one segment) helps 2-ary more
than 4-ary at these sizes.

### Lookup throughput (MOps/s, 50% hit rate)

2/4-ary at n = 262,144; standard 3-ary at n = 59,049 (3^10), segmented 3-ary at
n = 196,608 — 3-ary rows not directly comparable.

| Arity | b | Standard | Segmented | Diff |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 2 | 43.46 | 39.40 | −9.3% |
| 2 | 4 | 37.30 | 38.09 | +2.1% |
| 3 | 2 | 12.08 | 35.26 | n/a† |
| 3 | 4 | 13.31 | 26.50 | n/a† |
| 4 | 2 | 29.01 | 27.27 | −6.0% |
| 4 | 4 | 17.08 | 25.79 | +51.0% |

† Standard 3-ary slowed by modulo xor3 arithmetic; n values differ.

At matched n, differences range −9.3% to +51%. Throughput drops with higher arity
(more buckets to probe) and larger b (more slots per bucket).

### Delete throughput (MOps/s)

Same n values as insert.

| Arity | b | Standard | Segmented | Diff |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 1 | 48.47 | 44.44 | −8.3% |
| 2 | 2 | 32.54 | 32.82 | +0.9% |
| 2 | 3 | 29.50 | 30.57 | +3.6% |
| 2 | 4 | 28.15 | 27.81 | −1.2% |
| 3 | 1 | 13.93 | 41.97 | n/a† |
| 3 | 2 | 12.14 | 29.77 | n/a† |
| 3 | 3 | 11.36 | 26.29 | n/a† |
| 3 | 4 | 10.85 | 26.39 | n/a† |
| 4 | 1 | 29.30 | 33.23 | +13.4% |
| 4 | 2 | 25.69 | 25.87 | +0.7% |
| 4 | 3 | 24.82 | 23.89 | −3.7% |
| 4 | 4 | 24.19 | 24.03 | −0.7% |

† Standard 3-ary slowed by modulo xor3; n values differ.

At matched n, differences are small (−8% to +13%) with no consistent winner.
Delete scans candidate buckets and clears a match, so eviction locality is moot.

### False-positive rate

Theoretical FPR is `k·b / 2^f`. Measured at selected `fp_bits` (2-ary, n = 262,144, b = 4):

| fp_bits | Standard | Segmented | Theoretical (2b/2^f) |
|:---:|:---:|:---:|:---:|
| 8 | 2.986% | 2.973% | 3.125% |
| 10 | 0.745% | 0.746% | 0.781% |
| 12 | 0.188% | 0.188% | 0.195% |
| 16 | 0.012% | 0.012% | 0.012% |
| 20 | 0.001% | 0.001% | 0.001% |

Both variants track the theoretical curve closely. The indexing strategy has no
measurable effect on FPR, as expected — FPR depends only on fingerprint width and
bucket size, not on how indices are computed.

---

## Discussion

### Summary

The segmented cuckoo filter performs on par with or slightly better than the
standard filter across all metrics. The k-partite structure does not degrade
performance and gives a practical edge: slightly higher load factor for b ≥ 2.
No per-slot position storage is needed for any scheme — standard k > 2 uses the
xor3/xor4 cycling property, segmented derives position from the segment index.
The 3/4-ary extensions achieve load factors consistent with theoretical
thresholds, validating the partial-key approach at higher arities.

### Load factor decreases with table size

At fixed `max_kicks = 500`, achievable load factor falls as n grows (2-ary, b=4):

| n | Standard | Segmented |
|---:|:---:|:---:|
| 16,384 | 96.51% | 96.70% |
| 65,536 | 96.32% | 96.57% |
| 262,144 | 96.09% | 96.28% |
| 1,048,576 | 95.87% | 96.10% |

The kick budget is an absolute constant. At small tables 500 kicks is a large
fraction of capacity (3.1% at n=16,384), giving room to find free slots; at large
tables it is tiny (0.05% at n=1,048,576) and the budget is exhausted before
navigating the longer chains near saturation. Raise `max_kicks`
(`set_max_kicks()`) to recover, or accept the <1pp drop across a 64× size range.

### Why 2-ary b=1 segmented underperforms standard

At b=1, a 2-ary filter is edges of a random graph on n vertices; the threshold is
exactly 50%. **Standard** hits 51.76% (above): partial-key hashing correlates
index pairs, and at b=1 with a small fingerprint space these correlations
occasionally admit valid placements. **Segmented** hits 48.71% (below): the
bipartite constraint plus the limited fingerprint space makes the graph slightly
less connected. For **b ≥ 2** the effect vanishes — extra slots absorb the
connectivity difference and segmented matches or exceeds standard.

### Consistency with theory

Measured load factors sit 0.15–2.10% below the Sanders & Walzer thresholds. The
gap is partial-key hashing not being truly random: the offset `fingerprint_hash(fp)`
has at most `2^f` (or `3^f`) distinct values vs. uniform over `n`. This is the
same gap Fan et al. (2014) documented for 2-ary; we confirm it extends to 3/4-ary
with xor3/xor4. Segmented sits at or above standard for b ≥ 2, confirming the
k-partite structure does not hurt and may marginally help.

### Open questions

- **Scaling max_kicks.** Would `max_kicks = O(n)` or `O(n·b)` close the gap at large n?
- **Tighter bounds.** Can the thresholds be sharpened for partial-key (XOR-based) hashing?
- **Mixed-arity.** Could a filter start 2-ary at low load and raise arity as it fills?

---

## References

- B.-Y. Fan, D. G. Andersen, M. Kaminsky, M. D. Mitzenmacher. *Cuckoo Filter:
  Practically Better Than Bloom*. CoNEXT 2014.
- B. Liu, C. Li, Y. Lin, B. Vucetic, Y. Li. *D-Ary Cuckoo Filter: A Space
  Efficient Data Structure for Set Membership Lookup*. IEEE ICPADS 2017.
- P. Sanders, S. Walzer. *Load Thresholds for Cuckoo Hashing with Overlapping
  Blocks*. [arXiv:1707.06855](https://arxiv.org/pdf/1707.06855), 2022.
- M. Dietzfelbinger, A. Goerdt, M. Mitzenmacher, A. Montanari, R. Pagh, M. Rink.
  *Tight Thresholds for Cuckoo Hashing via XORSAT*. ICALP 2010.
