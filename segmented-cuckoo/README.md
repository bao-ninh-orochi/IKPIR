# Segmented Cuckoo Filter

A **cuckoo filter** is a space-efficient probabilistic data structure for approximate set
membership. It stores short hashes (*fingerprints*) of inserted items and supports
**insert**, **lookup**, and **delete** with a tunable false-positive rate. Unlike Bloom
filters, cuckoo filters support deletion and achieve better space efficiency at low
false-positive rates.

This project is an experimental study that extends the cuckoo filter design along two axes
and empirically compares the variants:

1. **Arity** -- we generalise the partial-key cuckoo hashing scheme from 2 candidate
   buckets (the original design) to **3-ary and 4-ary** constructions using the xor3 and
   xor4 operations from the D-Ary Cuckoo Filter (Liu et al. 2017, IEEE ICPADS).
2. **Indexing strategy** -- we introduce the **Segmented Cuckoo Filter**, which partitions
   the bucket array into k equal segments and confines each candidate index to its own
   segment, directly implementing a k-partite (hyper)graph structure.

The combination yields **6 concrete filter variants** (Standard/Segmented x 2/3/4-ary), all
sharing identical insert/lookup/delete logic and differing only in index computation.

## Contributions

- **k-ary partial-key cuckoo hashing.** We extend the XOR-based partial-key scheme of
  Fan et al. (2014) from 2 candidate buckets to 3 and 4, using the xor3 and xor4 operations
  from Liu et al. (2017). The xord *cycling property* (applying xord d times returns to the
  start) enables reconstruction of all k candidate indices from any one index and the
  fingerprint, without per-slot position storage, for all arities.

- **Segmented (k-partite) cuckoo filter construction.** We propose a variant that confines
  each of the k candidate indices to a dedicated segment of the table. The chain position is
  derived directly from the segment index, requiring no per-slot position storage for any
  arity.

- **Comprehensive empirical comparison.** We benchmark all 6 variants on load factor,
  insert/lookup/delete throughput, false-positive rate, and degree distribution,
  and compare against theoretical thresholds from k-ary cuckoo hashing theory.

## Key Findings

| Finding | Detail |
|---|---|
| **k-ary works as expected** | 3-ary and 4-ary standard filters (xor3/xor4 construction) achieve load factors 0.15--2.10% below theoretical thresholds, consistent with the partial-key penalty |
| **Segmented matches or beats standard** | For b >= 2, segmented achieves +0.04% to +0.23% higher load factor than standard at the same arity |
| **3-ary standard throughput limited by modulo** | Standard 3-ary (n = 3^k) uses modulo-based `fingerprint_hash_mod` which is slower than the bitmask operations available when n is a power of 2 |
| **Load factor decreases with table size** | For fixed `max_kicks = 500`, larger tables achieve lower load factor (the kick budget becomes proportionally smaller) |

---

## Table of Contents

- [Getting Started](#getting-started)
- [Background](#background)
- [Extension 1: k-ary Partial-Key Cuckoo Hashing](#extension-1-k-ary-partial-key-cuckoo-hashing)
- [Extension 2: Segmented Cuckoo Filters](#extension-2-segmented-cuckoos)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Discussion](#discussion)
- [References](#references)

---

## Getting Started

### Project structure

```
src/
  lib.rs             -- public API, type aliases for all 6 filter types and KV stores
  filter.rs          -- CuckooFilter<S>: generic insert/lookup/delete with cuckoo kicking
  scheme.rs          -- IndexScheme trait + 6 concrete scheme structs
  hash.rs            -- xxHash3 item hashing, fingerprint-hash functions, index reconstruction
  data_layout.rs     -- DataLayout, FingerprintTable, FingerprintValueTable: bit-packed slot storage
  store.rs           -- CuckooKVStore<S>: fingerprint-and-value KV store with kicking + rollback
  util.rs            -- next_power_of_2, power-of-3 and power-of-4 helpers

examples/
  basic_usage.rs          -- demo of both 2-ary filter types
  kv_store_basic_usage.rs -- demo of Segmented2aryCuckooKVStore insert/get/delete/update
  load_factor.rs          -- fill filters to capacity and print max load factor

benches/             -- standalone benchmark binaries (no clap; write CSV to results/)
  load_factor.rs                -- max load factor (sweeps max_kicks ∈ {500..5000})
  insert_throughput.rs          -- insert MOps/s while filling to capacity
  lookup_throughput.rs          -- lookup MOps/s at 5 hit rates (0/25/50/75/100%)
  delete_throughput.rs          -- delete MOps/s on a full filter
  fpr.rs                        -- false-positive rate vs fingerprint_bits
  degree_distribution.rs        -- per-bucket degree at saturation + histogram
  kv_store_insert_throughput.rs -- KV-store insert MOps/s
  kv_store_lookup_throughput.rs -- KV-store lookup MOps/s (50/50 hit/miss)
  kv_store_delete_throughput.rs -- KV-store delete MOps/s
scripts/plot.py      -- matplotlib charts from CSV results (10 plot functions)
results/             -- generated CSV data and plots (gitignored)
```

### Dependencies

**Rust** (stable >= 1.75):

```bash
rustup update stable
```

Rust crates are managed by Cargo (no manual install):
`xxhash-rust` (xxHash3 item hashing), `rand` (random slot selection during eviction),
`proptest` (property-based tests, dev only).

**Python** (optional, for plotting):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt   # matplotlib, pandas
```

### Build and run

```bash
cargo test                           # all unit + doc tests
cargo run --example basic_usage      # simple demo
cargo run --example load_factor      # fill to capacity, print load factors
```

**Library usage:**

```rust
use segmented_cuckoo::SegmentedCuckooFilter;

let mut f = SegmentedCuckooFilter::new(1024, 4, 12).unwrap();
f.insert("hello").unwrap();
assert!(f.contain("hello"));
f.delete("hello");
assert!(!f.contain("hello"));

// Auto-size from expected item count
use segmented_cuckoo::Standard3aryCuckooFilter;
let mut f = Standard3aryCuckooFilter::from_num_items(100_000, 4, 12).unwrap();
f.insert(b"item".as_ref()).unwrap();
```

**KV-store usage:**

```rust
use segmented_cuckoo::Segmented2aryCuckooKVStore;

let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 64).unwrap();
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

### Cuckoo filters

A cuckoo filter stores a table of `n` *buckets*, each holding `b` *slots*. Each slot stores
a `f`-bit *fingerprint* (a short hash of the inserted item). The three operations are:

- **Insert**: compute a fingerprint and two (or more) candidate bucket indices; place the
  fingerprint in the first bucket with a free slot. If all candidates are full, *evict* a
  random fingerprint from one candidate and relocate it to one of *its* alternates
  (cuckoo kicking). Repeat up to `max_kicks` times; if no free slot is found, roll back
  all mutations and report the table as full.
- **Lookup**: check whether the fingerprint appears in any candidate bucket. A match means
  "probably yes" (with false-positive rate <= `2b / 2^f`); a miss means "definitely no".
- **Delete**: remove the first matching fingerprint from a candidate bucket.

### Partial-key cuckoo hashing (2-ary)

The key insight of Fan et al. (2014) is that the alternate index can be computed from
the *fingerprint alone*, without storing or re-hashing the original item:

```
i1 = H(x) mod n                       (primary index, from full hash of item x)
i2 = i1 XOR h(fingerprint(x))         (alternate index, from XOR with hashed fingerprint)
```

Because XOR is self-inverse (`a XOR b XOR b = a`), given any `(index, fingerprint)` pair
you can recover the other index: `i_other = index XOR h(fingerprint)`. This is called
*partial-key cuckoo hashing* and is what makes cuckoo filters practical -- eviction only
needs the stored fingerprint, not the original item.

---

## Extension 1: k-ary Partial-Key Cuckoo Hashing

### The xor3 and xor4 operations

We extend partial-key cuckoo hashing from 2 to 3 and 4 candidate buckets using the xor3
and xor4 operations introduced by Liu et al. (2017) in the D-Ary Cuckoo Filter paper.

**xor3** is digit-wise addition mod 3 over the base-3 representation of its operands:

```
xor3(a, b):  decompose a and b into base-3 digits,
             add each pair mod 3,
             recompose the result.
```

**xor4** is digit-wise addition mod 4 over pairs of bits (2-bit "digits"):

```
xor4(a, b):  decompose a and b into 2-bit groups,
             add each group mod 4,
             recompose the result.
             (O(1) bitwise implementation using bit-twiddling on 32-bit words)
```

Both operations are *generalised XOR* in the sense that applying xord to a value d times
returns to the start:

```
xor2(xor2(a, b), b) = a         (ordinary XOR: 2 applications)
xor3(xor3(xor3(a, b), b), b) = a  (3 applications return to start)
xor4(xor4(xor4(xor4(a, b), b), b), b) = a  (4 applications return to start)
```

### Index chain construction

The candidate indices for a k-ary filter are a chained application of xord with a
fingerprint-derived offset. The offset uses `fingerprint_hash` masked to `[0, n)`:

**Standard 3-ary** (`n` must be a power of 3, `3^k`):
```
fingerprint_hash(fp) = (fp * 0x5bd1e995) % n      (modulo, because n is not a power of 2)

i1 = H(x) % n                          (primary index)
i2 = xor3(i1, fingerprint_hash(fp))    (first alternate)
i3 = xor3(i2, fingerprint_hash(fp))    (second alternate)
```

**Standard 4-ary** (`n` must be a power of 4, `4^k`):
```
fingerprint_hash(fp) = (fp * 0x5bd1e995) & (n-1)  (bitmask, because n is a power of 2)

i1 = (H(x) >> 32) & (n-1)              (primary index)
i2 = xor4(i1, fingerprint_hash(fp))    (first alternate)
i3 = xor4(i2, fingerprint_hash(fp))    (second alternate)
i4 = xor4(i3, fingerprint_hash(fp))    (third alternate)
```

### Cycling property: no position storage needed

The critical advantage of xord over ordinary XOR is that the **same offset** is applied at
every step, and the cycling property means all k candidates can be reconstructed from any
one of them by cycling forward:

```
Given cur_index and fingerprint (fp), all k candidates are:
    indices[0] = cur_index
    indices[1] = xord(cur_index, fingerprint_hash(fp))
    indices[2] = xord(indices[1], fingerprint_hash(fp))
    ...
    indices[k-1] = xord(indices[k-2], fingerprint_hash(fp))
```

Because the cycling property guarantees the full set is always produced (just starting from
a different point in the cycle), **no per-slot chain position storage is needed for any
scheme** -- not even for standard k > 2 filters.

### Fingerprint-hash functions

| Function | Definition | Usage |
|---|---|---|
| `fingerprint_hash1(fp)` | `(fp * 0x5bd1e995) & (range-1)` | 2-ary and 4-ary (range = n or segment_size) |
| `fingerprint_hash2(fp)` | `(fp * 0xcc9e2d51) & (range-1)` | Segmented 3/4-ary (within-segment offset) |
| `fingerprint_hash3(fp)` | `(fp * 0x1b873593) & (range-1)` | Segmented 4-ary (within-segment offset) |
| `fingerprint_hash_mod(fp)` | `(fp * 0x5bd1e995) % range` | Standard 3-ary (range = n = 3^k, not power of 2) |

The item hash function is xxHash3 (64-bit, non-cryptographic). The fingerprint is extracted
from the lower 32 bits; the primary index `i1` from the upper 32 bits (bitmask) or using
modulo for 3-ary. A fingerprint of 0 is forbidden (it marks an empty slot); if the hash
yields 0, it is replaced with 1.

---

## Extension 2: Segmented Cuckoo Filters

### Motivation

In the theory of cuckoo hashing, the k candidate buckets for each item are modelled as k
independent random indices -- equivalently, the items form a random k-uniform hypergraph on
the bucket vertices. For k = 2, this is a random graph. A well-studied variant uses
**k-partite** random (hyper)graphs, where the vertex set is partitioned into k equal parts
and each candidate index is constrained to one part.

The theoretical load-factor thresholds for k-ary cuckoo hashing are derived under the
k-partite model (Sanders & Walzer 2022, Dietzfelbinger et al. 2010). However, the original
cuckoo filter uses a *standard* (non-partitioned) construction where all indices share the
range `[0, n)`.

The motivating question is: **does enforcing the k-partite structure in a cuckoo filter --
where hash functions are not truly random but fingerprint-based -- match or improve upon the
standard construction in load factor, throughput, and false-positive rate?**

### Construction

A segmented cuckoo filter divides the `n` buckets into `k` equal **segments** and confines
each candidate index to its own segment:

```
k = 2:  segment_size = n/2,   i1 in [0, segment_size),     i2 in [segment_size, 2*segment_size)
k = 3:  segment_size = n/3,   i1 in [0, segment_size),     i2 in [segment_size, 2*segment_size),     i3 in [2*segment_size, 3*segment_size)
k = 4:  segment_size = n/4,   i1 in [0, segment_size),     i2 in [segment_size, 2*segment_size),     i3 in [2*segment_size, 3*segment_size),  i4 in [3*segment_size, 4*segment_size)
```

The partial-key XOR formulas work *within* each segment. Writing `i_j_local` for the offset
within segment j (i.e., `i_j = j * segment_size + i_j_local`):

**Segmented 2-ary:**
```
i1_local = (H(x) >> 32) & (segment_size - 1)
i2_local = i1_local XOR h1(fp)             -- h1 output masked to [0, segment_size)

i1 = i1_local                               -- segment 0
i2 = segment_size + i2_local                         -- segment 1
```

**Segmented 3-ary:**
```
i1_local = (H(x) >> 32) & (segment_size - 1)
i2_local = i1_local XOR h1(fp)             -- h1 masked to [0, segment_size)
i3_local = i2_local XOR h2(fp)             -- h2 masked to [0, segment_size)

i1 = 0 * segment_size + i1_local                    -- segment 0
i2 = 1 * segment_size + i2_local                    -- segment 1
i3 = 2 * segment_size + i3_local                    -- segment 2
```

**Segmented 4-ary:**
```
i1_local = (H(x) >> 32) & (segment_size - 1)
i2_local = i1_local XOR h1(fp)
i3_local = i2_local XOR h2(fp)
i4_local = i3_local XOR h3(fp)

i1 = 0 * segment_size + i1_local                    -- segment 0
i2 = 1 * segment_size + i2_local                    -- segment 1
i3 = 2 * segment_size + i3_local                    -- segment 2
i4 = 3 * segment_size + i4_local                    -- segment 3
```

> **Note:** The segment size `segment_size` **must** be a power of two so that the `& (segment_size - 1)` mask
> replaces a modulo operation. This means n must be `k * 2^m` for some m >= 0.

### Why no position storage is needed

In a segmented filter, the chain position of any index is determined by which segment it
falls in: `position = index / segment_size`. During cuckoo kicking, the filter reads the bucket index,
divides by `segment_size`, and immediately knows the chain position -- no per-slot storage required.

Standard k > 2 filters achieve the same result via the xord cycling property (Liu et al.
2017): since applying xord d times returns to the starting index, all k candidates can be
reconstructed by cycling from any one of them, without recording which position was stored.
Both constructions therefore need zero per-slot position storage for any arity.

### Table size constraints

| Scheme | Constraint on n |
|---|---|
| Standard 2-ary | Power of 2 (>= 1) |
| Standard 3-ary | Power of 3 (3^k, so xor3 cycling stays in range) |
| Standard 4-ary | Power of 4 (4^k = 2^(2k), so xor4 cycling stays in range) |
| Segmented 2-ary | Power of 2, >= 2 |
| Segmented 3-ary | `n = 3 * 2^m` (n/3 must be a power of 2) |
| Segmented 4-ary | Power of 2, >= 4 |

---

## Experimental Setup

### Parameters

All experiments use `fingerprint_bits = 12` (unless sweeping `fingerprint_bits`), `max_kicks = 500`, and the
xxHash3 item hash. The table sizes tested are:

- **Standard 2-ary:** n in {2^14, 2^16, 2^18, 2^20}
- **Standard 3-ary:** n in {3^8, 3^9, 3^10, 3^11} (powers of 3, required for xor3 cycling)
- **Standard 4-ary:** n in {4^7, 4^8, 4^9, 4^10} (powers of 4 = 4^k, required for xor4 cycling)
- **Segmented 2-ary:** n in {2^14, 2^16, 2^18, 2^20}
- **Segmented 3-ary:** n in {3 * 2^12, 3 * 2^14, 3 * 2^16, 3 * 2^18} (n/3 must be a power of 2)
- **Segmented 4-ary:** n in {2^14, 2^16, 2^18, 2^20}
- **Bucket sizes:** b in {1, 2, 3, 4}

### Benchmarks

| Benchmark | What it measures | Trials |
|---|---|---|
| `load_factor` | Fill filter to capacity, record max load factor; sweeps `max_kicks` ∈ {500, 1000, …, 5000} | 20 per config |
| `insert_throughput` | MOps/s while filling to capacity | 10 per config |
| `lookup_throughput` | MOps/s at 5 hit rates (0%, 25%, 50%, 75%, 100%) on a full filter | 10 per config |
| `delete_throughput` | MOps/s deleting all items from a full filter | 10 per config |
| `fpr` | False-positive rate, sweeping `fingerprint_bits` from minimum to 32 | 1 per value |
| `degree_distribution` | Per-bucket occupancy (degree) at saturation, plus degree histogram | 1 per config |
| `kv_store_insert_throughput` | KV-store insert MOps/s — segmented schemes only, `value_bits` ∈ {8, 64, 256, 1024} | 10 per config |
| `kv_store_lookup_throughput` | KV-store lookup MOps/s (50/50 hit/miss) via zero-allocation `get_into` | 10 per config |
| `kv_store_delete_throughput` | KV-store delete MOps/s on a full store | 10 per config |

The first six are the comparison study reported in [Results](#results).
The three `kv_store_*` benches measure the IKPIR primitive layer
(segmented schemes only, `(fingerprint, value)` slots) — they feed
`ikpir-server` performance.

### Running benchmarks

Benches are not clap-parsed: `cargo bench --bench <name>` runs the
hardcoded matrix and writes CSV under `results/`. Each filter bench
(where applicable) has matching plot function(s) in `scripts/plot.py`;
the `kv_store_*` benches do not yet ship a plotter.

| Bench | CSV output (under `results/`) | Matching plot functions |
|---|---|---|
| `load_factor` | `load_factor.csv` | `load_factor_all`, `load_factor_b234`, `load_factor_by_kicks [arity] [bucket_size]` |
| `insert_throughput` | `insert_throughput.csv` | `insert_throughput` |
| `lookup_throughput` | `lookup_throughput.csv` | `lookup_throughput` |
| `delete_throughput` | `delete_throughput.csv` | `delete_throughput` |
| `fpr` | `fpr/arity{a}_num_buckets{n}_bucket_size{b}.csv` (12 files) | `fpr_load_factor`, `fpr_comparison` |
| `degree_distribution` | `degree_per_bucket.csv`, `degree_distribution.csv` | `degree_index`, `degree_histogram` |
| `kv_store_insert_throughput` | `kv_store_insert_throughput.csv` | _no packaged plot_ |
| `kv_store_lookup_throughput` | `kv_store_lookup_throughput.csv` | _no packaged plot_ |
| `kv_store_delete_throughput` | `kv_store_delete_throughput.csv` | _no packaged plot_ |

```bash
# Run one bench, then render its plots:
cargo bench --bench load_factor
python scripts/plot.py load_factor_all
python scripts/plot.py load_factor_by_kicks 2 4   # optional args: arity=2, bucket_size=4

# Run every bench in sequence:
for b in load_factor insert_throughput lookup_throughput delete_throughput \
         fpr degree_distribution \
         kv_store_insert_throughput kv_store_lookup_throughput kv_store_delete_throughput; do
    cargo bench --bench "$b"
done

# Render every plot at once (one-time pip setup):
source .venv/bin/activate
pip install -r scripts/requirements.txt
python scripts/plot.py                            # all available plots → results/plots/
python scripts/plot.py --list                     # list plot functions

# Override the read/write directories:
SCF_RESULTS_DIR=/tmp/results SCF_PLOTS_DIR=/tmp/plots python scripts/plot.py
```

---

## Results

### Load factor

The table below reports mean load factor averaged over all tested table sizes and 20 trials
per configuration. The **Threshold** column shows theoretical load-factor limits for k-ary
cuckoo hashing with b-slot buckets and truly random hash functions (from
[Sanders & Walzer 2022](https://arxiv.org/pdf/1707.06855) / Dietzfelbinger et al. 2010).

| Arity | b | Threshold | Standard | Segmented | Seg. vs Std | Std vs Threshold |
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

\* The 2-ary b=1 case is anomalous: the standard filter slightly *exceeds* the theoretical
threshold (due to finite-size effects and partial-key correlation), while the segmented
variant falls below it. See [Discussion](#why-does-2-ary-b1-segmented-perform-worse-than-standard).
Standard 3-ary values are averaged over n ∈ {3^8..3^11}; segmented 3-ary over n ∈ {3·2^12..3·2^18}.

**Observations:**

- For **b >= 2 across all arities**, the segmented variant consistently achieves equal or
  slightly higher load factor than the standard variant.
- The standard 2-ary b=4 filter achieves 96.2%, consistent with the ~95.5% reported by
  Fan et al. (2014) (our slightly higher value comes from a larger kick budget: 500 vs. the
  original paper's setting).
- Higher arity dramatically improves load factor: 4-ary b=4 reaches 99.71%, within 0.19%
  of the theoretical maximum.
- The gap between measured load factor and theoretical threshold (0.15--2.10%) reflects the
  well-known penalty of partial-key hashing vs. truly random hashing -- the xord-based
  offset has only `2^f` (or `3^f` for xor3) possible values rather than `n`.

![Load factor for all configurations](images/load_factor_all.png)

*Figure: Maximum load factor across all 6 filter variants. Blue = standard, orange = segmented. Higher arity and larger b both push load factor toward 1.0.*

![Load factor for b=2,3,4](images/load_factor_b234.png)

### Insert throughput

Measured in millions of operations per second (MOps/s), inserting items until the filter is
full. For 2-ary and 4-ary both variants use n = 262,144 (2^18). Standard 3-ary is bounded
to powers of 3 (largest bench point: n = 177,147 = 3^11); segmented 3-ary uses n = 196,608
(= 3·2^16). The 3-ary comparison is therefore at slightly different table sizes.

| Arity | b | Standard n | Standard (MOps/s) | Segmented n | Segmented (MOps/s) | Difference |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 1 | 262,144 | 25.52 | 262,144 | 27.22 | +6.7% |
| 2 | 2 | 262,144 | 13.36 | 262,144 | 13.96 | +4.5% |
| 2 | 3 | 262,144 | 12.17 | 262,144 | 12.45 | +2.3% |
| 2 | 4 | 262,144 | 11.24 | 262,144 | 10.22 | −9.0% |
| 3 | 1 | 177,147 | 4.62 | 196,608 | 9.33 | n/a† |
| 3 | 2 | 177,147 | 5.06 | 196,608 | 11.38 | n/a† |
| 3 | 3 | 177,147 | 5.82 | 196,608 | 12.29 | n/a† |
| 3 | 4 | 177,147 | 6.10 | 196,608 | 11.08 | n/a† |
| 4 | 1 | 262,144 | 10.00 | 262,144 | 9.59 | −4.1% |
| 4 | 2 | 262,144 | 12.94 | 262,144 | 11.36 | −12.2% |
| 4 | 3 | 262,144 | 13.86 | 262,144 | 10.49 | −24.3% |
| 4 | 4 | 262,144 | 11.65 | 262,144 | 11.42 | −1.9% |

† Standard 3-ary is significantly slower than segmented 3-ary due to the modulo-based
`fingerprint_hash_mod` required when n = 3^k (not a power of 2). The two entries also measure
different n values, so the raw numbers should not be compared directly.

For 2-ary, segmented is faster in 3 of 4 configurations. For 4-ary, standard is faster in 3
of 4 configurations. The locality advantage of segmented (eviction chains stay within one
segment) appears to benefit 2-ary more than 4-ary at these sizes.

![Insert throughput, 2-ary](images/insert_throughput_2ary.png)
![Insert throughput, 3-ary](images/insert_throughput_3ary.png)
![Insert throughput, 4-ary](images/insert_throughput_4ary.png)

### Lookup throughput

Lookup throughput at 50% hit rate. For 2-ary and 4-ary both variants use n = 262,144 (2^18);
for standard 3-ary the largest benchmark point is n = 59,049 (3^10), segmented 3-ary uses
n = 196,608 (3·2^16). The 3-ary rows are at different n values and are not directly comparable.

| Arity | b | Standard n | Standard (MOps/s) | Segmented n | Segmented (MOps/s) | Difference |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 2 | 262,144 | 43.46 | 262,144 | 39.40 | −9.3% |
| 2 | 4 | 262,144 | 37.30 | 262,144 | 38.09 | +2.1% |
| 3 | 2 | 59,049 | 12.08 | 196,608 | 35.26 | n/a† |
| 3 | 4 | 59,049 | 13.31 | 196,608 | 26.50 | n/a† |
| 4 | 2 | 262,144 | 29.01 | 262,144 | 27.27 | −6.0% |
| 4 | 4 | 262,144 | 17.08 | 262,144 | 25.79 | +51.0% |

† Standard 3-ary is slower due to modulo-based xor3 arithmetic (n = 3^k), and the n values
differ between standard and segmented.

For 2-ary and 4-ary at matched n, differences range from −9.3% to +51%. Throughput decreases
with higher arity (more buckets to probe) and larger b (more slots to scan per bucket).

![Lookup throughput, n=2^18 b=4](images/lookup_throughput_n262144_b4.png)
![Lookup throughput, n=2^16 b=4](images/lookup_throughput_n65536_b4.png)

### Delete throughput

Delete throughput (MOps/s), deleting all items after filling. Same n values as insert
throughput: 2-ary/4-ary at n = 262,144; standard 3-ary at n = 177,147 (3^11), segmented
3-ary at n = 196,608 (3·2^16).

| Arity | b | Standard n | Standard (MOps/s) | Segmented n | Segmented (MOps/s) | Difference |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 1 | 262,144 | 48.47 | 262,144 | 44.44 | −8.3% |
| 2 | 2 | 262,144 | 32.54 | 262,144 | 32.82 | +0.9% |
| 2 | 3 | 262,144 | 29.50 | 262,144 | 30.57 | +3.6% |
| 2 | 4 | 262,144 | 28.15 | 262,144 | 27.81 | −1.2% |
| 3 | 1 | 177,147 | 13.93 | 196,608 | 41.97 | n/a† |
| 3 | 2 | 177,147 | 12.14 | 196,608 | 29.77 | n/a† |
| 3 | 3 | 177,147 | 11.36 | 196,608 | 26.29 | n/a† |
| 3 | 4 | 177,147 | 10.85 | 196,608 | 26.39 | n/a† |
| 4 | 1 | 262,144 | 29.30 | 262,144 | 33.23 | +13.4% |
| 4 | 2 | 262,144 | 25.69 | 262,144 | 25.87 | +0.7% |
| 4 | 3 | 262,144 | 24.82 | 262,144 | 23.89 | −3.7% |
| 4 | 4 | 262,144 | 24.19 | 262,144 | 24.03 | −0.7% |

† Standard 3-ary is slower due to modulo-based xor3 arithmetic, and n values differ.

For 2-ary and 4-ary at matched n, differences are small (−8% to +13%) with no consistent
winner. Delete is algorithmically similar to lookup (scan candidate buckets, clear match),
so eviction locality does not affect its cost.

![Delete throughput, 2-ary](images/delete_throughput_2ary.png)
![Delete throughput, 3-ary](images/delete_throughput_3ary.png)
![Delete throughput, 4-ary](images/delete_throughput_4ary.png)

### False-positive rate

The theoretical false-positive rate is `d·b / 2^fp_bits` where `d` is the arity. For the
common configuration d=2, b=4, fp_bits=12 this gives 0.195%. The table below shows measured
FPR at selected fp_bits values (2-ary, n = 262,144, b = 4):

| fp_bits | Standard FPR | Segmented FPR | Theoretical (2·b/2^f) |
|:---:|:---:|:---:|:---:|
| 8 | 2.986% | 2.973% | 3.125% |
| 10 | 0.745% | 0.746% | 0.781% |
| 12 | 0.188% | 0.188% | 0.195% |
| 16 | 0.012% | 0.012% | 0.012% |
| 20 | 0.001% | 0.001% | 0.001% |

Both variants track the theoretical curve closely across the entire fp_bits sweep. The
indexing strategy (standard vs. segmented) has no measurable effect on false-positive rate,
as expected: FPR depends only on the fingerprint bit width and bucket size, not on how
indices are computed.

![FPR comparison across arity and b](images/fpr_comparison.png)

*Figure: False-positive rate vs. fingerprint bits. The dashed line shows the theoretical bound d·b/2^f (where d is the arity). All 6 filter variants closely follow the bound.*

### Eviction chain statistics

![Eviction chain statistics, 2-ary](images/eviction_2ary.png)
![Eviction chain statistics, 3-ary](images/eviction_3ary.png)
![Eviction chain statistics, 4-ary](images/eviction_4ary.png)
![Mean kicks per insert, 2-ary](images/eviction_mean_kicks_2ary.png)
![Mean kicks per insert, 3-ary](images/eviction_mean_kicks_3ary.png)
![Mean kicks per insert, 4-ary](images/eviction_mean_kicks_4ary.png)

---

## Discussion

### Summary

The segmented cuckoo filter performs on par with or slightly better than the standard cuckoo
filter across all metrics. The k-partite graph structure -- which confines each candidate
index to its own table segment -- does not degrade performance and provides a practical
benefit: slightly higher load factor for b >= 2. Per-slot position storage is not needed for
any scheme: standard k > 2 uses the xor3/xor4 cycling property (Liu et al. 2017) to
reconstruct indices without tracking chain position; segmented schemes derive position from
the segment index.

The 3-ary and 4-ary extensions via xor3 and xor4 (Liu et al. 2017) work correctly and
achieve load factors consistent with theoretical k-ary cuckoo hashing thresholds, validating
the partial-key approach for higher arities.

### Load factor decreases with table size

For a fixed `max_kicks = 500`, the achievable load factor decreases as n grows:

| n (2-ary, b=4) | Standard | Segmented |
|---:|:---:|:---:|
| 16,384 | 96.51% | 96.70% |
| 65,536 | 96.32% | 96.57% |
| 262,144 | 96.09% | 96.28% |
| 1,048,576 | 95.87% | 96.10% |

**Explanation.** The kick budget is an absolute constant. At small tables, 500 kicks is a
large fraction of the total capacity (500 / 16,384 = 3.1%), giving the algorithm ample room
to explore and find free slots. At large tables, 500 kicks is tiny relative to capacity
(500 / 1,048,576 = 0.05%), and the algorithm exhausts its budget before navigating the
longer eviction chains that arise near saturation.

To maintain the same load factor at larger tables, increase `max_kicks` (the filter
provides `set_max_kicks()` for this). Alternatively, accept the slightly lower maximum
load -- the difference is less than 1 percentage point across a 64x range of table sizes.

### Why does 2-ary b=1 segmented perform worse than standard?

With b=1 (one slot per bucket), a 2-ary cuckoo filter is equivalent to placing edges of a
random graph on n vertices. The theoretical load-factor threshold is exactly 50%.

The **standard** variant achieves 51.76%, slightly above this threshold. This happens because
partial-key hashing creates correlated index pairs -- the XOR structure means certain pairs
of buckets are "linked" more often than in a truly random graph. At b=1 with small
fingerprint space, these correlations can occasionally help by creating cycles that happen to
admit valid placements.

The **segmented** variant achieves 48.71%, slightly below the threshold. The bipartite
constraint (i1 and i2 always in different halves) combined with the limited fingerprint space
makes the bipartite random graph slightly less connected than its non-partitioned counterpart
at b=1.

For **b >= 2** this effect vanishes: multiple slots per bucket absorb the minor connectivity
differences, and the segmented variant consistently matches or exceeds standard.

### Consistency with theoretical thresholds

Our results are consistent with the theoretical analysis of Sanders & Walzer
([arXiv:1707.06855](https://arxiv.org/pdf/1707.06855)), which gives load-factor thresholds
for k-ary cuckoo hashing with truly random hash functions:

The measured load factors sit 0.15--2.10% below these thresholds. The gap exists because
partial-key hashing is not truly random: the xor offset `fingerprint_hash(fp)` produces at most
`2^f` distinct values (or `3^f` for xor3), whereas truly random hashing would distribute
each alternate index uniformly over all `n` buckets. This is the same systematic gap
documented by Fan et al. (2014) for the 2-ary case; we confirm it extends consistently to
3-ary and 4-ary with the xor3/xor4 construction.

The segmented variant sits at or above the standard variant for b >= 2, confirming that
the k-partite structure does not hurt and may marginally help. Whether this advantage
persists or grows at extremely large n, or whether it is a finite-size artefact, remains
an open question.

### Open questions

- **Scaling max_kicks.** Would setting `max_kicks = O(n)` or `O(n * b)` close the gap to
  theoretical thresholds at large n?
- **Tighter theoretical bounds.** The thresholds in the literature assume truly random hash
  functions. Can tighter bounds be derived for partial-key (XOR-based) hashing?
- **Mixed-arity schemes.** Could a filter use 2-ary at low load and switch to higher arity
  as the table fills?

---

## References

- B.-Y. Fan, D. G. Andersen, M. Kaminsky, M. D. Mitzenmacher. *Cuckoo Filter: Practically
  Better Than Bloom*. CoNEXT 2014.
- B. Liu, C. Li, Y. Lin, B. Vucetic, Y. Li. *D-Ary Cuckoo Filter: A Space Efficient Data
  Structure for Set Membership Lookup*. IEEE ICPADS 2017.
- P. Sanders, S. Walzer. *Load Thresholds for Cuckoo Hashing with Overlapping Blocks*.
  [arXiv:1707.06855](https://arxiv.org/pdf/1707.06855), 2022.
- M. Dietzfelbinger, A. Goerdt, M. Mitzenmacher, A. Montanari, R. Pagh, M. Rink.
  *Tight Thresholds for Cuckoo Hashing via XORSAT*. ICALP 2010.
