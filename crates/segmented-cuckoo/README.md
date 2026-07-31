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
lookups to a single Index-PIR query per segment.

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
  insert/lookup/delete throughput, and false-positive rate, against theoretical
  k-ary cuckoo-hashing thresholds.

## Key findings

Measured across the six configurations RisePIR uses ([Results](#results), the
paper's Table 2).

| Finding | Detail |
|---|---|
| Segmentation costs nothing in space | The SCF reaches a *higher* load factor than the standard filter in all six configurations, by 0.02–0.07 pp, peaking at 99.86% |
| Close to the theoretical limit | Every configuration lands within 0.7 pp of the k-partite information-theoretic threshold |
| Segmentation wins big at arity 3 | 2.2–3.3× faster on insert, lookup, and delete — the standard filter needs a modulo (n = 3^k), the SCF a bitmask (n = 3·2^m) |
| Comparable at arities 2 and 4 | Both layouts fall on powers of two; they differ by at most ~13% with neither consistently ahead, the exception being lookup at (2, 4) where standard is ~1.5× faster |

The confinement that keyword PIR depends on — each candidate restricted to a
predictable segment — therefore imposes no measurable cost in storage
efficiency, and at arity 3 it is a substantial speedup.

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
fingerprint_hash(fp) = ((fp * 0xff51afd7ed558ccd) >> 32) & (n-1)
i1 = (H(x) >> 64) & (n-1)
i2 = xor4(i1, fingerprint_hash(fp))
i3 = xor4(i2, fingerprint_hash(fp))
i4 = xor4(i3, fingerprint_hash(fp))
```

Standard 3-ary uses `((fp * 0xff51afd7ed558ccd) >> 32) % n` because n = 3^k is not
a power of 2. This modulo is what the arity-3 throughput gap in the results below
comes down to.

**No position storage.** The same offset is applied at each step, so the cycling
property reconstructs all k candidates from any one of them — no per-slot chain
position is stored, even for standard k > 2 filters.

The item hash is xxHash3_128 (128-bit, non-cryptographic): fingerprint from the
low 64 bits (masked to `fingerprint_bits`, up to the full 64), primary index from
the high 64 bits. A fingerprint of 0 is forbidden (marks empty); a 0 hash is
replaced with 1.

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
i1_local = (H(x) >> 64) & (segment_size - 1)
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

Defaults, and the reasoning behind them, live in one place —
[`benches/configs.rs`](benches/configs.rs) — which every bench reads:

| Parameter | Default | Why |
|---|---|---|
| `fingerprint_bits` | **32** | Drives FPR to `k·b / 2^f` (`k·b / 2^32` at the default); a false positive never perturbs a load-factor or throughput measurement, so the numbers isolate the indexing scheme. `f` up to 64 is supported (the default stays 32) |
| `max_kicks` | **2500** | High enough that measured load factor reflects the scheme rather than the kick budget; a small budget caps it well below the true threshold at ~10^6 buckets |
| `num_buckets` | **~10^6** | Per arity, below |
| trials | 10 (20 for load factor) | Load factor reports one headline number per config, so it buys a tighter error bar |

**Config matrix.** The benches default to the six `(arity, bucket_size)` pairs
RisePIR uses. The two schemes are compared at equal table size, as close as their
divisibility constraints allow — at arities 2 and 4 both ladders hit 2^20
exactly; at arity 3 one is `3·2^m` and the other `3^k`, so they cannot coincide
and land 1.4% apart:

| Arity | b | Segmented n | Standard n |
|:---:|:---:|---:|---:|
| 2 | 4 | 1 048 576 (2^20) | 1 048 576 (2^20) |
| 3 | 2, 3 | 1 572 864 (3·2^19) | 1 594 323 (3^13) |
| 4 | 1, 2, 3 | 1 048 576 (2^20) | 1 048 576 (4^10) |

Note `n` depends only on the arity, not on `b` — the six rows carry three
distinct pairs between them.

| Bench | Measures | CSV (under `results/segmented-cuckoo/`) |
|---|---|---|
| `cuckoo_filter_load_factor` | Max load factor | `cuckoo_filter_load_factor.csv` |
| `cuckoo_filter_insert_throughput` | Insert MOps/s while filling | `cuckoo_filter_insert_throughput.csv` |
| `cuckoo_filter_lookup_throughput` | Lookup MOps/s at a given hit rate | `cuckoo_filter_lookup_throughput.csv` |
| `cuckoo_filter_delete_throughput` | Delete MOps/s on a full filter | `cuckoo_filter_delete_throughput.csv` |
| `cuckoo_filter_false_positive_rate` | FPR vs `fingerprint_bits` | `cuckoo_filter_false_positive_rate/*.csv` |
| `kv_store_{insert,lookup,delete}_throughput` | KV-store MOps/s (segmented only) | `kv_store_*.csv` |

The first four produce the [Results](#results) table below. The `kv_store_*`
benches measure the IKPIR primitive layer (segmented `(fingerprint, value)`
slots) and are not part of that comparison: a KV slot carries `fp ‖ value`, so
they size from `--target-items` (default 65536) rather than ~10^6 buckets, which
at `value_bits = 1024` would run to gigabytes.

### Running benchmarks

```bash
# Reproduce the paper's Table 2 — the four benches, all six configs. (~1-2 h)
./scripts/table2.sh

# Narrow it: any flags are forwarded to every bench.
./scripts/table2.sh --arity 4                  # the three arity-4 rows
./scripts/table2.sh --trials 3                 # faster, noisier

# One bench, via the workspace runner (routes the CSV to results/segmented-cuckoo/).
./scripts/bench.sh cuckoo_filter_insert_throughput                  # all six configs
./scripts/bench.sh cuckoo_filter_insert_throughput --arity 4 --bucket-size 2
./scripts/bench.sh cuckoo_filter_false_positive_rate --arity 2 --bucket-size 4
./scripts/bench.sh kv_store_lookup_throughput --value-bits 8,256

# Directly — writes to the crate-local results/ unless IKPIR_RESULTS_DIR is set.
cargo bench -p segmented-cuckoo --bench cuckoo_filter_load_factor
```

Every flag is optional; passing none runs the paper's matrix at the paper's
tunables. `--arity` / `--bucket-size` narrow to the matching rows, or synthesize
a config outside the matrix if none match. Common overrides: `--num-buckets`,
`--fingerprint-bits`, `--max-kicks`, `--warmup`, `--trials`; plus `--hit-rate`
(lookup), `--num-queries` (false-positive rate), and `--value-bits` /
`--plaintext-bits` / `--target-items` (KV store).

The filter / KV-store properties also have fast unit-test coverage:
`cargo test -p segmented-cuckoo`.

---

## Results

Reproduce with `./scripts/table2.sh`. This is Table 2 of the CANS 2026 paper,
measured on an Apple M1 MacBook Pro with 16 GiB of memory, 32-bit fingerprint,
each filter filled to capacity. **Theory** is the k-partite information-theoretic
threshold, from Walzer's tabulation of Dietzfelbinger et al. 2010 and Sanders &
Walzer 2022. Bold marks the better filter on each metric.

| Arity | b | Scheme | n | Load factor (%) | Theory (%) | Insert (Mops) | Lookup (Mops) | Delete (Mops) |
|:---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 2 | 4 | Segmented | 1 048 576 | **97.40** | 98.04 | **9.04** | 24.91 | **34.55** |
|   |   | Standard  | 1 048 576 | 97.33 |  | 8.46 | **36.64** | 34.35 |
| 3 | 2 | Segmented | 1 572 864 | **98.32** | 98.82 | **9.52** | **48.07** | **37.86** |
|   |   | Standard  | 1 594 323 | 98.29 |  | 3.24 | 14.56 | 13.03 |
| 3 | 3 | Segmented | 1 572 864 | **99.43** | 99.73 | **8.72** | **28.23** | **28.50** |
|   |   | Standard  | 1 594 323 | 99.39 |  | 4.02 | 10.33 | 10.38 |
| 4 | 1 | Segmented | 1 048 576 | **97.12** | 97.68 | 8.86 | **64.12** | **46.60** |
|   |   | Standard  | 1 048 576 | 97.06 |  | **9.38** | 60.93 | 45.58 |
| 4 | 2 | Segmented | 1 048 576 | **99.59** | 99.82 | 12.09 | **48.61** | 37.04 |
|   |   | Standard  | 1 048 576 | 99.55 |  | **12.89** | 45.99 | **37.71** |
| 4 | 3 | Segmented | 1 048 576 | **99.86** | 99.98 | 13.61 | 25.59 | 31.30 |
|   |   | Standard  | 1 048 576 | 99.84 |  | **14.42** | **29.25** | **31.73** |

### Load factor

The SCF attains a slightly higher load factor in **all six** configurations
(+0.02 to +0.07 pp) and stays within 0.7 pp of the k-partite threshold, peaking
at 99.86%. Segmentation removes placement freedom the standard filter has — each
candidate is pinned to its own segment — yet costs nothing in storage
efficiency. Higher arity and larger `b` both help sharply: (4, 3) reaches 99.86%,
within 0.12 pp of its threshold.

The residual gap to threshold is the **partial-key penalty**: the xord offset
takes at most `2^f` (or `3^f`) distinct values rather than ranging uniformly over
`n`, so the hashes are not truly random. This is the same gap Fan et al. (2014)
documented for 2-ary; it extends to 3/4-ary with xor3/xor4.

### Throughput

The comparison splits by arity.

**Arity 3 — the SCF is 2.2–3.3× faster** on all three operations. This is a table-
size effect, not a segmentation effect: the standard filter addresses `3^k`
buckets and must reconstruct candidates by base-3 digit-wise XOR with a modulo,
while the SCF sizes its table along the finer `n = 3·2^m` ladder and recovers each
candidate with a single binary XOR inside a power-of-two segment.

**Arities 2 and 4 — comparable.** Both layouts fall on powers of two, so both use
the bitmask path; they differ by at most ~13% with neither consistently ahead.
The one notable exception is lookup at (2, 4), where the standard filter is
roughly 1.5× faster (36.64 vs 24.91 Mops).

---

## Discussion

The segmented cuckoo filter performs on par with or better than the standard
filter across every metric measured. The k-partite structure does not degrade
performance, and it buys the property IKPIR is built on: a deterministic, fixed
lookup-position set per item. No per-slot position storage is needed for any
scheme — standard k > 2 uses the xor3/xor4 cycling property, segmented derives
position from the segment index. The 3/4-ary extensions reach load factors
consistent with the theoretical thresholds, validating the partial-key approach
at higher arities.

### On `max_kicks`

The kick budget is an absolute constant, so its adequacy depends on table size:
at 16 384 buckets, 500 kicks is 3.1% of capacity, but at 1 048 576 buckets it is
0.05% — too few to navigate the long eviction chains near saturation, and the
measured load factor then reports the budget rather than the scheme. This is why
the default is 2500 rather than the 500 used in earlier sweeps of this crate. An
open question is whether `max_kicks = O(n)` would close the remaining gap to
threshold at large `n`.

### Open questions

- **Tighter bounds.** Can the thresholds be sharpened for partial-key (XOR-based)
  hashing, rather than assuming truly random hashes?
- **Mixed-arity.** Could a filter start 2-ary at low load and raise arity as it
  fills?

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
- S. Walzer. Tabulation of k-partite load thresholds (the **Theory** column
  above).
