# Benchmark Documentation

All benchmarks are standalone binaries in `benches/`. Each writes CSV results to `results/`
and an ASCII summary to stdout. Run with `cargo bench --bench <name>`.

Common constants across all benchmarks: `MAX_KICKS = 500`, `FP_BITS = 12` (except `fpr`
which sweeps fp_bits).

---

## helpers.rs

Shared utility. Provides `csv_writer(filename, header)` which creates a `BufWriter` for
`results/<filename>`, creating any necessary parent directories. All other benchmarks call
this to open their CSV output.

---

## load_factor

**Intent:** Measure how full each filter scheme can get before it rejects insertions (table
full). This is the fundamental capacity comparison between standard and segmented indexing.

**Method:** For each (scheme, n, b) configuration, insert sequential u64 keys in LE-byte
form until `add` returns `TableFull`. Record the load factor at that point. Repeat 20 trials
per config to get a stable distribution (load factor has trial-to-trial variance because the
kicking strategy is randomized).

**Rationale:** Load factor is a prerequisite for interpreting all other benchmarks — a scheme
that achieves higher load means each slot is used more efficiently. Running 20 trials captures
variance in the random eviction chain, giving min/mean/max rather than a single noisy sample.

**Parameters:** n ∈ {2^14, 2^16, 2^18, 2^20} for all schemes except segmented 3-ary (which
requires n = 3·2^m), where n ∈ {3·2^12, 3·2^14, 3·2^16, 3·2^18}. b ∈ {1, 2, 3, 4}.

**Output:** `results/load_factor.csv`
Columns: `scheme, arity, n, b, fp_bits, trial, load_factor`

**Plotted by:** `plot_load_factor_all`, `plot_load_factor_b234`

---

## insert_throughput

**Intent:** Measure raw insert speed for each scheme, at the load factor each scheme naturally
reaches.

**Method:** For each (scheme, n, b), insert sequential keys until `TableFull` and time the
entire loop. Divide total insertions by elapsed time to get Mops/sec. Run 3 warmup trials
(JIT, CPU cache warm-up) then 10 timed trials. Report mean ± std dev.

**Rationale:** "Insert until full" is the only fair comparison method. If we inserted a fixed
number of items (e.g., 1M), we would conflate throughput with load factor: 1M items might fill
standard 3-ary to ~90% load (near full, expensive kicking) but segmented 3-ary to only ~70%
(sparse, cheap kicking), making standard appear slower purely because it was under more stress.
By measuring throughput over the full fill trajectory — from empty to capacity — each scheme is
evaluated under comparable conditions across its entire operating range.

**Parameters:** n ∈ {2^16, 2^18, 2^20}. b ∈ {1, 2, 3, 4}. 3 warmup + 10 measured trials.

**Output:** `results/insert_throughput.csv`
Columns: `scheme, arity, n, b, trial, inserted, load_factor, duration_ns, throughput_mops`

**Plotted by:** `plot_insert_throughput` (3 figures, one per arity)

---

## delete_throughput

**Intent:** Measure delete speed on a filter that is already at capacity.

**Method:** First, do a single fill run to determine the exact number of items the filter
accepts (`count`). Then for each trial: fill with exactly `count` items (untimed), start the
timer, delete all `count` items, stop the timer. Repeat 3 warmup + 10 measured trials. Only
the deletion loop is timed; the fill phase is excluded.

**Rationale:** Timing only the delete loop (not fill + delete) gives a clean signal for how
fast the hash-and-probe delete path is, independent of insert behavior. Using a fixed `count`
(determined once before trials begin) ensures every trial deletes the same number of items,
making throughput numbers directly comparable across trials. Items are deleted in the same
sequential order they were inserted, so the working set matches the filter's actual contents.

Note: A small fraction of deletes may return `NotFound` due to fingerprint collisions (two
items sharing the same tag and candidate buckets). This is expected and does not invalidate the
measurement — the full delete code path is still executed.

**Parameters:** n ∈ {2^16, 2^18, 2^20}. b ∈ {1, 2, 3, 4}. 3 warmup + 10 measured trials.

**Output:** `results/delete_throughput.csv`
Columns: `scheme, arity, n, b, trial, deleted, load_factor, duration_ns, throughput_mops`

**Plotted by:** `plot_delete_throughput` (3 figures, one per arity)

---

## lookup_throughput

**Intent:** Measure query speed at varying hit rates, on a full filter.

**Method:** Fill the filter until full. Then for each hit rate in {0%, 25%, 50%, 75%, 100%}:
construct a query set of `q = n*b/2` keys, mixing inserted keys (hits) and never-inserted keys
(misses) in the desired ratio. Time `q` `contain` calls. Run 3 warmup + 10 measured trials
per hit rate. A fixed query count `q = n*b/2` (half the filter capacity) is used to keep
runtime predictable without being trivially small.

**Rationale:** Hit rate matters because a lookup that finds a matching tag terminates early
(first matching bucket), while a miss must check all k buckets. The 5-point sweep (0%..100%)
reveals whether this early-exit benefit is significant. Testing at full load ensures the
filter is in a realistic operating state — the FPR is meaningful and cache pressure is real.

**Parameters:** n ∈ {2^16, 2^18}. b ∈ {2, 4} (b=1 and b=3 omitted to reduce runtime).
3 warmup + 10 measured trials per hit rate.

**Output:** `results/lookup_throughput.csv`
Columns: `scheme, arity, n, b, hit_rate_pct, trial, load_factor, num_queries, duration_ns, throughput_mops`

**Plotted by:** `plot_lookup_throughput` (one figure per (n, b))

---

## fpr

**Intent:** Measure the actual false positive rate (FPR) as fingerprint bit width varies, and
compare it against the theoretical bound `2b / 2^fp_bits`.

**Method:** For each (arity, n, b), sweep `fp_bits` from `min_fp_bits(b)` up to 32 (where
`min_fp_bits(b) = floor(log2(2b)) + 1` is the minimum that makes FPR < 1). At each fp_bits:
insert until full, then query `q = 10·n·b` items that were never inserted. Count how many
return `true` (false positives). Also record the theoretical FPR `2b / 2^fp_bits`.

Separate CSVs are created per (arity, n, b) because the arity changes both the filter type
and (for segmented 3-ary) the n value.

**Rationale:** The standard theoretical FPR for a cuckoo filter is `2b / 2^f`. This bench
verifies whether the segmented variant achieves the same FPR — theoretically it should, since
segmentation affects index placement but not fingerprint collision probability. Using
`q = 10·n·b` non-inserted queries gives at least 10× the filter capacity in miss lookups,
enough for ~1000+ expected false positives even at low FPR, which keeps the measured rate
statistically meaningful.

**Parameters:** n = 2^18. b ∈ {1, 2, 3, 4}. fp_bits swept from min to 32. Single run per
config (no trials — sweep runtime is already long).

**Output:** `results/fpr/arity{a}_n{n}_b{b}.csv` (one file per arity per (n, b))
Columns: `fp_bits, scheme, n, load_factor, num_inserted, num_queries, false_positives, fpr_pct, theoretical_pct`

**Plotted by:** `plot_fpr_load_factor` (load factor vs fp_bits), `plot_fpr_comparison` (FPR
vs fp_bits with theoretical line)

---

## eviction

**Intent:** Understand the eviction chain length distribution — how often an insert is a
direct placement (0 kicks) vs. requiring many displacements.

**Method:** Use `add_with_stats` instead of `add` to get per-insertion kick counts. For each
(scheme, n, b), insert until full, accumulating kick counts into 5 histogram buckets: 0,
1–10, 11–50, 51–100, 101–500. Also track total kicks, direct placements, and mean kicks per
insertion.

**Rationale:** `add_with_stats` is a separate code path (not delegating to `add`) that returns
`InsertStats { kicks: u32 }`, intentionally keeping the hot `add` path free of stats overhead.
The histogram buckets are chosen to reveal whether high-kick insertions are rare tail events or
a significant fraction of all insertions. Eviction behavior is expected to worsen as the filter
fills, so collecting stats over the full fill trajectory captures the entire distribution.

b=1 is excluded because with only 1 slot per bucket, every collision immediately triggers
kicking, making the histogram less informative — load factor with b=1 is also lower, so
including it would add noise without insight.

**Parameters:** n ∈ {2^14, 2^16, 2^18, 2^20}. b ∈ {2, 3, 4}. Single trial per config.

**Output:** `results/eviction.csv`
Columns: `scheme, arity, n, b, fp_bits, total_inserts, total_kicks, direct_placements, max_kicks, mean_kicks, hist_0, hist_1_10, hist_11_50, hist_51_100, hist_101_500`

**Plotted by:** `plot_eviction` (stacked bar of normalised histogram per arity),
`plot_eviction_mean_kicks` (mean kicks vs n per arity)

---

## degree_distribution

**Intent:** Analyze how many items are mapped to each bucket (bucket "degree") as the filter
fills, revealing load balance between buckets and any spatial structure introduced by
segmented indexing.

**Method:** Intercept the item hash via the `IndexScheme` trait directly (calling
`scheme.hash_item()` before passing the item to `filter.add()`). For each inserted item,
increment the degree counter for all k candidate bucket indices. Items that trigger
`TableFull` are not counted.

Two outputs are produced in one run:

- **Part 1 — degree-index**: Writes the degree of every bucket vs its index. Configured by
  `DI_ARITY` and `DI_N` constants (default: arity=2, n=65536). Change these constants to
  inspect other arities. Shows whether segmented schemes create degree "bands" across the
  index space.

- **Part 2 — histogram**: For all 6 schemes at fixed n (65536 for arity=2/4; 49152 for
  segmented 3-ary), writes the count of buckets with each degree value. Shows the shape of the
  degree distribution (Poisson-like for standard, possibly different for segmented).

**Rationale:** Bucket degree is a proxy for how balanced the hash function distributes items.
A scheme where some buckets have very high degree will exhaust those buckets first and force
more evictions. The segmented constraint (each index in its own segment) may create more
uniform degree within each segment compared to standard hashing.

**Parameters (degree-index):** DI_ARITY=2 (default), DI_N=2^16=65536. b ∈ {1, 2, 3, 4}.
**Parameters (histogram):** All arities. DH_N=2^16=65536 (seg3: 3·2^14=49152). b ∈ {1, 2, 3, 4}.

**Output:**
- `results/degree_per_bucket.csv` — columns: `scheme, arity, n, b, bucket_index, degree`
- `results/degree_distribution.csv` — columns: `scheme, arity, n, b, degree, count`

**Plotted by:** `plot_degree_index` (scatter: degree vs bucket index),
`plot_degree_histogram` (line: fraction of buckets vs degree)
