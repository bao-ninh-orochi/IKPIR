# Plot Documentation

`scripts/plot.py` reads CSV files from `results/` and generates PNG charts in
`results/plots/`. Run with `python scripts/plot.py` (requires a venv with matplotlib and
pandas — see `scripts/requirements.txt`).

The script only generates plots for benchmarks whose output CSV files already exist, so it is
safe to run after any subset of benchmarks.

**Colour convention:** blue (`#1f77b4`) = standard scheme, orange (`#ff7f0e`) = segmented scheme.
**Marker convention:** diamond = b=1, square = b=2, circle = b=3, triangle = b=4.

---

## Load Factor

### `plot_load_factor_all()`

- **Input:** `results/load_factor.csv`
- **What:** 3-subplot figure (one per arity). Each subplot plots mean load factor vs n for
  every (scheme, b) combination, with b encoded as marker shape and scheme as colour.
- **Output:** `results/plots/load_factor_all.png`

### `plot_load_factor_b234()`

- **Input:** `results/load_factor.csv`
- **What:** Same 3-subplot layout, but arity=2 shows only b=2,3,4 (b=1 is excluded for
  clarity since 1-slot buckets behave very differently). Arity=3 and 4 show all b values.
- **Output:** `results/plots/load_factor_b234.png`

---

## Insert Throughput

### `plot_insert_throughput()`

- **Input:** `results/insert_throughput.csv`
- **What:** 3 separate figures (one per arity). Each figure has 4 subplots (one per b value).
  Within each subplot, bars are grouped by n; each n group has two adjacent bars — segmented
  (orange) and standard (blue). Y-axis is mean Mops/sec across trials.
- **Output:** `results/plots/insert_throughput_2ary.png`, `insert_throughput_3ary.png`,
  `insert_throughput_4ary.png`

---

## Delete Throughput

### `plot_delete_throughput()`

- **Input:** `results/delete_throughput.csv`
- **What:** Identical layout to `plot_insert_throughput` — 3 figures × 4 subplots (per b),
  bars by n with segmented/standard pairs. Y-axis is mean Mops/sec for the deletion loop only.
- **Output:** `results/plots/delete_throughput_2ary.png`, `delete_throughput_3ary.png`,
  `delete_throughput_4ary.png`

---

## Lookup Throughput

### `plot_lookup_throughput()`

- **Input:** `results/lookup_throughput.csv`
- **What:** One figure per (n, b) pair. X-axis has 6 scheme groups (segmented 2-ary, standard
  2-ary, segmented 3-ary, standard 3-ary, segmented 4-ary, standard 4-ary). Within each group,
  5 side-by-side bars show throughput at hit rates 0%, 25%, 50%, 75%, 100%. Segmented 3-ary
  may have a slightly different n (nearest valid 3·2^m) and its label shows the actual n used.
- **Output:** `results/plots/lookup_throughput_n{n}_b{b}.png` (one per (n, b))

---

## False Positive Rate

### `plot_fpr_load_factor()`

- **Input:** `results/fpr/arity{a}_n{n}_b{b}.csv` (one file per config)
- **What:** Per CSV file, one figure showing load factor (y) vs fingerprint bits (x) for
  standard and segmented. Reveals whether FPR controls (wider fingerprints) also affect how
  full the filter can get.
- **Output:** `results/plots/fpr_lf_arity{a}_n{n}_b{b}.png` (one per config)

### `plot_fpr_comparison()`

- **Input:** `results/fpr/arity{a}_n{n}_b{b}.csv` — automatically selects the (n, b) pair
  with the largest capacity ≥ 1M (i.e., n·b ≥ 1,000,000) to ensure the FPR estimate is based
  on enough queries.
- **What:** 3-subplot figure (one per arity). Each subplot shows FPR% (log y-scale) vs
  fingerprint bits for standard and segmented, plus a dashed black line for the theoretical
  bound `2b / 2^fp_bits`. Shows how closely each scheme tracks the theoretical FPR.
- **Output:** `results/plots/fpr_comparison.png`

---

## Eviction

### `plot_eviction()`

- **Input:** `results/eviction.csv`
- **What:** 3 separate figures (one per arity). Each figure has subplots for each b value.
  Within each subplot, bars are grouped by (n, scheme) and stacked into 5 kick-count ranges:
  0, 1–10, 11–50, 51–100, 101–500, normalised to 100% of insertions. Shows what fraction of
  inserts required each level of eviction work.
- **Output:** `results/plots/eviction_2ary.png`, `eviction_3ary.png`, `eviction_4ary.png`

### `plot_eviction_mean_kicks()`

- **Input:** `results/eviction.csv`
- **What:** 3 separate figures (one per arity). Each figure is a line plot of mean kicks per
  insertion (y) vs n (log-scale x), with one line per (scheme, b) combination. Shows how
  eviction cost scales with table size.
- **Output:** `results/plots/eviction_mean_kicks_2ary.png`, `eviction_mean_kicks_3ary.png`,
  `eviction_mean_kicks_4ary.png`

---

## Degree Distribution

### `plot_degree_index()`

- **Input:** `results/degree_per_bucket.csv`
- **What:** For each (arity, n, b) in the CSV, one figure with 2 side-by-side scatter plots
  (standard left, segmented right). X-axis is bucket index, y-axis is degree. Downsampled to
  max 4000 points for readability. Reveals spatial patterns — segmented schemes show clear
  degree bands at segment boundaries.
- **Output:** `results/plots/degree_distribution/degree_{arity}ary_n{n}_b{b}.png`

### `plot_degree_histogram()`

- **Input:** `results/degree_distribution.csv`
- **What:** For each (arity, b) combination, one line plot where x-axis is bucket degree and
  y-axis is the fraction of all buckets with that degree. Compares the degree distribution
  shape between standard and segmented — standard follows a Poisson-like distribution; segmented
  may concentrate degrees differently.
- **Output:** `results/plots/degree_hist_{arity}ary_b{b}.png`
