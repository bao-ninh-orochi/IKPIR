# CSV schemas

Canonical column definitions for every file in `results/paper/`. Reviewers
who want to audit the numbers can cross-reference `scripts/plot.py` and
the per-bench module-level doc comments.

Types: `s`=string, `u32`=unsigned int, `f64`=floating point, `ns`=nanoseconds.

---

## `scf/load_factor.csv`

| Column      | Type | Description |
|-------------|------|-------------|
| scheme      | s    | `standard` or `segmented` |
| arity       | u32  | 2, 3, or 4 |
| n           | u32  | number of buckets |
| b           | u32  | slots per bucket |
| fp_bits     | u32  | fingerprint width in bits |
| max_kicks   | u32  | kick-chain budget per insert |
| mean_lf     | f64  | mean load factor at TableFull across trials |
| min_lf      | f64  | minimum load factor observed |
| max_lf      | f64  | maximum load factor observed |
| stddev_lf   | f64  | population stddev across trials |

## `scf/insert_throughput.csv`

| Column   | Type | Description |
|----------|------|-------------|
| scheme   | s    | `standard` / `segmented` |
| arity    | u32  | |
| n        | u32  | |
| b        | u32  | |
| fp_bits  | u32  | |
| target_lf| f64  | load factor at which throughput was sampled |
| mean_ns  | ns   | mean per-insert latency |
| min_ns   | ns   | |
| max_ns   | ns   | |
| stddev_ns| ns   | |

## `scf/lookup_throughput.csv`

Columns identical to `insert_throughput.csv` plus:

| Column | Type | Description |
|--------|------|-------------|
| hit_rate | f64 | fraction of lookups that are present items (0.0, 0.25, 0.5, 0.75, 1.0) |

## `scf/delete_throughput.csv`

Columns identical to `insert_throughput.csv`.

## `scf/eviction.csv`

| Column      | Type | Description |
|-------------|------|-------------|
| scheme      | s    | |
| arity       | u32  | |
| n           | u32  | |
| b           | u32  | |
| max_kicks   | u32  | |
| mean_kicks  | f64  | mean evictions per insert |
| p50_kicks   | f64  | median |
| p99_kicks   | f64  | 99th percentile |
| p100_kicks  | u32  | worst observed kick chain |

## `scf/degree_distribution.csv`

| Column  | Type | Description |
|---------|------|-------------|
| scheme  | s    | |
| arity   | u32  | |
| n       | u32  | |
| b       | u32  | |
| degree  | u32  | bucket in-degree (# of items that nominated this bucket) |
| count   | u32  | # of buckets with this degree |

## `scf/fpr/arity{a}_n{n}_b{b}.csv`

| Column | Type | Description |
|--------|------|-------------|
| fp_bits | u32 | |
| mean_lf | f64 | |
| observed_fpr | f64 | |
| theoretical_bound | f64 | `d*b / 2^fp_bits` |

---

## `pir/setup_latency.csv` *(Phase B)*

| Column | Type | Description |
|--------|------|-------------|
| arity     | u32 | |
| n         | u32 | |
| b         | u32 | |
| fp_bits   | u32 | |
| mean_ns   | ns  | total server-setup wall-clock |
| min_ns    | ns  | |
| max_ns    | ns  | |
| stddev_ns | ns  | |

## `pir/query_latency.csv` *(Phase B)*

As above, with per-query latency.

## `pir/insert_cost.csv` *(Phase C)*

| Column | Type | Description |
|--------|------|-------------|
| arity | u32 | |
| n     | u32 | DB size before the insert |
| mean_ns | ns | per-insert wall-clock |
| columns_touched | u32 | hint-matrix columns rewritten |
| kick_chain_len | u32 | SCF kick chain length for this insert |

## `pir/delete_cost.csv` *(Phase C)*

As above for delete.

## `pir/full_rebuild_vs_incremental.csv` *(Phase C)*

| Column | Type | Description |
|--------|------|-------------|
| n             | u32 | DB size |
| churn_pct     | f64 | fraction of DB rewritten in the batch |
| incremental_ns | ns | total wall-clock, incremental path |
| rebuild_ns    | ns | total wall-clock, full-rebuild path |
| break_even    | bool | whether incremental is faster |
