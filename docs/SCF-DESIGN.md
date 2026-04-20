# ARCH-25-03-26: Per-(arity, b) Load Factor Thresholds in `from_num_items`

**Status:** Implemented
**Date:** 2026-03-25

---

## Context

The `from_num_items` constructor is the primary user-facing way to build a filter when you know
the expected number of items. Internally it auto-sizes `n` by:

1. Computing the minimum buckets needed: `n = upper_power_of_2(ceil(max_items / b))`
2. Doubling if the projected load exceeds a threshold

The original threshold was a single hardcoded constant: `0.96` for all six filter types. This was
wrong in two ways:

1. **Too aggressive for low-arity, low-b configs.** The empirical maximum achievable load factor
   for arity=2, b=1 is only ~0.50. A threshold of 0.96 would leave the filter undersized, causing
   `TableFull` errors before all `max_items` could be inserted.

2. **Needlessly conservative for high-arity, high-b configs.** The empirical maximum for arity=4,
   b=4 is ~0.997. Using 0.96 wastes ~4% capacity.

Additionally, the original code used a single `if` for the threshold check. This is only correct
when the threshold is ≥ 0.5, because `upper_power_of_2` guarantees `capacity ≥ max_items`, so the
ratio ≤ 1.0, and one halving always drops below 0.5. But once any threshold falls below 0.5
(arity=2, b=1 target = 0.48), a single doubling may be insufficient, leaving the projected load
still above the threshold.

---

## Decision

Replace the single `0.96` constant with a 3×4 constant table `MAX_LOAD_FACTOR[arity-2][b-1]`,
and replace the `if` check with a `while` loop.

### Threshold values

Each value is **0.95 × the mean empirical max load factor** across all tested configurations:
- Scheme variants: standard and segmented (values differ by < 1%, so a shared table suffices)
- Table sizes: n ∈ {2^14, 2^16, 2^18, 2^20} (for arity=2/4) or n ∈ {3·2^12 … 3·2^18} (arity=3)
- Trials: 20 per (scheme, n, b) combination
- `MAX_KICKS = 500`

The 5% discount from the mean provides a safety margin that accounts for:
- Trial-to-trial variance in the kicking chain (randomised eviction order)
- The slight decrease in achievable load factor at larger n (larger tables have more potential
  collision paths for the fixed kick budget; using the mean across all n sizes naturally averages
  the pessimistic large-n end into the threshold)

| arity | b=1  | b=2  | b=3  | b=4  |
|-------|------|------|------|------|
| 2     | 0.48 | 0.83 | 0.89 | 0.91 |
| 3     | 0.85 | 0.93 | 0.94 | 0.94 |
| 4     | 0.91 | 0.94 | 0.95 | 0.95 |

Raw mean load factors (from `benches/load_factor.rs`, `results/load_factor.csv`):

| arity | b=1  | b=2  | b=3  | b=4  |
|-------|------|------|------|------|
| 2     | 0.502 | 0.873 | 0.938 | 0.963 |
| 3     | 0.897 | 0.975 | 0.989 | 0.993 |
| 4     | 0.961 | 0.991 | 0.996 | 0.997 |

### `while` loop instead of `if`

The threshold for arity=2, b=1 is 0.48 < 0.5. A single doubling of n drops the projected load
by half, but if the original ratio was between 0.96 and 1.0, one halving gives 0.48–0.50, which
may still exceed the target. The `while` loop handles this correctly in all cases (in practice it
iterates at most twice).

---

## Rationale

### Why 0.95 × mean (not minimum)?

The minimum across all tested (n, scheme, trial) combinations is very close to the mean for most
configurations — the load factor distribution is tight. Using the minimum would add unnecessary
conservatism without meaningful robustness improvement. The 5% discount below the mean provides
adequate headroom without wasting capacity.

### Why a shared table for standard and segmented?

Empirically, the two scheme families achieve nearly identical max load factors (difference < 1%)
for every (arity, b) combination. Separate tables would require twice the maintenance for no
measurable improvement in sizing accuracy.

### Why a free function rather than a trait method?

`target_load_factor` is only called from `from_num_items` constructors. Adding it to
`IndexScheme` would pollute the trait with a concern that belongs solely to the auto-sizing
constructor, not to the core index computation contract. A module-level free function keeps the
trait surface clean.

### n-dependence

Max load factor decreases slightly as n grows (larger tables have more collision probability for
a fixed kick budget). The mean across the tested n range (2^14 … 2^20) naturally incorporates
the pessimistic large-n end, making the threshold conservative for smaller tables (which can
actually sustain slightly higher load) and appropriately cautious for the range where most
production use cases land.

If significantly larger tables (n >> 2^20) are used, the achievable load factor may be somewhat
lower than these thresholds. In that case, users should construct filters manually via `new()` or
`Segmented3aryCuckooFilter::new()` with explicitly chosen n.

---

## Alternatives Considered

| Alternative | Reason Rejected |
|---|---|
| Use minimum across all trials (not mean×0.95) | Marginally more conservative with no benefit; min ≈ mean for tight distributions |
| Use a formula modeling n-dependence | Over-engineering; the n-decay across our tested range is ≤ 0.01 load factor units |
| Add to `IndexScheme` trait | Pollutes the trait; sizing logic is not part of the index computation contract |
| Algebraic sizing: `n = upper_power_of_2(ceil(max_items / (b * target)))` | Correct and elegant, but changes the meaning of the rounding step. The while-loop approach preserves the existing `upper_power_of_2(ceil(max_items/b))` initial estimate, which is a natural lower bound, and only enlarges from there. |

---

## Consequences

- Filters sized via `from_num_items` with b=1 or b=2 for arity=2 will now be **larger** than
  before (more headroom allocated). This is the correct behaviour — they were previously
  undersized.
- Filters with high arity or large b may be **slightly smaller** than before (0.96 → 0.94–0.95
  for arity=3/4, b=3/4). The difference is at most one doubling step.
- All 208 unit and doc tests pass unchanged.
- The `MAX_LOAD_FACTOR` table must be re-derived if `MAX_KICKS_DEFAULT` is changed, as a higher
  kick budget enables higher achievable load factors and vice versa.

---

## References

- `src/filter.rs` — implementation (constant, helper, six `from_num_items` bodies)
- `results/load_factor.csv` — raw experimental data
- `benches/load_factor.rs` — benchmark that produced the data
