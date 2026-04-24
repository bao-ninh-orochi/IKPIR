# Changelog

All notable changes to this crate are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Rename `util::upper_power_of_2` → `util::next_power_of_2`.** All three power-of-N
  helpers (`next_power_of_2`, `next_power_of_3`, `next_power_of_4`) now share the same
  `next_power_of_N` naming scheme and are declared in 2/3/4 order. Behavior unchanged.
  Breaking for any out-of-workspace caller that imported `upper_power_of_2`.
- **Reorder constructor `impl` blocks in `filter.rs`** so all three Segmented variants
  (2-ary, 3-ary, 4-ary) come before all three Standard variants. No behavior change.
- **Restrict supported parameter ranges.** `validate_common_params` now enforces
  `arity ∈ {2, 3, 4}` and `bucket_size ∈ 1..=4`; out-of-range values return
  `CuckooError::InvalidParams` instead of being silently accepted (or panicking via
  out-of-bounds access into `MAX_LOAD_FACTOR`). New public constants
  `SUPPORTED_ARITIES` and `SUPPORTED_BUCKET_SIZES` document the supported sets.
  All constructor doc comments updated to reflect the tightened constraint.



- **Rename: `TagTable` → `FingerprintTable`** (Change 1). All public methods
  renamed accordingly: `read_tag` → `read_fingerprint`, `write_tag` →
  `write_fingerprint`, `insert_tag_to_bucket` → `insert_fingerprint_to_bucket`,
  `find_tag_in_bucket_slot` → `find_fingerprint_in_bucket_slot`. Identifier
  `tag` replaced by `fingerprint` throughout source, tests, and benchmarks.
- **Rename: `fp_bits` → `fingerprint_bits`** (Change 2). All local variables,
  constants (`FP_BITS` → `FINGERPRINT_BITS`), and CSV column headers updated.
- **Rename: `k` → `arity`** (Change 3). Loop variables in `filter.rs` and
  bench harnesses use `arity` instead of `k`.
- **Removed `InsertStats` / `add_with_stats`** (Change 4). The eviction-stats
  insert variant and its bench (`benches/eviction.rs`) are deleted. Associated
  PNG images (`images/eviction_*.png`) and `scripts/plot.py` functions
  (`plot_eviction`, `plot_eviction_mean_kicks`) are removed.
- **Added `Standard2aryScheme` / `Segmented2aryScheme` aliases** (Change 5/6).
  Previous names `StandardScheme` and `SegmentedScheme` are removed; all
  consumers (including `ikpir-common` and `ikpir-server`) updated.
- **`IndexScheme::all_indices` — removed `position: u8` parameter** (Q4
  decision). The argument is no longer needed: for segmented schemes position
  is derived from the index; for standard 2-ary XOR symmetry makes it
  irrelevant; for standard 3-ary/4-ary xord cycling handles it implicitly.
  Standard 2-ary round-trip tests now use sort-and-compare since the returned
  pair order depends on which endpoint `cur_index` is.
- **Removed `reinsert_after_full_delete_standard_3ary/4ary` tests** (Q6). Both
  tests are subsumed by the `delete_contract_tests!` macro output.
- **Dropped nested type aliases in `filter.rs` test module** (Q7). Tests now
  use `CuckooFilter::<SchemeType>::` directly.
- **`KeywordFootprint.tag` → `KeywordFootprint.fingerprint`** in `ikpir-common`
  and `ikpir-client`. Breaking change for any out-of-workspace consumers.

### Added

- 6 cuckoo filter variants: Standard and Segmented, each in 2-ary, 3-ary,
  and 4-ary forms. Public type aliases: `Standard2aryCuckooFilter`,
  `Segmented2aryCuckooFilter`, `Standard3aryCuckooFilter`,
  `Segmented3aryCuckooFilter`, `Standard4aryCuckooFilter`,
  `Segmented4aryCuckooFilter`.
- `IndexScheme` trait abstracting the arity / segmented distinction; six
  scheme structs implement it.
- Bit-packed `FingerprintTable` for arbitrary fingerprint widths.
- Custom CSV-emitting bench harnesses for load factor, insert / lookup /
  delete throughput, false-positive rate, and degree distribution.
- Property-based tests for delete round-trip correctness (proptest).

## [0.1.0] — TBD

First tagged release alongside paper submission.
