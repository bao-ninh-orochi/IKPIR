# `segmented-cuckoo` finalisation plan

Scope: finish the `segmented-cuckoo` crate before the next task (ikpir).
Status: drafted, awaiting implementation. Tick each `[ ]` as it lands.

> Revised after a source-level audit of `filter.rs`, `store.rs`, `data_layout.rs`, the existing tests/examples/benches, and the `ikpir-*` stubs. Corrections folded in: `CuckooFilter` actually has **12** constructors (6 segmented + 6 standard), `value_size_in_bytes` already exists on `FingerprintValueTable` (delegate, don't reinvent), `update_value` keeps its `find` step (only the inner `write` is replaced), and the macro at `filter.rs:1722` covers most of the test rename churn. `ikpir-server`/`ikpir-client` are 1-line `// TODO` stubs, so the API rename is safe.

## Goals
1. Unify the public method names of `CuckooFilter` and `CuckooKVStore` (`add → insert`, `size → num_items`).
2. Eliminate per-call heap allocations on the `FingerprintValueTable` hot paths via `read_partial` / `write_partial`.
3. Move the rollback chain off the per-insert allocation path on both `CuckooFilter` and `CuckooKVStore`.
4. Add `get_into` + `value_size_in_bytes` on `CuckooKVStore` for zero-allocation reads from `ikpir-server`.
5. Cover `CuckooKVStore` with examples + benches matching the filter-side structure.
6. Sweep up doc drift in `CLAUDE.md` and `README.md`.

## Non-goals (this round)
- No changes to `IndexScheme`, hash functions, or scheme arithmetic.
- No new filter/scheme variants.
- No PIR / cryptographic logic (next-task scope: `ikpir-server` / `ikpir-client`).
- No `Cargo.toml` workspace reorganisation.

## Existing helpers to reuse (don't reinvent)
- `FingerprintValueTable::value_size_in_bytes` — `data_layout.rs:584`. The new `CuckooKVStore::value_size_in_bytes` is a one-line delegating wrapper.
- `FingerprintValueTable::read_value` — `data_layout.rs:610`. After commit 2 it is alloc-free, so the new `CuckooKVStore::get_into` just calls it.
- `DataLayout::read` / `write` — `data_layout.rs:183` / `233`. After commit 2 these become thin wrappers around the new `read_partial` / `write_partial`.

## Pre-flight (before commit 1)
- [ ] `cargo test -p segmented-cuckoo` — establish green baseline.
- [ ] `cargo clippy -p segmented-cuckoo --all-targets -- -D warnings` — lint baseline.
- [ ] `cargo bench -p segmented-cuckoo --no-run` — compile-check all benches.
- [ ] (Optional) capture pre-change ad-hoc `kv_store` insert/lookup numbers for before/after comparison after commit 2.

---

## Commit 1 — API renames + arithmetic consistency

Pure rename / cosmetic; no behaviour change. The bulk of the test churn collapses into a single edit inside the `delete_contract_tests!` macro body.

### Source changes
- [ ] `src/filter.rs`
  - `pub fn add` → `pub fn insert` (impl block at L944).
  - `pub fn size` → `pub fn num_items` (L1154). The struct field `num_items` already exists; the new method just shares the name.
  - Update every rustdoc example body in this file (≈19 `.add(` lines + 3 `.size()` lines): `f.add(...) → f.insert(...)`, `f.size() → f.num_items()`.
  - Update the `delete_contract_tests!` macro **body** at L1722 — one edit propagates to all 6 instantiations (`delete_seg2/3/4`, `delete_std2/3/4` at L1863–1893).
  - Update the standalone unit tests above the macro and the `prop_tests` block at L1900+.
  - Do **not** add `// TEST-CHANGE: …` comments.
- [ ] `src/store.rs`
  - L431: `self.num_items = self.num_items.saturating_sub(1)` → `self.num_items -= 1` in `delete`. (Pre-condition: `delete` only reaches this line after a successful `find`; underflow is impossible.)
  - L339 comment: `mirrors `CuckooFilter::add` defensiveness` → `mirrors `CuckooFilter::insert` defensiveness`.
- [ ] `src/lib.rs` — crate-level rustdoc Quick start (L64–76) and auto-size example (L80–86): `add → insert`.

### Examples
- [ ] `examples/basic_usage.rs` — `f.add(...) → f.insert(...)`.
- [ ] `examples/load_factor.rs` — 6× `filter.add(...) → filter.insert(...)`.

### Benches (rename only the call sites; enumerate with `grep -n '\.add(\|\.size()'` first)
- [ ] `benches/insert_throughput.rs`
- [ ] `benches/lookup_throughput.rs`
- [ ] `benches/delete_throughput.rs`
- [ ] `benches/load_factor.rs`
- [ ] `benches/fpr.rs`
- [ ] `benches/degree_distribution.rs`

### Docs
- [ ] `segmented-cuckoo/CLAUDE.md` section 6 row "Understand insert/delete/lookup flow": `filter.rs::add()` → `filter.rs::insert()`.
- [ ] `segmented-cuckoo/README.md` library-usage code block at L118 / L126.

### Validation
- [ ] `cargo test -p segmented-cuckoo` — must stay green; the renames are mechanical.
- [ ] `cargo build -p segmented-cuckoo --examples`.
- [ ] `cargo bench -p segmented-cuckoo --no-run`.
- [ ] `cargo clippy -p segmented-cuckoo --all-targets -- -D warnings`.

---

## Commit 2 — `read_partial` / `write_partial` + rewire `FingerprintValueTable`

Adds new bit-precise primitives, makes existing `read`/`write` thin wrappers, and removes per-call heap allocations from `read_fingerprint`, `read_value`, `write`, and `delete`.

### `src/data_layout.rs` — `DataLayout`
- [ ] Add `pub fn read_partial(&self, bucket: u32, slot: u32, bit_offset: u32, n_bits: u32, out: &mut [u8])`.
- [ ] Add `pub fn write_partial(&mut self, bucket: u32, slot: u32, bit_offset: u32, n_bits: u32, src: &[u8])`.
- [ ] Refactor `read` and `write` to delegate to the new methods (existing roundtrip tests then guard the partial primitives).
- [ ] Preconditions (panicking asserts):
  - `bit_offset + n_bits <= data_size_in_bit`
  - `bucket < num_buckets && slot < slots_per_bucket`
  - `out.len() == ceil(n_bits / 8)` / `src.len() == ceil(n_bits / 8)`
  - `n_bits >= 1` (CuckooKVStore rejects `value_bits == 0` at construction; see `store.rs:547` test)
- [ ] `write_partial` clears only the n_bits range it writes (not the whole slot) and masks any high bits in `src` beyond `n_bits`.

### `src/data_layout.rs` — `FingerprintValueTable`
- [ ] Add field `zero_buf: Vec<u8>` initialised in `new` to `vec![0u8; layout.slot_size_in_bytes()]`. Bit-math: `slot_size_in_bytes() == ceil((fp_bits + value_bits) / 8) == ceil(entry_bits / 8)`, so this buffer is exactly `entry_bits` worth — the right size for a full-entry zero write in `delete`.
- [ ] Refactor `read_fingerprint` — stack `[0u8; 4]` buffer; `read_partial(b, s, 0, fp_bits, &mut buf[..ceil(fp_bits/8)])`; assemble `u32` matching `unpack_fingerprint`'s current endian convention.
- [ ] Refactor `read_value` — single `read_partial(b, s, fp_bits, value_bits, out)`. Caller-provided `out` must be `value_size_in_bytes`.
- [ ] Refactor `write` — two calls: `write_partial(b, s, 0, fp_bits, &fp_bytes)` then `write_partial(b, s, fp_bits, value_bits, value)`.
- [ ] Refactor `update_value` — **keep** the `find(bucket, fingerprint)` step; replace **only** the inner `self.write(bucket, slot, fingerprint, new_value)` call with `self.layout.write_partial(bucket, slot, fp_bits, value_bits, new_value)`. Fingerprint bits stay untouched.
- [ ] Refactor `delete` — `self.layout.write_partial(bucket, slot, 0, fp_bits + value_bits, &self.zero_buf)`.
- [ ] Remove private helpers `pack_entry`, `unpack_value`, `unpack_fingerprint`.
- [ ] Drop the deferred-optimisation rustdoc at L493–498.

### Tests (add to existing `mod tests` in `data_layout.rs`)
DataLayout-direct:
- [ ] `read_partial_aligned_offset`
- [ ] `read_partial_misaligned_within_byte`
- [ ] `read_partial_crosses_byte_boundary`
- [ ] `read_partial_at_last_slot` (covers the trailing 8-byte tail-padding region)
- [ ] `write_partial_doesnt_disturb_remainder_of_slot`
- [ ] `write_partial_doesnt_disturb_neighbour_slot`
- [ ] `write_partial_then_read_partial_roundtrip` — widths 1, 7, 8, 9, 12, 16, 31, 33, 64, 100
- [ ] `write_partial_high_bits_in_src_are_masked`
- [ ] Panic tests: range OOB, wrong buffer length, OOB bucket/slot, `n_bits == 0`.

FingerprintValueTable:
- [ ] `delete_zeros_value_bits_too` (after delete, `read_value` returns zeros).
- [ ] `update_value_doesnt_touch_fingerprint_bits` (re-affirm against the new path).

### Validation
- [ ] `cargo test -p segmented-cuckoo` — primary gates are the existing `fvt_roundtrip_*` and `datalayout_roundtrip_*` tests at `data_layout.rs:843+`.
- [ ] `cargo clippy`.
- [ ] (Optional) before/after MOps comparison at wide value widths.

---

## Commit 3 — Pre-allocated rollback chains

Behaviour-preserving refactor of `insert` on both `CuckooKVStore` and `CuckooFilter`. All existing tests must pass unchanged; new tests cover the buffer-reuse contract.

### `src/store.rs` — `CuckooKVStore`
- [ ] Add fields:
  ```
  chain_meta:   Vec<(u32, u32, u32)>,  // (bucket, slot, evicted_fp); cap = max_kicks
  chain_values: Vec<u8>,                // size = max_kicks * vsize; indexed slab
  cur_value:    Vec<u8>,                // length = vsize; in-flight displaced value
  ```
- [ ] Initialise in **all 6** `CuckooKVStore` constructors:
  - `Segmented2aryScheme::new` (L61), `::from_num_items` (L95)
  - `Segmented3aryScheme::new` (L131), `::from_num_items` (L167)
  - `Segmented4aryScheme::new` (L205), `::from_num_items` (L240)
  - Each: `cur_value = vec![0u8; vsize]`, `chain_meta = Vec::with_capacity(MAX_KICKS_DEFAULT as usize)`, `chain_values = vec![0u8; MAX_KICKS_DEFAULT as usize * vsize]`.
- [ ] Rewrite `insert` (L295):
  - Replace `let mut cur_value: Vec<u8> = value.to_vec()` (L319) with `self.cur_value.copy_from_slice(value)`.
  - Replace `Vec::with_capacity(...)` chain (L322–323) with `self.chain_meta.clear()`.
  - Per kick:
    1. Read evicted fp into a local.
    2. `chain_values[kick_idx*vsize..(kick_idx+1)*vsize].copy_from_slice(&self.cur_value)`.
    3. `chain_meta.push((cur_index, slot, evicted_fp))`.
    4. Read evicted value into `self.cur_value` via `read_value`.
    5. Write `(cur_fingerprint, &chain_values[kick_idx*vsize..(kick_idx+1)*vsize])` to slot.
    6. Move evicted fp into `cur_fingerprint`.
  - Rollback (L374–377): iterate `chain_meta` in reverse; slice into `chain_values` for each restored value.
- [ ] Update `set_max_kicks` (L505):
  ```
  self.chain_meta.reserve(max_kicks as usize);
  self.chain_values.resize(max_kicks as usize * vsize, 0);
  ```
- [ ] Update `set_max_kicks` rustdoc — drop OOM caveats; mention buffer resize.

### `src/filter.rs` — `CuckooFilter`
- [ ] Add field `chain: Vec<(u32, u32, u32)>` to the struct.
- [ ] Initialise in **all 12** `CuckooFilter` constructors with `Vec::with_capacity(MAX_KICKS_DEFAULT as usize)`:
  - Segmented: L278, L328, L384, L436, L495, L547
  - Standard:  L603, L650, L706, L755, L811, L860
- [ ] `insert` (formerly `add`, L944): replace `let mut chain: Vec<...> = Vec::with_capacity(...)` (L965) with `self.chain.clear()`.
- [ ] `set_max_kicks` (L1246): add `self.chain.reserve(max_kicks as usize)`.

### Tests (both types)
- [ ] `repeated_inserts_reuse_buffers` — ≥1000 inserts; `capacity()` doesn't grow past `max_kicks` after the first.
- [ ] `set_max_kicks_grows_buffers` — increase `max_kicks`, observe `capacity()` reflects it.
- [ ] `failed_insert_then_successful_insert` — verify a rollback path leaves the next insert correct.

### Validation
- [ ] `cargo test -p segmented-cuckoo` — primary gates: `table_full_triggers_rollback` (`store.rs:608`), `kicking_under_load_2/3/4ary` (L638/658/676), `wide_value_roundtrip_through_kicking` (L694).
- [ ] `cargo clippy`.

---

## Commit 4 — `get_into`, `value_size_in_bytes`, examples, benches

### `src/store.rs`
- [ ] Add to `impl<S: IndexScheme> CuckooKVStore<S>`:
  ```rust
  /// Number of bytes one value occupies (== value_bits rounded up to a byte).
  #[inline]
  pub fn value_size_in_bytes(&self) -> usize {
      self.table.value_size_in_bytes()
  }

  /// Read the value for `key` into `out`. Returns `true` on hit.
  /// Panics if `out.len() != value_size_in_bytes()`.
  pub fn get_into<K: AsRef<[u8]>>(&self, key: K, out: &mut [u8]) -> bool {
      assert_eq!(out.len(), self.value_size_in_bytes(),
                 "out buffer length must equal value_size_in_bytes");
      let (fp, indices) = self.scheme.hash_item(key.as_ref(), self.table.fingerprint_bits());
      for i in 0..self.scheme.arity() {
          if let Some(slot) = self.table.find(indices[i], fp) {
              self.table.read_value(indices[i], slot, out);
              return true;
          }
      }
      false
  }
  ```
  `get_into` is alloc-free thanks to commit 2 (`read_value` no longer heaps).
- [ ] Tests:
  - `get_into_hit_writes_value_returns_true`
  - `get_into_miss_returns_false`
  - `get_into_panics_on_wrong_length` (`#[should_panic(expected = "value_size_in_bytes")]`)

### `src/lib.rs`
- [ ] Surface `value_size_in_bytes` and `get_into` in the rustdoc tour for one segmented KV-store alias.

### Examples
- [ ] `examples/kv_store_basic_usage.rs` — demo all 3 segmented variants (insert / get / get_into / update / delete / num_items / size_in_bytes / load_factor). Mirror `examples/basic_usage.rs`.

### Benches (new files; configs `arity ∈ {2,3,4}`, `bucket_size ∈ {2,4}`, `value_bits ∈ {8, 64, 256, 1024}`; CSV `mean_inserted, mean_lf, mean_mops`)
- [ ] `benches/kv_store_insert_throughput.rs`
- [ ] `benches/kv_store_lookup_throughput.rs` — uses `get_into` (zero-alloc); 50/50 hit/miss query mix.
- [ ] `benches/kv_store_delete_throughput.rs`

### `Cargo.toml` (segmented-cuckoo)
- [ ] Add three new `[[bench]]` entries after the existing six (L31–53).

### Docs
- [ ] `segmented-cuckoo/CLAUDE.md` section 5 — extend the "Adding a new scheme" recipe: for a segmented scheme, also add a constructor in `store.rs` (`CuckooKVStore` is segmented-only).
- [ ] `segmented-cuckoo/CLAUDE.md` section 2 — drop "planned" framing on the `store.rs` row now that `get_into` lands.
- [ ] `segmented-cuckoo/README.md` — add a sibling KV-store usage block under the existing library-usage block.

### Validation
- [ ] `cargo test -p segmented-cuckoo`.
- [ ] `cargo run -p segmented-cuckoo --example kv_store_basic_usage`.
- [ ] `cargo run -p segmented-cuckoo --example basic_usage` (regression).
- [ ] `cargo bench -p segmented-cuckoo --no-run`.
- [ ] `cargo clippy -p segmented-cuckoo --all-targets -- -D warnings`.

---

## Final-pass validation (after all four commits)
- [ ] `cargo test -p segmented-cuckoo`.
- [ ] `cargo clippy -p segmented-cuckoo --all-targets -- -D warnings`.
- [ ] `cargo doc -p segmented-cuckoo --no-deps`.
- [ ] Examples (all three): `basic_usage`, `load_factor`, `kv_store_basic_usage`.
- [ ] (Optional) one bench per family on a small config to validate output.

## Risks / things to watch
- **Commit 2 bit-math regression.** `read_partial` / `write_partial` are tricky around byte boundaries. Mitigation: keep the existing `read` / `write` as wrappers; existing roundtrip tests catch regressions; new explicit boundary tests above.
- **Commit 3 buffer indexing.** Off-by-one in `slab_off = kick_idx * vsize` would silently corrupt rollback. Mitigation: existing rollback tests on both types are the correct guards.
- **Bench compile-time vs runtime.** Plan only requires `--no-run`; full bench runs are separate and slower.
- **API break for `ikpir-*` crates.** `add → insert`, `size → num_items` are breaking. Verified safe: both `ikpir-server/src/lib.rs` and `ikpir-client/src/lib.rs` are 1-line `// TODO` stubs. Right time to do this.
- **Macro-driven test churn.** Most `.add(` test sites in `filter.rs` live inside `delete_contract_tests!` (L1722); editing the macro body propagates to all 6 instantiations.

## Open questions
None blocking. Confirm and commit 1 begins.
