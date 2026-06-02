# CLAUDE.md — segmented-cuckoo crate

## 1. Crate purpose

Provides two filter families and a key-value store layer:

- **Standard Cuckoo Filter** — the original 2/3/4-ary cuckoo filter used as a comparison baseline.
- **Segmented Cuckoo Filter (SCF)** — a novel variant where each candidate index is confined to its own fixed segment, giving every item a *deterministic, fixed* lookup position set. This property is what makes a single Index-PIR query sufficient in IKPIR.
- **`CuckooKVStore<S>`** — fingerprint-and-value KV store (lives in `store.rs`); same kicking + rollback as `CuckooFilter` with the chain widened to carry value cells. Takes `plaintext_bits` as a 5th constructor parameter (controls PIR cell width, 1–32; use 8 when values are byte-aligned). Exposes a zero-allocation `get_into` read path (streaming, no heap alloc) and an `as_cells` view for PIR matvec. Public aliases: `Segmented{2,3,4}aryCuckooKVStore`. Core KV methods: `insert`, `get`, `get_into`, `delete`, `update`, `contain`, `num_items`, `load_factor`, `set_max_kicks`, `value_size_in_bytes`, `size_in_bytes`. PIR bridge methods: `params()` → `CuckooParams`, `slot_cell_range`, `as_cells`, `snapshot_cells`, `from_cells`. Mutation log: `enable_mutation_log`, `disable_mutation_log`, `mutation_log_enabled`, `drain_mutations` → `Vec<SlotMutation>`, `apply_mutation`. Iteration: `iter_occupied_slots` → `OccupiedSlot`.

This crate does **not** contain PIR logic, cryptographic primitives, or network I/O. Those live in `ikpir-server` and `ikpir-client`.

## 2. File map

| File | Role |
|------|------|
| `src/scheme.rs` | `IndexScheme` trait + 6 concrete impls + `SchemeKind` enum + `SchemeMeta` marker trait (3 segmented impls); the only place index arithmetic is defined |
| `src/hash.rs` | `hash_item_*` / `all_indices_*` primitives; also tests the round-trip invariant |
| `src/fingerprint_table.rs` | `FingerprintTable`: bit-packed `Vec<u8>` storage for `CuckooFilter`; key methods: `read`, `write`, `contain`, `find`, `insert`, `delete` |
| `src/fingerprint_value_table.rs` | `FingerprintValueTable`: flat `Vec<u32>` cell-based storage for `CuckooKVStore`; each cell holds ≤ `plaintext_bits` payload in low bits (high bits always zero); key methods: `read_fingerprint`, `write_fingerprint`, `read_value`, `write_value`, `contain`, `find`, `insert`, `as_cells`, `from_cells`, `read_value_cells_chunk`, `read_value_to_box` |
| `src/filter.rs` | `CuckooFilter<S>` generic; add/delete/contain; rollback logic |
| `src/store.rs` | `CuckooKVStore<S>` generic + `CuckooParams` + `SlotMutation` + `OccupiedSlot`; `new` / `from_num_items` / `from_cells` for Segmented{2,3,4}ary; insert/get/delete/update with kicking + rollback + optional mutation log |
| `src/lib.rs` | Public type aliases and module-level docs; user-facing API surface |
| `src/util.rs` | `next_power_of_{2,3,4}` / `is_power_of_{3,4}`; used only in constructors |

## 3. Key design decisions (the WHY)

- **Rollback, not victim cache** — keeps the table shape stable (no extra rows); PIR requires a fixed-size array. A victim cache would add a variable-length overflow that breaks the fixed-index guarantee.
- **Non-zero fingerprint invariant** — 0 means "slot empty"; the hash layer remaps a 0 result to 1. This lets `FingerprintTable` use a single zero-test for occupancy without a separate presence bit.
- **`FingerprintTable`: bit-packed `Vec<u8>` + 8-byte tail padding** — avoids alignment waste; unaligned 8-byte load = single MOV on x86. The padding prevents the load from reading past the buffer end on the last slot. Used exclusively by `CuckooFilter`.
- **`FingerprintValueTable`: flat `Vec<u32>` cells, LSB-first, high bits zero** — each cell stores at most `plaintext_bits` bits of payload in its low bits; the high `(32 - plaintext_bits)` bits are always zero. This is the *ChalametPIR matvec invariant*: `vec_mult_u32_u32` accumulates `u32` cells into a `u64` accumulator and the high-bit guarantee prevents overflow. `cells_per_slot = ⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`; fingerprint occupies the leading bits within the cell-stream, value occupies the trailing bits. Used exclusively by `CuckooKVStore`.

## 4. Scheme/hash symmetry invariant

For any item, `all_indices(hash_item(item).indices[p], hash_item(item).fingerprint)` must return the same index set for all positions `p` (with `p < arity`). Tests in `hash.rs` verify this property for every scheme variant.

## 5. Adding a new scheme

1. Implement `IndexScheme` in `scheme.rs`
2. Add `hash_item_*` / `all_indices_*` in `hash.rs`
3. Add a constructor in `filter.rs`
4. If the scheme is segmented, also add a constructor in `store.rs` (`CuckooKVStore` is segmented-only)
5. Add a public type alias in `lib.rs`

## 6. Entry points for common tasks

| Task | Where to look |
|------|---------------|
| Understand index arithmetic | `scheme.rs` + `hash.rs` |
| Understand filter storage layout | `fingerprint_table.rs` (`FingerprintTable`: bit-packed `Vec<u8>`, 8-byte tail padding, unaligned load) |
| Understand KV-store storage layout | `fingerprint_value_table.rs` (`FingerprintValueTable`: flat `Vec<u32>` cells, LSB-first, high bits zero; `read_bits`/`write_bits` private helpers handle cross-cell boundaries) |
| Understand insert/delete/lookup flow | `filter.rs::insert()`, `::contain()`, `::delete()` |
| Understand KV-store insert flow | `store.rs::CuckooKVStore::insert` (kick chain: `chain_meta: Vec<(u32,u32,u32)>` = `(bucket, slot, evicted_fp)` + flat `chain_values: Vec<u32>` slab (cell units) + `cur_value: Vec<u32>` scratch + `pending_mutations: Vec<SlotMutation>` scratch flushed to `mutation_log` on commit; byte↔cell conversion via `pack_value_bytes_to_cells` / `unpack_value_streaming` at the public boundary) |
| Understand IKPIR client geometry | `CuckooParams::candidate_buckets` + `slot_cell_range`; see `examples/pir_plaintext_recovery.rs` |
| Understand mutation log | `enable_mutation_log` / `drain_mutations`; `SlotMutation` fields; emission rules in store.rs doc comments; replay via `apply_mutation` |
| Add a new scheme variant | `scheme.rs` → `hash.rs` → `filter.rs` → `store.rs` (if segmented) → `lib.rs` |
| Change filter slot storage | `fingerprint_table.rs` only; scheme/hash untouched |
| Change KV-store slot storage | `fingerprint_value_table.rs` only; scheme/hash untouched |
