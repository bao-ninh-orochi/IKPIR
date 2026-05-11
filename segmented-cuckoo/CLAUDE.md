# CLAUDE.md — segmented-cuckoo crate

## 1. Crate purpose

Provides two filter families and a key-value store layer:

- **Standard Cuckoo Filter** — the original 2/3/4-ary cuckoo filter used as a comparison baseline.
- **Segmented Cuckoo Filter (SCF)** — a novel variant where each candidate index is confined to its own fixed segment, giving every item a *deterministic, fixed* lookup position set. This property is what makes a single Index-PIR query sufficient in IKPIR.
- **`CuckooKVStore<S>`** — fingerprint-and-value KV store (lives in `store.rs`); same kicking + rollback as `CuckooFilter` with the chain widened to carry value bytes. Exposes a zero-allocation `get_into` read path. Public aliases: `Segmented{2,3,4}aryCuckooKVStore`. Key methods: `insert`, `get`, `get_into`, `delete`, `update`, `set_max_kicks`, `value_size_in_bytes`, `contain`, `num_items`, `size_in_bytes`, `load_factor`.

This crate does **not** contain PIR logic, cryptographic primitives, or network I/O. Those live in `ikpir-server` and `ikpir-client`.

## 2. File map

| File | Role |
|------|------|
| `src/scheme.rs` | `IndexScheme` trait + 6 concrete impls; the only place index arithmetic is defined |
| `src/hash.rs` | `hash_item_*` / `all_indices_*` primitives; also tests the round-trip invariant |
| `src/data_layout.rs` | `DataLayout` raw bit-packed slot storage (any width) + `FingerprintTable` and `FingerprintValueTable` wrappers; the only place byte-level I/O happens |
| `src/filter.rs` | `CuckooFilter<S>` generic; add/delete/contain; rollback logic |
| `src/store.rs` | `CuckooKVStore<S>` generic; `new` + `from_num_items` for Segmented{2,3,4}ary; insert/get/delete/update with cuckoo kicking + rollback |
| `src/lib.rs` | Public type aliases and module-level docs; user-facing API surface |
| `src/util.rs` | `next_power_of_{2,3,4}` / `is_power_of_{3,4}`; used only in constructors |

## 3. Key design decisions (the WHY)

- **Rollback, not victim cache** — keeps the table shape stable (no extra rows); PIR requires a fixed-size array. A victim cache would add a variable-length overflow that breaks the fixed-index guarantee.
- **Non-zero fingerprint invariant** — 0 means "slot empty"; the hash layer remaps a 0 result to 1. This lets `FingerprintTable` use a single zero-test for occupancy without a separate presence bit.
- **Bit-packed `Vec<u8>` + 8-byte tail padding** — avoids alignment waste; unaligned 8-byte load = single MOV on x86. The padding prevents the load from reading past the buffer end on the last slot.

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
| Understand storage layout | `data_layout.rs` (`DataLayout` for the raw primitive; `FingerprintTable` for the filter wrapper; `FingerprintValueTable` for the KV-store wrapper) |
| Understand insert/delete/lookup flow | `filter.rs::insert()`, `::contain()`, `::delete()` |
| Understand KV-store insert flow | `store.rs::CuckooKVStore::insert` (kick chain: `chain_meta: Vec<(u32,u32,u32)>` = `(bucket, slot, evicted_fp)` + flat `chain_values: Vec<u8>` slab + `cur_value: Vec<u8>` scratch; see `store.rs:44–51`); `data_layout.rs::FingerprintValueTable` |
| Add a new scheme variant | `scheme.rs` → `hash.rs` → `filter.rs` → `store.rs` (if segmented) → `lib.rs` |
| Change slot storage | `data_layout.rs` only; scheme/hash untouched |
| Add a new wrapper over `DataLayout` | `data_layout.rs` for `DataLayout::read`/`write`; new wrapper struct in its own module |
