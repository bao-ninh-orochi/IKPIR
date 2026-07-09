# CLAUDE.md — segmented-cuckoo crate

## Purpose

Two filter families plus a KV-store layer. No PIR, crypto, or network I/O
(those live in `ikpir-server` / `ikpir-client`).

- **Standard Cuckoo Filter** — original 2/3/4-ary filter; comparison baseline.
- **Segmented Cuckoo Filter (SCF)** — each candidate index is confined to its
  own fixed segment, so every item has a *deterministic, fixed* lookup-position
  set. This is what makes a single Index-PIR query sufficient in IKPIR.
- **`CuckooKVStore<S>`** — stores `fp ‖ value` per slot; same kicking + rollback
  as the filter, chain widened to carry value cells. Constructor takes a 5th
  param `plaintext_bits` (PIR cell width, 1–32; use 8 for byte-aligned values).
  Aliases `Segmented{2,3,4}aryCuckooKVStore`.

## File map

| File | Role |
|------|------|
| `src/scheme.rs` | `IndexScheme` trait + 6 impls + `SchemeKind` + `SchemeMeta`; the only place index arithmetic lives |
| `src/hash.rs` | `hash_item_*` / `all_indices_*`; tests the scheme/hash round-trip invariant |
| `src/fingerprint_table.rs` | `FingerprintTable`: bit-packed `Vec<u8>` storage for `CuckooFilter` |
| `src/fingerprint_value_table.rs` | `FingerprintValueTable`: flat `Vec<u32>` cell storage for `CuckooKVStore` (high-bits-zero invariant) |
| `src/filter.rs` | `CuckooFilter<S>`: add/delete/contain + rollback |
| `src/store.rs` | `CuckooKVStore<S>` + `CuckooParams` + `SlotMutation` + `OccupiedSlot` |
| `src/lib.rs` | Public type aliases + module docs |
| `src/util.rs` | `next_power_of_{2,3,4}` / `is_power_of_{3,4}` (constructors only) |

## Key design decisions (the WHY)

- **Rollback, not victim cache** — keeps the table shape fixed; PIR needs a
  fixed-size array. A victim cache adds variable-length overflow that breaks the
  fixed-index guarantee.
- **Non-zero fingerprint invariant** — 0 means "empty slot"; the hash layer
  remaps a 0 result to 1, so occupancy is a single zero-test, no presence bit.
- **`FingerprintTable`: bit-packed `Vec<u8>` + 8-byte tail padding** — unaligned
  8-byte load = single MOV; padding stops the last-slot load reading past the end.
- **`FingerprintValueTable`: flat `Vec<u32>` cells, LSB-first, high bits zero** —
  each cell holds ≤ `plaintext_bits` payload in its low bits. This is the
  *ChalametPIR matvec invariant*: `vec_mult_u32_u32` accumulates `u32` cells into
  a `u64`, and the high-bit guarantee prevents overflow.
  `cells_per_slot = ⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`;
  fingerprint leads, value trails.

## Scheme/hash symmetry invariant

For any item, `all_indices(hash_item(item).indices[p], fingerprint)` must return
the same index set for every position `p < arity`. Verified in `hash.rs`.

## Adding a new scheme

1. Implement `IndexScheme` in `scheme.rs`
2. Add `hash_item_*` / `all_indices_*` in `hash.rs`
3. Add a constructor in `filter.rs`
4. If segmented, add a constructor in `store.rs` (KV-store is segmented-only)
5. Add a public alias in `lib.rs`
