# `FingerprintTable` — storage design notes

`FingerprintTable` (`src/bucket.rs:32-46`) is the physical storage layer of every
`CuckooFilter`. It packs fingerprints of arbitrary bit width (1–32 bits) back-to-back
in a single `Vec<u8>` and exposes O(1) `read_fingerprint` / `write_fingerprint`
primitives. This document explains four design choices that are not obvious from the
code alone.

---

## 1. Flat byte buffer

```rust
pub struct FingerprintTable {
    data: Vec<u8>,                  // src/bucket.rs:39
    pub num_buckets: u32,
    pub fingerprints_per_bucket: u32,
    pub fingerprint_bits: u32,
}
```

`data` is one contiguous heap allocation of

```
ceil(num_buckets * fingerprints_per_bucket * fingerprint_bits / 8) + 8
```

bytes (`src/bucket.rs:71-81`). The first term is the minimum space needed to hold
all fingerprint bits packed without padding; the trailing `+ 8` is read-window
padding (see §3). Buckets and slots are a purely arithmetic notion at this layer:
fingerprint `(bucket, slot)` lives at bit offset
`(bucket * fingerprints_per_bucket + slot) * fingerprint_bits`
(`src/bucket.rs:113-120`). There are no per-bucket headers, no inter-slot padding,
and no byte alignment between fingerprints.

**Layout for `fingerprint_bits = 12`, `fingerprints_per_bucket = 4`:**

```text
byte:   |   0    |   1    |   2    |   3    |   4    |   5    | ...
bits:   |76543210|76543210|76543210|76543210|76543210|76543210|
fp #0:   <----- 12 ----->                                          bits  0..11
fp #1:                       <----- 12 ----->                      bits 12..23
fp #2:                                                <-- 12 ...   bits 24..35  (crosses byte 3 -> 4)
```

**Alternatives rejected:**

| Alternative | Cost vs. flat `Vec<u8>` |
|---|---|
| `Vec<u16>` / `Vec<u32>` slot array | Wastes the unused high bits. A 12-bit fingerprint in `u16` costs 33% extra memory; in `u32`, 167% extra. The cuckoo filter's FPR/space tradeoff is *not* byte-quantised, so byte-aligning the storage throws away a real lever. |
| `bitvec` / `bitfield` crate | Adds a dependency, an extra indirection or block-management layer, and a layout that varies with feature flags — making (de)serialisation fragile. |
| Per-bucket `[u32; N]` (typed slots) | Same waste as `Vec<u32>`, plus loses the single-allocation property and the stable on-disk layout. |

The flat buffer wins on three axes simultaneously: minimal memory, single
allocation (good for cache and prefetching — a 2-bucket probe usually touches
1–2 cache lines), and a layout that *is* the wire format with zero conversion.
Its only cost is unaligned access, which is free on x86_64 and aarch64.

---

## 2. Why `u64` masks despite the 32-bit cap

The fingerprint width is constrained to `1..=32` upstream, in
`validate_common_params` (`src/filter.rs:225-228`):

```text
let min_fingerprint_bits = (arity * bucket_size).ilog2() + 1;
if fingerprint_bits < min_fingerprint_bits || fingerprint_bits > 32 { ... }
```

Every `CuckooFilter` constructor calls `validate_common_params` before
`FingerprintTable::new`, so `bucket.rs` can rely on `fingerprint_bits ≤ 32`.

Yet the mask is built in `u64`:

```rust
let mask = (1u64 << self.fingerprint_bits) - 1;   // src/bucket.rs:162, 210
```

Three reasons:

- **Avoids `1u32 << 32` UB at the boundary.** With `fingerprint_bits == 32`,
  `1u32 << 32` is undefined behaviour in Rust (shift by ≥ bit-width). `1u64 << 32`
  is well-defined and equals `0x1_0000_0000`, so `mask = 0xFFFF_FFFF` is the
  correct full-width mask. Computing in `u64` removes the need for a special case.
- **The window is already a `u64`.** `read_fingerprint` loads a `u64` from
  `data[byte_pos..byte_pos + 8]` (`src/bucket.rs:165-169`) and shifts by up to 7
  bits. The masked field naturally lives in `u64` too; using `u32` here would
  force an extra truncation and re-widen.
- **No cost.** A `u64` mask fits in one register on every 64-bit target. The final
  cast back to `u32` (`src/bucket.rs:170`) is lossless because `mask ≤ 2^32 − 1`.

The `1..=32` cap is the *contract* the storage relies on; the `u64` arithmetic
is what lets a single code path serve the entire contracted range without a
branch on width.

---

## 3. Why a fixed 8-byte load (not the minimum 5)

The minimum bytes that can possibly contain a 32-bit field starting at a
sub-byte offset is

```text
ceil((bit_shift + fingerprint_bits) / 8)  ≤  ceil((7 + 32) / 8)  =  5
```

so a 5-byte unaligned load would suffice. The code reads 8 (`src/bucket.rs:165-169`,
`src/bucket.rs:212-216`) for these reasons:

- **One instruction.** On every 64-bit ISA Rust targets, an unaligned 8-byte load
  is a single `MOV` (x86_64) or `LDR` (aarch64). A 5-byte load has no
  corresponding instruction; it would have to be synthesised from a 4-byte load
  plus a 1-byte load and a shift — strictly more work, more branches, more
  register pressure.
- **Constant length → better codegen.** `from_le_bytes::<[u8; 8]>` produces
  monomorphic, branch-free code. A variable-length read keyed on `fingerprint_bits`
  defeats that.
- **The 8-byte tail padding makes it unconditional.** `new` allocates
  `total_bytes + 8` (`src/bucket.rs:76`). For *any* legal `(bucket, slot)` the
  slice `data[byte_pos..byte_pos + 8]` is in-bounds, so there is no end-of-buffer
  branch and `try_into::<[u8; 8]>` cannot fail. The `expect` on
  `src/bucket.rs:168, 215` documents this as an invariant, not error handling.

The cost is 8 bytes of overhead per table — irrelevant compared to the data
buffer for any realistic filter.

---

## 4. Why little-endian / LSB-first packing

The packing is "LSB-first" in two senses:

1. **Across bytes:** the window is a little-endian `u64`
   (`u64::from_le_bytes` / `to_le_bytes`, `src/bucket.rs:165, 219`). Byte
   `byte_pos + k` of the buffer maps to bits `8k..8k+7` of the loaded `u64`.
2. **Within a byte:** the lowest-numbered fingerprint occupies the *low* bits.
   A fingerprint at bit offset 0 lives in bits 0..(f-1) of byte 0.

This combination buys two concrete properties:

- **`shift count = bit_pos % 8`, both directions.** Extraction is
  `(val >> bit_shift) & mask`; insertion is `val |= (fp & mask) << bit_shift`.
  Same shift count, both shifts go "the natural way". With MSB-first packing,
  the shift count would be something like `64 - (bit_pos % 8) - fingerprint_bits`,
  which is fragile and easy to off-by-one for byte-crossing fields.
- **Byte-crossing fields need no special case.** Because higher byte indices
  correspond to higher bit positions in the loaded `u64`, a 12-bit field starting
  at bit 4 of byte 0 occupies bits 4..15 of `val` — one mask, one shift, done.
  An MSB-first layout would require splitting the field into a "this byte" and
  "next byte" piece with two masks.

**Why little-endian specifically:** every platform Rust currently targets in
practice (x86, x86_64, aarch64, riscv64, wasm) is natively little-endian, so
`from_le_bytes` / `to_le_bytes` compile to a plain unaligned load/store with no
byte swap. The choice also pins the on-memory and on-disk layout: a filter built
on one host reads identically on another regardless of host endianness.

---

## Summary

| Property | Cost per access |
|---|---|
| `read_fingerprint` | 1 unaligned `u64` load + shift + mask |
| `write_fingerprint` | 1 unaligned `u64` load + clear + OR + 1 store |
| `find_fingerprint_in_bucket` | `O(fingerprints_per_bucket)` — short fixed loop, ≤ 8 in practice |

**Caller-enforced invariants (not checked here):**

- `fingerprint_bits ∈ 1..=32` — enforced in `validate_common_params`
  (`src/filter.rs:225-228`).
- Fingerprint value `0` is reserved to mean "empty slot"; the hash layer must
  never produce it for a real key.
- `bucket < num_buckets` and `slot < fingerprints_per_bucket` — bounds-checked
  only by `Vec` indexing.
