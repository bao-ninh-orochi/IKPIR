# `HintDeltaBundle` wire format, version 1

Normative specification of the byte encoding of a `HintDeltaBundle`, the
mutation-phase transcript an `IkpirServer` publishes after each successful
`insert` / `update` / `delete` and every `IkpirClient` folds into its hint.
`crates/ikpir-common/src/wire.rs` implements it (`encode`, `decode`,
`wire_stats`, `wire_byte_size`); `crates/ikpir-client/tests/wire_transcript.rs`
and the unit tests in `wire.rs` enforce it. Where prose and code disagree, this
file is the specification and the code is wrong.

## 1. Why the format is what it is

The transcript names the cells of the per-segment database `D ∈ Z_p^{ρ×ω}` that
a mutation rewrote, each with its signed difference `γ = new − old`. The client
holds the hint `H = A·D` over `Z_q` (q = 2³²) and applies `H[·][c] += A[·][r]·γ`
for every named cell, so `γ` must be the exact integer difference: reducing it
modulo `p` would move `H` by `A[·][r]·p`. With `new, old ∈ [0, p)` the
difference lies in `(−p, p)`, `2p − 1` values, so it needs exactly
`⌈log₂(2p − 1)⌉ = plaintext_bits + 1` bits: 10 at p = 2⁹, 9 at p = 2⁸. Every
other field is an index into a domain both sides know from `CuckooParams`, and
is carried at the width of that domain. Nothing in this format is a parameter
that is not already fixed by the setup bundle, so nothing but the epoch and the
deltas themselves crosses the wire.

The paper prices the online phase at natural algebraic width (4 B per `Z_q`
element). This format prices the mutation phase the same way. It is a
specification of what the protocol sends, not a compression scheme.

## 2. Parameters and derived widths

All widths derive from the receiver's `CuckooParams` `P`; the encoder uses the
same `P`. `P` is never on the wire.

| symbol | definition | meaning |
|---|---|---|
| `d`  | `P.arity()` | number of segments |
| `n_b`| `P.num_buckets` | buckets in the whole store, `n_b = d·ρ` |
| `ρ`  | `P.segment_size()` = `n_b / d` | rows per segment (SCF buckets, **not** the SimplePIR reshape rows) |
| `ω`  | `P.bucket_size · P.cells_per_slot()` | cells per row (`row_width`, **not** the SimplePIR reshape width) |
| `p`  | `2^P.plaintext_bits` | cell modulus |
| `bitlen(x)` | `0` if `x = 0`, else `⌊log₂ x⌋ + 1` | bits needed to write `x` |
| **`IB`** | `bitlen(n_b − 1)` | width of a global bucket index `g ∈ [0, n_b)` |
| **`OW`** | `bitlen(ω − 1)` | width of a cell offset `∈ [0, ω)` and of a run length minus one `∈ [0, ω)` |
| **`DB`** | `P.plaintext_bits + 1` | width of one delta |
| **`G`**  | `(2·OW + 1) div DB` | longest gap of zero cells bridged inside one run |

`bitlen` is the width of an *index* (values `0 … n−1` need `bitlen(n−1)`
bits). A width may be 0 (a 0-bit field is not written and reads as 0).

Paper geometry, fingerprint 64 bits:

| shape | `n_b` | `IB` | ℓ | `pb` | `cells_per_slot` | `ω` | `OW` | `DB` | `G` |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| (2,4) | 2¹⁸ | 18 | 256 B | 9 | 235 | 940 | 10 | 10 | 2 |
| (2,4) | 2¹⁸ | 18 | 1 kB | 9 | 918 | 3672 | 12 | 10 | 2 |
| (2,4) RisePIR-S | 2¹⁸ | 18 | 1 kB | 8 | 1032 | 4128 | 13 | 9 | 3 |
| (4,1) | 2²⁰ | 20 | 256 B | 9 | 235 | 235 | 8 | 10 | 1 |
| (4,2) | 2¹⁹ | 19 | 256 B | 9 | 235 | 470 | 9 | 10 | 1 |
| (3,2) | 3·2¹⁸ | 20 | 256 B | 9 | 235 | 470 | 9 | 10 | 1 |
| (3,3) | 3·2¹⁷ | 19 | 256 B | 9 | 235 | 705 | 10 | 10 | 2 |

## 3. Bit stream

The encoding is a bit stream packed **least-significant bit first**: stream bit
`i` is bit `(i mod 8)` of byte `(i div 8)`. A field of width `w` and value `v`
occupies `w` consecutive stream bits; the `j`-th of them (from the field's
start) carries bit `j` of `v`, `j = 0` the least significant. After the last
field the stream is padded with zero bits to a byte boundary. The byte length
is `⌈bits / 8⌉`.

## 4. Grammar

```
bundle := epoch(64)  { 1(1) row }*  0(1)
row    := bucket(IB)  run  { 1(1) run }*  0(1)
run    := start(OW)  len_minus_1(OW)  delta(DB) × len
delta  := u = γ + (p − 1)        γ ∈ (−p, p),  u ∈ [0, 2p − 2]
```

- `epoch`: the bundle's `epoch` as an unsigned 64-bit integer.
- Each row is preceded by a 1-bit flag `1`; a `0` flag ends the bundle. A bundle
  with no touched rows is `epoch(64) 0(1)`, 9 bytes.
- `bucket`: the global bucket index `g = j·ρ + r` of the touched row, for
  segment `j ∈ [0, d)` and row-in-segment `r ∈ [0, ρ)`. The receiver splits it
  back as `j = g div ρ`, `r = g mod ρ`.
- A row holds one or more runs. Each run after the first is preceded by a 1-bit
  flag `1`; a `0` flag ends the row.
- `start`: offset of the run's first cell, `∈ [0, ω)`. `len_minus_1`: the run
  covers cells `start … start + len − 1`, `len ∈ [1, ω − start]`. Then `len`
  deltas in offset order, one per covered cell (zero deltas included).
- The code point `u = 2p − 1` (all `DB` bits set) is invalid.

## 5. Canonical form

Every conforming encoder emits, and every decoder accepts only, the canonical
form. Let `S` be the sparse set of a row's cells with nonzero `γ`, ordered by
offset.

1. **Rows** appear in strictly ascending `bucket` order, each at most once, and
   every row has at least one run. (Rows whose net delta is all zero are not
   rows: the fold drops them.)
2. **Runs** are the maximal groups of consecutive elements of `S` such that any
   two neighbours `a < b` in one group satisfy `b − a − 1 ≤ G`. Hence within a
   row runs are strictly ascending, two consecutive runs are separated by at
   least `G + 1` cells with zero delta, a run's first and last deltas are
   nonzero, and any interior zero stretch has length `≤ G`.
3. **Padding** bits are zero.

Rule 2 is the cost-optimal split for this grammar: bridging a gap of `g` zeros
costs `g·DB` bits, opening a new run costs `2·OW + 1`, and the decision at each
gap is independent of every other, so the greedy rule is optimal and unique.
Since the fold emits sorted rows and sorted nonzero cells, the canonical form is
a pure function of the in-memory bundle, and `decode(encode(b)) == b` holds
structurally as well as functionally.

## 6. Size

```
bits  = 64 + 1 + rows·(1 + IB) + runs·(2·OW + 1) + cells·DB
bytes = ⌈bits / 8⌉
```

where `rows` is the number of touched rows, `runs` the number of runs, and
`cells` the number of delta literals (run lengths summed, interior zeros
included). `HintDeltaBundle::wire_byte_size()` returns exactly this, from the
same `DeltaWireLayout` the encoder writes with, and
`encode(b).len() == b.wire_byte_size()` is enforced by test for every bundle.
`wire_stats()` additionally reports `nonzero_cells`, the size of `S`, which is
the `Θ(τ·w)` quantity the paper's asymptotics count.

Worked example, (2,4), ℓ = 256 B, pb = 9, an `update` of one slot whose 64-bit
fingerprint occupies cells 0–7 and is unchanged, value bits change in cells
7–234 (228 cells, all nonzero): one row, one run,
`64 + 1 + 1·(1 + 18) + 1·(2·10 + 1) + 228·10 = 2 385` bits = **299 bytes**. The pre-v1
accounting (`8 B/row + 10 B/cell + 12 + 4d`) reported 2 290 bytes for the same
bundle.

## 7. Decoding and validation

The decoder's input crosses a trust boundary. It must reject, with
`WireError`, any of: a stream that ends before a field is complete; `bucket ≥
n_b`; `start ≥ ω`; `start + len > ω`; a delta code `u > 2p − 2`; rows not
strictly ascending; runs not strictly ascending or closer than `G + 1` zero
cells; a run whose first or last delta is zero, or with an interior zero
stretch longer than `G`; non-zero padding bits; bytes after the padding.
Bounds are checked before any index is used. The decoder drops literal zero
deltas, so the result is the sparse in-memory form.

The result carries the `CuckooParams` it was decoded under, and
`IkpirClient::apply_delta` rejects a bundle whose params differ from the
client's.

## 8. Encoding preconditions

The encoder's input is internal (the fold's output). It **panics** if the
bundle is not canonical or a field is out of range: `|γ| ≥ p`, an offset `≥
ω`, a row `≥ ρ`, a segment count `≠ d`, unsorted or duplicate rows or offsets,
or a zero delta in the sparse input. A violation is a bug in the fold and must
be loud, never silently truncated. `|γ| < p` is a protocol invariant: repeated
writes to one cell within a commit telescope to `final − initial`, which is
again in `(−p, p)`.

## 9. Out of scope

Message framing (length prefixes), version negotiation, and compression belong
to the transport. A server may publish, to a client `k` epochs behind, either
the `k` bundles or a fresh `ServerSetupBundle`, whichever is smaller; that
choice is a deployment policy, and both sizes are reported by
`benches/server_mutation.rs` so the comparison is arithmetic on the CSV row. A
batched `DBMutation(ops_1..ops_τ)` sharing one epoch across a batch is a
separate protocol change and is not part of this format.
