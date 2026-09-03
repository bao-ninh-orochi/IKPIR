# Response-rewind client update mode

Design note for the client's second update strategy. `IkpirClient` keeps its
per-segment state consistent with server mutations in one of two selectable
[`ClientUpdateMode`](../crates/ikpir-common/src/backend/mod.rs)s. Both consume
the same `HintDeltaBundle` stream (`docs/hint-delta-wire-format.md`) and return
the **same** decoded value for every query; only *when* and *where* the
published `ΔD` is spent differs, so the mode is a purely local client choice the
server never sees — like `HintPatchMode`, one level up. The implementation lives
in `crates/ikpir-client/src/{client.rs,pending.rs}` and
`crates/ikpir-common/src/backend/` (`ResponseRewind`); the equivalence is pinned
by `crates/ikpir-client/tests/rewind_equivalence.rs`. Where prose and code
disagree, the tests are the specification.

## 1. The two modes

The client holds per-segment state `(A, H)` and an LWE secret per query. A
mutation batch publishes the sparse per-cell `ΔD` (`new − old`, exact integers,
`docs/hint-delta-wire-format.md`).

- **Hint-patch** (`HintPatchMode`; the pre-existing path). Fold each delta into
  the hint immediately: `H ← H + Σ A[:,c]·γ`, cost **Θ(n·τ·ω)** per batch
  (`n` = LWE dimension, `τ` = mutations, `ω` = row width). Decode is a direct
  `client_decode` against the up-to-date hint. Entry points: `apply_delta`,
  `decode`.

- **Response-rewind** (the default). Pin the bootstrap hint `H₀ = Aᵀ·D₀` and
  never patch it; instead **accumulate** the published deltas into a running
  `ΔD = D_head − D₀`, cost **Θ(τ·ω)** per batch — a factor-`n` cheaper client
  maintenance. Pay a per-query correction that grows with the staleness `|ΔD|`,
  reclaimable by folding `ΔD` into the hint on demand. Entry points:
  `accumulate_delta`, `decode_rewind`, `collect_garbage`.

`ClientUpdateMode::Rewind` is the default (matching the eth_getBalance
deployment this technique came from). Hint-patch is fully supported when
selected; `decode` / `apply_delta` are behaviour-preserving there, and return
`WrongUpdateMode` in rewind mode.

## 2. The mechanism

The server answers at its **head** `D'`; a rewind client decodes against its
**pinned** `H₀`. For a FrodoPIR segment, the query is
`q = A·s + e + Δ·u_row` (the marker-bearing vector the server multiplied) and
the response is `a = qᵀ·D'`. In four steps (`decode_rewind`, per segment):

1. **rewind** `a ← a − qᵀ·ΔD` (`ResponseRewind::rewind_response`). Since
   `D' − ΔD = D₀`, this is `a = qᵀ·D₀`, exact in `Z_2³²` — **no added noise**.
2. **decode** `client_decode(H₀, a)` → the row **as of the pin**, `D₀[row]`.
3. **add** `cells += ΔD[row]` → the row **as of the head**, `D'[row]`. Must
   precede step 4.
4. **scan** the same branchless fingerprint scan as `decode`.

Per backend, the map from a segment cell `(row, offset)` to the response index
it contributes to differs: FrodoPIR keeps the tall-skinny layout
(`a[off] -= q.b[row]·γ`); SimplePIR folds through its near-square reshape
(`big_r = row/k`, `big_c = (row%k)·ω + off`, `a[big_c] -= q.b[big_r]·γ`, reshape
parameters read off the public client state). Both read only public fields.
`γ as u32` truncating the two's-complement `i64` to its low 32 bits **is** the
reduction mod `2³²`.

## 3. Correctness

**Why the correction is exact.** `qᵀ·D'` and `qᵀ·D₀` differ by exactly `qᵀ·ΔD`
in `Z_2³²`; subtracting it recovers `qᵀ·D₀` with the *same* noise budget as a
fresh query against `D₀`. Decoding against `H₀` therefore succeeds under the
same Lemma-2 per-cell bound the scheme already targets.

**Why step 3 never overflows.** The server publishes each per-cell delta as
`new − old` with `new, old ∈ [0, p)`, so the accumulated `ΔD[cell]` telescopes
to `current − pinned`, both in `[0, p)`, hence `ΔD ∈ (−p, p)` and
`pinned + ΔD = current ∈ [0, p)` exactly. `decode_rewind`'s `[0, p)` check on
each corrected cell is therefore a pure integrity check on a corrupt or
inconsistent delta/response (`CellOutOfRange`) — it never fires in honest
operation, and never returns a wrong value.

**Equivalence.** rewind, hint-patch (H patched to `Aᵀ·D'`), and a fresh setup at
the head all recover the true `D'[row]` whenever decode succeeds, so the
returned value is identical across all three. Pinned by
`tests/rewind_equivalence.rs` for both backends and arities 2/3/4, including a
post-pin insert, a garbage-collected client, and negative controls.

## 4. Epoch model

`epoch` is the head the client tracks; `pin_epoch` is where `H₀` sits.

- **Hint-patch:** `pin_epoch == epoch` always (the hint tracks the head).
- **Rewind:** `pin_epoch ≤ epoch`; `ΔD` covers `[pin_epoch, epoch]`. The pin
  advances only on `collect_garbage` (fold `ΔD` into `H`, re-pin at the head,
  clear `ΔD`) or a resync (`reset_from`).

`accumulate_delta` is **strict-monotone** exactly like `apply_delta`: only
`delta.epoch == epoch + 1` is accepted (`StaleDelta` / `FutureDelta`); the
library consumes one `HintDeltaBundle` per epoch, and a gap recovers via
`reset_from` (which clears `ΔD` and re-pins). `decode_rewind` gates
`resp.epoch == epoch` (`EpochMismatch`). Garbage collection is **never required
for correctness** — a rewind client stays correct indefinitely; GC only trades a
one-off `Θ(|ΔD|·n)` patch for a cheaper steady-state decode.

## 5. API

| Task | Hint-patch | Rewind |
|---|---|---|
| Consume a published delta | `apply_delta` | `accumulate_delta` |
| Decode a response | `decode(key, resp)` | `decode_rewind(key, query, resp)` |
| Reclaim staleness | — | `collect_garbage` |
| Mode | `set_update_mode(HintPatch)` | default |

`decode_rewind` threads the query bundle the caller already holds (its
marker-bearing `b` vectors drive the correction) and works in either mode (in
hint-patch mode `ΔD` is empty, so it reduces to `decode`). `build_query`,
`reset_from`, precompute, and the `&self` `decode` contract are unchanged. No
Cargo feature: the backend is monomorphised on `B` and the mode is a runtime
enum. Both modes are preserved across `reset_from`.

## 6. Scope

- Backend-generic over RisePIR-F and RisePIR-S (`ResponseRewind` impls for both).
- The **server is unchanged**: `HintDeltaBundle` already carries the raw sparse
  `ΔD`; rewind is a pure client-side alternative *consumer* of the same
  transcript.
- No value codec — the existing fingerprint scan and `Option<Vec<u8>>` output
  are reused; only the correction and the step-3 add are new before the scan.
- The timed benchmark path stays single-threaded and non-SIMD.
- **Side channel (client-local).** `decode_rewind`'s step 3 (the per-row `ΔD`
  add) iterates a `BTreeMap` range keyed on the queried row, so its timing
  depends on that row — a client-local leak of a row index the client itself
  chose, never reaching the server or the response. Steps 1–2 and the
  fingerprint scan retain `decode`'s slot-independent hardening; a fully
  constant-time decode is out of scope for this prototype
  (`ikpir-common/CLAUDE.md` §3). Since rewind is the default mode, this is the
  default decode path.
