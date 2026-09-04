# Response-rewind: the client's update strategy

Design note for the client's (sole) update strategy. `IkpirClient` keeps its
per-segment state consistent with server mutations via **response-rewind**:
it never patches its hint, and instead accumulates the published sparse
`HintDeltaBundle` stream (`docs/hint-delta-wire-format.md`) into a rolling
`ΔD`, correcting each response back to its pinned hint's epoch at decode time.
This is a factor-`n` cheaper client maintenance than the classical
hint-patching alternative (`n` = LWE dimension) — the technique this repo
adopted from the eth_getBalance deployment it originated in.

The classical hint-patching path — fold every delta into the hint immediately
— survives **only** as a benchmark comparator (`HintPatchClient`, gated
behind the `hint-patch-bench` Cargo feature, disabled by default) so
`benches/client_mutation.rs` can still measure the two head-to-head for the
paper's §6.2 evaluation. It is not part of the production client. The
implementation lives in `crates/ikpir-client/src/{client.rs,pending.rs}`
(production) and `crates/ikpir-client/src/bench_comparator/` (the
comparator), plus `crates/ikpir-common/src/backend/` (`ResponseRewind`); the
equivalence between the two is pinned by
`crates/ikpir-client/tests/rewind_equivalence.rs` (also feature-gated). Where
prose and code disagree, the tests are the specification.

## 1. The two paths

The client holds per-segment state `(A, H)` and an LWE secret per query. A
mutation batch publishes the sparse per-cell `ΔD` (`new − old`, exact integers,
`docs/hint-delta-wire-format.md`).

- **Response-rewind** (the production client, `ikpir-client::IkpirClient`).
  Pin the bootstrap hint `H₀ = Aᵀ·D₀` and never patch it; instead
  **accumulate** the published deltas into a running `ΔD = D_head − D₀`, cost
  **Θ(τ·ω)** per batch (`τ` = mutations, `ω` = row width) — a factor-`n`
  cheaper client maintenance than hint-patching. Pay a per-query correction
  that grows with the staleness `|ΔD|`, reclaimable by folding `ΔD` into the
  hint on demand. Entry points: `accumulate_delta`, `decode`,
  `collect_garbage`.

- **Hint-patch** (the bench-only comparator, `ikpir-client::HintPatchClient`,
  behind `hint-patch-bench`). Fold each delta into the hint immediately:
  `H ← H + Σ A[:,c]·γ`, cost **Θ(n·τ·ω)** per batch (`n` = LWE dimension).
  Decode is a direct `client_decode` against the up-to-date hint. Entry
  points: `apply_delta`, `decode`.

Both consume the same `HintDeltaBundle` stream and return the **same**
decoded value for every query — only *when* and *where* the published `ΔD` is
spent differs. That equivalence is what `tests/rewind_equivalence.rs` pins;
it is what lets the paper's §6.2 comparison trust that `HintPatchClient`'s
numbers describe the same protocol response-rewind implements more cheaply.

## 2. The mechanism

The server answers at its **head** `D'`; the client decodes against its
**pinned** `H₀`. For a FrodoPIR segment, the query is
`q = A·s + e + Δ·u_row` (the marker-bearing vector the server multiplied) and
the response is `a = qᵀ·D'`. In four steps (`IkpirClient::decode`, per
segment):

1. **rewind** `a ← a − qᵀ·ΔD` (`ResponseRewind::rewind_response`). Since
   `D' − ΔD = D₀`, this is `a = qᵀ·D₀`, exact in `Z_2³²` — **no added noise**.
2. **decode** `client_decode(H₀, a)` → the row **as of the pin**, `D₀[row]`.
3. **add** `cells += ΔD[row]` → the row **as of the head**, `D'[row]`. Must
   precede step 4.
4. **scan** the branchless fingerprint scan.

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
`pinned + ΔD = current ∈ [0, p)` exactly. `decode`'s `[0, p)` check on
each corrected cell is therefore a pure integrity check on a corrupt or
inconsistent delta/response (`CellOutOfRange`) — it never fires in honest
operation, and never returns a wrong value.

**Equivalence.** rewind, hint-patch (`H` patched to `Aᵀ·D'`), and a fresh setup
at the head all recover the true `D'[row]` whenever decode succeeds, so the
returned value is identical across all three. Pinned by
`tests/rewind_equivalence.rs` for both backends and arities 2/3/4, including a
post-pin insert and a garbage-collected client.

## 4. Epoch model

`epoch` is the head the client tracks; `pin_epoch` is where `H₀` sits.
`pin_epoch ≤ epoch`; `ΔD` covers `[pin_epoch, epoch]`. The pin advances only
on `collect_garbage` (fold `ΔD` into `H`, re-pin at the head, clear `ΔD`) or a
resync (`reset_from`). (The bench-only `HintPatchClient` has no separate pin:
its hint tracks the head directly, so its `epoch` alone describes its state.)

`accumulate_delta` is **strict-monotone**: only `delta.epoch == epoch + 1` is
accepted (`StaleDelta` / `FutureDelta`); the library consumes one
`HintDeltaBundle` per epoch, and a gap recovers via `reset_from` (which clears
`ΔD` and re-pins). `decode` gates `resp.epoch == epoch` (`EpochMismatch`).
Garbage collection is **never required for correctness** — a rewind client
stays correct indefinitely; GC only trades a one-off `Θ(|ΔD|·n)` patch for a
cheaper steady-state decode.

## 5. API

| Task | Production (rewind) | Bench comparator (hint-patch) |
|---|---|---|
| Consume a published delta | `IkpirClient::accumulate_delta` | `HintPatchClient::apply_delta` |
| Decode a response | `IkpirClient::decode(key, query, resp)` | `HintPatchClient::decode(key, resp)` |
| Reclaim staleness | `collect_garbage` | — (hint is always current) |
| Availability | always (`ikpir-client` default features) | `hint-patch-bench` feature only |

`IkpirClient::decode` threads the query bundle the caller already holds (its
marker-bearing `b` vectors drive the correction); `build_query`, `reset_from`,
and precompute are otherwise unchanged from before the pivot. No Cargo
feature gates the production path — the backend is monomorphised on `B` and
there is exactly one client type. The comparator path is a separate type
(`HintPatchClient<B>`, not a runtime mode on `IkpirClient`), so there is
nothing to "switch" — the two never coexist on one client instance.

## 6. Scope

- Backend-generic over RisePIR-F and RisePIR-S (`ResponseRewind` impls for both).
- The **server is unchanged**: `HintDeltaBundle` already carries the raw sparse
  `ΔD`; rewind is a pure client-side *consumer* of the same transcript. The
  server's live hint (patched in `commit_mutations`, handed out at
  `setup()` / `full_rebuild()`) is the sole hint-maintenance path in the
  system — it is what lets a newly-bootstrapping client (or one recovering
  from a `FutureDelta` gap) start from an up-to-date hint without replaying
  history.
- No value codec — the existing fingerprint scan and `Option<Vec<u8>>` output
  are reused; only the correction and the step-3 add are new before the scan.
- The timed benchmark path stays single-threaded and non-SIMD.
- **Side channel (client-local).** `decode`'s step 3 (the per-row `ΔD`
  add) iterates a `BTreeMap` range keyed on the queried row, so its timing
  depends on that row — a client-local leak of a row index the client itself
  chose, never reaching the server or the response. Steps 1–2 and the
  fingerprint scan retain slot-independent hardening; a fully
  constant-time decode is out of scope for this prototype
  (`ikpir-common/CLAUDE.md` §3).
