//! `FrodoPirBackend` — IndexPirBackend impl over `Z_{2^32}` with ternary
//! errors.
//!
//! # Purpose
//!
//! The shipped IKPIR backend: implements
//! [`IndexPirBackend`](crate::IndexPirBackend),
//! [`IncrementalPirBackend`](crate::IncrementalPirBackend),
//! [`PrecomputingPirBackend`](crate::PrecomputingPirBackend), and
//! [`BackendWireSize`](crate::BackendWireSize) for FrodoPIR.
//!
//! # Design / architecture
//!
//! - **Witness type.** [`FrodoPirBackend`] is zero-sized; all behaviour
//!   lives on the trait impls.
//! - **Per-segment state.** Each segment has its own
//!   [`FrodoServerParams`] / [`FrodoHint`] (server side) and
//!   [`FrodoClientState`] (client side). The IKPIR server constructs
//!   `arity` instances of each.
//! - **Precomputation queue.** `FrodoClientState` holds a `VecDeque` of
//!   [`PreparedSlot`]s and at most one in-flight slot. The math in
//!   `frodo/mod.rs` corresponds 1:1 to the field names (`b = A·s + e`,
//!   `c = sᵀ·H`).
//! - **Hot loops.** `server_answer`'s matvec and `client_decode`'s
//!   `residual − c` (or `residual − sᵀ·H` on the cold path) dominate
//!   CPU time. Both are deliberately unconditional (no `sk == 0`
//!   short-circuit) so timing does not leak the secret's Hamming
//!   weight.
//!
//! # Related files
//!
//! - `mod.rs` — re-exports the public types here; carries the math
//!   summary for the whole module.
//! - `params.rs` — `FrodoConfig` / `FrodoParams`.
//! - `arith.rs` — `round_p_to_q` / `round_q_to_p` (Δ-scaling).
//! - `sampler.rs` — `sample_a` / `sample_ternary_into`.

use std::collections::VecDeque;

use rand::RngCore;

use super::{FrodoConfig, FrodoParams, round_p_to_q, round_q_to_p, sample_a, sample_ternary_into};
use crate::backend::{BackendWireSize, IndexPirBackend, IncrementalPirBackend, PrecomputingPirBackend};

/// Zero-sized witness type that carries the [`IndexPirBackend`] /
/// [`IncrementalPirBackend`] / [`PrecomputingPirBackend`] /
/// [`BackendWireSize`] impls for FrodoPIR.
///
/// # Purpose
///
/// The IKPIR server / client are generic over the backend; this type
/// names the FrodoPIR specialisation.
///
/// # Rationale
///
/// Zero-sized so the value is free to construct and pass by value; all
/// methods are static — see the trait impls below.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrodoPirBackend;

/// Per-segment public parameters: LWE dimensions, plaintext modulus,
/// and the 16-byte seed used to expand the public matrix `A`.
///
/// # Purpose
///
/// One instance per segment; combined with [`FrodoHint`] it forms the
/// `(ServerParams, Hint)` pair every IKPIR client receives at setup. The
/// matrix `A` itself is **not** carried here — see [`FrodoHintMaterial`].
#[derive(Clone, Debug)]
pub struct FrodoServerParams {
    /// LWE dimension, plaintext bits, and 16-byte seed used to sample `a`.
    pub params:    FrodoParams,
    /// Number of database rows in this segment.
    pub n_rows:    u32,
    /// Number of `u32` cells per database row.
    pub row_width: u32,
}

/// Server-local working state: the LWE public matrix `A` in row-major
/// shape `n_rows × lwe_dim`, expanded deterministically from
/// [`FrodoServerParams::params`]`.seed` via [`sample_a`].
///
/// # Purpose
///
/// Used by [`FrodoPirBackend::server_setup`] to compute the hint and by
/// [`FrodoPirBackend::server_patch_hint`] to keep the hint coherent
/// across mutations. **Not part of the wire payload** — the client
/// re-expands its own copy from the seed during
/// [`FrodoPirBackend::client_setup`], and the server may drop and
/// re-expand its copy via
/// [`IkpirServer::drop_hint_material`](../../../ikpir_server/struct.IkpirServer.html#method.drop_hint_material).
///
/// # Rationale
///
/// Pulling `A` out of [`FrodoServerParams`] keeps the wire bundle small
/// (only the 16-byte seed travels) and lets the server free `A` on
/// pure-read workloads. Not `Clone`: every "extra" `A` buffer must be
/// an explicit [`FrodoPirBackend::expand_hint_material`] call so
/// accidental duplication is impossible.
#[derive(Debug, Default)]
pub struct FrodoHintMaterial {
    /// Public matrix `A` in row-major shape `n_rows × lwe_dim`.
    pub a: Vec<u32>,
}

/// FrodoPIR hint matrix `H = Aᵀ · D mod 2³²` in row-major shape
/// `lwe_dim × row_width`.
///
/// # Purpose
///
/// Held by both server and (a copy at) client; the matvec
/// `client_decode` performs against this matrix is the dominant CPU
/// cost.
///
/// # Rationale
///
/// Derived once at setup and patched in place thereafter via
/// [`FrodoPirBackend::server_patch_hint`] /
/// [`FrodoPirBackend::client_patch_state`] — never recomputed unless
/// the server triggers `full_rebuild`.
#[derive(Clone, Debug, PartialEq)]
pub struct FrodoHint {
    /// Flat row-major buffer of length `lwe_dim × row_width`.
    pub data: Vec<u32>,
}

/// One precomputed query slot: the LWE secret, the matching public-half
/// query vector `b = A·s + e`, and (optionally) the decode-side material
/// `c = sᵀ·H`. `c` is `None` until [`FrodoPirBackend::client_precompute_decodes`]
/// runs. Internal to this module — consumers go through the public `prepared`
/// / `in_flight` accessor methods on `FrodoClientState`.
struct PreparedSlot {
    secret: Vec<u32>,
    b:      Vec<u32>,
    c:      Option<Vec<u32>>,
}

/// Per-segment client-held FrodoPIR state.
///
/// # Purpose
///
/// Holds everything the client needs to build queries and decode
/// responses for one segment: a copy of [`FrodoServerParams`], the
/// patched [`FrodoHint`], a FIFO queue of slots prepared by
/// [`FrodoPirBackend::client_precompute_queries`], and the slot
/// consumed by the most recent
/// [`FrodoPirBackend::client_query`].
///
/// # Constraints
///
/// **Single in-flight query.** Each `client_query` overwrites the
/// in-flight slot; issuing a second `client_query` before decoding the
/// first discards the first's secret and that first decode will return
/// garbage. This matches the protocol cadence in `IkpirClient` (one
/// query → one answer → one decode per round per segment).
pub struct FrodoClientState {
    /// Public parameters for this segment.
    pub params:        FrodoServerParams,
    /// Locally expanded copy of the LWE public matrix `A`, re-derived
    /// from `params.params.seed` during `client_setup`.
    pub hint_material: FrodoHintMaterial,
    /// Locally maintained copy of the segment hint matrix.
    pub hint:          FrodoHint,
    /// Prepared but unconsumed slots, FIFO. Front is the next slot a
    /// `client_query` will use; back is where `client_precompute_queries`
    /// appends.
    prepared:          VecDeque<PreparedSlot>,
    /// The slot consumed by the most recent `client_query`, if any. Read
    /// by the matching `client_decode`.
    in_flight:         Option<PreparedSlot>,
}

impl FrodoClientState {
    /// Number of prepared-but-unconsumed slots (filled by
    /// `client_precompute_queries`, drained by `client_query`).
    pub fn prepared_len(&self) -> usize { self.prepared.len() }

    /// Whether there is an in-flight query awaiting decode (always 0 or 1).
    pub fn in_flight_len(&self) -> usize { self.in_flight.is_some() as usize }
}

/// Wire-level FrodoPIR query: `b = A·s + e + Δ·u_row` of length `n_rows`.
///
/// # Purpose
///
/// The client-side ciphertext shipped to the server. One per segment.
#[derive(Clone, Debug)]
pub struct FrodoQuery {
    /// Encrypted query vector (`b = A·s + e + Δ·u_row`).
    pub b: Vec<u32>,
}

/// Wire-level FrodoPIR response: `a = bᵀ·D` of length `row_width`.
///
/// # Purpose
///
/// The server-side ciphertext shipped back to the client. One per
/// segment.
#[derive(Clone, Debug)]
pub struct FrodoResponse {
    /// Encrypted response vector (`a = bᵀ·D`).
    pub a: Vec<u32>,
}

impl IndexPirBackend for FrodoPirBackend {
    type Config       = FrodoConfig;
    type ServerParams = FrodoServerParams;
    type HintMaterial = FrodoHintMaterial;
    type Hint         = FrodoHint;
    type ClientState  = FrodoClientState;
    type Query        = FrodoQuery;
    type Response     = FrodoResponse;

    fn server_setup(
        config: &FrodoConfig,
        db: &[u32],
        n_rows: u32,
        row_width: u32,
        plaintext_bits: u32,
    ) -> (FrodoServerParams, FrodoHintMaterial, FrodoHint) {
        debug_assert_eq!(db.len(), (n_rows as usize) * (row_width as usize));
        let lwe_dim = config.lwe_dim;

        let mut seed = [0u8; 16];
        rand::rng().fill_bytes(&mut seed);
        let params = FrodoParams::new(lwe_dim, plaintext_bits, seed);

        let a = sample_a(&seed, n_rows, lwe_dim);
        let hint_data = compute_hint(&a, db, n_rows, lwe_dim, row_width);

        (
            FrodoServerParams { params, n_rows, row_width },
            FrodoHintMaterial { a },
            FrodoHint { data: hint_data },
        )
    }

    fn expand_hint_material(params: &FrodoServerParams) -> FrodoHintMaterial {
        let a = sample_a(&params.params.seed, params.n_rows, params.params.lwe_dim);
        FrodoHintMaterial { a }
    }

    fn client_setup(params: &FrodoServerParams, hint: &FrodoHint) -> FrodoClientState {
        FrodoClientState {
            params:        params.clone(),
            hint_material: Self::expand_hint_material(params),
            hint:          hint.clone(),
            prepared:      VecDeque::new(),
            in_flight:     None,
        }
    }

    fn client_query(state: &mut FrodoClientState, row: u32) -> FrodoQuery {
        let n_rows         = state.params.n_rows;
        let plaintext_bits = state.params.params.plaintext_bits;
        debug_assert!(row < n_rows, "row {row} out of range (n_rows={n_rows})");

        // Cheap path if a prepared slot is available; otherwise sample inline
        // (matches the pre-precomputation behaviour for backward compatibility).
        let slot = state.prepared.pop_front()
            .unwrap_or_else(|| sample_slot(&state.params, &state.hint_material));

        let mut b = slot.b.clone();
        let delta = round_p_to_q(1, plaintext_bits);
        b[row as usize] = b[row as usize].wrapping_add(delta);

        state.in_flight = Some(slot);
        FrodoQuery { b }
    }

    fn server_answer(
        _params: &FrodoServerParams,
        db: &[u32],
        n_rows: u32,
        row_width: u32,
        query: &FrodoQuery,
    ) -> FrodoResponse {
        debug_assert_eq!(query.b.len(), n_rows as usize);
        debug_assert_eq!(db.len(), (n_rows as usize) * (row_width as usize));
        let mut a = vec![0u32; row_width as usize];
        for i in 0..n_rows as usize {
            let qi = query.b[i];
            let row_off = i * row_width as usize;
            for j in 0..row_width as usize {
                a[j] = a[j].wrapping_add(qi.wrapping_mul(db[row_off + j]));
            }
        }
        FrodoResponse { a }
    }

    /// Inner loop is unconditional (no `sk == 0` short-circuit) so timing
    /// does not leak the secret's Hamming weight: `sk.wrapping_mul(0) = 0`
    /// and `x.wrapping_sub(0) = x`, so the arithmetic is unchanged.
    fn client_decode(state: &FrodoClientState, response: &FrodoResponse) -> Vec<u32> {
        let row_width      = state.params.row_width as usize;
        let plaintext_bits = state.params.params.plaintext_bits;
        debug_assert_eq!(response.a.len(), row_width);

        let slot = state.in_flight.as_ref()
            .expect("client_decode called without a matching client_query");

        // residual = response - c, where c = sᵀ·H (precomputed if present,
        // otherwise materialised on the fly). The two paths are arithmetically
        // identical; the cheap path just reuses already-multiplied values.
        let mut residual = response.a.clone();
        match slot.c.as_ref() {
            Some(c) => {
                debug_assert_eq!(c.len(), row_width);
                for j in 0..row_width {
                    residual[j] = residual[j].wrapping_sub(c[j]);
                }
            }
            None => {
                let lwe_dim = state.params.params.lwe_dim as usize;
                let h = &state.hint.data;
                for k in 0..lwe_dim {
                    let sk = slot.secret[k];
                    let row_off = k * row_width;
                    for j in 0..row_width {
                        residual[j] = residual[j].wrapping_sub(sk.wrapping_mul(h[row_off + j]));
                    }
                }
            }
        }

        residual.iter().map(|&y| round_q_to_p(y, plaintext_bits)).collect()
    }
}

/// Sample a fresh `(secret, error)` pair and compute the public-half
/// query vector `b = A·s + e` for one query slot.
///
/// # Purpose
///
/// Centralises the LWE matvec used by both the inline `client_query`
/// fallback path and the explicit `client_precompute_queries` path so
/// the two paths cannot diverge.
///
/// # Complexity
///
/// `O(n_rows · lwe_dim)` wrapping multiply-add — this is the per-query
/// LWE cost amortised by `precompute_queries`.
fn sample_slot(params: &FrodoServerParams, material: &FrodoHintMaterial) -> PreparedSlot {
    let lwe_dim = params.params.lwe_dim as usize;
    let n_rows  = params.n_rows  as usize;

    let mut rng = rand::rng();
    let mut secret = vec![0u32; lwe_dim];
    sample_ternary_into(&mut rng, &mut secret);
    let mut e = vec![0u32; n_rows];
    sample_ternary_into(&mut rng, &mut e);

    // b = A·s + e
    let mut b = e;
    let a = &material.a;
    for (i, bi) in b.iter_mut().enumerate() {
        let row_off = i * lwe_dim;
        let mut acc = *bi;
        for k in 0..lwe_dim {
            acc = acc.wrapping_add(a[row_off + k].wrapping_mul(secret[k]));
        }
        *bi = acc;
    }
    PreparedSlot { secret, b, c: None }
}

/// Compute the decode-side material `c = sᵀ·H` for one slot.
///
/// # Purpose
///
/// Used by `client_precompute_decodes` to materialise Phase C; the
/// matching `client_decode` then takes the cheap `residual − c` path.
///
/// # Rationale
///
/// Inner loop is unconditional (no `sk == 0` short-circuit) so timing
/// does not leak the secret's Hamming weight: `sk.wrapping_mul(0) = 0`
/// is a no-op in the wrapping-add accumulator.
///
/// # Complexity
///
/// `O(lwe_dim · row_width)` wrapping multiply-add — the most expensive
/// per-slot operation; this is what `precompute_decodes` amortises
/// across a batch.
fn compute_c(secret: &[u32], hint: &[u32], lwe_dim: usize, row_width: usize) -> Vec<u32> {
    debug_assert_eq!(secret.len(), lwe_dim);
    debug_assert_eq!(hint.len(), lwe_dim * row_width);
    let mut c = vec![0u32; row_width];
    for (k, &sk) in secret.iter().enumerate() {
        let row_off = k * row_width;
        for j in 0..row_width {
            c[j] = c[j].wrapping_add(sk.wrapping_mul(hint[row_off + j]));
        }
    }
    c
}

impl PrecomputingPirBackend for FrodoPirBackend {
    fn client_precompute_queries(state: &mut FrodoClientState, count: u32) {
        state.prepared.reserve(count as usize);
        for _ in 0..count {
            state.prepared.push_back(sample_slot(&state.params, &state.hint_material));
        }
    }

    fn client_precompute_decodes(state: &mut FrodoClientState) {
        let lwe_dim   = state.params.params.lwe_dim as usize;
        let row_width = state.params.row_width as usize;
        let h         = &state.hint.data;

        for slot in state.prepared.iter_mut() {
            if slot.c.is_none() {
                slot.c = Some(compute_c(&slot.secret, h, lwe_dim, row_width));
            }
        }
        if let Some(slot) = state.in_flight.as_mut() {
            if slot.c.is_none() {
                slot.c = Some(compute_c(&slot.secret, h, lwe_dim, row_width));
            }
        }
    }

    fn prepared_slot_count(state: &FrodoClientState) -> usize {
        state.prepared_len()
    }

    fn in_flight_slot_count(state: &FrodoClientState) -> usize {
        state.in_flight_len()
    }
}

/// Compute the FrodoPIR hint `H = Aᵀ · D mod 2³²`.
///
/// # Purpose
///
/// Setup-time matvec that produces the per-segment server hint. Output
/// is row-major shape `lwe_dim × row_width`:
/// `H[k, j] = Σ_i A[i, k] · D[i, j] mod q`.
///
/// # Rationale
///
/// Loop nest is `i, k, j` to keep `A` and `D` row accesses sequential
/// per `i` — the inner-loop pattern LLVM auto-vectorises. The
/// `aik == 0` early-exit is a sparsity shortcut on the random ternary
/// `A` (about 1/3 of cells); harmless to timing since `A` is public.
///
/// # Complexity
///
/// `O(n_rows · lwe_dim · row_width)` wrapping multiply-add — dominates
/// the cost of `server_setup` and `full_rebuild`.
fn compute_hint(a: &[u32], db: &[u32], n_rows: u32, lwe_dim: u32, row_width: u32) -> Vec<u32> {
    let lwe_dim_us   = lwe_dim as usize;
    let row_width_us = row_width as usize;
    let mut h = vec![0u32; lwe_dim_us * row_width_us];
    for i in 0..n_rows as usize {
        let a_row = &a[i * lwe_dim_us..(i + 1) * lwe_dim_us];
        let d_row = &db[i * row_width_us..(i + 1) * row_width_us];
        for k in 0..lwe_dim_us {
            let aik = a_row[k];
            if aik == 0 { continue; }
            let h_row = &mut h[k * row_width_us..(k + 1) * row_width_us];
            for j in 0..row_width_us {
                h_row[j] = h_row[j].wrapping_add(aik.wrapping_mul(d_row[j]));
            }
        }
    }
    h
}

/// Wire-size accounting for FrodoPIR.
///
/// Reports the minimum fixed-width little-endian encoding of each wire
/// type. `FrodoQuery` and `FrodoResponse` are dense `u32` vectors;
/// `FrodoHint` is the `lwe_dim × row_width` matrix; `FrodoServerParams`
/// carries `(lwe_dim, plaintext_bits, n_rows, row_width)` and a 16-byte
/// seed — the public matrix `A` is **not** on the wire (it lives in
/// [`FrodoHintMaterial`] and the client re-expands it from the seed).
impl BackendWireSize for FrodoPirBackend {
    fn query_byte_size(q: &FrodoQuery) -> usize {
        q.b.len() * 4
    }
    fn response_byte_size(r: &FrodoResponse) -> usize {
        r.a.len() * 4
    }
    fn hint_byte_size(h: &FrodoHint) -> usize {
        h.data.len() * 4
    }
    fn server_params_byte_size(_p: &FrodoServerParams) -> usize {
        // FrodoParams = { lwe_dim: u32, plaintext_bits: u32, seed: [u8; 16] }
        // FrodoServerParams = { params, n_rows: u32, row_width: u32 }.
        // The public matrix A lives in FrodoHintMaterial and never travels on the wire.
        let frodo_params = 4 + 4 + 16;          // lwe_dim + plaintext_bits + seed
        let dims         = 4 + 4;               // n_rows + row_width
        frodo_params + dims
    }
}

impl IncrementalPirBackend for FrodoPirBackend {
    fn server_patch_hint(
        params: &FrodoServerParams,
        material: &FrodoHintMaterial,
        hint: &mut FrodoHint,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
    ) {
        apply_patch(
            &material.a,
            params.params.lwe_dim,
            params.n_rows,
            params.row_width,
            &mut hint.data,
            row_deltas,
        );
    }

    fn client_patch_state(
        state: &mut FrodoClientState,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
    ) {
        // Pull params + hint_material snapshots out of the state —
        // `client_setup` already stashed both, so no separate arguments
        // are threaded through.
        let FrodoClientState { params, hint_material, hint, prepared, in_flight } = state;
        apply_patch(
            &hint_material.a,
            params.params.lwe_dim,
            params.n_rows,
            params.row_width,
            &mut hint.data,
            row_deltas,
        );
        // Slots that already carry `c = sᵀ·H` need their `c` patched in
        // lock-step. Slots with `c == None` are skipped — they will lazily
        // pick up the post-patch hint on first use.
        patch_slot_c(&hint_material.a, params.params.lwe_dim, params.row_width,
                     prepared.iter_mut().chain(in_flight.as_mut()),
                     row_deltas);
    }
}

/// Apply the same hint-row deltas to every slot's precomputed `c`
/// vector.
///
/// # Purpose
///
/// Keeps the Phase-C precomputation consistent with the patched hint —
/// without it, `client_decode` on a precomputed slot would return
/// garbage after any mutation.
///
/// # Rationale
///
/// From `H_new[k, off] = H_old[k, off] + A[row_idx, k] · Δ`,
/// `c_new[off] = c_old[off] + (Σ_k secret[k] · A[row_idx, k]) · Δ
///             = c_old[off] + dot · Δ`.
///
/// `dot = secret · A_row` is **independent of cell offset**, so we
/// compute it once per `(slot, row_delta)` pair, then apply it to every
/// `(off, Δ)` cell edit on that row. This is the optimisation that
/// makes patching asymptotically cheaper than recomputing `c`.
///
/// # Complexity
///
/// `O(slots · row_deltas · (lwe_dim + n_cells))` — versus
/// `O(slots · row_deltas · lwe_dim · n_cells)` if `dot` were
/// recomputed per cell.
fn patch_slot_c<'a, I>(
    a: &[u32],
    lwe_dim: u32,
    row_width: u32,
    slots: I,
    row_deltas: &[(u32, Vec<(u16, i64)>)],
)
where
    I: IntoIterator<Item = &'a mut PreparedSlot>,
{
    let lwe_dim_us   = lwe_dim as usize;
    let row_width_us = row_width as usize;

    for slot in slots {
        let Some(c) = slot.c.as_mut() else { continue; };
        debug_assert_eq!(c.len(), row_width_us);
        debug_assert_eq!(slot.secret.len(), lwe_dim_us);

        for (row_idx, cells) in row_deltas {
            let a_row = &a[(*row_idx as usize) * lwe_dim_us..(*row_idx as usize + 1) * lwe_dim_us];
            // dot = secret · A_row (offset-independent).
            let dot: u32 = slot.secret.iter().zip(a_row.iter())
                .fold(0u32, |acc, (&s, &ai)| acc.wrapping_add(s.wrapping_mul(ai)));
            for (off, delta) in cells {
                if *delta == 0 { continue; }
                let delta_u32 = *delta as u32;
                c[*off as usize] = c[*off as usize].wrapping_add(dot.wrapping_mul(delta_u32));
            }
        }
    }
}

/// Apply sparse cell deltas to a hint laid out as
/// `lwe_dim × row_width` row-major.
///
/// # Purpose
///
/// Core of incremental hint patching for both server
/// (`server_patch_hint`) and client (`client_patch_state`); both
/// delegate here so the two paths cannot diverge.
///
/// # Rationale
///
/// Math: `H[k, cell] += A[row_idx, k] · Δ mod 2³²`. Iterates touched
/// `(row, cell, Δ)` triples and slides `A`'s `row_idx`-th column
/// against the hint column at `cell`. The `aik == 0` early-exit
/// captures sparsity in the random ternary `A`.
///
/// # Complexity
///
/// `O(Σ row_deltas · lwe_dim)` wrapping multiply-add — sparse in the
/// mutation footprint, dense in `lwe_dim`. Vastly cheaper than a full
/// `compute_hint` when the mutation count is small.
fn apply_patch(
    a: &[u32],
    lwe_dim: u32,
    n_rows: u32,
    row_width: u32,
    hint: &mut [u32],
    row_deltas: &[(u32, Vec<(u16, i64)>)],
) {
    let lwe_dim_us   = lwe_dim as usize;
    let row_width_us = row_width as usize;
    debug_assert_eq!(a.len(), (n_rows as usize) * lwe_dim_us);
    debug_assert_eq!(hint.len(), lwe_dim_us * row_width_us);

    for (row_idx, cells) in row_deltas {
        debug_assert!(
            (*row_idx as usize) < n_rows as usize,
            "row_idx {row_idx} out of range (n_rows={n_rows})",
        );
        let a_row = &a[(*row_idx as usize) * lwe_dim_us..(*row_idx as usize + 1) * lwe_dim_us];

        for (cell_offset, delta) in cells {
            debug_assert!(
                (*cell_offset as usize) < row_width_us,
                "cell_offset {cell_offset} out of range (row_width={row_width})",
            );
            if *delta == 0 { continue; }
            let delta_u32 = *delta as u32;
            let cell_us = *cell_offset as usize;

            for (k, &aik) in a_row.iter().enumerate() {
                if aik == 0 { continue; }
                let idx = k * row_width_us + cell_us;
                hint[idx] = hint[idx].wrapping_add(aik.wrapping_mul(delta_u32));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db(n_rows: u32, row_width: u32, pb: u32) -> Vec<u32> {
        let mask = (1u32 << pb) - 1;
        (0..n_rows * row_width)
            .map(|i| (i.wrapping_mul(2_654_435_761)) & mask)
            .collect()
    }

    fn roundtrip_all_rows(n_rows: u32, row_width: u32) {
        let pb = 8u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);
        for row in 0..n_rows {
            let q = FrodoPirBackend::client_query(&mut state, row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            let expected: Vec<u32> = db[row as usize * row_width as usize
                ..(row as usize + 1) * row_width as usize]
                .to_vec();
            assert_eq!(decoded, expected, "row {row} mismatch");
        }
    }

    #[test]
    fn roundtrip_square_4x4() {
        roundtrip_all_rows(4, 4);
    }

    #[test]
    fn roundtrip_wide_4x16() {
        roundtrip_all_rows(4, 16);
    }

    #[test]
    fn roundtrip_tall_32x4() {
        roundtrip_all_rows(32, 4);
    }

    #[test]
    fn roundtrip_segment_shape_64x16() {
        let pb = 8u32;
        let n_rows = 64u32;
        let row_width = 16u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);
        // Query 100 random rows (with wrapping) to stress the matvecs.
        for i in 0u32..100 {
            let row = (i.wrapping_mul(2_654_435_761)) % n_rows;
            let q = FrodoPirBackend::client_query(&mut state, row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            let expected: Vec<u32> = db[row as usize * row_width as usize
                ..(row as usize + 1) * row_width as usize]
                .to_vec();
            assert_eq!(decoded, expected, "iteration {i}, row {row} mismatch");
        }
    }

    #[test]
    fn decode_yields_high_bits_zero() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);
        for row in 0..n_rows {
            let q = FrodoPirBackend::client_query(&mut state, row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            for &cell in &decoded {
                assert!(cell < (1 << pb), "cell {cell} exceeds p=2^{pb}");
            }
        }
    }

    #[test]
    fn independent_queries_dont_interfere() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        let q3 = FrodoPirBackend::client_query(&mut state, 3);
        let r3 = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q3);
        let d3 = FrodoPirBackend::client_decode(&state, &r3);
        let expected3: Vec<u32> = db[3 * row_width as usize..4 * row_width as usize].to_vec();
        assert_eq!(d3, expected3, "first query row 3 mismatch");

        let q7 = FrodoPirBackend::client_query(&mut state, 7);
        let r7 = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q7);
        let d7 = FrodoPirBackend::client_decode(&state, &r7);
        let expected7: Vec<u32> = db[7 * row_width as usize..8 * row_width as usize].to_vec();
        assert_eq!(d7, expected7, "second query row 7 mismatch");
    }

    #[test]
    fn setup_is_random_per_call() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp1, _, _) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let (sp2, _, _) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        assert_ne!(sp1.params.seed, sp2.params.seed, "seeds must differ across calls");
    }

    #[test]
    fn hint_matches_explicit_atd() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        let lwe_dim = sp.params.lwe_dim;
        let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
        let expected = compute_hint(&a, &db, n_rows, lwe_dim, row_width);

        assert_eq!(hint.data, expected, "hint != Aᵀ·D");
    }

    // Returns the actual signed change applied to the cell (may differ from `delta` when
    // the cell value wraps mod 2^plaintext_bits).
    fn apply_cell_delta(
        db: &mut [u32],
        row_idx: u32,
        cell_offset: u16,
        delta: i64,
        row_width: u32,
        plaintext_bits: u32,
    ) -> i64 {
        let idx = row_idx as usize * row_width as usize + cell_offset as usize;
        let mask = (1i64 << plaintext_bits) - 1;
        let old = db[idx] as i64;
        let new = (old + delta) & mask;
        db[idx] = new as u32;
        new - old
    }

    #[test]
    fn patch_single_cell_matches_oracle() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, mut hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        let actual_dlt = apply_cell_delta(&mut db, 1, 2, 1, row_width, pb);
        let row_deltas = vec![(1u32, vec![(2u16, actual_dlt)])];
        FrodoPirBackend::server_patch_hint(&sp, &_mat, &mut hint, &row_deltas);

        let lwe_dim = sp.params.lwe_dim;
        let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
        let expected = compute_hint(&a, &db, n_rows, lwe_dim, row_width);
        assert_eq!(hint.data, expected);
    }

    #[test]
    fn patch_multi_cell_single_row_matches_oracle() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, mut hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        let raw_deltas: &[(u16, i64)] = &[(0, 3), (1, -1), (3, 2)];
        let cells: Vec<(u16, i64)> = raw_deltas
            .iter()
            .map(|&(off, dlt)| (off, apply_cell_delta(&mut db, 2, off, dlt, row_width, pb)))
            .collect();
        let row_deltas = vec![(2u32, cells)];
        FrodoPirBackend::server_patch_hint(&sp, &_mat, &mut hint, &row_deltas);

        let lwe_dim = sp.params.lwe_dim;
        let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
        let expected = compute_hint(&a, &db, n_rows, lwe_dim, row_width);
        assert_eq!(hint.data, expected);
    }

    #[test]
    fn patch_multi_row_matches_oracle() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, mut hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        let raw: &[(u32, &[(u16, i64)])] = &[
            (0, &[(0, 1)]),
            (3, &[(1, -2), (3, -4)]),
            (7, &[(0, 3), (2, -1), (3, 1)]),
        ];
        let row_deltas: Vec<(u32, Vec<(u16, i64)>)> = raw
            .iter()
            .map(|&(row, cells)| {
                let actual: Vec<(u16, i64)> = cells
                    .iter()
                    .map(|&(off, dlt)| (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb)))
                    .collect();
                (row, actual)
            })
            .collect();
        FrodoPirBackend::server_patch_hint(&sp, &_mat, &mut hint, &row_deltas);

        let lwe_dim = sp.params.lwe_dim;
        let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
        let expected = compute_hint(&a, &db, n_rows, lwe_dim, row_width);
        assert_eq!(hint.data, expected);
    }

    #[test]
    fn patch_zero_delta_is_noop() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, mut hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let hint_before = hint.data.clone();

        let row_deltas = vec![(2u32, vec![(1u16, 0i64)])];
        FrodoPirBackend::server_patch_hint(&sp, &_mat, &mut hint, &row_deltas);

        assert_eq!(hint.data, hint_before);
    }

    #[test]
    fn patch_negative_delta_correct() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, mut hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        // Row 2, off 0: make_db value is 72 (≥ 5), so delta = −5 gives actual change = −5.
        let actual_dlt = apply_cell_delta(&mut db, 2, 0, -5, row_width, pb);
        let row_deltas = vec![(2u32, vec![(0u16, actual_dlt)])];
        FrodoPirBackend::server_patch_hint(&sp, &_mat, &mut hint, &row_deltas);

        let lwe_dim = sp.params.lwe_dim;
        let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
        let expected = compute_hint(&a, &db, n_rows, lwe_dim, row_width);
        assert_eq!(hint.data, expected);
    }

    #[test]
    fn patch_random_burst_matches_oracle() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let pb = 8u32;
        let n_rows = 32u32;
        let row_width = 8u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, mut hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mask = (1u32 << pb) - 1;

        let mut rng = ChaCha20Rng::seed_from_u64(0xCAFE_F00D);
        for _iter in 0..50 {
            let n_rows_in_batch = rng.random_range(1..=4u32);
            let mut row_deltas: Vec<(u32, Vec<(u16, i64)>)> = Vec::new();
            for _ in 0..n_rows_in_batch {
                let row = rng.random_range(0..n_rows);
                let n_cells = rng.random_range(1..=row_width);
                let mut cells: Vec<(u16, i64)> = Vec::new();
                for _ in 0..n_cells {
                    let off = rng.random_range(0..row_width) as u16;
                    let new_val: u32 = rng.random_range(0..=mask);
                    let idx = row as usize * row_width as usize + off as usize;
                    let actual_dlt = new_val as i64 - db[idx] as i64;
                    db[idx] = new_val;
                    cells.push((off, actual_dlt));
                }
                row_deltas.push((row, cells));
            }
            FrodoPirBackend::server_patch_hint(&sp, &_mat, &mut hint, &row_deltas);

            let lwe_dim = sp.params.lwe_dim;
            let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
            let expected = compute_hint(&a, &db, n_rows, lwe_dim, row_width);
            assert_eq!(hint.data, expected, "patched hint diverged at iter {_iter}");
        }
    }

    #[test]
    fn decode_after_patch_returns_patched_value() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        let target_row = 3u32;
        let dlt1 = apply_cell_delta(&mut db, target_row, 1, 7, row_width, pb);
        let dlt2 = apply_cell_delta(&mut db, target_row, 2, -3, row_width, pb);
        let row_deltas = vec![(target_row, vec![(1u16, dlt1), (2u16, dlt2)])];

        let mut server_hint = hint;
        FrodoPirBackend::server_patch_hint(&sp, &_mat, &mut server_hint, &row_deltas);
        FrodoPirBackend::client_patch_state(&mut state, &row_deltas);

        assert_eq!(state.hint.data, server_hint.data,
            "server and client patched hints must be identical");

        let q = FrodoPirBackend::client_query(&mut state, target_row);
        let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
        let decoded = FrodoPirBackend::client_decode(&state, &r);
        let expected = db[target_row as usize * row_width as usize
            ..(target_row as usize + 1) * row_width as usize].to_vec();
        assert_eq!(decoded, expected, "decode after patch did not recover patched row");
    }

    #[test]
    fn decode_after_multi_row_patch_returns_patched_value() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 8u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        let raw: &[(u32, &[(u16, i64)])] = &[
            (2,  &[(0, 5), (3, -2)]),
            (7,  &[(1, 3)]),
            (11, &[(2, -4), (6, 1)]),
        ];
        let row_deltas: Vec<(u32, Vec<(u16, i64)>)> = raw
            .iter()
            .map(|&(row, cells)| {
                let actual: Vec<(u16, i64)> = cells
                    .iter()
                    .map(|&(off, dlt)| (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb)))
                    .collect();
                (row, actual)
            })
            .collect();

        let mut server_hint = hint;
        FrodoPirBackend::server_patch_hint(&sp, &_mat, &mut server_hint, &row_deltas);
        FrodoPirBackend::client_patch_state(&mut state, &row_deltas);

        assert_eq!(state.hint.data, server_hint.data,
            "server and client patched hints must be identical");

        for &(target_row, _) in raw {
            let q = FrodoPirBackend::client_query(&mut state, target_row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            let expected = db[target_row as usize * row_width as usize
                ..(target_row as usize + 1) * row_width as usize].to_vec();
            assert_eq!(decoded, expected, "decode failed for row {target_row}");
        }
    }

    // -------- preprocessing (Phase B + Phase C) --------

    /// Sanity: with both phases warm, every query/decode round-trip still
    /// returns the right plaintext for every row.
    #[test]
    fn precomputed_roundtrip_all_rows() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 8u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        FrodoPirBackend::client_precompute_queries(&mut state, n_rows);
        FrodoPirBackend::client_precompute_decodes(&mut state);
        assert_eq!(state.prepared_len(), n_rows as usize);
        assert_eq!(state.in_flight_len(), 0);

        for row in 0..n_rows {
            let q = FrodoPirBackend::client_query(&mut state, row);
            assert_eq!(state.in_flight_len(), 1);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            let expected = db[row as usize * row_width as usize
                ..(row as usize + 1) * row_width as usize].to_vec();
            assert_eq!(decoded, expected, "row {row} mismatch");
        }
        // All prepared slots consumed.
        assert_eq!(state.prepared_len(), 0);
    }

    /// `client_precompute_decodes` is idempotent: a second call leaves the
    /// queue contents unchanged.
    #[test]
    fn precompute_decodes_idempotent() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        FrodoPirBackend::client_precompute_queries(&mut state, 4);
        FrodoPirBackend::client_precompute_decodes(&mut state);
        let snapshot: Vec<Option<Vec<u32>>> = state.prepared.iter().map(|s| s.c.clone()).collect();
        FrodoPirBackend::client_precompute_decodes(&mut state);
        let after: Vec<Option<Vec<u32>>> = state.prepared.iter().map(|s| s.c.clone()).collect();
        assert_eq!(snapshot, after, "second precompute_decodes must be a no-op");
    }

    /// Cheap path (precomputed) and on-the-fly path produce the same
    /// decoded plaintext when fed the same `(secret, b)`.
    #[test]
    fn precomputed_decode_matches_on_the_fly() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        // Warm path: precompute queries + decodes.
        let mut warm = FrodoPirBackend::client_setup(&sp, &hint);
        FrodoPirBackend::client_precompute_queries(&mut warm, 4);
        FrodoPirBackend::client_precompute_decodes(&mut warm);

        // Cold path: clone the prepared slots over to a sibling state with
        // c stripped. Both states then issue queries against the same rows.
        let mut cold = FrodoPirBackend::client_setup(&sp, &hint);
        for slot in warm.prepared.iter() {
            cold.prepared.push_back(PreparedSlot {
                secret: slot.secret.clone(),
                b:      slot.b.clone(),
                c:      None,
            });
        }

        for row in 0..4u32 {
            let q_warm = FrodoPirBackend::client_query(&mut warm, row);
            let q_cold = FrodoPirBackend::client_query(&mut cold, row);
            assert_eq!(q_warm.b, q_cold.b, "queries diverged at row {row}");

            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q_warm);
            let dec_warm = FrodoPirBackend::client_decode(&warm, &r);
            let dec_cold = FrodoPirBackend::client_decode(&cold, &r);
            assert_eq!(dec_warm, dec_cold, "decodes diverged at row {row}");
        }
    }

    /// After patching the hint, the precomputed `c` of every queued slot
    /// matches what `client_setup` on the post-patch hint would compute.
    #[test]
    fn patched_c_matches_recomputed_c() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        FrodoPirBackend::client_precompute_queries(&mut state, 5);
        FrodoPirBackend::client_precompute_decodes(&mut state);

        // Apply some sparse patches.
        let raw: &[(u32, &[(u16, i64)])] = &[
            (1, &[(0, 3), (2, -1)]),
            (5, &[(1, 7)]),
            (7, &[(0, -2), (3, 4)]),
        ];
        let row_deltas: Vec<(u32, Vec<(u16, i64)>)> = raw
            .iter()
            .map(|&(row, cells)| {
                let actual: Vec<(u16, i64)> = cells
                    .iter()
                    .map(|&(off, dlt)| (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb)))
                    .collect();
                (row, actual)
            })
            .collect();

        // Snapshot the secrets so we can recompute c against the patched hint.
        let secrets: Vec<Vec<u32>> = state.prepared.iter().map(|s| s.secret.clone()).collect();
        FrodoPirBackend::client_patch_state(&mut state, &row_deltas);

        // Oracle: compute c = sᵀ · patched_hint directly from H.
        let lwe_dim = sp.params.lwe_dim;
        let a       = sample_a(&sp.params.seed, n_rows, lwe_dim);
        let h_patched = compute_hint(&a, &db, n_rows, lwe_dim, row_width);

        for (slot, secret) in state.prepared.iter().zip(secrets.iter()) {
            let oracle = compute_c(secret, &h_patched, lwe_dim as usize, row_width as usize);
            assert_eq!(slot.c.as_ref().unwrap(), &oracle, "patched c diverged");
        }
    }

    /// Full round-trip: precompute, patch, then decode — the patched-c path
    /// recovers the patched plaintext exactly.
    #[test]
    fn decode_after_patch_with_precomputed_c_returns_patched_value() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 8u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);
        let mut server_hint = hint;

        FrodoPirBackend::client_precompute_queries(&mut state, 4);
        FrodoPirBackend::client_precompute_decodes(&mut state);

        let raw: &[(u32, &[(u16, i64)])] = &[
            (2,  &[(0, 5), (3, -2)]),
            (11, &[(2, -4), (6, 1)]),
        ];
        let row_deltas: Vec<(u32, Vec<(u16, i64)>)> = raw
            .iter()
            .map(|&(row, cells)| {
                let actual: Vec<(u16, i64)> = cells
                    .iter()
                    .map(|&(off, dlt)| (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb)))
                    .collect();
                (row, actual)
            })
            .collect();

        FrodoPirBackend::server_patch_hint(&sp, &_mat, &mut server_hint, &row_deltas);
        FrodoPirBackend::client_patch_state(&mut state, &row_deltas);

        for &(target_row, _) in raw {
            let q = FrodoPirBackend::client_query(&mut state, target_row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            let expected = db[target_row as usize * row_width as usize
                ..(target_row as usize + 1) * row_width as usize].to_vec();
            assert_eq!(decoded, expected, "decode failed for row {target_row}");
        }
    }

    /// `client_query` falls back to inline sampling when the prepared queue
    /// is empty (preserves pre-precomputation behaviour).
    #[test]
    fn client_query_falls_back_when_queue_empty() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);
        assert_eq!(state.prepared_len(), 0);

        let q = FrodoPirBackend::client_query(&mut state, 3);
        assert_eq!(state.in_flight_len(), 1);
        let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
        let decoded = FrodoPirBackend::client_decode(&state, &r);
        let expected = db[3 * row_width as usize..4 * row_width as usize].to_vec();
        assert_eq!(decoded, expected);
    }

    /// `client_precompute_decodes` also fills `c` for an already-in-flight
    /// slot whose `c` was None — that slot's decode then takes the cheap path.
    #[test]
    fn precompute_decodes_fills_in_flight_slot() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        let q = FrodoPirBackend::client_query(&mut state, 1);
        assert!(state.in_flight.as_ref().unwrap().c.is_none(),
            "fresh inline-sampled slot starts with no c");

        FrodoPirBackend::client_precompute_decodes(&mut state);
        assert!(state.in_flight.as_ref().unwrap().c.is_some(),
            "precompute_decodes must fill in-flight slot too");

        let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
        let decoded = FrodoPirBackend::client_decode(&state, &r);
        let expected = db[row_width as usize..2 * row_width as usize].to_vec();
        assert_eq!(decoded, expected);
    }
}
