//! `SimplePirBackend` — IndexPirBackend impl over `Z_{2^32}` with a
//! square-ish reshape and discrete-Gaussian errors.
//!
//! # Purpose
//!
//! The second shipped IKPIR backend, alongside [`crate::FrodoPirBackend`].
//! Implements [`IndexPirBackend`](crate::IndexPirBackend),
//! [`IncrementalPirBackend`](crate::IncrementalPirBackend),
//! [`PrecomputingPirBackend`](crate::PrecomputingPirBackend), and
//! [`BackendWireSize`](crate::BackendWireSize) for SimplePIR.
//!
//! # Design / architecture
//!
//! - **Witness type.** [`SimplePirBackend`] is zero-sized; all behaviour
//!   lives on the trait impls.
//! - **Per-segment state.** Each segment has its own
//!   [`SimpleServerParams`] / [`SimpleHint`] (server side) and
//!   [`SimpleClientState`] (client side). The IKPIR server constructs
//!   `arity` instances of each.
//! - **Internal reshape.** The per-segment cell array of shape
//!   `(n_rows × row_width)` is logically viewed as a near-square
//!   `(R × C)` matrix where `R = ⌈n_rows / k⌉`, `C = k · row_width`,
//!   and `k = max(1, round(√(n_rows / row_width)))`. The flat buffer is
//!   reinterpreted in place — there is no reshape copy. The trait-level
//!   `row` argument (a bucket index within the segment) translates to
//!   `reshape_row = row / k` plus an intra-row `bucket_within = row % k`
//!   that the client uses to slice the right `row_width`-cell window out
//!   of the decoded response. `bucket_within` is **never** transmitted
//!   to the server.
//! - **Precomputation queue.** Same shape as FrodoPIR: `VecDeque` of
//!   prepared slots and at most one in-flight slot.
//!   `bucket_within` lives on the in-flight slot only — it's per-query,
//!   not per-prepared-slot.
//! - **Hot loops.** `server_answer`'s matvec and `client_decode`'s
//!   `residual − c` (or `residual − sᵀ·H` on the cold path) dominate
//!   CPU time. Loop bodies mirror FrodoPIR with reshape-coordinate
//!   translations woven through the index math.
//!
//! # Related files
//!
//! - `mod.rs`     — re-exports the public types here; carries the math
//!   summary for the whole module.
//! - `params.rs`  — `SimpleConfig` / `SimpleParams`.
//! - `arith.rs`   — `round_p_to_q` / `round_q_to_p` (Δ-scaling).
//! - `sampler.rs` — `sample_a` / `sample_uniform_zq_into`
//!   / `sample_discrete_gaussian_into`.

use std::collections::VecDeque;

use rand::RngCore;

use super::{
    round_p_to_q, round_q_to_p, sample_a, sample_a_parallel, sample_discrete_gaussian_into,
    sample_uniform_zq_into, SimpleConfig, SimpleParams,
};
use crate::backend::matvec::matvec_accumulate;
use crate::backend::{
    parallel, patch::TouchedRuns, BackendWireSize, HintPatchMode, IncrementalPirBackend,
    IndexPirBackend, ParallelSetupBackend, PrecomputingPirBackend, ResponseRewind,
};
use std::collections::BTreeMap;

/// Zero-sized witness type that carries the [`IndexPirBackend`] /
/// [`IncrementalPirBackend`] / [`PrecomputingPirBackend`] /
/// [`BackendWireSize`] impls for SimplePIR.
///
/// # Purpose
///
/// The IKPIR server / client are generic over the backend; this type
/// names the SimplePIR specialisation.
///
/// # Rationale
///
/// Zero-sized so the value is free to construct and pass by value; all
/// methods are static — see the trait impls below.
#[derive(Clone, Copy, Debug, Default)]
pub struct SimplePirBackend;

/// Per-segment public parameters: LWE dimensions, plaintext modulus,
/// reshape dimensions, and the 16-byte seed used to expand the public
/// matrix `A`.
///
/// # Purpose
///
/// One instance per segment; combined with [`SimpleHint`] it forms the
/// `(ServerParams, Hint)` pair every IKPIR client receives at setup. The
/// matrix `A` itself is **not** carried here — see [`SimpleHintMaterial`].
///
/// # Rationale
///
/// `n_rows` / `row_width` are the **original** segment shape (cuckoo
/// buckets × bucket cells); `k`, `reshape_rows`, `reshape_row_width`
/// are the SimplePIR-internal reshape parameters. Storing both spares
/// every call site from recomputing the reshape arithmetic and makes
/// the wire bundle self-describing.
#[derive(Clone, Debug)]
pub struct SimpleServerParams {
    /// LWE dimension, plaintext bits, σ, and 16-byte seed used to sample `a`.
    pub params: SimpleParams,
    /// Number of database rows in the **original** segment layout
    /// (i.e. cuckoo buckets per segment).
    pub n_rows: u32,
    /// Number of `u32` cells per database row in the **original** layout
    /// (i.e. cells per cuckoo bucket).
    pub row_width: u32,
    /// Buckets per reshape row. `k ≥ 1`; equals
    /// `max(1, round(√(n_rows / row_width)))`.
    pub k: u32,
    /// Number of rows in the reshaped matrix `D`.
    /// `reshape_rows = ⌈n_rows / k⌉`.
    pub reshape_rows: u32,
    /// Cells per row in the reshaped matrix `D`.
    /// `reshape_row_width = k · row_width`.
    pub reshape_row_width: u32,
}

/// Server-local working state: the LWE public matrix `A` in row-major
/// shape `reshape_rows × lwe_dim`, expanded deterministically from
/// [`SimpleServerParams::params`]`.seed` via `sample_a` (private to this
/// module).
///
/// # Purpose
///
/// Used by [`SimplePirBackend::server_setup`] to compute the hint and by
/// [`SimplePirBackend::server_patch_hint`] to keep the hint coherent
/// across mutations. **Not part of the wire payload** — the client
/// re-expands its own copy from the seed during
/// [`SimplePirBackend::client_setup`], and the server may drop and
/// re-expand its copy via
/// [`IkpirServer::drop_hint_material`](../../../ikpir_server/struct.IkpirServer.html#method.drop_hint_material).
///
/// # Rationale
///
/// Pulling `A` out of [`SimpleServerParams`] keeps the wire bundle small
/// (only the 16-byte seed travels) and lets the server free `A` on
/// pure-read workloads. Not `Clone`: every "extra" `A` buffer must be
/// an explicit [`SimplePirBackend::expand_hint_material`] call so
/// accidental duplication is impossible.
#[derive(Debug, Default)]
pub struct SimpleHintMaterial {
    /// Public matrix `A` in row-major shape `reshape_rows × lwe_dim`.
    pub a: Vec<u32>,
}

/// SimplePIR hint matrix `H = Aᵀ · D mod 2³²` in row-major shape
/// `lwe_dim × reshape_row_width`.
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
/// [`SimplePirBackend::server_patch_hint`] /
/// [`SimplePirBackend::client_patch_state`] — never recomputed unless
/// the server triggers `full_rebuild`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimpleHint {
    /// Flat row-major buffer of length `lwe_dim × reshape_row_width`.
    pub data: Vec<u32>,
}

/// One precomputed query slot: the LWE secret, the matching public-half
/// query vector `b = A·s + e`, and (optionally) the decode-side material
/// `c = sᵀ·H`. `c` is `None` until [`SimplePirBackend::client_precompute_decodes`]
/// runs. Internal to this module — consumers go through the public
/// `prepared` / `in_flight` accessor methods on `SimpleClientState`.
struct PreparedSlot {
    secret: Vec<u32>,
    b: Vec<u32>,
    c: Option<Vec<u32>>,
}

/// An in-flight slot: the [`PreparedSlot`] consumed by the most recent
/// `client_query`, plus the intra-reshape-row `bucket_within` that the
/// matching `client_decode` uses to slice the right `row_width`-cell
/// window out of the response.
///
/// # Rationale
///
/// `bucket_within = row % k` is per-query (it depends on the target
/// `row`), so it cannot live on a prepared slot — only on the in-flight
/// slot that a `client_query` produces.
struct InFlightSlot {
    slot: PreparedSlot,
    bucket_within: u32,
}

/// Per-segment client-held SimplePIR state.
///
/// # Purpose
///
/// Holds everything the client needs to build queries and decode
/// responses for one segment: a copy of [`SimpleServerParams`], the
/// patched [`SimpleHint`], a FIFO queue of slots prepared by
/// [`SimplePirBackend::client_precompute_queries`], and the slot
/// consumed by the most recent
/// [`SimplePirBackend::client_query`].
///
/// # Constraints
///
/// **Single in-flight query.** Each `client_query` overwrites the
/// in-flight slot; issuing a second `client_query` before decoding the
/// first discards the first's secret and that first decode will return
/// garbage. This matches the protocol cadence in `IkpirClient` (one
/// query → one answer → one decode per round per segment).
pub struct SimpleClientState {
    /// Public parameters for this segment.
    pub params: SimpleServerParams,
    /// Locally expanded copy of the LWE public matrix `A`, re-derived
    /// from `params.params.seed` during `client_setup`.
    pub hint_material: SimpleHintMaterial,
    /// Locally maintained copy of the segment hint matrix.
    pub hint: SimpleHint,
    /// Prepared but unconsumed slots, FIFO. Front is the next slot a
    /// `client_query` will use; back is where `client_precompute_queries`
    /// appends.
    prepared: VecDeque<PreparedSlot>,
    /// The slot consumed by the most recent `client_query`, if any. Read
    /// by the matching `client_decode`.
    in_flight: Option<InFlightSlot>,
}

impl SimpleClientState {
    /// Number of prepared-but-unconsumed slots (filled by
    /// `client_precompute_queries`, drained by `client_query`).
    pub fn prepared_len(&self) -> usize {
        self.prepared.len()
    }

    /// Whether there is an in-flight query awaiting decode (always 0 or 1).
    pub const fn in_flight_len(&self) -> usize {
        self.in_flight.is_some() as usize
    }
}

/// Wire-level SimplePIR query: `b = A·s + e + Δ·u_{reshape_row}` of
/// length `reshape_rows`.
///
/// # Purpose
///
/// The client-side ciphertext shipped to the server. One per segment.
/// Crucially, does **not** carry `bucket_within` — that stays
/// client-side so the server learns at most the reshape-row index
/// (hidden inside `b` under LWE).
#[derive(Clone, Debug)]
pub struct SimpleQuery {
    /// Encrypted query vector
    /// (`b = A·s + e + Δ·u_{reshape_row}`), length `reshape_rows`.
    pub b: Vec<u32>,
}

/// Wire-level SimplePIR response: `a = bᵀ·D` of length
/// `reshape_row_width`.
///
/// # Purpose
///
/// The server-side ciphertext shipped back to the client. One per
/// segment. The client slices the `bucket_within·row_width
/// .. (bucket_within+1)·row_width` window out of the decoded response.
#[derive(Clone, Debug)]
pub struct SimpleResponse {
    /// Encrypted response vector (`a = bᵀ·D`), length `reshape_row_width`.
    pub a: Vec<u32>,
}

impl IndexPirBackend for SimplePirBackend {
    type Config = SimpleConfig;
    type ServerParams = SimpleServerParams;
    type HintMaterial = SimpleHintMaterial;
    type Hint = SimpleHint;
    type ClientState = SimpleClientState;
    type Query = SimpleQuery;
    type Response = SimpleResponse;

    fn server_setup(
        config: &SimpleConfig,
        db: &[u32],
        n_rows: u32,
        row_width: u32,
        plaintext_bits: u32,
    ) -> (SimpleServerParams, SimpleHintMaterial, SimpleHint) {
        debug_assert_eq!(db.len(), (n_rows as usize) * (row_width as usize));
        let lwe_dim = config.lwe_dim;
        let sigma = config.sigma;

        let (k, reshape_rows, reshape_row_width) = reshape_dims(n_rows, row_width);

        let mut seed = [0u8; 16];
        rand::rng().fill_bytes(&mut seed);
        let params = SimpleParams::new(lwe_dim, plaintext_bits, sigma, seed);

        let a = sample_a(&seed, reshape_rows, lwe_dim);
        let hint_data = compute_hint(&a, db, n_rows, row_width, k, lwe_dim, reshape_row_width);

        (
            SimpleServerParams {
                params,
                n_rows,
                row_width,
                k,
                reshape_rows,
                reshape_row_width,
            },
            SimpleHintMaterial { a },
            SimpleHint { data: hint_data },
        )
    }

    fn db_matrix_shape(params: &SimpleServerParams) -> (u32, u32) {
        // Post-reshape dims as `reshape_dims` fixed them at setup time.
        (params.reshape_rows, params.reshape_row_width)
    }

    fn expand_hint_material(params: &SimpleServerParams) -> SimpleHintMaterial {
        let a = sample_a(
            &params.params.seed,
            params.reshape_rows,
            params.params.lwe_dim,
        );
        SimpleHintMaterial { a }
    }

    fn client_setup(params: &SimpleServerParams, hint: &SimpleHint) -> SimpleClientState {
        SimpleClientState {
            params: params.clone(),
            hint_material: Self::expand_hint_material(params),
            hint: hint.clone(),
            prepared: VecDeque::new(),
            in_flight: None,
        }
    }

    fn client_query(state: &mut SimpleClientState, row: u32) -> SimpleQuery {
        let n_rows = state.params.n_rows;
        let k = state.params.k;
        let plaintext_bits = state.params.params.plaintext_bits;
        debug_assert!(row < n_rows, "row {row} out of range (n_rows={n_rows})");

        let reshape_row = row / k;
        let bucket_within = row % k;

        // Cheap path if a prepared slot is available; otherwise sample inline.
        let slot = state
            .prepared
            .pop_front()
            .unwrap_or_else(|| sample_slot(&state.params, &state.hint_material));

        let mut b = slot.b.clone();
        let delta = round_p_to_q(1, plaintext_bits);
        b[reshape_row as usize] = b[reshape_row as usize].wrapping_add(delta);

        state.in_flight = Some(InFlightSlot {
            slot,
            bucket_within,
        });
        SimpleQuery { b }
    }

    fn server_answer(
        params: &SimpleServerParams,
        db: &[u32],
        n_rows: u32,
        row_width: u32,
        query: &SimpleQuery,
    ) -> SimpleResponse {
        debug_assert_eq!(query.b.len(), params.reshape_rows as usize);
        debug_assert_eq!(db.len(), (n_rows as usize) * (row_width as usize));
        debug_assert_eq!(params.n_rows, n_rows);
        debug_assert_eq!(params.row_width, row_width);

        let k = params.k as usize;
        let row_width_us = row_width as usize;
        let reshape_row_width = params.reshape_row_width as usize;
        let mut a = vec![0u32; reshape_row_width];

        // The reshaped matrix is `db` itself reinterpreted with rows of
        // `reshape_row_width = k · row_width` cells (original row `r` lands
        // at reshape row `r / k`, offset `(r % k) · row_width` — contiguous).
        // The first `n_rows / k` reshape rows are dense; run them through the
        // shared blocked kernel.
        let full = n_rows as usize / k;
        matvec_accumulate(&mut a, &db[..full * reshape_row_width], &query.b[..full]);

        // Partial last reshape row (`n_rows mod k` original rows), scalar.
        let tail_rows = db[full * reshape_row_width..].chunks_exact(row_width_us);
        for (t, row) in tail_rows.enumerate() {
            let qi = query.b[full];
            let off_within = t * row_width_us;
            for (x, &cell) in a[off_within..off_within + row_width_us].iter_mut().zip(row) {
                *x = x.wrapping_add(qi.wrapping_mul(cell));
            }
        }
        SimpleResponse { a }
    }

    /// Inner loop is unconditional (no `sk == 0` short-circuit) so timing
    /// does not leak the secret's value. With uniform `Z_q` secret this
    /// is essentially never zero anyway; the unconditional path is kept
    /// for parity with FrodoPIR and to avoid timing variance.
    fn client_decode(state: &SimpleClientState, response: &SimpleResponse) -> Vec<u32> {
        let reshape_row_width = state.params.reshape_row_width as usize;
        let row_width = state.params.row_width as usize;
        let plaintext_bits = state.params.params.plaintext_bits;
        debug_assert_eq!(response.a.len(), reshape_row_width);

        // PROTOCOL INVARIANT (internal): `in_flight` is populated by the
        // matching `client_query`. `IkpirClient::decode` always issues a
        // query before decoding, so a `None` here is an unreachable backend
        // bug, never reachable from user input; there is no `IkpirError`
        // variant for it and this trait method is infallible by contract.
        let inflight = state
            .in_flight
            .as_ref()
            .expect("client_decode invariant: in_flight set by matching client_query");
        let slot = &inflight.slot;
        let bucket_within = inflight.bucket_within as usize;
        debug_assert!(
            (bucket_within + 1) * row_width <= reshape_row_width,
            "bucket_within slice out of range",
        );

        // residual = response - c, where c = sᵀ·H (precomputed if present,
        // otherwise materialised on the fly). The two paths are arithmetically
        // identical; the cheap path just reuses already-multiplied values.
        let mut residual = response.a.clone();
        match slot.c.as_ref() {
            Some(c) => {
                debug_assert_eq!(c.len(), reshape_row_width);
                for j in 0..reshape_row_width {
                    residual[j] = residual[j].wrapping_sub(c[j]);
                }
            }
            None => {
                // residual −= sᵀ·H, computed as residual += (−s)ᵀ·H — exact
                // mod 2³², and the one-time negation (data-independent) lets
                // the shared blocked kernel carry the heavy pass.
                let neg_secret: Vec<u32> = slot.secret.iter().map(|s| s.wrapping_neg()).collect();
                matvec_accumulate(&mut residual, &state.hint.data, &neg_secret);
            }
        }

        let slice_start = bucket_within * row_width;
        residual[slice_start..slice_start + row_width]
            .iter()
            .map(|&y| round_q_to_p(y, plaintext_bits))
            .collect()
    }
}

/// Compute the reshape parameters `(k, reshape_rows, reshape_row_width)`
/// from the original `(n_rows, row_width)`.
///
/// # Rationale
///
/// `k = max(1, round(√(n_rows / row_width)))` minimises the perimeter of
/// the reshaped matrix subject to `R · k ≥ n_rows`. The `max(1, ...)`
/// guard handles the degenerate case `n_rows < row_width / 4` where the
/// rounded value would be `0`.
///
/// # Complexity
///
/// `O(1)` — one division, one sqrt, one round.
#[inline]
pub(crate) fn reshape_dims(n_rows: u32, row_width: u32) -> (u32, u32, u32) {
    debug_assert!(n_rows > 0 && row_width > 0);
    let ratio = (n_rows as f64) / (row_width as f64);
    let k = ratio.sqrt().round().max(1.0) as u32;
    let reshape_rows = n_rows.div_ceil(k);
    let reshape_row_width = k
        .checked_mul(row_width)
        .expect("reshape_row_width = k * row_width overflowed u32");
    (k, reshape_rows, reshape_row_width)
}

/// Translate an original `(orig_row, orig_off)` cell coordinate into
/// the reshape coordinate `(reshape_row, reshape_off)`.
///
/// # Rationale
///
/// Sole point of coordinate translation — used by both
/// `server_patch_hint` and `client_patch_state` so the two paths cannot
/// diverge in subtle ways.
///
/// Both components are returned as `u32`: `reshape_off` can exceed
/// `u16::MAX` whenever `reshape_row_width > 65535` (large databases
/// with wide values), so a narrower return type would silently truncate.
#[inline]
const fn translate(orig_row: u32, orig_off: u16, k: u32, row_width: u32) -> (u32, u32) {
    let reshape_row = orig_row / k;
    let reshape_off = (orig_row % k) * row_width + orig_off as u32;
    (reshape_row, reshape_off)
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
/// `O(reshape_rows · lwe_dim)` wrapping multiply-add — this is the
/// per-query LWE cost amortised by `precompute_queries`.
fn sample_slot(params: &SimpleServerParams, material: &SimpleHintMaterial) -> PreparedSlot {
    let lwe_dim = params.params.lwe_dim as usize;
    let reshape_rows = params.reshape_rows as usize;
    let sigma = params.params.sigma;

    let mut rng = rand::rng();
    let mut secret = vec![0u32; lwe_dim];
    sample_uniform_zq_into(&mut rng, &mut secret);
    let mut e = vec![0u32; reshape_rows];
    sample_discrete_gaussian_into(&mut rng, sigma, &mut e);

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
/// # Complexity
///
/// `O(lwe_dim · reshape_row_width)` wrapping multiply-add — the most
/// expensive per-slot operation; this is what `precompute_decodes`
/// amortises across a batch.
fn compute_c(secret: &[u32], hint: &[u32], lwe_dim: usize, reshape_row_width: usize) -> Vec<u32> {
    debug_assert_eq!(secret.len(), lwe_dim);
    debug_assert_eq!(hint.len(), lwe_dim * reshape_row_width);
    let mut c = vec![0u32; reshape_row_width];
    matvec_accumulate(&mut c, hint, secret);
    c
}

impl PrecomputingPirBackend for SimplePirBackend {
    fn client_precompute_queries(state: &mut SimpleClientState, count: u32) {
        state.prepared.reserve(count as usize);
        for _ in 0..count {
            state
                .prepared
                .push_back(sample_slot(&state.params, &state.hint_material));
        }
    }

    fn client_precompute_decodes(state: &mut SimpleClientState) {
        let lwe_dim = state.params.params.lwe_dim as usize;
        let reshape_row_width = state.params.reshape_row_width as usize;
        let h = &state.hint.data;

        for slot in state.prepared.iter_mut() {
            if slot.c.is_none() {
                slot.c = Some(compute_c(&slot.secret, h, lwe_dim, reshape_row_width));
            }
        }
        if let Some(inflight) = state.in_flight.as_mut() {
            if inflight.slot.c.is_none() {
                inflight.slot.c = Some(compute_c(
                    &inflight.slot.secret,
                    h,
                    lwe_dim,
                    reshape_row_width,
                ));
            }
        }
    }

    fn prepared_slot_count(state: &SimpleClientState) -> usize {
        state.prepared_len()
    }

    fn in_flight_slot_count(state: &SimpleClientState) -> usize {
        state.in_flight_len()
    }
}

/// Compute the SimplePIR hint `H = Aᵀ · D mod 2³²` over the reshaped
/// matrix.
///
/// # Purpose
///
/// Setup-time matvec that produces the per-segment server hint. Output
/// is row-major shape `lwe_dim × reshape_row_width`:
/// `H[k, off] = Σ_R A[R, k] · D[R, off] mod q`,
/// where `R` indexes reshape rows and `off` indexes the
/// `reshape_row_width`-wide column space.
///
/// # Rationale
///
/// Loop nest is `r, k_idx, j` over the **original** `(n_rows, lwe_dim,
/// row_width)` shape so the flat `db` slice is read sequentially per
/// original row. Each original row contributes to a `row_width`-wide
/// sub-segment of one reshape row, starting at `off_within = (r % k) *
/// row_width`.
///
/// There is deliberately **no `aik == 0` shortcut**: `A` is uniform over
/// `Z_q` via ChaCha20, so a zero cell occurs with probability `2⁻³²` and
/// the test can never pay for itself. FrodoPIR's `compute_hint` omits it
/// on the same grounds, so the two backends stay comparable.
///
/// # Complexity
///
/// `O(n_rows · lwe_dim · row_width)` wrapping multiply-add — the same
/// asymptotic cost as FrodoPIR (the reshape changes the matrix shape,
/// not the total number of operations).
fn compute_hint(
    a: &[u32],
    db: &[u32],
    n_rows: u32,
    row_width: u32,
    k: u32,
    lwe_dim: u32,
    reshape_row_width: u32,
) -> Vec<u32> {
    let lwe_dim_us = lwe_dim as usize;
    let row_width_us = row_width as usize;
    let reshape_row_width_us = reshape_row_width as usize;
    let k_us = k as usize;
    let mut h = vec![0u32; lwe_dim_us * reshape_row_width_us];

    for r in 0..n_rows as usize {
        let reshape_row = r / k_us;
        let off_within = (r % k_us) * row_width_us;
        let a_row = &a[reshape_row * lwe_dim_us..(reshape_row + 1) * lwe_dim_us];
        let d_row = &db[r * row_width_us..(r + 1) * row_width_us];
        for k_idx in 0..lwe_dim_us {
            let aik = a_row[k_idx];
            let h_row = &mut h[k_idx * reshape_row_width_us..(k_idx + 1) * reshape_row_width_us];
            for j in 0..row_width_us {
                h_row[off_within + j] =
                    h_row[off_within + j].wrapping_add(aik.wrapping_mul(d_row[j]));
            }
        }
    }
    h
}

/// Multi-threaded twin of [`compute_hint`] — **bit-identical output**.
///
/// # Purpose
///
/// The optimized setup path's hint precompute (see
/// [`ParallelSetupBackend`]). This is where essentially all of setup's
/// `Θ(n_rows · lwe_dim · row_width)` arithmetic lives.
///
/// # Rationale
///
/// `H` splits by **bands of rows**: worker `t` owns hint rows
/// `[k₀, k₀ + band)` and runs the reference's `r, k, j` nest — reshape
/// coordinate translation included — restricted to that band. The
/// bands are disjoint output regions, so there is no reduction and no
/// cross-thread synchronisation, and each cell still accumulates over
/// `r` in the reference's order, making the result bit-identical.
///
/// Falls back to [`compute_hint`] on a single core or below
/// [`parallel::PAR_MIN_HINT_MACS`].
fn compute_hint_parallel(
    a: &[u32],
    db: &[u32],
    n_rows: u32,
    row_width: u32,
    k: u32,
    lwe_dim: u32,
    reshape_row_width: u32,
) -> Vec<u32> {
    let lwe_dim_us = lwe_dim as usize;
    let row_width_us = row_width as usize;
    let reshape_row_width_us = reshape_row_width as usize;
    let k_us = k as usize;
    let macs = u64::from(n_rows) * u64::from(lwe_dim) * u64::from(row_width);
    let threads = parallel::setup_threads();
    if threads <= 1 || macs < parallel::PAR_MIN_HINT_MACS {
        return compute_hint(a, db, n_rows, row_width, k, lwe_dim, reshape_row_width);
    }

    let mut h = vec![0u32; lwe_dim_us * reshape_row_width_us];
    // Chunk length is a whole multiple of `reshape_row_width`, so every
    // band starts on a hint-row boundary and `offset / reshape_row_width`
    // is the band's first `k`.
    let chunk = parallel::balanced_chunk_len(h.len(), reshape_row_width_us, threads);
    parallel::par_chunks_mut(&mut h, chunk, |offset, band| {
        let k0 = offset / reshape_row_width_us;
        // Whole hint rows per band — see the FrodoPIR twin for why the
        // floor is safe and why it is asserted anyway.
        debug_assert_eq!(
            band.len() % reshape_row_width_us,
            0,
            "band must be whole hint rows"
        );
        let band_rows = band.len() / reshape_row_width_us;
        for r in 0..n_rows as usize {
            let reshape_row = r / k_us;
            let off_within = (r % k_us) * row_width_us;
            let a_row =
                &a[reshape_row * lwe_dim_us + k0..reshape_row * lwe_dim_us + k0 + band_rows];
            let d_row = &db[r * row_width_us..(r + 1) * row_width_us];
            for (k_idx, &aik) in a_row.iter().enumerate() {
                let h_row =
                    &mut band[k_idx * reshape_row_width_us..(k_idx + 1) * reshape_row_width_us];
                for j in 0..row_width_us {
                    h_row[off_within + j] =
                        h_row[off_within + j].wrapping_add(aik.wrapping_mul(d_row[j]));
                }
            }
        }
    });
    h
}

/// Optimized setup for SimplePIR: same `(ServerParams, HintMaterial,
/// Hint)`, computed across cores.
///
/// Both heavy kernels fan out — `sample_a_parallel` (in `sampler.rs`) for
/// `A` and `compute_hint_parallel` (above) for `H = Aᵀ·D` over the reshaped database
/// — and both are bit-identical to their reference twins, so a server
/// set up on this path is indistinguishable from one set up on
/// [`IndexPirBackend::server_setup`].
impl ParallelSetupBackend for SimplePirBackend {
    fn server_setup_parallel(
        config: &SimpleConfig,
        db: &[u32],
        n_rows: u32,
        row_width: u32,
        plaintext_bits: u32,
    ) -> (SimpleServerParams, SimpleHintMaterial, SimpleHint) {
        debug_assert_eq!(db.len(), (n_rows as usize) * (row_width as usize));
        let lwe_dim = config.lwe_dim;
        let sigma = config.sigma;

        let (k, reshape_rows, reshape_row_width) = reshape_dims(n_rows, row_width);

        let mut seed = [0u8; 16];
        rand::rng().fill_bytes(&mut seed);
        let params = SimpleParams::new(lwe_dim, plaintext_bits, sigma, seed);

        let a = sample_a_parallel(&seed, reshape_rows, lwe_dim);
        let hint_data =
            compute_hint_parallel(&a, db, n_rows, row_width, k, lwe_dim, reshape_row_width);

        (
            SimpleServerParams {
                params,
                n_rows,
                row_width,
                k,
                reshape_rows,
                reshape_row_width,
            },
            SimpleHintMaterial { a },
            SimpleHint { data: hint_data },
        )
    }

    fn expand_hint_material_parallel(params: &SimpleServerParams) -> SimpleHintMaterial {
        let a = sample_a_parallel(
            &params.params.seed,
            params.reshape_rows,
            params.params.lwe_dim,
        );
        SimpleHintMaterial { a }
    }

    fn client_setup_parallel(params: &SimpleServerParams, hint: &SimpleHint) -> SimpleClientState {
        SimpleClientState {
            params: params.clone(),
            hint_material: Self::expand_hint_material_parallel(params),
            hint: hint.clone(),
            prepared: VecDeque::new(),
            in_flight: None,
        }
    }
}

/// Wire-size accounting for SimplePIR.
///
/// Reports the minimum fixed-width little-endian encoding of each wire
/// type. `SimpleQuery` is a dense `u32` vector of length `reshape_rows`;
/// `SimpleResponse` is a dense `u32` vector of length
/// `reshape_row_width`; `SimpleHint` is the `lwe_dim × reshape_row_width`
/// matrix; `SimpleServerParams` carries `(lwe_dim, plaintext_bits, sigma,
/// n_rows, row_width, k, reshape_rows, reshape_row_width)` and a 16-byte
/// seed — the public matrix `A` is **not** on the wire (it lives in
/// [`SimpleHintMaterial`] and the client re-expands it from the seed).
impl BackendWireSize for SimplePirBackend {
    fn query_byte_size(q: &SimpleQuery) -> usize {
        q.b.len() * 4
    }
    fn response_byte_size(r: &SimpleResponse) -> usize {
        r.a.len() * 4
    }
    fn hint_byte_size(h: &SimpleHint) -> usize {
        h.data.len() * 4
    }
    fn server_params_byte_size(_p: &SimpleServerParams) -> usize {
        // SimpleParams: { lwe_dim: u32, plaintext_bits: u32, sigma: f64, seed: [u8; 16] }
        let simple_params = 4 + 4 + 8 + 16;
        // Outer dims: n_rows + row_width + k + reshape_rows + reshape_row_width = 5 * u32
        let dims = 5 * 4;
        // The public matrix A lives in SimpleHintMaterial and never travels on the wire.
        simple_params + dims
    }
}

impl IncrementalPirBackend for SimplePirBackend {
    fn server_patch_hint(
        params: &SimpleServerParams,
        material: &SimpleHintMaterial,
        hint: &mut SimpleHint,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
        mode: HintPatchMode,
    ) {
        apply_patch(
            &material.a,
            params.params.lwe_dim,
            params.n_rows,
            params.row_width,
            params.k,
            params.reshape_rows,
            params.reshape_row_width,
            &mut hint.data,
            row_deltas,
            mode,
        );
    }

    fn client_patch_state(
        state: &mut SimpleClientState,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
        mode: HintPatchMode,
    ) {
        // Pull params + hint_material snapshots out of the state —
        // `client_setup` already stashed both, so no separate arguments
        // are threaded through.
        let SimpleClientState {
            params,
            hint_material,
            hint,
            prepared,
            in_flight,
        } = state;
        apply_patch(
            &hint_material.a,
            params.params.lwe_dim,
            params.n_rows,
            params.row_width,
            params.k,
            params.reshape_rows,
            params.reshape_row_width,
            &mut hint.data,
            row_deltas,
            mode,
        );
        // Slots that already carry `c = sᵀ·H` need their `c` patched in
        // lock-step. Slots with `c == None` are skipped — they will lazily
        // pick up the post-patch hint on first use.
        let prepared_iter = prepared.iter_mut();
        let in_flight_iter = in_flight.as_mut().map(|inflight| &mut inflight.slot);
        patch_slot_c(
            &hint_material.a,
            params.params.lwe_dim,
            params.row_width,
            params.k,
            params.reshape_row_width,
            prepared_iter.chain(in_flight_iter),
            row_deltas,
        );
    }
}

/// Apply the same hint-row deltas to every slot's precomputed `c`
/// vector.
///
/// # Rationale
///
/// `H_new[k, reshape_off] = H_old[k, reshape_off] + A[reshape_row, k] · Δ`,
/// so `c_new[reshape_off] = c_old[reshape_off]
///                        + (Σ_k secret[k] · A[reshape_row, k]) · Δ
///                        = c_old[reshape_off] + dot · Δ`,
/// where `dot = secret · A_{reshape_row}` is independent of the cell
/// offset. Compute `dot` once per `(slot, original-row)` pair, then
/// apply it to every `(off, Δ)` cell edit on that row.
///
/// # Complexity
///
/// `O(slots · row_deltas · (lwe_dim + n_cells))`.
#[allow(clippy::too_many_arguments)]
fn patch_slot_c<'a, I>(
    a: &[u32],
    lwe_dim: u32,
    row_width: u32,
    k: u32,
    reshape_row_width: u32,
    slots: I,
    row_deltas: &[(u32, Vec<(u16, i64)>)],
) where
    I: IntoIterator<Item = &'a mut PreparedSlot>,
{
    let lwe_dim_us = lwe_dim as usize;
    let reshape_row_width_us = reshape_row_width as usize;

    for slot in slots {
        let Some(c) = slot.c.as_mut() else {
            continue;
        };
        debug_assert_eq!(c.len(), reshape_row_width_us);
        debug_assert_eq!(slot.secret.len(), lwe_dim_us);

        for (orig_row, cells) in row_deltas {
            let (reshape_row, _) = translate(*orig_row, 0, k, row_width);
            let a_row =
                &a[(reshape_row as usize) * lwe_dim_us..(reshape_row as usize + 1) * lwe_dim_us];
            let dot: u32 = slot
                .secret
                .iter()
                .zip(a_row.iter())
                .fold(0u32, |acc, (&s, &ai)| acc.wrapping_add(s.wrapping_mul(ai)));
            for (orig_off, delta) in cells {
                if *delta == 0 {
                    continue;
                }
                let (_, reshape_off) = translate(*orig_row, *orig_off, k, row_width);
                let delta_u32 = *delta as u32;
                c[reshape_off as usize] =
                    c[reshape_off as usize].wrapping_add(dot.wrapping_mul(delta_u32));
            }
        }
    }
}

/// Apply sparse cell deltas to a hint laid out as
/// `lwe_dim × reshape_row_width` row-major, using the realization
/// selected by `mode`.
///
/// # Rationale
///
/// Dispatches to [`apply_patch_entry_level`] or
/// [`apply_patch_row_level`]; the two realizations produce
/// bit-identical hints (all arithmetic mod `2³²`) and differ only in
/// cost — see [`HintPatchMode`]. The distinction matters most for this
/// backend: the reshape makes the hint width `reshape_row_width ≈
/// √(segment cells)` grow with the database, so the dense row-level
/// pass grows with the database size while the entry-level pass stays
/// proportional to the touched cells.
#[allow(clippy::too_many_arguments)]
fn apply_patch(
    a: &[u32],
    lwe_dim: u32,
    n_rows: u32,
    row_width: u32,
    k: u32,
    reshape_rows: u32,
    reshape_row_width: u32,
    hint: &mut [u32],
    row_deltas: &[(u32, Vec<(u16, i64)>)],
    mode: HintPatchMode,
) {
    match mode {
        HintPatchMode::EntryLevel => apply_patch_entry_level(
            a,
            lwe_dim,
            n_rows,
            row_width,
            k,
            reshape_rows,
            reshape_row_width,
            hint,
            row_deltas,
        ),
        HintPatchMode::RowLevel => apply_patch_row_level(
            a,
            lwe_dim,
            n_rows,
            row_width,
            k,
            reshape_rows,
            reshape_row_width,
            hint,
            row_deltas,
        ),
    }
}

/// Entry-level realization (iSimplePIR): patch only the touched hint
/// columns.
///
/// # Rationale
///
/// `H[k, reshape_off] += A[reshape_row, k] · Δ mod 2³²`, where
/// `(reshape_row, reshape_off) = translate(orig_row, orig_off, k,
/// row_width)`. Iterates touched `(orig_row, orig_off, Δ)` triples and
/// slides `A`'s `reshape_row`-th column against the hint column at
/// `reshape_off`. No `aik == 0` shortcut — see [`compute_hint`] for why
/// `A` carries no exploitable sparsity.
///
/// The execution order is [`TouchedRuns`]': `k_idx` (the hint row)
/// outside, the row's touched columns — coalesced into contiguous runs —
/// inside, so the patch sweeps the hint once. `translate` is affine in
/// `orig_off` at fixed `orig_row`, so one slot's cells stay contiguous
/// after the reshape and coalesce into a single run. Getting this order
/// right matters most for this backend: the reshape makes the hint the
/// widest, so a per-column sweep of it is the most expensive.
///
/// # Complexity
///
/// `O(touched_cells · lwe_dim)` wrapping multiply-add — `Θ(n)` per
/// touched cell, independent of the reshape width, against
/// [`apply_patch_row_level`]'s `Θ(n · reshape_row_width)` per touched
/// reshape row.
#[allow(clippy::too_many_arguments)]
fn apply_patch_entry_level(
    a: &[u32],
    lwe_dim: u32,
    n_rows: u32,
    row_width: u32,
    k: u32,
    reshape_rows: u32,
    reshape_row_width: u32,
    hint: &mut [u32],
    row_deltas: &[(u32, Vec<(u16, i64)>)],
) {
    let lwe_dim_us = lwe_dim as usize;
    let reshape_row_width_us = reshape_row_width as usize;
    debug_assert_eq!(a.len(), (reshape_rows as usize) * lwe_dim_us);
    debug_assert_eq!(hint.len(), lwe_dim_us * reshape_row_width_us);

    // Hoisted out of the row loop: a batch of mutations allocates at most
    // once, however many rows it touches.
    let mut touched = TouchedRuns::new();

    for (orig_row, cells) in row_deltas {
        debug_assert!(
            *orig_row < n_rows,
            "orig_row {orig_row} out of range (n_rows={n_rows})",
        );
        touched.rebuild(cells, |orig_off| {
            debug_assert!(
                (orig_off as u32) < row_width,
                "orig_off {orig_off} out of range (row_width={row_width})",
            );
            translate(*orig_row, orig_off, k, row_width).1 as usize
        });
        if touched.is_empty() {
            continue;
        }
        let (reshape_row, _) = translate(*orig_row, 0, k, row_width);
        let a_row =
            &a[(reshape_row as usize) * lwe_dim_us..(reshape_row as usize + 1) * lwe_dim_us];
        touched.apply(a_row, hint, reshape_row_width_us);
    }
}

/// Row-level realization (SimplePIR): refresh the full reshape width for
/// every touched **reshape** row.
///
/// # Rationale
///
/// The paper's row-level patch operates on rows of the reshaped matrix
/// `D`, and the reshape packs `k` original (bucket) rows into each
/// reshape row — so edits from different original rows can land in the
/// same reshape row and should share **one** dense rank-one update.
///
/// Deltas arrive sorted by original row (`fold_mutations_into_row_deltas`
/// drains a `BTreeMap`), and `reshape_row = orig_row / k` is monotone in
/// `orig_row`, so the rows sharing a reshape row are already adjacent.
/// The scan therefore consumes one *run* at a time through a single
/// reused dense buffer: densify the run into a `reshape_row_width`-wide
/// delta vector, apply
/// `H[k_idx, ·] += A[reshape_row, k_idx] · δ[·] mod 2³²` across the
/// entire hint width, then clear only the cells that run wrote. Exactly
/// one dense vector is ever live. Materialising the whole grouping up
/// front instead would hold one per touched reshape row — ~70 MB against
/// ~31 KB for a τ = 1 % batch at the (4, 1) paper config.
///
/// Unsorted input stays **correct**, and is merely slower: the update is
/// linear and wrapping `u32` addition is associative, so splitting one
/// reshape row across several rank-one updates reaches the same hint. It
/// only costs extra full-width passes, which is why this is a run scan
/// and not a defensive sort. No `aik == 0` shortcut — see
/// [`compute_hint`].
///
/// # Complexity
///
/// `O(touched_reshape_rows · lwe_dim · reshape_row_width)` wrapping
/// multiply-add — `Θ(n·ω)` per touched reshape row with
/// `ω = reshape_row_width ≈ √(segment cells)`, the paper's row-level
/// cost. This is the term that grows with the database size and that
/// the entry-level realization avoids.
#[allow(clippy::too_many_arguments)]
fn apply_patch_row_level(
    a: &[u32],
    lwe_dim: u32,
    n_rows: u32,
    row_width: u32,
    k: u32,
    reshape_rows: u32,
    reshape_row_width: u32,
    hint: &mut [u32],
    row_deltas: &[(u32, Vec<(u16, i64)>)],
) {
    let lwe_dim_us = lwe_dim as usize;
    let reshape_row_width_us = reshape_row_width as usize;
    debug_assert_eq!(a.len(), (reshape_rows as usize) * lwe_dim_us);
    debug_assert_eq!(hint.len(), lwe_dim_us * reshape_row_width_us);

    // The working set, not an accumulator: exactly one dense delta vector
    // is live at any point, reused across every run.
    let mut dense = vec![0u32; reshape_row_width_us];

    let mut i = 0;
    while i < row_deltas.len() {
        let reshape_row = row_deltas[i].0 / k;
        let run_start = i;

        // Densify the run of original rows sharing this reshape row.
        while i < row_deltas.len() && row_deltas[i].0 / k == reshape_row {
            let (orig_row, cells) = &row_deltas[i];
            debug_assert!(
                *orig_row < n_rows,
                "orig_row {orig_row} out of range (n_rows={n_rows})",
            );
            // `off_within` is invariant across the row's cells, so it is
            // hoisted out of the inner loop rather than recomputed per cell.
            let off_within = (*orig_row % k) * row_width;
            for (orig_off, delta) in cells {
                debug_assert!(
                    (*orig_off as u32) < row_width,
                    "orig_off {orig_off} out of range (row_width={row_width})",
                );
                let cell_us = (off_within + u32::from(*orig_off)) as usize;
                dense[cell_us] = dense[cell_us].wrapping_add(*delta as u32);
            }
            i += 1;
        }

        let a_row =
            &a[(reshape_row as usize) * lwe_dim_us..(reshape_row as usize + 1) * lwe_dim_us];
        for (k_idx, &aik) in a_row.iter().enumerate() {
            let h_row = &mut hint[k_idx * reshape_row_width_us..(k_idx + 1) * reshape_row_width_us];
            for (h, &d) in h_row.iter_mut().zip(dense.iter()) {
                *h = h.wrapping_add(aik.wrapping_mul(d));
            }
        }

        // Clear only what this run wrote — `O(touched cells)`, not
        // `O(reshape_row_width)`. Idempotent under duplicate offsets.
        for (orig_row, cells) in &row_deltas[run_start..i] {
            let off_within = (*orig_row % k) * row_width;
            for (orig_off, _) in cells {
                dense[(off_within + u32::from(*orig_off)) as usize] = 0;
            }
        }
    }
}

impl ResponseRewind for SimplePirBackend {
    /// SimplePIR folds the segment into a near-square matrix, so a segment cell
    /// `(row, off)` maps to reshape coordinates `big_r = row/k`,
    /// `big_c = (row%k)·row_width + off`, and `resp.a[big_c] -= q.b[big_r]·δ`.
    /// The reshape parameters `k` / `row_width` are read off the client state's
    /// public `params` (the original, pre-reshape row width).
    fn rewind_response(
        state: &SimpleClientState,
        query: &SimpleQuery,
        resp: &mut SimpleResponse,
        deltas: &BTreeMap<(u32, u16), i64>,
    ) {
        let k = state.params.k;
        let row_width = state.params.row_width;
        for (&(row, off), &d) in deltas {
            let big_r = (row / k) as usize;
            let big_c = ((row % k) * row_width + u32::from(off)) as usize;
            // `d as u32` keeps the low 32 bits — exactly reduction mod 2³².
            let term = query.b[big_r].wrapping_mul(d as u32);
            resp.a[big_c] = resp.a[big_c].wrapping_sub(term);
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

    /// The optimized hint kernel is bit-identical to the reference —
    /// including the reshape coordinate translation and the ragged last
    /// reshape row (`n_rows % k != 0`, which the first shape triggers).
    /// Every shape exceeds `parallel::PAR_MIN_HINT_MACS` (asserted, so the test
    /// cannot silently go vacuous if the threshold moves) and the
    /// `lwe_dim` values are both multiples and non-multiples of any
    /// plausible worker count.
    #[test]
    fn compute_hint_parallel_matches_reference() {
        for (n_rows, row_width, lwe_dim) in
            [(259u32, 33u32, 256u32), (256, 16, 1031), (1024, 64, 17)]
        {
            assert!(
                u64::from(n_rows) * u64::from(lwe_dim) * u64::from(row_width)
                    >= parallel::PAR_MIN_HINT_MACS,
                "shape ({n_rows}, {row_width}, {lwe_dim}) must exceed the parallel threshold"
            );
            let (k, reshape_rows, reshape_row_width) = reshape_dims(n_rows, row_width);
            let db = make_db(n_rows, row_width, 8);
            let a = sample_a(&[0x9Eu8; 16], reshape_rows, lwe_dim);
            assert_eq!(
                compute_hint_parallel(&a, &db, n_rows, row_width, k, lwe_dim, reshape_row_width),
                compute_hint(&a, &db, n_rows, row_width, k, lwe_dim, reshape_row_width),
                "mismatch at n_rows={n_rows} row_width={row_width} lwe_dim={lwe_dim}"
            );
        }
    }

    /// End-to-end equivalence contract of [`ParallelSetupBackend`]: a
    /// server set up on the optimized path holds exactly the `(A, H)`
    /// the reference path would have derived from the same seed.
    #[test]
    fn parallel_setup_matches_reference_for_its_own_seed() {
        let (n_rows, row_width, pb) = (256u32, 32u32, 8u32);
        let db = make_db(n_rows, row_width, pb);
        let cfg = SimpleConfig::with_lwe_dim(256);

        let (sp, mat, hint) =
            SimplePirBackend::server_setup_parallel(&cfg, &db, n_rows, row_width, pb);

        // Same `A` as the reference expansion of this segment's seed …
        let reference_mat = SimplePirBackend::expand_hint_material(&sp);
        assert_eq!(mat.a, reference_mat.a);
        // … and the same hint the reference would have computed from it.
        assert_eq!(
            hint.data,
            compute_hint(
                &reference_mat.a,
                &db,
                n_rows,
                row_width,
                sp.k,
                sp.params.lwe_dim,
                sp.reshape_row_width
            )
        );
        // The client's optimized re-expansion agrees too.
        let state = SimplePirBackend::client_setup_parallel(&sp, &hint);
        assert_eq!(state.hint_material.a, reference_mat.a);
        assert_eq!(state.hint, hint);
    }

    /// Reshape arithmetic produces sane `(k, R, C)` for a range of
    /// shapes. Specifically checks: `k ≥ 1`, `R · k ≥ n_rows`,
    /// `C = k · row_width`, and the rough near-square property.
    #[test]
    fn reshape_dims_sane() {
        for &(n, w) in &[(8u32, 16), (16, 16), (64, 4), (8192, 16), (1, 1), (17, 4)] {
            let (k, r, c) = reshape_dims(n, w);
            assert!(k >= 1, "k must be ≥ 1, got {k} for ({n},{w})");
            assert!(r.saturating_mul(k) >= n, "R·k ≥ n: R={r} k={k} n={n}");
            assert_eq!(c, k * w, "C = k·row_width");
        }
    }

    /// `translate` round-trips per the math: orig (r, off) maps to
    /// reshape (r/k, (r%k)·row_width + off).
    #[test]
    fn translate_matches_math() {
        let k = 4u32;
        let row_width = 8u32;
        for r in 0..16u32 {
            for off in 0..row_width as u16 {
                let (rr, ro) = translate(r, off, k, row_width);
                assert_eq!(rr, r / k);
                assert_eq!(ro, (r % k) * row_width + off as u32);
            }
        }
    }

    /// Regression for the u16-truncation bug: when `reshape_row_width`
    /// exceeds `u16::MAX`, `translate` must return the un-narrowed offset
    /// so the downstream `as usize` cast indexes the hint correctly.
    /// Triggered when a database is large enough that `k · row_width ≥ 65536`.
    #[test]
    fn translate_handles_reshape_offsets_beyond_u16() {
        // k=128, row_width=520 → reshape_row_width = 66 560 (> u16::MAX).
        let k = 128u32;
        let row_width = 520u32;
        let last_bucket_within = k - 1; // 127
        let last_off_in_row = (row_width - 1) as u16; // 519

        // Original row in the *first* reshape row, last bucket-within slot:
        let (rr, ro) = translate(last_bucket_within, last_off_in_row, k, row_width);
        assert_eq!(rr, 0, "still in reshape_row 0");
        assert_eq!(
            ro,
            last_bucket_within * row_width + (last_off_in_row as u32)
        );
        assert!(
            ro > u16::MAX as u32,
            "reshape_off {ro} must exceed u16::MAX to exercise the regression"
        );
    }

    fn roundtrip_all_rows(n_rows: u32, row_width: u32) {
        let pb = 8u32;
        let db = make_db(n_rows, row_width, pb);
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        }; // smaller dim for test speed
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);
        for row in 0..n_rows {
            let q = SimplePirBackend::client_query(&mut state, row);
            let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = SimplePirBackend::client_decode(&state, &r);
            let expected: Vec<u32> = db
                [row as usize * row_width as usize..(row as usize + 1) * row_width as usize]
                .to_vec();
            assert_eq!(
                decoded, expected,
                "row {row} mismatch (n={n_rows}, w={row_width})"
            );
        }
    }

    #[test]
    fn roundtrip_square_16x16() {
        roundtrip_all_rows(16, 16);
    }

    #[test]
    fn roundtrip_wide_4x16() {
        roundtrip_all_rows(4, 16);
    }

    #[test]
    fn roundtrip_tall_64x4() {
        roundtrip_all_rows(64, 4);
    }

    /// Non-divisible reshape: 17 rows with row_width 4 → k = round(√4.25) = 2,
    /// R = 9 (padded). Last reshape row holds only 1 original row, the
    /// other "bucket slot" is zero-padded. All 17 rows should still
    /// roundtrip correctly.
    #[test]
    fn roundtrip_non_divisible_17x4() {
        roundtrip_all_rows(17, 4);
    }

    #[test]
    fn roundtrip_segment_shape_64x16() {
        let pb = 8u32;
        let n_rows = 64u32;
        let row_width = 16u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);
        for i in 0u32..100 {
            let row = (i.wrapping_mul(2_654_435_761)) % n_rows;
            let q = SimplePirBackend::client_query(&mut state, row);
            let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = SimplePirBackend::client_decode(&state, &r);
            let expected: Vec<u32> = db
                [row as usize * row_width as usize..(row as usize + 1) * row_width as usize]
                .to_vec();
            assert_eq!(decoded, expected, "iteration {i}, row {row} mismatch");
        }
    }

    #[test]
    fn decode_yields_high_bits_zero() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);
        for row in 0..n_rows {
            let q = SimplePirBackend::client_query(&mut state, row);
            let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = SimplePirBackend::client_decode(&state, &r);
            for &cell in &decoded {
                assert!(cell < (1 << pb), "cell {cell} exceeds p=2^{pb}");
            }
        }
    }

    #[test]
    fn setup_is_random_per_call() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp1, _, _) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let (sp2, _, _) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        assert_ne!(
            sp1.params.seed, sp2.params.seed,
            "seeds must differ across calls"
        );
    }

    /// Oracle: rebuilding the hint from scratch over the post-mutation
    /// `db` must match the incrementally patched hint.
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
        let n_rows = 8u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, mut hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);

        let actual_dlt = apply_cell_delta(&mut db, 1, 2, 1, row_width, pb);
        let row_deltas = vec![(1u32, vec![(2u16, actual_dlt)])];
        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );

        let expected = compute_hint(
            &mat.a,
            &db,
            n_rows,
            row_width,
            sp.k,
            sp.params.lwe_dim,
            sp.reshape_row_width,
        );
        assert_eq!(hint.data, expected);
    }

    #[test]
    fn patch_multi_cell_single_row_matches_oracle() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, mut hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);

        let raw_deltas: &[(u16, i64)] = &[(0, 3), (1, -1), (3, 2)];
        let cells: Vec<(u16, i64)> = raw_deltas
            .iter()
            .map(|&(off, dlt)| (off, apply_cell_delta(&mut db, 2, off, dlt, row_width, pb)))
            .collect();
        let row_deltas = vec![(2u32, cells)];
        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );

        let expected = compute_hint(
            &mat.a,
            &db,
            n_rows,
            row_width,
            sp.k,
            sp.params.lwe_dim,
            sp.reshape_row_width,
        );
        assert_eq!(hint.data, expected);
    }

    #[test]
    fn patch_multi_row_matches_oracle() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, mut hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);

        let raw: &[(u32, &[(u16, i64)])] = &[
            (0, &[(0, 1)]),
            (3, &[(1, -2), (3, -4)]),
            (7, &[(0, 3), (2, -1), (3, 1)]),
            (15, &[(2, 5)]),
        ];
        let row_deltas: Vec<(u32, Vec<(u16, i64)>)> = raw
            .iter()
            .map(|&(row, cells)| {
                let actual: Vec<(u16, i64)> = cells
                    .iter()
                    .map(|&(off, dlt)| {
                        (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb))
                    })
                    .collect();
                (row, actual)
            })
            .collect();
        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );

        let expected = compute_hint(
            &mat.a,
            &db,
            n_rows,
            row_width,
            sp.k,
            sp.params.lwe_dim,
            sp.reshape_row_width,
        );
        assert_eq!(hint.data, expected);
    }

    #[test]
    fn patch_zero_delta_is_noop() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp, mat, mut hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let hint_before = hint.data.clone();

        let row_deltas = vec![(2u32, vec![(1u16, 0i64)])];
        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );
        assert_eq!(hint.data, hint_before);
    }

    #[test]
    fn decode_after_patch_returns_patched_value() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);

        let target_row = 3u32;
        let dlt1 = apply_cell_delta(&mut db, target_row, 1, 7, row_width, pb);
        let dlt2 = apply_cell_delta(&mut db, target_row, 2, -3, row_width, pb);
        let row_deltas = vec![(target_row, vec![(1u16, dlt1), (2u16, dlt2)])];

        let mut server_hint = hint;
        SimplePirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut server_hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );
        SimplePirBackend::client_patch_state(&mut state, &row_deltas, HintPatchMode::EntryLevel);
        assert_eq!(
            state.hint.data, server_hint.data,
            "server and client patched hints must be identical"
        );

        let q = SimplePirBackend::client_query(&mut state, target_row);
        let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
        let decoded = SimplePirBackend::client_decode(&state, &r);
        let expected = db[target_row as usize * row_width as usize
            ..(target_row as usize + 1) * row_width as usize]
            .to_vec();
        assert_eq!(
            decoded, expected,
            "decode after patch did not recover patched row"
        );
    }

    /// With both phases warm, every query/decode round-trip still
    /// returns the right plaintext for every row.
    #[test]
    fn precomputed_roundtrip_all_rows() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 8u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);

        SimplePirBackend::client_precompute_queries(&mut state, n_rows);
        SimplePirBackend::client_precompute_decodes(&mut state);
        assert_eq!(state.prepared_len(), n_rows as usize);
        assert_eq!(state.in_flight_len(), 0);

        for row in 0..n_rows {
            let q = SimplePirBackend::client_query(&mut state, row);
            assert_eq!(state.in_flight_len(), 1);
            let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = SimplePirBackend::client_decode(&state, &r);
            let expected = db
                [row as usize * row_width as usize..(row as usize + 1) * row_width as usize]
                .to_vec();
            assert_eq!(decoded, expected, "row {row} mismatch");
        }
        assert_eq!(state.prepared_len(), 0);
    }

    #[test]
    fn precompute_decodes_idempotent() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);

        SimplePirBackend::client_precompute_queries(&mut state, 4);
        SimplePirBackend::client_precompute_decodes(&mut state);
        let snapshot: Vec<Option<Vec<u32>>> = state.prepared.iter().map(|s| s.c.clone()).collect();
        SimplePirBackend::client_precompute_decodes(&mut state);
        let after: Vec<Option<Vec<u32>>> = state.prepared.iter().map(|s| s.c.clone()).collect();
        assert_eq!(snapshot, after, "second precompute_decodes must be a no-op");
    }

    #[test]
    fn precomputed_decode_matches_on_the_fly() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);

        let mut warm = SimplePirBackend::client_setup(&sp, &hint);
        SimplePirBackend::client_precompute_queries(&mut warm, 4);
        SimplePirBackend::client_precompute_decodes(&mut warm);

        let mut cold = SimplePirBackend::client_setup(&sp, &hint);
        for slot in warm.prepared.iter() {
            cold.prepared.push_back(PreparedSlot {
                secret: slot.secret.clone(),
                b: slot.b.clone(),
                c: None,
            });
        }

        for row in 0..4u32 {
            let q_warm = SimplePirBackend::client_query(&mut warm, row);
            let q_cold = SimplePirBackend::client_query(&mut cold, row);
            assert_eq!(q_warm.b, q_cold.b, "queries diverged at row {row}");

            let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q_warm);
            let dec_warm = SimplePirBackend::client_decode(&warm, &r);
            let dec_cold = SimplePirBackend::client_decode(&cold, &r);
            assert_eq!(dec_warm, dec_cold, "decodes diverged at row {row}");
        }
    }

    /// After patching the hint, the precomputed `c` of every queued slot
    /// matches what `client_setup` on the post-patch hint would compute.
    #[test]
    fn patched_c_matches_recomputed_c() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);

        SimplePirBackend::client_precompute_queries(&mut state, 5);
        SimplePirBackend::client_precompute_decodes(&mut state);

        let raw: &[(u32, &[(u16, i64)])] = &[
            (1, &[(0, 3), (2, -1)]),
            (5, &[(1, 7)]),
            (13, &[(0, -2), (3, 4)]),
        ];
        let row_deltas: Vec<(u32, Vec<(u16, i64)>)> = raw
            .iter()
            .map(|&(row, cells)| {
                let actual: Vec<(u16, i64)> = cells
                    .iter()
                    .map(|&(off, dlt)| {
                        (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb))
                    })
                    .collect();
                (row, actual)
            })
            .collect();

        let secrets: Vec<Vec<u32>> = state.prepared.iter().map(|s| s.secret.clone()).collect();
        SimplePirBackend::client_patch_state(&mut state, &row_deltas, HintPatchMode::EntryLevel);

        let h_patched = compute_hint(
            &mat.a,
            &db,
            n_rows,
            row_width,
            sp.k,
            sp.params.lwe_dim,
            sp.reshape_row_width,
        );

        for (slot, secret) in state.prepared.iter().zip(secrets.iter()) {
            let oracle = compute_c(
                secret,
                &h_patched,
                sp.params.lwe_dim as usize,
                sp.reshape_row_width as usize,
            );
            assert_eq!(slot.c.as_ref().unwrap(), &oracle, "patched c diverged");
        }
    }

    /// `client_query` falls back to inline sampling when the prepared
    /// queue is empty.
    #[test]
    fn client_query_falls_back_when_queue_empty() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);
        assert_eq!(state.prepared_len(), 0);

        let q = SimplePirBackend::client_query(&mut state, 3);
        assert_eq!(state.in_flight_len(), 1);
        let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
        let decoded = SimplePirBackend::client_decode(&state, &r);
        let expected = db[3 * row_width as usize..4 * row_width as usize].to_vec();
        assert_eq!(decoded, expected);
    }

    /// `client_precompute_decodes` also fills `c` for an already-in-flight
    /// slot whose `c` was None.
    #[test]
    fn precompute_decodes_fills_in_flight_slot() {
        let pb = 8u32;
        let n_rows = 8u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);

        let q = SimplePirBackend::client_query(&mut state, 1);
        assert!(
            state.in_flight.as_ref().unwrap().slot.c.is_none(),
            "fresh inline-sampled slot starts with no c"
        );

        SimplePirBackend::client_precompute_decodes(&mut state);
        assert!(
            state.in_flight.as_ref().unwrap().slot.c.is_some(),
            "precompute_decodes must fill in-flight slot too"
        );

        let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
        let decoded = SimplePirBackend::client_decode(&state, &r);
        let expected = db[row_width as usize..2 * row_width as usize].to_vec();
        assert_eq!(decoded, expected);
    }

    /// FrodoPIR-symmetry test: 16 precomputed slots have 16 distinct secrets
    /// (verifies independent secret-per-slot sampling, per the FrodoPIR
    /// preprocessing audit in project_session9 memory).
    #[test]
    fn precomputed_slots_have_distinct_secrets() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);

        SimplePirBackend::client_precompute_queries(&mut state, 16);
        let secrets: Vec<&Vec<u32>> = state.prepared.iter().map(|s| &s.secret).collect();
        for i in 0..secrets.len() {
            for j in (i + 1)..secrets.len() {
                assert_ne!(
                    secrets[i], secrets[j],
                    "slot {i} and {j} share the same secret — must be pairwise distinct"
                );
            }
        }
    }

    // -------- row-level vs entry-level patch realizations --------

    /// The two [`HintPatchMode`] realizations must produce bit-identical
    /// hints across random multi-row bursts, and both must match the
    /// recomputed oracle at the end of the run. With `k = 2` here, random
    /// bursts regularly touch original rows that share a reshape row, so
    /// the row-level grouping is exercised throughout.
    #[test]
    fn row_level_patch_matches_entry_level_and_oracle() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, hint0) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        assert!(sp.k > 1, "test intends k > 1 so reshape rows are shared");
        let mut hint_entry = hint0.clone();
        let mut hint_row = hint0;
        let mask = (1u32 << pb) - 1;

        let mut rng = ChaCha20Rng::seed_from_u64(0xB0B5_CAFE);
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
            SimplePirBackend::server_patch_hint(
                &sp,
                &mat,
                &mut hint_entry,
                &row_deltas,
                HintPatchMode::EntryLevel,
            );
            SimplePirBackend::server_patch_hint(
                &sp,
                &mat,
                &mut hint_row,
                &row_deltas,
                HintPatchMode::RowLevel,
            );
            assert_eq!(
                hint_entry.data, hint_row.data,
                "realizations diverged at iter {_iter}"
            );
        }

        let expected = compute_hint(
            &mat.a,
            &db,
            n_rows,
            row_width,
            sp.k,
            sp.params.lwe_dim,
            sp.reshape_row_width,
        );
        assert_eq!(hint_row.data, expected, "row-level diverged from oracle");
    }

    /// Edits to two original rows that share one reshape row must fold
    /// into a single dense rank-one update and still match the oracle.
    #[test]
    fn row_level_groups_rows_sharing_reshape_row() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, mut hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        assert_eq!(
            sp.k, 2,
            "fixture intends orig rows 4 and 5 to share reshape row 2"
        );

        // Orig rows 4 and 5 → same reshape row (4/2 == 5/2 == 2).
        let d4 = apply_cell_delta(&mut db, 4, 1, 7, row_width, pb);
        let d5 = apply_cell_delta(&mut db, 5, 3, -2, row_width, pb);
        let row_deltas = vec![(4u32, vec![(1u16, d4)]), (5u32, vec![(3u16, d5)])];
        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::RowLevel,
        );

        let expected = compute_hint(
            &mat.a,
            &db,
            n_rows,
            row_width,
            sp.k,
            sp.params.lwe_dim,
            sp.reshape_row_width,
        );
        assert_eq!(hint.data, expected);
    }

    /// Row-level patch on the zero-padded final reshape row of a
    /// non-divisible reshape (17 rows, k = 2 → last reshape row holds a
    /// single original row) must match the oracle.
    #[test]
    fn row_level_patch_on_padded_tail_row_matches_oracle() {
        let pb = 8u32;
        let n_rows = 17u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, mut hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        assert_eq!(sp.k, 2);
        assert_eq!(sp.reshape_rows, 9, "17 rows at k=2 pad the 9th reshape row");

        let d = apply_cell_delta(&mut db, 16, 2, 5, row_width, pb);
        let row_deltas = vec![(16u32, vec![(2u16, d)])];
        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::RowLevel,
        );

        let expected = compute_hint(
            &mat.a,
            &db,
            n_rows,
            row_width,
            sp.k,
            sp.params.lwe_dim,
            sp.reshape_row_width,
        );
        assert_eq!(hint.data, expected);
    }

    /// Decode after a row-level `client_patch_state` recovers the patched
    /// plaintext, and the client hint equals the server's row-level hint.
    #[test]
    fn decode_after_row_level_patch_returns_patched_value() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);

        let target_row = 3u32;
        let dlt1 = apply_cell_delta(&mut db, target_row, 1, 7, row_width, pb);
        let dlt2 = apply_cell_delta(&mut db, target_row, 2, -3, row_width, pb);
        let row_deltas = vec![(target_row, vec![(1u16, dlt1), (2u16, dlt2)])];

        let mut server_hint = hint;
        SimplePirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut server_hint,
            &row_deltas,
            HintPatchMode::RowLevel,
        );
        SimplePirBackend::client_patch_state(&mut state, &row_deltas, HintPatchMode::RowLevel);
        assert_eq!(
            state.hint.data, server_hint.data,
            "server and client row-level patched hints must be identical"
        );

        let q = SimplePirBackend::client_query(&mut state, target_row);
        let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
        let decoded = SimplePirBackend::client_decode(&state, &r);
        let expected = db[target_row as usize * row_width as usize
            ..(target_row as usize + 1) * row_width as usize]
            .to_vec();
        assert_eq!(
            decoded, expected,
            "decode after row-level patch did not recover patched row"
        );
    }

    /// A zero delta is a no-op under the row-level realization too. Worth
    /// pinning separately from the entry-level case: row-level does not
    /// skip zero deltas — it densifies them into the shared buffer and
    /// runs the full-width pass anyway — so this also catches a buffer
    /// that the previous run failed to clear.
    #[test]
    fn row_level_patch_zero_delta_is_noop() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, mut hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        assert_eq!(sp.k, 2, "fixture intends a k > 1 reshape");

        // A real edit first, so the second (zero) patch runs against a
        // buffer the previous run has written to and cleared.
        let d = apply_cell_delta(&mut db, 4, 1, 9, row_width, pb);
        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint,
            &[(4u32, vec![(1u16, d)])],
            HintPatchMode::RowLevel,
        );
        let hint_before = hint.data.clone();

        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint,
            &[(5u32, vec![(3u16, 0i64)])],
            HintPatchMode::RowLevel,
        );

        assert_eq!(hint.data, hint_before, "zero delta must not move the hint");
    }

    /// Duplicate edits to the same `(row, cell)` within one call must
    /// accumulate identically under both realizations: entry-level applies
    /// them one by one, row-level sums them while densifying. The batch
    /// also spans two original rows sharing one reshape row (`4 / 2 ==
    /// 5 / 2`), so the duplicates land in the *same* dense buffer that the
    /// row-level run scan reuses.
    #[test]
    fn duplicate_cell_edits_accumulate_identically() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, hint0) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        assert_eq!(
            sp.k, 2,
            "fixture intends rows 4 and 5 to share reshape row 2"
        );

        let d1 = apply_cell_delta(&mut db, 4, 2, 3, row_width, pb);
        let d2 = apply_cell_delta(&mut db, 4, 2, 2, row_width, pb);
        let d3 = apply_cell_delta(&mut db, 5, 2, -6, row_width, pb);
        let row_deltas = vec![
            (4u32, vec![(2u16, d1), (2u16, d2)]),
            (5u32, vec![(2u16, d3)]),
        ];

        let mut hint_entry = hint0.clone();
        let mut hint_row = hint0;
        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint_entry,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );
        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint_row,
            &row_deltas,
            HintPatchMode::RowLevel,
        );
        assert_eq!(
            hint_entry.data, hint_row.data,
            "realizations diverged on duplicate cell edits"
        );

        let expected = compute_hint(
            &mat.a,
            &db,
            n_rows,
            row_width,
            sp.k,
            sp.params.lwe_dim,
            sp.reshape_row_width,
        );
        assert_eq!(hint_row.data, expected, "patched hint diverged from oracle");
    }

    /// Full round-trip under `RowLevel` with a **warm** Phase-B/Phase-C
    /// queue: precompute, patch via `client_patch_state(.., RowLevel)`,
    /// then decode. Exercises the interplay between the row-level hint
    /// patch and the (deliberately mode-independent) `patch_slot_c`, which
    /// the entry-level `patched_c_matches_recomputed_c` test leaves
    /// uncovered on this side.
    #[test]
    fn decode_after_row_level_patch_with_precomputed_c() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 4u32;
        let cfg = SimpleConfig {
            lwe_dim: 256,
            sigma: 6.4,
        };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, hint) = SimplePirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);
        let mut state = SimplePirBackend::client_setup(&sp, &hint);
        let mut server_hint = hint;

        SimplePirBackend::client_precompute_queries(&mut state, 4);
        SimplePirBackend::client_precompute_decodes(&mut state);

        // Rows 4 and 5 share a reshape row; row 11 sits in another one.
        let raw: &[(u32, &[(u16, i64)])] =
            &[(4, &[(0, 5), (3, -2)]), (5, &[(2, -4)]), (11, &[(1, 7)])];
        let row_deltas: Vec<(u32, Vec<(u16, i64)>)> = raw
            .iter()
            .map(|&(row, cells)| {
                let actual: Vec<(u16, i64)> = cells
                    .iter()
                    .map(|&(off, dlt)| {
                        (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb))
                    })
                    .collect();
                (row, actual)
            })
            .collect();

        SimplePirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut server_hint,
            &row_deltas,
            HintPatchMode::RowLevel,
        );
        SimplePirBackend::client_patch_state(&mut state, &row_deltas, HintPatchMode::RowLevel);
        assert_eq!(
            state.hint.data, server_hint.data,
            "server and client row-level patched hints must be identical"
        );

        for &(target_row, _) in raw {
            let q = SimplePirBackend::client_query(&mut state, target_row);
            let r = SimplePirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = SimplePirBackend::client_decode(&state, &r);
            let expected = db[target_row as usize * row_width as usize
                ..(target_row as usize + 1) * row_width as usize]
                .to_vec();
            assert_eq!(decoded, expected, "decode failed for row {target_row}");
        }
    }

    // ── Noise-margin probes (paper-scale, `#[ignore]`d) ─────────────────
    //
    // Empirical counterpart of `crate::pir_params`: measure the decode
    // noise `e·D` at real per-segment geometries with the exact wrapping
    // arithmetic of `server_answer` and the real error sampler, without
    // paying for the (noise-irrelevant) hint. Decode of a cell is wrong
    // iff the centered noise coordinate reaches `Δ/2`, so probing
    // `e·D` alone reproduces the correctness event exactly.
    //
    // Run with:  cargo test -p ikpir-common --release -- --ignored noise_margin

    /// Simulate `draws` independent decodes: each draw samples a fresh
    /// error vector (as `client_query` would) and checks the
    /// `row_width`-cell window a decode actually reads.
    ///
    /// The dominant noise component `(p/2)·Σᵣ eᵣ` is shared by every
    /// cell of one response, so decode failures cluster per query —
    /// counting failed *draws* over the real window is the metric that
    /// matches the protocol, and many draws (not many columns) is what
    /// gives the probe statistical power. Cells are uniform in
    /// `[0, 2^pb)` — the distribution `pack_slot_cells` produces for
    /// random fingerprints and values. `fingerprint_bits` feeds
    /// `row_width` the same way it feeds
    /// `crate::pir_params::simple_max_plaintext_bits`, so the probe
    /// window matches the geometry the selector was evaluated at.
    /// Returns `(failed_draws, overflowed_cells, max |noise| / (Δ/2))`.
    fn measure_decode_noise(
        segment_rows: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        pb: u32,
        draws: usize,
    ) -> (u64, u64, f64) {
        use rand::SeedableRng;
        let row_width = bucket_size * (fingerprint_bits + value_bits).div_ceil(pb);
        let (_, reshape_rows, _) = reshape_dims(segment_rows, row_width);
        let rows = reshape_rows as usize;
        let window = row_width as usize;
        let mask = (1u32 << pb) - 1;
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0x5A; 32]);
        let mut d = vec![0u32; rows * window];
        for cell in d.iter_mut() {
            *cell = rng.next_u32() & mask;
        }
        let delta_half = 1i64 << (32 - pb - 1);
        let mut e = vec![0u32; rows];
        let mut acc = vec![0u32; window];
        let (mut failed_draws, mut overflowed_cells, mut max_abs) = (0u64, 0u64, 0i64);
        for _ in 0..draws {
            sample_discrete_gaussian_into(&mut rng, 6.4, &mut e);
            acc.fill(0);
            for (row, &ei) in e.iter().enumerate() {
                let off = row * window;
                for (c, a) in acc.iter_mut().enumerate() {
                    *a = a.wrapping_add(ei.wrapping_mul(d[off + c]));
                }
            }
            let mut bad = 0u64;
            for &x in &acc {
                let mut v = i64::from(x);
                if v >= 1i64 << 31 {
                    v -= 1i64 << 32;
                }
                let a = v.abs();
                bad += u64::from(a >= delta_half);
                max_abs = max_abs.max(a);
            }
            overflowed_cells += bad;
            failed_draws += u64::from(bad > 0);
        }
        #[allow(clippy::cast_precision_loss)]
        (
            failed_draws,
            overflowed_cells,
            max_abs as f64 / delta_half as f64,
        )
    }

    /// EMPIRICAL EVIDENCE for the `pir_params` history note: the
    /// pre-fix operating point (`pb = 10` keyed on total capacity) at
    /// the paper's largest SimplePIR geometry — `(d, b) = (4, 1)`,
    /// `n_b = 2^22` (per-segment rows `2^20`), 1 KiB values, 32-bit
    /// fingerprint (as the pre-fix era ran, before fingerprints widened
    /// to 64 bits) — pushes the decode noise past `Δ/2` on a few
    /// percent of ordinary queries against random data, versus the
    /// `δ = 2⁻⁴⁰` the scheme promises. The old bench sweep was
    /// measuring a scheme that does not reliably decode.
    #[test]
    #[ignore = "paper-scale probe (~10 s in release; run with --release)"]
    fn noise_margin_rejects_old_pb10_operating_point() {
        let draws = 256;
        let (failed, cells, ratio) = measure_decode_noise(1 << 20, 1, 32, 8192, 10, draws);
        println!(
            "simple pb=10 @ (s=2^20, b=1, f=32, ℓ=8192): failed decodes {failed}/{draws} \
             ({cells} cells), max|noise|/(Δ/2) = {ratio:.3}"
        );
        assert!(
            failed > 0,
            "expected failed decodes at the rejected operating point (got none)"
        );
    }

    /// The operating points `pir_params::simple_max_plaintext_bits`
    /// selects at the new paper geometries (`fingerprint_bits = 64`)
    /// keep the measured noise strictly inside `Δ/2`.
    #[test]
    #[ignore = "paper-scale probe (~10 s in release; run with --release)"]
    fn noise_margin_validates_selected_operating_points() {
        let draws = 128;
        for (s, b, arity, vb, expected_pb) in [
            (1u32 << 17, 4u32, 2u32, 8192u32, 8u32),
            (1 << 18, 2, 3, 8192, 8),
            (1 << 18, 1, 4, 8192, 9),
            (1 << 17, 2, 4, 2048, 9),
        ] {
            let pb = crate::pir_params::simple_max_plaintext_bits(arity, s, b, 64, vb, 6.4);
            assert_eq!(
                pb, expected_pb,
                "selector regressed at (s={s}, b={b}, arity={arity}, ℓ={vb})"
            );
            let (failed, _, ratio) = measure_decode_noise(s, b, 64, vb, pb, draws);
            println!(
                "simple pb={pb} @ (s={s}, b={b}, arity={arity}, ℓ={vb}): \
                 failed decodes {failed}/{draws}, max|noise|/(Δ/2) = {ratio:.3}"
            );
            assert_eq!(failed, 0, "selected pb={pb} must not fail decodes");
            assert!(ratio < 1.0);
        }
    }
}
