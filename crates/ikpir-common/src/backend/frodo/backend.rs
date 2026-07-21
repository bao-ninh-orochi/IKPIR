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

use super::{
    round_p_to_q, round_q_to_p, sample_a, sample_a_parallel, sample_ternary_into, FrodoConfig,
    FrodoParams,
};
use crate::backend::gemm::gemm_at_d_accumulate;
use crate::backend::matvec::{matvec_accumulate, matvec_rows_accumulate};
#[cfg(feature = "parallel")]
use crate::backend::patch;
use crate::backend::patch::{apply_dense_rows, row_level_batch_rows, TouchedRuns};
use crate::backend::{
    parallel, BackendWireSize, HintPatchMode, IncrementalPirBackend, IndexPirBackend,
    ParallelSetupBackend, PrecomputingPirBackend,
};

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
    pub params: FrodoParams,
    /// Number of database rows in this segment.
    pub n_rows: u32,
    /// Number of `u32` cells per database row.
    pub row_width: u32,
}

/// Server-local working state: the LWE public matrix `A` in row-major
/// shape `n_rows × lwe_dim`, expanded deterministically from
/// [`FrodoServerParams::params`]`.seed` via `sample_a` (private to this
/// module).
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
    b: Vec<u32>,
    c: Option<Vec<u32>>,
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
    pub params: FrodoServerParams,
    /// Locally expanded copy of the LWE public matrix `A`, re-derived
    /// from `params.params.seed` during `client_setup`.
    pub hint_material: FrodoHintMaterial,
    /// Locally maintained copy of the segment hint matrix.
    pub hint: FrodoHint,
    /// Prepared but unconsumed slots, FIFO. Front is the next slot a
    /// `client_query` will use; back is where `client_precompute_queries`
    /// appends.
    prepared: VecDeque<PreparedSlot>,
    /// The slot consumed by the most recent `client_query`, if any. Read
    /// by the matching `client_decode`.
    in_flight: Option<PreparedSlot>,
}

impl FrodoClientState {
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
    type Config = FrodoConfig;
    type ServerParams = FrodoServerParams;
    type HintMaterial = FrodoHintMaterial;
    type Hint = FrodoHint;
    type ClientState = FrodoClientState;
    type Query = FrodoQuery;
    type Response = FrodoResponse;

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
            FrodoServerParams {
                params,
                n_rows,
                row_width,
            },
            FrodoHintMaterial { a },
            FrodoHint { data: hint_data },
        )
    }

    fn db_matrix_shape(params: &FrodoServerParams) -> (u32, u32) {
        // FrodoPIR multiplies the segment as handed to it — no reshape.
        (params.n_rows, params.row_width)
    }

    fn expand_hint_material(params: &FrodoServerParams) -> FrodoHintMaterial {
        let a = sample_a(&params.params.seed, params.n_rows, params.params.lwe_dim);
        FrodoHintMaterial { a }
    }

    fn client_setup(params: &FrodoServerParams, hint: &FrodoHint) -> FrodoClientState {
        FrodoClientState {
            params: params.clone(),
            hint_material: Self::expand_hint_material(params),
            hint: hint.clone(),
            prepared: VecDeque::new(),
            in_flight: None,
        }
    }

    fn client_query(state: &mut FrodoClientState, row: u32) -> FrodoQuery {
        let n_rows = state.params.n_rows;
        let plaintext_bits = state.params.params.plaintext_bits;
        debug_assert!(row < n_rows, "row {row} out of range (n_rows={n_rows})");

        // Cheap path if a prepared slot is available; otherwise sample inline
        // (matches the pre-precomputation behaviour for backward compatibility).
        let slot = state
            .prepared
            .pop_front()
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
        matvec_accumulate(&mut a, db, &query.b);
        FrodoResponse { a }
    }

    /// Inner loop is unconditional (no `sk == 0` short-circuit) so timing
    /// does not leak the secret's Hamming weight: `sk.wrapping_mul(0) = 0`
    /// and `x.wrapping_sub(0) = x`, so the arithmetic is unchanged.
    fn client_decode(state: &FrodoClientState, response: &FrodoResponse) -> Vec<u32> {
        let row_width = state.params.row_width as usize;
        let plaintext_bits = state.params.params.plaintext_bits;
        debug_assert_eq!(response.a.len(), row_width);

        // PROTOCOL INVARIANT (internal): see SimplePirBackend::client_decode.
        // `in_flight` is set by the matching `client_query`; a `None` here is
        // an unreachable backend bug, not a user-input error.
        let slot = state
            .in_flight
            .as_ref()
            .expect("client_decode invariant: in_flight set by matching client_query");

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
                // residual −= sᵀ·H, computed as residual += (−s)ᵀ·H — exact
                // mod 2³², and the one-time negation (data-independent) lets
                // the shared blocked kernel carry the heavy pass.
                let neg_secret: Vec<u32> = slot.secret.iter().map(|s| s.wrapping_neg()).collect();
                matvec_accumulate(&mut residual, &state.hint.data, &neg_secret);
            }
        }

        residual
            .iter()
            .map(|&y| round_q_to_p(y, plaintext_bits))
            .collect()
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
    let n_rows = params.n_rows as usize;

    let mut rng = rand::rng();
    let mut secret = vec![0u32; lwe_dim];
    sample_ternary_into(&mut rng, &mut secret);
    let mut e = vec![0u32; n_rows];
    sample_ternary_into(&mut rng, &mut e);

    // b = A·s + e — the size-adaptive row-blocked kernel (see
    // matvec_rows_accumulate) folds A·s into the error vector in place.
    let mut b = e;
    matvec_rows_accumulate(&mut b, &material.a, &secret);
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
    matvec_accumulate(&mut c, hint, secret);
    c
}

impl PrecomputingPirBackend for FrodoPirBackend {
    /// Slots are mutually independent random samples, so a Phase-B
    /// batch parallelises over slots (each task draws from its own
    /// thread-local RNG; queue order among fresh random slots carries
    /// no meaning, and rayon's indexed collect preserves it anyway).
    fn client_precompute_queries(state: &mut FrodoClientState, count: u32) {
        #[cfg(feature = "parallel")]
        if count >= 2 {
            use rayon::prelude::*;
            let (params, material) = (&state.params, &state.hint_material);
            let slots: Vec<PreparedSlot> = (0..count)
                .into_par_iter()
                .map(|_| sample_slot(params, material))
                .collect();
            state.prepared.extend(slots);
            return;
        }
        state.prepared.reserve(count as usize);
        for _ in 0..count {
            state
                .prepared
                .push_back(sample_slot(&state.params, &state.hint_material));
        }
    }

    /// Phase-C materialisation is one independent `sᵀ·H` per slot —
    /// parallel over the pending slots.
    fn client_precompute_decodes(state: &mut FrodoClientState) {
        let FrodoClientState {
            params,
            hint,
            prepared,
            in_flight,
            ..
        } = state;
        let lwe_dim = params.params.lwe_dim as usize;
        let row_width = params.row_width as usize;
        let h = &hint.data;

        let pending = prepared
            .iter_mut()
            .chain(in_flight.as_mut())
            .filter(|slot| slot.c.is_none());
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let pending: Vec<&mut PreparedSlot> = pending.collect();
            if pending.len() >= 2 {
                pending.into_par_iter().for_each(|slot| {
                    slot.c = Some(compute_c(&slot.secret, h, lwe_dim, row_width));
                });
            } else {
                for slot in pending {
                    slot.c = Some(compute_c(&slot.secret, h, lwe_dim, row_width));
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        for slot in pending {
            slot.c = Some(compute_c(&slot.secret, h, lwe_dim, row_width));
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
/// Delegates to the shared register-tiled [`gemm_at_d_accumulate`]
/// kernel (see `backend/gemm.rs` for the blocking scheme and the
/// bit-exactness argument) — the reference `i, k, j` rank-one-update
/// loop streamed the whole hint once per DB row and was bound on `H`
/// cache traffic, not multiply throughput.
///
/// There is deliberately **no `aik == 0` shortcut**. `A` is uniform over
/// `Z_{2³²}` (`sample_a` draws raw `next_u32` words), so a zero cell
/// occurs with probability `2⁻³²` and skipping it can never pay for the
/// test. The ternary distribution in this backend belongs to the LWE
/// *secret and error* (`sample_ternary_into`), not to `A`.
///
/// # Complexity
///
/// `O(n_rows · lwe_dim · row_width)` wrapping multiply-add — dominates
/// the cost of `server_setup` and `full_rebuild`.
fn compute_hint(a: &[u32], db: &[u32], n_rows: u32, lwe_dim: u32, row_width: u32) -> Vec<u32> {
    let lwe_dim_us = lwe_dim as usize;
    let row_width_us = row_width as usize;
    debug_assert_eq!(a.len(), n_rows as usize * lwe_dim_us);
    let mut h = vec![0u32; lwe_dim_us * row_width_us];
    gemm_at_d_accumulate(&mut h, a, db, lwe_dim_us, row_width_us);
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
/// On this branch [`compute_hint`] *is* the optimized kernel: the
/// register-tiled GEMM in `backend/gemm.rs` already fans its disjoint
/// `H` tiles out over rayon. So the twin is the same function, and
/// `ParallelSetupBackend`'s equivalence contract holds trivially.
/// Wrapping a second fan-out around it would only oversubscribe the
/// machine. `--no-default-features` selects the scoped-thread banding
/// instead — see the sibling definition below.
#[cfg(feature = "parallel")]
fn compute_hint_parallel(
    a: &[u32],
    db: &[u32],
    n_rows: u32,
    lwe_dim: u32,
    row_width: u32,
) -> Vec<u32> {
    compute_hint(a, db, n_rows, lwe_dim, row_width)
}

/// Multi-threaded twin of [`compute_hint`] — **bit-identical output**.
///
/// # Purpose
///
/// The `--no-default-features` build's optimized setup path: with the
/// `parallel` feature off, [`compute_hint`] runs the register-tiled
/// GEMM single-threaded, so [`ParallelSetupBackend`] still needs a
/// fan-out of its own.
///
/// # Rationale
///
/// `H` splits by **bands of rows**: worker `t` owns hint rows
/// `[k₀, k₀ + band)` and runs the reference's `i, k, j` nest restricted
/// to that band. The bands are disjoint output regions, so there is no
/// reduction and no cross-thread synchronisation — and each cell
/// `H[k, j]` still accumulates over `i` in the same increasing order
/// the reference uses, which is what makes the result bit-identical
/// rather than merely equivalent.
///
/// Banding by `k` (not by `i`) is also the cache-friendly choice: each
/// worker keeps a `band × row_width` slice of `H` hot while all workers
/// stream the same `D` rows in lockstep, sharing them in the outer
/// cache levels.
///
/// Falls back to [`compute_hint`] on a single core or below
/// [`parallel::PAR_MIN_HINT_MACS`].
#[cfg(not(feature = "parallel"))]
fn compute_hint_parallel(
    a: &[u32],
    db: &[u32],
    n_rows: u32,
    lwe_dim: u32,
    row_width: u32,
) -> Vec<u32> {
    let lwe_dim_us = lwe_dim as usize;
    let row_width_us = row_width as usize;
    let macs = u64::from(n_rows) * u64::from(lwe_dim) * u64::from(row_width);
    let threads = parallel::setup_threads();
    if threads <= 1 || macs < parallel::PAR_MIN_HINT_MACS {
        return compute_hint(a, db, n_rows, lwe_dim, row_width);
    }

    let mut h = vec![0u32; lwe_dim_us * row_width_us];
    // Chunk length is a whole multiple of `row_width`, so every band
    // starts on a hint-row boundary and `offset / row_width` is the
    // band's first `k`.
    let chunk = parallel::balanced_chunk_len(h.len(), row_width_us, threads);
    parallel::par_chunks_mut(&mut h, chunk, |offset, band| {
        let k0 = offset / row_width_us;
        // Whole hint rows per band: `balanced_chunk_len` was given
        // `row_width` as its unit and `h.len()` is `lwe_dim · row_width`,
        // so every chunk — including the ragged last one — divides
        // exactly. The floor below would silently drop the tail rows if
        // that ever stopped holding.
        debug_assert_eq!(band.len() % row_width_us, 0, "band must be whole hint rows");
        let band_rows = band.len() / row_width_us;
        for i in 0..n_rows as usize {
            let a_row = &a[i * lwe_dim_us + k0..i * lwe_dim_us + k0 + band_rows];
            let d_row = &db[i * row_width_us..(i + 1) * row_width_us];
            for (k, &aik) in a_row.iter().enumerate() {
                let h_row = &mut band[k * row_width_us..(k + 1) * row_width_us];
                for j in 0..row_width_us {
                    h_row[j] = h_row[j].wrapping_add(aik.wrapping_mul(d_row[j]));
                }
            }
        }
    });
    h
}

/// Optimized setup for FrodoPIR: same `(ServerParams, HintMaterial,
/// Hint)`, computed across cores.
///
/// Both heavy kernels fan out — `sample_a_parallel` (in `sampler.rs`) for
/// `A` and `compute_hint_parallel` (above) for `H = Aᵀ·D` — and both are bit-identical
/// to their reference twins, so a server set up on this path is
/// indistinguishable from one set up on [`IndexPirBackend::server_setup`].
impl ParallelSetupBackend for FrodoPirBackend {
    fn server_setup_parallel(
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

        let a = sample_a_parallel(&seed, n_rows, lwe_dim);
        let hint_data = compute_hint_parallel(&a, db, n_rows, lwe_dim, row_width);

        (
            FrodoServerParams {
                params,
                n_rows,
                row_width,
            },
            FrodoHintMaterial { a },
            FrodoHint { data: hint_data },
        )
    }

    fn expand_hint_material_parallel(params: &FrodoServerParams) -> FrodoHintMaterial {
        let a = sample_a_parallel(&params.params.seed, params.n_rows, params.params.lwe_dim);
        FrodoHintMaterial { a }
    }

    fn client_setup_parallel(params: &FrodoServerParams, hint: &FrodoHint) -> FrodoClientState {
        FrodoClientState {
            params: params.clone(),
            hint_material: Self::expand_hint_material_parallel(params),
            hint: hint.clone(),
            prepared: VecDeque::new(),
            in_flight: None,
        }
    }
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
        let frodo_params = 4 + 4 + 16; // lwe_dim + plaintext_bits + seed
        let dims = 4 + 4; // n_rows + row_width
        frodo_params + dims
    }
}

impl IncrementalPirBackend for FrodoPirBackend {
    fn server_patch_hint(
        params: &FrodoServerParams,
        material: &FrodoHintMaterial,
        hint: &mut FrodoHint,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
        mode: HintPatchMode,
    ) {
        apply_patch(
            &material.a,
            params.params.lwe_dim,
            params.n_rows,
            params.row_width,
            &mut hint.data,
            row_deltas,
            mode,
        );
    }

    fn client_patch_state(
        state: &mut FrodoClientState,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
        mode: HintPatchMode,
    ) {
        // Pull params + hint_material snapshots out of the state —
        // `client_setup` already stashed both, so no separate arguments
        // are threaded through.
        let FrodoClientState {
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
            &mut hint.data,
            row_deltas,
            mode,
        );
        // Slots that already carry `c = sᵀ·H` need their `c` patched in
        // lock-step. Slots with `c == None` are skipped — they will lazily
        // pick up the post-patch hint on first use.
        patch_slot_c(
            &hint_material.a,
            params.params.lwe_dim,
            params.row_width,
            prepared.iter_mut().chain(in_flight.as_mut()),
            row_deltas,
        );
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
/// Deliberately **independent of [`HintPatchMode`]**: the `c`-patch is
/// inherently sparse (`dot · Δ` per touched cell, with the `lwe_dim`-long
/// `dot` shared per row), and a dense per-row variant would compute the
/// same values while only inflating the Phase-C maintenance cost. The
/// mode knob governs the hint-matrix patch — the paper's `HintUpdate` —
/// not this bookkeeping.
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
) where
    I: IntoIterator<Item = &'a mut PreparedSlot>,
{
    let lwe_dim_us = lwe_dim as usize;
    let row_width_us = row_width as usize;

    // Slots are independent (each owns its secret and c), so a warm queue
    // parallelises trivially — but only once there is enough work to pay
    // for the fan-out. The per-slot cost is `row_deltas · lwe_dim`
    // multiply-adds, so a single-mutation delta against a warm queue of
    // sixteen slots is ~25 k MACs total: two orders of magnitude under
    // the gate, and measurably *slower* threaded. A τ-sized batch clears
    // it comfortably.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let slot_refs: Vec<&mut PreparedSlot> = slots.into_iter().collect();
        let work = slot_refs
            .len()
            .saturating_mul(row_deltas.len())
            .saturating_mul(lwe_dim_us);
        if slot_refs.len() >= 2 && work >= patch::PATCH_PAR_MIN_MACS && parallel::kernels_parallel()
        {
            slot_refs.into_par_iter().for_each(|slot| {
                patch_one_slot_c(a, lwe_dim_us, row_width_us, slot, row_deltas);
            });
        } else {
            for slot in slot_refs {
                patch_one_slot_c(a, lwe_dim_us, row_width_us, slot, row_deltas);
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    for slot in slots {
        patch_one_slot_c(a, lwe_dim_us, row_width_us, slot, row_deltas);
    }
}

/// Patch a single slot's `c` (no-op when Phase C hasn't run for it).
fn patch_one_slot_c(
    a: &[u32],
    lwe_dim_us: usize,
    row_width_us: usize,
    slot: &mut PreparedSlot,
    row_deltas: &[(u32, Vec<(u16, i64)>)],
) {
    let Some(c) = slot.c.as_mut() else {
        return;
    };
    debug_assert_eq!(c.len(), row_width_us);
    debug_assert_eq!(slot.secret.len(), lwe_dim_us);

    for (row_idx, cells) in row_deltas {
        let a_row = &a[(*row_idx as usize) * lwe_dim_us..(*row_idx as usize + 1) * lwe_dim_us];
        // dot = secret · A_row (offset-independent).
        let dot: u32 = slot
            .secret
            .iter()
            .zip(a_row.iter())
            .fold(0u32, |acc, (&s, &ai)| acc.wrapping_add(s.wrapping_mul(ai)));
        for (off, delta) in cells {
            if *delta == 0 {
                continue;
            }
            let delta_u32 = *delta as u32;
            c[*off as usize] = c[*off as usize].wrapping_add(dot.wrapping_mul(delta_u32));
        }
    }
}

/// Apply sparse cell deltas to a hint laid out as
/// `lwe_dim × row_width` row-major, using the realization selected by
/// `mode`.
///
/// # Purpose
///
/// Core of incremental hint patching for both server
/// (`server_patch_hint`) and client (`client_patch_state`); both
/// delegate here so the two paths cannot diverge. Dispatches to
/// [`apply_patch_entry_level`] or [`apply_patch_row_level`]; the two
/// realizations produce bit-identical hints (all arithmetic mod `2³²`)
/// and differ only in cost — see [`HintPatchMode`].
fn apply_patch(
    a: &[u32],
    lwe_dim: u32,
    n_rows: u32,
    row_width: u32,
    hint: &mut [u32],
    row_deltas: &[(u32, Vec<(u16, i64)>)],
    mode: HintPatchMode,
) {
    match mode {
        HintPatchMode::EntryLevel => {
            apply_patch_entry_level(a, lwe_dim, n_rows, row_width, hint, row_deltas);
        }
        HintPatchMode::RowLevel => {
            apply_patch_row_level(a, lwe_dim, n_rows, row_width, hint, row_deltas);
        }
    }
}

/// Entry-level realization (iSimplePIR): patch only the touched hint
/// columns.
///
/// # Rationale
///
/// Math: `H[k, cell] += A[row_idx, k] · Δ mod 2³²`. Iterates touched
/// `(row, cell, Δ)` triples and slides `A`'s `row_idx`-th column
/// against the hint column at `cell`; untouched columns are never read
/// or written. No `aik == 0` shortcut — see [`compute_hint`] for why `A`
/// carries no exploitable sparsity.
///
/// The execution order is [`TouchedRuns`]': `k` (the hint row) outside,
/// the row's touched columns — coalesced into contiguous runs — inside,
/// so the patch sweeps the hint once. FrodoPIR patches column
/// `cell_offset` directly; there is no reshape to translate through.
///
/// Large bursts then split that single sweep by `k`: each rayon task
/// owns a contiguous **band** of hint rows and replays every row's runs
/// into it. Bands are disjoint output regions, so each `(row, cell, k)`
/// term is still added exactly once and the result is bit-identical to
/// the one-band pass for any banding. The two optimisations are
/// orthogonal — coalescing fixes *what order* the hint is walked in,
/// banding fixes *how many cores* walk it — and compose without either
/// giving anything up.
///
/// # Complexity
///
/// `O(touched_cells · lwe_dim)` wrapping multiply-add — `Θ(n)` per
/// touched cell, the paper's entry-level cost. Vastly cheaper than a
/// full `compute_hint` when the mutation count is small, and
/// `bucket_size×` cheaper than [`apply_patch_row_level`] on the same
/// batch.
fn apply_patch_entry_level(
    a: &[u32],
    lwe_dim: u32,
    n_rows: u32,
    row_width: u32,
    hint: &mut [u32],
    row_deltas: &[(u32, Vec<(u16, i64)>)],
) {
    let lwe_dim_us = lwe_dim as usize;
    let row_width_us = row_width as usize;
    debug_assert_eq!(a.len(), (n_rows as usize) * lwe_dim_us);
    debug_assert_eq!(hint.len(), lwe_dim_us * row_width_us);
    if row_width_us == 0 {
        return;
    }

    #[cfg(feature = "parallel")]
    {
        let touched_cells: usize = row_deltas.iter().map(|(_, cells)| cells.len()).sum();
        if touched_cells * lwe_dim_us >= patch::PATCH_PAR_MIN_MACS && parallel::kernels_parallel() {
            patch::apply_banded(
                a,
                lwe_dim_us,
                hint,
                row_width_us,
                row_deltas,
                |row_idx, cells| {
                    let mut runs = TouchedRuns::new();
                    runs.rebuild(cells, |off| off as usize);
                    (row_idx as usize, runs)
                },
            );
            return;
        }
    }

    // Hoisted out of the row loop: a batch of mutations allocates at most
    // once, however many rows it touches.
    let mut touched = TouchedRuns::new();

    for (row_idx, cells) in row_deltas {
        debug_assert!(
            (*row_idx as usize) < n_rows as usize,
            "row_idx {row_idx} out of range (n_rows={n_rows})",
        );
        touched.rebuild(cells, |off| {
            debug_assert!(
                (off as usize) < row_width_us,
                "cell_offset {off} out of range (row_width={row_width})",
            );
            off as usize
        });
        if touched.is_empty() {
            continue;
        }
        let a_row = &a[(*row_idx as usize) * lwe_dim_us..(*row_idx as usize + 1) * lwe_dim_us];
        touched.apply(a_row, hint, row_width_us);
    }
}

/// Row-level realization (SimplePIR): refresh the full hint width for
/// every touched row.
///
/// # Rationale
///
/// The literal reading is one dense rank-one update per touched row:
/// densify that row's sparse edits into a `row_width`-wide delta vector
/// `δ` (zero at untouched columns) and fold
/// `H[k, ·] += A[row_idx, k] · δ[·]` across the whole hint. Columns with
/// `δ = 0` are still multiplied — that is the point: this is the
/// row-granular patch of SimplePIR, kept as the baseline the entry-level
/// sharpening is measured against.
///
/// Executed one row at a time it also streams the entire hint once
/// **per touched row**, and a τ = 1 % mutation batch touches thousands.
/// So rows are densified in *chunks*, and each chunk applied as a single
/// `H += A_selᵀ · Δ` product through the shared register-tiled
/// [`gemm_at_d_accumulate`] — the same kernel that computes hints. The
/// hint is then streamed once per chunk rather than once per row, and
/// the multiply-adds run tiled instead of one accumulator load and store
/// apiece. A chunk too shallow to give the tiles any contraction to
/// amortise keeps the reference rank-one pass; `patch::apply_dense_rows`
/// owns that choice and the measurement behind it.
///
/// Chunking rather than one product over the whole batch is what keeps
/// the working set bounded: `Δ` plus the gathered `A` rows is capped at
/// [`patch::ROW_LEVEL_BATCH_CELLS`], a few megabytes, where the whole batch at
/// paper scale would be tens to hundreds. Bit-exact either way — each
/// `H[k, j]` gains `Σ_r A[r, k] · δ_r[j]` and wrapping addition is
/// associative and commutative, so any chunking sums the same terms. No
/// `aik == 0` shortcut — see [`compute_hint`].
///
/// # Complexity
///
/// `O(touched_rows · lwe_dim · row_width)` wrapping multiply-add —
/// `Θ(n·ω)` per touched row, the paper's row-level cost.
fn apply_patch_row_level(
    a: &[u32],
    lwe_dim: u32,
    n_rows: u32,
    row_width: u32,
    hint: &mut [u32],
    row_deltas: &[(u32, Vec<(u16, i64)>)],
) {
    let lwe_dim_us = lwe_dim as usize;
    let row_width_us = row_width as usize;
    debug_assert_eq!(a.len(), (n_rows as usize) * lwe_dim_us);
    debug_assert_eq!(hint.len(), lwe_dim_us * row_width_us);

    let batch_rows = row_level_batch_rows(row_width_us, lwe_dim_us);
    // Reused across chunks: `clear` + `resize` refills with zeros without
    // giving the capacity back, so only the first chunk allocates.
    let mut delta: Vec<u32> = Vec::new();
    let mut a_sel: Vec<u32> = Vec::new();

    for chunk in row_deltas.chunks(batch_rows) {
        let t = chunk.len();
        delta.clear();
        delta.resize(t * row_width_us, 0);
        a_sel.clear();
        a_sel.resize(t * lwe_dim_us, 0);

        for (r, (row_idx, cells)) in chunk.iter().enumerate() {
            debug_assert!(
                (*row_idx as usize) < n_rows as usize,
                "row_idx {row_idx} out of range (n_rows={n_rows})",
            );
            let delta_row = &mut delta[r * row_width_us..(r + 1) * row_width_us];
            for (cell_offset, d) in cells {
                debug_assert!(
                    (*cell_offset as usize) < row_width_us,
                    "cell_offset {cell_offset} out of range (row_width={row_width})",
                );
                let cell_us = *cell_offset as usize;
                delta_row[cell_us] = delta_row[cell_us].wrapping_add(*d as u32);
            }
            a_sel[r * lwe_dim_us..(r + 1) * lwe_dim_us].copy_from_slice(
                &a[(*row_idx as usize) * lwe_dim_us..(*row_idx as usize + 1) * lwe_dim_us],
            );
        }

        apply_dense_rows(hint, &a_sel, &delta, lwe_dim_us, row_width_us);
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);
        for row in 0..n_rows {
            let q = FrodoPirBackend::client_query(&mut state, row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            let expected: Vec<u32> = db
                [row as usize * row_width as usize..(row as usize + 1) * row_width as usize]
                .to_vec();
            assert_eq!(decoded, expected, "row {row} mismatch");
        }
    }

    /// The optimized hint kernel is bit-identical to the reference.
    /// Every shape exceeds `parallel::PAR_MIN_HINT_MACS` (asserted, so the test
    /// cannot silently go vacuous if the threshold moves) and the
    /// `lwe_dim` values are both multiples and non-multiples of any
    /// plausible worker count, exercising the ragged last band.
    #[test]
    fn compute_hint_parallel_matches_reference() {
        for (n_rows, row_width, lwe_dim) in
            [(37u32, 257u32, 256u32), (32, 64, 1031), (24, 4096, 17)]
        {
            assert!(
                u64::from(n_rows) * u64::from(lwe_dim) * u64::from(row_width)
                    >= parallel::PAR_MIN_HINT_MACS,
                "shape ({n_rows}, {row_width}, {lwe_dim}) must exceed the parallel threshold"
            );
            let db = make_db(n_rows, row_width, 8);
            let a = sample_a(&[0x9Eu8; 16], n_rows, lwe_dim);
            assert_eq!(
                compute_hint_parallel(&a, &db, n_rows, lwe_dim, row_width),
                compute_hint(&a, &db, n_rows, lwe_dim, row_width),
                "mismatch at n_rows={n_rows} row_width={row_width} lwe_dim={lwe_dim}"
            );
        }
    }

    /// End-to-end equivalence contract of [`ParallelSetupBackend`]: a
    /// server set up on the optimized path holds exactly the `(A, H)`
    /// the reference path would have derived from the same seed.
    #[test]
    fn parallel_setup_matches_reference_for_its_own_seed() {
        let (n_rows, row_width, pb) = (64u32, 300u32, 8u32);
        let db = make_db(n_rows, row_width, pb);
        let cfg = FrodoConfig::with_lwe_dim(256);

        let (sp, mat, hint) =
            FrodoPirBackend::server_setup_parallel(&cfg, &db, n_rows, row_width, pb);

        // Same `A` as the reference expansion of this segment's seed …
        let reference_mat = FrodoPirBackend::expand_hint_material(&sp);
        assert_eq!(mat.a, reference_mat.a);
        // … and the same hint the reference would have computed from it.
        assert_eq!(
            hint.data,
            compute_hint(&reference_mat.a, &db, n_rows, sp.params.lwe_dim, row_width)
        );
        // The client's optimized re-expansion agrees too.
        let state = FrodoPirBackend::client_setup_parallel(&sp, &hint);
        assert_eq!(state.hint_material.a, reference_mat.a);
        assert_eq!(state.hint, hint);
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);
        // Query 100 random rows (with wrapping) to stress the matvecs.
        for i in 0u32..100 {
            let row = (i.wrapping_mul(2_654_435_761)) % n_rows;
            let q = FrodoPirBackend::client_query(&mut state, row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
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
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
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
        let (sp1, _, _) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let (sp2, _, _) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        assert_ne!(
            sp1.params.seed, sp2.params.seed,
            "seeds must differ across calls"
        );
    }

    #[test]
    fn hint_matches_explicit_atd() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

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
        let (sp, _mat, mut hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        let actual_dlt = apply_cell_delta(&mut db, 1, 2, 1, row_width, pb);
        let row_deltas = vec![(1u32, vec![(2u16, actual_dlt)])];
        FrodoPirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );

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
        let (sp, _mat, mut hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        let raw_deltas: &[(u16, i64)] = &[(0, 3), (1, -1), (3, 2)];
        let cells: Vec<(u16, i64)> = raw_deltas
            .iter()
            .map(|&(off, dlt)| (off, apply_cell_delta(&mut db, 2, off, dlt, row_width, pb)))
            .collect();
        let row_deltas = vec![(2u32, cells)];
        FrodoPirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );

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
        let (sp, _mat, mut hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

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
                    .map(|&(off, dlt)| {
                        (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb))
                    })
                    .collect();
                (row, actual)
            })
            .collect();
        FrodoPirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );

        let lwe_dim = sp.params.lwe_dim;
        let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
        let expected = compute_hint(&a, &db, n_rows, lwe_dim, row_width);
        assert_eq!(hint.data, expected);
    }

    /// A mutation batch big enough to cross `PATCH_PAR_MIN_MACS` — the
    /// threshold is asserted, so the test cannot go vacuous if it moves —
    /// patches identically under both realizations and matches a
    /// recomputed hint. This is the only coverage of the banded
    /// entry-level sweep at a shape a caller actually produces.
    #[test]
    fn patch_batch_above_parallel_gate_matches_oracle() {
        let pb = 8u32;
        let (n_rows, row_width) = (128u32, 16u32);
        let cfg = FrodoConfig { lwe_dim: 512 };
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, hint0) = FrodoPirBackend::server_setup(&cfg, &db, n_rows, row_width, pb);

        // Every cell of every row: 2048 touched cells × 512 = 2²⁰ MACs.
        let row_deltas: Vec<(u32, Vec<(u16, i64)>)> = (0..n_rows)
            .map(|row| {
                let cells = (0..row_width as u16)
                    .map(|off| {
                        let dlt = i64::from(off).wrapping_sub(7).wrapping_mul(3);
                        (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb))
                    })
                    .collect();
                (row, cells)
            })
            .collect();
        #[cfg(feature = "parallel")]
        {
            let touched: usize = row_deltas.iter().map(|(_, c)| c.len()).sum();
            assert!(
                touched * cfg.lwe_dim as usize >= crate::backend::patch::PATCH_PAR_MIN_MACS,
                "fixture must exceed the entry-level fan-out threshold"
            );
        }

        let mut hint_entry = hint0.clone();
        let mut hint_row = hint0;
        FrodoPirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint_entry,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );
        FrodoPirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint_row,
            &row_deltas,
            HintPatchMode::RowLevel,
        );

        let expected = compute_hint(&mat.a, &db, n_rows, cfg.lwe_dim, row_width);
        assert_eq!(
            hint_entry.data, expected,
            "entry-level diverged from oracle"
        );
        assert_eq!(hint_row.data, expected, "row-level diverged from oracle");
    }

    /// The row-level pass fires one GEMM per chunk of touched rows, so a
    /// batch spanning several chunks must reach the same hint as one
    /// spanning one — in particular the chunk buffer has to be zeroed
    /// between products. The shape is synthetic (a hint far wider than
    /// any real segment) purely to make `row_level_batch_rows` small
    /// enough that a cheap test crosses the boundary; the loop it
    /// exercises depends on nothing else.
    #[test]
    fn row_level_patch_spans_several_chunks() {
        let (lwe_dim, row_width) = (4u32, 262_140u32);
        let cap = crate::backend::patch::row_level_batch_rows(row_width as usize, lwe_dim as usize);
        let n_rows = (cap as u32) * 2 + 3;
        assert!(
            cap >= 2 && n_rows as usize > 2 * cap,
            "fixture must span 3 chunks"
        );

        let a: Vec<u32> = (0..n_rows * lwe_dim)
            .map(|i| i.wrapping_mul(2_654_435_761).wrapping_add(17))
            .collect();
        let hint0: Vec<u32> = (0..lwe_dim * row_width)
            .map(|i| i.wrapping_mul(40_503))
            .collect();
        let row_deltas: Vec<(u32, Vec<(u16, i64)>)> = (0..n_rows)
            .map(|row| {
                (
                    row,
                    vec![
                        (u16::try_from(row % 1000).unwrap(), i64::from(row) + 1),
                        (u16::try_from(row % 1000).unwrap() + 1, -3),
                    ],
                )
            })
            .collect();

        // Oracle: the sparse formula, one touched cell at a time.
        let mut expected = hint0.clone();
        for (row, cells) in &row_deltas {
            for &(off, dlt) in cells {
                for k in 0..lwe_dim as usize {
                    let idx = k * row_width as usize + off as usize;
                    let aik = a[*row as usize * lwe_dim as usize + k];
                    expected[idx] = expected[idx].wrapping_add(aik.wrapping_mul(dlt as u32));
                }
            }
        }

        let mut got = hint0;
        apply_patch_row_level(&a, lwe_dim, n_rows, row_width, &mut got, &row_deltas);
        assert_eq!(got, expected, "chunked row-level patch diverged");
    }

    #[test]
    fn patch_zero_delta_is_noop() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, _mat, mut hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let hint_before = hint.data.clone();

        let row_deltas = vec![(2u32, vec![(1u16, 0i64)])];
        FrodoPirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );

        assert_eq!(hint.data, hint_before);
    }

    #[test]
    fn patch_negative_delta_correct() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, mut hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        // Row 2, off 0: make_db value is 72 (≥ 5), so delta = −5 gives actual change = −5.
        let actual_dlt = apply_cell_delta(&mut db, 2, 0, -5, row_width, pb);
        let row_deltas = vec![(2u32, vec![(0u16, actual_dlt)])];
        FrodoPirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );

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
        let (sp, _mat, mut hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
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
            FrodoPirBackend::server_patch_hint(
                &sp,
                &_mat,
                &mut hint,
                &row_deltas,
                HintPatchMode::EntryLevel,
            );

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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        let target_row = 3u32;
        let dlt1 = apply_cell_delta(&mut db, target_row, 1, 7, row_width, pb);
        let dlt2 = apply_cell_delta(&mut db, target_row, 2, -3, row_width, pb);
        let row_deltas = vec![(target_row, vec![(1u16, dlt1), (2u16, dlt2)])];

        let mut server_hint = hint;
        FrodoPirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut server_hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );
        FrodoPirBackend::client_patch_state(&mut state, &row_deltas, HintPatchMode::EntryLevel);

        assert_eq!(
            state.hint.data, server_hint.data,
            "server and client patched hints must be identical"
        );

        let q = FrodoPirBackend::client_query(&mut state, target_row);
        let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
        let decoded = FrodoPirBackend::client_decode(&state, &r);
        let expected = db[target_row as usize * row_width as usize
            ..(target_row as usize + 1) * row_width as usize]
            .to_vec();
        assert_eq!(
            decoded, expected,
            "decode after patch did not recover patched row"
        );
    }

    #[test]
    fn decode_after_multi_row_patch_returns_patched_value() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 8u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        let raw: &[(u32, &[(u16, i64)])] = &[
            (2, &[(0, 5), (3, -2)]),
            (7, &[(1, 3)]),
            (11, &[(2, -4), (6, 1)]),
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

        let mut server_hint = hint;
        FrodoPirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut server_hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );
        FrodoPirBackend::client_patch_state(&mut state, &row_deltas, HintPatchMode::EntryLevel);

        assert_eq!(
            state.hint.data, server_hint.data,
            "server and client patched hints must be identical"
        );

        for &(target_row, _) in raw {
            let q = FrodoPirBackend::client_query(&mut state, target_row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            let expected = db[target_row as usize * row_width as usize
                ..(target_row as usize + 1) * row_width as usize]
                .to_vec();
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
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
            let expected = db
                [row as usize * row_width as usize..(row as usize + 1) * row_width as usize]
                .to_vec();
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

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
                b: slot.b.clone(),
                c: None,
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
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
                    .map(|&(off, dlt)| {
                        (off, apply_cell_delta(&mut db, row, off, dlt, row_width, pb))
                    })
                    .collect();
                (row, actual)
            })
            .collect();

        // Snapshot the secrets so we can recompute c against the patched hint.
        let secrets: Vec<Vec<u32>> = state.prepared.iter().map(|s| s.secret.clone()).collect();
        FrodoPirBackend::client_patch_state(&mut state, &row_deltas, HintPatchMode::EntryLevel);

        // Oracle: compute c = sᵀ · patched_hint directly from H.
        let lwe_dim = sp.params.lwe_dim;
        let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);
        let mut server_hint = hint;

        FrodoPirBackend::client_precompute_queries(&mut state, 4);
        FrodoPirBackend::client_precompute_decodes(&mut state);

        let raw: &[(u32, &[(u16, i64)])] = &[(2, &[(0, 5), (3, -2)]), (11, &[(2, -4), (6, 1)])];
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

        FrodoPirBackend::server_patch_hint(
            &sp,
            &_mat,
            &mut server_hint,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );
        FrodoPirBackend::client_patch_state(&mut state, &row_deltas, HintPatchMode::EntryLevel);

        for &(target_row, _) in raw {
            let q = FrodoPirBackend::client_query(&mut state, target_row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            let expected = db[target_row as usize * row_width as usize
                ..(target_row as usize + 1) * row_width as usize]
                .to_vec();
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
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
        let (sp, _mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);

        let q = FrodoPirBackend::client_query(&mut state, 1);
        assert!(
            state.in_flight.as_ref().unwrap().c.is_none(),
            "fresh inline-sampled slot starts with no c"
        );

        FrodoPirBackend::client_precompute_decodes(&mut state);
        assert!(
            state.in_flight.as_ref().unwrap().c.is_some(),
            "precompute_decodes must fill in-flight slot too"
        );

        let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
        let decoded = FrodoPirBackend::client_decode(&state, &r);
        let expected = db[row_width as usize..2 * row_width as usize].to_vec();
        assert_eq!(decoded, expected);
    }

    // -------- row-level vs entry-level patch realizations --------

    /// The two [`HintPatchMode`] realizations must produce bit-identical
    /// hints across random multi-row bursts, and both must match the
    /// recomputed `Aᵀ·D` oracle at the end of the run.
    #[test]
    fn row_level_patch_matches_entry_level_and_oracle() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let pb = 8u32;
        let n_rows = 32u32;
        let row_width = 8u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, hint0) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
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
            FrodoPirBackend::server_patch_hint(
                &sp,
                &mat,
                &mut hint_entry,
                &row_deltas,
                HintPatchMode::EntryLevel,
            );
            FrodoPirBackend::server_patch_hint(
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

        let lwe_dim = sp.params.lwe_dim;
        let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
        let expected = compute_hint(&a, &db, n_rows, lwe_dim, row_width);
        assert_eq!(hint_row.data, expected, "row-level diverged from oracle");
    }

    /// A zero delta is a no-op under the row-level realization too (the
    /// dense pass multiplies a zero delta vector into the hint).
    #[test]
    fn row_level_patch_zero_delta_is_noop() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let db = make_db(n_rows, row_width, pb);
        let (sp, mat, mut hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let hint_before = hint.data.clone();

        let row_deltas = vec![(2u32, vec![(1u16, 0i64)])];
        FrodoPirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint,
            &row_deltas,
            HintPatchMode::RowLevel,
        );

        assert_eq!(hint.data, hint_before);
    }

    /// Duplicate edits to the same `(row, cell)` within one call must
    /// accumulate identically under both realizations: entry-level applies
    /// them one by one, row-level sums them while densifying.
    #[test]
    fn duplicate_cell_edits_accumulate_identically() {
        let pb = 8u32;
        let n_rows = 4u32;
        let row_width = 4u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, hint0) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);

        let d1 = apply_cell_delta(&mut db, 1, 2, 3, row_width, pb);
        let d2 = apply_cell_delta(&mut db, 1, 2, 2, row_width, pb);
        let row_deltas = vec![(1u32, vec![(2u16, d1), (2u16, d2)])];

        let mut hint_entry = hint0.clone();
        let mut hint_row = hint0;
        FrodoPirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint_entry,
            &row_deltas,
            HintPatchMode::EntryLevel,
        );
        FrodoPirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut hint_row,
            &row_deltas,
            HintPatchMode::RowLevel,
        );
        assert_eq!(hint_entry.data, hint_row.data);

        let lwe_dim = sp.params.lwe_dim;
        let a = sample_a(&sp.params.seed, n_rows, lwe_dim);
        let expected = compute_hint(&a, &db, n_rows, lwe_dim, row_width);
        assert_eq!(hint_row.data, expected);
    }

    /// Full round-trip under `RowLevel`: precompute Phase B + C, patch via
    /// `client_patch_state(.., RowLevel)`, then decode — exercises the
    /// mode-independent `patch_slot_c` interplay and recovers the patched
    /// plaintext exactly.
    #[test]
    fn decode_after_row_level_patch_with_precomputed_c() {
        let pb = 8u32;
        let n_rows = 16u32;
        let row_width = 8u32;
        let mut db = make_db(n_rows, row_width, pb);
        let (sp, mat, hint) =
            FrodoPirBackend::server_setup(&FrodoConfig::default(), &db, n_rows, row_width, pb);
        let mut state = FrodoPirBackend::client_setup(&sp, &hint);
        let mut server_hint = hint;

        FrodoPirBackend::client_precompute_queries(&mut state, 4);
        FrodoPirBackend::client_precompute_decodes(&mut state);

        let raw: &[(u32, &[(u16, i64)])] = &[(2, &[(0, 5), (3, -2)]), (11, &[(2, -4), (6, 1)])];
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

        FrodoPirBackend::server_patch_hint(
            &sp,
            &mat,
            &mut server_hint,
            &row_deltas,
            HintPatchMode::RowLevel,
        );
        FrodoPirBackend::client_patch_state(&mut state, &row_deltas, HintPatchMode::RowLevel);
        assert_eq!(
            state.hint.data, server_hint.data,
            "server and client row-level patched hints must be identical"
        );

        for &(target_row, _) in raw {
            let q = FrodoPirBackend::client_query(&mut state, target_row);
            let r = FrodoPirBackend::server_answer(&sp, &db, n_rows, row_width, &q);
            let decoded = FrodoPirBackend::client_decode(&state, &r);
            let expected = db[target_row as usize * row_width as usize
                ..(target_row as usize + 1) * row_width as usize]
                .to_vec();
            assert_eq!(decoded, expected, "decode failed for row {target_row}");
        }
    }

    // ── Noise-margin probes (paper-scale, `#[ignore]`d) ─────────────────
    //
    // Empirical counterpart of `crate::pir_params`: measure the decode
    // noise `e·D` at real per-segment row counts with the exact wrapping
    // arithmetic of `server_answer` and the real ternary sampler,
    // without paying for the (noise-irrelevant) hint. Decode of a cell
    // is wrong iff the centered noise coordinate reaches `Δ/2`.
    //
    // Run with:  cargo test -p ikpir-common --release -- --ignored noise_margin

    /// Simulate `draws` independent decodes over a `probe_cols`-cell
    /// window: each draw samples a fresh ternary error vector (as
    /// `client_query` would) and checks every window cell against
    /// `Δ/2`.
    ///
    /// The dominant noise component `(p/2)·Σᵣ eᵣ` is shared by every
    /// cell of one response, so decode failures cluster per query —
    /// many draws (not many columns) is what gives the probe
    /// statistical power, and a window smaller than a real decode's
    /// `row_width` only *under*-counts failures. Cells are uniform in
    /// `[0, 2^pb)` — the distribution `pack_slot_cells` produces for
    /// random fingerprints and values. Returns
    /// `(failed_draws, overflowed_cells, max |noise| / (Δ/2))`.
    fn measure_decode_noise(
        segment_rows: u32,
        pb: u32,
        probe_cols: usize,
        draws: usize,
    ) -> (u64, u64, f64) {
        use rand::SeedableRng;
        let rows = segment_rows as usize;
        let mask = (1u32 << pb) - 1;
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0xA5; 32]);
        let mut d = vec![0u32; rows * probe_cols];
        for cell in d.iter_mut() {
            *cell = rng.next_u32() & mask;
        }
        let delta_half = 1i64 << (32 - pb - 1);
        let mut e = vec![0u32; rows];
        let mut acc = vec![0u32; probe_cols];
        let (mut failed_draws, mut overflowed_cells, mut max_abs) = (0u64, 0u64, 0i64);
        for _ in 0..draws {
            sample_ternary_into(&mut rng, &mut e);
            acc.fill(0);
            for (row, &ei) in e.iter().enumerate() {
                // Skip the ~1/3 of rows with e_i = 0 — the probe measures
                // noise, not timing, so the shortcut is safe here.
                if ei == 0 {
                    continue;
                }
                let off = row * probe_cols;
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

    /// The operating points `pir_params::frodo_max_plaintext_bits`
    /// selects — including the exact Eq. 8 equality boundary
    /// `(s, pb) = (2^18, 10)` where `8·p²·√m = 2^32` — keep the measured
    /// noise strictly inside `Δ/2`.
    #[test]
    #[ignore = "paper-scale probe (~5 s in release; run with --release)"]
    fn noise_margin_validates_selected_operating_points() {
        let draws = 32;
        for s in [1u32 << 18, 1 << 19] {
            let pb = crate::pir_params::frodo_max_plaintext_bits(s);
            let (failed, _, ratio) = measure_decode_noise(s, pb, 256, draws);
            println!(
                "frodo pb={pb} @ s={s}: failed decodes {failed}/{draws}, \
                 max|noise|/(Δ/2) = {ratio:.3}"
            );
            assert_eq!(failed, 0, "selected pb={pb} must not fail decodes");
            assert!(ratio < 1.0);
        }
    }

    /// One plaintext bit past the Eq. 8 boundary (`pb = 11` at
    /// `s = 2^18`) overflows `Δ/2` on ordinary random data — the
    /// equation is tight in practice, not just a safety margin.
    #[test]
    #[ignore = "paper-scale probe (~3 s in release; run with --release)"]
    fn noise_margin_rejects_one_bit_past_boundary() {
        let draws = 64;
        let (failed, cells, ratio) = measure_decode_noise(1 << 18, 11, 256, draws);
        println!(
            "frodo pb=11 @ s=2^18: failed decodes {failed}/{draws} ({cells} cells), \
             max|noise|/(Δ/2) = {ratio:.3}"
        );
        assert!(
            failed > 0,
            "expected failed decodes one bit past the Eq. 8 boundary (got none)"
        );
    }
}
