//! Pluggable Index-PIR backends for [`IkpirServer`](crate::IkpirServer).
//!
//! [`IndexPirBackend`] is the minimum contract a backend must satisfy
//! ([`server_setup`] / [`client_setup`] / [`client_query`] / [`server_answer`] /
//! [`client_decode`]). [`IncrementalPirBackend`] adds two extension methods
//! ([`server_patch_hint`] / [`client_patch_state`]) that let the server keep
//! its preprocessing matrix in sync with the database without a full
//! recompute. [`PrecomputingPirBackend`] adds two further extension methods
//! ([`client_precompute_queries`] / [`client_precompute_decodes`]) that let
//! the client amortise per-query LWE work across a batch of upcoming queries
//! (FrodoPIR Fig. 1 amortisation).
//!
//! [`server_setup`]: IndexPirBackend::server_setup
//! [`client_setup`]: IndexPirBackend::client_setup
//! [`client_query`]: IndexPirBackend::client_query
//! [`server_answer`]: IndexPirBackend::server_answer
//! [`client_decode`]: IndexPirBackend::client_decode
//! [`server_patch_hint`]: IncrementalPirBackend::server_patch_hint
//! [`client_patch_state`]: IncrementalPirBackend::client_patch_state
//! [`client_precompute_queries`]: PrecomputingPirBackend::client_precompute_queries
//! [`client_precompute_decodes`]: PrecomputingPirBackend::client_precompute_decodes
//!
//! # Implementing a new backend
//!
//! 1. Define a [`Config`](IndexPirBackend::Config) struct holding the backend's
//!    tunable knobs (e.g. LWE dimension for FrodoPIR). It must be `Clone +
//!    Default` so callers can either pass an explicit config or fall back to
//!    sensible defaults.
//! 2. `server_setup` returns a `(ServerParams, Hint)` from the DB slice and
//!    the backend `Config` — same `(config, db)` → same hint, modulo any
//!    internal random seed.
//! 3. `client_setup` materialises a `ClientState` that captures whatever the
//!    decode step needs (e.g. the LWE secret + a copy of `Hint`).
//! 4. The triple `(client_query, server_answer, client_decode)` must satisfy
//!    `client_decode(server_answer(client_query(state, row))) == db[row*row_width..(row+1)*row_width]`
//!    for every valid `row`.
//! 5. If implementing [`IncrementalPirBackend`]: `server_patch_hint` and
//!    `client_patch_state` must keep `Hint` and `ClientState` consistent
//!    with the updated DB for **all** future queries.

pub mod frodo;
pub use frodo::{FrodoConfig, FrodoPirBackend};

/// A single-server Index-PIR backend, expressed as five operations over
/// associated types.
///
/// All cells are `u32`. The PIR-level plaintext modulus `p = 2^plaintext_bits`
/// is encoded as the convention "every cell's high `(32 - plaintext_bits)`
/// bits are zero". Backends are free to choose their own ciphertext
/// representation (LWE, Paillier, Regev) as long as the contract above is
/// honoured.
pub trait IndexPirBackend {
    /// Backend-specific tunable knobs supplied at server construction time
    /// (e.g. LWE dimension, error distribution width). Persisted by
    /// [`IkpirServer`](crate::IkpirServer) and re-used by `full_rebuild` so
    /// every regenerated `Hint` has the same dimensions as the originals.
    type Config: Clone + Default;
    /// Public preprocessing parameters produced at setup time
    /// (e.g. an LWE matrix `A`, plaintext modulus, dimensions).
    type ServerParams: Clone;
    /// Server-held preprocessing matrix; for FrodoPIR this is `H = Aᵀ · D`.
    type Hint: Clone;
    /// Client-held state derived from `(ServerParams, Hint)` and used to
    /// build queries and decode responses.
    type ClientState;
    /// Wire-level PIR query, one per segment.
    type Query;
    /// Wire-level PIR response, one per segment.
    type Response;

    /// Server-side preprocessing on a single segment's cell array.
    ///
    /// `db.len()` must equal `n_rows × row_width`. Implementations may sample
    /// fresh randomness on each call.
    fn server_setup(
        config: &Self::Config,
        db: &[u32],
        n_rows: u32,
        row_width: u32,
        plaintext_bits: u32,
    ) -> (Self::ServerParams, Self::Hint);

    /// Client-side preprocessing for one segment.
    fn client_setup(params: &Self::ServerParams, hint: &Self::Hint) -> Self::ClientState;

    /// Build the query that retrieves row `row` from the server's segment.
    /// Mutates `state` to record the secret randomness needed by the
    /// matching `client_decode`.
    fn client_query(state: &mut Self::ClientState, row: u32) -> Self::Query;

    /// Compute the response for `query` over the segment cell array `db`.
    fn server_answer(
        params: &Self::ServerParams,
        db: &[u32],
        n_rows: u32,
        row_width: u32,
        query: &Self::Query,
    ) -> Self::Response;

    /// Decode a response into the cell row that the matching `client_query`
    /// targeted. Output length is `row_width` cells.
    fn client_decode(state: &Self::ClientState, response: &Self::Response) -> Vec<u32>;
}

/// Extension of [`IndexPirBackend`] for backends that support sparse
/// hint updates without a full recompute.
///
/// `row_deltas` is a list of `(row_index, sparse cell edits)` where each
/// edit is `(cell_offset_within_row, signed_delta_mod_2^plaintext_bits)`.
/// Both methods must apply the *same* mathematical change to their
/// respective state, so the client and server stay in lock-step after the
/// patch.
pub trait IncrementalPirBackend: IndexPirBackend {
    /// Apply `row_deltas` to the server-held [`Hint`](IndexPirBackend::Hint).
    fn server_patch_hint(
        params: &Self::ServerParams,
        hint: &mut Self::Hint,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
    );

    /// Apply `row_deltas` to the client-held
    /// [`ClientState`](IndexPirBackend::ClientState).
    ///
    /// **Contract:** every piece of decode-side state derived from the hint
    /// (whether the hint copy itself, or further reductions like the
    /// precomputed `c = sᵀ·H` held by [`PrecomputingPirBackend`])
    /// must be updated so that all future
    /// [`client_query`](IndexPirBackend::client_query) /
    /// [`client_decode`](IndexPirBackend::client_decode) calls produce the
    /// same outputs they would have produced from a freshly issued
    /// [`client_setup`](IndexPirBackend::client_setup) on the post-patch
    /// hint. (Queue contents preserved by precomputation are *not*
    /// considered part of the equivalence — a fresh setup always yields an
    /// empty queue.)
    fn client_patch_state(
        state: &mut Self::ClientState,
        params: &Self::ServerParams,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
    );
}

/// Extension of [`IndexPirBackend`] for backends that can report the wire
/// (byte) size of their associated types.
///
/// Used by the wire-bundle `wire_byte_size()` helpers in [`crate::wire`]
/// and by benches that compare incremental hint deltas against full-hint
/// bytes. A backend that does not implement this trait can still be used
/// with [`IkpirServer`](crate::IkpirServer); wire-size accounting is
/// simply unavailable for that backend.
///
/// **Encoding convention.** Sizes returned here are *minimum* on-wire
/// representations assuming fixed-width little-endian encoding of every
/// scalar — `u32` is 4 bytes, `u64` is 8 bytes, `u16` is 2 bytes, `i64`
/// is 8 bytes. Sizes do not include framing, length prefixes, or
/// compression. A real serialiser will be at least this size.
pub trait BackendWireSize: IndexPirBackend {
    /// Encoded size of one [`Query`](IndexPirBackend::Query) in bytes.
    fn query_byte_size(q: &Self::Query) -> usize;
    /// Encoded size of one [`Response`](IndexPirBackend::Response) in bytes.
    fn response_byte_size(r: &Self::Response) -> usize;
    /// Encoded size of one [`Hint`](IndexPirBackend::Hint) in bytes.
    fn hint_byte_size(h: &Self::Hint) -> usize;
    /// Encoded size of one [`ServerParams`](IndexPirBackend::ServerParams) in bytes.
    fn server_params_byte_size(p: &Self::ServerParams) -> usize;
}

/// Extension of [`IndexPirBackend`] for backends that benefit from
/// pre-sampling per-query randomness in batches (FrodoPIR Fig. 1 style).
///
/// The amortisation has two phases that may be measured independently:
///
/// - **Phase B (query side):** sample the per-query secrets and compute
///   the public-matrix half of the query (`b = A·s + e` for FrodoPIR).
///   This material is independent of the database, so it remains valid
///   across any number of database mutations.
/// - **Phase C (decode side):** compute the hint-dependent decode material
///   (`c = sᵀ·H` for FrodoPIR). This depends on the current hint and must
///   be patched in lock-step with each [`IncrementalPirBackend::client_patch_state`]
///   call (see that trait's contract).
///
/// Once both phases have been run for a slot, the corresponding
/// [`client_query`](IndexPirBackend::client_query) consumes the prepared
/// `b` (cheap vector add) and the matching
/// [`client_decode`](IndexPirBackend::client_decode) consumes the prepared
/// `c` (cheap vector subtract + rounding). Slots are consumed in FIFO
/// order, segment-locally; out-of-order responses produce undefined
/// plaintext.
///
/// Backends that have no useful preprocessing simply do not implement this
/// trait. The base [`IndexPirBackend`] surface stays minimal for them.
pub trait PrecomputingPirBackend: IndexPirBackend {
    /// Phase B — append `count` slots of pre-sampled query material
    /// (`(secret_i, b_i)` for FrodoPIR, no decode material yet) to the
    /// state's internal queue. Subsequent
    /// [`client_query`](IndexPirBackend::client_query) calls consume one
    /// slot each before falling back to inline sampling.
    fn client_precompute_queries(state: &mut Self::ClientState, count: u32);

    /// Phase C — for every slot in the state's queue that does not yet
    /// carry decode material, compute and attach it (`c_i = secret_iᵀ·H`
    /// for FrodoPIR). Idempotent: slots that already have decode material
    /// are skipped. Includes already-in-flight slots so a caller who issued
    /// queries before precomputing decodes still gets the cheap decode
    /// path.
    fn client_precompute_decodes(state: &mut Self::ClientState);

    /// Number of slots currently prepared but not yet consumed by a
    /// [`client_query`](IndexPirBackend::client_query). Surface for
    /// benchmarks and tests; consumers should treat as observational.
    fn prepared_slot_count(state: &Self::ClientState) -> usize;

    /// Number of slots currently in flight (consumed by `client_query`,
    /// awaiting `client_decode`).
    fn in_flight_slot_count(state: &Self::ClientState) -> usize;
}
