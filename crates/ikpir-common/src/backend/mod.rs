//! Pluggable Index-PIR backends for `IkpirServer`.
//!
//! # Purpose
//!
//! Defines the trait contract every PIR backend must satisfy and the two
//! optional extensions (incremental hint, precomputation) that
//! `IkpirServer` and `IkpirClient` opt into when available.
//!
//! # Design / architecture
//!
//! Three layered traits, from minimum to richest:
//!
//! - [`IndexPirBackend`] is the minimum contract
//!   ([`server_setup`] / [`client_setup`] / [`client_query`] /
//!   [`server_answer`] / [`client_decode`]).
//! - [`IncrementalPirBackend`] adds two extension methods
//!   ([`server_patch_hint`] / [`client_patch_state`]) that let the
//!   server keep its preprocessing matrix in sync with the database
//!   without a full recompute. Both take a [`HintPatchMode`] selecting
//!   the row-level or entry-level realization of the patch.
//! - [`PrecomputingPirBackend`] adds two further extension methods
//!   ([`client_precompute_queries`] / [`client_precompute_decodes`])
//!   that let the client amortise per-query LWE work across a batch of
//!   upcoming queries (FrodoPIR Fig. 1 amortisation).
//! - [`ParallelSetupBackend`] adds a second, multi-threaded
//!   *realization* of the three setup entry points, producing
//!   bit-identical output. It exists purely so that a benchmark which
//!   does not measure setup can stop paying for it single-threaded.
//!
//! [`BackendWireSize`] is an orthogonal helper for backends that can
//! report byte-level sizes; benches use it to compare wire footprints
//! between incremental deltas and full hints.
//!
//! # Related files
//!
//! - `frodo/`, `simple/` — the two shipped backend implementations.
//! - `ikpir-server::IkpirServer` / `ikpir-client::IkpirClient` — the
//!   two protocol-level consumers of `IndexPirBackend` /
//!   `IncrementalPirBackend`.
//! - `wire.rs` in `ikpir-common` — `wire_byte_size` helpers consume `BackendWireSize`.
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
//! 2. Define [`HintMaterial`](IndexPirBackend::HintMaterial) — server-local
//!    working state derived deterministically from `ServerParams` (e.g. the
//!    LWE public matrix `A` expanded from a 16-byte seed). Not part of the
//!    wire payload. Backends with no analogous material set
//!    `type HintMaterial = ()`.
//! 3. `server_setup` returns `(ServerParams, HintMaterial, Hint)` from the DB
//!    slice and the backend `Config` — same `(config, db)` → same hint,
//!    modulo any internal random seed.
//! 4. `expand_hint_material(params)` re-derives the `HintMaterial` from
//!    `ServerParams` whenever the server has dropped its copy. Must be
//!    bit-identically deterministic in any seed/state inside `params`.
//! 5. `client_setup` materialises a `ClientState` that captures whatever the
//!    decode step needs (e.g. the LWE secret + a copy of `Hint` + a freshly
//!    expanded `HintMaterial`).
//! 6. The triple `(client_query, server_answer, client_decode)` must satisfy
//!    `client_decode(server_answer(client_query(state, row))) == db[row*row_width..(row+1)*row_width]`
//!    for every valid `row`.
//! 7. If implementing [`IncrementalPirBackend`]: `server_patch_hint` and
//!    `client_patch_state` must keep `Hint` and `ClientState` consistent
//!    with the updated DB for **all** future queries. `server_patch_hint`
//!    takes a `&HintMaterial` so the server can replay seed-derived state
//!    without re-cloning it.

pub mod frodo;
pub(crate) mod matvec;
pub mod parallel;
pub(crate) mod patch;
pub mod simple;
pub use frodo::{FrodoConfig, FrodoPirBackend};
pub use simple::{SimpleConfig, SimplePirBackend};

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
    /// `IkpirServer` and re-used by `full_rebuild` so
    /// every regenerated `Hint` has the same dimensions as the originals.
    type Config: Clone + Default;
    /// Public preprocessing parameters produced at setup time. Carries the
    /// wire-shippable description of the segment (shape constants, seeds);
    /// the bulky derived state lives in [`Self::HintMaterial`].
    type ServerParams: Clone;
    /// Server-local working state derived deterministically from
    /// [`Self::ServerParams`] — e.g. the LWE public matrix `A` expanded
    /// from a 16-byte seed. Used by [`Self::server_setup`] to compute the
    /// hint and by [`IncrementalPirBackend::server_patch_hint`] to keep
    /// the hint coherent across mutations. **Not part of the wire payload.**
    /// The server may drop its copy and re-expand it on demand via
    /// [`Self::expand_hint_material`]; backends with no analogous material
    /// set `type HintMaterial = ()`.
    type HintMaterial: Default + Send + 'static;
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
    /// fresh randomness on each call. Returns the wire-shippable
    /// [`Self::ServerParams`], the server-local [`Self::HintMaterial`]
    /// (which the caller may immediately drop if not needed), and the
    /// derived [`Self::Hint`].
    fn server_setup(
        config: &Self::Config,
        db: &[u32],
        n_rows: u32,
        row_width: u32,
        plaintext_bits: u32,
    ) -> (Self::ServerParams, Self::HintMaterial, Self::Hint);

    /// Shape `(rows, cols)` of the matrix the backend actually multiplies
    /// for one segment, read back from `params`.
    ///
    /// A backend may reshape the `(n_rows, row_width)` segment it was
    /// handed into whatever layout its algorithm wants: FrodoPIR keeps the
    /// tall-skinny original, SimplePIR folds it into a near-square matrix.
    /// The reshape is chosen inside [`Self::server_setup`] and recorded in
    /// `params`, so this must *report* those stored dimensions — never
    /// recompute them. Callers that need the real geometry (the benches'
    /// `db_rows` / `db_cols` CSV columns) then cannot drift from the
    /// backend's own choice.
    fn db_matrix_shape(params: &Self::ServerParams) -> (u32, u32);

    /// Re-derive [`Self::HintMaterial`] from [`Self::ServerParams`].
    ///
    /// **Determinism contract.** Implementations must be bit-identically
    /// deterministic in any seed/state inside `params`: the server may
    /// drop and re-expand the result mid-protocol and require the same
    /// matrix back. The client also calls this during [`Self::client_setup`]
    /// to materialise its own copy from the wire-shipped seed. Backends
    /// with no analogous material return `Default::default()`.
    fn expand_hint_material(params: &Self::ServerParams) -> Self::HintMaterial;

    /// Client-side preprocessing for one segment. The implementation
    /// should call [`Self::expand_hint_material`] to materialise its own
    /// copy of the seed-derived state; the wire bundle does not carry it.
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
    ///
    /// # Panics
    ///
    /// May panic if called without a preceding `client_query` on the same
    /// `state`. This is an internal protocol invariant: the public
    /// `IkpirClient` API always queries before decoding, so it is never
    /// reachable from user input.
    fn client_decode(state: &Self::ClientState, response: &Self::Response) -> Vec<u32>;
}

/// Granularity at which an [`IncrementalPirBackend`] realizes a hint
/// patch.
///
/// Both realizations consume the same sparse `(row, [(cell, Δ)])`
/// transcript and leave the hint equal to `A · D` for the post-mutation
/// database — the wire format and the resulting state are identical (all
/// arithmetic is mod `2³²`); only the arithmetic schedule, and therefore
/// the patch cost, differs. Server and client may even run different
/// modes without ever diverging.
///
/// With `n` the LWE dimension and `ω` the width of the (possibly
/// backend-reshaped) database matrix `D`:
///
/// - [`RowLevel`](Self::RowLevel) — the row-granular patch of SimplePIR:
///   for each touched (reshaped) row `r`, densify its edits into a
///   full-width delta vector `δ ∈ Z_q^ω` and apply the dense rank-one
///   update `H[k'][c] += A[r][k'] · δ_c` for all `k' ∈ [n]`, `c ∈ [ω]` —
///   `Θ(n·ω)` per touched row.
/// - [`EntryLevel`](Self::EntryLevel) — the per-cell sharpening of
///   iSimplePIR: for each touched cell `(r, c, γ)`, patch column `c`
///   alone, `H[k'][c] += A[r][k'] · γ` for all `k' ∈ [n]` — `Θ(n)` per
///   cell.
///
/// For a FrodoPIR-shaped (tall) database a mutation rewrites a few cells
/// of one row, so the two modes differ by a constant factor
/// (`ω / touched cells` per row). For a SimplePIR-shaped (near-square)
/// database `ω` grows with the database size, so only the entry-level
/// patch keeps the per-mutation cost independent of the database size.
/// The default is [`EntryLevel`](Self::EntryLevel).
///
/// Realizing that asymptotic advantage as wall-clock takes care with the
/// execution order — `H` is row-major, so patching one column at a time
/// sweeps the whole hint once *per touched column*. Both backends
/// therefore run the entry-level patch through the crate-internal
/// `backend::patch::TouchedRuns`, which inverts the loops and coalesces
/// touched columns into contiguous runs; see its module docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HintPatchMode {
    /// Dense per-row rank-one update, `Θ(n·ω)` per touched row — the
    /// row-level patch of SimplePIR.
    RowLevel,
    /// Sparse per-cell column update, `Θ(n)` per touched cell — the
    /// entry-level patch of iSimplePIR.
    #[default]
    EntryLevel,
}

/// Extension of [`IndexPirBackend`] for backends that support sparse
/// hint updates without a full recompute.
///
/// `row_deltas` is a list of `(row_index, sparse cell edits)` where each
/// edit is `(cell_offset_within_row, signed_delta_mod_2^plaintext_bits)`.
/// Both methods must apply the *same* mathematical change to their
/// respective state, so the client and server stay in lock-step after the
/// patch. `mode` selects the [`HintPatchMode`] realization; every mode
/// must produce the same post-patch state, so mode choices are local and
/// never need to be coordinated across the wire.
pub trait IncrementalPirBackend: IndexPirBackend {
    /// Apply `row_deltas` to the server-held [`Hint`](IndexPirBackend::Hint).
    ///
    /// Takes the server-local
    /// [`HintMaterial`](IndexPirBackend::HintMaterial) by reference so the
    /// patch can read seed-derived state (e.g. the LWE public matrix `A`)
    /// without re-cloning it. Callers must materialise `material` via
    /// [`IndexPirBackend::expand_hint_material`] before this call if the
    /// server has dropped its copy.
    fn server_patch_hint(
        params: &Self::ServerParams,
        material: &Self::HintMaterial,
        hint: &mut Self::Hint,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
        mode: HintPatchMode,
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
    ///
    /// Implementations should read the backend's
    /// [`ServerParams`](IndexPirBackend::ServerParams) and
    /// [`HintMaterial`](IndexPirBackend::HintMaterial) from the
    /// `ClientState` itself; the caller does **not** thread a separate
    /// `params` or `material` argument through, since
    /// [`client_setup`](IndexPirBackend::client_setup) already stashes
    /// both at setup time.
    fn client_patch_state(
        state: &mut Self::ClientState,
        row_deltas: &[(u32, Vec<(u16, i64)>)],
        mode: HintPatchMode,
    );
}

/// Extension of [`IndexPirBackend`] offering a **second realization of
/// the setup phase** that trades single-threadedness for wall-clock
/// time while producing bit-identical output.
///
/// # Why this exists
///
/// Setup is `Θ(n_rows · lwe_dim · row_width)` — orders of magnitude
/// more arithmetic than any online operation, and the dominant cost of
/// getting a server or client into the state a benchmark actually wants
/// to measure. The paper reports setup in the single-threaded,
/// non-SIMD regime, so [`IndexPirBackend::server_setup`] and friends
/// must stay exactly that. Every *other* benchmark, however, only needs
/// setup's **result**; how it was computed is unobservable downstream.
///
/// This trait is that second path. Nothing else about the protocol
/// changes: same `ServerParams`, same `HintMaterial`, same `Hint`, same
/// wire bytes.
///
/// # Equivalence contract
///
/// For every `(config, db, n_rows, row_width, plaintext_bits)` and
/// every `params`:
///
/// - [`expand_hint_material_parallel`](Self::expand_hint_material_parallel)`(p)`
///   must equal [`IndexPirBackend::expand_hint_material`]`(p)` **bit for
///   bit**, and
/// - [`server_setup_parallel`](Self::server_setup_parallel) must return
///   the same `(HintMaterial, Hint)` that
///   [`IndexPirBackend::server_setup`] would have returned **given the
///   same internally sampled seed** — i.e. `hint == Aᵀ·D` for the
///   `A` that the returned `ServerParams`' seed expands to, with `A`
///   itself unchanged, and
/// - [`client_setup_parallel`](Self::client_setup_parallel) must yield a
///   `ClientState` observationally identical to
///   [`IndexPirBackend::client_setup`]'s: same queries, same decodes,
///   same patch behaviour.
///
/// Implementations get this for free by partitioning only the
/// **output** — disjoint bands of the hint, disjoint runs of the
/// keystream — so each output cell still accumulates the same terms in
/// the same order. `u32` wrapping arithmetic is associative and
/// commutative, but relying on that is not even necessary under a
/// disjoint-output partition. See
/// [`backend::parallel`](crate::backend::parallel) for the fan-out
/// primitives and the worker-count knob (`IKPIR_SETUP_THREADS`).
///
/// # Constraints
///
/// The schedule depends only on public shape
/// (`n_rows`, `row_width`, `lwe_dim`) and the machine's core count —
/// never on database contents or on any secret — so the parallel path
/// inherits the base trait's side-channel posture. Implementations must
/// preserve that; do not branch on cell values to skip work.
///
/// # Choosing a path
///
/// | Caller | Use |
/// |---|---|
/// | `benches/server_setup.rs` (the measurement) | [`IndexPirBackend::server_setup`] |
/// | every other bench's preamble | [`Self::server_setup_parallel`] |
/// | library / test / production code | either; they are interchangeable |
///
/// `IkpirServer::new_parallel` / `IkpirServer::full_rebuild_parallel` /
/// `IkpirClient::from_setup_parallel` are the protocol-level entry
/// points that route here.
pub trait ParallelSetupBackend: IndexPirBackend {
    /// Multi-threaded realization of [`IndexPirBackend::server_setup`].
    ///
    /// Returns the same triple, under the equivalence contract in the
    /// trait docs. Fresh randomness (the public seed) is still sampled
    /// per call, exactly as the reference does.
    fn server_setup_parallel(
        config: &Self::Config,
        db: &[u32],
        n_rows: u32,
        row_width: u32,
        plaintext_bits: u32,
    ) -> (Self::ServerParams, Self::HintMaterial, Self::Hint);

    /// Multi-threaded realization of
    /// [`IndexPirBackend::expand_hint_material`].
    ///
    /// **Bit-identical** to the reference for the same `params` — the
    /// determinism contract on the reference method applies here
    /// verbatim, and across the two methods: a server that expanded
    /// with one may re-expand with the other.
    fn expand_hint_material_parallel(params: &Self::ServerParams) -> Self::HintMaterial;

    /// Multi-threaded realization of [`IndexPirBackend::client_setup`].
    ///
    /// The client's cost here is dominated by re-expanding `A` from the
    /// wire-shipped seed, so this is `expand_hint_material_parallel`
    /// plus the same cheap clones the reference performs.
    fn client_setup_parallel(params: &Self::ServerParams, hint: &Self::Hint) -> Self::ClientState;
}

/// Extension of [`IndexPirBackend`] for backends that can report the wire
/// (byte) size of their associated types.
///
/// Used by the wire-bundle `wire_byte_size()` helpers (see
/// `ServerSetupBundle`, `PirQueryBundle`, `PirResponseBundle`,
/// `HintDeltaBundle`) and by benches that compare incremental hint
/// deltas against full-hint bytes. A backend that does not implement
/// this trait can still be used with
/// `IkpirServer`; wire-size accounting is simply
/// unavailable for that backend.
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
