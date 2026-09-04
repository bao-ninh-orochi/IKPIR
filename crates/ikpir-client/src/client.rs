//! `IkpirClient<B>` — the client-side state machine.
//!
//! # Purpose
//!
//! Translates user-level `(key, query)` operations into the per-segment
//! Index-PIR `(Query, Response)` exchanges with the IKPIR server.
//! Maintains the strict-monotone client epoch and the response-rewind `ΔD`
//! accumulator, syncing from the `HintDeltaBundle`s the server emits on
//! every mutation.
//!
//! # Design / architecture
//!
//! - **State.** [`CuckooParams`] + one `B::ServerParams` and
//!   `B::ClientState` per segment + the epoch counter + the rolling
//!   response-rewind `ΔD` (`PendingDelta`) and its pin epoch. Never owns a
//!   `CuckooKVStore` — that's server-side.
//! - **Lookup geometry is public.** `candidate_buckets` is
//!   re-derivable from `params` alone, so the client computes it on
//!   every query without privacy cost.
//! - **Side-channel hardening in `decode`.** Every candidate slot is
//!   unpacked and merged into a fixed-size accumulator via a
//!   branchless OR-masked select; a timing observer learns at most
//!   `(arity, bucket_size)`.
//! - **Response-rewind is the sole update strategy.** The client pins its
//!   bootstrap hint `H₀` and never patches it: `accumulate_delta` rolls the
//!   published `ΔD` forward (`Θ(τ·ω)` per batch — a factor-`n` cheaper
//!   maintenance than patching the hint), and `decode` corrects each
//!   response back to `H₀`'s epoch before decoding. `collect_garbage`
//!   reclaims the staleness-growing per-query correction by folding `ΔD`
//!   into the hint on demand. `reset_from` recovers after a `full_rebuild`
//!   or an unbridgeable `FutureDelta` gap. The classical alternative —
//!   patching the hint immediately — survives only as a benchmark
//!   comparator behind the `hint-patch-bench` Cargo feature
//!   (`crate::bench_comparator::HintPatchClient`); see
//!   `docs/rewind-client-mode.md`.
//!
//! # Related files
//!
//! - `lib.rs` — re-exports `IkpirClient`, `DeltaApplyOutcome`, and the
//!   ikpir-server wire/backend types.
//! - `error.rs` — `IkpirClientError` variants.
//! - `bench_comparator/` — the feature-gated hint-patch comparator client.
//! - `ikpir-server::IkpirServer` — counterpart on the server side.

use ikpir_common::{
    HintDeltaBundle, HintPatchMode, IncrementalPirBackend, IndexPirBackend, ParallelSetupBackend,
    PirQueryBundle, PirResponseBundle, PrecomputingPirBackend, ResponseRewind, ServerSetupBundle,
};
use segmented_cuckoo::{unpack_slot_cells, CuckooParams};

/// The per-segment client-setup primitive, as a function pointer.
///
/// # Purpose
///
/// [`IndexPirBackend::client_setup`] (the single-threaded reference)
/// and [`ParallelSetupBackend::client_setup_parallel`] (the optimized
/// twin) have identical signatures and, by the latter's equivalence
/// contract, produce observationally identical `ClientState`s. Passing
/// one as a value lets [`IkpirClient::from_setup`] and
/// [`IkpirClient::from_setup_parallel`] share a single body.
type PerSegmentClientSetup<B> = fn(
    &<B as IndexPirBackend>::ServerParams,
    &<B as IndexPirBackend>::Hint,
) -> <B as IndexPirBackend>::ClientState;

use crate::error::IkpirClientError;
use crate::pending::PendingDelta;

/// Client-side IKPIR engine, generic over the PIR backend `B`.
///
/// # Purpose
///
/// Holds the per-segment client state, the SCF [`CuckooParams`], and a
/// strict-monotone epoch counter. Does **not** hold any `CuckooKVStore`
/// cells — those live on the server.
///
/// # Rationale
///
/// Stateless w.r.t. the database content (the server is the
/// authoritative store) but stateful w.r.t. the per-segment LWE
/// preprocessing (`B::ClientState`) and the rolling `ΔD`; the latter is
/// what `accumulate_delta` keeps in lock-step with server mutations.
///
/// # Threading
///
/// All methods are synchronous. `build_query` and `accumulate_delta` take
/// `&mut self`. Wrap in `Mutex` if exposing across threads.
pub struct IkpirClient<B: IndexPirBackend> {
    params: CuckooParams,
    states: Vec<B::ClientState>,
    epoch: u64,
    /// Rolling public `ΔD` accumulated since the pinned hint `H₀` — see
    /// [`Self::accumulate_delta`] / [`Self::decode`] / [`Self::collect_garbage`].
    pending: PendingDelta,
    /// Epoch the per-segment states (the pinned hint `H₀`) are at. Trails
    /// [`Self::epoch`] by the accumulated deltas and advances only on
    /// [`Self::collect_garbage`] or a resync ([`Self::reset_from`]).
    pin_epoch: u64,
}

/// Outcome of [`IkpirClient::try_accumulate_delta_or_resync`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaApplyOutcome {
    /// The delta was applied incrementally (the common case).
    Synced,
    /// The delta was too far ahead; the fetched fresh bundle was used to
    /// [`IkpirClient::reset_from`] the client.
    Resynced,
}

impl<B: IndexPirBackend> IkpirClient<B> {
    /// Build a fresh client from a server setup bundle.
    ///
    /// # Purpose
    ///
    /// First-time client bootstrap; the inverse of
    /// `IkpirServer::setup()`. Also used (via [`Self::reset_from`])
    /// to recover from a `FutureDelta` gap.
    ///
    /// # Arguments
    ///
    /// - `bundle` — setup material from the server.
    ///
    /// # Constraints
    ///
    /// `bundle.params.plaintext_bits` (and the rest of the SCF geometry)
    /// is fixed for this client's lifetime; every subsequent
    /// [`Self::accumulate_delta`] or [`Self::reset_from`] call must come from
    /// a server built with the same `plaintext_bits`. See
    /// `IkpirServer::new` for the upgrade path that would be required
    /// to lift this invariant.
    ///
    /// # Returns
    ///
    /// A ready-to-query client at `bundle.epoch`.
    ///
    /// # Complexity
    ///
    /// `O(arity)` calls to
    /// [`B::client_setup`](IndexPirBackend::client_setup), each of
    /// which clones one `ServerParams` and one `Hint` — and, for both
    /// shipped LWE backends, re-expands that segment's public matrix
    /// `A` from its seed. That expansion is the whole cost and it runs
    /// **single-threaded** here; see [`Self::from_setup_parallel`] for
    /// the optimized twin.
    pub fn from_setup(bundle: ServerSetupBundle<B>) -> Self {
        Self::assemble(bundle, B::client_setup)
    }

    /// Shared body of [`Self::from_setup`] and
    /// [`Self::from_setup_parallel`].
    fn assemble(bundle: ServerSetupBundle<B>, client_setup: PerSegmentClientSetup<B>) -> Self {
        let arity = bundle.params.arity();
        debug_assert_eq!(bundle.backend_params.len(), arity);
        debug_assert_eq!(bundle.hints.len(), arity);

        // `client_setup` stashes a clone of each `ServerParams` inside the
        // returned `ClientState`, so the client does not need to retain
        // its own `Vec<B::ServerParams>` — the per-state copy is the
        // sole source of truth on this side of the wire.
        let states: Vec<B::ClientState> = bundle
            .backend_params
            .iter()
            .zip(bundle.hints.iter())
            .map(|(p, h)| client_setup(p, h))
            .collect();

        Self {
            params: bundle.params,
            states,
            epoch: bundle.epoch,
            pending: PendingDelta::new(arity),
            pin_epoch: bundle.epoch,
        }
    }

    /// Replace all internal state from a fresh setup bundle.
    ///
    /// # Purpose
    ///
    /// Resync entry point. Use after a server-side `full_rebuild`, or
    /// whenever [`IkpirClient::accumulate_delta`] reports
    /// [`IkpirClientError::FutureDelta`] (the gap is unbridgeable
    /// incrementally).
    ///
    /// # Arguments
    ///
    /// - `bundle` — fresh setup material from the server, at any
    ///   epoch ≥ the current client epoch.
    ///
    /// # Rationale
    ///
    /// Equivalent to `*self = Self::from_setup(bundle)` — implemented
    /// as a swap so existing callers don't need to take ownership. Clears
    /// any accumulated `ΔD` and re-pins `H₀` at the fresh bundle's epoch.
    pub fn reset_from(&mut self, bundle: ServerSetupBundle<B>) {
        *self = Self::from_setup(bundle);
    }

    /// Current client epoch. Strictly monotone across `accumulate_delta`
    /// and `reset_from`.
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// SCF geometry parameters in use.
    pub const fn params(&self) -> CuckooParams {
        self.params
    }

    /// Epoch the pinned hint `H₀` is at. Equals [`Self::epoch`] on a freshly
    /// bootstrapped or just-garbage-collected client; otherwise trails by the
    /// accumulated `ΔD` span and advances only on [`Self::collect_garbage`] /
    /// [`Self::reset_from`].
    pub const fn pin_epoch(&self) -> u64 {
        self.pin_epoch
    }

    /// Number of nonzero `ΔD` cells currently accumulated — the staleness
    /// measure the per-query correction cost in [`Self::decode`] scales with.
    pub fn pending_cells(&self) -> usize {
        self.pending.cells()
    }
}

/// Optimized (multi-threaded) bootstrap, available for every backend
/// implementing [`ParallelSetupBackend`] — both shipped LWE backends
/// do.
///
/// # Purpose
///
/// Bootstrapping a client re-expands the public matrix `A` from each
/// segment's seed — `Θ(arity · n_rows · lwe_dim)` ChaCha20 words, which
/// is the entire cost of [`IkpirClient::from_setup`] and reaches
/// gigabytes at paper scale. These twins produce the identical client
/// across all cores.
///
/// # Constraints
///
/// The resulting client is **observationally identical** to one built
/// by the reference path: same queries, same decodes, same patch
/// behaviour, same epoch (see the equivalence contract on
/// [`ParallelSetupBackend`]). Only the wall-clock differs, so any
/// bench that does not itself report client-bootstrap cost should
/// prefer these. Worker count follows `IKPIR_SETUP_THREADS`, else the
/// machine's available parallelism.
impl<B: ParallelSetupBackend> IkpirClient<B> {
    /// Multi-threaded twin of [`Self::from_setup`] — identical
    /// resulting client, computed across cores.
    pub fn from_setup_parallel(bundle: ServerSetupBundle<B>) -> Self {
        Self::assemble(bundle, B::client_setup_parallel)
    }

    /// Multi-threaded twin of [`Self::reset_from`]: clears any accumulated
    /// `ΔD` and re-pins `H₀` at the fresh bundle's epoch, computed across
    /// cores.
    pub fn reset_from_parallel(&mut self, bundle: ServerSetupBundle<B>) {
        *self = Self::from_setup_parallel(bundle);
    }
}

impl<B: IndexPirBackend> IkpirClient<B>
where
    B::Query: Clone,
{
    /// Build a per-segment PIR query for `key`.
    ///
    /// # Purpose
    ///
    /// Client hot path. Re-derives the SCF candidate buckets locally
    /// ([`CuckooParams::candidate_buckets`]) and emits one
    /// [`B::Query`](IndexPirBackend::Query) per segment, where the
    /// j-th query targets row `indices[j] % segment_size` in segment
    /// `j`.
    ///
    /// # Arguments
    ///
    /// - `key` — lookup key bytes.
    ///
    /// # Returns
    ///
    /// A [`PirQueryBundle`] carrying the current client epoch. The
    /// server rejects with
    /// [`IkpirError::StaleEpoch`](ikpir_common::IkpirError::StaleEpoch)
    /// if it has moved past this epoch.
    ///
    /// # Complexity
    ///
    /// One xxh3 hash of `key` + `arity` calls to
    /// [`B::client_query`](IndexPirBackend::client_query). Each LWE
    /// query is `O(n_rows · lwe_dim)` matvec on the cold path, `O(n_rows)`
    /// (single vector add) on the precomputed cheap path.
    pub fn build_query(&mut self, key: &[u8]) -> PirQueryBundle<B> {
        let (_fp, indices) = self.params.candidate_buckets(key);
        let arity = self.params.arity();
        let segment_size = self.params.segment_size();

        let queries: Vec<B::Query> = (0..arity)
            .map(|j| B::client_query(&mut self.states[j], indices[j] % segment_size))
            .collect();

        PirQueryBundle {
            epoch: self.epoch,
            queries,
        }
    }
}

/// Branchless `u64` equality mask: returns `u64::MAX` if `a == b`, else `0`.
///
/// Standard constant-time trick: `x ^ b == 0` iff `a == b`; squeeze that
/// zero/non-zero into bit 63 via `x | -x`, shift down, then subtract 1 to
/// flip the meaning.
#[inline]
const fn ct_eq_u64_mask(a: u64, b: u64) -> u64 {
    let x = a ^ b;
    ((x | x.wrapping_neg()) >> 63).wrapping_sub(1)
}

impl<B: PrecomputingPirBackend> IkpirClient<B> {
    /// Phase B amortisation — pre-sample `count` query slots **per segment**.
    ///
    /// Each subsequent [`IkpirClient::build_query`] call consumes one slot
    /// per segment off the prepared queue (cheap path: one vector add)
    /// before falling back to inline LWE sampling. The prepared material is
    /// independent of the database, so it stays valid across mutations.
    ///
    /// Cost per call: `count × arity × n_rows × lwe_dim` matvec work.
    pub fn precompute_queries(&mut self, count: u32) {
        for state in &mut self.states {
            B::client_precompute_queries(state, count);
        }
    }

    /// Phase C amortisation — fill in the decode-side material `c = sᵀ·H`
    /// for every prepared (and currently in-flight) slot per segment that
    /// does not already have it. Idempotent.
    ///
    /// After this call, every matching [`IkpirClient::decode`] takes the
    /// cheap path (one vector subtract + rounding) against the pinned hint.
    /// [`IkpirClient::collect_garbage`] keeps the precomputed `c` values
    /// consistent with the hint it folds `ΔD` into.
    ///
    /// Cost per call: `prepared_count × arity × lwe_dim × row_width` matvec work.
    pub fn precompute_decodes(&mut self) {
        for state in &mut self.states {
            B::client_precompute_decodes(state);
        }
    }

    /// Per-segment count of prepared-but-unconsumed query slots.
    /// `vec[j]` is segment `j`'s queue length.
    pub fn prepared_per_segment(&self) -> Vec<usize> {
        self.states.iter().map(B::prepared_slot_count).collect()
    }

    /// Per-segment count of in-flight queries (each entry is 0 or 1; the
    /// FrodoPIR contract is "at most one in-flight query per segment").
    pub fn in_flight_per_segment(&self) -> Vec<usize> {
        self.states.iter().map(B::in_flight_slot_count).collect()
    }
}

impl<B: IncrementalPirBackend> IkpirClient<B> {
    /// Accumulate a hint-delta bundle into the rolling `ΔD`.
    ///
    /// # Purpose
    ///
    /// Steady-state synchronisation entry point: instead of patching the
    /// hint, folds `delta` into the client's running `ΔD = D_head − D₀`
    /// (per-cell sum, dropping cells that net to zero), leaving the pinned
    /// hint `H₀` untouched. The tracked head epoch advances; the pin does
    /// not.
    ///
    /// # Arguments
    ///
    /// - `delta` — bundle emitted by the server's
    ///   `insert` / `update` / `delete`.
    ///
    /// # Constraints
    ///
    /// **Strict-monotone:** the only accepted shape is
    /// `delta.epoch == self.epoch + 1`. Older deltas are
    /// [`IkpirClientError::StaleDelta`]; gaps are
    /// [`IkpirClientError::FutureDelta`] — the caller must recover by
    /// calling [`IkpirClient::reset_from`] with a fresh server bundle.
    /// `delta.params` must equal the client's cached `params`, and
    /// `delta.per_segment_row_deltas.len()` must equal `params.arity()`.
    ///
    /// # Returns
    ///
    /// - `Ok(())` and `self.epoch += 1` on success.
    /// - `Err(IkpirClientError::StaleDelta { expected, got })` when
    ///   `delta.epoch ≤ self.epoch`.
    /// - `Err(IkpirClientError::FutureDelta { expected, got })` when
    ///   `delta.epoch > self.epoch + 1`.
    /// - `Err(IkpirClientError::MalformedBundle)` if `delta.params` differs
    ///   from the client's `params`, `per_segment_row_deltas.len()` doesn't
    ///   match arity, or (never reachable with real cell deltas) an `i64`
    ///   overflow while summing.
    ///
    /// # Complexity
    ///
    /// `O(Σ |touched cells|)` `BTreeMap` work per segment — independent of
    /// the LWE dimension `n`, the factor-`n` maintenance saving over
    /// patching the hint directly (see the bench-only
    /// `bench_comparator::HintPatchClient` behind the `hint-patch-bench`
    /// feature).
    pub fn accumulate_delta(&mut self, delta: HintDeltaBundle<B>) -> Result<(), IkpirClientError> {
        let expected = self.epoch + 1;
        if delta.epoch < expected {
            return Err(IkpirClientError::StaleDelta {
                expected,
                got: delta.epoch,
            });
        }
        if delta.epoch > expected {
            return Err(IkpirClientError::FutureDelta {
                expected,
                got: delta.epoch,
            });
        }
        if delta.params != self.params {
            return Err(IkpirClientError::MalformedBundle);
        }
        if delta.per_segment_row_deltas.len() != self.params.arity() {
            return Err(IkpirClientError::MalformedBundle);
        }
        // Never false with real cell deltas (the running sum stays in (−p, p)).
        if !self.pending.merge(&delta.per_segment_row_deltas) {
            return Err(IkpirClientError::MalformedBundle);
        }
        self.epoch = delta.epoch;
        Ok(())
    }

    /// Fold the accumulated `ΔD` into the pinned hint and re-pin at the
    /// head — the client's garbage collection.
    ///
    /// # Purpose
    ///
    /// Reclaims the per-query rewind correction: patches each segment's hint by
    /// the whole accumulated `ΔD` via
    /// [`client_patch_state`](IncrementalPirBackend::client_patch_state)
    /// (entry-level), advancing the pinned hint `H₀` to the current head and
    /// clearing `ΔD`. **Never required for correctness** — a rewind client stays
    /// correct indefinitely without it (the correction merely grows with
    /// staleness); GC trades a one-off `Θ(|ΔD|·n)` patch for a cheaper
    /// steady-state decode. A no-op when nothing is pending.
    ///
    /// # Returns
    ///
    /// `Ok(())`, after folding (or immediately, if nothing was pending).
    pub fn collect_garbage(&mut self) -> Result<(), IkpirClientError> {
        if self.pin_epoch == self.epoch {
            return Ok(()); // nothing pending
        }
        for (j, state) in self.states.iter_mut().enumerate() {
            let row_deltas = self.pending.as_row_deltas(j);
            if !row_deltas.is_empty() {
                B::client_patch_state(state, &row_deltas, HintPatchMode::EntryLevel);
            }
        }
        self.pin_epoch = self.epoch;
        self.pending.clear();
        Ok(())
    }

    /// Accumulate a delta, falling back to a fresh server bundle on
    /// [`IkpirClientError::FutureDelta`].
    ///
    /// # Purpose
    ///
    /// Sugar for the common gap-handling pattern: try the incremental
    /// accumulate, fall back to a full resync only when the gap is too big.
    ///
    /// # Arguments
    ///
    /// - `delta`        — bundle emitted by the server.
    /// - `fetch_bundle` — closure that obtains a fresh
    ///   [`ServerSetupBundle`]; called only on `FutureDelta`.
    ///
    /// # Rationale
    ///
    /// In the common path (`delta.epoch == self.epoch + 1`) this
    /// delegates to [`Self::accumulate_delta`] and returns
    /// [`DeltaApplyOutcome::Synced`]. If the delta epoch jumps past
    /// `self.epoch + 1`, `fetch_bundle` is invoked, the client is
    /// rebuilt via [`Self::reset_from`], and
    /// [`DeltaApplyOutcome::Resynced`] is returned. All other errors
    /// from `accumulate_delta` are propagated as-is — in particular
    /// [`IkpirClientError::StaleDelta`] (caller should drop the delta)
    /// and [`IkpirClientError::MalformedBundle`] (bug or protocol
    /// mismatch — recovery is not automatic).
    ///
    /// # Returns
    ///
    /// - `Ok(DeltaApplyOutcome::Synced)` on the common path.
    /// - `Ok(DeltaApplyOutcome::Resynced)` when the resync path fires.
    /// - `Err(IkpirClientError::*)` for any error other than
    ///   `FutureDelta`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ikpir_client::{
    ///     DeltaApplyOutcome, FrodoPirBackend, HintDeltaBundle, IkpirClient,
    ///     ServerSetupBundle,
    /// };
    ///
    /// fn handle(
    ///     client: &mut IkpirClient<FrodoPirBackend>,
    ///     delta:  HintDeltaBundle<FrodoPirBackend>,
    ///     fresh:  impl FnOnce() -> ServerSetupBundle<FrodoPirBackend>,
    /// ) {
    ///     match client.try_accumulate_delta_or_resync(delta, fresh).unwrap() {
    ///         DeltaApplyOutcome::Synced   => { /* steady state */ }
    ///         DeltaApplyOutcome::Resynced => { /* dropped any preprocessed queries */ }
    ///     }
    /// }
    /// ```
    pub fn try_accumulate_delta_or_resync<F>(
        &mut self,
        delta: HintDeltaBundle<B>,
        fetch_bundle: F,
    ) -> Result<DeltaApplyOutcome, IkpirClientError>
    where
        F: FnOnce() -> ServerSetupBundle<B>,
    {
        match self.accumulate_delta(delta) {
            Ok(()) => Ok(DeltaApplyOutcome::Synced),
            Err(IkpirClientError::FutureDelta { .. }) => {
                self.reset_from(fetch_bundle());
                Ok(DeltaApplyOutcome::Resynced)
            }
            Err(e) => Err(e),
        }
    }
}

/// Response-rewind read path — available for every backend that implements
/// [`ResponseRewind`] (both shipped LWE backends do). This is the client's
/// **sole decode path**.
impl<B: IncrementalPirBackend + ResponseRewind> IkpirClient<B>
where
    B::Query: Clone,
    B::Response: Clone,
{
    /// Decode a response answered at the server head against the client's
    /// *pinned* hint `H₀`, correcting for the accumulated `ΔD`.
    ///
    /// # Purpose
    ///
    /// Pins `H₀` and bridges the gap to the server's head with the running
    /// `ΔD` accumulated by [`Self::accumulate_delta`]. For each segment it:
    /// (1) subtracts `qᵀ·ΔD` from the response
    /// ([`ResponseRewind::rewind_response`]) — exact in `Z_2³²`, no added
    /// noise — so it decodes as of the pin; (2) runs
    /// [`client_decode`](IndexPirBackend::client_decode) against `H₀`,
    /// recovering the *stale* row; (3) adds `ΔD[row]` to reach the current
    /// row; then (4) runs a branchless fingerprint scan. The decoded value
    /// is identical to what a fresh setup at the head would return (and to
    /// the bench-only `HintPatchClient` comparator's decode — pinned by
    /// `tests/rewind_equivalence.rs` behind the `hint-patch-bench` feature).
    ///
    /// When `ΔD` is empty (a freshly bootstrapped or just-garbage-collected
    /// client, `pin_epoch() == epoch()`), steps 1 and 3 are no-ops and this
    /// reduces to a direct decode against the pinned hint.
    ///
    /// # Arguments
    ///
    /// - `key`   — the same key passed to the matching [`Self::build_query`].
    /// - `query` — the bundle `build_query` returned for `key`, whose secret is
    ///   still in flight; its marker-bearing `b` vectors (the exact ones the
    ///   server answered) drive the correction.
    /// - `resp`  — the response bundle, answered at the current tracked head.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(value))` if a candidate slot carried the matching
    ///   fingerprint.
    /// - `Ok(None)` if no slot matched.
    /// - `Err(EpochMismatch)` if `resp.epoch != self.epoch`.
    /// - `Err(MalformedBundle)` on a wrong segment count or row width, or if
    ///   `query.epoch != resp.epoch` (mismatched `(query, resp)` pair).
    /// - `Err(CellOutOfRange)` if a corrected cell escapes
    ///   `[0, 2^plaintext_bits)` — a corrupt or inconsistent delta/response,
    ///   never a returned wrong value.
    ///
    /// # Rationale
    ///
    /// **Fingerprint-collision behaviour.** If two candidate slots happen to
    /// share `fp` (rare false positive at the cuckoo layer), the returned
    /// `Vec<u8>` is the bitwise OR of their values. This matches the
    /// cuckoo-layer FPR bound; callers that need certainty should retry on a
    /// different fingerprint.
    ///
    /// # Side channels
    ///
    /// Steps 1 and 2 are key-independent (step 1 walks the whole segment map;
    /// step 2 is `client_decode`), and the step-4 fingerprint scan is a
    /// branchless, slot-independent OR-masked select — a timing observer
    /// learns at most `(arity, bucket_size)`. **Step 3 is not constant-time
    /// in the query, though:** it iterates `pending.row(seg, queried_row)`, a
    /// `BTreeMap` range keyed on the queried row, so both the traversal and
    /// the iteration count depend on which row the key hashed to. The leak
    /// is **client-local** — it concerns the row the client itself chose and
    /// never reaches the server or the response — and a fully
    /// constant-time decode is out of scope for this prototype
    /// (`ikpir-common/CLAUDE.md` §3); see `docs/rewind-client-mode.md` §3.
    ///
    /// # Complexity
    ///
    /// `O(arity)` calls to `client_decode` (cold path:
    /// `O(arity · lwe_dim · row_width)` matvec; cheap path with `c`
    /// precomputed: `O(arity · row_width)` vector subtract) plus
    /// `O(Σ |ΔD|)` per-segment correction work — the staleness-growing cost
    /// — plus `O(arity · bucket_size · cells_per_slot)` slot scan work.
    pub fn decode(
        &self,
        key: &[u8],
        query: &PirQueryBundle<B>,
        resp: &PirResponseBundle<B>,
    ) -> Result<Option<Vec<u8>>, IkpirClientError> {
        if resp.epoch != self.epoch {
            return Err(IkpirClientError::EpochMismatch {
                client: self.epoch,
                response: resp.epoch,
            });
        }
        // The `(query, resp)` pair must be from the same round; a mismatched
        // pairing would rewind with the wrong `qᵀ`, silently corrupting the
        // correction rather than erroring.
        if query.epoch != resp.epoch {
            return Err(IkpirClientError::MalformedBundle);
        }
        let arity = self.params.arity();
        if resp.responses.len() != arity || query.queries.len() != arity {
            return Err(IkpirClientError::MalformedBundle);
        }
        let (fp, indices) = self.params.candidate_buckets(key);
        let segment_size = self.params.segment_size();
        let bucket_size = self.params.bucket_size as usize;
        let cps = self.params.cells_per_slot() as usize;
        let value_size = self.params.value_size_in_bytes();
        let plaintext_bound: i64 = 1i64 << self.params.plaintext_bits;

        let mut acc = vec![0u8; value_size];
        let mut found_mask: u64 = 0;

        // `j` addresses several parallel per-segment structures (states,
        // queries, responses, the pending ΔD, and `indices`); the indexed form
        // keeps them visibly in lockstep.
        #[allow(clippy::needless_range_loop)]
        for j in 0..arity {
            // Step 1: resp -= qᵀ·ΔD over the whole segment (on a local copy).
            let mut corrected = resp.responses[j].clone();
            B::rewind_response(
                &self.states[j],
                &query.queries[j],
                &mut corrected,
                self.pending.segment(j),
            );
            // Step 2: decode against the stale, pinned hint H₀ — the row as of
            // the pin.
            let mut cells: Vec<u32> = B::client_decode(&self.states[j], &corrected);
            if cells.len() != bucket_size * cps {
                return Err(IkpirClientError::MalformedBundle);
            }
            // Step 3: cells += ΔD[queried row] — the row as of the head. Must
            // precede the scan.
            let queried_row = indices[j] % segment_size;
            for (off, d) in self.pending.row(j, queried_row) {
                let idx = off as usize;
                if idx >= cells.len() {
                    return Err(IkpirClientError::CellOutOfRange {
                        segment: j,
                        row: queried_row,
                        offset: off,
                    });
                }
                let corrected_cell = i64::from(cells[idx]) + d;
                if corrected_cell < 0 || corrected_cell >= plaintext_bound {
                    return Err(IkpirClientError::CellOutOfRange {
                        segment: j,
                        row: queried_row,
                        offset: off,
                    });
                }
                cells[idx] = corrected_cell as u32;
            }
            // Step 4: branchless fingerprint scan.
            for s in 0..bucket_size {
                let slot = &cells[s * cps..(s + 1) * cps];
                let (decoded_fp, value_bytes) = unpack_slot_cells(&self.params, slot);
                let mask64 = ct_eq_u64_mask(decoded_fp, fp);
                let mask8 = (mask64 & 0xFF) as u8;
                for (a, v) in acc.iter_mut().zip(value_bytes.iter()) {
                    *a |= mask8 & *v;
                }
                found_mask |= mask64;
            }
        }
        Ok(if found_mask != 0 { Some(acc) } else { None })
    }
}
