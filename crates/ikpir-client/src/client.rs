//! `IkpirClient<B>` — the client-side state machine.
//!
//! # Purpose
//!
//! Translates user-level `(key, query)` operations into the per-segment
//! Index-PIR `(Query, Response)` exchanges with the IKPIR server.
//! Maintains the strict-monotone client epoch and patches its
//! per-segment `B::ClientState` from `HintDeltaBundle`s.
//!
//! # Design / architecture
//!
//! - **State.** [`CuckooParams`] + one `B::ServerParams` and
//!   `B::ClientState` per segment + the epoch counter. Never owns a
//!   `CuckooKVStore` — that's server-side.
//! - **Lookup geometry is public.** `candidate_buckets` is
//!   re-derivable from `params` alone, so the client computes it on
//!   every query without privacy cost.
//! - **Side-channel hardening in `decode`.** Every candidate slot is
//!   unpacked and merged into a fixed-size accumulator via a
//!   branchless OR-masked select; a timing observer learns at most
//!   `(arity, bucket_size)`.
//! - **Two recovery paths.** `apply_delta` for the strict-monotone
//!   steady state, `reset_from` for after a `full_rebuild` or a
//!   `FutureDelta` gap.
//!
//! # Related files
//!
//! - `lib.rs` — re-exports `IkpirClient`, `DeltaApplyOutcome`, and the
//!   ikpir-server wire/backend types.
//! - `error.rs` — `IkpirClientError` variants.
//! - `ikpir-server::IkpirServer` — counterpart on the server side.

use ikpir_common::{
    HintDeltaBundle, HintPatchMode, IncrementalPirBackend, IndexPirBackend, ParallelSetupBackend,
    PirQueryBundle, PirResponseBundle, PrecomputingPirBackend, ServerSetupBundle,
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
/// preprocessing (`B::ClientState`); the latter is what
/// `apply_delta` patches in lock-step with server mutations.
///
/// # Threading
///
/// All methods are synchronous. `build_query` and `apply_delta` take
/// `&mut self`. Wrap in `Mutex` if exposing across threads.
pub struct IkpirClient<B: IndexPirBackend> {
    params: CuckooParams,
    states: Vec<B::ClientState>,
    epoch: u64,
    /// Realization used by [`Self::apply_delta`] — see [`HintPatchMode`].
    /// A purely local compute choice: the accepted [`HintDeltaBundle`]
    /// and the resulting state are identical under either mode, so the
    /// client's mode never needs to match the server's.
    hint_patch_mode: HintPatchMode,
}

/// Outcome of [`IkpirClient::try_apply_delta_or_resync`].
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
    /// [`Self::apply_delta`] or [`Self::reset_from`] call must come from
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
            hint_patch_mode: HintPatchMode::default(),
        }
    }

    /// Replace all internal state from a fresh setup bundle.
    ///
    /// # Purpose
    ///
    /// Resync entry point. Use after a server-side `full_rebuild`, or
    /// whenever [`IkpirClient::apply_delta`] reports
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
    /// as a swap so existing callers don't need to take ownership. The
    /// configured [`HintPatchMode`] is a client preference, not server
    /// state, so it survives the reset.
    pub fn reset_from(&mut self, bundle: ServerSetupBundle<B>) {
        let mode = self.hint_patch_mode;
        *self = Self::from_setup(bundle);
        self.hint_patch_mode = mode;
    }

    /// Current client epoch. Strictly monotone across `apply_delta`
    /// and `reset_from`.
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// SCF geometry parameters in use.
    pub const fn params(&self) -> CuckooParams {
        self.params
    }

    /// Realization used by [`Self::apply_delta`] for incremental hint
    /// patches. Defaults to [`HintPatchMode::EntryLevel`].
    pub const fn hint_patch_mode(&self) -> HintPatchMode {
        self.hint_patch_mode
    }

    /// Select the [`HintPatchMode`] realization for future
    /// [`Self::apply_delta`] calls.
    ///
    /// # Rationale
    ///
    /// The mode is a **local compute choice**: either realization leaves
    /// the per-segment state identical (all arithmetic mod `2³²`), so
    /// the client's mode never needs to match the server's and may be
    /// switched between deltas at will. Entry-level is the default (and
    /// the cheaper realization — `Θ(n)` per touched cell instead of
    /// `Θ(n·ω)` per touched row); row-level exists as the
    /// SimplePIR-style baseline the benches compare against. Survives
    /// [`Self::reset_from`].
    pub fn set_hint_patch_mode(&mut self, mode: HintPatchMode) {
        self.hint_patch_mode = mode;
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

    /// Multi-threaded twin of [`Self::reset_from`]. Like it, the
    /// configured [`HintPatchMode`] is a client preference and survives
    /// the reset.
    pub fn reset_from_parallel(&mut self, bundle: ServerSetupBundle<B>) {
        let mode = self.hint_patch_mode;
        *self = Self::from_setup_parallel(bundle);
        self.hint_patch_mode = mode;
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

impl<B: IndexPirBackend> IkpirClient<B>
where
    B::Response: Clone,
{
    /// Decode a server response.
    ///
    /// # Purpose
    ///
    /// Inverse of [`Self::build_query`]. Re-derives the fingerprint and
    /// candidate buckets from `key`, runs
    /// [`B::client_decode`](IndexPirBackend::client_decode) on each
    /// per-segment response, slices the recovered cell row into the
    /// SCF `bucket_size` slot streams, and returns the matching value
    /// or `Ok(None)` if no slot in the candidate buckets carries the
    /// matching fingerprint.
    ///
    /// # Arguments
    ///
    /// - `key`  — the same key passed to the matching
    ///   `build_query` call.
    /// - `resp` — the response bundle returned by the server's
    ///   `answer`.
    ///
    /// # Rationale
    ///
    /// **Side-channel hardening.** Every `arity × bucket_size`
    /// candidate slot is unpacked and merged into a fixed-size
    /// accumulator via a branchless OR-masked select. The probe path
    /// is independent of which slot (if any) holds the match, so a
    /// co-located timing observer learns nothing beyond
    /// `(arity, bucket_size)` from the decode call. The `if sk == 0`
    /// shortcut in the underlying LWE decode is a separate concern;
    /// see `crates/ikpir-server/CLAUDE.md` for the server-side threat model.
    ///
    /// **Fingerprint-collision behaviour.** If two candidate slots
    /// happen to share `fp` (rare false positive at the cuckoo layer),
    /// the returned `Vec<u8>` is the bitwise OR of their values. This
    /// matches the cuckoo-layer FPR bound; callers that need certainty
    /// should retry on a different fingerprint. This OR-of-matching-values
    /// under ambiguity is the same failure event the scheme's correctness
    /// lemma charges (the collision event), so the bound covers it.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(value))` if a candidate slot carried the matching
    ///   fingerprint.
    /// - `Ok(None)` if no slot matched.
    /// - `Err(IkpirClientError::EpochMismatch)` if
    ///   `resp.epoch != self.epoch` (the server moved between query
    ///   and answer).
    /// - `Err(IkpirClientError::MalformedBundle)` if the response has
    ///   the wrong number of segments or an inner row of the wrong
    ///   width.
    ///
    /// # Complexity
    ///
    /// `O(arity)` calls to `B::client_decode` (cold path:
    /// `O(arity · lwe_dim · row_width)` matvec; cheap path with `c`
    /// precomputed: `O(arity · row_width)` vector subtract) plus
    /// `O(arity · bucket_size · cells_per_slot)` slot scan work.
    pub fn decode(
        &self,
        key: &[u8],
        resp: &PirResponseBundle<B>,
    ) -> Result<Option<Vec<u8>>, IkpirClientError> {
        if resp.epoch != self.epoch {
            return Err(IkpirClientError::EpochMismatch {
                client: self.epoch,
                response: resp.epoch,
            });
        }
        let arity = self.params.arity();
        if resp.responses.len() != arity {
            return Err(IkpirClientError::MalformedBundle);
        }
        let (fp, _) = self.params.candidate_buckets(key);
        let bucket_size = self.params.bucket_size as usize;
        let cps = self.params.cells_per_slot() as usize;
        let value_size = self.params.value_size_in_bytes();

        let mut acc = vec![0u8; value_size];
        let mut found_mask: u64 = 0;

        for j in 0..arity {
            let row: Vec<u32> = B::client_decode(&self.states[j], &resp.responses[j]);
            if row.len() != bucket_size * cps {
                return Err(IkpirClientError::MalformedBundle);
            }
            for s in 0..bucket_size {
                let slot = &row[s * cps..(s + 1) * cps];
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
    /// cheap path (one vector subtract + rounding). [`IkpirClient::apply_delta`]
    /// keeps the precomputed `c` values consistent with the patched hint.
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
    /// Apply a hint-delta bundle from the server.
    ///
    /// # Purpose
    ///
    /// Steady-state synchronisation entry point. Patches the
    /// per-segment hint copy in `B::ClientState` so subsequent queries
    /// agree with the server's post-mutation database.
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
    ///   from the client's `params`, or `per_segment_row_deltas.len()`
    ///   doesn't match arity.
    ///
    /// # Complexity
    ///
    /// `O(Σ row_deltas · lwe_dim)` per segment — see
    /// `IncrementalPirBackend::client_patch_state`. Empty segments
    /// short-circuit.
    pub fn apply_delta(&mut self, delta: HintDeltaBundle<B>) -> Result<(), IkpirClientError> {
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
        let arity = self.params.arity();
        if delta.per_segment_row_deltas.len() != arity {
            return Err(IkpirClientError::MalformedBundle);
        }
        for (j, deltas) in delta.per_segment_row_deltas.iter().enumerate() {
            if !deltas.is_empty() {
                B::client_patch_state(&mut self.states[j], deltas, self.hint_patch_mode);
            }
        }
        self.epoch = delta.epoch;
        Ok(())
    }

    /// Apply a delta, falling back to a fresh server bundle on
    /// [`IkpirClientError::FutureDelta`].
    ///
    /// # Purpose
    ///
    /// Sugar for the common gap-handling pattern: try the incremental
    /// patch, fall back to a full resync only when the gap is too big.
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
    /// delegates to [`Self::apply_delta`] and returns
    /// [`DeltaApplyOutcome::Synced`]. If the delta epoch jumps past
    /// `self.epoch + 1`, `fetch_bundle` is invoked, the client is
    /// rebuilt via [`Self::reset_from`], and
    /// [`DeltaApplyOutcome::Resynced`] is returned. All other errors
    /// from `apply_delta` are propagated as-is — in particular
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
    ///     match client.try_apply_delta_or_resync(delta, fresh).unwrap() {
    ///         DeltaApplyOutcome::Synced   => { /* steady state */ }
    ///         DeltaApplyOutcome::Resynced => { /* dropped any preprocessed queries */ }
    ///     }
    /// }
    /// ```
    pub fn try_apply_delta_or_resync<F>(
        &mut self,
        delta: HintDeltaBundle<B>,
        fetch_bundle: F,
    ) -> Result<DeltaApplyOutcome, IkpirClientError>
    where
        F: FnOnce() -> ServerSetupBundle<B>,
    {
        match self.apply_delta(delta) {
            Ok(()) => Ok(DeltaApplyOutcome::Synced),
            Err(IkpirClientError::FutureDelta { .. }) => {
                self.reset_from(fetch_bundle());
                Ok(DeltaApplyOutcome::Resynced)
            }
            Err(e) => Err(e),
        }
    }
}
