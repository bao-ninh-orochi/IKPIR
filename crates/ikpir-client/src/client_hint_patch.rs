//! [`HintPatchClient`] — the client-side state machine for the
//! **client-hint-patch** flow.
//!
//! # Purpose
//!
//! `HintPatchClient<B>` is the classical client that folds every published
//! [`HintDeltaBundle`] into its own hint immediately
//! (`H ← H + Σ A[:,col]·δ`, `Θ(n·τ·ω)` per batch of `τ` mutations over row
//! width `ω` — the iSimplePIR-style realization), via a selectable
//! [`HintPatchMode`], and decodes directly against the patched hint. This is
//! the flow the CANS 2026 camera-ready's client numbers were measured with.
//!
//! Contrast with the client-rewind flow ([`crate::RewindClient`]), which
//! pins its bootstrap hint and instead accumulates the published `ΔD`
//! (`Θ(τ·ω)` maintenance per batch, plus a staleness-growing per-query
//! correction at decode time). Both flows consume the same
//! [`HintDeltaBundle`] stream from the server and return the same decoded
//! value for every query — pinned by `tests/client_flow_parity.rs`. See
//! `docs/rewind-client-mode.md` for the full comparison.
//!
//! # Design / architecture
//!
//! [`HintPatchClient<B>`] holds `CuckooParams` + one `B::ClientState` per
//! segment + the epoch counter + a selectable [`HintPatchMode`] realization.
//! It carries no `ΔD` accumulator — it patches the hint directly, so there
//! is nothing to roll forward.
//!
//! # Related files
//!
//! - `client_rewind.rs` — the client-rewind flow, `RewindClient`.
//! - `benches/client_mutation.rs` — the `--update-mode patch` sweep.
//! - `tests/client_flow_parity.rs` — the patch == rewind == fresh pin.

use ikpir_common::{
    HintDeltaBundle, HintPatchMode, IncrementalPirBackend, IndexPirBackend, ParallelSetupBackend,
    PirQueryBundle, PirResponseBundle, PrecomputingPirBackend, ServerSetupBundle,
};
use segmented_cuckoo::{unpack_slot_cells, CuckooParams};

use crate::ct::ct_eq_u64_mask;
use crate::error::IkpirClientError;
use crate::outcome::DeltaApplyOutcome;

/// The per-segment client-setup primitive, as a function pointer — see
/// `crate::client_rewind::PerSegmentClientSetup`, which this mirrors so
/// [`HintPatchClient::from_setup`] and [`HintPatchClient::from_setup_parallel`]
/// can share one body.
type PerSegmentClientSetup<B> = fn(
    &<B as IndexPirBackend>::ServerParams,
    &<B as IndexPirBackend>::Hint,
) -> <B as IndexPirBackend>::ClientState;

/// Client-side IKPIR engine for the **client-hint-patch** flow, generic over
/// the PIR backend `B`. See the module docs.
pub struct HintPatchClient<B: IndexPirBackend> {
    params: CuckooParams,
    states: Vec<B::ClientState>,
    epoch: u64,
    /// Realization used by [`Self::apply_delta`] — see [`HintPatchMode`]. A
    /// purely local compute choice: the accepted [`HintDeltaBundle`] and the
    /// resulting state are identical under either mode.
    hint_patch_mode: HintPatchMode,
}

impl<B: IndexPirBackend> HintPatchClient<B> {
    /// Build a fresh client from a server setup bundle — the
    /// client-hint-patch counterpart of [`crate::RewindClient::from_setup`].
    pub fn from_setup(bundle: ServerSetupBundle<B>) -> Self {
        Self::assemble(bundle, B::client_setup)
    }

    fn assemble(bundle: ServerSetupBundle<B>, client_setup: PerSegmentClientSetup<B>) -> Self {
        let arity = bundle.params.arity();
        debug_assert_eq!(bundle.backend_params.len(), arity);
        debug_assert_eq!(bundle.hints.len(), arity);

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
    /// whenever [`HintPatchClient::apply_delta`] reports
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

    /// Current client epoch. Strictly monotone across [`Self::apply_delta`].
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

/// Optimized (multi-threaded) bootstrap — the client-hint-patch counterpart
/// of [`crate::RewindClient::from_setup_parallel`]; identical resulting
/// client, computed across cores.
impl<B: ParallelSetupBackend> HintPatchClient<B> {
    /// Multi-threaded twin of [`Self::from_setup`].
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

impl<B: IndexPirBackend> HintPatchClient<B>
where
    B::Query: Clone,
{
    /// Build a per-segment PIR query for `key` — identical contract to
    /// [`crate::RewindClient::build_query`].
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

impl<B: IndexPirBackend> HintPatchClient<B>
where
    B::Response: Clone,
{
    /// Decode a server response against the directly-patched hint — the
    /// client-hint-patch counterpart of `RewindClient::decode`.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(value))` if a candidate slot carried the matching
    ///   fingerprint, `Ok(None)` if none did.
    /// - `Err(EpochMismatch)` if `resp.epoch != self.epoch`.
    /// - `Err(MalformedBundle)` on a wrong segment count or row width.
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

impl<B: PrecomputingPirBackend> HintPatchClient<B> {
    /// Phase B amortisation — pre-sample `count` query slots **per segment**.
    ///
    /// Each subsequent [`HintPatchClient::build_query`] call consumes one
    /// slot per segment off the prepared queue (cheap path: one vector add)
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
    /// After this call, every matching [`HintPatchClient::decode`] takes the
    /// cheap path (one vector subtract + rounding). [`HintPatchClient::apply_delta`]
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

impl<B: IncrementalPirBackend> HintPatchClient<B> {
    /// Apply a hint-delta bundle from the server, patching the hint
    /// immediately — the client-hint-patch counterpart of
    /// `RewindClient::accumulate_delta`.
    ///
    /// # Constraints
    ///
    /// **Strict-monotone:** the only accepted shape is
    /// `delta.epoch == self.epoch + 1`. Older deltas are
    /// [`IkpirClientError::StaleDelta`]; gaps are
    /// [`IkpirClientError::FutureDelta`] — the caller must recover by
    /// calling [`HintPatchClient::reset_from`] with a fresh server bundle.
    ///
    /// # Complexity
    ///
    /// `O(Σ row_deltas · lwe_dim)` per segment.
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
    ///     DeltaApplyOutcome, FrodoPirBackend, HintDeltaBundle, HintPatchClient,
    ///     ServerSetupBundle,
    /// };
    ///
    /// fn handle(
    ///     client: &mut HintPatchClient<FrodoPirBackend>,
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
