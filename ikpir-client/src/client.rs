use ikpir_server::{
    HintDeltaBundle, IndexPirBackend, IncrementalPirBackend, PrecomputingPirBackend,
    PirQueryBundle, PirResponseBundle, ServerSetupBundle,
};
use segmented_cuckoo::{unpack_slot_cells, CuckooParams};

use crate::error::IkpirClientError;

/// Client-side IKPIR engine, generic over the PIR backend `B`.
///
/// Owns one [`B::ClientState`](IndexPirBackend::ClientState) per segment,
/// the SCF [`CuckooParams`], and a strict-monotone epoch counter. Does not
/// hold any `CuckooKVStore` cells — those live on the server.
///
/// # Threading
///
/// All methods are synchronous. `build_query` and `apply_delta` take
/// `&mut self`. Wrap in `Mutex` if exposing across threads.
pub struct IkpirClient<B: IndexPirBackend> {
    params:         CuckooParams,
    backend_params: Vec<B::ServerParams>,
    states:         Vec<B::ClientState>,
    epoch:          u64,
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
    /// Materialises one [`B::ClientState`](IndexPirBackend::ClientState) per
    /// segment via [`B::client_setup`](IndexPirBackend::client_setup),
    /// caches the SCF [`CuckooParams`], and adopts the server's epoch.
    ///
    /// # Invariants
    ///
    /// `bundle.params.plaintext_bits` (and the rest of the SCF geometry) is
    /// fixed for this client's lifetime; all subsequent
    /// [`Self::apply_delta`] and [`Self::reset_from`] calls must come from
    /// servers built with the same `plaintext_bits`. See `IkpirServer::new`
    /// for the upgrade path that would be required to lift this invariant.
    pub fn from_setup(bundle: ServerSetupBundle<B>) -> Self {
        let arity = bundle.params.arity();
        debug_assert_eq!(bundle.backend_params.len(), arity);
        debug_assert_eq!(bundle.hints.len(), arity);

        let states: Vec<B::ClientState> = bundle.backend_params.iter()
            .zip(bundle.hints.iter())
            .map(|(p, h)| B::client_setup(p, h))
            .collect();

        Self {
            params:         bundle.params,
            backend_params: bundle.backend_params,
            states,
            epoch:          bundle.epoch,
        }
    }

    /// Replace all internal state from a fresh setup bundle.
    ///
    /// Use after a server-side `full_rebuild`, or whenever
    /// [`IkpirClient::apply_delta`] reports
    /// [`IkpirClientError::FutureDelta`] (the gap is unbridgeable
    /// incrementally).
    pub fn reset_from(&mut self, bundle: ServerSetupBundle<B>) {
        *self = Self::from_setup(bundle);
    }

    /// Current client epoch. Strictly monotone across `apply_delta`
    /// and `reset_from`.
    pub fn epoch(&self) -> u64 { self.epoch }

    /// SCF geometry parameters in use.
    pub fn params(&self) -> CuckooParams { self.params }
}

impl<B: IndexPirBackend> IkpirClient<B>
where
    B::Query: Clone,
{
    /// Build a per-segment PIR query for `key`.
    ///
    /// Re-derives the SCF candidate buckets locally
    /// ([`CuckooParams::candidate_buckets`]) and emits one
    /// [`B::Query`](IndexPirBackend::Query) per segment, where the j-th
    /// query targets row `indices[j] % segment_size` in segment `j`. The
    /// returned [`PirQueryBundle`] carries the current client epoch; the
    /// server rejects the query with
    /// [`IkpirError::StaleEpoch`](ikpir_server::IkpirError::StaleEpoch) if
    /// it has moved on.
    pub fn build_query(&mut self, key: &[u8]) -> PirQueryBundle<B> {
        let (_fp, indices) = self.params.candidate_buckets(key);
        let arity          = self.params.arity();
        let segment_size   = self.params.segment_size();

        let queries: Vec<B::Query> = (0..arity)
            .map(|j| B::client_query(&mut self.states[j], indices[j] % segment_size))
            .collect();

        PirQueryBundle { epoch: self.epoch, queries }
    }
}

impl<B: IndexPirBackend> IkpirClient<B>
where
    B::Response: Clone,
{
    /// Decode a server response.
    ///
    /// Re-derives the fingerprint and candidate buckets from `key`, runs
    /// [`B::client_decode`](IndexPirBackend::client_decode) on each
    /// per-segment response, slices the recovered cell row into the SCF
    /// `bucket_size` slot streams, and returns the matching value or
    /// `Ok(None)` if no slot in the candidate buckets carries the matching
    /// fingerprint.
    ///
    /// # Side-channel hardening
    ///
    /// Every `arity × bucket_size` candidate slot is unpacked and merged
    /// into a fixed-size accumulator via a branchless OR-masked select.
    /// The probe path is independent of which slot (if any) holds the
    /// match, so a co-located timing observer learns nothing beyond
    /// `(arity, bucket_size)` from the decode call. The `if sk == 0`
    /// shortcut in the underlying LWE decode is a separate concern; see
    /// `ikpir-server/CLAUDE.md` for the server-side threat model.
    ///
    /// # Fingerprint-collision behaviour
    ///
    /// If two candidate slots happen to share `fp` (rare false positive),
    /// the returned `Vec<u8>` is the bitwise OR of their values. This
    /// matches the cuckoo-layer FPR bound; callers that need certainty
    /// should retry on a different fingerprint.
    ///
    /// # Errors
    ///
    /// - [`IkpirClientError::EpochMismatch`] if `resp.epoch != self.epoch`
    ///   (the server moved between query and answer).
    /// - [`IkpirClientError::MalformedBundle`] if the response has the wrong
    ///   number of segments or an inner row of the wrong width.
    pub fn decode(
        &self,
        key:  &[u8],
        resp: &PirResponseBundle<B>,
    ) -> Result<Option<Vec<u8>>, IkpirClientError> {
        if resp.epoch != self.epoch {
            return Err(IkpirClientError::EpochMismatch {
                client: self.epoch, response: resp.epoch,
            });
        }
        let arity = self.params.arity();
        if resp.responses.len() != arity {
            return Err(IkpirClientError::MalformedBundle);
        }
        let (fp, _) = self.params.candidate_buckets(key);
        let bucket_size = self.params.bucket_size as usize;
        let cps         = self.params.cells_per_slot() as usize;
        let value_size  = self.params.value_size_in_bytes();

        let mut acc = vec![0u8; value_size];
        let mut found_mask: u32 = 0;

        for j in 0..arity {
            let row: Vec<u32> = B::client_decode(&self.states[j], &resp.responses[j]);
            if row.len() != bucket_size * cps {
                return Err(IkpirClientError::MalformedBundle);
            }
            for s in 0..bucket_size {
                let slot = &row[s * cps..(s + 1) * cps];
                let (decoded_fp, value_bytes) = unpack_slot_cells(&self.params, slot);
                let mask32 = ct_eq_u32_mask(decoded_fp, fp);
                let mask8  = (mask32 & 0xFF) as u8;
                for (a, v) in acc.iter_mut().zip(value_bytes.iter()) {
                    *a |= mask8 & *v;
                }
                found_mask |= mask32;
            }
        }
        Ok(if found_mask != 0 { Some(acc) } else { None })
    }
}

/// Branchless `u32` equality mask: returns `0xFFFFFFFF` if `a == b`, else `0`.
///
/// Standard constant-time trick: `x ^ b == 0` iff `a == b`; squeeze that
/// zero/non-zero into bit 31 via `x | -x`, shift down, then subtract 1 to
/// flip the meaning.
#[inline]
fn ct_eq_u32_mask(a: u32, b: u32) -> u32 {
    let x = a ^ b;
    ((x | x.wrapping_neg()) >> 31).wrapping_sub(1)
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
    /// **Strict-monotone:** the only accepted shape is
    /// `delta.epoch == self.epoch + 1`. Older deltas are
    /// [`IkpirClientError::StaleDelta`]; gaps are
    /// [`IkpirClientError::FutureDelta`] — the caller must recover by
    /// calling [`IkpirClient::reset_from`] with a fresh server bundle.
    ///
    /// # Errors
    ///
    /// - [`IkpirClientError::StaleDelta`]
    /// - [`IkpirClientError::FutureDelta`]
    /// - [`IkpirClientError::MalformedBundle`] if `per_segment_row_deltas.len()`
    ///   doesn't match arity.
    pub fn apply_delta(
        &mut self,
        delta: HintDeltaBundle<B>,
    ) -> Result<(), IkpirClientError> {
        let expected = self.epoch + 1;
        if delta.epoch < expected {
            return Err(IkpirClientError::StaleDelta  { expected, got: delta.epoch });
        }
        if delta.epoch > expected {
            return Err(IkpirClientError::FutureDelta { expected, got: delta.epoch });
        }
        let arity = self.params.arity();
        if delta.per_segment_row_deltas.len() != arity {
            return Err(IkpirClientError::MalformedBundle);
        }
        for (j, deltas) in delta.per_segment_row_deltas.iter().enumerate() {
            if !deltas.is_empty() {
                B::client_patch_state(
                    &mut self.states[j],
                    &self.backend_params[j],
                    deltas,
                );
            }
        }
        self.epoch = delta.epoch;
        Ok(())
    }

    /// Apply a delta, falling back to a fresh server bundle on
    /// [`IkpirClientError::FutureDelta`].
    ///
    /// In the common path (`delta.epoch == self.epoch + 1`) this delegates
    /// to [`Self::apply_delta`] and returns [`DeltaApplyOutcome::Synced`].
    /// If the delta epoch jumps past `self.epoch + 1`, the caller-supplied
    /// `fetch_bundle` closure is invoked to obtain a fresh
    /// [`ServerSetupBundle`], the client is rebuilt via
    /// [`Self::reset_from`], and [`DeltaApplyOutcome::Resynced`] is returned.
    ///
    /// All other errors from `apply_delta` are propagated as-is — in
    /// particular [`IkpirClientError::StaleDelta`] (caller should drop the
    /// delta) and [`IkpirClientError::MalformedBundle`] (bug or protocol
    /// mismatch — recovery is not automatic).
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
            Ok(())                                        => Ok(DeltaApplyOutcome::Synced),
            Err(IkpirClientError::FutureDelta { .. })     => {
                self.reset_from(fetch_bundle());
                Ok(DeltaApplyOutcome::Resynced)
            }
            Err(e)                                        => Err(e),
        }
    }
}
