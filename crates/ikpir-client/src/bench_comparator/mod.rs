//! [`HintPatchClient`] — the client-side hint-patch **benchmark comparator**.
//!
//! # Purpose
//!
//! Response-rewind (`IkpirClient::{accumulate_delta, decode, collect_garbage}`)
//! is the client's sole *production* update strategy — see the crate-level
//! docs and `docs/rewind-client-mode.md`. Before that pivot, the client could
//! also fold every delta into its hint immediately
//! (`H ← H + Σ A[:,col]·δ`, `Θ(n·τ·ω)` per batch of `τ` mutations over row
//! width `ω`, versus rewind's `Θ(τ·ω)` — a factor-`n` cheaper maintenance).
//! That classical path is kept **only** as this benchmark comparator, gated
//! behind the `hint-patch-bench` Cargo feature (disabled by default — a
//! production build never links this module): `client_mutation` sweeps
//! `--update-mode patch,rewind` head-to-head, and
//! `tests/rewind_equivalence.rs` pins that the two decode identically.
//!
//! # Design / architecture
//!
//! [`HintPatchClient<B>`] mirrors the shape `IkpirClient<B>` had before the
//! rewind pivot: `CuckooParams` + one `B::ClientState` per segment + the
//! epoch counter + a selectable [`HintPatchMode`] realization. It carries no
//! `ΔD` accumulator — it patches the hint directly, so there is nothing to
//! roll forward. It is not part of the production client and has no
//! `reset_from` / precomputation surface — the comparator only needs to
//! bootstrap, apply deltas, and decode.
//!
//! # Related files
//!
//! - `crate::client` — the production rewind-only `IkpirClient`.
//! - `benches/client_mutation.rs` — the `--update-mode patch` sweep.
//! - `tests/rewind_equivalence.rs` — the patch == rewind == fresh pin.

use ikpir_common::{
    HintDeltaBundle, HintPatchMode, IncrementalPirBackend, IndexPirBackend, ParallelSetupBackend,
    PirQueryBundle, PirResponseBundle, ServerSetupBundle,
};
use segmented_cuckoo::{unpack_slot_cells, CuckooParams};

use crate::error::IkpirClientError;

/// The per-segment client-setup primitive, as a function pointer — see
/// `crate::client::PerSegmentClientSetup`, which this mirrors so
/// [`HintPatchClient::from_setup`] and [`HintPatchClient::from_setup_parallel`]
/// can share one body.
type PerSegmentClientSetup<B> = fn(
    &<B as IndexPirBackend>::ServerParams,
    &<B as IndexPirBackend>::Hint,
) -> <B as IndexPirBackend>::ClientState;

/// Benchmark-only client that patches its hint immediately on every delta —
/// the classical alternative response-rewind replaced in production. See the
/// module docs.
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
    /// Build a fresh comparator client from a server setup bundle — the
    /// hint-patch counterpart of [`crate::IkpirClient::from_setup`].
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
    /// [`Self::apply_delta`] calls. A purely local compute choice — either
    /// realization leaves the per-segment state identical (all arithmetic
    /// mod `2³²`).
    pub fn set_hint_patch_mode(&mut self, mode: HintPatchMode) {
        self.hint_patch_mode = mode;
    }
}

/// Optimized (multi-threaded) bootstrap — the comparator counterpart of
/// [`crate::IkpirClient::from_setup_parallel`]; identical resulting client,
/// computed across cores.
impl<B: ParallelSetupBackend> HintPatchClient<B> {
    /// Multi-threaded twin of [`Self::from_setup`].
    pub fn from_setup_parallel(bundle: ServerSetupBundle<B>) -> Self {
        Self::assemble(bundle, B::client_setup_parallel)
    }
}

impl<B: IndexPirBackend> HintPatchClient<B>
where
    B::Query: Clone,
{
    /// Build a per-segment PIR query for `key` — identical contract to
    /// [`crate::IkpirClient::build_query`].
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
    /// hint-patch counterpart of `IkpirClient::decode`.
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

/// Branchless `u64` equality mask: returns `u64::MAX` if `a == b`, else `0`.
/// Local copy of `crate::client`'s helper — kept private to each module so
/// the production client carries no bench-comparator dependency.
#[inline]
const fn ct_eq_u64_mask(a: u64, b: u64) -> u64 {
    let x = a ^ b;
    ((x | x.wrapping_neg()) >> 63).wrapping_sub(1)
}

impl<B: IncrementalPirBackend> HintPatchClient<B> {
    /// Apply a hint-delta bundle from the server, patching the hint
    /// immediately — the hint-patch counterpart of
    /// `IkpirClient::accumulate_delta`.
    ///
    /// # Constraints
    ///
    /// **Strict-monotone:** the only accepted shape is
    /// `delta.epoch == self.epoch + 1`. Older deltas are
    /// [`IkpirClientError::StaleDelta`]; gaps are
    /// [`IkpirClientError::FutureDelta`] (unhandled here — the comparator has
    /// no `reset_from`; rebuild it from a fresh bundle instead).
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
}
