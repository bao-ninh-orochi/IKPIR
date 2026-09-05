//! [`DeltaApplyOutcome`] — shared result type for the "apply a delta, resync
//! on an unbridgeable gap" convenience wrapper both client flows expose.

/// Outcome of [`RewindClient::try_accumulate_delta_or_resync`](crate::RewindClient::try_accumulate_delta_or_resync)
/// / [`HintPatchClient::try_apply_delta_or_resync`](crate::HintPatchClient::try_apply_delta_or_resync).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaApplyOutcome {
    /// The delta was applied incrementally (the common case).
    Synced,
    /// The delta was too far ahead; the fetched fresh bundle was used to
    /// reset the client (`RewindClient::reset_from` /
    /// `HintPatchClient::reset_from`).
    Resynced,
}
