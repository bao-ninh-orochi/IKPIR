//! Client-side error type [`IkpirClientError`].
//!
//! # Purpose
//!
//! Enumerates every failure mode reachable from `IkpirClient`'s public
//! API plus a wrapper variant that forwards server errors for
//! ergonomic `?` propagation in synchronous in-process composition.
//!
//! # Design / architecture
//!
//! Three categories:
//!
//! - **Epoch-coherence failures** ([`StaleDelta`](IkpirClientError::StaleDelta),
//!   [`FutureDelta`](IkpirClientError::FutureDelta),
//!   [`EpochMismatch`](IkpirClientError::EpochMismatch)) — caller chose
//!   the wrong incremental shape.
//! - **Bundle-shape failures**
//!   ([`MalformedBundle`](IkpirClientError::MalformedBundle)) — bug or
//!   protocol-version mismatch.
//! - **Forwarded server errors** ([`Server`](IkpirClientError::Server))
//!   — wraps `IkpirError` from the server crate.
//!
//! # Related files
//!
//! - `client.rs` — sole producer of these variants.
//! - `ikpir-server::IkpirError` — wrapped by the `Server` variant.

use std::fmt;

use ikpir_common::{ClientUpdateMode, IkpirError};

/// Errors returned by [`IkpirClient`](crate::IkpirClient) methods.
///
/// # Purpose
///
/// Single error type the IKPIR client surfaces. See the module-level
/// docs for the three categories of variants.
///
/// # Rationale
///
/// `IkpirClientError: From<IkpirError>` lets a chained client-server
/// flow use a single `?` operator everywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IkpirClientError {
    /// Returned by [`IkpirClient::apply_delta`](crate::IkpirClient::apply_delta)
    /// when `delta.epoch ≤ self.epoch` — the delta has already been applied
    /// or arrived out of order.
    StaleDelta {
        /// Epoch the client expected (`self.epoch + 1`).
        expected: u64,
        /// Epoch the delta carried.
        got: u64,
    },
    /// Returned by [`IkpirClient::apply_delta`](crate::IkpirClient::apply_delta)
    /// when `delta.epoch > self.epoch + 1` — the client missed at least one
    /// update. Caller must call
    /// [`IkpirClient::reset_from`](crate::IkpirClient::reset_from) with a
    /// fresh setup bundle.
    FutureDelta {
        /// Epoch the client expected (`self.epoch + 1`).
        expected: u64,
        /// Epoch the delta carried.
        got: u64,
    },
    /// Returned by [`IkpirClient::decode`](crate::IkpirClient::decode) when
    /// `resp.epoch != self.epoch` — the server moved between query and answer.
    EpochMismatch {
        /// Client's current epoch.
        client: u64,
        /// Epoch the response carried.
        response: u64,
    },
    /// Returned by [`IkpirClient::apply_delta`](crate::IkpirClient::apply_delta)
    /// when `delta.params` does not equal the client's cached `params`, or
    /// `delta.per_segment_row_deltas.len()` does not match `params.arity()`.
    /// Also returned by `decode` when a bundle's segment count or row width
    /// does not match the cached `params.arity()` /
    /// `bucket_size × cells_per_slot`.
    MalformedBundle,
    /// A method was called in the wrong
    /// [`ClientUpdateMode`](ikpir_common::ClientUpdateMode): `apply_delta` /
    /// `decode` require [`HintPatch`](ikpir_common::ClientUpdateMode::HintPatch);
    /// `accumulate_delta` / `collect_garbage` require
    /// [`Rewind`](ikpir_common::ClientUpdateMode::Rewind). Switch the entry point
    /// (or the mode) to match — the call is a caller-side logic error, never a
    /// wrong answer.
    WrongUpdateMode {
        /// The mode the called method requires.
        expected: ClientUpdateMode,
        /// The mode the client is currently in.
        actual: ClientUpdateMode,
    },
    /// Returned by
    /// [`decode_rewind`](crate::IkpirClient::decode_rewind) when a decoded cell,
    /// after adding its accumulated `ΔD`, falls outside `[0, 2^plaintext_bits)`.
    /// Honest operation never triggers this (the running `ΔD` telescopes to
    /// `current − pinned`, both in range); it is a loud integrity check on a
    /// corrupt or inconsistent delta/response, never a returned wrong value.
    CellOutOfRange {
        /// Segment whose decoded row went out of range.
        segment: usize,
        /// Row within the segment.
        row: u32,
        /// Cell offset within the row.
        offset: u16,
    },
    /// Forwarded [`IkpirError`] from a server call. Present for ergonomic
    /// `?` propagation in synchronous in-process composition.
    Server(IkpirError),
}

impl From<IkpirError> for IkpirClientError {
    fn from(e: IkpirError) -> Self {
        Self::Server(e)
    }
}

impl fmt::Display for IkpirClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleDelta { expected, got } => {
                write!(f, "stale delta: expected epoch {expected}, got {got}")
            }
            Self::FutureDelta { expected, got } => {
                write!(
                    f,
                    "future delta: expected epoch {expected}, got {got} \
                     (missed at least one update; recover via reset_from)"
                )
            }
            Self::EpochMismatch { client, response } => {
                write!(
                    f,
                    "epoch mismatch: client is at epoch {client}, response carries epoch {response}"
                )
            }
            Self::MalformedBundle => {
                write!(
                    f,
                    "malformed bundle: params, segment count, or row width does not match client parameters"
                )
            }
            Self::WrongUpdateMode { expected, actual } => {
                write!(
                    f,
                    "wrong update mode: this method requires {expected:?}, but the client is in {actual:?}"
                )
            }
            Self::CellOutOfRange {
                segment,
                row,
                offset,
            } => {
                write!(
                    f,
                    "cell out of range after rewind: segment {segment}, row {row}, offset {offset} \
                     (corrupt or inconsistent delta/response)"
                )
            }
            Self::Server(e) => write!(f, "server error: {e}"),
        }
    }
}

impl std::error::Error for IkpirClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Server(e) => Some(e),
            _ => None,
        }
    }
}
