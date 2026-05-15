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

use ikpir_common::IkpirError;

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
    /// Either `apply_delta` or `decode` was given a bundle whose segment count
    /// or row width does not match the cached `params.arity()` /
    /// `bucket_size × cells_per_slot`.
    MalformedBundle,
    /// Forwarded [`IkpirError`] from a server call. Present for ergonomic
    /// `?` propagation in synchronous in-process composition.
    Server(IkpirError),
}

impl From<IkpirError> for IkpirClientError {
    fn from(e: IkpirError) -> Self { Self::Server(e) }
}
