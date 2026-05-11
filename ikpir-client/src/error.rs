use ikpir_server::IkpirError;

/// Errors returned by [`IkpirClient`](crate::IkpirClient) methods.
///
/// Variants split into three categories:
///
/// - **Epoch-coherence failures** — [`StaleDelta`](Self::StaleDelta),
///   [`FutureDelta`](Self::FutureDelta), [`EpochMismatch`](Self::EpochMismatch).
/// - **Bundle-shape failures** — [`MalformedBundle`](Self::MalformedBundle).
/// - **Forwarded server errors** — [`Server`](Self::Server), present so a
///   single `?` operator works in synchronous in-process composition.
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
