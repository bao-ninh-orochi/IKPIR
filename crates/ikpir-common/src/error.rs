//! [`IkpirError`] — shared server error enum.
//!
//! # Purpose
//!
//! Single error type returned by `IkpirServer` methods, shared here so
//! `IkpirClient` can wrap it in `IkpirClientError::Server(IkpirError)`
//! without depending on `ikpir-server` at compile time.

use std::fmt;

/// Errors returned by `IkpirServer` methods.
///
/// All variants are recoverable in the sense that they leave the server's
/// internal state consistent: the mutation log is drained on insert/update
/// failure to prevent leaked deltas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IkpirError {
    /// The query carries an epoch that does not match the server's current
    /// epoch. Returned by `IkpirServer::answer`.
    StaleEpoch {
        /// Current server epoch.
        expected: u64,
        /// Epoch the client query announced.
        got: u64,
    },
    /// The query bundle has the wrong number of segments for the active arity.
    /// Returned by `IkpirServer::answer`.
    MalformedQuery,
    /// The cuckoo kicking budget was exhausted on `insert`. The store and
    /// hint are rolled back to the pre-insert state.
    ///
    /// Recovery requires rebuilding the server with a larger `num_buckets`
    /// from an externally-held `(key, value)` map — the SCF store discards
    /// keys after hashing, so IKPIR cannot self-recover. Clients then call
    /// `IkpirClient::reset_from` on the new bundle. See `IkpirServer::insert`
    /// for the full recovery recipe.
    TableFull,
    /// `update` or `delete` was called with a key that is not present.
    NotFound,
    /// Caller-provided value width does not match the store's `value_size_in_bytes()`.
    InvalidInput,
}

impl fmt::Display for IkpirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleEpoch { expected, got } => {
                write!(
                    f,
                    "stale query epoch: server is at epoch {expected}, query announced {got}"
                )
            }
            Self::MalformedQuery => {
                write!(
                    f,
                    "malformed query: wrong segment count for the active arity"
                )
            }
            Self::TableFull => write!(f, "table is full: cuckoo kicking budget exhausted"),
            Self::NotFound => write!(f, "key not found"),
            Self::InvalidInput => {
                write!(
                    f,
                    "invalid input: value width does not match the store's value size"
                )
            }
        }
    }
}

impl std::error::Error for IkpirError {}
