#![warn(missing_docs)]
//! Client library for Incremental-Keyword-PIR.
//!
//! # End-to-end flow
//!
//! ```text
//!   Client::new(seed_μ, hint_bytes, params_bytes)  →  Client state (k A_j, k H_j, params)
//!   Client::query(key, rng)                         →  (QueryState, query_bytes)
//!   ── send query_bytes to server ──
//!   Server::respond(query_bytes)                    →  response_bytes
//!   ── receive response_bytes ──
//!   Client::decrypt(state, response_bytes)          →  Ok(Some(value)) | Ok(None) | Err
//! ```
//!
//! `Ok(None)` indicates the keyword was never inserted into the server's
//! database (legitimate filter miss).

pub mod decrypt;
pub mod query;
pub mod setup;

use ikpir_common::params::{FilterParams, SEED_LEN};
use rand::{CryptoRng, RngCore};

pub use decrypt::{decrypt, DecryptError};
pub use query::{QueryError, QueryState};
pub use setup::{ClientSetup, SetupError};

/// A configured PIR client.
///
/// Wraps [`ClientSetup`] and exposes the three protocol operations as methods.
#[derive(Debug, Clone)]
pub struct Client {
    /// Preprocessed setup state (k `A_j` and `H_j` matrices).
    setup: ClientSetup,
    /// Master seed `μ` — kept for API symmetry; not transmitted on queries.
    seed_mu: [u8; SEED_LEN],
}

impl Client {
    /// Build a client from the server's published `(seed_μ, hint_bytes, params_bytes)`.
    ///
    /// # Errors
    ///
    /// * [`SetupError::Serde`] — malformed bytes.
    /// * [`SetupError::DimensionMismatch`] — hint dimensions inconsistent with params.
    pub fn new(
        seed_mu: &[u8; SEED_LEN],
        hint_bytes: &[u8],
        params_bytes: &[u8],
    ) -> Result<Self, SetupError> {
        let s = setup::setup(seed_mu, hint_bytes, params_bytes)?;
        Ok(Self {
            setup: s,
            seed_mu: *seed_mu,
        })
    }

    /// Active filter parameters.
    pub fn params(&self) -> &FilterParams {
        &self.setup.params
    }

    /// Build a PIR query for `key`. Returns `(state, query_bytes)`.
    ///
    /// `query_bytes` is sent to the server; `state` is retained until
    /// [`Client::decrypt`] is called with the server's response.
    pub fn query<R: RngCore + CryptoRng>(
        &self,
        key: &[u8],
        rng: &mut R,
    ) -> Result<(QueryState, Vec<u8>), QueryError> {
        query::query(&self.setup, &self.seed_mu, key, rng)
    }

    /// Decrypt a server response for `state`.
    ///
    /// Returns:
    ///   * `Ok(Some(value))` — the stored value.
    ///   * `Ok(None)` — keyword was never inserted (filter miss).
    ///   * `Err(DecryptError)` — wire-format or noise-overflow error.
    pub fn decrypt(
        &self,
        state: &QueryState,
        response_bytes: &[u8],
    ) -> Result<Option<Vec<u8>>, DecryptError> {
        decrypt::decrypt(state, response_bytes)
    }
}
