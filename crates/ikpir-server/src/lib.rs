#![warn(missing_docs)]
//! Server library for Incremental-Keyword-PIR.
//!
//! # Per-segment PIR protocol
//!
//! The server holds `k = params.degree()` segment triples `(A_j, D_j, H_j)`:
//!   * `A_j` (`LWE_DIMENSION × seg`) — public matrix derived deterministically
//!     from per-segment seed `seeds[j]`, itself derived from the master seed `μ`;
//!   * `D_j` (`seg × row_elems_bucket`) — encoded cuckoo-map bucket rows for
//!     segment `j` with fingerprint + value per slot;
//!   * `H_j = A_j · D_j` (`LWE_DIMENSION × row_elems_bucket`) — hint published to
//!     the client.
//!
//! The protocol round-trip:
//!
//! ```text
//!   Client                                          Server
//!     │  ── Client::new(seed_μ, hints, params) ──►  │
//!     │  ── query_bytes (k segment vectors) ───────► │
//!     │                                              │ Server::respond
//!     │ ◄──── response_bytes (k segment vectors) ───┤
//!     │  Client::decrypt → Option<value>
//! ```
//!
//! # Incremental updates
//!
//! [`Server::insert`], [`Server::delete`], and [`Server::update_value`] keep
//! `H_j = A_j · D_j` after every mutation via [`update::apply`].
//!
//! # Memory
//!
//! Total `4 · k · (LWE_DIMENSION · (seg + row_elems_bucket) + seg · row_elems_bucket)`
//! bytes — the sum of the `A_j`, `D_j`, and `H_j` matrices across all `k` segments.
//! Since `k · seg = num_buckets`, the `A_j` and `D_j` portion matches the old
//! single-segment design; the `H_j` portion is `k`× larger (one hint per segment).

pub mod respond;
pub mod setup;
pub mod update;

use std::collections::HashMap;

use ikpir_common::cuckoo_store::{MapError, PerSegmentRow, SegmentedCuckooMap, UpdateBatch};
use ikpir_common::matrix::{Matrix, MatrixError};
use ikpir_common::params::{FilterParams, SEED_LEN};
use ikpir_common::serialization::{
    batch_hint_to_bytes, batch_query_from_bytes, batch_response_to_bytes, params_to_bytes,
    SerdeError,
};
use segmented_cuckoo_filter::{Segmented3aryScheme, Segmented4aryScheme, SegmentedScheme};

/// LWE matrix triple for one segment.
#[derive(Clone)]
pub(crate) struct PerSegmentState {
    /// `A_j`: `LWE_DIMENSION × seg` — public matrix.
    pub a: Matrix,
    /// `D_j`: `seg × row_elems_bucket` — encoded database.
    pub d: Matrix,
    /// `H_j = A_j · D_j`: `LWE_DIMENSION × row_elems_bucket` — hint.
    pub h: Matrix,
}

/// Enum dispatch over the three supported segmented cuckoo map arities.
///
/// Avoids boxing / dynamic dispatch on the insert/delete paths.
#[derive(Clone)]
pub(crate) enum SegmentedMap {
    /// Segmented-2ary map.
    Seg2(SegmentedCuckooMap<SegmentedScheme>),
    /// Segmented-3ary map.
    Seg3(SegmentedCuckooMap<Segmented3aryScheme>),
    /// Segmented-4ary map.
    Seg4(SegmentedCuckooMap<Segmented4aryScheme>),
}

impl SegmentedMap {
    pub(crate) fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<UpdateBatch, MapError> {
        match self {
            Self::Seg2(m) => m.insert(key, value),
            Self::Seg3(m) => m.insert(key, value),
            Self::Seg4(m) => m.insert(key, value),
        }
    }

    pub(crate) fn delete(&mut self, key: &[u8]) -> Result<UpdateBatch, MapError> {
        match self {
            Self::Seg2(m) => m.delete(key),
            Self::Seg3(m) => m.delete(key),
            Self::Seg4(m) => m.delete(key),
        }
    }

    pub(crate) fn update_value(
        &mut self,
        key: &[u8],
        new_value: &[u8],
    ) -> Result<UpdateBatch, MapError> {
        match self {
            Self::Seg2(m) => m.update_value(key, new_value),
            Self::Seg3(m) => m.update_value(key, new_value),
            Self::Seg4(m) => m.update_value(key, new_value),
        }
    }

    pub(crate) fn to_matrices(&self) -> Result<Vec<Matrix>, MapError> {
        match self {
            Self::Seg2(m) => m.to_matrices(),
            Self::Seg3(m) => m.to_matrices(),
            Self::Seg4(m) => m.to_matrices(),
        }
    }
}

/// Server state required to answer PIR queries and apply incremental updates.
///
/// Construct via [`Server::setup`]. The server holds `k` triples
/// `(A_j, D_j, H_j)` indexed by segment `j ∈ [0, k)`.
///
/// # Memory
///
/// Approximately `4 · k · (LWE_DIMENSION · (seg + row_elems_bucket) + seg · row_elems_bucket)`
/// bytes. For Segmented-4ary with 1000 items and 32-byte values (`k=4`,
/// `seg=512`, `row_elems_bucket=104`) this is roughly 18 MB.
#[derive(Clone)]
pub struct Server {
    /// Filter parameters.
    params: FilterParams,
    /// Master seed `μ` — published to the client at setup.
    seed_mu: [u8; SEED_LEN],
    /// Per-segment LWE state (`k` triples).
    segments: Vec<PerSegmentState>,
    /// Live `(key, value)` map for insert/delete mutations.
    map: SegmentedMap,
}

impl Server {
    /// Run server setup: populate the cuckoo map, derive per-segment matrices,
    /// and return `(Server, hint_bytes, params_bytes)`.
    ///
    /// # Arguments
    ///
    /// * `params` — filter parameters (use [`FilterParams::recommended`]).
    /// * `seed_mu` — 32-byte public seed; published to the client.
    /// * `entries` — initial `(key, value)` pairs. All values must satisfy
    ///   `value.len() == params.value_len`.
    ///
    /// # Errors
    ///
    /// Returns [`SetupError`] if the map is full, a value length mismatches, or
    /// matrix dimensions overflow.
    pub fn setup<K, V>(
        params: &FilterParams,
        seed_mu: &[u8; SEED_LEN],
        entries: &HashMap<K, V>,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), SetupError>
    where
        K: AsRef<[u8]> + Eq + std::hash::Hash,
        V: AsRef<[u8]>,
    {
        let art = setup::setup(params, seed_mu, entries)?;
        let hint_bytes = batch_hint_to_bytes(&art.hints);
        let params_bytes = params_to_bytes(&art.params);
        let segments: Vec<PerSegmentState> = art
            .a_segs
            .into_iter()
            .zip(art.databases)
            .zip(art.hints)
            .map(|((a, d), h)| PerSegmentState { a, d, h })
            .collect();
        let server = Self {
            params: art.params,
            seed_mu: art.seed_mu,
            segments,
            map: art.map,
        };
        Ok((server, hint_bytes, params_bytes))
    }

    /// Filter parameters in use.
    pub fn params(&self) -> &FilterParams {
        &self.params
    }

    /// Master seed `μ`.
    pub fn seed_mu(&self) -> &[u8; SEED_LEN] {
        &self.seed_mu
    }

    /// Public matrix `A_j` for segment `j`.
    pub fn segment_a(&self, j: usize) -> &Matrix {
        &self.segments[j].a
    }

    /// Database matrix `D_j` for segment `j`.
    pub fn segment_d(&self, j: usize) -> &Matrix {
        &self.segments[j].d
    }

    /// Hint matrix `H_j` for segment `j`.
    pub fn segment_h(&self, j: usize) -> &Matrix {
        &self.segments[j].h
    }

    /// Answer a client's batched PIR query.
    ///
    /// Parses `k` query vectors from `query_bytes`, computes `r_j = b_j · D_j`
    /// for each segment, and serialises the `k` response vectors.
    ///
    /// # Errors
    ///
    /// * [`RespondError::Serde`] — malformed query bytes.
    /// * [`RespondError::Matrix`] — shape inconsistency (should not occur on a
    ///   correctly constructed server).
    pub fn respond(&self, query_bytes: &[u8]) -> Result<Vec<u8>, RespondError> {
        let k = self.params.degree();
        let seg = self.params.seg();
        let per_seg_b = batch_query_from_bytes(query_bytes, k, seg)?;
        let d_refs: Vec<&Matrix> = self.segments.iter().map(|s| &s.d).collect();
        let per_seg_r = respond::respond(&d_refs, &per_seg_b)?;
        Ok(batch_response_to_bytes(&per_seg_r))
    }

    /// Insert `(key, value)` and update all per-segment hints.
    ///
    /// # Errors
    ///
    /// * [`InsertError::Map`] — key already present or cuckoo map full.
    /// * [`InsertError::Update`] — matrix shape error (invariant: does not occur
    ///   with well-formed params).
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), InsertError> {
        let batch = self.map.insert(key, value).map_err(InsertError::Map)?;
        update::apply(&mut self.segments, &self.params, &batch.rows).map_err(InsertError::Update)
    }

    /// Delete `key` and update all per-segment hints.
    ///
    /// # Errors
    ///
    /// * [`DeleteError::Map`] — key not found.
    /// * [`DeleteError::Update`] — matrix shape error.
    pub fn delete(&mut self, key: &[u8]) -> Result<(), DeleteError> {
        let batch = self.map.delete(key).map_err(DeleteError::Map)?;
        update::apply(&mut self.segments, &self.params, &batch.rows).map_err(DeleteError::Update)
    }

    /// Update the value associated with `key` to `new_value`.
    ///
    /// In-place replacement: keeps the existing slot (same `tag`, same segment)
    /// and overwrites only the value cell. Emits a single `PerSegmentRow` —
    /// half the matrix work of a delete-then-insert round-trip.
    ///
    /// # Errors
    ///
    /// * [`UpdateKwError::Map`] — key not found, or `new_value.len() != value_len`.
    /// * [`UpdateKwError::Update`] — matrix shape error (should not occur with
    ///   well-formed params).
    pub fn update_value(&mut self, key: &[u8], new_value: &[u8]) -> Result<(), UpdateKwError> {
        let batch = self
            .map
            .update_value(key, new_value)
            .map_err(UpdateKwError::Map)?;
        update::apply(&mut self.segments, &self.params, &batch.rows)
            .map_err(UpdateKwError::Update)
    }

    /// Apply a batch of pre-encoded per-segment row updates.
    ///
    /// Low-level entry point for callers that drive updates externally. For
    /// keyword-level mutations prefer [`insert`][Self::insert] /
    /// [`delete`][Self::delete] / [`update_value`][Self::update_value].
    ///
    /// # Errors
    ///
    /// See [`update::UpdateError`].
    pub fn apply_row_updates(&mut self, rows: &[PerSegmentRow]) -> Result<(), update::UpdateError> {
        update::apply(&mut self.segments, &self.params, rows)
    }
}

// ── Error types ───────────────────────────────────────────────────────────────

/// Error returned by [`Server::setup`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupError {
    /// Encoding, map, or matrix failure during setup.
    Setup(setup::SetupError),
}

impl From<setup::SetupError> for SetupError {
    fn from(e: setup::SetupError) -> Self {
        Self::Setup(e)
    }
}

impl core::fmt::Display for SetupError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Setup(e) => write!(f, "setup: {e}"),
        }
    }
}

impl std::error::Error for SetupError {}

/// Error returned by [`Server::respond`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RespondError {
    /// Wire-format deserialization error.
    Serde(SerdeError),
    /// Matrix shape mismatch.
    Matrix(MatrixError),
}

impl From<SerdeError> for RespondError {
    fn from(e: SerdeError) -> Self {
        Self::Serde(e)
    }
}
impl From<MatrixError> for RespondError {
    fn from(e: MatrixError) -> Self {
        Self::Matrix(e)
    }
}

impl core::fmt::Display for RespondError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Serde(e) => write!(f, "wire format error: {e}"),
            Self::Matrix(e) => write!(f, "matrix error: {e}"),
        }
    }
}

impl std::error::Error for RespondError {}

/// Error returned by [`Server::insert`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError {
    /// Cuckoo-map failure (duplicate key or map full).
    Map(MapError),
    /// Matrix-update failure.
    Update(update::UpdateError),
}

impl core::fmt::Display for InsertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Map(e) => write!(f, "map error: {e}"),
            Self::Update(e) => write!(f, "update error: {e}"),
        }
    }
}

impl std::error::Error for InsertError {}

/// Error returned by [`Server::delete`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeleteError {
    /// Cuckoo-map failure (key not found).
    Map(MapError),
    /// Matrix-update failure.
    Update(update::UpdateError),
}

impl core::fmt::Display for DeleteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Map(e) => write!(f, "map error: {e}"),
            Self::Update(e) => write!(f, "update error: {e}"),
        }
    }
}

impl std::error::Error for DeleteError {}

/// Error returned by [`Server::update_value`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateKwError {
    /// Cuckoo-map failure.
    Map(MapError),
    /// Matrix-update failure.
    Update(update::UpdateError),
}

impl core::fmt::Display for UpdateKwError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Map(e) => write!(f, "map error: {e}"),
            Self::Update(e) => write!(f, "update error: {e}"),
        }
    }
}

impl std::error::Error for UpdateKwError {}
