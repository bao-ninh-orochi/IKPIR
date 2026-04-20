//! Build per-segment LWE queries for a keyword.
//!
//! For each segment `j ∈ [0, k)`:
//!
//! ```text
//!   s    ← ternary^{LWE_DIMENSION}           (client secret, shared across segments)
//!   e_j  ← ternary^{seg}                     (per-segment error)
//!   b_j  = sᵀ · A_j  +  e_j  +  Δ · u_j     (u_j = indicator at local_j)
//!   c_j  = sᵀ · H_j                          (hint precomputation)
//! ```
//!
//! where `local_j = idx[j] − j · seg` is the segment-local candidate index.
//!
//! `c_j` is retained in [`QueryState`] until the response arrives; the client
//! subtracts it from `r_j` before rounding.
//!
//! # Non-goals
//!
//! No zeroisation of `s` on drop. For a deployed system, wrap secrets in a
//! `zeroize`-aware container.

use ikpir_common::hash::candidates;
use ikpir_common::lwe::sample_ternary_vec;
use ikpir_common::matrix::{vec_add_assign, MatrixError};
use ikpir_common::params::{FilterParams, LWE_DIMENSION, SEED_LEN};
use ikpir_common::serialization::batch_query_to_bytes;
use rand::{CryptoRng, RngCore};

use crate::setup::ClientSetup;

/// Per-query secret state retained by the client until the response arrives.
#[derive(Debug, Clone)]
pub struct QueryState {
    /// The keyword this query was built for.
    pub key: Vec<u8>,
    /// SCF fingerprint tag for the keyword (used for slot matching on decrypt).
    pub tag: u32,
    /// `c_j = sᵀ · H_j` for each segment, length `row_elems_bucket` each.
    pub c: Vec<Vec<u32>>,
    /// Echo of the active filter parameters.
    pub params: FilterParams,
}

/// Errors raised during query construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryError {
    /// Matrix dimension mismatch.
    Matrix(MatrixError),
}

impl From<MatrixError> for QueryError {
    fn from(e: MatrixError) -> Self {
        Self::Matrix(e)
    }
}

impl core::fmt::Display for QueryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Matrix(e) => write!(f, "matrix error: {e}"),
        }
    }
}

impl std::error::Error for QueryError {}

/// Build `k` per-segment PIR queries for `key`.
///
/// Returns `(QueryState, query_bytes)`:
///   * `state` must be passed to [`crate::decrypt::decrypt`] together with the
///     server's response bytes.
///   * `query_bytes` is the wire payload — `batch_query_to_bytes(&[b_0, …, b_{k-1}])`,
///     total `4 · k · seg` bytes.
///
/// # Notes
///
/// `seed_mu` is kept in the signature for API symmetry; the v2 wire format does
/// NOT include a seed prefix on the query bytes (the server already holds `μ`).
pub fn query<R: RngCore + CryptoRng>(
    setup: &ClientSetup,
    _seed_mu: &[u8; SEED_LEN],
    key: &[u8],
    rng: &mut R,
) -> Result<(QueryState, Vec<u8>), QueryError> {
    let params = setup.params.clone();
    let k = params.degree() as usize;
    let seg = params.seg() as usize;
    let scale = params.scale();

    // 1. Sample shared LWE secret s (length LWE_DIMENSION).
    let s = sample_ternary_vec(rng, LWE_DIMENSION);

    // 2. Derive keyword footprint (tag + k candidate indices).
    let foot = candidates(key, &params);

    let mut b_segs = Vec::with_capacity(k);
    let mut c_segs = Vec::with_capacity(k);

    for j in 0..k {
        // 3. Sample per-segment error e_j (length seg).
        let e_j = sample_ternary_vec(rng, seg);

        // 4. b_j = sᵀ · A_j  +  e_j  +  Δ · u_j
        let mut b_j = setup.a[j].left_mul_row_vec(&s)?; // sᵀ · A_j, length seg
        vec_add_assign(&mut b_j, &e_j); // + e_j

        // u_j has a single '1' at the segment-local candidate index.
        let local_j = (foot.indices[j] - (j as u32) * (seg as u32)) as usize;
        b_j[local_j] = b_j[local_j].wrapping_add(scale); // + Δ · u_j

        // 5. c_j = sᵀ · H_j (length row_elems_bucket).
        let c_j = setup.hint[j].left_mul_row_vec(&s)?;

        b_segs.push(b_j);
        c_segs.push(c_j);
    }

    let bytes = batch_query_to_bytes(&b_segs);
    let state = QueryState {
        key: key.to_vec(),
        tag: foot.tag,
        c: c_segs,
        params,
    };
    Ok((state, bytes))
}
