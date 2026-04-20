//! Client setup: consume the server's `k` hint matrices and params; re-derive
//! the `k` public `A_j` matrices from the shared seed.

use ikpir_common::hash::derive_segment_seeds;
use ikpir_common::matrix::Matrix;
use ikpir_common::params::{FilterParams, LWE_DIMENSION, SEED_LEN};
use ikpir_common::serialization::{batch_hint_from_bytes, params_from_bytes, SerdeError};

/// Preprocessed state a client needs to issue PIR queries.
#[derive(Debug, Clone)]
pub struct ClientSetup {
    /// Filter parameters (received from server).
    pub params: FilterParams,
    /// `k` public matrices `A_j` (`LWE_DIMENSION × seg`), re-derived locally.
    pub a: Vec<Matrix>,
    /// `k` hint matrices `H_j = A_j · D_j` (`LWE_DIMENSION × row_elems_bucket`).
    pub hint: Vec<Matrix>,
}

/// Errors raised during client setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupError {
    /// Deserialization failure.
    Serde(SerdeError),
    /// A hint matrix's dimensions are inconsistent with the declared parameters.
    DimensionMismatch,
}

impl From<SerdeError> for SetupError {
    fn from(e: SerdeError) -> Self {
        Self::Serde(e)
    }
}

impl core::fmt::Display for SetupError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Serde(e) => write!(f, "wire format error: {e}"),
            Self::DimensionMismatch => f.write_str("hint dimensions mismatch params"),
        }
    }
}

impl std::error::Error for SetupError {}

/// Build the client-side state from the server's published artefacts.
///
/// # Steps
///
/// 1. Parse `FilterParams` from `params_bytes`.
/// 2. Parse `k` hint matrices from `hint_bytes` via `batch_hint_from_bytes`.
/// 3. Validate each hint's shape: `LWE_DIMENSION × row_elems_bucket`.
/// 4. Derive `k` per-segment seeds from `seed_mu`.
/// 5. Build `A_j = Matrix::from_seed(LWE_DIMENSION, seg, seeds[j])`.
///
/// # Errors
///
/// * [`SetupError::Serde`] — malformed wire bytes.
/// * [`SetupError::DimensionMismatch`] — hint shape inconsistent with params.
pub fn setup(
    seed_mu: &[u8; SEED_LEN],
    hint_bytes: &[u8],
    params_bytes: &[u8],
) -> Result<ClientSetup, SetupError> {
    let params = params_from_bytes(params_bytes)?;
    let k = params.degree();
    let hints = batch_hint_from_bytes(hint_bytes, k)?;

    let row_elems_bucket = params.row_elems_bucket();
    for h in &hints {
        if h.rows() != LWE_DIMENSION as u32 || h.cols() != row_elems_bucket {
            return Err(SetupError::DimensionMismatch);
        }
    }

    let seeds = derive_segment_seeds(seed_mu, k);
    let seg = params.seg();
    let mut a_segs = Vec::with_capacity(k as usize);
    for seed in seeds.iter().take(k as usize) {
        let a = Matrix::from_seed(LWE_DIMENSION as u32, seg, seed)
            .map_err(|_| SetupError::DimensionMismatch)?;
        a_segs.push(a);
    }

    Ok(ClientSetup {
        params,
        a: a_segs,
        hint: hints,
    })
}
