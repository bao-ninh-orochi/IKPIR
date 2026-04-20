//! Server setup: build the segmented cuckoo map, derive per-segment LWE
//! matrices, and produce the hint artefacts published to the client.
//!
//! # Dataflow
//!
//! ```text
//! keys + values  ──►  SegmentedCuckooMap  ──►  D_j (seg × row_elems_bucket)  ∀j
//!
//! seed_μ  ──►  derive_segment_seeds  ──►  seeds[j]
//!                                              │
//!                  Matrix::from_seed(seeds[j]) ──►  A_j (LWE_DIMENSION × seg)
//!                                              │
//!                                    H_j = A_j · D_j   ──►  published to client
//! ```

use std::collections::HashMap;

use ikpir_common::cuckoo_store::{MapError, SegmentedCuckooMap, DEFAULT_MAX_KICKS};
use ikpir_common::hash::derive_segment_seeds;
use ikpir_common::matrix::{Matrix, MatrixError};
use ikpir_common::params::{Arity, FilterParams, ParamError, LWE_DIMENSION, SEED_LEN};
use segmented_cuckoo_filter::{Segmented3aryScheme, Segmented4aryScheme, SegmentedScheme};

/// Bundle of values produced at setup time.
///
/// Consumed by `lib.rs::Server::setup` to build the `Server` struct and
/// serialise the wire artefacts for the client.
pub(crate) struct SetupArtifacts {
    /// 32-byte public seed `μ`.
    pub seed_mu: [u8; SEED_LEN],
    /// `k` hint matrices `H_j`, shape `LWE_DIMENSION × row_elems_bucket`.
    /// Sent to the client via `batch_hint_to_bytes`.
    pub hints: Vec<Matrix>,
    /// `k` database matrices `D_j`, shape `seg × row_elems_bucket`.
    /// Retained by the server.
    pub databases: Vec<Matrix>,
    /// `k` public matrices `A_j`, shape `LWE_DIMENSION × seg`.
    /// Retained by the server for incremental updates.
    pub a_segs: Vec<Matrix>,
    /// The live `(key, value)` map for insert/delete operations.
    pub map: crate::SegmentedMap,
    /// Filter parameters.
    pub params: FilterParams,
}

/// Error returned by [`setup`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupError {
    /// Parameter selection or arity error.
    Params(ParamError),
    /// Cuckoo-map construction or insert failed.
    Map(MapError),
    /// Matrix construction or multiplication failed.
    Matrix(MatrixError),
}

impl From<ParamError> for SetupError {
    fn from(e: ParamError) -> Self {
        Self::Params(e)
    }
}
impl From<MapError> for SetupError {
    fn from(e: MapError) -> Self {
        Self::Map(e)
    }
}
impl From<MatrixError> for SetupError {
    fn from(e: MatrixError) -> Self {
        Self::Matrix(e)
    }
}

impl core::fmt::Display for SetupError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Params(e) => write!(f, "parameter error: {e}"),
            Self::Map(e) => write!(f, "cuckoo map error: {e}"),
            Self::Matrix(e) => write!(f, "matrix error: {e}"),
        }
    }
}

impl std::error::Error for SetupError {}

/// Build a [`crate::SegmentedMap`] appropriate for `params.arity`.
fn build_map(params: &FilterParams, rng_seed: [u8; 32]) -> Result<crate::SegmentedMap, SetupError> {
    let b = params.slots_per_bucket();
    match params.arity {
        Arity::Segmented2 => {
            let scheme = SegmentedScheme { half: params.seg() };
            let m =
                SegmentedCuckooMap::new(scheme, params.clone(), b, DEFAULT_MAX_KICKS, rng_seed)?;
            Ok(crate::SegmentedMap::Seg2(m))
        }
        Arity::Segmented3 => {
            let scheme = Segmented3aryScheme {
                segment_size: params.seg(),
            };
            let m =
                SegmentedCuckooMap::new(scheme, params.clone(), b, DEFAULT_MAX_KICKS, rng_seed)?;
            Ok(crate::SegmentedMap::Seg3(m))
        }
        Arity::Segmented4 => {
            let scheme = Segmented4aryScheme {
                segment_size: params.seg(),
            };
            let m =
                SegmentedCuckooMap::new(scheme, params.clone(), b, DEFAULT_MAX_KICKS, rng_seed)?;
            Ok(crate::SegmentedMap::Seg4(m))
        }
        _ => Err(SetupError::Params(ParamError::ArityNotSupportedForPir)),
    }
}

/// Run the per-segment server setup protocol.
///
/// # Errors
///
/// * [`SetupError::Params`] for unsupported arities.
/// * [`SetupError::Map`] if the cuckoo map is full or a value length mismatches.
/// * [`SetupError::Matrix`] on dimension overflow.
pub(crate) fn setup<K, V>(
    params: &FilterParams,
    seed_mu: &[u8; SEED_LEN],
    entries: &HashMap<K, V>,
) -> Result<SetupArtifacts, SetupError>
where
    K: AsRef<[u8]> + Eq + std::hash::Hash,
    V: AsRef<[u8]>,
{
    // Build map and insert all entries.
    let mut map = build_map(params, *seed_mu)?;
    for (key, value) in entries {
        map.insert(key.as_ref(), value.as_ref())?;
    }

    // Materialise k per-segment D matrices.
    let d_segs: Vec<Matrix> = map.to_matrices()?;
    let k = params.degree() as usize;
    debug_assert_eq!(d_segs.len(), k);

    // Derive k per-segment seeds and build A matrices.
    let seeds = derive_segment_seeds(seed_mu, params.degree());
    let seg = params.seg();
    let mut a_segs = Vec::with_capacity(k);
    let mut h_segs = Vec::with_capacity(k);

    for j in 0..k {
        let a = Matrix::from_seed(LWE_DIMENSION as u32, seg, &seeds[j])?;
        let h = a.mul(&d_segs[j])?;
        a_segs.push(a);
        h_segs.push(h);
    }

    Ok(SetupArtifacts {
        seed_mu: *seed_mu,
        hints: h_segs,
        databases: d_segs,
        a_segs,
        map,
        params: params.clone(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ikpir_common::params::{Arity, FilterParams, LWE_DIMENSION};
    use ikpir_common::serialization::{batch_hint_from_bytes, batch_hint_to_bytes};

    use crate::Server;

    fn build_db(num: u32, value_len: u32) -> HashMap<Vec<u8>, Vec<u8>> {
        (0..num)
            .map(|i| {
                let key = format!("key-{i}").into_bytes();
                let value: Vec<u8> = (0..value_len as usize)
                    .map(|j| ((i as usize).wrapping_add(j)) as u8)
                    .collect();
                (key, value)
            })
            .collect()
    }

    #[test]
    fn setup_produces_k_matrices_of_expected_shape() {
        let params = FilterParams::recommended(Arity::Segmented4, 64, 8).expect("params");
        let db = build_db(64, 8);
        let (server, _, _) = Server::setup(&params, &[1u8; 32], &db).expect("setup");

        let k = params.degree() as usize;
        let seg = params.seg() as usize;
        let row_elems_bucket = params.row_elems_bucket() as usize;

        assert_eq!(k, 4);
        for j in 0..k {
            assert_eq!(
                server.segment_a(j).rows() as usize,
                LWE_DIMENSION,
                "A_{j} rows"
            );
            assert_eq!(server.segment_a(j).cols() as usize, seg, "A_{j} cols");
            assert_eq!(server.segment_d(j).rows() as usize, seg, "D_{j} rows");
            assert_eq!(
                server.segment_d(j).cols() as usize,
                row_elems_bucket,
                "D_{j} cols"
            );
            assert_eq!(
                server.segment_h(j).rows() as usize,
                LWE_DIMENSION,
                "H_{j} rows"
            );
            assert_eq!(
                server.segment_h(j).cols() as usize,
                row_elems_bucket,
                "H_{j} cols"
            );
        }
    }

    #[test]
    fn setup_h_equals_a_times_d_per_segment() {
        let params = FilterParams::recommended(Arity::Segmented4, 64, 8).expect("params");
        let db = build_db(64, 8);
        let (server, _, _) = Server::setup(&params, &[2u8; 32], &db).expect("setup");
        let k = params.degree() as usize;
        for j in 0..k {
            let expected = server
                .segment_a(j)
                .mul(server.segment_d(j))
                .expect("recompute");
            assert_eq!(server.segment_h(j), &expected, "H_{j} != A_{j}·D_{j}");
        }
    }

    #[test]
    fn setup_round_trip_hint_bytes() {
        let params = FilterParams::recommended(Arity::Segmented4, 64, 8).expect("params");
        let db = build_db(64, 8);
        let (server, hint_bytes, _) = Server::setup(&params, &[3u8; 32], &db).expect("setup");
        let k = params.degree();
        let hints = batch_hint_from_bytes(&hint_bytes, k).expect("parse hints");
        assert_eq!(hints.len(), k as usize);
        for (j, hint) in hints.iter().enumerate() {
            assert_eq!(hint, server.segment_h(j), "hint {j} round-trip mismatch");
        }
        // Re-serialise and check equality.
        let h_refs: Vec<_> = (0..k as usize)
            .map(|j| server.segment_h(j).clone())
            .collect();
        let re_bytes = batch_hint_to_bytes(&h_refs);
        assert_eq!(hint_bytes, re_bytes);
    }

    #[test]
    fn setup_fails_cleanly_on_map_full() {
        // Use tiny params and try to insert far too many entries.
        let params = FilterParams::recommended(Arity::Segmented4, 4, 4).expect("params");
        // Build 200 entries — far exceeds the 4-item capacity.
        let db = build_db(200, 4);
        let err = Server::setup(&params, &[0u8; 32], &db);
        assert!(err.is_err(), "expected setup to fail on oversized db");
    }
}
