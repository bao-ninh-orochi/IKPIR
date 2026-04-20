//! Per-segment PIR response: compute `b_j · D_j` for each segment `j`.

use ikpir_common::matrix::{Matrix, MatrixError};

/// Compute the per-segment response vectors: `r_j = b_j · D_j` for each `j`.
///
/// # Arguments
///
/// * `databases` — `k` references to per-segment `D_j` matrices
///   (`seg × row_elems_bucket`).
/// * `queries` — `k` query vectors `b_j`, each of length `seg`.
///
/// # Returns
///
/// `k` response vectors `r_j`, each of length `row_elems_bucket`.
///
/// # Errors
///
/// [`MatrixError::ShapeMismatch`] if any `queries[j].len() != databases[j].rows()`.
pub(crate) fn respond(
    databases: &[&Matrix],
    queries: &[Vec<u32>],
) -> Result<Vec<Vec<u32>>, MatrixError> {
    debug_assert_eq!(databases.len(), queries.len());
    let mut out = Vec::with_capacity(databases.len());
    for (d, q) in databases.iter().zip(queries.iter()) {
        out.push(d.left_mul_row_vec(q)?);
    }
    Ok(out)
}
