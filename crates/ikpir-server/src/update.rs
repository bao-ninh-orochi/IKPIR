//! Row-level incremental matrix update for per-segment PIR hints.
//!
//! # Correctness invariant
//!
//! After any call to [`apply`], the following holds for every segment `j`:
//!
//! ```text
//! H_j == A_j · D_j
//! ```
//!
//! Each update replaces `D_j[row_local_idx, :]` with the new payload and
//! adjusts `H_j` by `A_j[:, row_local_idx] ⊗ diff`, where
//! `diff = new_payload − old_payload` (wrapping). By linearity:
//!
//! ```text
//! H_j + A_j[:, i] ⊗ diff = A_j · D_j_new
//! ```
//!
//! # Complexity
//!
//! `O(LWE_DIMENSION · t · row_elems_bucket)` multiply-adds for `t` row updates.

use core::fmt;

use ikpir_common::matrix::MatrixError;
use ikpir_common::params::FilterParams;

pub use ikpir_common::cuckoo_store::PerSegmentRow;

use crate::PerSegmentState;

/// Errors returned by [`apply`] and [`Server::apply_row_updates`][crate::Server::apply_row_updates].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UpdateError {
    /// `segment_idx >= params.degree()`.
    SegmentOutOfBounds,
    /// `row_local_idx >= params.seg()`.
    RowIndexOutOfBounds,
    /// `new_payload.len() != params.row_elems_bucket()`.
    ShapeMismatch,
    /// A payload value exceeds `2^mat_elem_bit_len - 1`.
    PayloadOutOfRange,
    /// Underlying matrix operation failed (should not occur with well-formed inputs).
    Matrix(MatrixError),
}

impl From<MatrixError> for UpdateError {
    fn from(e: MatrixError) -> Self {
        Self::Matrix(e)
    }
}

impl fmt::Display for UpdateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SegmentOutOfBounds => f.write_str("segment index out of bounds"),
            Self::RowIndexOutOfBounds => f.write_str("row_local_idx exceeds segment size"),
            Self::ShapeMismatch => {
                f.write_str("new_payload length does not match row_elems_bucket")
            }
            Self::PayloadOutOfRange => f.write_str("payload value exceeds 2^mat_elem_bit_len - 1"),
            Self::Matrix(e) => write!(f, "matrix error: {e}"),
        }
    }
}

impl std::error::Error for UpdateError {}

/// Apply a batch of per-segment row overwrites incrementally.
///
/// Maintains `H_j = A_j · D_j` for every touched segment `j` without a
/// full matrix recompute. All updates are validated before any state is
/// mutated (fail-fast, atomic within the batch).
///
/// # Arguments
///
/// * `segments` — mutable slice of `k` per-segment states.
/// * `params` — filter parameters.
/// * `updates` — ordered list of per-segment row overwrites. Multiple
///   updates for the same `(segment_idx, row_local_idx)` are applied in
///   order; the last one wins.
///
/// # Errors
///
/// Returns the first validation error; no state is modified.
#[allow(clippy::needless_range_loop)]
pub(crate) fn apply(
    segments: &mut [PerSegmentState],
    params: &FilterParams,
    updates: &[PerSegmentRow],
) -> Result<(), UpdateError> {
    let mask: u32 = if params.mat_elem_bit_len >= 32 {
        u32::MAX
    } else {
        (1u32 << params.mat_elem_bit_len) - 1
    };
    let row_elems_bucket = params.row_elems_bucket() as usize;
    let seg = params.seg();
    let degree = params.degree();

    // Validate all updates before mutating anything.
    for u in updates {
        if (u.segment_idx as u32) >= degree {
            return Err(UpdateError::SegmentOutOfBounds);
        }
        if u.row_local_idx >= seg {
            return Err(UpdateError::RowIndexOutOfBounds);
        }
        if u.new_payload.len() != row_elems_bucket {
            return Err(UpdateError::ShapeMismatch);
        }
        if u.new_payload.iter().any(|&p| p & !mask != 0) {
            return Err(UpdateError::PayloadOutOfRange);
        }
    }

    for u in updates {
        let j = u.segment_idx as usize;

        // diff = new_payload − D_j[row_local_idx, :]  (wrapping Z_{2^32})
        let diff: Vec<u32> = {
            let old_row = segments[j].d.row(u.row_local_idx)?;
            u.new_payload
                .iter()
                .zip(old_row.iter())
                .map(|(&nw, &old)| nw.wrapping_sub(old))
                .collect()
        };

        // a_col = A_j[:, row_local_idx]  (length LWE_DIMENSION)
        let a_col = segments[j].a.column(u.row_local_idx)?;
        let n = a_col.len();

        // H_j[i, :] += A_j[i, row_local_idx] * diff[:]  for each i
        for i in 0..n {
            let a_i = a_col[i];
            if a_i == 0 {
                continue;
            }
            let hint_row = segments[j].h.row_mut(i as u32)?;
            for k in 0..row_elems_bucket {
                hint_row[k] = hint_row[k].wrapping_add(a_i.wrapping_mul(diff[k]));
            }
        }

        // Overwrite D_j[row_local_idx, :]
        segments[j]
            .d
            .row_mut(u.row_local_idx)?
            .copy_from_slice(&u.new_payload);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ikpir_common::cuckoo_store::PerSegmentRow;
    use ikpir_common::params::{Arity, FilterParams};
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use crate::Server;

    use super::UpdateError;

    fn build_db(num: u32, value_len: u32) -> HashMap<Vec<u8>, Vec<u8>> {
        (0..num)
            .map(|i| {
                let key = format!("key-{i}").into_bytes();
                let value: Vec<u8> = (0..value_len as usize)
                    .map(|j| ((i as usize).wrapping_add(j.wrapping_mul(7))) as u8)
                    .collect();
                (key, value)
            })
            .collect()
    }

    fn check_invariant(server: &Server) {
        for j in 0..server.params().degree() as usize {
            let expected = server
                .segment_a(j)
                .mul(server.segment_d(j))
                .expect("recompute h");
            assert_eq!(
                server.segment_h(j),
                &expected,
                "H_{j} != A_{j}·D_{j} invariant violated"
            );
        }
    }

    #[test]
    fn apply_empty_is_noop() {
        let params = FilterParams::recommended(Arity::Segmented4, 32, 4).expect("params");
        let db = build_db(32, 4);
        let (mut server, _, _) = Server::setup(&params, &[5u8; 32], &db).expect("setup");
        let h_before: Vec<_> = (0..params.degree() as usize)
            .map(|j| server.segment_h(j).clone())
            .collect();
        let d_before: Vec<_> = (0..params.degree() as usize)
            .map(|j| server.segment_d(j).clone())
            .collect();
        server.apply_row_updates(&[]).expect("empty update");
        for j in 0..params.degree() as usize {
            assert_eq!(server.segment_h(j), &h_before[j]);
            assert_eq!(server.segment_d(j), &d_before[j]);
        }
    }

    #[test]
    fn apply_single_row_matches_full_recompute_seg4() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let params = FilterParams::recommended(Arity::Segmented4, 64, 8).expect("params");
        let db = build_db(64, 8);
        let (mut server, _, _) = Server::setup(&params, &[11u8; 32], &db).expect("setup");

        let mask: u32 = (1u32 << params.mat_elem_bit_len) - 1;
        let row_elems_bucket = params.row_elems_bucket() as usize;

        let update = PerSegmentRow {
            segment_idx: 0,
            row_local_idx: 0,
            new_payload: (0..row_elems_bucket)
                .map(|_| rng.next_u32() & mask)
                .collect(),
        };
        server.apply_row_updates(&[update]).expect("apply");
        check_invariant(&server);
    }

    #[test]
    fn apply_single_row_matches_full_recompute_seg2() {
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        let params = FilterParams::recommended(Arity::Segmented2, 64, 8).expect("params");
        let db = build_db(64, 8);
        let (mut server, _, _) = Server::setup(&params, &[22u8; 32], &db).expect("setup");

        let mask: u32 = (1u32 << params.mat_elem_bit_len) - 1;
        let row_elems_bucket = params.row_elems_bucket() as usize;
        let update = PerSegmentRow {
            segment_idx: 0,
            row_local_idx: 0,
            new_payload: (0..row_elems_bucket)
                .map(|_| rng.next_u32() & mask)
                .collect(),
        };
        server.apply_row_updates(&[update]).expect("apply");
        check_invariant(&server);
    }

    #[test]
    fn apply_single_row_matches_full_recompute_seg3() {
        let mut rng = ChaCha20Rng::from_seed([99u8; 32]);
        let params = FilterParams::recommended(Arity::Segmented3, 64, 8).expect("params");
        let db = build_db(64, 8);
        let (mut server, _, _) = Server::setup(&params, &[33u8; 32], &db).expect("setup");

        let mask: u32 = (1u32 << params.mat_elem_bit_len) - 1;
        let row_elems_bucket = params.row_elems_bucket() as usize;
        let update = PerSegmentRow {
            segment_idx: 1,
            row_local_idx: 0,
            new_payload: (0..row_elems_bucket)
                .map(|_| rng.next_u32() & mask)
                .collect(),
        };
        server.apply_row_updates(&[update]).expect("apply");
        check_invariant(&server);
    }

    #[test]
    fn apply_batch_same_segment_different_rows() {
        let mut rng = ChaCha20Rng::from_seed([17u8; 32]);
        let params = FilterParams::recommended(Arity::Segmented4, 64, 8).expect("params");
        let db = build_db(64, 8);
        let (mut server, _, _) = Server::setup(&params, &[55u8; 32], &db).expect("setup");

        let mask: u32 = (1u32 << params.mat_elem_bit_len) - 1;
        let row_elems_bucket = params.row_elems_bucket() as usize;
        let updates: Vec<PerSegmentRow> = (0..4)
            .map(|row| PerSegmentRow {
                segment_idx: 2,
                row_local_idx: row,
                new_payload: (0..row_elems_bucket)
                    .map(|_| rng.next_u32() & mask)
                    .collect(),
            })
            .collect();
        server.apply_row_updates(&updates).expect("apply batch");
        check_invariant(&server);
    }

    #[test]
    fn apply_batch_across_all_segments() {
        let mut rng = ChaCha20Rng::from_seed([31u8; 32]);
        let params = FilterParams::recommended(Arity::Segmented4, 64, 8).expect("params");
        let db = build_db(64, 8);
        let (mut server, _, _) = Server::setup(&params, &[77u8; 32], &db).expect("setup");

        let mask: u32 = (1u32 << params.mat_elem_bit_len) - 1;
        let row_elems_bucket = params.row_elems_bucket() as usize;
        let updates: Vec<PerSegmentRow> = (0..params.degree() as u8)
            .map(|seg_idx| PerSegmentRow {
                segment_idx: seg_idx,
                row_local_idx: 0,
                new_payload: (0..row_elems_bucket)
                    .map(|_| rng.next_u32() & mask)
                    .collect(),
            })
            .collect();
        server
            .apply_row_updates(&updates)
            .expect("apply across all segments");
        check_invariant(&server);
    }

    #[test]
    fn apply_sequential_same_cell_last_wins() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let params = FilterParams::recommended(Arity::Segmented4, 32, 4).expect("params");
        let db = build_db(32, 4);
        let (mut server, _, _) = Server::setup(&params, &[3u8; 32], &db).expect("setup");

        let mask: u32 = (1u32 << params.mat_elem_bit_len) - 1;
        let row_elems_bucket = params.row_elems_bucket() as usize;
        let first_payload: Vec<u32> = (0..row_elems_bucket)
            .map(|_| rng.next_u32() & mask)
            .collect();
        let second_payload: Vec<u32> = (0..row_elems_bucket)
            .map(|_| rng.next_u32() & mask)
            .collect();

        let updates = vec![
            PerSegmentRow {
                segment_idx: 0,
                row_local_idx: 0,
                new_payload: first_payload,
            },
            PerSegmentRow {
                segment_idx: 0,
                row_local_idx: 0,
                new_payload: second_payload.clone(),
            },
        ];
        server.apply_row_updates(&updates).expect("apply");

        assert_eq!(
            server.segment_d(0).row(0).expect("row"),
            second_payload.as_slice(),
            "last update must win"
        );
        check_invariant(&server);
    }

    #[test]
    fn apply_row_oob_is_error() {
        let params = FilterParams::recommended(Arity::Segmented4, 32, 4).expect("params");
        let db = build_db(32, 4);
        let (mut server, _, _) = Server::setup(&params, &[3u8; 32], &db).expect("setup");

        let bad = PerSegmentRow {
            segment_idx: 0,
            row_local_idx: params.seg(),
            new_payload: vec![0u32; params.row_elems_bucket() as usize],
        };
        assert_eq!(
            server.apply_row_updates(&[bad]),
            Err(UpdateError::RowIndexOutOfBounds)
        );
    }

    #[test]
    fn apply_segment_oob_is_error() {
        let params = FilterParams::recommended(Arity::Segmented4, 32, 4).expect("params");
        let db = build_db(32, 4);
        let (mut server, _, _) = Server::setup(&params, &[3u8; 32], &db).expect("setup");

        let bad = PerSegmentRow {
            segment_idx: params.degree() as u8,
            row_local_idx: 0,
            new_payload: vec![0u32; params.row_elems_bucket() as usize],
        };
        assert_eq!(
            server.apply_row_updates(&[bad]),
            Err(UpdateError::SegmentOutOfBounds)
        );
    }

    #[test]
    fn apply_shape_mismatch_is_error() {
        let params = FilterParams::recommended(Arity::Segmented4, 32, 4).expect("params");
        let db = build_db(32, 4);
        let (mut server, _, _) = Server::setup(&params, &[3u8; 32], &db).expect("setup");

        let bad = PerSegmentRow {
            segment_idx: 0,
            row_local_idx: 0,
            new_payload: vec![0u32; params.row_elems_bucket() as usize + 1],
        };
        assert_eq!(
            server.apply_row_updates(&[bad]),
            Err(UpdateError::ShapeMismatch)
        );
    }

    #[test]
    fn apply_payload_out_of_range_is_error() {
        let params = FilterParams::recommended(Arity::Segmented4, 32, 4).expect("params");
        let db = build_db(32, 4);
        let (mut server, _, _) = Server::setup(&params, &[3u8; 32], &db).expect("setup");

        let mask: u32 = (1u32 << params.mat_elem_bit_len) - 1;
        let mut payload = vec![0u32; params.row_elems_bucket() as usize];
        payload[0] = mask + 1;

        let bad = PerSegmentRow {
            segment_idx: 0,
            row_local_idx: 0,
            new_payload: payload,
        };
        assert_eq!(
            server.apply_row_updates(&[bad]),
            Err(UpdateError::PayloadOutOfRange)
        );
    }
}
