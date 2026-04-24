//! Decode the server's per-segment responses into the stored record value.
//!
//! For each segment `j`:
//!
//! ```text
//!   r_j − c_j  =  e_jᵀ · D_j  +  Δ · D_j[local_j, :]     (mod 2^32)
//! ```
//!
//! Rounding to multiples of `Δ` recovers `D_j[local_j, :]`, which encodes
//! `b = slots_per_bucket` `(fingerprint, value)` slots. We scan each slot
//! for a fingerprint matching `state.fingerprint` and return the decoded value on
//! a hit, or `Ok(None)` if no segment contains a matching slot.
//!
//! # Edge cases
//!
//! * Empty slot: `fp == 0` — never equals a valid tag (SCF guarantees `tag != 0`).
//! * Fingerprint collision: two distinct keys may produce the same tag at a
//!   false-positive rate `d·b/2^{fp_bits} ≈ 0.4 %` (default params). The first
//!   matching slot in the first matching segment is returned.
//! * Noise overflow: `decode_value` returns `None` — propagated as
//!   [`DecryptError::DecodeFailed`].

use ikpir_common::cuckoo_store::decode_fp;
use ikpir_common::lwe::{decode_value, round_and_extract};
use ikpir_common::matrix::vec_sub_assign;
use ikpir_common::serialization::{batch_response_from_bytes, SerdeError};

use crate::query::QueryState;

/// Errors raised during decrypt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecryptError {
    /// Response wire-format error or length mismatch.
    Serde(SerdeError),
    /// Rounding produced a slot that could not be decoded to `value_len` bytes
    /// (noise overflow or corrupted response).
    DecodeFailed,
}

impl From<SerdeError> for DecryptError {
    fn from(e: SerdeError) -> Self {
        Self::Serde(e)
    }
}

impl core::fmt::Display for DecryptError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Serde(e) => write!(f, "wire format error: {e}"),
            Self::DecodeFailed => f.write_str("response could not be decoded (noise overflow?)"),
        }
    }
}

impl std::error::Error for DecryptError {}

/// Decrypt the server's batched response for a pending [`QueryState`].
///
/// Returns:
///   * `Ok(Some(value))` — the stored value for the queried keyword.
///   * `Ok(None)` — the keyword was never inserted (filter miss); no slot in any
///     segment matched the fingerprint tag.
///   * `Err(DecryptError::Serde)` — malformed `response_bytes`.
///   * `Err(DecryptError::DecodeFailed)` — noise overflow or corrupted response.
pub fn decrypt(state: &QueryState, response_bytes: &[u8]) -> Result<Option<Vec<u8>>, DecryptError> {
    let k = state.params.degree() as usize;
    let row_elems_bucket = state.params.row_elems_bucket();
    let per_seg_r =
        batch_response_from_bytes(response_bytes, state.params.degree(), row_elems_bucket)?;

    let fp_elems = state.params.fp_elems() as usize;
    let val_elems = state.params.val_elems() as usize;
    let slot_elems = fp_elems + val_elems;
    let b = state.params.slots_per_bucket() as usize;

    for (r_j, c_j) in per_seg_r.iter().zip(state.c.iter()).take(k) {
        // r_j − c_j
        let mut r = r_j.clone();
        vec_sub_assign(&mut r, c_j);

        // Round to nearest scaled payload, extract raw plaintext.
        let slots = round_and_extract(&r, &state.params);

        // Scan b slots for a matching fingerprint.
        for s in 0..b {
            let base = s * slot_elems;
            let fp = decode_fp(
                &slots[base..base + fp_elems],
                state.params.fingerprint_bits,
                state.params.mat_elem_bit_len,
            );
            if fp == state.fingerprint {
                let val = decode_value(&slots[base + fp_elems..base + slot_elems], &state.params)
                    .ok_or(DecryptError::DecodeFailed)?;
                return Ok(Some(val));
            }
        }
    }

    Ok(None)
}
