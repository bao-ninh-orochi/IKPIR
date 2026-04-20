//! Wire format for params, hint matrix, queries, and responses.
//!
//! The encoding is a tight hand-rolled little-endian layout so the bytes
//! reported by the paper's bandwidth figures match the implementation
//! exactly. No `serde`, no `bincode` — every field is a plain `u32` / `u8`.
//!
//! # Layout summary
//!
//! ```text
//!  FilterParams (header, versioned):
//!    magic    u32  = b"IKPR"                     4 B
//!    version  u8   = 1                           1 B
//!    arity    u8                                  1 B
//!    fp_bits  u8                                  1 B
//!    mat_bits u8                                  1 B
//!    num_buckets u32                              4 B
//!    row_elems   u32                              4 B
//!    value_len   u32                              4 B
//!    → 20 B total
//!
//!  Matrix (`rows × cols` of `u32`, row-major):
//!    rows u32                                     4 B
//!    cols u32                                     4 B
//!    elems [u32; rows*cols]                       4·rows·cols B
//!
//!  Query  = seed(32B) ‖ Matrix(1 × num_buckets)   (rows=1, cols=num_buckets)
//!  Response = Matrix(1 × row_elems)
//! ```
//!
//! The `Query` header includes the 32-byte public seed so that the server
//! can re-derive `A` locally without extra state — matching
//! ChalametPIR's on-the-wire shape.

use crate::matrix::{Matrix, MatrixError};
use crate::params::{Arity, FilterParams};

/// Magic header for a serialized `FilterParams`. ASCII "IKPR".
pub const MAGIC: [u8; 4] = *b"IKPR";
/// Wire-format version.
///
/// Version 2 introduces the per-segment PIR: `FilterParams::row_elems`
/// now carries `val_elems` (per-slot value width) rather than the old
/// peeling-scheme's full-row width; queries and responses are batched
/// per segment via `batch_{query,response}_{to,from}_bytes`.
pub const WIRE_VERSION: u8 = 2;

/// Errors raised during (de)serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerdeError {
    /// Buffer too short to hold the expected structure.
    TooShort,
    /// Magic bytes mismatched.
    BadMagic,
    /// Unknown wire version.
    BadVersion,
    /// Encoded `Arity` byte was out of range.
    BadArity,
    /// Matrix dimensions did not match the declared shape.
    BadMatrix(MatrixError),
    /// `FilterParams` fields were structurally invalid.
    BadParams,
}

impl core::fmt::Display for SerdeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooShort => f.write_str("buffer too short"),
            Self::BadMagic => f.write_str("missing or wrong magic"),
            Self::BadVersion => f.write_str("unknown wire version"),
            Self::BadArity => f.write_str("invalid arity byte"),
            Self::BadMatrix(e) => write!(f, "matrix error: {e}"),
            Self::BadParams => f.write_str("inconsistent FilterParams fields"),
        }
    }
}

impl std::error::Error for SerdeError {}

impl From<MatrixError> for SerdeError {
    fn from(e: MatrixError) -> Self {
        Self::BadMatrix(e)
    }
}

const ARITY_STD2: u8 = 0;
const ARITY_SEG2: u8 = 1;
const ARITY_STD3: u8 = 2;
const ARITY_SEG3: u8 = 3;
const ARITY_STD4: u8 = 4;
const ARITY_SEG4: u8 = 5;

fn arity_to_byte(a: Arity) -> u8 {
    match a {
        Arity::Standard2 => ARITY_STD2,
        Arity::Segmented2 => ARITY_SEG2,
        Arity::Standard3 => ARITY_STD3,
        Arity::Segmented3 => ARITY_SEG3,
        Arity::Standard4 => ARITY_STD4,
        Arity::Segmented4 => ARITY_SEG4,
    }
}

fn byte_to_arity(b: u8) -> Option<Arity> {
    match b {
        ARITY_STD2 => Some(Arity::Standard2),
        ARITY_SEG2 => Some(Arity::Segmented2),
        ARITY_STD3 => Some(Arity::Standard3),
        ARITY_SEG3 => Some(Arity::Segmented3),
        ARITY_STD4 => Some(Arity::Standard4),
        ARITY_SEG4 => Some(Arity::Segmented4),
        _ => None,
    }
}

/// Serialize a `FilterParams` into the 20-byte canonical form.
pub fn params_to_bytes(params: &FilterParams) -> Vec<u8> {
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(&MAGIC);
    out.push(WIRE_VERSION);
    out.push(arity_to_byte(params.arity));
    out.push(params.fingerprint_bits as u8);
    out.push(params.mat_elem_bit_len as u8);
    out.extend_from_slice(&params.num_buckets.to_le_bytes());
    out.extend_from_slice(&params.row_elems.to_le_bytes());
    out.extend_from_slice(&params.value_len.to_le_bytes());
    out
}

/// Inverse of [`params_to_bytes`].
pub fn params_from_bytes(bytes: &[u8]) -> Result<FilterParams, SerdeError> {
    if bytes.len() < 20 {
        return Err(SerdeError::TooShort);
    }
    if bytes[0..4] != MAGIC {
        return Err(SerdeError::BadMagic);
    }
    if bytes[4] != WIRE_VERSION {
        return Err(SerdeError::BadVersion);
    }
    let arity = byte_to_arity(bytes[5]).ok_or(SerdeError::BadArity)?;
    let fingerprint_bits = bytes[6] as u32;
    let mat_elem_bit_len = bytes[7] as u32;
    let num_buckets = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let row_elems = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let value_len = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);

    if fingerprint_bits == 0
        || fingerprint_bits > 32
        || mat_elem_bit_len == 0
        || mat_elem_bit_len >= 32
        || num_buckets == 0
        || row_elems == 0
        || value_len == 0
    {
        return Err(SerdeError::BadParams);
    }

    Ok(FilterParams {
        arity,
        num_buckets,
        fingerprint_bits,
        row_elems,
        mat_elem_bit_len,
        value_len,
    })
}

/// Serialize a matrix as `rows:u32 ‖ cols:u32 ‖ elems:[u32; rows*cols]`.
pub fn matrix_to_bytes(m: &Matrix) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + m.num_bytes());
    out.extend_from_slice(&m.rows().to_le_bytes());
    out.extend_from_slice(&m.cols().to_le_bytes());
    for &x in m.as_slice() {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Inverse of [`matrix_to_bytes`].
pub fn matrix_from_bytes(bytes: &[u8]) -> Result<Matrix, SerdeError> {
    if bytes.len() < 8 {
        return Err(SerdeError::TooShort);
    }
    let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let cols = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let expected_bytes = 8usize
        .checked_add(
            (rows as usize)
                .checked_mul(cols as usize)
                .and_then(|x| x.checked_mul(4))
                .ok_or(SerdeError::TooShort)?,
        )
        .ok_or(SerdeError::TooShort)?;
    if bytes.len() < expected_bytes {
        return Err(SerdeError::TooShort);
    }
    let n = (rows as usize) * (cols as usize);
    let mut elems = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * 4;
        elems.push(u32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]));
    }
    Ok(Matrix::from_elems(rows, cols, elems)?)
}

/// Serialize a batched k-segment query: concatenated `b_j` vectors of
/// length `seg` in little-endian u32 form.
///
/// Layout: `b_0[u32; seg] ‖ b_1[u32; seg] ‖ … ‖ b_{k-1}[u32; seg]`.
/// Size: `4 · k · seg` bytes.
pub fn batch_query_to_bytes(per_seg: &[Vec<u32>]) -> Vec<u8> {
    let total: usize = per_seg.iter().map(|v| v.len() * 4).sum();
    let mut out = Vec::with_capacity(total);
    for vec in per_seg {
        for &x in vec {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// Inverse of [`batch_query_to_bytes`]. Expects exactly `k · seg · 4` bytes.
pub fn batch_query_from_bytes(bytes: &[u8], k: u32, seg: u32) -> Result<Vec<Vec<u32>>, SerdeError> {
    let total = (k as usize) * (seg as usize) * 4;
    if bytes.len() != total {
        return Err(SerdeError::TooShort);
    }
    let mut out = Vec::with_capacity(k as usize);
    for j in 0..k as usize {
        let base = j * (seg as usize) * 4;
        let mut v = Vec::with_capacity(seg as usize);
        for i in 0..seg as usize {
            let off = base + i * 4;
            v.push(u32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]));
        }
        out.push(v);
    }
    Ok(out)
}

/// Serialize a batched k-segment response: concatenated `r_j` vectors of
/// length `row_elems_bucket`.
pub fn batch_response_to_bytes(per_seg: &[Vec<u32>]) -> Vec<u8> {
    let total: usize = per_seg.iter().map(|v| v.len() * 4).sum();
    let mut out = Vec::with_capacity(total);
    for vec in per_seg {
        for &x in vec {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// Inverse of [`batch_response_to_bytes`]. Expects exactly
/// `k · row_elems_bucket · 4` bytes.
pub fn batch_response_from_bytes(
    bytes: &[u8],
    k: u32,
    row_elems_bucket: u32,
) -> Result<Vec<Vec<u32>>, SerdeError> {
    let total = (k as usize) * (row_elems_bucket as usize) * 4;
    if bytes.len() != total {
        return Err(SerdeError::TooShort);
    }
    let mut out = Vec::with_capacity(k as usize);
    for j in 0..k as usize {
        let base = j * (row_elems_bucket as usize) * 4;
        let mut v = Vec::with_capacity(row_elems_bucket as usize);
        for i in 0..row_elems_bucket as usize {
            let off = base + i * 4;
            v.push(u32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]));
        }
        out.push(v);
    }
    Ok(out)
}

/// Serialize k hint matrices as a concatenated sequence of
/// self-delimiting [`matrix_to_bytes`] blobs.
pub fn batch_hint_to_bytes(hints: &[Matrix]) -> Vec<u8> {
    let mut out = Vec::new();
    for h in hints {
        out.extend_from_slice(&matrix_to_bytes(h));
    }
    out
}

/// Inverse of [`batch_hint_to_bytes`]. Parses exactly `k` matrices; fails
/// if trailing bytes remain or if any matrix is truncated.
pub fn batch_hint_from_bytes(bytes: &[u8], k: u32) -> Result<Vec<Matrix>, SerdeError> {
    let mut cursor = 0usize;
    let mut out = Vec::with_capacity(k as usize);
    for _ in 0..k {
        if bytes.len().saturating_sub(cursor) < 8 {
            return Err(SerdeError::TooShort);
        }
        let rows = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]);
        let cols = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]);
        let elem_bytes = (rows as usize)
            .checked_mul(cols as usize)
            .and_then(|x| x.checked_mul(4))
            .ok_or(SerdeError::TooShort)?;
        let blob_len = 8usize.checked_add(elem_bytes).ok_or(SerdeError::TooShort)?;
        if bytes.len().saturating_sub(cursor) < blob_len {
            return Err(SerdeError::TooShort);
        }
        out.push(matrix_from_bytes(&bytes[cursor..cursor + blob_len])?);
        cursor += blob_len;
    }
    if cursor != bytes.len() {
        return Err(SerdeError::TooShort);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::FilterParams;

    fn make_params(arity: Arity) -> FilterParams {
        if arity.is_segmented() {
            FilterParams::recommended(arity, 256, 32).unwrap()
        } else {
            FilterParams {
                arity,
                num_buckets: 256,
                fingerprint_bits: 12,
                row_elems: 32,
                mat_elem_bit_len: 8,
                value_len: 32,
            }
        }
    }

    #[test]
    fn params_round_trip_all_arities() {
        for &a in &[
            Arity::Standard2,
            Arity::Segmented2,
            Arity::Standard3,
            Arity::Segmented3,
            Arity::Standard4,
            Arity::Segmented4,
        ] {
            let p = make_params(a);
            let bytes = params_to_bytes(&p);
            assert_eq!(bytes.len(), 20);
            let got = params_from_bytes(&bytes).unwrap();
            assert_eq!(got, p);
        }
    }

    #[test]
    fn params_rejects_bad_magic() {
        let p = FilterParams::recommended(Arity::Segmented4, 256, 32).unwrap();
        let mut bytes = params_to_bytes(&p);
        bytes[0] ^= 1;
        assert!(matches!(
            params_from_bytes(&bytes),
            Err(SerdeError::BadMagic)
        ));
    }

    #[test]
    fn params_rejects_bad_version() {
        let p = FilterParams::recommended(Arity::Segmented4, 256, 32).unwrap();
        let mut bytes = params_to_bytes(&p);
        bytes[4] = 42;
        assert!(matches!(
            params_from_bytes(&bytes),
            Err(SerdeError::BadVersion)
        ));
    }

    #[test]
    fn matrix_round_trip() {
        let m = Matrix::from_seed(7, 13, &[3u8; 32]).unwrap();
        let bytes = matrix_to_bytes(&m);
        let got = matrix_from_bytes(&bytes).unwrap();
        assert_eq!(got, m);
    }

    #[test]
    fn batch_query_round_trip() {
        let per_seg: Vec<Vec<u32>> = (0..4)
            .map(|j| (0..50).map(|i| (j as u32) * 1000 + i).collect())
            .collect();
        let bytes = batch_query_to_bytes(&per_seg);
        let got = batch_query_from_bytes(&bytes, 4, 50).unwrap();
        assert_eq!(got, per_seg);
    }

    #[test]
    fn batch_query_rejects_wrong_size() {
        let per_seg: Vec<Vec<u32>> = (0..4).map(|_| vec![0u32; 50]).collect();
        let bytes = batch_query_to_bytes(&per_seg);
        assert!(batch_query_from_bytes(&bytes, 4, 49).is_err());
        assert!(batch_query_from_bytes(&bytes, 3, 50).is_err());
    }

    #[test]
    fn batch_response_round_trip() {
        let per_seg: Vec<Vec<u32>> = (0..3)
            .map(|j| (0..20).map(|i| (j * 17 + i) as u32).collect())
            .collect();
        let bytes = batch_response_to_bytes(&per_seg);
        let got = batch_response_from_bytes(&bytes, 3, 20).unwrap();
        assert_eq!(got, per_seg);
    }

    #[test]
    fn batch_hint_round_trip_three_matrices() {
        let m0 = Matrix::from_seed(3, 4, &[1u8; 32]).unwrap();
        let m1 = Matrix::from_seed(3, 4, &[2u8; 32]).unwrap();
        let m2 = Matrix::from_seed(3, 4, &[3u8; 32]).unwrap();
        let bytes = batch_hint_to_bytes(&[m0.clone(), m1.clone(), m2.clone()]);
        let got = batch_hint_from_bytes(&bytes, 3).unwrap();
        assert_eq!(got, vec![m0, m1, m2]);
    }

    #[test]
    fn batch_hint_rejects_trailing_bytes() {
        let m = Matrix::from_seed(3, 4, &[7u8; 32]).unwrap();
        let mut bytes = batch_hint_to_bytes(&[m]);
        bytes.push(0);
        assert!(batch_hint_from_bytes(&bytes, 1).is_err());
    }
}
