//! Matrix types and arithmetic over `Z_q` with `q = 2^32`.
//!
//! All matrix elements are `u32`; arithmetic is `wrapping_add` / `wrapping_sub`
//! / `wrapping_mul`, which agrees with `mod 2^32` reduction. This lets the
//! whole PIR stack be built from plain-old `u32` ops without tracking modular
//! reductions explicitly.
//!
//! Storage is row-major (`elems[row * cols + col]`). Mat-vec products are
//! laid out so that the hot loops run contiguously over rows, which the
//! compiler auto-vectorises cleanly on modern CPUs without any unsafe / SIMD
//! intrinsics.

use core::fmt;

use crate::hash;
use crate::params::SEED_LEN;

/// Row-major `u32` matrix over `Z_{2^32}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix {
    rows: u32,
    cols: u32,
    elems: Vec<u32>,
}

/// Errors emitted by matrix construction or arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixError {
    /// Requested dimensions overflow `usize` / `u32`.
    DimensionOverflow,
    /// Caller supplied a backing `Vec<u32>` whose length mismatches `rows * cols`.
    LengthMismatch,
    /// Dimensions incompatible for a multiply / add / mat-vec.
    ShapeMismatch,
    /// Row / column index out of bounds.
    OutOfBounds,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOverflow => f.write_str("matrix dimensions overflow usize"),
            Self::LengthMismatch => f.write_str("backing buffer length does not match dimensions"),
            Self::ShapeMismatch => f.write_str("incompatible matrix shapes"),
            Self::OutOfBounds => f.write_str("matrix index out of bounds"),
        }
    }
}

impl std::error::Error for MatrixError {}

impl Matrix {
    /// Build a zero matrix of shape `rows × cols`.
    pub fn zeros(rows: u32, cols: u32) -> Result<Self, MatrixError> {
        let len = Self::flat_len(rows, cols)?;
        Ok(Self {
            rows,
            cols,
            elems: vec![0u32; len],
        })
    }

    /// Wrap an existing backing buffer into a matrix.
    pub fn from_elems(rows: u32, cols: u32, elems: Vec<u32>) -> Result<Self, MatrixError> {
        let len = Self::flat_len(rows, cols)?;
        if elems.len() != len {
            return Err(MatrixError::LengthMismatch);
        }
        Ok(Self { rows, cols, elems })
    }

    /// Expand a `rows × cols` matrix from a 32-byte seed using ChaCha20.
    pub fn from_seed(rows: u32, cols: u32, seed: &[u8; SEED_LEN]) -> Result<Self, MatrixError> {
        let len = Self::flat_len(rows, cols)?;
        let elems = hash::prg_u32_stream(seed, len);
        Ok(Self { rows, cols, elems })
    }

    fn flat_len(rows: u32, cols: u32) -> Result<usize, MatrixError> {
        (rows as usize)
            .checked_mul(cols as usize)
            .ok_or(MatrixError::DimensionOverflow)
    }

    /// Number of rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Borrow the flat element buffer.
    pub fn as_slice(&self) -> &[u32] {
        &self.elems
    }

    /// Borrow the flat element buffer mutably.
    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        &mut self.elems
    }

    /// Borrow one row as a slice.
    pub fn row(&self, r: u32) -> Result<&[u32], MatrixError> {
        if r >= self.rows {
            return Err(MatrixError::OutOfBounds);
        }
        let start = (r as usize) * (self.cols as usize);
        Ok(&self.elems[start..start + self.cols as usize])
    }

    /// Borrow one row mutably.
    pub fn row_mut(&mut self, r: u32) -> Result<&mut [u32], MatrixError> {
        if r >= self.rows {
            return Err(MatrixError::OutOfBounds);
        }
        let start = (r as usize) * (self.cols as usize);
        let end = start + self.cols as usize;
        Ok(&mut self.elems[start..end])
    }

    /// Copy column `col` into a fresh `Vec<u32>` of length `rows`.
    ///
    /// The matrix is row-major so this is a stride-`cols` gather; O(rows).
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::OutOfBounds`] if `col >= self.cols`.
    pub fn column(&self, col: u32) -> Result<Vec<u32>, MatrixError> {
        if col >= self.cols {
            return Err(MatrixError::OutOfBounds);
        }
        let cols = self.cols as usize;
        let col_us = col as usize;
        let mut out = Vec::with_capacity(self.rows as usize);
        for r in 0..self.rows as usize {
            out.push(self.elems[r * cols + col_us]);
        }
        Ok(out)
    }

    /// Get `self[row][col]`.
    pub fn get(&self, row: u32, col: u32) -> Result<u32, MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::OutOfBounds);
        }
        Ok(self.elems[(row as usize) * (self.cols as usize) + col as usize])
    }

    /// Set `self[row][col] = v`.
    pub fn set(&mut self, row: u32, col: u32, v: u32) -> Result<(), MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::OutOfBounds);
        }
        self.elems[(row as usize) * (self.cols as usize) + col as usize] = v;
        Ok(())
    }

    /// Compute `self * rhs` over `Z_{2^32}`.
    ///
    /// Shapes: `(m × k) * (k × n) → (m × n)`. `O(m · k · n)`.
    // Index-based triple loop is deliberate — `p` is used for both the
    // `a_row` lookup AND to slice into `rhs.elems[p*n_us..(p+1)*n_us]`, which
    // an iterator form would obscure.
    #[allow(clippy::needless_range_loop)]
    pub fn mul(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        if self.cols != rhs.rows {
            return Err(MatrixError::ShapeMismatch);
        }
        let m = self.rows;
        let k = self.cols;
        let n = rhs.cols;
        let mut out = Matrix::zeros(m, n)?;
        let k_us = k as usize;
        let n_us = n as usize;
        // Row-major triple loop: (i, p, j). Swapping the j loop inside accesses
        // rhs[p, ..] contiguously, which the compiler vectorises.
        for i in 0..m as usize {
            let a_row = &self.elems[i * k_us..(i + 1) * k_us];
            let c_row = &mut out.elems[i * n_us..(i + 1) * n_us];
            for p in 0..k_us {
                let a = a_row[p];
                let b_row = &rhs.elems[p * n_us..(p + 1) * n_us];
                for j in 0..n_us {
                    c_row[j] = c_row[j].wrapping_add(a.wrapping_mul(b_row[j]));
                }
            }
        }
        Ok(out)
    }

    /// Right-multiply the row vector `vec` by `self`: `vec · self` where
    /// `self` is `rows × cols` and `vec.len() == rows`. Returns a vector of
    /// length `cols`.
    #[allow(clippy::needless_range_loop)]
    pub fn left_mul_row_vec(&self, vec: &[u32]) -> Result<Vec<u32>, MatrixError> {
        if vec.len() != self.rows as usize {
            return Err(MatrixError::ShapeMismatch);
        }
        let cols = self.cols as usize;
        let mut out = vec![0u32; cols];
        for (i, &coeff) in vec.iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            let row = &self.elems[i * cols..(i + 1) * cols];
            for j in 0..cols {
                out[j] = out[j].wrapping_add(coeff.wrapping_mul(row[j]));
            }
        }
        Ok(out)
    }

    /// Right-multiply `self` by a column vector `vec`: `self · vec` where
    /// `vec.len() == cols`. Returns a vector of length `rows`.
    pub fn right_mul_col_vec(&self, vec: &[u32]) -> Result<Vec<u32>, MatrixError> {
        if vec.len() != self.cols as usize {
            return Err(MatrixError::ShapeMismatch);
        }
        let cols = self.cols as usize;
        let mut out = vec![0u32; self.rows as usize];
        for (i, out_i) in out.iter_mut().enumerate() {
            let row = &self.elems[i * cols..(i + 1) * cols];
            let mut acc: u32 = 0;
            for j in 0..cols {
                acc = acc.wrapping_add(row[j].wrapping_mul(vec[j]));
            }
            *out_i = acc;
        }
        Ok(out)
    }

    /// Rough wire-size estimate (element count × 4 bytes).
    pub fn num_bytes(&self) -> usize {
        self.elems.len() * 4
    }

    /// Total element count.
    pub fn num_elems(&self) -> usize {
        self.elems.len()
    }
}

/// Add two `u32` vectors over `Z_{2^32}` in place: `a[i] = a[i] + b[i]`.
pub fn vec_add_assign(a: &mut [u32], b: &[u32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, &y) in a.iter_mut().zip(b.iter()) {
        *x = x.wrapping_add(y);
    }
}

/// Subtract two `u32` vectors over `Z_{2^32}` in place: `a[i] = a[i] - b[i]`.
pub fn vec_sub_assign(a: &mut [u32], b: &[u32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, &y) in a.iter_mut().zip(b.iter()) {
        *x = x.wrapping_sub(y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_expected_shape() {
        let m = Matrix::zeros(3, 4).unwrap();
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.as_slice(), &[0u32; 12]);
    }

    #[test]
    fn from_seed_is_deterministic() {
        let a = Matrix::from_seed(5, 7, &[1u8; 32]).unwrap();
        let b = Matrix::from_seed(5, 7, &[1u8; 32]).unwrap();
        assert_eq!(a, b);
        let c = Matrix::from_seed(5, 7, &[2u8; 32]).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn mul_multiplies_small_matrices() {
        // Identity × arbitrary = arbitrary.
        let mut id = Matrix::zeros(3, 3).unwrap();
        for i in 0..3 {
            id.set(i, i, 1).unwrap();
        }
        let a = Matrix::from_elems(3, 2, vec![1, 2, u32::MAX, 0, u32::MAX - 1, 100_000]).unwrap();
        let prod = id.mul(&a).unwrap();
        assert_eq!(prod, a);
    }

    #[test]
    fn wrapping_semantics() {
        let a = Matrix::from_elems(1, 2, vec![u32::MAX, u32::MAX]).unwrap();
        let b = Matrix::from_elems(2, 1, vec![1, 1]).unwrap();
        let prod = a.mul(&b).unwrap();
        // (2^32 - 1) + (2^32 - 1) = 2^33 - 2 ≡ -2 mod 2^32 = 2^32 - 2
        assert_eq!(prod.get(0, 0).unwrap(), u32::MAX - 1);
    }

    #[test]
    fn mul_shape_mismatch_is_error() {
        let a = Matrix::zeros(2, 3).unwrap();
        let b = Matrix::zeros(2, 2).unwrap(); // should be 3 rows
        assert!(matches!(a.mul(&b), Err(MatrixError::ShapeMismatch)));
    }

    #[test]
    fn left_mul_row_vec_agrees_with_mul() {
        // Build a small matrix and cross-check.
        let m = Matrix::from_elems(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let v = [10u32, 20u32];
        let lhs_as_mat = Matrix::from_elems(1, 2, v.to_vec()).unwrap();
        let mat_product = lhs_as_mat.mul(&m).unwrap();
        let direct = m.left_mul_row_vec(&v).unwrap();
        assert_eq!(direct, mat_product.as_slice());
    }

    #[test]
    fn right_mul_col_vec_agrees_with_mul() {
        let m = Matrix::from_elems(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let v = [10u32, 20u32];
        let rhs_as_mat = Matrix::from_elems(2, 1, v.to_vec()).unwrap();
        let mat_product = m.mul(&rhs_as_mat).unwrap();
        let direct = m.right_mul_col_vec(&v).unwrap();
        let expected: Vec<u32> = (0..3).map(|i| mat_product.get(i, 0).unwrap()).collect();
        assert_eq!(direct, expected);
    }

    #[test]
    fn vec_add_sub_round_trip() {
        let mut a = vec![1u32, 2, 3, u32::MAX];
        let b = vec![10u32, 20, 30, 1];
        let original = a.clone();
        vec_add_assign(&mut a, &b);
        vec_sub_assign(&mut a, &b);
        assert_eq!(a, original);
    }

    #[test]
    fn column_matches_get() {
        let m = Matrix::from_elems(3, 4, (0u32..12).collect()).unwrap();
        for c in 0..4u32 {
            let col = m.column(c).unwrap();
            assert_eq!(col.len(), 3);
            for r in 0..3u32 {
                assert_eq!(
                    col[r as usize],
                    m.get(r, c).unwrap(),
                    "mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn column_out_of_bounds() {
        let m = Matrix::zeros(2, 3).unwrap();
        assert!(matches!(m.column(3), Err(MatrixError::OutOfBounds)));
    }
}
