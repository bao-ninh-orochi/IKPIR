//! [`ct_eq_u64_mask`] — shared constant-time equality mask used by both
//! client flows' fingerprint scans.

/// Branchless `u64` equality mask: returns `u64::MAX` if `a == b`, else `0`.
///
/// Standard constant-time trick: `x ^ b == 0` iff `a == b`; squeeze that
/// zero/non-zero into bit 63 via `x | -x`, shift down, then subtract 1 to
/// flip the meaning.
#[inline]
pub(crate) const fn ct_eq_u64_mask(a: u64, b: u64) -> u64 {
    let x = a ^ b;
    ((x | x.wrapping_neg()) >> 63).wrapping_sub(1)
}
