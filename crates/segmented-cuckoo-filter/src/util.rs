//! Shared arithmetic utilities used by filter constructors.
//!
//! This module contains small, pure helpers that are too general to belong to any single
//! domain module. Currently the only export is [`upper_power_of_2`], which drives the
//! auto-sizing logic in every `from_num_items` constructor.

/// Return the smallest power of 2 that is ≥ `x`.
///
/// Used by every `from_num_items` constructor to round a required bucket count up to the
/// nearest valid table size. All filter indexing schemes require `n` to be a power of 2
/// (or a multiple of a power of 2 for segmented-3ary), so this function provides the
/// canonical rounding step.
///
/// # Arguments
///
/// - `x` — the value to round up. Any `u64` is accepted; `0` is treated as `1` so the
///   return value is always ≥ 1.
///
/// # Returns
///
/// The smallest `p` such that `p` is a power of 2 and `p >= x`. Always in `[1, 2^63]`.
///
/// # Performance
///
/// O(1) — delegates to [`u64::next_power_of_two`] which is a single hardware intrinsic
/// (`BSR` / `LZCNT`) on x86-64.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::util::upper_power_of_2;
///
/// assert_eq!(upper_power_of_2(0), 1);   // special-cased: 0 → 1
/// assert_eq!(upper_power_of_2(1), 1);
/// assert_eq!(upper_power_of_2(3), 4);   // rounded up
/// assert_eq!(upper_power_of_2(16), 16); // already a power of 2 → unchanged
/// assert_eq!(upper_power_of_2(17), 32); // rounded up to next power
/// ```
pub fn upper_power_of_2(x: u64) -> u64 {
    if x == 0 {
        return 1;
    }
    x.next_power_of_two()
}

/// Return `true` if `n` is a power of 3 (3^k for some k ≥ 0).
pub fn is_power_of_3(n: u32) -> bool {
    if n == 0 {
        return false;
    }
    let mut v = n;
    while v % 3 == 0 {
        v /= 3;
    }
    v == 1
}

/// Return the smallest power of 3 that is ≥ `x`. Returns 1 for x=0.
pub fn next_power_of_3(x: u64) -> u64 {
    if x <= 1 {
        return 1;
    }
    let mut p = 1u64;
    while p < x {
        p *= 3;
    }
    p
}

/// Return `true` if `n` is a power of 4 (4^k = 2^(2k) for k ≥ 0).
pub fn is_power_of_4(n: u32) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 2 == 0)
}

/// Return the smallest power of 4 that is ≥ `x`. Returns 1 for x=0.
pub fn next_power_of_4(x: u64) -> u64 {
    if x <= 1 {
        return 1;
    }
    let mut p = 1u64;
    while p < x {
        p *= 4;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upper_power_of_2() {
        assert_eq!(upper_power_of_2(0), 1);
        assert_eq!(upper_power_of_2(1), 1);
        assert_eq!(upper_power_of_2(2), 2);
        assert_eq!(upper_power_of_2(3), 4);
        assert_eq!(upper_power_of_2(5), 8);
        assert_eq!(upper_power_of_2(16), 16);
        assert_eq!(upper_power_of_2(17), 32);
    }

    #[test]
    fn test_is_power_of_3() {
        assert!(!is_power_of_3(0));
        assert!(is_power_of_3(1));
        assert!(is_power_of_3(3));
        assert!(is_power_of_3(9));
        assert!(is_power_of_3(27));
        assert!(is_power_of_3(243));
        assert!(!is_power_of_3(2));
        assert!(!is_power_of_3(4));
        assert!(!is_power_of_3(6));
    }

    #[test]
    fn test_next_power_of_3() {
        assert_eq!(next_power_of_3(0), 1);
        assert_eq!(next_power_of_3(1), 1);
        assert_eq!(next_power_of_3(2), 3);
        assert_eq!(next_power_of_3(3), 3);
        assert_eq!(next_power_of_3(4), 9);
        assert_eq!(next_power_of_3(9), 9);
        assert_eq!(next_power_of_3(10), 27);
        assert_eq!(next_power_of_3(243), 243);
        assert_eq!(next_power_of_3(244), 729);
    }

    #[test]
    fn test_is_power_of_4() {
        assert!(!is_power_of_4(0));
        assert!(is_power_of_4(1));
        assert!(!is_power_of_4(2));
        assert!(!is_power_of_4(3));
        assert!(is_power_of_4(4));
        assert!(!is_power_of_4(8));
        assert!(is_power_of_4(16));
        assert!(!is_power_of_4(32));
        assert!(is_power_of_4(64));
        assert!(is_power_of_4(256));
    }

    #[test]
    fn test_next_power_of_4() {
        assert_eq!(next_power_of_4(0), 1);
        assert_eq!(next_power_of_4(1), 1);
        assert_eq!(next_power_of_4(2), 4);
        assert_eq!(next_power_of_4(4), 4);
        assert_eq!(next_power_of_4(5), 16);
        assert_eq!(next_power_of_4(16), 16);
        assert_eq!(next_power_of_4(17), 64);
        assert_eq!(next_power_of_4(256), 256);
        assert_eq!(next_power_of_4(257), 1024);
    }
}
