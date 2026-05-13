//! Shared arithmetic utilities used by filter and KV-store constructors.
//!
//! # Purpose
//!
//! Small, pure helpers that are too general to belong to any single domain
//! module: `next_power_of_*` for rounding up requested capacity to a valid
//! `num_buckets`, and `is_power_of_*` predicates for the matching
//! validation checks.
//!
//! # Design / architecture
//!
//! Each `from_num_items` constructor rounds the requested item count up to
//! a valid `num_buckets` via one of these helpers; `is_power_of_*` then
//! flags caller-supplied `num_buckets` that would violate the scheme's
//! shape constraint. Hot path is `next_power_of_2`, which delegates to
//! the `BSR` / `LZCNT` hardware intrinsic; the 3- and 4-ary helpers
//! iterate (constructors are not on a hot path).
//!
//! # Related files
//!
//! - `store.rs` / `filter.rs` — every `new` / `from_num_items` constructor.
//! - `scheme.rs` — segmented-3ary geometry depends on
//!   `is_power_of_3`-style assertions to enforce
//!   `num_buckets = 3 · 2^t`.

/// Smallest power of 2 that is ≥ `x`.
///
/// # Purpose
///
/// Canonical rounding step used by every `from_num_items` constructor:
/// turns a requested bucket count into the nearest valid power-of-2 table
/// size.
///
/// # Arguments
///
/// - `x` — value to round up. `0` is special-cased to return `1`.
///
/// # Constraints
///
/// Panics if `x > u64::MAX / 2` (the result would overflow `u64`).
///
/// # Returns
///
/// The smallest `p` such that `p` is a power of 2 and `p ≥ x`. Always in
/// `[1, 2^63]`.
///
/// # Complexity
///
/// `O(1)` — delegates to [`u64::next_power_of_two`], a single hardware
/// intrinsic (`BSR` / `LZCNT`) on x86-64.
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::util::next_power_of_2;
///
/// assert_eq!(next_power_of_2(0), 1);   // special-cased: 0 → 1
/// assert_eq!(next_power_of_2(1), 1);
/// assert_eq!(next_power_of_2(3), 4);   // rounded up
/// assert_eq!(next_power_of_2(16), 16); // already a power of 2 → unchanged
/// assert_eq!(next_power_of_2(17), 32); // rounded up to next power
/// ```
pub fn next_power_of_2(x: u64) -> u64 {
    if x == 0 {
        return 1;
    }
    assert!(x <= 1u64 << 63, "next_power_of_2 input exceeds u64::MAX/2 ({x})");
    x.next_power_of_two()
}

/// Smallest power of 3 that is ≥ `x`.
///
/// # Purpose
///
/// Used by `Standard3aryCuckooFilter::from_num_items` to round up to a
/// valid `num_buckets = 3^t`.
///
/// # Arguments
///
/// - `x` — value to round up. `0` and `1` both return `1`.
///
/// # Returns
///
/// The smallest `p = 3^t` with `p ≥ x`.
///
/// # Complexity
///
/// `O(log_3(x))` iterations — bounded above by ~40 for any `u64`. Not on a
/// hot path.
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

/// Smallest power of 4 that is ≥ `x`.
///
/// # Purpose
///
/// Used by `Standard4aryCuckooFilter::from_num_items` to round up to a
/// valid `num_buckets = 4^t`.
///
/// # Arguments
///
/// - `x` — value to round up. `0` and `1` both return `1`.
///
/// # Returns
///
/// The smallest `p = 4^t` with `p ≥ x`.
///
/// # Complexity
///
/// `O(log_4(x))` iterations — bounded above by ~32 for any `u64`.
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

/// `true` iff `n` is a power of 3 (`3^t` for some `t ≥ 0`).
///
/// # Arguments
///
/// - `n` — value to test. `0` is not a power of 3 (returns `false`).
///
/// # Returns
///
/// `true` if `n ∈ {1, 3, 9, 27, …}`, `false` otherwise.
///
/// # Complexity
///
/// `O(log_3(n))` divisions in the worst case.
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

/// `true` iff `n` is a power of 4 (`4^t = 2^(2t)` for `t ≥ 0`).
///
/// # Arguments
///
/// - `n` — value to test. `0` returns `false`.
///
/// # Returns
///
/// `true` if `n ∈ {1, 4, 16, 64, …}`, `false` otherwise.
///
/// # Rationale
///
/// A power of 4 is also a power of 2 with an even number of trailing
/// zeros — checking both predicates is `O(1)` (two intrinsics).
pub fn is_power_of_4(n: u32) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % 2 == 0)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the power-of-`k` helpers. Each test pins a
    //! representative cross-section of the input range, including the
    //! boundary cases (`0`, `1`) and the failure mode for
    //! `next_power_of_2`.

    use super::*;

    /// `next_power_of_2` rounds up correctly and is identity on a power of 2.
    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(16), 16);
        assert_eq!(next_power_of_2(17), 32);
    }

    /// `next_power_of_3` rounds up to `3^t` correctly.
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

    /// `next_power_of_4` rounds up to `4^t` correctly.
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

    /// `is_power_of_3` recognises `3^t` and rejects everything else.
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

    /// `is_power_of_4` recognises `4^t` and rejects everything else
    /// (including powers of 2 that are not powers of 4, like 2, 8, 32).
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

    /// `next_power_of_2` panics on inputs that would overflow `u64`.
    #[test]
    #[should_panic(expected = "exceeds")]
    fn next_power_of_2_panics_on_huge() {
        next_power_of_2(u64::MAX);
    }
}
