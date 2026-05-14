//! `round_p_to_q` / `round_q_to_p` — Δ-scaling between plaintext and
//! ciphertext rings.
//!
//! # Purpose
//!
//! Move a value between `Z_p = Z_{2^plaintext_bits}` (the IKPIR cell
//! ring) and `Z_q = Z_{2^32}` (the FrodoPIR ciphertext ring). The
//! one-line description: multiply / divide by `Δ = q / p` with
//! round-to-nearest on the way back.
//!
//! # Design / architecture
//!
//! - `round_p_to_q(x)` = `x << (32 - plaintext_bits)` (Δ is always a
//!   power of two with our parameter set, so the multiply is a shift).
//! - `round_q_to_p(y)` = `⌊(y + Δ/2) / Δ⌋` with wrapping `+ Δ/2` —
//!   wrapping is safe because the shift drops the carry-out bits.
//!
//! Both functions are `#[inline]` and run in `O(1)`; they are called
//! once per cell on the FrodoPIR hot path.
//!
//! # Related files
//!
//! - `backend.rs` — sole caller: `client_decode` calls `round_q_to_p`
//!   per cell; `server_answer` and the patcher call `round_p_to_q`.

/// Lift `x ∈ Z_p` into `Z_q` (`q = 2^32`).
///
/// # Arguments
///
/// - `x`              — value to lift; must satisfy
///   `x < 2^plaintext_bits` (the high-bits-zero invariant).
/// - `plaintext_bits` — log₂ of `p`; in `1..=31`.
///
/// # Returns
///
/// `x · Δ` where `Δ = q / p = 2^(32 − plaintext_bits)`. Equivalent to
/// `x << (32 − plaintext_bits)`.
///
/// # Complexity
///
/// `O(1)` — a single shift; expected to be inlined into the matvec.
#[inline]
pub(crate) fn round_p_to_q(x: u32, plaintext_bits: u32) -> u32 {
    debug_assert!((1..=31).contains(&plaintext_bits));
    debug_assert!(x < (1u32 << plaintext_bits), "x exceeds plaintext range");
    x << (32 - plaintext_bits)
}

/// Round `y ∈ Z_q` to the nearest multiple of `Δ = q / p`, then divide
/// by `Δ` to recover the `Z_p` value.
///
/// # Arguments
///
/// - `y`              — ciphertext-side scalar.
/// - `plaintext_bits` — log₂ of `p`; in `1..=31`.
///
/// # Returns
///
/// `⌊(y + Δ/2) / Δ⌋ mod p`, computed without an explicit modulo (the
/// shift discards the high bits).
///
/// # Rationale
///
/// `wrapping_add(Δ/2)` is intentional — the carry-out bits would be
/// shifted away by `>> (32 − plaintext_bits)` even if `+ Δ/2` overflowed.
///
/// # Complexity
///
/// `O(1)` — one add and one shift.
#[inline]
pub(crate) fn round_q_to_p(y: u32, plaintext_bits: u32) -> u32 {
    debug_assert!((1..=31).contains(&plaintext_bits));
    let shift = 32 - plaintext_bits;
    // Add Δ/2 with wrapping (the high bits we discard absorb any overflow).
    y.wrapping_add(1u32 << (shift - 1)) >> shift
}

#[cfg(test)]
mod tests {
    //! Unit tests for the Δ-scaling round-trip. Two invariants:
    //!
    //! 1. `round_q_to_p(round_p_to_q(x)) == x` for every valid `x`.
    //! 2. `round_q_to_p` rounds toward the nearest Δ-multiple (with
    //!    ties going up).

    use super::*;

    /// `round_p_to_q` then `round_q_to_p` is the identity on `Z_p` for
    /// every supported `plaintext_bits` and boundary value.
    #[test]
    fn round_trip_identity() {
        for &bits in &[1u32, 8, 9, 10, 16, 31] {
            let p = 1u32 << bits;
            // Test boundaries: 0, 1, p-2, p-1, plus a few mid-range values.
            let candidates: &[u32] = &[0, 1, p / 2, p.saturating_sub(2), p - 1];
            for &x in candidates {
                if x < p {
                    let lifted = round_p_to_q(x, bits);
                    let recovered = round_q_to_p(lifted, bits);
                    assert_eq!(
                        recovered, x,
                        "round-trip failed for x={x}, plaintext_bits={bits}"
                    );
                }
            }
        }
    }

    /// `round_q_to_p` snaps to the nearest Δ-multiple: values just
    /// below the midpoint round down, the midpoint itself rounds up.
    #[test]
    fn rounding_toward_nearest() {
        // plaintext_bits=8 → Δ = 2^24 = 16_777_216
        let bits = 8u32;
        let delta = 1u32 << (32 - bits); // 2^24

        // y = Δ*k + (Δ/2 - 1): should round down to k
        let k: u32 = 5;
        let y_low = delta.wrapping_mul(k).wrapping_add(delta / 2 - 1);
        assert_eq!(round_q_to_p(y_low, bits), k);

        // y = Δ*k + Δ/2: should round up to k+1 (or wrap)
        let y_high = delta.wrapping_mul(k).wrapping_add(delta / 2);
        assert_eq!(round_q_to_p(y_high, bits), k + 1);
    }

}
