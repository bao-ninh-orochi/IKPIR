/// Lift x ∈ Z_p into Z_q where q = 2^32 by multiplying by Δ = q / p
/// (i.e. shift left by `32 - plaintext_bits`). Cells must already satisfy
/// the high-bits-zero invariant `x < 2^plaintext_bits`.
#[inline]
pub(crate) fn round_p_to_q(x: u32, plaintext_bits: u32) -> u32 {
    debug_assert!((1..=31).contains(&plaintext_bits));
    debug_assert!(x < (1u32 << plaintext_bits), "x exceeds plaintext range");
    x << (32 - plaintext_bits)
}

/// Round y ∈ Z_q to nearest multiple of Δ = q / p, then divide by Δ to
/// recover the Z_p value: `⌊(y + Δ/2) / Δ⌋ mod p`.
#[inline]
pub(crate) fn round_q_to_p(y: u32, plaintext_bits: u32) -> u32 {
    debug_assert!((1..=31).contains(&plaintext_bits));
    let shift = 32 - plaintext_bits;
    // Add Δ/2 with wrapping (the high bits we discard absorb any overflow).
    y.wrapping_add(1u32 << (shift - 1)) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

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
