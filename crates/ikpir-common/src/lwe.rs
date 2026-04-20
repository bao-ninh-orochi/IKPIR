//! LWE-style primitives used by the PIR protocol.
//!
//! We use the same parameterisation as the reference ChalametPIR: a ciphertext
//! modulus `q = 2^32`, a secret sampled from the uniform ternary distribution
//! `{-1, 0, 1}`, and an identical ternary-error distribution for masking.
//! Ternary distributions are chosen both for simplicity (they need no rejection
//! sampling and no pre-computed tables) and because at the chosen LWE
//! dimension of `1774` they deliver roughly 128-bit classical security.
//!
//! # What lives here
//!
//! * [`sample_ternary`] — uniform ternary sample via a single `u32` draw.
//! * [`sample_ternary_vec`] — length-`n` ternary vector.
//! * [`encode_value`] / [`decode_value`] — map record bytes ⇆ a vector of
//!   **raw plaintext payloads** in `[0, 2^{mat_elem_bit_len})`. The `Δ`
//!   scaling is applied separately by the client/server at the LWE-ciphertext
//!   boundary (see [`crate::params::FilterParams::scale`]); keeping the
//!   database matrix in the plaintext space is what makes the LWE noise
//!   bound close.
//! * [`round_and_extract`] — nearest-multiple-of-`Δ` rounding followed by
//!   extraction of the plaintext payload; used by the client on decrypt.
//!
//! # Non-goals
//!
//! * No `zeroize` of secret material (tracked as hardening follow-up).
//!   Phase B is a research artefact; callers who need it can wrap secrets
//!   in their own zeroizing container.

use rand::RngCore;

use crate::params::{FilterParams, CIPHERTEXT_MODULUS_BITS};

/// Sample one `u32` from the uniform ternary distribution `{-1, 0, 1}`,
/// represented in `Z_{2^32}` (so `-1` maps to `u32::MAX`).
///
/// Uses two bits of a single RNG draw: `(b0, b1) ∈ {00, 01, 10, 11}` → `-1`,
/// `0`, `0`, `+1` with probabilities `1/4, 1/4, 1/4, 1/4` respectively.
/// That yields the target support with a minor imbalance at `0` (mass `1/2`
/// instead of `1/3`), which matches the reference implementation and is
/// standard for LWE "uniform ternary" instantiations.
pub fn sample_ternary<R: RngCore>(rng: &mut R) -> u32 {
    let bits = rng.next_u32();
    match bits & 0b11 {
        0b00 => u32::MAX, // -1 mod 2^32
        0b11 => 1,
        _ => 0,
    }
}

/// Length-`n` ternary sample (writes a fresh vector).
pub fn sample_ternary_vec<R: RngCore>(rng: &mut R, n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(sample_ternary(rng));
    }
    out
}

/// Encode a byte slice into `params.row_elems` raw plaintext payloads.
///
/// Splits the value into chunks of `mat_elem_bit_len` bits (least-significant
/// bit first) and returns the resulting `u32` vector, each element in
/// `[0, 2^{mat_elem_bit_len})`. Padding (bits beyond `value_len * 8`) is zero.
///
/// **No scaling is applied.** Callers that need the LWE-scaled form should
/// multiply each payload by [`FilterParams::scale`] at use-site.
///
/// Returns `None` if `value.len() != params.value_len`.
pub fn encode_value(value: &[u8], params: &FilterParams) -> Option<Vec<u32>> {
    if value.len() != params.value_len as usize {
        return None;
    }
    let bit_len = params.mat_elem_bit_len;
    if bit_len == 0 || bit_len >= CIPHERTEXT_MODULUS_BITS {
        return None;
    }
    let row_elems = params.row_elems as usize;
    let mask = (1u32 << bit_len) - 1;
    let mut out = vec![0u32; row_elems];

    for (idx, slot) in out.iter_mut().enumerate() {
        let bit_cursor = idx as u64 * bit_len as u64;
        let byte_pos = (bit_cursor / 8) as usize;
        let bit_off = (bit_cursor % 8) as u32;
        if byte_pos >= value.len() {
            break;
        }
        // Load up to 8 bytes from `byte_pos` into a 64-bit window, then mask.
        // `bit_len ≤ 14 < 32`, so the windowed-shift fits comfortably.
        let mut window: u64 = 0;
        for i in 0..8 {
            if byte_pos + i < value.len() {
                window |= (value[byte_pos + i] as u64) << (i * 8);
            }
        }
        *slot = ((window >> bit_off) as u32) & mask;
    }
    Some(out)
}

/// Inverse of [`encode_value`]: convert `row_elems` raw payloads (each in
/// `[0, 2^{mat_elem_bit_len})`) back into a `value_len`-byte vector.
pub fn decode_value(payloads: &[u32], params: &FilterParams) -> Option<Vec<u8>> {
    if payloads.len() != params.row_elems as usize {
        return None;
    }
    let bit_len = params.mat_elem_bit_len;
    if bit_len == 0 || bit_len >= CIPHERTEXT_MODULUS_BITS {
        return None;
    }
    let mask = (1u32 << bit_len) - 1;
    let value_len = params.value_len as usize;
    let mut out = vec![0u8; value_len];

    for (idx, &payload_raw) in payloads.iter().enumerate() {
        let bit_cursor = idx as u64 * bit_len as u64;
        let byte_pos = (bit_cursor / 8) as usize;
        let bit_off = (bit_cursor % 8) as u32;
        if byte_pos >= value_len {
            break;
        }
        let payload = payload_raw & mask;
        let shifted = (payload as u64) << bit_off;
        // `shifted` occupies at most `bit_off + bit_len ≤ 7 + 14 = 21` bits,
        // so 3 bytes of span is always enough.
        for i in 0..3usize {
            if byte_pos + i < value_len {
                out[byte_pos + i] |= (shifted >> (i * 8)) as u8;
            }
        }
    }
    Some(out)
}

/// Round every element of `slots` to the nearest multiple of
/// `Δ = 2^{32 − mat_elem_bit_len}` and extract the raw payload.
///
/// Used by the client after subtracting the noise-mask `s · H` from the
/// server's response. Rounding succeeds as long as the accumulated LWE error
/// stayed within `Δ/2` of the nearest scaled-payload target — which the
/// parameter selector guarantees.
///
/// Returns payloads in `[0, 2^{mat_elem_bit_len})`, one per input slot.
pub fn round_and_extract(slots: &[u32], params: &FilterParams) -> Vec<u32> {
    let bit_len = params.mat_elem_bit_len;
    let scale = params.scale();
    let half_scale = scale / 2;
    let mask = (1u32 << bit_len) - 1;
    slots
        .iter()
        .map(|&s| {
            // Round-half-up toward the nearest multiple of Δ. `s + Δ/2` overflows
            // deliberately: we work in Z_{2^32} so the carry out is the right
            // thing (it rolls payload value `2^bit_len - 1` back to `0`).
            let rounded = s.wrapping_add(half_scale);
            (rounded / scale) & mask
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Arity, FilterParams};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn params_for(value_len: u32) -> FilterParams {
        FilterParams::recommended(Arity::Segmented4, 1024, value_len).unwrap()
    }

    #[test]
    fn ternary_sample_in_support() {
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        for _ in 0..10_000 {
            let s = sample_ternary(&mut rng);
            assert!(
                s == 0 || s == 1 || s == u32::MAX,
                "value {} out of support",
                s
            );
        }
    }

    #[test]
    fn ternary_vector_length() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let v = sample_ternary_vec(&mut rng, 500);
        assert_eq!(v.len(), 500);
    }

    #[test]
    fn encode_decode_round_trip_32_bytes() {
        let params = params_for(32);
        let mut value = [0u8; 32];
        for (i, b) in value.iter_mut().enumerate() {
            *b = i as u8;
        }
        let payloads = encode_value(&value, &params).unwrap();
        let dec = decode_value(&payloads, &params).unwrap();
        assert_eq!(dec, value.to_vec());
    }

    #[test]
    fn encode_decode_round_trip_various_lengths() {
        for len in [1u32, 7, 16, 31, 63, 128] {
            let params = params_for(len);
            let mut v = vec![0u8; len as usize];
            for (i, b) in v.iter_mut().enumerate() {
                *b = ((i * 37) & 0xFF) as u8;
            }
            let payloads = encode_value(&v, &params).unwrap();
            let dec = decode_value(&payloads, &params).unwrap();
            assert_eq!(dec, v);
        }
    }

    #[test]
    fn encode_rejects_wrong_length() {
        let params = params_for(32);
        assert!(encode_value(&[0u8; 31], &params).is_none());
        assert!(encode_value(&[0u8; 33], &params).is_none());
    }

    #[test]
    fn round_and_extract_tolerates_small_noise() {
        let params = params_for(32);
        let mut value = [0u8; 32];
        for (i, b) in value.iter_mut().enumerate() {
            *b = (i * 13) as u8;
        }
        let payloads = encode_value(&value, &params).unwrap();
        let scale = params.scale();
        let tolerance = (scale / 2).saturating_sub(16);
        // Lift to scaled representation, add noise, round-and-extract, decode.
        let noisy: Vec<u32> = payloads
            .iter()
            .map(|&p| p.wrapping_mul(scale).wrapping_add(tolerance))
            .collect();
        let recovered = round_and_extract(&noisy, &params);
        let dec = decode_value(&recovered, &params).unwrap();
        assert_eq!(dec, value.to_vec());
    }
}
