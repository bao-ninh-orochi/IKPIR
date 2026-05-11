use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Build a ChaCha20Rng from the 16-byte seed (zero-padded to 32 bytes).
fn rng_from_seed(seed: &[u8; 16]) -> ChaCha20Rng {
    let mut padded = [0u8; 32];
    padded[..16].copy_from_slice(seed);
    ChaCha20Rng::from_seed(padded)
}

/// Sample the public matrix A in row-major shape `(n_rows × lwe_dim)`.
/// Each cell is a fresh `u32` from ChaCha20. Output length = `n_rows * lwe_dim`.
///
/// Determinism: same `seed` + same `(n_rows, lwe_dim)` → identical output.
pub(crate) fn sample_a(seed: &[u8; 16], n_rows: u32, lwe_dim: u32) -> Vec<u32> {
    let len = (n_rows as usize)
        .checked_mul(lwe_dim as usize)
        .expect("A dimensions overflow usize");
    let mut rng = rng_from_seed(seed);
    let mut out = vec![0u32; len];
    for cell in &mut out {
        *cell = rng.next_u32();
    }
    out
}

/// Fill `dst` with uniform ternary `{-1, 0, +1}` samples, using `u32`
/// wraparound to encode `-1` as `u32::MAX`. Uses 2-bit chunks from the
/// PRG with rejection sampling on `0b11` to avoid bias.
pub(crate) fn sample_ternary_into<R: RngCore>(rng: &mut R, dst: &mut [u32]) {
    let mut idx = 0;
    while idx < dst.len() {
        let mut word = rng.next_u32();
        for _ in 0..16 {
            if idx == dst.len() {
                return;
            }
            match word & 0b11 {
                0 => {
                    dst[idx] = 0;
                    idx += 1;
                }
                1 => {
                    dst[idx] = 1;
                    idx += 1;
                }
                2 => {
                    dst[idx] = u32::MAX; // -1 mod 2^32
                    idx += 1;
                }
                _ => { /* 0b11 → reject */ }
            }
            word >>= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LWE_DIM: u32 = crate::backend::frodo::FrodoParams::DEFAULT_LWE_DIM;

    #[test]
    fn sample_a_determinism() {
        let seed = [0x42u8; 16];
        let a1 = sample_a(&seed, 4, LWE_DIM);
        let a2 = sample_a(&seed, 4, LWE_DIM);
        assert_eq!(a1, a2);
    }

    #[test]
    fn sample_a_distinctness() {
        let seed1 = [0x11u8; 16];
        let seed2 = [0x22u8; 16];
        let a1 = sample_a(&seed1, 4, LWE_DIM);
        let a2 = sample_a(&seed2, 4, LWE_DIM);
        let differ = a1.iter().zip(a2.iter()).filter(|(x, y)| x != y).count();
        assert!(
            differ * 100 >= a1.len() * 99,
            "fewer than 99% cells differ: {differ}/{}",
            a1.len()
        );
    }

    #[test]
    fn sample_a_shape() {
        let seed = [0u8; 16];
        let n_rows = 4u32;
        let a = sample_a(&seed, n_rows, LWE_DIM);
        assert_eq!(a.len(), (n_rows * LWE_DIM) as usize);
    }

    #[test]
    fn ternary_distribution() {
        let n: usize = 100_000;
        let mut rng = rng_from_seed(&[0xABu8; 16]);
        let mut dst = vec![0u32; n];
        sample_ternary_into(&mut rng, &mut dst);

        let count_zero = dst.iter().filter(|&&x| x == 0).count();
        let count_one = dst.iter().filter(|&&x| x == 1).count();
        let count_neg = dst.iter().filter(|&&x| x == u32::MAX).count();

        // σ = sqrt(2N/9) for each bucket under uniform multinomial
        let sigma = ((2.0 * n as f64) / 9.0).sqrt();
        let lo = (n as f64 / 3.0 - 5.0 * sigma) as usize;
        let hi = (n as f64 / 3.0 + 5.0 * sigma) as usize;

        assert!(
            (lo..=hi).contains(&count_zero),
            "zero count {count_zero} out of [{lo}, {hi}]"
        );
        assert!(
            (lo..=hi).contains(&count_one),
            "one count {count_one} out of [{lo}, {hi}]"
        );
        assert!(
            (lo..=hi).contains(&count_neg),
            "neg count {count_neg} out of [{lo}, {hi}]"
        );
    }

    #[test]
    fn ternary_determinism() {
        let mut rng1 = rng_from_seed(&[0x77u8; 16]);
        let mut rng2 = rng_from_seed(&[0x77u8; 16]);
        let mut dst1 = vec![0u32; LWE_DIM as usize];
        let mut dst2 = vec![0u32; LWE_DIM as usize];
        sample_ternary_into(&mut rng1, &mut dst1);
        sample_ternary_into(&mut rng2, &mut dst2);
        assert_eq!(dst1, dst2);
    }

    #[test]
    fn ternary_support() {
        let mut rng = rng_from_seed(&[0x55u8; 16]);
        let mut dst = vec![0u32; LWE_DIM as usize];
        sample_ternary_into(&mut rng, &mut dst);
        for &v in &dst {
            assert!(
                v == 0 || v == 1 || v == u32::MAX,
                "unexpected ternary value: {v}"
            );
        }
    }
}
