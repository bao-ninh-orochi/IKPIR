//! Hashing helpers layered on top of the SCF's own hash schedule.
//!
//! Two concerns live here, both cheap and free of secret data:
//!
//!   * **Candidate derivation.** Given a keyword and a [`FilterParams`], return
//!     the SCF candidate bucket indices and fingerprint. This is the bridge
//!     between application-level keys and the PIR row layout. The core work
//!     delegates to [`segmented_cuckoo_filter::IndexScheme::hash_item`]; the
//!     wrapper just picks the right concrete scheme from the [`Arity`].
//!
//!   * **Seed expansion for the public matrix.** Given a 32-byte server seed,
//!     deterministically stream `u32` values for the LWE matrix `A`. We use
//!     ChaCha20 as a PRG — it is already a workspace dependency, fast, and
//!     widely trusted.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use segmented_cuckoo_filter::{
    IndexScheme, Segmented3aryScheme, Segmented4aryScheme, SegmentedScheme, Standard3aryScheme,
    Standard4aryScheme, StandardScheme,
};

use crate::params::{Arity, FilterParams, SEED_LEN};

/// A keyword's footprint in the PIR row space.
///
/// Returned by [`candidates`]. The first `degree` entries of `indices` hold
/// valid bucket indices; remaining slots are padded with zero. This mirrors
/// the SCF's own return shape so the two APIs compose cleanly.
#[derive(Debug, Clone, Copy)]
pub struct KeywordFootprint {
    /// SCF fingerprint tag (non-zero for any valid keyword).
    pub tag: u32,
    /// Candidate bucket indices — valid entries at `[0, degree)`.
    pub indices: [u32; 4],
    /// Arity degree (`2`, `3`, or `4`); `indices[degree..]` are padding zeros.
    pub degree: u32,
}

impl KeywordFootprint {
    /// Slice of actually-valid candidate indices.
    pub fn slots(&self) -> &[u32] {
        &self.indices[..self.degree as usize]
    }
}

/// Derive the SCF fingerprint + candidate bucket indices for `key`.
///
/// Thin dispatcher over the six concrete `IndexScheme` types. `num_buckets`
/// must already be shaped for the arity (power-of-2 / power-of-3 / k·2^m as
/// appropriate); construction-time sizing lives in
/// [`crate::params::FilterParams::recommended`].
pub fn candidates(key: &[u8], params: &FilterParams) -> KeywordFootprint {
    let (tag, indices) = match params.arity {
        Arity::Standard2 => StandardScheme {
            num_buckets: params.num_buckets,
        }
        .hash_item(key, params.fingerprint_bits),
        Arity::Segmented2 => {
            let half = params.num_buckets / 2;
            SegmentedScheme { half }.hash_item(key, params.fingerprint_bits)
        }
        Arity::Standard3 => Standard3aryScheme {
            num_buckets: params.num_buckets,
        }
        .hash_item(key, params.fingerprint_bits),
        Arity::Segmented3 => {
            let segment_size = params.num_buckets / 3;
            Segmented3aryScheme { segment_size }.hash_item(key, params.fingerprint_bits)
        }
        Arity::Standard4 => Standard4aryScheme {
            num_buckets: params.num_buckets,
        }
        .hash_item(key, params.fingerprint_bits),
        Arity::Segmented4 => {
            let segment_size = params.num_buckets / 4;
            Segmented4aryScheme { segment_size }.hash_item(key, params.fingerprint_bits)
        }
    };
    KeywordFootprint {
        tag,
        indices,
        degree: params.degree(),
    }
}

/// Stream a deterministic sequence of `u32`s from a 32-byte seed.
///
/// Used to expand the public LWE matrix `A`. ChaCha20 is a cryptographic PRG,
/// which is more than strong enough for a public matrix — the choice is driven
/// by availability (`rand_chacha` is already in workspace deps) and speed.
pub fn prg_u32_stream(seed: &[u8; SEED_LEN], len: usize) -> Vec<u32> {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(rand::RngCore::next_u32(&mut rng));
    }
    out
}

/// Derive `k` per-segment seeds from a single master seed.
///
/// Seeds a ChaCha20 PRG with `seed_mu`, pulls `k · SEED_LEN` bytes in one
/// shot, and chunks the output into k fixed-size seeds. Deterministic for
/// identical `(seed_mu, k)`; every segment seed differs with overwhelming
/// probability because they are consecutive non-overlapping PRG outputs.
///
/// Used by both client and server to expand the per-segment public
/// matrices `A_j` independently.
pub fn derive_segment_seeds(seed_mu: &[u8; SEED_LEN], k: u32) -> Vec<[u8; SEED_LEN]> {
    let mut rng = ChaCha20Rng::from_seed(*seed_mu);
    let k_us = k as usize;
    let mut buf = vec![0u8; SEED_LEN * k_us];
    rand::RngCore::fill_bytes(&mut rng, &mut buf);
    let mut out = Vec::with_capacity(k_us);
    for chunk in buf.chunks_exact(SEED_LEN) {
        let mut seed = [0u8; SEED_LEN];
        seed.copy_from_slice(chunk);
        out.push(seed);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Arity, FilterParams};

    fn make_params_for_arity(arity: Arity, _approx_buckets: u32, value_len: u32) -> FilterParams {
        if arity.is_segmented() {
            FilterParams::recommended(arity, _approx_buckets, value_len).unwrap()
        } else {
            // Standard arities are not supported by `recommended`; construct directly.
            // Use valid num_buckets per the scheme's constraint:
            //   Standard2: power of 2  → 256 = 2^8
            //   Standard3: power of 3  → 243 = 3^5
            //   Standard4: power of 4  → 256 = 4^4
            let num_buckets = match arity {
                Arity::Standard3 => 243,
                _ => 256,
            };
            FilterParams {
                arity,
                num_buckets,
                fingerprint_bits: 12,
                row_elems: (value_len as u64 * 8).div_ceil(8) as u32,
                mat_elem_bit_len: 8,
                value_len,
            }
        }
    }

    #[test]
    fn candidates_are_in_range_for_all_arities() {
        let value_len = 32;
        for &arity in &[
            Arity::Standard2,
            Arity::Segmented2,
            Arity::Standard3,
            Arity::Segmented3,
            Arity::Standard4,
            Arity::Segmented4,
        ] {
            let p = make_params_for_arity(arity, 256, value_len);
            for i in 0u32..100 {
                let foot = candidates(&i.to_le_bytes(), &p);
                assert_eq!(foot.degree, arity.degree(), "arity={:?}", arity);
                assert_ne!(foot.tag, 0);
                for &idx in foot.slots() {
                    assert!(idx < p.num_buckets, "arity={:?} idx={}", arity, idx);
                }
            }
        }
    }

    #[test]
    fn candidates_are_deterministic() {
        let p = FilterParams::recommended(Arity::Segmented4, 256, 32).unwrap();
        let a = candidates(b"hello", &p);
        let b = candidates(b"hello", &p);
        assert_eq!(a.tag, b.tag);
        assert_eq!(a.indices, b.indices);
    }

    #[test]
    fn prg_is_deterministic_and_seed_sensitive() {
        let seed_a = [7u8; SEED_LEN];
        let mut seed_b = [7u8; SEED_LEN];
        seed_b[0] = 8;
        let a1 = prg_u32_stream(&seed_a, 100);
        let a2 = prg_u32_stream(&seed_a, 100);
        let b = prg_u32_stream(&seed_b, 100);
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn derive_segment_seeds_is_deterministic_and_seed_sensitive() {
        let seed_a = [42u8; SEED_LEN];
        let mut seed_b = [42u8; SEED_LEN];
        seed_b[0] ^= 0xFF;
        let a1 = derive_segment_seeds(&seed_a, 4);
        let a2 = derive_segment_seeds(&seed_a, 4);
        let b = derive_segment_seeds(&seed_b, 4);
        assert_eq!(a1, a2, "identical inputs must yield identical outputs");
        assert_eq!(a1.len(), 4);
        assert_ne!(
            a1, b,
            "different master seed must yield different segment seeds"
        );
    }

    #[test]
    fn derive_segment_seeds_are_pairwise_distinct() {
        let seed = [7u8; SEED_LEN];
        for k in [2u32, 3, 4] {
            let seeds = derive_segment_seeds(&seed, k);
            assert_eq!(seeds.len(), k as usize);
            for i in 0..k as usize {
                for j in (i + 1)..k as usize {
                    assert_ne!(seeds[i], seeds[j], "segment seeds {i} and {j} collided");
                }
            }
        }
    }

    #[test]
    fn derive_segment_seeds_prefix_stable_across_k() {
        // With the same master seed, the first two derived seeds must not depend on k —
        // both come from the same ChaCha20 PRG prefix regardless of total length.
        let seed = [13u8; SEED_LEN];
        let k2 = derive_segment_seeds(&seed, 2);
        let k4 = derive_segment_seeds(&seed, 4);
        assert_eq!(k2[0], k4[0]);
        assert_eq!(k2[1], k4[1]);
    }

    #[test]
    fn segmented4_candidates_live_in_distinct_segments() {
        let p = FilterParams::recommended(Arity::Segmented4, 1000, 32).unwrap();
        let seg = p.num_buckets / 4;
        for i in 0u32..200 {
            let foot = candidates(&i.to_le_bytes(), &p);
            for (pos, &idx) in foot.slots().iter().enumerate() {
                let low = pos as u32 * seg;
                let high = low + seg;
                assert!(idx >= low && idx < high);
            }
        }
    }
}
