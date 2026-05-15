//! FrodoPIR parameter types.
//!
//! # Purpose
//!
//! Two types:
//!
//! - [`FrodoParams`] — per-segment runtime values
//!   (`lwe_dim`, `plaintext_bits`, public seed). Built once per
//!   `server_setup` call and threaded into the matvec routines.
//! - [`FrodoConfig`] — user-facing tunable knobs persisted by
//!   `IkpirServer` and re-used across every `full_rebuild`.
//!
//! # Design / architecture
//!
//! `FrodoConfig` is the small, stable surface a deployment commits to;
//! `FrodoParams` is the larger object actually consumed by the math.
//! The split keeps the IKPIR server holding only the user-facing knob
//! and assembling the runtime params per-segment.
//!
//! # Related files
//!
//! - `backend.rs` — sole consumer; reads `FrodoParams` per call and
//!   `FrodoConfig` once per setup.
//! - `ikpir-server::IkpirServer` — persists `FrodoConfig` for the server's
//!   lifetime.

/// FrodoPIR LWE parameters.
///
/// # Purpose
///
/// Per-segment runtime values needed by the FrodoPIR matvec
/// (`lwe_dim`, `plaintext_bits`, public seed). One instance is built
/// per `server_setup` call.
///
/// # Rationale
///
/// `q = 2^32` is implicit (native `u32` wraparound; matches the
/// `Vec<u32>` hint shape required by `IndexPirBackend`). The error and
/// secret distributions are uniform ternary `{-1, 0, +1}` per
/// FrodoPIR §4.1 (Assumption 1); no σ is needed in the type.
#[derive(Clone, Debug)]
pub struct FrodoParams {
    /// LWE dimension n in the paper (default 1774 for 128-bit security).
    pub lwe_dim: u32,
    /// log₂ of the plaintext modulus p. Set per-call from the trait's
    /// `plaintext_bits` argument. Required: `1 ≤ plaintext_bits ≤ 31`.
    pub plaintext_bits: u32,
    /// Public seed for sampling A (paper's λ-bit μ; we use 128 bits).
    pub seed: [u8; 16],
}

impl FrodoParams {
    /// Default LWE dimension for 128-bit security (FrodoPIR Table 5).
    pub const DEFAULT_LWE_DIM: u32 = 1774;

    /// Construct LWE parameters with explicit dimension, plaintext bit
    /// width, and 128-bit seed used to sample the public matrix `A`.
    ///
    /// # Arguments
    ///
    /// - `lwe_dim`        — LWE dimension `n`; must be positive.
    /// - `plaintext_bits` — log₂ of the plaintext modulus; in `1..=31`.
    /// - `seed`           — 128-bit public seed for the `A` sampler.
    ///
    /// # Constraints
    ///
    /// Panics if `plaintext_bits` is outside `1..=31` or `lwe_dim == 0`.
    pub fn new(lwe_dim: u32, plaintext_bits: u32, seed: [u8; 16]) -> Self {
        assert!(
            (1..=31).contains(&plaintext_bits),
            "plaintext_bits must be in 1..=31, got {plaintext_bits}"
        );
        assert!(lwe_dim > 0, "lwe_dim must be positive");
        Self { lwe_dim, plaintext_bits, seed }
    }
}

/// User-facing tunable knobs for [`FrodoPirBackend`](crate::FrodoPirBackend).
///
/// # Purpose
///
/// The small, stable backend surface persisted by
/// `IkpirServer`. Threaded into every
/// `B::server_setup` call (initial setup *and* every `full_rebuild`),
/// so a single config controls the LWE dimension across the server's
/// lifetime.
///
/// # Rationale
///
/// Use [`FrodoConfig::default`] for the FrodoPIR Table-5 128-bit
/// security setting (`lwe_dim = 1774`); use
/// [`FrodoConfig::with_lwe_dim`] to override (e.g. for benchmarking a
/// reduced parameter set).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrodoConfig {
    /// LWE dimension `n`. Larger values give stronger security at higher
    /// preprocessing/query cost. Must be > 0.
    pub lwe_dim: u32,
}

impl FrodoConfig {
    /// New config with the supplied LWE dimension.
    ///
    /// # Arguments
    ///
    /// - `lwe_dim` — LWE dimension `n`; must be positive.
    ///
    /// # Constraints
    ///
    /// Panics if `lwe_dim == 0`.
    pub fn with_lwe_dim(lwe_dim: u32) -> Self {
        assert!(lwe_dim > 0, "lwe_dim must be positive");
        Self { lwe_dim }
    }
}

impl Default for FrodoConfig {
    /// 128-bit security default (`lwe_dim = 1774`, FrodoPIR Table 5).
    fn default() -> Self {
        Self { lwe_dim: FrodoParams::DEFAULT_LWE_DIM }
    }
}
