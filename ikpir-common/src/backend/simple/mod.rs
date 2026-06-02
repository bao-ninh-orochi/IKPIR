//! SimplePIR backend — LWE Index-PIR with discrete-Gaussian errors and a
//! `√N × √N` database reshape.
//!
//! # Purpose
//!
//! The second shipped backend, alongside [`crate::FrodoPirBackend`]. Same
//! trait family (mandatory [`crate::IndexPirBackend`] + the three optional
//! extensions), same wire-format bundles, same role inside
//! `IkpirServer<S, B>` / `IkpirClient<B>` — only the LWE distributions and
//! the per-segment matrix shape differ.
//!
//! # Math summary
//!
//! `q = 2³²` (implicit), `p = 2^plaintext_bits`, `Δ = q / p`. Per segment,
//! the cell array of shape `(n_rows × row_width)` is **logically reshaped**
//! into a square-ish matrix `D` of shape `(R × C)` where each row of `D`
//! contains exactly `k` consecutive cuckoo buckets. `A` of shape
//! `(R × lwe_dim)` is sampled from a per-segment public seed; the hint
//! `H = Aᵀ · D` has shape `(lwe_dim × C)`. Per query:
//!
//! ```text
//! reshape_row    = bucket_idx / k
//! bucket_within  = bucket_idx % k        (client-only; never sent to server)
//! b = A·s + e + Δ·u_{reshape_row}        length R
//! a = bᵀ·D                               length C
//! decode: residual = a − sᵀ·H, then slice [bucket_within·row_width
//!                                          .. (bucket_within+1)·row_width]
//!         round each cell to nearest Δ-multiple
//! ```
//!
//! # Distributions (vs FrodoPIR)
//!
//! | Quantity | FrodoPIR | SimplePIR |
//! |---|---|---|
//! | LWE secret `s` | uniform ternary `{-1, 0, +1}` | uniform `Z_q` (`u32`) |
//! | LWE error  `e` | uniform ternary `{-1, 0, +1}` | discrete Gaussian, σ = 6.4 |
//! | Default `lwe_dim` | 1566 (lattice estimator, ADPS16) | 1275 (lattice estimator, ADPS16) |
//!
//! Both choices give 128-bit security on `q = 2³²` per the respective
//! parameter analyses. SimplePIR's LWE assumption (§3.1 / Fig. 2) samples
//! the secret `s ←R Z_q^n` uniformly and the error `e ← χ^√N` from the
//! discrete-Gaussian distribution `χ` of standard deviation σ; §4.2 sets
//! `σ = 6.4` for 128-bit security at correctness error `δ = 2⁻⁴⁰`. The
//! reference implementation at `github.com/ahenzinger/simplepir` matches.
//!
//! The implementation approximates the discrete Gaussian by a rounded
//! continuous Gaussian (Box–Muller in `sampler.rs`); for `σ ≈ 6.4` this
//! is statistically indistinguishable from a strict discrete-Gaussian
//! sampler for the lattice-attack regimes relevant here.
//!
//! # Reshape derivation
//!
//! Goal: pick `k` so the reshaped `(R × C)` matrix is roughly square.
//! `R = ⌈n_rows / k⌉`, `C = k · row_width`. Setting `R ≈ C` gives
//! `k² ≈ n_rows / row_width`, hence
//! `k = max(1, round(√(n_rows / row_width)))`. If `k ∤ n_rows`, the last
//! reshape row is implicitly padded with zero cells — zero contributes
//! nothing to `bᵀ·D`, so decode is unaffected.
//!
//! # Concurrency contract
//!
//! Same as FrodoPIR: `client_query` mutates the in-flight slot in
//! [`SimpleClientState`]; `client_decode`
//! consumes it. Single in-flight query per segment per state.
//!
//! # Related files
//!
//! - `backend.rs` — [`SimplePirBackend`] + the five associated types + the
//!   four trait impls.
//! - `params.rs`  — user-facing [`SimpleConfig`] and per-segment runtime
//!   [`SimpleParams`].
//! - `arith.rs`   — `Δ`-scaling between `Z_p` and `Z_q` (duplicated from
//!   `frodo/arith.rs` per project rule "don't refactor beyond what the
//!   task requires").
//! - `sampler.rs` — `sample_a` (deterministic public matrix), uniform
//!   `Z_q` secret sampler, and Box–Muller discrete-Gaussian error sampler.

mod arith;
mod backend;
mod params;
mod sampler;

pub use backend::{
    SimpleClientState, SimpleHint, SimpleHintMaterial, SimplePirBackend, SimpleQuery,
    SimpleResponse, SimpleServerParams,
};
pub use params::{SimpleConfig, SimpleParams};

pub(crate) use arith::{round_p_to_q, round_q_to_p};
pub(crate) use sampler::{sample_a, sample_discrete_gaussian_into, sample_uniform_zq_into};
