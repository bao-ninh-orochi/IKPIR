# Reproducing paper results

This document is the **artifact-evaluation entry point**. A reviewer should
be able to clone the repo, install the prerequisites, run one command, and
reproduce every figure in the paper. If anything here is unclear or wrong,
please open an issue.

## Prerequisites

- **Rust toolchain.** Pinned to the version in [`rust-toolchain.toml`](rust-toolchain.toml).
  `rustup` auto-installs it on first `cargo` invocation.
- **`just`** (command runner). Install via `cargo install just` or your package
  manager.
- **Python 3.10+** for plotting. Run `just plots-setup` once to create a local
  `.venv` with the pinned dependencies from `scripts/requirements.txt`.

That's it — no system libraries, no CUDA, no network access at runtime.

## Hardware used for the paper

<!-- Fill in before submission. -->

| Field | Value |
|---|---|
| CPU | <TODO-CPU-MODEL, e.g. Apple M2 Pro / AMD 7950X> |
| Cores | <TODO-CORE-COUNT> (used: <TODO-CORES-USED>) |
| RAM | <TODO-RAM-GB> |
| OS | <TODO-OS, e.g. macOS 14.3 / Ubuntu 22.04> |
| `rustc` | (from `rust-toolchain.toml`) |

## Commands

| What you want | Command | Wall-clock | RAM |
|---|---|---|---|
| Smoke test | `just repro-smoke` | <1 min | <1 GB |
| All SCF benches | `just bench-scf-portable` | <TODO-HH:MM> | <TODO-GB> |
| All PIR benches | `just bench-pir-portable` | <TODO-HH:MM> | <TODO-GB> |
| Plots only | `just plots` | <10 s | <1 GB |
| Numerical verification | `just verify` | <10 s | <1 GB |
| Everything | `just repro-all` | <TODO-HH:MM> | <TODO-GB> |

Benches emit CSVs into `crates/<crate>/results/`. `just bench-*-portable`
auto-copies the canonical subset into `results/paper/`. Plots read from
`results/paper/` and write into `results/plots/`.

## What "success" means numerically

`scripts/verify_results.py` encodes these assertions (run via `just verify`).
Tolerances are quoted in-file.

| Claim | Source | Tolerance |
|---|---|---|
| SCF 4-ary, b=4 load factor ≥ 0.94 | Table <TODO>, Figure <TODO> | ±0.005 |
| SCF segmented vs. standard: +<TODO>% load factor | Figure <TODO> | relative |
| PIR per-query latency ≤ <TODO> ms (n=2^20) | Table <TODO> | ±10% |
| Incremental update vs. full rebuild break-even at n=<TODO> | Figure <TODO> | ±1 bucket |

## Variance and reporting

- Every bench reports mean + min + max + population stddev over `TRIALS`
  trials (see each bench's module doc comment for the exact count).
- RNG seeds are fixed inside each bench so runs are deterministic modulo
  parallel scheduling.
- Portable codegen is required for quoted paper numbers; native codegen
  (`just bench-*-native`) produces *different* numbers on different µarchs.

## Troubleshooting

- **`cargo` not found.** Install rustup: <https://rustup.rs>.
- **`just: command not found.`** `cargo install just`.
- **Out of memory on 2^20 configs.** Lower the top bench size by editing the
  `sizes` vectors in `crates/segmented-cuckoo-filter/benches/*.rs`. Paper
  numbers require 2^20; smaller runs are still directionally correct.

## Archival DOI

Once the paper is accepted, a tagged release of this repo is archived on
Zenodo. The DOI appears in [`CITATION.cff`](CITATION.cff).
