# Contributing

Thanks for your interest in IKPIR! This document covers the mechanics of
getting a change accepted; the architecture itself is documented in the root
[`CLAUDE.md`](CLAUDE.md) and the per-crate `CLAUDE.md`/`README.md` files.

## Toolchain

The workspace pins Rust **1.85.0** via [`rust-toolchain.toml`](rust-toolchain.toml)
(rustup selects it automatically). 1.85 is also the declared MSRV
(`rust-version` in the root manifest). Bump the pin deliberately and in its
own PR — the paper's benchmark numbers are measured on the pinned channel.

## Local gates

Every PR must pass (the first four are enforced by CI; the rustdoc gate
is a good local habit):

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo bench --workspace --no-run --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
```

## Benches

- One bench at one config: `./scripts/bench.sh <name> [flags]` (auto-derives
  `--plaintext-bits` / `--lwe-dim`; CSVs land under `results/<crate>/`).
- Fast correctness pass over every PIR bench on both backends:
  `./scripts/smoke.sh` (~a couple of minutes).
- Never commit `results/` — it is git-ignored and regenerated on demand.

## Workspace conventions

- Crates live under `crates/`; version, edition, MSRV, license, lints, and
  `publish = false` are inherited from the root manifest — new crates opt in
  with `.workspace = true` keys and `[lints] workspace = true`.
- Add new dependencies to `[workspace.dependencies]` in the root manifest
  first, then reference them with `{ workspace = true }`. Keep the external
  dependency set lean; hand-rolled error types are house style (no
  `thiserror`/`anyhow`).
- Avoid dynamic dispatch on hot paths; the PIR backend is selected at the
  `B: IndexPirBackend` type parameter.

## Branches, commits, PRs

- Branch from an up-to-date `main`: `feature/…`, `perf/…`, `fix/…`,
  `chore/…`, `ci/…`, `docs/…`.
- Commit style: `type(scope): summary` (lowercase summary), e.g.
  `perf(ikpir-common): width-adaptive blocked matvec`.
- PRs target `main`. Keep each commit building green; CI must pass before
  review.

## Licensing

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be licensed as in [README § License](README.md#license) —
Apache-2.0 — without any additional terms or conditions.

Dependencies are held to the same standard: every crate in the resolved graph
must carry a permissive license that allows redistribution inside an Apache-2.0
work. `deny.toml` is the allow-list, enforced by the `licenses` CI job. Adding a
crate under a copyleft license (GPL, AGPL, LGPL, MPL, SSPL) will fail that job.
