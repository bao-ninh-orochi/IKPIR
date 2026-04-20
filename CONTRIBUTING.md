# Contributing

This is a research repository accompanying an academic paper. External
contributions are welcome but discretionary — if you'd like to propose a
substantive change, please open an issue first to discuss fit.

## Development setup

```bash
git clone https://github.com/<TODO-GITHUB-USER>/incremental-keyword-pir
cd incremental-keyword-pir
rustup show           # trigger auto-install of the pinned toolchain
cargo install just    # if you don't have it
just plots-setup      # one-time Python venv for scripts/plot.py
```

## Pre-merge gates

Every PR must pass the same matrix CI runs. You can run the full gate
locally with one command:

```bash
just ci
```

Which runs:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo build --workspace --release`

Paper figures (`just repro-smoke`) are smoke-tested in CI too — if your
change alters a bench output schema, update `scripts/plot.py` and
`results/SCHEMA.md` in the same PR.

## Branch naming

- `feat/<short-desc>` for new functionality.
- `fix/<short-desc>` for bug fixes.
- `bench/<short-desc>` for bench-only additions.
- `docs/<short-desc>` for documentation-only changes.

## Commit messages

Conventional Commits-ish. The first line is a short imperative sentence
under 70 characters. Body explains *why*, not *what* — the diff already
shows what changed.

## Adding a benchmark

1. Create `crates/<crate>/benches/<name>.rs`. Mirror the pattern of an
   existing bench: doc comment explaining **Intent / Method / Sweep /
   Parameters / Output**, then CSV writer from `helpers.rs`.
2. Register in that crate's `Cargo.toml` under `[[bench]]` with
   `harness = false`.
3. Document the output schema in `results/SCHEMA.md`.
4. Wire into `scripts/plot.py` if the CSV backs a paper figure.
5. Commit a small "reference" CSV to `results/paper/<crate>/` so
   `scripts/verify_results.py` can sanity-check it.

## Running only one bench

```bash
cargo bench -p segmented-cuckoo-filter --bench load_factor
```

## Code of Conduct

By participating, you agree to uphold the [Code of Conduct](CODE_OF_CONDUCT.md).
