# One-command entry point for development, CI, and paper reproduction.
#
# Usage:
#   just                   # default = `just help`
#   just build             # build all workspace crates
#   just test              # run the full test matrix
#   just repro-smoke       # <1 min: build + test + tiny bench + plots
#   just repro-all         # full paper reproduction (slow)
#
# Portable vs. native: Portable recipes set neutral RUSTFLAGS to make numbers
# comparable across machines. Native recipes enable `target-cpu=native` for
# local exploration — do NOT use them for paper figures.

default: help

help:
    @just --list

# ---------------------------------------------------------------------------
# Build / test / lint
# ---------------------------------------------------------------------------

build:
    cargo build --workspace --all-targets

build-release:
    cargo build --workspace --release

# Runs lib + integration tests + examples + doctests across the workspace.
# Intentionally excludes benches (they are long-running standalone binaries;
# `cargo bench` drives them).
test:
    cargo test --workspace

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

# Lints everything (including benches) so CI still catches issues in bench
# code. Tests/benches are free to use unwrap/expect/panic — the workspace
# lints do not deny them (see Cargo.toml).
lint:
    cargo clippy --workspace --all-targets -- -D warnings

docs:
    cargo doc --workspace --no-deps

docs-open:
    cargo doc --workspace --no-deps --open

clean:
    cargo clean
    rm -rf crates/*/results

# Run every pre-merge gate (what CI runs).
ci: fmt-check lint test build-release

# ---------------------------------------------------------------------------
# Benchmarks — portable (paper) variants use neutral RUSTFLAGS.
# ---------------------------------------------------------------------------

# SCF benches, portable codegen (paper-grade).
bench-scf-portable:
    RUSTFLAGS="" cargo bench -p segmented-cuckoo-filter
    @just _collect-scf-csvs

# SCF benches, native codegen (local exploration; NOT for paper figures).
bench-scf-native:
    RUSTFLAGS="-C target-cpu=native" cargo bench -p segmented-cuckoo-filter

# PIR benches, portable codegen.
bench-pir-portable:
    RUSTFLAGS="" cargo bench -p ikpir-server
    @just _collect-pir-csvs

# PIR benches, native codegen.
bench-pir-native:
    RUSTFLAGS="-C target-cpu=native" cargo bench -p ikpir-server

# Copy freshly-generated bench CSVs into the committed `results/paper/` tree.
# Only the canonical "final" CSVs backing paper figures live there.
_collect-scf-csvs:
    @mkdir -p results/paper/scf
    @if [ -d crates/segmented-cuckoo-filter/results ]; then \
        cp -v crates/segmented-cuckoo-filter/results/*.csv results/paper/scf/ 2>/dev/null || true; \
    fi

_collect-pir-csvs:
    @mkdir -p results/paper/pir
    @if [ -d crates/ikpir-server/results ]; then \
        cp -v crates/ikpir-server/results/*.csv results/paper/pir/ 2>/dev/null || true; \
    fi

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# plot.py lives inside the segmented-cuckoo-filter crate so the crate can
# be published standalone; workspace `just plots` points it at the
# workspace-level paper CSV tree instead of the crate's local `results/`.

# Regenerate all figures in `results/plots/` from `results/paper/scf/*.csv`.
plots:
    SCF_RESULTS_DIR=results/paper/scf SCF_PLOTS_DIR=results/plots \
        python3 crates/segmented-cuckoo-filter/scripts/plot.py

# Install Python plotting dependencies into a local `.venv`.
plots-setup:
    python3 -m venv .venv
    .venv/bin/pip install -r crates/segmented-cuckoo-filter/scripts/requirements.txt
    .venv/bin/pip install -r scripts/requirements.txt

# ---------------------------------------------------------------------------
# Reproducibility entry points
# ---------------------------------------------------------------------------

# <1 minute: compile, run tests, run tiny benches, regenerate plots.
repro-smoke: ci plots

# Full paper-grade reproduction. Expect multi-hour runtime on a laptop.
repro-all: ci bench-scf-portable bench-pir-portable plots verify

# Numerical sanity-check of `results/paper/*.csv` against paper claims.
verify:
    python3 scripts/verify_results.py
