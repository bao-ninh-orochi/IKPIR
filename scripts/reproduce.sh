#!/usr/bin/env bash
# Single entry point for artifact-evaluation reviewers.
#
# Usage: `./scripts/reproduce.sh` from anywhere; it cd's to the repo root.
# Invokes `just repro-all`. See REPRODUCING.md for expected wall-clock and
# hardware assumptions.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/.."

exec just repro-all
