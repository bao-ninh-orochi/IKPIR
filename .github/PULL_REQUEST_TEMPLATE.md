## What

<!-- 1-2 sentences: what does this PR change? -->

## Why

<!-- The motivation: bug report, benchmark regression, paper-section alignment, … -->

## Pre-merge checklist

- [ ] `just ci` passes locally (`fmt-check`, `lint`, `test`, `build-release`).
- [ ] If bench schema changed, `scripts/plot.py` and `results/SCHEMA.md` updated.
- [ ] `CHANGELOG.md` updated (workspace and/or crate-level, as applicable).
- [ ] Docs updated if public API changed.

## Paper impact

<!-- Fill in if this PR changes a number quoted in the paper. -->

- Affected figure(s) / table(s):
- Regenerated `results/paper/<subdir>/*.csv`:  Yes / No
- Regenerated `results/plots/*.png`:  Yes / No
