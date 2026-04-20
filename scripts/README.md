# `scripts/` — automation for paper reproduction

| Script                 | What it does | Run via |
|------------------------|--------------|---------|
| `plot.py`              | Regenerate every figure from `results/paper/*.csv` into `results/plots/` | `just plots` |
| `verify_results.py`    | Numerical sanity-check `results/paper/*.csv` against paper claims | `just verify` |
| `reproduce.sh`         | Thin wrapper around `just repro-all` for AEC reviewers | `./scripts/reproduce.sh` |
| `requirements.txt`     | Pinned Python deps for `plot.py` and `verify_results.py` | `just plots-setup` |

## One-time setup

```bash
just plots-setup     # creates .venv and installs the pinned dependencies
```

## Environment overrides

All three Python scripts respect these variables if you want to drive them
against a scratch directory instead of the committed CSVs:

- `IKPIR_SCF_RESULTS_DIR` (default: `results/paper/scf`)
- `IKPIR_PIR_RESULTS_DIR` (default: `results/paper/pir`)
- `IKPIR_PLOTS_DIR`       (default: `results/plots`)
