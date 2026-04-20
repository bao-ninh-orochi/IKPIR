#!/usr/bin/env python3
"""Numerical sanity-check for `results/paper/*.csv` against paper claims.

Each assertion below corresponds to a specific sentence / figure / table in
the paper. If you edit the paper, update the assertions here in the same
PR. Tolerances are quoted per-assertion.

Exit 0 on success, 1 on any failure.

Usage:
    python scripts/verify_results.py
    IKPIR_SCF_RESULTS_DIR=./my-run/scf python scripts/verify_results.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


SCF_DIR = Path(os.environ.get("IKPIR_SCF_RESULTS_DIR", "results/paper/scf"))
PIR_DIR = Path(os.environ.get("IKPIR_PIR_RESULTS_DIR", "results/paper/pir"))


class Check:
    """Lightweight assertion helper that accumulates failures instead of
    aborting on the first one — reviewers see the whole picture at once."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def assert_true(self, cond: bool, msg: str) -> None:
        if cond:
            print(f"[OK ] {msg}")
        else:
            print(f"[ERR] {msg}")
            self.failures.append(msg)

    def assert_approx(
        self, actual: float, expected: float, tol: float, msg: str
    ) -> None:
        ok = abs(actual - expected) <= tol
        note = f"actual={actual:.4f} expected={expected:.4f} tol={tol:.4f}"
        if ok:
            print(f"[OK ] {msg} — {note}")
        else:
            print(f"[ERR] {msg} — {note}")
            self.failures.append(f"{msg} ({note})")


def verify_scf_load_factor(check: Check) -> None:
    path = SCF_DIR / "load_factor.csv"
    if not path.exists():
        check.assert_true(
            False,
            f"{path} missing — run `just bench-scf-portable` first",
        )
        return

    df = pd.read_csv(path)
    # Paper claim (placeholder — edit to match table/figure at submission time):
    # "Segmented 4-ary with b=4 reaches load factor ≥ 0.94."
    seg4 = df[
        (df["scheme"] == "segmented")
        & (df["arity"] == 4)
        & (df["b"] == 4)
    ]
    if seg4.empty:
        check.assert_true(False, "segmented 4-ary b=4 rows missing from load_factor.csv")
        return

    peak = seg4["mean_lf"].max()
    check.assert_true(
        peak >= 0.94,
        f"Segmented 4-ary b=4 peak load factor ≥ 0.94 (observed {peak:.4f})",
    )


def verify_pir_query_latency(check: Check) -> None:
    path = PIR_DIR / "query_latency.csv"
    if not path.exists():
        # Phase B not yet implemented — silently skip.
        print(f"[skip] {path} not present (Phase B not yet complete)")
        return
    # TODO: add concrete latency assertion once Phase B commits reference numbers.


def main() -> int:
    check = Check()
    verify_scf_load_factor(check)
    verify_pir_query_latency(check)

    if check.failures:
        print(f"\n{len(check.failures)} assertion(s) failed.")
        return 1
    print("\nAll assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
