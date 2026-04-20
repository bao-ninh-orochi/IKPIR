#!/usr/bin/env python3
"""Generate plots from benchmark CSV results.

Usage:
    python scripts/plot.py                            # generate all plots
    python scripts/plot.py <function>                 # run one plot function
    python scripts/plot.py <function> [args...]       # run with optional args
    python scripts/plot.py --list                     # list available functions

Examples:
    python scripts/plot.py load_factor_all
    python scripts/plot.py load_factor_by_kicks           # all (arity, b) pairs
    python scripts/plot.py load_factor_by_kicks 2 4       # arity=2, b=4 only

Reads CSV files from `results/paper/{scf,pir}/` and generates PNG charts in
`results/plots/`.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Paper-grade CSVs live in `results/paper/{scf,pir}/`; all plots emit to
# `results/plots/`. Override via `IKPIR_RESULTS_DIR` env var for local runs
# against a scratch directory.
SCF_RESULTS_DIR = os.environ.get("IKPIR_SCF_RESULTS_DIR", "results/paper/scf")
PIR_RESULTS_DIR = os.environ.get("IKPIR_PIR_RESULTS_DIR", "results/paper/pir")
PLOTS_DIR       = os.environ.get("IKPIR_PLOTS_DIR",       "results/plots")

# Backwards-compat alias for plots written before the scf/pir split.
RESULTS_DIR = SCF_RESULTS_DIR

# Consistent scheme colours: blue = standard, orange = segmented
SCHEME_COLORS = {
    "standard": "#1f77b4",
    "segmented": "#ff7f0e",
}

# Consistent bucket-size markers: diamond=b1, square=b2, circle=b3, triangle=b4
B_MARKERS = {1: "D", 2: "s", 3: "o", 4: "^"}

# Markers for MAX_KICKS sweep values (500, 1000, ..., 5000)
KICKS_MARKERS = {
    500: "o",
    1000: "s",
    1500: "D",
    2000: "^",
    2500: "v",
    3000: "<",
    3500: ">",
    4000: "p",
    4500: "*",
    5000: "h",
}
KICKS_LINESTYLES = {
    "standard": "-",
    "segmented": "--",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load Factor
# ═══════════════════════════════════════════════════════════════════════════════

def plot_load_factor_all():
    """Load factor — all b values.

    Input:  results/load_factor.csv
    Plot:   3 subplots (one per arity). Mean load factor (y) vs n (x, log2 scale).
            Colour = scheme, marker shape = b value.
            Uses max_kicks=500 slice (first kick budget) for comparison.
    Output: results/plots/load_factor_all.png
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "load_factor.csv"))
    # New CSV has mean_lf directly; filter to a single max_kicks for comparability
    if "max_kicks" in df.columns:
        df = df[df["max_kicks"] == df["max_kicks"].min()]
    if "mean_lf" not in df.columns:
        # Old format: aggregate
        df = df.groupby(["scheme", "arity", "n", "b"]).agg(mean_lf=("load_factor", "mean")).reset_index()

    arities = [2, 3, 4]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)

    for idx, arity in enumerate(arities):
        ax = axes[0][idx]
        data = df[df["arity"] == arity]
        for (scheme, b), grp in data.groupby(["scheme", "b"]):
            grp = grp.sort_values("n")
            ax.plot(
                grp["n"], grp["mean_lf"],
                marker=B_MARKERS.get(b, "x"),
                color=SCHEME_COLORS.get(scheme, "gray"),
                label=f"{scheme} b={b}",
                linewidth=1.2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n (buckets)")
        ax.set_ylabel("Load Factor")
        ax.set_title(f"{arity}-ary")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Maximum Load Factor (b=1,2,3,4) by Arity", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "load_factor_all.png"), dpi=150)
    plt.close(fig)


def plot_load_factor_b234():
    """Load factor — selective b values for clarity.

    Input:  results/load_factor.csv
    Plot:   Same 3-subplot layout as plot_load_factor_all, but arity=2 shows only b=2,3,4
            (b=1 excluded because 1-slot buckets behave very differently).
            Uses max_kicks=500 slice.
    Output: results/plots/load_factor_b234.png
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "load_factor.csv"))
    if "max_kicks" in df.columns:
        df = df[df["max_kicks"] == df["max_kicks"].min()]
    if "mean_lf" not in df.columns:
        df = df.groupby(["scheme", "arity", "n", "b"]).agg(mean_lf=("load_factor", "mean")).reset_index()

    arities = [2, 3, 4]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)

    for idx, arity in enumerate(arities):
        ax = axes[0][idx]
        if arity == 2:
            data = df[(df["arity"] == arity) & df["b"].isin([2, 3, 4])]
        else:
            data = df[df["arity"] == arity]
        for (scheme, b), grp in data.groupby(["scheme", "b"]):
            grp = grp.sort_values("n")
            ax.plot(
                grp["n"], grp["mean_lf"],
                marker=B_MARKERS.get(b, "x"),
                color=SCHEME_COLORS.get(scheme, "gray"),
                label=f"{scheme} b={b}",
                linewidth=1.2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n (buckets)")
        ax.set_ylabel("Load Factor")
        ax.set_title(f"{arity}-ary")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Maximum Load Factor (arity=2: b=2,3,4 | arity=3,4: b=1,2,3,4)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "load_factor_b234.png"), dpi=150)
    plt.close(fig)


def plot_load_factor_by_kicks(arity=None, b=None):
    """Load factor vs MAX_KICKS sweep.

    For each (arity, b) pair: one figure. X-axis = n (log2 scale), Y-axis = mean load
    factor. Each MAX_KICKS value is one line. Blue = standard, orange = segmented.
    Solid lines = standard, dashed lines = segmented. Different markers per MAX_KICKS.

    Args:
        arity: If given (int), plot only this arity. Otherwise plot all.
        b:     If given (int), plot only this b value. Otherwise plot all.

    Input:  results/load_factor.csv  (must have max_kicks column)
    Output: results/plots/load_factor_kicks_arity{a}_b{b}.png  (one per (arity, b))
    """
    csv_path = os.path.join(RESULTS_DIR, "load_factor.csv")
    df = pd.read_csv(csv_path)

    if "max_kicks" not in df.columns:
        print("  load_factor.csv has no max_kicks column — run the updated benchmark first.")
        return

    arities = [int(arity)] if arity is not None else sorted(df["arity"].unique())
    b_values = [int(b)] if b is not None else sorted(df["b"].unique())

    for a in arities:
        for bv in b_values:
            subset = df[(df["arity"] == a) & (df["b"] == bv)]
            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(11, 6))

            for (scheme, mk), grp in subset.groupby(["scheme", "max_kicks"]):
                grp = grp.sort_values("n")
                ax.plot(
                    grp["n"], grp["mean_lf"],
                    marker=KICKS_MARKERS.get(int(mk), "x"),
                    color=SCHEME_COLORS.get(scheme, "gray"),
                    linestyle=KICKS_LINESTYLES.get(scheme, "-"),
                    label=f"{scheme} kicks={int(mk)}",
                    linewidth=1.0,
                    markersize=5,
                )

            ax.set_xscale("log", base=2)
            ax.set_xlabel("n (buckets)")
            ax.set_ylabel("Mean Load Factor")
            ax.set_title(f"Load Factor vs MAX_KICKS \u2014 {a}-ary, b={bv}")
            ax.legend(fontsize=7, ncol=2, loc="lower right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = os.path.join(PLOTS_DIR, f"load_factor_kicks_arity{a}_b{bv}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"    Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Insert Throughput
# ═══════════════════════════════════════════════════════════════════════════════

def plot_insert_throughput():
    """Insert throughput — insert until full.

    Input:  results/insert_throughput.csv
    Plot:   3 figures (one per arity), each with 4 subplots (one per b value).
            Within each subplot, bars are grouped by n; each n group has two adjacent bars —
            segmented (orange) and standard (blue). Y-axis = mean Mops/sec across trials.
    Output: results/plots/insert_throughput_{arity}ary.png  (one per arity)
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "insert_throughput.csv"))
    # Support both old (per-trial) and new (aggregated) CSV formats
    if "mean_mops" not in df.columns:
        df = df.groupby(["scheme", "arity", "n", "b"]).agg(mean_mops=("throughput_mops", "mean")).reset_index()

    b_values = sorted(df["b"].unique())

    for arity in sorted(df["arity"].unique()):
        adata = df[df["arity"] == arity]
        n_values = sorted(adata["n"].unique())

        fig, axes = plt.subplots(1, len(b_values), figsize=(4 * len(b_values), 5), squeeze=False)

        for col_idx, b in enumerate(b_values):
            ax = axes[0][col_idx]
            bdata = adata[adata["b"] == b]

            x = list(range(len(n_values)))
            width = 0.35

            for i, n_val in enumerate(n_values):
                for j, scheme in enumerate(["segmented", "standard"]):
                    match = bdata[(bdata["n"] == n_val) & (bdata["scheme"] == scheme)]
                    if not match.empty:
                        pos = i + (j - 0.5) * width
                        ax.bar(
                            pos, match.iloc[0]["mean_mops"], width,
                            color=SCHEME_COLORS[scheme],
                            label=scheme if i == 0 else "",
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([f"n={int(n)}" for n in n_values], fontsize=8)
            ax.set_ylabel("M ops/sec")
            ax.set_title(f"b={b}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Insert Throughput — {arity}-ary (insert until full)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"insert_throughput_{arity}ary.png"), dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Delete Throughput
# ═══════════════════════════════════════════════════════════════════════════════

def plot_delete_throughput():
    """Delete throughput — delete all items from a full filter.

    Input:  results/delete_throughput.csv
    Plot:   Identical layout to plot_insert_throughput (3 figures × 4 subplots per b,
            bars by n with segmented/standard pairs). Y-axis = mean Mops/sec for the
            deletion loop only (fill phase excluded from timing).
    Output: results/plots/delete_throughput_{arity}ary.png  (one per arity)
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "delete_throughput.csv"))
    if "mean_mops" not in df.columns:
        df = df.groupby(["scheme", "arity", "n", "b"]).agg(mean_mops=("throughput_mops", "mean")).reset_index()

    b_values = sorted(df["b"].unique())

    for arity in sorted(df["arity"].unique()):
        adata = df[df["arity"] == arity]
        n_values = sorted(adata["n"].unique())

        fig, axes = plt.subplots(1, len(b_values), figsize=(4 * len(b_values), 5), squeeze=False)

        for col_idx, b in enumerate(b_values):
            ax = axes[0][col_idx]
            bdata = adata[adata["b"] == b]

            x = list(range(len(n_values)))
            width = 0.35

            for i, n_val in enumerate(n_values):
                for j, scheme in enumerate(["segmented", "standard"]):
                    match = bdata[(bdata["n"] == n_val) & (bdata["scheme"] == scheme)]
                    if not match.empty:
                        pos = i + (j - 0.5) * width
                        ax.bar(
                            pos, match.iloc[0]["mean_mops"], width,
                            color=SCHEME_COLORS[scheme],
                            label=scheme if i == 0 else "",
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([f"n={int(n)}" for n in n_values], fontsize=8)
            ax.set_ylabel("M ops/sec")
            ax.set_title(f"b={b}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Delete Throughput — {arity}-ary (delete all from full filter)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"delete_throughput_{arity}ary.png"), dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Lookup Throughput
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lookup_throughput():
    """Lookup throughput — query speed at 5 hit rates on a full filter.

    Input:  results/lookup_throughput.csv
    Plot:   One figure per (n, b) pair.
            X-axis: 6 scheme groups (seg 2-ary, std 2-ary, seg 3-ary, std 3-ary,
            seg 4-ary, std 4-ary). Within each group: 5 bars for hit rates 0/25/50/75/100%.
            Segmented 3-ary may have a different n (nearest valid 3·2^m); its label shows
            the actual n used.
    Output: results/plots/lookup_throughput_n{n}_b{b}.png  (one per (n, b))
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "lookup_throughput.csv"))
    # Support both old (per-trial) and new (aggregated) CSV formats
    if "mean_mops" not in df.columns:
        df = df.groupby(["scheme", "arity", "n", "b", "hit_rate_pct"]).agg(
            mean_mops=("throughput_mops", "mean")
        ).reset_index()

    # Determine the reference n (majority n for arity≠3) used for each (n_ref, b) figure.
    ref_n_values = sorted(df[df["arity"] == 2]["n"].unique())
    b_values = sorted(df["b"].unique())

    for ref_n in ref_n_values:
        for b in b_values:
            slice2 = df[(df["arity"] == 2) & (df["n"] == ref_n) & (df["b"] == b)]
            slice4 = df[(df["arity"] == 4) & (df["n"] == ref_n) & (df["b"] == b)]
            slice3 = df[(df["arity"] == 3) & (df["b"] == b)]

            if slice2.empty and slice3.empty and slice4.empty:
                continue

            std3_ns = sorted(slice3[slice3["scheme"] == "standard"]["n"].unique())
            seg3_ns = sorted(slice3[slice3["scheme"] == "segmented"]["n"].unique())

            def closest(ns, ref):
                return min(ns, key=lambda v: abs(v - ref)) if len(ns) else None

            std3_n = closest(std3_ns, ref_n)
            seg3_n = closest(seg3_ns, ref_n)

            SCHEME_ORDER = [
                ("segmented 2-ary",               "segmented", 2, ref_n),
                ("standard 2-ary",                "standard",  2, ref_n),
                (f"segmented 3-ary (n={seg3_n})", "segmented", 3, seg3_n),
                (f"standard 3-ary (n={std3_n})",  "standard",  3, std3_n),
                ("segmented 4-ary",               "segmented", 4, ref_n),
                ("standard 4-ary",                "standard",  4, ref_n),
            ]

            hit_rates = sorted(df["hit_rate_pct"].unique())
            fig, ax = plt.subplots(figsize=(14, 6))
            n_schemes = len(SCHEME_ORDER)
            width = 0.8 / max(len(hit_rates), 1)

            for i, hr in enumerate(hit_rates):
                means = []
                for label, scheme, arity, n_val in SCHEME_ORDER:
                    if n_val is None:
                        means.append(0.0)
                        continue
                    row = df[
                        (df["scheme"] == scheme) &
                        (df["arity"] == arity) &
                        (df["n"] == n_val) &
                        (df["b"] == b) &
                        (df["hit_rate_pct"] == hr)
                    ]
                    means.append(row.iloc[0]["mean_mops"] if not row.empty else 0.0)
                positions = [xi + i * width for xi in range(n_schemes)]
                ax.bar(positions, means, width, label=f"{int(hr)}%")

            ax.set_xticks([xi + width * (len(hit_rates) - 1) / 2 for xi in range(n_schemes)])
            ax.set_xticklabels(
                [s[0] for s in SCHEME_ORDER], rotation=25, ha="right", fontsize=9
            )
            ax.set_ylabel("M ops/sec")
            ax.set_title(f"Lookup Throughput \u2014 n={int(ref_n)}, b={int(b)}")
            ax.legend(title="Hit Rate")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(
                os.path.join(PLOTS_DIR, f"lookup_throughput_n{int(ref_n)}_b{int(b)}.png"),
                dpi=150,
            )
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. False Positive Rate
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fpr_load_factor():
    """FPR — load factor vs fingerprint bits.

    Input:  results/fpr/arity{a}_n{n}_b{b}.csv  (one file per config)
    Plot:   Per CSV file: one figure, load factor (y) vs fingerprint bits (x),
            one line per scheme. Shows whether wider fingerprints affect how full
            the filter gets.
    Output: results/plots/fpr_lf_arity{a}_n{n}_b{b}.png  (one per config)
    """
    fpr_dir = os.path.join(RESULTS_DIR, "fpr")
    if not os.path.isdir(fpr_dir):
        return

    for csv_file in sorted(glob.glob(os.path.join(fpr_dir, "arity*_n*_b*.csv"))):
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        m = re.match(r"arity(\d+)_n(\d+)_b(\d+)", basename)
        if not m:
            continue
        arity, n, b = int(m.group(1)), int(m.group(2)), int(m.group(3))

        df = pd.read_csv(csv_file)

        fig, ax = plt.subplots(figsize=(8, 5))
        for scheme, grp in df.groupby("scheme"):
            grp = grp.sort_values("fp_bits")
            ax.plot(
                grp["fp_bits"], grp["load_factor"],
                marker="o",
                color=SCHEME_COLORS.get(scheme, "gray"),
                label=scheme,
                linewidth=1.2,
            )
        ax.set_xlabel("Fingerprint Bits")
        ax.set_ylabel("Load Factor")
        ax.set_title(f"Load Factor vs Fingerprint Bits \u2014 {arity}-ary, n={n}, b={b}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"fpr_lf_{basename}.png"), dpi=150)
        plt.close(fig)


def plot_fpr_comparison():
    """FPR vs fingerprint bits with theoretical bound.

    Input:  results/fpr/arity{a}_n{n}_b{b}.csv — auto-selects the (n, b) pair with the
            largest capacity >= 1M (n*b >= 1_000_000) for statistical significance.
    Plot:   3 subplots (one per arity). FPR% (log y-scale) vs fingerprint bits for standard
            and segmented, plus a dashed black theoretical line d*b/2^fp_bits (where d=arity).
    Output: results/plots/fpr_comparison.png
    """
    fpr_dir = os.path.join(RESULTS_DIR, "fpr")
    if not os.path.isdir(fpr_dir):
        return

    best_n, best_b, best_cap = None, None, 0
    for csv_file in glob.glob(os.path.join(fpr_dir, "arity*_n*_b*.csv")):
        m = re.match(r"arity(\d+)_n(\d+)_b(\d+)", os.path.basename(csv_file))
        if m:
            n_val, b_val = int(m.group(2)), int(m.group(3))
            cap = n_val * b_val
            if cap >= 1_000_000 and cap > best_cap:
                best_n, best_b, best_cap = n_val, b_val, cap

    if best_n is None:
        print("  No FPR config with capacity >= 1M found")
        return

    arities = [2, 3, 4]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)

    for idx, arity in enumerate(arities):
        ax = axes[0][idx]
        csv_file = os.path.join(fpr_dir, f"arity{arity}_n{best_n}_b{best_b}.csv")
        if not os.path.exists(csv_file):
            ax.set_visible(False)
            continue

        df = pd.read_csv(csv_file)

        for scheme, grp in df.groupby("scheme"):
            grp = grp.sort_values("fp_bits")
            ax.plot(
                grp["fp_bits"], grp["fpr_pct"],
                marker="o",
                color=SCHEME_COLORS.get(scheme, "gray"),
                label=scheme,
                linewidth=1.2,
            )

        # Theoretical line: d*b / 2^f (corrected from 2b/2^f)
        fp_range = sorted(df["fp_bits"].unique())
        theoretical = [arity * best_b / (2 ** f) * 100.0 for f in fp_range]
        ax.plot(fp_range, theoretical, "k--", linewidth=2, alpha=0.5,
                label=f"theoretical ({arity}b/2^f)")

        ax.set_yscale("log")
        ax.set_xlabel("Fingerprint Bits")
        ax.set_ylabel("False Positive Rate (%)")
        ax.set_title(f"{arity}-ary")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"FPR vs Fingerprint Bits (n={best_n}, b={best_b})", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fpr_comparison.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Eviction
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eviction():
    """Eviction chain distribution — stacked bar of kick-count ranges.

    Input:  results/eviction.csv
    Plot:   3 figures (one per arity), each with subplots per b value. Within each subplot,
            bars are grouped by (n, scheme) and stacked into 5 kick-count ranges:
            0, 1-10, 11-50, 51-100, 101-500 — normalised to 100% of insertions.
    Output: results/plots/eviction_{arity}ary.png  (one per arity)
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "eviction.csv"))

    hist_cols = ["hist_0", "hist_1_10", "hist_11_50", "hist_51_100", "hist_101_500"]
    hist_labels = ["0", "1-10", "11-50", "51-100", "101-500"]
    hist_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    for col in hist_cols:
        df[col + "_pct"] = df[col] / df["total_inserts"] * 100.0
    pct_cols = [c + "_pct" for c in hist_cols]

    arities = [2, 3, 4]
    b_values = sorted(df["b"].unique())

    for arity in arities:
        adf = df[df["arity"] == arity].copy()
        n_cols = len(b_values)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), squeeze=False)

        for col_idx, b in enumerate(b_values):
            ax = axes[0][col_idx]
            bdf = adf[adf["b"] == b].sort_values(["n", "scheme"]).reset_index(drop=True)
            labels = [
                f"{row['scheme']}\nn={row['n']:,}" for _, row in bdf.iterrows()
            ]
            x = range(len(labels))
            bottom = [0.0] * len(labels)

            for pct_col, hlabel, color in zip(pct_cols, hist_labels, hist_colors):
                values = bdf[pct_col].tolist()
                ax.bar(x, values, bottom=bottom, label=f"kicks=[{hlabel}]", color=color)
                bottom = [bv + v for bv, v in zip(bottom, values)]

            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("% of insertions")
            ax.set_ylim(0, 105)
            ax.set_title(f"b={b}")
            ax.grid(True, alpha=0.3, axis="y")
            if col_idx == 0:
                ax.legend(fontsize=7, loc="upper left")

        fig.suptitle(
            f"Eviction Distribution \u2014 {arity}-ary (insert until full, fp_bits=12)",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"eviction_{arity}ary.png"), dpi=150)
        plt.close(fig)


def plot_eviction_mean_kicks():
    """Mean evictions per insertion vs table size.

    Input:  results/eviction.csv
    Plot:   3 figures (one per arity). Mean kicks per insertion (y) vs n (x, log2 scale),
            one line per (scheme, b) combination. Shows how eviction cost scales with n.
    Output: results/plots/eviction_mean_kicks_{arity}ary.png  (one per arity)
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "eviction.csv"))

    b_markers = {2: "s", 3: "o", 4: "^"}
    scheme_styles = {"standard": "-", "segmented": "--"}

    for arity in [2, 3, 4]:
        adf = df[df["arity"] == arity].copy()
        fig, ax = plt.subplots(figsize=(8, 5))

        for (scheme, b), grp in adf.groupby(["scheme", "b"]):
            grp = grp.sort_values("n")
            ax.plot(
                grp["n"], grp["mean_kicks"],
                marker=b_markers.get(b, "x"),
                color=SCHEME_COLORS.get(scheme, "gray"),
                linestyle=scheme_styles.get(scheme, "-"),
                label=f"{scheme} b={b}",
                linewidth=1.4,
            )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("n (buckets)")
        ax.set_ylabel("Mean kicks per insertion")
        ax.set_title(f"Mean Evictions per Insert \u2014 {arity}-ary (fp_bits=12)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"eviction_mean_kicks_{arity}ary.png"), dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Degree Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_degree_index():
    """Bucket degree vs bucket index — spatial pattern of load distribution.

    Input:  results/degree_per_bucket.csv
    Plot:   For each (arity, n, b): one figure with 2 side-by-side scatter plots
            (standard left, segmented right). X = bucket index, Y = degree.
            Downsampled to max 4000 points for readability. Reveals degree bands at
            segment boundaries in segmented schemes.
    Output: results/plots/degree_distribution/degree_{arity}ary_n{n}_b{b}.png
    """
    csv_path = os.path.join(RESULTS_DIR, "degree_per_bucket.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    degree_dir = os.path.join(PLOTS_DIR, "degree_distribution")
    os.makedirs(degree_dir, exist_ok=True)

    arities = sorted(df["arity"].unique())
    b_values = sorted(df["b"].unique())

    for arity in arities:
        adf = df[df["arity"] == arity]
        std_ns = sorted(adf[adf["scheme"] == "standard"]["n"].unique())
        seg_ns = sorted(adf[adf["scheme"] == "segmented"]["n"].unique())

        for n_std, n_seg in zip(std_ns, seg_ns):
            for b in b_values:
                std_rows = adf[
                    (adf["scheme"] == "standard") & (adf["n"] == n_std) & (adf["b"] == b)
                ].sort_values("bucket_index")
                seg_rows = adf[
                    (adf["scheme"] == "segmented") & (adf["n"] == n_seg) & (adf["b"] == b)
                ].sort_values("bucket_index")

                if std_rows.empty and seg_rows.empty:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(14, 4), squeeze=False)

                for ax, (rows, scheme, n_val, color) in zip(axes[0], [
                    (std_rows, "standard", n_std, SCHEME_COLORS["standard"]),
                    (seg_rows, "segmented", n_seg, SCHEME_COLORS["segmented"]),
                ]):
                    if rows.empty:
                        ax.set_visible(False)
                        continue
                    stride = max(1, len(rows) // 4000)
                    rows = rows.iloc[::stride]
                    ax.scatter(
                        rows["bucket_index"], rows["degree"],
                        s=2, color=color, alpha=0.4,
                    )
                    ax.plot(
                        rows["bucket_index"], rows["degree"],
                        color=color, linewidth=0.5, alpha=0.6,
                    )
                    ax.set_xlabel("Bucket index")
                    ax.set_ylabel("Bucket degree")
                    ax.set_title(f"{scheme} (n={n_val:,})")
                    ax.grid(True, alpha=0.3)

                fig.suptitle(
                    f"Bucket Degree vs Index \u2014 {arity}-ary, b={b}, fp_bits=12",
                    fontsize=12,
                )
                fig.tight_layout()
                fname = f"degree_{arity}ary_n{n_std}_b{b}.png"
                fig.savefig(os.path.join(degree_dir, fname), dpi=150)
                plt.close(fig)


def plot_degree_histogram():
    """Degree distribution — fraction of buckets at each degree level.

    Input:  results/degree_distribution.csv
    Plot:   For each (arity, b): one line plot. X = bucket degree, Y = fraction of all
            buckets with that degree. Compares distribution shape between standard
            (Poisson-like) and segmented schemes.
    Output: results/plots/degree_hist_{arity}ary_b{b}.png
    """
    csv_path = os.path.join(RESULTS_DIR, "degree_distribution.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    for arity in sorted(df["arity"].unique()):
        for b in sorted(df["b"].unique()):
            subset = df[(df["arity"] == arity) & (df["b"] == b)]
            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))

            for (scheme, n), grp in subset.groupby(["scheme", "n"]):
                total_buckets = grp["count"].sum()
                grp = grp.sort_values("degree")
                ax.plot(
                    grp["degree"],
                    grp["count"] / total_buckets,
                    marker=B_MARKERS.get(b, "o"),
                    color=SCHEME_COLORS.get(scheme, "gray"),
                    label=f"{scheme} (n={n})",
                    linewidth=1.2,
                )

            ax.set_xlabel("Bucket Degree")
            ax.set_ylabel("Fraction of Buckets")
            ax.set_title(f"Degree Distribution \u2014 {arity}-ary, b={b}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(PLOTS_DIR, f"degree_hist_{arity}ary_b{b}.png"),
                dpi=150,
            )
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI registry and main
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_FUNCTIONS = {
    "load_factor_all":        plot_load_factor_all,
    "load_factor_b234":       plot_load_factor_b234,
    "load_factor_by_kicks":   plot_load_factor_by_kicks,
    "insert_throughput":      plot_insert_throughput,
    "delete_throughput":      plot_delete_throughput,
    "lookup_throughput":      plot_lookup_throughput,
    "fpr_load_factor":        plot_fpr_load_factor,
    "fpr_comparison":         plot_fpr_comparison,
    "eviction":               plot_eviction,
    "eviction_mean_kicks":    plot_eviction_mean_kicks,
    "degree_index":           plot_degree_index,
    "degree_histogram":       plot_degree_histogram,
}


def _run_all():
    """Run all plot functions (same as no-argument invocation)."""
    if os.path.exists(os.path.join(RESULTS_DIR, "load_factor.csv")):
        plot_load_factor_all()
        print("  Generated load_factor_all.png")
        plot_load_factor_b234()
        print("  Generated load_factor_b234.png")
        # Only generate kicks sweep if the CSV has a max_kicks column
        df_lf = pd.read_csv(os.path.join(RESULTS_DIR, "load_factor.csv"))
        if "max_kicks" in df_lf.columns:
            plot_load_factor_by_kicks()
            print("  Generated load_factor_kicks_*.png")

    if os.path.exists(os.path.join(RESULTS_DIR, "insert_throughput.csv")):
        plot_insert_throughput()
        print("  Generated insert_throughput_2ary.png, insert_throughput_3ary.png, insert_throughput_4ary.png")

    if os.path.exists(os.path.join(RESULTS_DIR, "delete_throughput.csv")):
        plot_delete_throughput()
        print("  Generated delete_throughput_2ary.png, delete_throughput_3ary.png, delete_throughput_4ary.png")

    if os.path.exists(os.path.join(RESULTS_DIR, "lookup_throughput.csv")):
        plot_lookup_throughput()
        print("  Generated lookup_throughput_*.png")

    if os.path.isdir(os.path.join(RESULTS_DIR, "fpr")):
        plot_fpr_load_factor()
        print("  Generated fpr_lf_*.png")
        plot_fpr_comparison()
        print("  Generated fpr_comparison.png")

    if os.path.exists(os.path.join(RESULTS_DIR, "eviction.csv")):
        plot_eviction()
        print("  Generated eviction_*.png")
        plot_eviction_mean_kicks()
        print("  Generated eviction_mean_kicks_*.png")

    if os.path.exists(os.path.join(RESULTS_DIR, "degree_per_bucket.csv")):
        plot_degree_index()
        print("  Generated degree_distribution/ (degree-index scatter)")

    if os.path.exists(os.path.join(RESULTS_DIR, "degree_distribution.csv")):
        plot_degree_histogram()
        print("  Generated degree_hist_*.png")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from benchmark CSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available plot functions:
  load_factor_all          Load factor — all b values
  load_factor_b234         Load factor — selective b for clarity
  load_factor_by_kicks     Load factor vs MAX_KICKS sweep  [arity] [b]
  insert_throughput        Insert throughput bars per arity
  delete_throughput        Delete throughput bars per arity
  lookup_throughput        Lookup throughput at 5 hit rates
  fpr_load_factor          FPR: load factor vs fp_bits
  fpr_comparison           FPR vs fp_bits with theoretical bound
  eviction                 Eviction distribution stacked bars
  eviction_mean_kicks      Mean evictions per insert vs n
  degree_index             Bucket degree vs index scatter
  degree_histogram         Degree histogram per (arity, b)

Examples:
  python scripts/plot.py
  python scripts/plot.py load_factor_by_kicks
  python scripts/plot.py load_factor_by_kicks 2 4
""",
    )
    parser.add_argument(
        "function",
        nargs="?",
        default=None,
        choices=list(PLOT_FUNCTIONS.keys()),
        metavar="function",
        help="Plot function to run. Omit to run all.",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Optional positional arguments passed to the plot function.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available plot functions and exit.",
    )

    args = parser.parse_args()

    if args.list:
        print("Available plot functions:")
        for name, fn in PLOT_FUNCTIONS.items():
            first_line = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:<26} {first_line}")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    if args.function is None:
        _run_all()
        return

    fn = PLOT_FUNCTIONS[args.function]
    # Convert string args to int where possible
    fn_args = []
    for a in args.args:
        try:
            fn_args.append(int(a))
        except ValueError:
            fn_args.append(a)

    try:
        fn(*fn_args)
    except TypeError as exc:
        print(f"Error calling {args.function}: {exc}", file=sys.stderr)
        doc = fn.__doc__ or ""
        print(f"\nFunction docstring:\n{doc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Generated {args.function} plot(s) → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
