#!/usr/bin/env python3
"""Generate plots from segmented-cuckoo-filter benchmark CSV results.

Usage:
    python scripts/plot.py                            # generate all plots
    python scripts/plot.py <function>                 # run one plot function
    python scripts/plot.py <function> [args...]       # run with optional args
    python scripts/plot.py --list                     # list available functions

Examples:
    python scripts/plot.py load_factor_all
    python scripts/plot.py load_factor_by_kicks           # all (arity, bucket_size) pairs
    python scripts/plot.py load_factor_by_kicks 2 4       # arity=2, bucket_size=4 only

Reads CSV files from `results/` (the default bench output directory) and
writes PNG charts to `results/plots/`. Override via `SCF_RESULTS_DIR` /
`SCF_PLOTS_DIR` env vars to plot against a scratch directory.

Expected CSV schemas (written by the benches in `benches/`):
    load_factor.csv         scheme, arity, num_buckets, bucket_size, fingerprint_bits,
                            max_kicks, mean_lf, min_lf, max_lf, stddev_lf
    insert_throughput.csv   scheme, arity, num_buckets, bucket_size, mean_inserted,
                            mean_lf, mean_duration_ns, mean_mops, min_mops, max_mops,
                            stddev_mops
    delete_throughput.csv   scheme, arity, num_buckets, bucket_size, deleted, mean_lf,
                            mean_duration_ns, mean_mops, min_mops, max_mops, stddev_mops
    lookup_throughput.csv   scheme, arity, num_buckets, bucket_size, hit_rate_pct,
                            load_factor, num_queries, mean_mops, min_mops, max_mops,
                            stddev_mops
    fpr/arity{a}_num_buckets{n}_bucket_size{b}.csv
                            fingerprint_bits, scheme, num_buckets, load_factor,
                            num_inserted, num_queries, false_positives, fpr_pct,
                            theoretical_pct
    degree_distribution.csv scheme, arity, num_buckets, bucket_size, degree, count
    degree_per_bucket.csv   scheme, arity, num_buckets, bucket_size, bucket_index, degree
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Default read/write locations are the crate-local `results/` tree. Override
# via env vars when running against a scratch directory or a workspace-level
# collection of CSVs.
RESULTS_DIR = os.environ.get("SCF_RESULTS_DIR", "results")
PLOTS_DIR   = os.environ.get("SCF_PLOTS_DIR",   "results/plots")

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
    """Load factor — all bucket_size values.

    Input:  results/load_factor.csv
    Plot:   3 subplots (one per arity). Mean load factor (y) vs num_buckets
            (x, log2 scale). Colour = scheme, marker shape = bucket_size.
            Uses max_kicks=500 slice (first kick budget) for comparison.
    Output: results/plots/load_factor_all.png
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "load_factor.csv"))
    if "max_kicks" in df.columns:
        df = df[df["max_kicks"] == df["max_kicks"].min()]
    if "mean_lf" not in df.columns:
        df = df.groupby(["scheme", "arity", "num_buckets", "bucket_size"]).agg(
            mean_lf=("load_factor", "mean")
        ).reset_index()

    arities = [2, 3, 4]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)

    for idx, arity in enumerate(arities):
        ax = axes[0][idx]
        data = df[df["arity"] == arity]
        for (scheme, bucket_size), grp in data.groupby(["scheme", "bucket_size"]):
            grp = grp.sort_values("num_buckets")
            ax.plot(
                grp["num_buckets"], grp["mean_lf"],
                marker=B_MARKERS.get(bucket_size, "x"),
                color=SCHEME_COLORS.get(scheme, "gray"),
                label=f"{scheme} b={bucket_size}",
                linewidth=1.2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("num_buckets")
        ax.set_ylabel("Load Factor")
        ax.set_title(f"{arity}-ary")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Maximum Load Factor (b=1,2,3,4) by Arity", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "load_factor_all.png"), dpi=150)
    plt.close(fig)


def plot_load_factor_b234():
    """Load factor — selective bucket_size values for clarity.

    Input:  results/load_factor.csv
    Plot:   Same 3-subplot layout as plot_load_factor_all, but arity=2 shows
            only bucket_size=2,3,4 (b=1 excluded because 1-slot buckets behave
            very differently). Uses max_kicks=500 slice.
    Output: results/plots/load_factor_b234.png
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "load_factor.csv"))
    if "max_kicks" in df.columns:
        df = df[df["max_kicks"] == df["max_kicks"].min()]
    if "mean_lf" not in df.columns:
        df = df.groupby(["scheme", "arity", "num_buckets", "bucket_size"]).agg(
            mean_lf=("load_factor", "mean")
        ).reset_index()

    arities = [2, 3, 4]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)

    for idx, arity in enumerate(arities):
        ax = axes[0][idx]
        if arity == 2:
            data = df[(df["arity"] == arity) & df["bucket_size"].isin([2, 3, 4])]
        else:
            data = df[df["arity"] == arity]
        for (scheme, bucket_size), grp in data.groupby(["scheme", "bucket_size"]):
            grp = grp.sort_values("num_buckets")
            ax.plot(
                grp["num_buckets"], grp["mean_lf"],
                marker=B_MARKERS.get(bucket_size, "x"),
                color=SCHEME_COLORS.get(scheme, "gray"),
                label=f"{scheme} b={bucket_size}",
                linewidth=1.2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("num_buckets")
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


def plot_load_factor_by_kicks(arity=None, bucket_size=None):
    """Load factor vs MAX_KICKS sweep.

    For each (arity, bucket_size) pair: one figure. X-axis = num_buckets (log2
    scale), Y-axis = mean load factor. Each MAX_KICKS value is one line.
    Blue = standard, orange = segmented. Solid = standard, dashed = segmented.

    Args:
        arity:       If given (int), plot only this arity.
        bucket_size: If given (int), plot only this bucket_size.

    Input:  results/load_factor.csv  (must have a max_kicks column)
    Output: results/plots/load_factor_kicks_arity{a}_b{b}.png  (one per pair)
    """
    csv_path = os.path.join(RESULTS_DIR, "load_factor.csv")
    df = pd.read_csv(csv_path)

    if "max_kicks" not in df.columns:
        print("  load_factor.csv has no max_kicks column — run the updated benchmark first.")
        return

    arities = [int(arity)] if arity is not None else sorted(df["arity"].unique())
    bucket_sizes = (
        [int(bucket_size)] if bucket_size is not None else sorted(df["bucket_size"].unique())
    )

    for a in arities:
        for bs in bucket_sizes:
            subset = df[(df["arity"] == a) & (df["bucket_size"] == bs)]
            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(11, 6))

            for (scheme, mk), grp in subset.groupby(["scheme", "max_kicks"]):
                grp = grp.sort_values("num_buckets")
                ax.plot(
                    grp["num_buckets"], grp["mean_lf"],
                    marker=KICKS_MARKERS.get(int(mk), "x"),
                    color=SCHEME_COLORS.get(scheme, "gray"),
                    linestyle=KICKS_LINESTYLES.get(scheme, "-"),
                    label=f"{scheme} kicks={int(mk)}",
                    linewidth=1.0,
                    markersize=5,
                )

            ax.set_xscale("log", base=2)
            ax.set_xlabel("num_buckets")
            ax.set_ylabel("Mean Load Factor")
            ax.set_title(f"Load Factor vs MAX_KICKS — {a}-ary, b={bs}")
            ax.legend(fontsize=7, ncol=2, loc="lower right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = os.path.join(PLOTS_DIR, f"load_factor_kicks_arity{a}_b{bs}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"    Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Insert Throughput
# ═══════════════════════════════════════════════════════════════════════════════

def plot_insert_throughput():
    """Insert throughput — insert until full.

    Input:  results/insert_throughput.csv
    Plot:   One figure per arity, each with one subplot per bucket_size. Within
            each subplot, bars are grouped by num_buckets; each group has two
            adjacent bars — segmented (orange) and standard (blue). Y-axis =
            mean Mops/sec across trials.
    Output: results/plots/insert_throughput_{arity}ary.png  (one per arity)
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "insert_throughput.csv"))
    if "mean_mops" not in df.columns:
        df = df.groupby(["scheme", "arity", "num_buckets", "bucket_size"]).agg(
            mean_mops=("throughput_mops", "mean")
        ).reset_index()

    bucket_sizes = sorted(df["bucket_size"].unique())

    for arity in sorted(df["arity"].unique()):
        adata = df[df["arity"] == arity]
        n_values = sorted(adata["num_buckets"].unique())

        fig, axes = plt.subplots(
            1, len(bucket_sizes), figsize=(4 * len(bucket_sizes), 5), squeeze=False
        )

        for col_idx, bs in enumerate(bucket_sizes):
            ax = axes[0][col_idx]
            bdata = adata[adata["bucket_size"] == bs]

            x = list(range(len(n_values)))
            width = 0.35

            for i, n_val in enumerate(n_values):
                for j, scheme in enumerate(["segmented", "standard"]):
                    match = bdata[(bdata["num_buckets"] == n_val) & (bdata["scheme"] == scheme)]
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
            ax.set_title(f"b={bs}")
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
    Plot:   Same layout as plot_insert_throughput. Y-axis = mean Mops/sec for
            the deletion loop only (fill phase excluded from timing).
    Output: results/plots/delete_throughput_{arity}ary.png  (one per arity)
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "delete_throughput.csv"))
    if "mean_mops" not in df.columns:
        df = df.groupby(["scheme", "arity", "num_buckets", "bucket_size"]).agg(
            mean_mops=("throughput_mops", "mean")
        ).reset_index()

    bucket_sizes = sorted(df["bucket_size"].unique())

    for arity in sorted(df["arity"].unique()):
        adata = df[df["arity"] == arity]
        n_values = sorted(adata["num_buckets"].unique())

        fig, axes = plt.subplots(
            1, len(bucket_sizes), figsize=(4 * len(bucket_sizes), 5), squeeze=False
        )

        for col_idx, bs in enumerate(bucket_sizes):
            ax = axes[0][col_idx]
            bdata = adata[adata["bucket_size"] == bs]

            x = list(range(len(n_values)))
            width = 0.35

            for i, n_val in enumerate(n_values):
                for j, scheme in enumerate(["segmented", "standard"]):
                    match = bdata[(bdata["num_buckets"] == n_val) & (bdata["scheme"] == scheme)]
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
            ax.set_title(f"b={bs}")
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
    Plot:   One figure per (num_buckets, bucket_size) pair.
            X-axis: 6 scheme groups (seg 2-ary, std 2-ary, seg 3-ary, std 3-ary,
            seg 4-ary, std 4-ary). Within each group: 5 bars for hit rates
            0/25/50/75/100%. Segmented 3-ary may have a different num_buckets
            (nearest valid 3·2^m); its label shows the actual num_buckets used.
    Output: results/plots/lookup_throughput_n{n}_b{b}.png  (one per pair)
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "lookup_throughput.csv"))
    if "mean_mops" not in df.columns:
        df = df.groupby(
            ["scheme", "arity", "num_buckets", "bucket_size", "hit_rate_pct"]
        ).agg(mean_mops=("throughput_mops", "mean")).reset_index()

    # 2-ary num_buckets is the reference grid; 3-ary segmented may differ.
    ref_n_values = sorted(df[df["arity"] == 2]["num_buckets"].unique())
    bucket_sizes = sorted(df["bucket_size"].unique())

    for ref_n in ref_n_values:
        for bs in bucket_sizes:
            slice2 = df[(df["arity"] == 2) & (df["num_buckets"] == ref_n) & (df["bucket_size"] == bs)]
            slice4 = df[(df["arity"] == 4) & (df["num_buckets"] == ref_n) & (df["bucket_size"] == bs)]
            slice3 = df[(df["arity"] == 3) & (df["bucket_size"] == bs)]

            if slice2.empty and slice3.empty and slice4.empty:
                continue

            std3_ns = sorted(slice3[slice3["scheme"] == "standard"]["num_buckets"].unique())
            seg3_ns = sorted(slice3[slice3["scheme"] == "segmented"]["num_buckets"].unique())

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
                        (df["num_buckets"] == n_val) &
                        (df["bucket_size"] == bs) &
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
            ax.set_title(f"Lookup Throughput — n={int(ref_n)}, b={int(bs)}")
            ax.legend(title="Hit Rate")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(
                os.path.join(PLOTS_DIR, f"lookup_throughput_n{int(ref_n)}_b{int(bs)}.png"),
                dpi=150,
            )
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. False Positive Rate
# ═══════════════════════════════════════════════════════════════════════════════

# FPR CSVs are named `arity{a}_num_buckets{n}_bucket_size{b}.csv` by the bench.
FPR_FILE_GLOB  = "arity*_num_buckets*_bucket_size*.csv"
FPR_FILE_REGEX = re.compile(r"arity(\d+)_num_buckets(\d+)_bucket_size(\d+)")


def plot_fpr_load_factor():
    """FPR — load factor vs fingerprint bits.

    Input:  results/fpr/arity{a}_num_buckets{n}_bucket_size{b}.csv (one per config)
    Plot:   Per CSV file: one figure, load factor (y) vs fingerprint bits (x),
            one line per scheme.
    Output: results/plots/fpr_lf_arity{a}_n{n}_b{b}.png  (one per config)
    """
    fpr_dir = os.path.join(RESULTS_DIR, "fpr")
    if not os.path.isdir(fpr_dir):
        return

    for csv_file in sorted(glob.glob(os.path.join(fpr_dir, FPR_FILE_GLOB))):
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        m = FPR_FILE_REGEX.match(basename)
        if not m:
            continue
        arity, num_buckets, bucket_size = int(m.group(1)), int(m.group(2)), int(m.group(3))

        df = pd.read_csv(csv_file)

        fig, ax = plt.subplots(figsize=(8, 5))
        for scheme, grp in df.groupby("scheme"):
            grp = grp.sort_values("fingerprint_bits")
            ax.plot(
                grp["fingerprint_bits"], grp["load_factor"],
                marker="o",
                color=SCHEME_COLORS.get(scheme, "gray"),
                label=scheme,
                linewidth=1.2,
            )
        ax.set_xlabel("Fingerprint Bits")
        ax.set_ylabel("Load Factor")
        ax.set_title(
            f"Load Factor vs Fingerprint Bits — {arity}-ary, "
            f"n={num_buckets}, b={bucket_size}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(
            PLOTS_DIR,
            f"fpr_lf_arity{arity}_n{num_buckets}_b{bucket_size}.png",
        )
        fig.savefig(out, dpi=150)
        plt.close(fig)


def plot_fpr_comparison():
    """FPR vs fingerprint bits with theoretical bound.

    Input:  results/fpr/arity{a}_num_buckets{n}_bucket_size{b}.csv — auto-selects
            the (num_buckets, bucket_size) pair with the largest capacity
            >= 1M (num_buckets*bucket_size >= 1_000_000).
    Plot:   3 subplots (one per arity). FPR% (log y-scale) vs fingerprint bits
            for standard and segmented, plus a dashed theoretical line
            arity*bucket_size/2^fp_bits.
    Output: results/plots/fpr_comparison.png
    """
    fpr_dir = os.path.join(RESULTS_DIR, "fpr")
    if not os.path.isdir(fpr_dir):
        return

    best_n, best_b, best_cap = None, None, 0
    for csv_file in glob.glob(os.path.join(fpr_dir, FPR_FILE_GLOB)):
        m = FPR_FILE_REGEX.match(os.path.basename(csv_file))
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
        csv_file = os.path.join(
            fpr_dir,
            f"arity{arity}_num_buckets{best_n}_bucket_size{best_b}.csv",
        )
        if not os.path.exists(csv_file):
            ax.set_visible(False)
            continue

        df = pd.read_csv(csv_file)

        for scheme, grp in df.groupby("scheme"):
            grp = grp.sort_values("fingerprint_bits")
            ax.plot(
                grp["fingerprint_bits"], grp["fpr_pct"],
                marker="o",
                color=SCHEME_COLORS.get(scheme, "gray"),
                label=scheme,
                linewidth=1.2,
            )

        fp_range = sorted(df["fingerprint_bits"].unique())
        theoretical = [arity * best_b / (2 ** f) * 100.0 for f in fp_range]
        ax.plot(fp_range, theoretical, "k--", linewidth=2, alpha=0.5,
                label=f"theoretical ({arity}·b/2^f)")

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
# 6. Degree Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_degree_index():
    """Bucket degree vs bucket index — spatial pattern of load distribution.

    Input:  results/degree_per_bucket.csv
    Plot:   For each (arity, num_buckets, bucket_size): one figure with 2
            side-by-side scatter plots (standard left, segmented right).
            X = bucket index, Y = degree. Downsampled to max 4000 points for
            readability.
    Output: results/plots/degree_distribution/degree_{arity}ary_n{n}_b{b}.png
    """
    csv_path = os.path.join(RESULTS_DIR, "degree_per_bucket.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    degree_dir = os.path.join(PLOTS_DIR, "degree_distribution")
    os.makedirs(degree_dir, exist_ok=True)

    arities = sorted(df["arity"].unique())
    bucket_sizes = sorted(df["bucket_size"].unique())

    for arity in arities:
        adf = df[df["arity"] == arity]
        std_ns = sorted(adf[adf["scheme"] == "standard"]["num_buckets"].unique())
        seg_ns = sorted(adf[adf["scheme"] == "segmented"]["num_buckets"].unique())

        for n_std, n_seg in zip(std_ns, seg_ns):
            for bs in bucket_sizes:
                std_rows = adf[
                    (adf["scheme"] == "standard")
                    & (adf["num_buckets"] == n_std)
                    & (adf["bucket_size"] == bs)
                ].sort_values("bucket_index")
                seg_rows = adf[
                    (adf["scheme"] == "segmented")
                    & (adf["num_buckets"] == n_seg)
                    & (adf["bucket_size"] == bs)
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
                    f"Bucket Degree vs Index — {arity}-ary, b={bs}, fp_bits=12",
                    fontsize=12,
                )
                fig.tight_layout()
                fname = f"degree_{arity}ary_n{n_std}_b{bs}.png"
                fig.savefig(os.path.join(degree_dir, fname), dpi=150)
                plt.close(fig)


def plot_degree_histogram():
    """Degree distribution — fraction of buckets at each degree level.

    Input:  results/degree_distribution.csv
    Plot:   For each (arity, bucket_size): one line plot. X = bucket degree,
            Y = fraction of all buckets with that degree. Compares shape
            between standard (Poisson-like) and segmented schemes.
    Output: results/plots/degree_hist_{arity}ary_b{b}.png
    """
    csv_path = os.path.join(RESULTS_DIR, "degree_distribution.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    for arity in sorted(df["arity"].unique()):
        for bs in sorted(df["bucket_size"].unique()):
            subset = df[(df["arity"] == arity) & (df["bucket_size"] == bs)]
            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))

            for (scheme, n), grp in subset.groupby(["scheme", "num_buckets"]):
                total_buckets = grp["count"].sum()
                grp = grp.sort_values("degree")
                ax.plot(
                    grp["degree"],
                    grp["count"] / total_buckets,
                    marker=B_MARKERS.get(bs, "o"),
                    color=SCHEME_COLORS.get(scheme, "gray"),
                    label=f"{scheme} (n={n})",
                    linewidth=1.2,
                )

            ax.set_xlabel("Bucket Degree")
            ax.set_ylabel("Fraction of Buckets")
            ax.set_title(f"Degree Distribution — {arity}-ary, b={bs}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(PLOTS_DIR, f"degree_hist_{arity}ary_b{bs}.png"),
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
    "degree_index":           plot_degree_index,
    "degree_histogram":       plot_degree_histogram,
}


def _run_all():
    """Run every plot function that has corresponding CSV data present."""
    if os.path.exists(os.path.join(RESULTS_DIR, "load_factor.csv")):
        plot_load_factor_all()
        print("  Generated load_factor_all.png")
        plot_load_factor_b234()
        print("  Generated load_factor_b234.png")
        df_lf = pd.read_csv(os.path.join(RESULTS_DIR, "load_factor.csv"))
        if "max_kicks" in df_lf.columns:
            plot_load_factor_by_kicks()
            print("  Generated load_factor_kicks_*.png")

    if os.path.exists(os.path.join(RESULTS_DIR, "insert_throughput.csv")):
        plot_insert_throughput()
        print("  Generated insert_throughput_{2,3,4}ary.png")

    if os.path.exists(os.path.join(RESULTS_DIR, "delete_throughput.csv")):
        plot_delete_throughput()
        print("  Generated delete_throughput_{2,3,4}ary.png")

    if os.path.exists(os.path.join(RESULTS_DIR, "lookup_throughput.csv")):
        plot_lookup_throughput()
        print("  Generated lookup_throughput_*.png")

    if os.path.isdir(os.path.join(RESULTS_DIR, "fpr")):
        plot_fpr_load_factor()
        print("  Generated fpr_lf_*.png")
        plot_fpr_comparison()
        print("  Generated fpr_comparison.png")

    if os.path.exists(os.path.join(RESULTS_DIR, "degree_per_bucket.csv")):
        plot_degree_index()
        print("  Generated degree_distribution/ (degree-index scatter)")

    if os.path.exists(os.path.join(RESULTS_DIR, "degree_distribution.csv")):
        plot_degree_histogram()
        print("  Generated degree_hist_*.png")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from segmented-cuckoo-filter benchmark CSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available plot functions:
  load_factor_all          Load factor — all bucket_size values
  load_factor_b234         Load factor — selective bucket_size for clarity
  load_factor_by_kicks     Load factor vs MAX_KICKS sweep  [arity] [bucket_size]
  insert_throughput        Insert throughput bars per arity
  delete_throughput        Delete throughput bars per arity
  lookup_throughput        Lookup throughput at 5 hit rates
  fpr_load_factor          FPR: load factor vs fp_bits
  fpr_comparison           FPR vs fp_bits with theoretical bound
  degree_index             Bucket degree vs index scatter
  degree_histogram         Degree histogram per (arity, bucket_size)

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
