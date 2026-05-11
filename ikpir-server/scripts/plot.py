#!/usr/bin/env python3
"""Generate plots from ikpir-server benchmark CSV results.

Usage:
    python scripts/plot.py                      # generate all plots
    python scripts/plot.py <function>           # run one plot function
    python scripts/plot.py --list               # list available functions

Reads CSV files from `results/` (the default bench output directory) and
writes PNG charts to `results/plots/`. Override via `IKPIR_SERVER_RESULTS_DIR`
/ `IKPIR_SERVER_PLOTS_DIR` env vars to point at a scratch directory.

Expected CSV schemas (written by the benches in `benches/`):
    ikpir_server_setup_throughput.csv
        arity, num_buckets, bucket_size, value_bits, lwe_dim,
        mean_setup_ms, min_setup_ms, max_setup_ms, stddev_setup_ms

    ikpir_server_answer_throughput.csv
        arity, num_buckets, bucket_size, value_bits, batch,
        mean_qps, min_qps, max_qps, stddev_qps

    ikpir_server_incremental_vs_rebuild.csv
        arity, num_buckets, bucket_size, value_bits, n_mutations,
        incremental_ms, rebuild_ms, ratio_inc_over_rebuild
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.environ.get("IKPIR_SERVER_RESULTS_DIR", "results")
PLOTS_DIR   = os.environ.get("IKPIR_SERVER_PLOTS_DIR",   "results/plots")

# Consistent colours across all server plots: one per value_bits.
VALUE_BITS_COLORS = {8: "#1f77b4", 64: "#ff7f0e", 256: "#2ca02c"}
# Marker per bucket_size.
BUCKET_MARKERS = {1: "D", 2: "s", 3: "^", 4: "o"}
# Line style per mode (incremental vs rebuild).
MODE_STYLES = {"incremental": ("-",  "#ff7f0e"), "rebuild": ("--", "#1f77b4")}
# Line style per arity — applied as a secondary dimension when more than
# one arity is present in the same CSV.
ARITY_LINESTYLES = {2: "-", 3: "--", 4: ":"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Setup throughput
# ═══════════════════════════════════════════════════════════════════════════════

def plot_setup_throughput():
    """Setup wall-clock cost vs num_buckets, grouped by (arity, bucket_size, value_bits).

    Input:  results/ikpir_server_setup_throughput.csv
    Plot:   One subplot per bucket_size. X = num_buckets (log2 scale),
            Y = mean_setup_ms with stddev error bars. Colour = value_bits,
            line style = arity (when more than one arity is present).
    Output: results/plots/setup_throughput.png
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ikpir_server_setup_throughput.csv"))
    bucket_sizes = sorted(df["bucket_size"].unique())
    arities = sorted(df["arity"].unique())
    fig, axes = plt.subplots(1, len(bucket_sizes), figsize=(6 * len(bucket_sizes), 5), squeeze=False)

    for idx, bs in enumerate(bucket_sizes):
        ax = axes[0][idx]
        for ar in arities:
            for vb, grp in df[(df["bucket_size"] == bs) & (df["arity"] == ar)].groupby("value_bits"):
                grp = grp.sort_values("num_buckets")
                label = f"vb={int(vb)}" if len(arities) == 1 else f"vb={int(vb)} ar={int(ar)}"
                ax.errorbar(
                    grp["num_buckets"], grp["mean_setup_ms"], yerr=grp["stddev_setup_ms"],
                    marker=BUCKET_MARKERS.get(int(bs), "x"),
                    color=VALUE_BITS_COLORS.get(int(vb), "gray"),
                    linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
                    label=label,
                    linewidth=1.2, capsize=3,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("num_buckets")
        ax.set_ylabel("Setup time (ms)")
        ax.set_title(f"bucket_size = {int(bs)}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("ikpir-server setup throughput (FrodoPirBackend)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "setup_throughput.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Answer throughput
# ═══════════════════════════════════════════════════════════════════════════════

def plot_answer_throughput():
    """Answer throughput (queries/sec) vs num_buckets, grouped by (arity, bucket_size, value_bits).

    Input:  results/ikpir_server_answer_throughput.csv
    Plot:   One subplot per bucket_size. X = num_buckets (log2 scale),
            Y = mean_qps with stddev error bars. Colour = value_bits,
            line style = arity (when more than one arity is present).
    Output: results/plots/answer_throughput.png
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ikpir_server_answer_throughput.csv"))
    bucket_sizes = sorted(df["bucket_size"].unique())
    arities = sorted(df["arity"].unique())
    fig, axes = plt.subplots(1, len(bucket_sizes), figsize=(6 * len(bucket_sizes), 5), squeeze=False)

    for idx, bs in enumerate(bucket_sizes):
        ax = axes[0][idx]
        for ar in arities:
            for vb, grp in df[(df["bucket_size"] == bs) & (df["arity"] == ar)].groupby("value_bits"):
                grp = grp.sort_values("num_buckets")
                label = f"vb={int(vb)}" if len(arities) == 1 else f"vb={int(vb)} ar={int(ar)}"
                ax.errorbar(
                    grp["num_buckets"], grp["mean_qps"], yerr=grp["stddev_qps"],
                    marker=BUCKET_MARKERS.get(int(bs), "x"),
                    color=VALUE_BITS_COLORS.get(int(vb), "gray"),
                    linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
                    label=label,
                    linewidth=1.2, capsize=3,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("num_buckets")
        ax.set_ylabel("Answer throughput (qps)")
        ax.set_title(f"bucket_size = {int(bs)}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("ikpir-server answer throughput (FrodoPirBackend)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "answer_throughput.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Incremental vs full_rebuild — the headline plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_incremental_vs_rebuild():
    """Incremental hint patch vs full_rebuild, time vs n_mutations.

    The headline IKPIR result: for small mutation footprints, incremental
    patching is dramatically cheaper than rebuilding; the curves cross
    once N grows large enough.

    Input:  results/ikpir_server_incremental_vs_rebuild.csv
    Plot:   One subplot per (arity, num_buckets) cell. X = n_mutations (log),
            Y = ms (log). Two lines per subplot: incremental_ms, rebuild_ms.
    Output: results/plots/incremental_vs_rebuild.png
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ikpir_server_incremental_vs_rebuild.csv"))
    arities = sorted(df["arity"].unique())
    nbs_per_arity = {ar: sorted(df[df["arity"] == ar]["num_buckets"].unique()) for ar in arities}
    n_cols = max(len(v) for v in nbs_per_arity.values())
    n_rows = len(arities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for r, ar in enumerate(arities):
        for c, nb in enumerate(nbs_per_arity[ar]):
            ax = axes[r][c]
            sub = df[(df["arity"] == ar) & (df["num_buckets"] == nb)].sort_values("n_mutations")
            if sub.empty:
                ax.set_visible(False)
                continue
            for mode in ("incremental", "rebuild"):
                style, color = MODE_STYLES[mode]
                ax.plot(
                    sub["n_mutations"], sub[f"{mode}_ms"],
                    marker="o", linestyle=style, color=color, label=mode, linewidth=1.4,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("n_mutations")
            ax.set_ylabel("Time (ms, log scale)")
            ax.set_title(f"arity = {int(ar)}, num_buckets = {int(nb)}")
            ax.legend()
            ax.grid(True, alpha=0.3, which="both")
        # Hide unused columns when this arity has fewer num_buckets than the widest row.
        for c in range(len(nbs_per_arity[ar]), n_cols):
            axes[r][c].set_visible(False)

    fig.suptitle("Incremental hint patch vs full_rebuild (FrodoPirBackend)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "incremental_vs_rebuild.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_FUNCTIONS = {
    "setup_throughput":        plot_setup_throughput,
    "answer_throughput":       plot_answer_throughput,
    "incremental_vs_rebuild":  plot_incremental_vs_rebuild,
}


def _run_all():
    """Run every plot function whose CSV is present."""
    if os.path.exists(os.path.join(RESULTS_DIR, "ikpir_server_setup_throughput.csv")):
        plot_setup_throughput()
        print("  Generated setup_throughput.png")
    if os.path.exists(os.path.join(RESULTS_DIR, "ikpir_server_answer_throughput.csv")):
        plot_answer_throughput()
        print("  Generated answer_throughput.png")
    if os.path.exists(os.path.join(RESULTS_DIR, "ikpir_server_incremental_vs_rebuild.csv")):
        plot_incremental_vs_rebuild()
        print("  Generated incremental_vs_rebuild.png")
    print(f"\nAll plots saved to {PLOTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from ikpir-server benchmark CSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available plot functions:
  setup_throughput          Setup wall-clock cost vs num_buckets
  answer_throughput         Answer rate (qps) vs num_buckets
  incremental_vs_rebuild    Headline: incremental patch vs full rebuild

Examples:
  python scripts/plot.py
  python scripts/plot.py incremental_vs_rebuild
""",
    )
    parser.add_argument("function", nargs="?", default=None,
                        choices=list(PLOT_FUNCTIONS.keys()), metavar="function",
                        help="Plot function to run. Omit to run all.")
    parser.add_argument("--list", action="store_true",
                        help="List available plot functions and exit.")

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
    try:
        fn()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Run the corresponding bench first to generate the CSV.", file=sys.stderr)
        sys.exit(1)
    print(f"  Generated {args.function} plot → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
