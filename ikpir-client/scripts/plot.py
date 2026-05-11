#!/usr/bin/env python3
"""Generate plots from ikpir-client benchmark CSV results.

Usage:
    python scripts/plot.py                      # generate all plots
    python scripts/plot.py <function>           # run one plot function
    python scripts/plot.py --list               # list available functions

Reads CSV files from `results/` (the default bench output directory) and
writes PNG charts to `results/plots/`. Override via `IKPIR_CLIENT_RESULTS_DIR`
/ `IKPIR_CLIENT_PLOTS_DIR` env vars to point at a scratch directory.

Expected CSV schemas (written by the benches in `benches/`):
    ikpir_client_query_throughput.csv
        arity, num_buckets, bucket_size, value_bits, batch,
        mean_qps, min_qps, max_qps, stddev_qps

    ikpir_client_decode_throughput.csv
        arity, num_buckets, bucket_size, value_bits, batch,
        mean_qps, min_qps, max_qps, stddev_qps

    ikpir_client_apply_delta_throughput.csv
        arity, num_buckets, bucket_size, value_bits, batch,
        mean_dps, min_dps, max_dps, stddev_dps
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.environ.get("IKPIR_CLIENT_RESULTS_DIR", "results")
PLOTS_DIR   = os.environ.get("IKPIR_CLIENT_PLOTS_DIR",   "results/plots")

VALUE_BITS_COLORS = {8: "#1f77b4", 64: "#ff7f0e", 256: "#2ca02c"}
BUCKET_MARKERS    = {1: "D", 2: "s", 3: "^", 4: "o"}
# Line style per arity — applied as a secondary dimension when more than
# one arity is present in the same CSV.
ARITY_LINESTYLES  = {2: "-", 3: "--", 4: ":"}


def _throughput_grid(csv_name, title, ylabel, mean_col, stddev_col, out_name):
    """Shared shape: one subplot per bucket_size, X = num_buckets (log2),
    Y = mean ± stddev. Colour = value_bits, line style = arity (when
    more than one arity is present).
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, csv_name))
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
                    grp["num_buckets"], grp[mean_col], yerr=grp[stddev_col],
                    marker=BUCKET_MARKERS.get(int(bs), "x"),
                    color=VALUE_BITS_COLORS.get(int(vb), "gray"),
                    linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
                    label=label,
                    linewidth=1.2, capsize=3,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("num_buckets")
        ax.set_ylabel(ylabel)
        ax.set_title(f"bucket_size = {int(bs)}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, out_name), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. build_query throughput
# ═══════════════════════════════════════════════════════════════════════════════

def plot_query_throughput():
    """build_query throughput (queries/sec) vs num_buckets, by (arity, bucket_size, value_bits).

    Input:  results/ikpir_client_query_throughput.csv
    Output: results/plots/query_throughput.png
    """
    _throughput_grid(
        "ikpir_client_query_throughput.csv",
        "ikpir-client build_query throughput (FrodoPirBackend)",
        "Query construction (qps)",
        "mean_qps", "stddev_qps",
        "query_throughput.png",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. decode throughput
# ═══════════════════════════════════════════════════════════════════════════════

def plot_decode_throughput():
    """decode throughput (queries/sec) vs num_buckets, by (arity, bucket_size, value_bits).

    Input:  results/ikpir_client_decode_throughput.csv
    Output: results/plots/decode_throughput.png
    """
    _throughput_grid(
        "ikpir_client_decode_throughput.csv",
        "ikpir-client decode throughput (FrodoPirBackend)",
        "Decode (qps)",
        "mean_qps", "stddev_qps",
        "decode_throughput.png",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. apply_delta throughput
# ═══════════════════════════════════════════════════════════════════════════════

def plot_apply_delta_throughput():
    """apply_delta throughput (deltas/sec) vs num_buckets, by (arity, bucket_size, value_bits).

    Input:  results/ikpir_client_apply_delta_throughput.csv
    Output: results/plots/apply_delta_throughput.png
    """
    _throughput_grid(
        "ikpir_client_apply_delta_throughput.csv",
        "ikpir-client apply_delta throughput (FrodoPirBackend)",
        "apply_delta (dps)",
        "mean_dps", "stddev_dps",
        "apply_delta_throughput.png",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_FUNCTIONS = {
    "query_throughput":        plot_query_throughput,
    "decode_throughput":       plot_decode_throughput,
    "apply_delta_throughput":  plot_apply_delta_throughput,
}


def _run_all():
    """Run every plot function whose CSV is present."""
    if os.path.exists(os.path.join(RESULTS_DIR, "ikpir_client_query_throughput.csv")):
        plot_query_throughput()
        print("  Generated query_throughput.png")
    if os.path.exists(os.path.join(RESULTS_DIR, "ikpir_client_decode_throughput.csv")):
        plot_decode_throughput()
        print("  Generated decode_throughput.png")
    if os.path.exists(os.path.join(RESULTS_DIR, "ikpir_client_apply_delta_throughput.csv")):
        plot_apply_delta_throughput()
        print("  Generated apply_delta_throughput.png")
    print(f"\nAll plots saved to {PLOTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from ikpir-client benchmark CSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available plot functions:
  query_throughput          build_query rate (qps) vs num_buckets
  decode_throughput         decode rate (qps) vs num_buckets
  apply_delta_throughput    apply_delta rate (dps) vs num_buckets

Examples:
  python scripts/plot.py
  python scripts/plot.py decode_throughput
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
