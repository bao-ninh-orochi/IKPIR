#!/usr/bin/env python3
"""Generate plots from ikpir-client benchmark CSV results.

Usage:
    python scripts/plot.py                      # generate every plot whose CSV is present
    python scripts/plot.py <function>           # one specific plot
    python scripts/plot.py --list               # list available plot functions

Reads CSV files from `results/` (the default bench output directory) and
writes PNG charts to `results/plots/`. Override via `IKPIR_CLIENT_RESULTS_DIR`
/ `IKPIR_CLIENT_PLOTS_DIR` to point at a scratch directory.

Each plotter is multi-config aware: when the CSV holds one row, you get
one data point; when an orchestrator (e.g. `scripts/run_benches.sh`) has
swept across (m, w, mode) the same plotter draws comparison curves.

Anchor configurations the plots are aligned with:
    ChalametPIR Table 2   query/parse client timings at w = 1 kB
    Hao-2025 Table 1      client side of online time
    Hao-2025 Figure 10    value-length curve
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.environ.get("IKPIR_CLIENT_RESULTS_DIR", "results")
PLOTS_DIR   = os.environ.get("IKPIR_CLIENT_PLOTS_DIR",   "results/plots")

# ─────────────────────────────────────────────────────────────────────────────
# Style maps (shared semantics with the server plot.py)
# ─────────────────────────────────────────────────────────────────────────────

VALUE_BITS_COLORS = {
    32:   "#1f77b4",
    64:   "#ff7f0e",
    128:  "#2ca02c",
    256:  "#d62728",
    512:  "#9467bd",
    1024: "#8c564b",
    2048: "#e377c2",
    4096: "#7f7f7f",
    8192: "#17becf",
}

ARITY_LINESTYLES = {2: "-", 3: "--", 4: ":"}
ARITY_MARKERS    = {2: "o", 3: "s", 4: "^"}

MODE_COLORS = {
    "cold":    "#1f77b4",
    "warm-b":  "#ff7f0e",
    "warm-bc": "#2ca02c",
}
MODE_MARKERS = {
    "cold":    "o",
    "warm-b":  "s",
    "warm-bc": "^",
}


def _color_for_vb(vb):
    return VALUE_BITS_COLORS.get(int(vb), "gray")


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _load(csv_name):
    return pd.read_csv(os.path.join(RESULTS_DIR, csv_name))


def _throughput_two_panel(df, title, ylabel, mean_col, stddev_col, out_name, has_mode=True):
    """Two-panel: (a) qps vs num_buckets, (b) qps vs value_bits.

    When `has_mode` is True, groups are coloured by mode.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — vs num_buckets, at each mode (when present).
    if has_mode and "mode" in df.columns:
        for (mode, ar, vb), grp in df.groupby(["mode", "arity", "value_bits"]):
            grp = grp.sort_values("num_buckets")
            if grp["num_buckets"].nunique() < 2:
                continue
            ax1.errorbar(
                grp["num_buckets"], grp[mean_col], yerr=grp[stddev_col],
                marker=MODE_MARKERS.get(mode, "x"),
                color=MODE_COLORS.get(mode, "gray"),
                linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
                label=f"{mode} arity={int(ar)} vb={int(vb)}",
                linewidth=1.4, capsize=3,
            )
    else:
        for (ar, vb), grp in df.groupby(["arity", "value_bits"]):
            grp = grp.sort_values("num_buckets")
            if grp["num_buckets"].nunique() < 2:
                continue
            ax1.errorbar(
                grp["num_buckets"], grp[mean_col], yerr=grp[stddev_col],
                marker=ARITY_MARKERS.get(int(ar), "x"),
                color=_color_for_vb(vb),
                linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
                label=f"arity={int(ar)} vb={int(vb)}",
                linewidth=1.4, capsize=3,
            )
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("num_buckets (log₂)")
    ax1.set_ylabel(ylabel)
    ax1.set_title("(a) m sweep")
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B — vs value_bits, at each num_buckets (limit to cold to keep readable).
    df_b = df
    if has_mode and "mode" in df.columns:
        df_b = df[df["mode"] == "cold"]
    for (ar, nb), grp in df_b.groupby(["arity", "num_buckets"]):
        grp = grp.sort_values("value_bits")
        if grp["value_bits"].nunique() < 2:
            continue
        ax2.errorbar(
            grp["value_bits"], grp[mean_col], yerr=grp[stddev_col],
            marker=ARITY_MARKERS.get(int(ar), "x"),
            linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
            label=f"arity={int(ar)} nb={int(nb)}{' cold' if has_mode else ''}",
            linewidth=1.4, capsize=3,
        )
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("value_bits (log₂)")
    ax2.set_ylabel(ylabel)
    ax2.set_title("(b) w sweep")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(title, fontsize=13)
    _save(fig, out_name)


# ═════════════════════════════════════════════════════════════════════════════
# 1. query_throughput
# ═════════════════════════════════════════════════════════════════════════════

def plot_query_throughput():
    """build_query throughput (qps) vs num_buckets / value_bits, per mode.

    Input:  results/ikpir_client_query_throughput.csv
    Output: results/plots/query_throughput.png
    Anchor: ChalametPIR Table 2 Query timings.
    """
    df = _load("ikpir_client_query_throughput.csv")
    _throughput_two_panel(
        df,
        "ikpir-client build_query throughput (FrodoPIR)",
        "build_query (qps, log)",
        "mean_qps", "stddev_qps",
        "query_throughput.png",
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. decode_throughput
# ═════════════════════════════════════════════════════════════════════════════

def plot_decode_throughput():
    """decode throughput (qps) vs num_buckets / value_bits, per mode.

    Input:  results/ikpir_client_decode_throughput.csv
    Output: results/plots/decode_throughput.png
    Anchor: ChalametPIR Table 2 Parsing column.
    """
    df = _load("ikpir_client_decode_throughput.csv")
    _throughput_two_panel(
        df,
        "ikpir-client decode throughput (FrodoPIR)",
        "decode (qps, log)",
        "mean_qps", "stddev_qps",
        "decode_throughput.png",
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. apply_delta_throughput
# ═════════════════════════════════════════════════════════════════════════════

def plot_apply_delta_throughput():
    """apply_delta throughput (deltas/sec) vs num_buckets, by precomputed_slots.

    Input:  results/ikpir_client_apply_delta_throughput.csv
    Output: results/plots/apply_delta_throughput.png
    """
    df = _load("ikpir_client_apply_delta_throughput.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    for (ar, pre), grp in df.groupby(["arity", "precomputed_slots"]):
        grp = grp.sort_values("num_buckets")
        warm_label = "warm" if int(pre) > 0 else "cold"
        ax.errorbar(
            grp["num_buckets"], grp["mean_dps"], yerr=grp["stddev_dps"],
            marker=ARITY_MARKERS.get(int(ar), "x"),
            linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
            color="#1f77b4" if warm_label == "cold" else "#ff7f0e",
            label=f"arity={int(ar)} pre={int(pre)} ({warm_label})",
            linewidth=1.4, capsize=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("num_buckets (log₂)")
    ax.set_ylabel("apply_delta (dps, log)")
    ax.set_title("ikpir-client apply_delta throughput (FrodoPIR)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, "apply_delta_throughput.png")


# ═════════════════════════════════════════════════════════════════════════════
# 4. preprocess_throughput — Phase B and Phase C rates side-by-side
# ═════════════════════════════════════════════════════════════════════════════

def plot_preprocess_throughput():
    """Phase B (precompute_queries) and Phase C (precompute_decodes) slots/sec.

    Input:  results/ikpir_client_preprocess_throughput.csv
    Output: results/plots/preprocess_throughput.png
    Two panels (B / C) — both vs num_buckets, grouped by (arity, value_bits).
    """
    df = _load("ikpir_client_preprocess_throughput.csv")
    fig, (axB, axC) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, mean_col, std_col, title in (
        (axB, "mean_phase_b_sps", "stddev_phase_b_sps", "(a) Phase B (b = A·s + e)"),
        (axC, "mean_phase_c_sps", "stddev_phase_c_sps", "(b) Phase C (c = sᵀ·H)"),
    ):
        for (ar, vb), grp in df.groupby(["arity", "value_bits"]):
            grp = grp.sort_values("num_buckets")
            ax.errorbar(
                grp["num_buckets"], grp[mean_col], yerr=grp[std_col],
                marker=ARITY_MARKERS.get(int(ar), "x"),
                color=_color_for_vb(vb),
                linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
                label=f"arity={int(ar)} vb={int(vb)}",
                linewidth=1.4, capsize=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("num_buckets (log₂)")
        ax.set_ylabel("Throughput (slots/sec, log)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("ikpir-client preprocessing throughput (FrodoPIR)", fontsize=13)
    _save(fig, "preprocess_throughput.png")


# ═════════════════════════════════════════════════════════════════════════════
# 5. client_setup_latency
# ═════════════════════════════════════════════════════════════════════════════

def plot_client_setup_latency():
    """`IkpirClient::from_setup` latency, with vs without precompute.

    Input:  results/ikpir_client_setup_latency.csv
    Output: results/plots/client_setup_latency.png
    """
    df = _load("ikpir_client_setup_latency.csv")
    fig, ax = plt.subplots(figsize=(9, 6))

    for (ar, wp), grp in df.groupby(["arity", "with_precompute"]):
        grp = grp.sort_values("num_buckets")
        label = f"arity={int(ar)} " + ("from_setup+precompute" if int(wp) else "from_setup only")
        ax.errorbar(
            grp["num_buckets"], grp["mean_setup_ms"], yerr=grp["stddev_setup_ms"],
            marker=ARITY_MARKERS.get(int(ar), "x"),
            linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
            color="#1f77b4" if int(wp) == 0 else "#d62728",
            label=label, linewidth=1.4, capsize=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("num_buckets (log₂)")
    ax.set_ylabel("Client setup latency (ms, log)")
    ax.set_title("ikpir-client from_setup latency (FrodoPIR)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, "client_setup_latency.png")


# ═════════════════════════════════════════════════════════════════════════════
# 6. client_memory_footprint
# ═════════════════════════════════════════════════════════════════════════════

def plot_client_memory_footprint():
    """Client heap+stack footprint vs num_buckets, grouped by preprocessing mode.

    Input:  results/ikpir_client_memory_footprint.csv
    Output: results/plots/client_memory_footprint.png
    """
    df = _load("ikpir_client_memory_footprint.csv")
    fig, ax = plt.subplots(figsize=(9, 6))

    for (mode, ar), grp in df.groupby(["mode", "arity"]):
        grp = grp.sort_values("num_buckets")
        ax.plot(
            grp["num_buckets"], grp["total_bytes"],
            marker=MODE_MARKERS.get(mode, "x"),
            color=MODE_COLORS.get(mode, "gray"),
            linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
            label=f"{mode} arity={int(ar)}",
            linewidth=1.4,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("num_buckets (log₂)")
    ax.set_ylabel("Client memory (bytes, log)")
    ax.set_title("ikpir-client memory footprint (FrodoPIR)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, "client_memory_footprint.png")


# ═════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═════════════════════════════════════════════════════════════════════════════

PLOT_FUNCTIONS = {
    "query_throughput":        plot_query_throughput,
    "decode_throughput":       plot_decode_throughput,
    "apply_delta_throughput":  plot_apply_delta_throughput,
    "preprocess_throughput":   plot_preprocess_throughput,
    "client_setup_latency":    plot_client_setup_latency,
    "client_memory_footprint": plot_client_memory_footprint,
}

CSV_FOR_PLOT = {
    "query_throughput":        "ikpir_client_query_throughput.csv",
    "decode_throughput":       "ikpir_client_decode_throughput.csv",
    "apply_delta_throughput":  "ikpir_client_apply_delta_throughput.csv",
    "preprocess_throughput":   "ikpir_client_preprocess_throughput.csv",
    "client_setup_latency":    "ikpir_client_setup_latency.csv",
    "client_memory_footprint": "ikpir_client_memory_footprint.csv",
}


def _run_all():
    any_run = False
    for name, fn in PLOT_FUNCTIONS.items():
        csv_path = os.path.join(RESULTS_DIR, CSV_FOR_PLOT[name])
        if not os.path.exists(csv_path):
            continue
        try:
            fn()
            print(f"  Generated {name}.png")
            any_run = True
        except Exception as exc:
            print(f"  Skip {name}: {exc}", file=sys.stderr)
    if not any_run:
        print(f"No CSVs found under {RESULTS_DIR}/. Run scripts/run_benches.sh first.")
    else:
        print(f"\nPlots saved to {PLOTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from ikpir-client benchmark CSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Plot ↔ bench mapping:\n" + "\n".join(
            f"  {name:<24} ← {CSV_FOR_PLOT[name]}" for name in PLOT_FUNCTIONS),
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
            print(f"  {name:<24} {first_line}")
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
    print(f"  Generated {args.function}.png → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
