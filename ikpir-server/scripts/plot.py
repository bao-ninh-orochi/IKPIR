#!/usr/bin/env python3
"""Generate plots from ikpir-server benchmark CSV results.

Usage:
    python scripts/plot.py                      # generate every plot whose CSV is present
    python scripts/plot.py <function>           # one specific plot
    python scripts/plot.py --list               # list available plot functions

Reads CSV files from `results/` (the default bench output directory) and
writes PNG charts to `results/plots/`. Override via `IKPIR_SERVER_RESULTS_DIR`
/ `IKPIR_SERVER_PLOTS_DIR` to point at a scratch directory.

Each plotter is multi-config aware: when the CSV holds one row, you get
one data point; when an orchestrator (e.g. `scripts/run_benches.sh`) has
swept across (m, w, k) the same plotter draws comparison curves. Lines
are grouped by (arity, value_bits) / (arity, num_buckets) / (mode, arity)
depending on which axis the bench varies along.

Anchor configurations the plots are aligned with (so the eye can land
on the comparable cells from each paper):
    ChalametPIR Table 1   m × w grid (FrodoPIR, k ∈ {3, 4})
    ChalametPIR Table 2   per-op timings at w = 1 kB
    Hao-2025 Table 1      m × w grid (m ∈ {2^18, 2^20, 2^22}, w ∈ {32..256}B)
    Hao-2025 Figure 10    value-length curve at m = 2^20
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.environ.get("IKPIR_SERVER_RESULTS_DIR", "results")
PLOTS_DIR   = os.environ.get("IKPIR_SERVER_PLOTS_DIR",   "results/plots")

# ─────────────────────────────────────────────────────────────────────────────
# Style maps
# ─────────────────────────────────────────────────────────────────────────────

# Colours by value_bits (covers 32 .. 8192 spanning Hao-2025 & ChalametPIR).
VALUE_BITS_COLORS = {
    32:   "#1f77b4",   # 4 B   — Hao-2025 Figure 10
    64:   "#ff7f0e",   # 8 B
    128:  "#2ca02c",   # 16 B
    256:  "#d62728",   # 32 B  — Hao-2025 Table 1
    512:  "#9467bd",   # 64 B  — Hao-2025 Table 1
    1024: "#8c564b",   # 128 B — Hao-2025 Table 1
    2048: "#e377c2",   # 256 B — Hao-2025 Table 1 + ChalametPIR Case I
    4096: "#7f7f7f",   # 512 B
    8192: "#17becf",   # 1 kB  — ChalametPIR Table 1
}

ARITY_LINESTYLES = {2: "-", 3: "--", 4: ":"}
ARITY_MARKERS    = {2: "o", 3: "s", 4: "^"}

MUTATION_KIND_COLORS = {
    "insert": "#1f77b4",
    "update": "#2ca02c",
    "delete": "#d62728",
}

MODE_STYLES = {
    "incremental": ("-",  "#ff7f0e"),
    "rebuild":     ("--", "#1f77b4"),
}

PREPROCESS_MODE_COLORS = {
    "cold":    "#1f77b4",
    "warm-b":  "#ff7f0e",
    "warm-bc": "#2ca02c",
}

FAILURE_KIND_COLORS = {
    "stale_epoch": "#1f77b4",
    "table_full":  "#d62728",
}


def _color_for_vb(vb):
    return VALUE_BITS_COLORS.get(int(vb), "gray")


def _vb_to_bytes_label(vb):
    return f"{int(vb)//8}B" if int(vb) >= 8 else f"{int(vb)}b"


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _load(csv_name):
    return pd.read_csv(os.path.join(RESULTS_DIR, csv_name))


# ═════════════════════════════════════════════════════════════════════════════
# 1. setup_latency — server-side `IkpirServer::new`
# ═════════════════════════════════════════════════════════════════════════════

def plot_setup_latency():
    """Setup latency vs num_buckets, grouped by (arity, value_bits).

    Input:  results/ikpir_server_setup_latency.csv
    Output: results/plots/setup_latency.png
    Anchor: ChalametPIR Table 1 setup column (preproc), Hao-2025 Table 1 Setup-Time.
    """
    df = _load("ikpir_server_setup_latency.csv")
    fig, ax = plt.subplots(figsize=(9, 6))

    for (ar, vb), grp in df.groupby(["arity", "value_bits"]):
        grp = grp.sort_values("num_buckets")
        ax.errorbar(
            grp["num_buckets"], grp["mean_setup_ms"], yerr=grp["stddev_setup_ms"],
            marker=ARITY_MARKERS.get(int(ar), "x"),
            color=_color_for_vb(vb),
            linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
            label=f"arity={int(ar)} vb={int(vb)} ({_vb_to_bytes_label(vb)})",
            linewidth=1.4, capsize=3, markersize=6,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("num_buckets (log₂)")
    ax.set_ylabel("Setup time (ms)")
    ax.set_title("ikpir-server setup latency (FrodoPIR)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, "setup_latency.png")


# ═════════════════════════════════════════════════════════════════════════════
# 2. answer_throughput — server PIR matvec rate
# ═════════════════════════════════════════════════════════════════════════════

def plot_answer_throughput():
    """Answer throughput (queries/sec) vs num_buckets / value_bits.

    Input:  results/ikpir_server_answer_throughput.csv
    Output: results/plots/answer_throughput.png
    Anchor: ChalametPIR Table 2 (response), Hao-2025 Table 1 (online time).
    Two panels: (a) qps vs num_buckets at fixed value_bits;
                (b) qps vs value_bits at fixed num_buckets.
    """
    df = _load("ikpir_server_answer_throughput.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — sweep num_buckets at each (arity, value_bits) where multiple num_buckets exist.
    grouped = df.groupby(["arity", "value_bits"])
    plotted_panel_a = False
    for (ar, vb), grp in grouped:
        grp = grp.sort_values("num_buckets")
        if grp["num_buckets"].nunique() < 2:
            continue
        ax1.errorbar(
            grp["num_buckets"], grp["mean_qps"], yerr=grp["stddev_qps"],
            marker=ARITY_MARKERS.get(int(ar), "x"),
            color=_color_for_vb(vb),
            linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
            label=f"arity={int(ar)} vb={int(vb)}",
            linewidth=1.4, capsize=3,
        )
        plotted_panel_a = True
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("num_buckets (log₂)")
    ax1.set_ylabel("Answer throughput (qps)")
    ax1.set_title("(a) m sweep")
    if plotted_panel_a:
        ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B — sweep value_bits at each (arity, num_buckets) where multiple value_bits exist.
    plotted_panel_b = False
    for (ar, nb), grp in df.groupby(["arity", "num_buckets"]):
        grp = grp.sort_values("value_bits")
        if grp["value_bits"].nunique() < 2:
            continue
        ax2.errorbar(
            grp["value_bits"], grp["mean_qps"], yerr=grp["stddev_qps"],
            marker=ARITY_MARKERS.get(int(ar), "x"),
            linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
            label=f"arity={int(ar)} nb={int(nb)}",
            linewidth=1.4, capsize=3,
        )
        plotted_panel_b = True
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("value_bits (log₂)")
    ax2.set_ylabel("Answer throughput (qps)")
    ax2.set_title("(b) w sweep")
    if plotted_panel_b:
        ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("ikpir-server answer throughput (FrodoPIR)", fontsize=13)
    _save(fig, "answer_throughput.png")


# ═════════════════════════════════════════════════════════════════════════════
# 3. incremental_vs_rebuild — the headline incremental crossover
# ═════════════════════════════════════════════════════════════════════════════

def plot_incremental_vs_rebuild():
    """Incremental hint patch (total + per-op) vs full_rebuild.

    Input:  results/ikpir_server_incremental_vs_rebuild.csv
    Output: results/plots/incremental_vs_rebuild.png
    Anchor: no paper analog — this is the headline result that justifies IKPIR.
    Three panels (one per mutation_kind). For each (arity, num_buckets) cell:
      - solid line  = total incremental_ms (N patches summed)
      - dashed line = rebuild_ms (single rebuild, N-independent)
    The crossover N is where the curves intersect.
    """
    df = _load("ikpir_server_incremental_vs_rebuild.csv")
    kinds = ["insert", "update", "delete"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 5), squeeze=False)

    for col, kind in enumerate(kinds):
        ax = axes[0][col]
        sub = df[df["mutation_kind"] == kind]
        if sub.empty:
            ax.set_visible(False)
            continue
        groups = sub.groupby(["arity", "num_buckets"])
        cmap = plt.get_cmap("tab10")
        for idx, ((ar, nb), grp) in enumerate(groups):
            grp = grp.sort_values("n_mutations")
            base_color = cmap(idx % 10)
            ax.plot(
                grp["n_mutations"], grp["incremental_ms"],
                marker="o", linestyle="-", color=base_color,
                label=f"inc arity={int(ar)} nb={int(nb)}", linewidth=1.5,
            )
            ax.plot(
                grp["n_mutations"], grp["rebuild_ms"],
                marker="x", linestyle="--", color=base_color, alpha=0.6,
                label=f"reb arity={int(ar)} nb={int(nb)}", linewidth=1.2,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("n_mutations (log₂)")
        ax.set_ylabel("Time (ms, log)")
        ax.set_title(f"mutation_kind = {kind}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Incremental hint patch vs full_rebuild (FrodoPIR)", fontsize=14)
    _save(fig, "incremental_vs_rebuild.png")


def plot_incremental_per_op():
    """Per-op incremental latency vs rebuild — the apples-to-apples view.

    Input:  results/ikpir_server_incremental_vs_rebuild.csv
    Output: results/plots/incremental_per_op.png
    Per-op = incremental_ms / n_succeeded. The rebuild line is plotted at the
    same N range for visual comparison.
    """
    df = _load("ikpir_server_incremental_vs_rebuild.csv")
    kinds = ["insert", "update", "delete"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 5), squeeze=False)

    for col, kind in enumerate(kinds):
        ax = axes[0][col]
        sub = df[df["mutation_kind"] == kind]
        if sub.empty:
            ax.set_visible(False)
            continue
        groups = sub.groupby(["arity", "num_buckets"])
        cmap = plt.get_cmap("tab10")
        for idx, ((ar, nb), grp) in enumerate(groups):
            grp = grp.sort_values("n_mutations")
            base_color = cmap(idx % 10)
            ax.plot(
                grp["n_mutations"], grp["incremental_per_op_ms"],
                marker="o", linestyle="-", color=base_color,
                label=f"inc/op arity={int(ar)} nb={int(nb)}", linewidth=1.5,
            )
            ax.plot(
                grp["n_mutations"], grp["rebuild_ms"],
                marker="x", linestyle="--", color=base_color, alpha=0.6,
                label=f"reb arity={int(ar)} nb={int(nb)}", linewidth=1.2,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("n_mutations (log₂)")
        ax.set_ylabel("Per-op latency (ms, log)")
        ax.set_title(f"mutation_kind = {kind}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Incremental per-op latency vs full rebuild (FrodoPIR)", fontsize=14)
    _save(fig, "incremental_per_op.png")


# ═════════════════════════════════════════════════════════════════════════════
# 4. end_to_end_fpr — false-positive rate at varying fingerprint widths
# ═════════════════════════════════════════════════════════════════════════════

def plot_end_to_end_fpr():
    """Observed FPR (n_fp / n_queried) vs fingerprint_bits.

    Input:  results/ikpir_end_to_end_fpr.csv
    Output: results/plots/end_to_end_fpr.png
    Anchor: ChalametPIR §6 false-positive analysis.
    Plots empirical FPR alongside the theoretical 2⁻fp_bits curve.
    """
    df = _load("ikpir_end_to_end_fpr.csv")
    fig, ax = plt.subplots(figsize=(9, 6))

    for (ar, nb), grp in df.groupby(["arity", "num_buckets"]):
        grp = grp.sort_values("fingerprint_bits")
        # Use a tiny positive floor for the log scale; mark zero-FPR rows explicitly.
        observed = grp["fpr"].clip(lower=1e-12)
        ax.plot(
            grp["fingerprint_bits"], observed,
            marker=ARITY_MARKERS.get(int(ar), "x"),
            linestyle=ARITY_LINESTYLES.get(int(ar), "-"),
            label=f"arity={int(ar)} nb={int(nb)}",
            linewidth=1.4,
        )

    fp_bits = sorted(df["fingerprint_bits"].unique())
    if fp_bits:
        theoretical = [2.0 ** (-fb) for fb in fp_bits]
        ax.plot(fp_bits, theoretical, linestyle=":", color="black",
                label="theoretical 2⁻ᶠᵇ", linewidth=1.4)

    ax.set_yscale("log")
    ax.set_xlabel("fingerprint_bits")
    ax.set_ylabel("FPR (observed)")
    ax.set_title("End-to-end FPR — ikpir-server")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, "end_to_end_fpr.png")


# ═════════════════════════════════════════════════════════════════════════════
# 5. failure_modes — rejection-path latencies
# ═════════════════════════════════════════════════════════════════════════════

def plot_failure_modes():
    """StaleEpoch vs TableFull rejection-path latencies.

    Input:  results/ikpir_failure_modes.csv
    Output: results/plots/failure_modes.png
    Bar chart per (failure_kind, arity).
    """
    df = _load("ikpir_failure_modes.csv")
    fig, ax = plt.subplots(figsize=(9, 5))

    kinds   = sorted(df["failure_kind"].unique())
    arities = sorted(df["arity"].unique())

    bar_w  = 0.8 / max(len(arities), 1)
    offsets = [(i - (len(arities) - 1) / 2) * bar_w for i in range(len(arities))]

    for i, ar in enumerate(arities):
        means, errs, xs = [], [], []
        for j, kind in enumerate(kinds):
            row = df[(df["failure_kind"] == kind) & (df["arity"] == ar)]
            if row.empty:
                continue
            r = row.iloc[0]
            means.append(r["mean_us"])
            errs.append(r["stddev_us"])
            xs.append(j + offsets[i])
        if not means:
            continue
        ax.bar(xs, means, width=bar_w, yerr=errs,
               label=f"arity={int(ar)}", capsize=3,
               color=plt.get_cmap("tab10")(i % 10), edgecolor="black", linewidth=0.4)

    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds)
    ax.set_ylabel("Rejection latency (µs)")
    ax.set_yscale("log")
    ax.set_title("Failure-mode rejection latency — ikpir-server")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    _save(fig, "failure_modes.png")


# ═════════════════════════════════════════════════════════════════════════════
# 6. wire_sizes — bundle byte-size catalogue
# ═════════════════════════════════════════════════════════════════════════════

def plot_wire_sizes():
    """Stacked / multi-series bandwidth costs by num_buckets and value_bits.

    Input:  results/ikpir_wire_sizes.csv
    Output: results/plots/wire_sizes.png
    Anchor: ChalametPIR Table 1 (query / response columns), Hao-2025 Table 1 Comm.
    """
    df = _load("ikpir_wire_sizes.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — vs num_buckets (m sweep), at each value_bits.
    for (ar, vb), grp in df.groupby(["arity", "value_bits"]):
        grp = grp.sort_values("num_buckets")
        if grp["num_buckets"].nunique() < 2:
            continue
        c = _color_for_vb(vb)
        ax1.plot(grp["num_buckets"], grp["query_bytes"],    marker="o", linestyle="-",  color=c, label=f"query vb={int(vb)}")
        ax1.plot(grp["num_buckets"], grp["response_bytes"], marker="s", linestyle="--", color=c, label=f"resp vb={int(vb)}")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("num_buckets (log₂)")
    ax1.set_ylabel("Wire bytes (log)")
    ax1.set_title("(a) m sweep")
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B — vs value_bits (w sweep), at each num_buckets.
    for (ar, nb), grp in df.groupby(["arity", "num_buckets"]):
        grp = grp.sort_values("value_bits")
        if grp["value_bits"].nunique() < 2:
            continue
        ax2.plot(grp["value_bits"], grp["query_bytes"],    marker="o", linestyle="-",  label=f"query nb={int(nb)}")
        ax2.plot(grp["value_bits"], grp["response_bytes"], marker="s", linestyle="--", label=f"resp nb={int(nb)}")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("value_bits (log₂)")
    ax2.set_ylabel("Wire bytes (log)")
    ax2.set_title("(b) w sweep")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("Wire sizes — ikpir-server (FrodoPIR)", fontsize=13)
    _save(fig, "wire_sizes.png")


def plot_wire_deltas():
    """Hint-delta bandwidth (insert/update/delete) vs num_buckets, value_bits.

    Input:  results/ikpir_wire_sizes.csv
    Output: results/plots/wire_deltas.png
    """
    df = _load("ikpir_wire_sizes.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    for (ar, vb), grp in df.groupby(["arity", "value_bits"]):
        grp = grp.sort_values("num_buckets")
        if grp["num_buckets"].nunique() < 2:
            continue
        c = _color_for_vb(vb)
        for col, marker, ls in (("hint_delta_insert_bytes", "o", "-"),
                                ("hint_delta_update_bytes", "s", "--"),
                                ("hint_delta_delete_bytes", "^", ":")):
            ax.plot(grp["num_buckets"], grp[col], marker=marker, linestyle=ls, color=c,
                    label=f"{col.replace('hint_delta_', '').replace('_bytes', '')} vb={int(vb)}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("num_buckets (log₂)")
    ax.set_ylabel("Hint-delta bytes (log)")
    ax.set_title("Hint-delta wire size by mutation kind — ikpir-server")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, "wire_deltas.png")


# ═════════════════════════════════════════════════════════════════════════════
# 7. setup_to_first_query — cold-start latency breakdown
# ═════════════════════════════════════════════════════════════════════════════

def plot_setup_to_first_query():
    """Stacked per-phase breakdown of setup-to-first-query latency.

    Input:  results/ikpir_setup_to_first_query.csv
    Output: results/plots/setup_to_first_query.png
    For each (mode, num_buckets) point, a stacked bar of the 7 phase columns.
    """
    df = _load("ikpir_setup_to_first_query.csv")
    modes = sorted(df["mode"].unique())
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 6), squeeze=False, sharey=True)

    phase_cols = [
        ("server_setup_ms",      "server_setup"),
        ("client_from_setup_ms", "client_from_setup"),
        ("precompute_b_ms",      "precompute_b"),
        ("precompute_c_ms",      "precompute_c"),
        ("build_query_ms",       "build_query"),
        ("answer_ms",            "answer"),
        ("decode_ms",            "decode"),
    ]
    cmap = plt.get_cmap("tab10")

    for idx, mode in enumerate(modes):
        ax = axes[0][idx]
        sub = df[df["mode"] == mode].sort_values("num_buckets")
        if sub.empty:
            ax.set_visible(False)
            continue
        x_labels = [f"nb={int(nb)}" for nb in sub["num_buckets"]]
        bottom = [0.0] * len(sub)
        for c, (col, label) in enumerate(phase_cols):
            vals = sub[col].fillna(0.0).tolist()
            ax.bar(x_labels, vals, bottom=bottom, label=label, color=cmap(c % 10),
                   edgecolor="black", linewidth=0.4)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(f"mode = {mode}")
        ax.set_ylabel("Latency (ms)")
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        if idx == len(modes) - 1:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Setup-to-first-query latency breakdown — ikpir-server", fontsize=13)
    _save(fig, "setup_to_first_query.png")


# ═════════════════════════════════════════════════════════════════════════════
# 8. steady_state_workload — mixed insert+query op latency
# ═════════════════════════════════════════════════════════════════════════════

def plot_steady_state_workload():
    """Per-op latency split by op kind, vs num_buckets.

    Input:  results/ikpir_steady_state_workload.csv
    Output: results/plots/steady_state_workload.png
    Three lines: mean_insert_ms (with apply_delta), mean_query_ms (full round-trip),
    mean_ms_per_op (workload average).
    """
    df = _load("ikpir_steady_state_workload.csv")
    fig, ax = plt.subplots(figsize=(9, 6))

    df = df.sort_values("num_buckets")
    ax.plot(df["num_buckets"], df["mean_insert_ms"], marker="o", linestyle="-",
            color="#1f77b4", label="mean_insert_ms (insert+apply_delta)", linewidth=1.4)
    ax.plot(df["num_buckets"], df["mean_query_ms"], marker="s", linestyle="-",
            color="#2ca02c", label="mean_query_ms (build_query→answer→decode)", linewidth=1.4)
    ax.plot(df["num_buckets"], df["mean_ms_per_op"], marker="^", linestyle="--",
            color="#d62728", label="mean_ms_per_op (workload average)", linewidth=1.4)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("num_buckets (log₂)")
    ax.set_ylabel("Per-op latency (ms, log)")
    ax.set_title("Steady-state workload — ikpir-server")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, "steady_state_workload.png")


# ═════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═════════════════════════════════════════════════════════════════════════════

PLOT_FUNCTIONS = {
    "setup_latency":           plot_setup_latency,
    "answer_throughput":       plot_answer_throughput,
    "incremental_vs_rebuild":  plot_incremental_vs_rebuild,
    "incremental_per_op":      plot_incremental_per_op,
    "end_to_end_fpr":          plot_end_to_end_fpr,
    "failure_modes":           plot_failure_modes,
    "wire_sizes":              plot_wire_sizes,
    "wire_deltas":             plot_wire_deltas,
    "setup_to_first_query":    plot_setup_to_first_query,
    "steady_state_workload":   plot_steady_state_workload,
}

CSV_FOR_PLOT = {
    "setup_latency":           "ikpir_server_setup_latency.csv",
    "answer_throughput":       "ikpir_server_answer_throughput.csv",
    "incremental_vs_rebuild":  "ikpir_server_incremental_vs_rebuild.csv",
    "incremental_per_op":      "ikpir_server_incremental_vs_rebuild.csv",
    "end_to_end_fpr":          "ikpir_end_to_end_fpr.csv",
    "failure_modes":           "ikpir_failure_modes.csv",
    "wire_sizes":              "ikpir_wire_sizes.csv",
    "wire_deltas":             "ikpir_wire_sizes.csv",
    "setup_to_first_query":    "ikpir_setup_to_first_query.csv",
    "steady_state_workload":   "ikpir_steady_state_workload.csv",
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
        description="Generate plots from ikpir-server benchmark CSV results.",
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
