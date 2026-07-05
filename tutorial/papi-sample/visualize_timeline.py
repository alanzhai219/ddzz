#!/usr/bin/env python3
"""
visualize_timeline.py — Render PAPI timeline trace as Gantt chart with metrics.

Usage:
    python3 visualize_timeline.py [trace.json] [output.png]
    python3 visualize_timeline.py trace.json          # saves trace_timeline.png
    python3 visualize_timeline.py trace.json out.png  # custom output name

Produces a 2-panel figure:
  Top:    Gantt chart — function duration on timeline,
          bar color = IPC (green=high, red=low).
          Each bar annotated with key metrics.
  Bottom: Summary table — all metrics per function.
"""

import json
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trace(path):
    with open(path) as f:
        return json.load(f)

def format_num(n):
    """Human-readable number."""
    if abs(n) >= 1_000_000_000:
        return f"{n/1e9:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1e3:.2f}K"
    return str(n)

# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render(data, out_path):
    traces = data["traces"]
    event_names = data["event_names"]
    names = [t["name"] for t in traces]

    # ---- Figure layout ----
    fig = plt.figure(figsize=(16, max(8, len(traces) * 1.2 + 4)))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)

    ax_gantt = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # ---- Color mapping (IPC) ----
    ipc_vals = [t["ipc"] for t in traces]
    ipc_min = min(ipc_vals)
    ipc_max = max(ipc_vals)
    if ipc_max == ipc_min:
        ipc_max = ipc_min + 1.0

    colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
    cmap = LinearSegmentedColormap.from_list("ipc_cmap", colors)
    norm = Normalize(ipc_min, ipc_max)

    # ---- Gantt Chart ----
    bar_height = 0.6
    for i, t in enumerate(traces):
        y = len(traces) - 1 - i
        start = t["start_us"] / 1_000_000.0  # convert to seconds
        dur   = t["dur_us"]   / 1_000_000.0

        color = cmap(norm(t["ipc"]))
        ax_gantt.barh(y, dur, bar_height, left=start,
                      color=color, edgecolor="#333333", linewidth=0.8, zorder=3)

        # Annotate bar with key metrics
        label = (
            f"{t['name']}\n"
            f"IPC={t['ipc']:.2f} | "
            f"Cycles={format_num(t['cycles'])} | "
            f"L1miss={format_num(t.get('PAPI_L1_DCM', 0))}"
        )
        ax_gantt.text(start + max(dur * 0.01, 0.005), y,
                      label, va="center", fontsize=7,
                      color="black", fontweight="bold")

    ax_gantt.set_yticks(range(len(traces)))
    ax_gantt.set_yticklabels([""] * len(traces))
    ax_gantt.set_xlabel("Time (seconds)", fontsize=10)
    ax_gantt.set_title("PAPI Function Timeline Profiler",
                       fontsize=14, fontweight="bold")
    ax_gantt.grid(axis="x", alpha=0.3, linestyle="--")
    ax_gantt.set_xlim(left=0)

    # Colorbar for IPC
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_gantt, aspect=40, pad=0.02)
    cbar.set_label("IPC (higher = better)", fontsize=9)

    # ---- Summary Table ----
    ax_table.axis("off")
    ax_table.set_title("Performance Metrics Summary", fontsize=12,
                       fontweight="bold", pad=10)

    # Build table data
    col_labels = ["Function", "Time(s)", "Cycles", "Insns", "IPC",
                  "L1D Miss", "L2 Miss", "TLB Miss", "BranchMisp"]
    cell_data = []
    for t in traces:
        cell_data.append([
            t["name"],
            f"{t['dur_us']/1e6:.4f}",
            format_num(t["cycles"]),
            format_num(t["insns"]),
            f"{t['ipc']:.4f}",
            format_num(t.get("PAPI_L1_DCM", 0)),
            format_num(t.get("PAPI_L2_TCM", 0)),
            format_num(t.get("PAPI_TLB_DM", 0)),
            format_num(t.get("PAPI_BR_MSP", 0)),
        ])

    table = ax_table.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    # Color header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(len(cell_data)):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    # ---- Save ----
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved timeline chart to {out_path}")
    print(f"  Functions: {len(traces)}")
    print(f"  Total time: {traces[-1]['start_us']/1e6 + traces[-1]['dur_us']/1e6:.4f}s")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trace_file = sys.argv[1] if len(sys.argv) > 1 else "trace.json"
    out_file   = sys.argv[2] if len(sys.argv) > 2 else "trace_timeline.png"

    if not os.path.exists(trace_file):
        print(f"Error: {trace_file} not found.", file=sys.stderr)
        print("Usage: python3 visualize_timeline.py [trace.json] [output.png]",
              file=sys.stderr)
        sys.exit(1)

    data = load_trace(trace_file)
    render(data, out_file)