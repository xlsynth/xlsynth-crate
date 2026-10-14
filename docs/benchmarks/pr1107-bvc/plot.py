# SPDX-License-Identifier: Apache-2.0
"""Render PR1107's full-cohort raw-g8r comparison with Matplotlib."""

import argparse
import collections
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


COLORS = {"win": "#007c78", "trade": "#aa6900", "loss": "#bd3044", "tie": "#78838d"}
MARKERS = {"win": "o", "trade": "^", "loss": "X", "tie": "D"}
EPS = 1e-8


def classify(row):
    """Use exact integer area and the recorded absolute LE tolerance."""
    da = row["candidate_and_nodes"] - row["baseline_and_nodes"]
    dl = row["candidate_graph_le"] - row["baseline_graph_le"]
    if da == 0 and abs(dl) < EPS:
        return "tie"
    if da <= 0 and dl <= EPS:
        return "win"
    if da >= 0 and dl >= -EPS:
        return "loss"
    return "trade"


def load_rows(directory):
    """Read the entire population, retaining unrounded measured values."""
    with gzip.open(directory / "measurements.csv.gz", "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len({row["sample_id"] for row in rows}) == len(rows)
    for row in rows:
        for key in ("baseline_and_nodes", "candidate_and_nodes"):
            row[key] = int(row[key])
        for key in ("baseline_graph_le", "candidate_graph_le"):
            row[key] = float(row[key])
        row["byte_identical"] = row["byte_identical"] == "true"
        row["class"] = classify(row)
        if not row["byte_identical"]:
            assert row["baseline_and_nodes"] > 0 and row["baseline_graph_le"] > 0
            row["and_pct"] = 100 * (
                row["candidate_and_nodes"] / row["baseline_and_nodes"] - 1
            )
            row["le_pct"] = 100 * (
                row["candidate_graph_le"] / row["baseline_graph_le"] - 1
            )
        else:
            assert row["class"] == "tie"
    return rows


def short_name(sample_id):
    if sample_id.startswith("mffc-"):
        return "mffc " + sample_id.split("-")[1][:8]
    if sample_id.startswith("k3-cone-"):
        return "k3 " + sample_id.split("-")[2][:8]
    return (
        "fixed-point aligned" if sample_id.startswith("fixed-point") else "hfloat16 fma"
    )


def render(directory, output):
    rows = load_rows(directory)
    meta = json.loads((directory / "provenance.json").read_text())
    counts = collections.Counter(row["class"] for row in rows)
    assert dict(counts) == meta["classification"]
    assert len(rows) == meta["population"]
    affected = [row for row in rows if not row["byte_identical"]]
    assert len(affected) == meta["changed_graphs"]
    with (directory / "affected.csv").open(newline="") as stream:
        detail = {row["sample_id"]: row for row in csv.DictReader(stream)}
    assert set(detail) == {row["sample_id"] for row in affected}
    for row in affected:
        row["short_id"] = detail[row["sample_id"]]["short_id"]
    changed = sorted(
        (row for row in affected if row["class"] != "tie"),
        key=lambda row: (
            "win trade loss".split().index(row["class"]),
            row["sample_id"],
        ),
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263440",
            "text.color": "#263440",
            "axes.edgecolor": "#aeb7bf",
        }
    )
    fig = plt.figure(figsize=(13.8, 8.2), facecolor="white")
    fig.text(
        0.045,
        0.947,
        "BVC: arithmetic-row scheduling changes few samples",
        size=19,
        weight="bold",
    )
    fig.text(
        0.045,
        0.901,
        f"{len(rows):,} samples  ·  {len(rows)-len(affected):,} byte-identical AIG files  ·  "
        f"{len(affected)} byte-different AIG files ({100*len(affected)/len(rows):.3f}%)",
        size=12,
    )
    fig.text(
        0.045,
        0.861,
        "24 wiring changes  ·  18 metric changes  ·  13 serialization-only differences",
        size=11.5,
    )
    fig.text(
        0.045,
        0.826,
        "7 wins  ·  10 area–delay tradeoffs  ·  1 loss  ·  19 byte-different files with tied metrics",
        size=11.5,
    )
    ax = fig.add_axes((0.07, 0.225, 0.51, 0.55))
    ax.set(
        xlim=(-11.1, 4.2),
        ylim=(-0.92, 0.24),
        xlabel="Change in live AND count (%)",
        ylabel="Change in actual graph LE delay (%)",
    )
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4])
    ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+g}" if x else "0"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1f}" if x else "0"))
    ax.grid(color="#e7ebef", linewidth=0.8, zorder=0)
    ax.axhline(0, color="#6d7882", linewidth=1)
    ax.axvline(0, color="#6d7882", linewidth=1)
    ax.set_title("Negative = improvement on either axis", loc="left", size=11, pad=16)
    grouped = collections.defaultdict(list)
    for row in affected:
        grouped[(row["and_pct"], row["le_pct"], row["class"])].append(row["short_id"])
        ax.scatter(
            row["and_pct"],
            row["le_pct"],
            s=53,
            color=COLORS[row["class"]],
            marker=MARKERS[row["class"]],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
    for (x, y, category), ids in grouped.items():
        if category == "tie" or category == "loss" or (-2 < x < 0 and y == 0):
            continue
        if category == "trade":
            positions = {
                "T1": (0.15, -0.79),
                "T8": (0.30, -0.56),
                "T10": (1.55, -0.49),
                "T3": (3.1, -0.59),
                "T2": (3.0, -0.86),
            }
            ax.annotate(
                ", ".join(ids),
                (x, y),
                xytext=positions[ids[0]],
                ha="center",
                fontsize=9,
                color=COLORS[category],
                arrowprops={"arrowstyle": "-", "color": COLORS[category], "lw": 0.7},
            )
            continue
        offset = (0, 13)
        ax.annotate(
            ", ".join(ids),
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS[category],
            arrowprops={"arrowstyle": "-", "color": COLORS[category], "lw": 0.7},
        )
    ax.annotate(
        "W5–W7",
        (-1.67, 0),
        xytext=(-3.9, 0.12),
        ha="center",
        size=9,
        color=COLORS["win"],
        arrowprops={"arrowstyle": "-", "color": COLORS["win"]},
    )
    ax.annotate(
        "Near origin: see zoom",
        (0, 0),
        xytext=(1.75, 0.13),
        ha="center",
        size=9,
        arrowprops={"arrowstyle": "-", "color": "#6d7882"},
    )
    zoom = ax.inset_axes((0.08, 0.13, 0.38, 0.24))
    zoom.set(xlim=(-0.25, 0.085), ylim=(-0.04, 0.04))
    zoom.axhline(0, color="#9ca8b2", lw=0.8)
    zoom.set_xticks([-0.2, 0, 0.04], labels=["−0.20%", "0", "+0.04%"])
    zoom.set_yticks([0], labels=["0%"])
    zoom.tick_params(labelsize=8)
    zoom.set_title("Origin zoom · same axes", size=9, loc="left", pad=7)
    for row in affected:
        if -0.25 <= row["and_pct"] <= 0.085 and abs(row["le_pct"]) < EPS:
            zoom.scatter(
                row["and_pct"],
                row["le_pct"],
                s=48,
                color=COLORS[row["class"]],
                marker=MARKERS[row["class"]],
                zorder=4,
            )
    for x, label, color in [
        (-0.18552875695732746, "W2", COLORS["win"]),
        (0, "Z1–Z19", COLORS["tie"]),
        (100 * 2 / 5238, "L1", COLORS["loss"]),
    ]:
        zoom.annotate(
            label,
            (x, 0),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            size=8,
            color=color,
        )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLORS[k],
                marker=MARKERS[k],
                linestyle="",
                label=v,
                markersize=7,
            )
            for k, v in [
                ("win", "Win"),
                ("trade", "Tradeoff"),
                ("loss", "Loss"),
                ("tie", "Metric tie"),
            ]
        ],
        loc="upper left",
        bbox_to_anchor=(0.065, 0.163),
        ncol=4,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1,
        fontsize=9.5,
    )
    table = fig.add_axes((0.64, 0.225, 0.325, 0.56))
    table.axis("off")
    table.text(0, 1.02, "Every metric-changing sample", weight="bold", size=11)
    for x, label, ha in [
        (0, "ID / sample", "left"),
        (0.77, "AND Δ%", "right"),
        (1, "LE Δ%", "right"),
    ]:
        table.text(x, 0.95, label, size=10, weight="bold", ha=ha)
    for i, row in enumerate(changed):
        y = 0.905 - i * 0.047
        color = COLORS[row["class"]]
        table.text(0, y, row["short_id"], color=color, weight="bold", size=9.5)
        table.text(0.10, y, short_name(row["sample_id"]), size=9.5)
        for x, key in [(0.77, "and_pct"), (1, "le_pct")]:
            value = row[key]
            table.text(
                x,
                y,
                "0.00" if abs(value) < EPS else f"{value:+.2f}",
                ha="right",
                size=9.5,
                color=color,
            )
    fig.text(
        0.64,
        0.19,
        "L1: 5,238 → 5,240 ANDs; LE 703.09 → 703.09",
        size=10,
        color=COLORS["loss"],
        weight="bold",
    )
    fig.text(
        0.045,
        0.118,
        "All 37 byte-different AIG files are represented; exact overlaps share labels. "
        "Full IDs and known origins: affected.csv.",
        size=9.5,
    )
    fig.text(
        0.045,
        0.091,
        "Z1–Z19: 13 serialization-only differences and 6 wiring changes with tied metrics. "
        "All lie at (0, 0).",
        size=9.5,
    )
    fig.text(
        0.045,
        0.065,
        "Raw g8r · folding / hashing / DCE on · ABC / FRAIG / cut rewriting / reassociation off · "
        "Brent–Kung adders · graph LE β₁=1, β₂=0",
        size=9.5,
    )
    fig.text(
        0.045,
        0.036,
        "Base 2954ad29 → PR code 67dcf1b1 · additive-row effects only; no eligible multiply fusion in this cohort. "
        "Δ% = 100 × (PR/base − 1).",
        size=9.5,
    )
    fig.savefig(
        output,
        dpi=180,
        facecolor="white",
        metadata={"Software": "Matplotlib " + matplotlib.__version__},
    )
    plt.close(fig)
    print(
        json.dumps(
            {
                "population": len(rows),
                "classification": dict(counts),
                "changed_graphs": len(affected),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    render(args.data_dir, args.output or args.data_dir / "bvc-impact.png")
