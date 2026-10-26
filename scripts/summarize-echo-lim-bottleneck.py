#!/usr/bin/env python3
"""Summarize graph PMU and per-core msprof op diagnostics for LIM."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def stats(values):
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--all-hit", type=Path, required=True)
    parser.add_argument("--op", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph = json.loads((args.graph / "result.json").read_text())
    control = json.loads((args.all_hit / "result.json").read_text())
    assert graph["status"] == control["status"] == "completed"
    assert graph["union_misses_per_request"] == control["union_misses_per_request"] == [400] * 16
    # Invocation 0 constructs residency using the original prefix cache.
    # Invocation 1 is the first baseline pass with exactly 400 union misses.
    paths = list(args.op.glob("OPPROF*/*/1/PipeUtilization*.csv"))
    if len(paths) != 1:
        raise ValueError("Expected one invocation-1 per-core profile")
    with paths[0].open(newline="") as source:
        cores = list(csv.DictReader(source))
    assert len(cores) == 72
    groups = {}
    for name in ("cube0", "vector0", "vector1"):
        rows = [r for r in cores if r["sub_block_id"] == name]
        assert len(rows) == 24
        groups[name] = {}
        for key in rows[0]:
            if key and key.endswith("(us)") and rows[0][key] != "NA":
                groups[name][key] = stats([float(row[key]) for row in rows])
    owner = [float(r["aiv_time(us)"]) for r in cores
             if r["sub_block_id"] == "vector0" and int(r["block_id"]) < 16]
    nonowner = [float(r["aiv_time(us)"]) for r in cores
                if r["sub_block_id"] == "vector0" and int(r["block_id"]) >= 16]
    rows = []
    for record in graph["results"]:
        rows.append({"mode": record["mode"], "event_mean_us": record["event_mean_us"],
                     "profiles": {name: {"calls": profile["calls"], "metrics": profile["metrics"]}
                                  for name, profile in record["profiles"].items()}})
    control_row = control["results"][0]
    all_hit = control_row["all_hit_control"]
    before = control_row["profiles"]["PipeUtilization"]["metrics"]
    after = all_hit["profile"]["metrics"]
    result = {
        "graph_source": str(args.graph), "all_hit_source": str(args.all_hit),
        "op_pipe_source": str(paths[0]), "mtp": 3, "batch": 16,
        "graph": rows, "per_core": groups,
        "union_owner_aiv_us": stats(owner), "other_even_aiv_us": stats(nonowner),
        "all_hit_control": {
            "miss400": before, "all_hit": after,
            "duration_reduction_us": before["Duration(us)"]["mean"] - after["Duration(us)"]["mean"],
            "event_miss400_us": control_row["event_mean_us"],
            "event_all_hit_us": all_hit["event_mean_us"],
        },
        "interpretation": "AIV-side limited at this shape; no evidence of saturated Cube arithmetic; "
                          "miss management causes an AIV tail; prefetch adds scalar/synchronization cost",
        "limits": ["PMU pipe times overlap; per-core means are not kernel elapsed time",
                   "msprof op per-core diagnostic uses eager; accepted latency uses actual graph replay",
                   "all-hit delta includes cache-dependent paths, not a directly instrumented union phase",
                   "ID0 wait counter agrees with source handshake; do not add it to pipeline busy time",
                   "synthetic fixed query and controlled residency; target verification only"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    x = np.arange(24)
    colors = ["#2878a0", "#c74f43", "#3a936b"]
    for i, (name, label) in enumerate((("cube0", "AIC"), ("vector0", "Even AIV"), ("vector1", "Odd AIV"))):
        selected = sorted((r for r in cores if r["sub_block_id"] == name), key=lambda r: int(r["block_id"]))
        key = "aic_time(us)" if name == "cube0" else "aiv_time(us)"
        axes[0].bar(x + (i - 1) * .26, [float(r[key]) for r in selected], .25, label=label, color=colors[i])
    axes[0].axvline(15.5, color="#444444", linestyle="--", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xlabel("AIC block ID (one AIC plus two AIVs)")
    axes[0].set_ylabel("Per-core duration (us)")
    axes[0].set_title("Original LIM: even AIVs on blocks 0-15 own request-union management")
    axes[0].legend(loc="lower right", ncol=3)
    axes[0].set_ylim(0, 540)
    axes[0].grid(axis="y", alpha=.2)
    labels = ["Original LIM", "Context, budget 0", "Prefetch 64/query"]
    keys = ["Duration(us)", "aic_mac_time(us)", "aiv_vec_time(us)", "aiv_scalar_time(us)"]
    names = ["Kernel duration", "AIC MAC", "AIV Vector", "AIV Scalar"]
    x = np.arange(3)
    for i, (key, label, color) in enumerate(zip(keys, names, ["#545454", *colors])):
        values = [r["profiles"]["PipeUtilization"]["metrics"][key]["mean"] for r in rows]
        bars = axes[1].bar(x + (i - 1.5) * .2, values, .19, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.0f", padding=2, fontsize=9)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Time (us)")
    axes[1].set_title("Graph replay: prefetch increases elapsed and scalar time; arithmetic work stays stable")
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].grid(axis="y", alpha=.2)
    axes[1].set_ylim(0, 2500)
    fig.suptitle("LIM bottleneck profile | B16, MTP3, context 40960, union misses 400", fontsize=14)
    fig.savefig(args.output / "lim-bottleneck.png", dpi=180)
    fig.savefig(args.output / "lim-bottleneck.svg")
    print(json.dumps({"summary": str(args.output / "summary.json"),
                      "all_hit_reduction_us": result["all_hit_control"]["duration_reduction_us"],
                      "union_owner_aiv_us": result["union_owner_aiv_us"]}, indent=2))


if __name__ == "__main__":
    main()
