#!/usr/bin/env python3
"""Print the top costly kernels from a torch Chrome trace."""

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the most expensive kernels in a torch Chrome trace."
    )
    parser.add_argument(
        "trace_path",
        type=Path,
        help="Path to a .trace.json or .trace.json.gz file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of kernels to print. Default: 50.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="kernel",
        help="Trace category to aggregate. Default: kernel.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("total", "avg", "max", "calls"),
        default="total",
        help="Metric used to rank rows. Default: total.",
    )
    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help="Only keep events whose name contains this substring.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Default: topXX-kernels.csv where XX is --top.",
    )
    return parser.parse_args()


def load_trace(trace_path: Path) -> List[Dict]:
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("traceEvents", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported trace format in {trace_path}")


def iter_duration_events(
    events: Iterable[Dict], category: str, contains: str | None
) -> Iterable[Tuple[str, float]]:
    contains_lower = contains.lower() if contains else None
    for event in events:
        if event.get("cat") != category or event.get("ph") != "X":
            continue
        name = event.get("name")
        dur = event.get("dur")
        if not name or dur is None:
            continue
        if contains_lower and contains_lower not in name.lower():
            continue
        yield name, float(dur)


def summarize(events: Iterable[Tuple[str, float]]) -> Tuple[List[Dict], float]:
    total_by_name: Counter[str] = Counter()
    calls_by_name: Counter[str] = Counter()
    max_by_name: Dict[str, float] = {}

    total_duration_us = 0.0
    for name, dur_us in events:
        total_by_name[name] += dur_us
        calls_by_name[name] += 1
        max_by_name[name] = max(max_by_name.get(name, 0.0), dur_us)
        total_duration_us += dur_us

    rows = []
    for name, total_us in total_by_name.items():
        calls = calls_by_name[name]
        rows.append(
            {
                "name": name,
                "total_us": total_us,
                "calls": calls,
                "avg_us": total_us / calls,
                "max_us": max_by_name[name],
            }
        )
    return rows, total_duration_us


def sort_rows(rows: List[Dict], sort_by: str) -> List[Dict]:
    key_map = {
        "total": lambda row: (row["total_us"], row["calls"]),
        "avg": lambda row: (row["avg_us"], row["total_us"]),
        "max": lambda row: (row["max_us"], row["total_us"]),
        "calls": lambda row: (row["calls"], row["total_us"]),
    }
    return sorted(rows, key=key_map[sort_by], reverse=True)


def print_rows(
    rows: List[Dict], total_duration_us: float, trace_path: Path, category: str, top_n: int
) -> None:
    print(f"trace: {trace_path}")
    print(f"category: {category}")
    print(f"total events: {len(rows)} unique names")
    print(f"total duration: {total_duration_us / 1000:.3f} ms")
    print()
    header = (
        f"{'rank':>4}  {'total_ms':>11}  {'share':>7}  {'calls':>8}  "
        f"{'avg_us':>10}  {'max_us':>10}  name"
    )
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(rows[:top_n], start=1):
        share = 100.0 * row["total_us"] / total_duration_us if total_duration_us else 0.0
        print(
            f"{idx:4d}  "
            f"{row['total_us'] / 1000:11.3f}  "
            f"{share:6.2f}%  "
            f"{row['calls']:8d}  "
            f"{row['avg_us']:10.3f}  "
            f"{row['max_us']:10.3f}  "
            f"{row['name']}"
        )


def write_csv(
    rows: List[Dict], total_duration_us: float, output_path: Path, top_n: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "total_ms", "share_pct", "calls", "avg_us", "max_us", "name"])
        for idx, row in enumerate(rows[:top_n], start=1):
            share = 100.0 * row["total_us"] / total_duration_us if total_duration_us else 0.0
            writer.writerow(
                [
                    idx,
                    f"{row['total_us'] / 1000:.3f}",
                    f"{share:.2f}",
                    row["calls"],
                    f"{row['avg_us']:.3f}",
                    f"{row['max_us']:.3f}",
                    row["name"],
                ]
            )


def main() -> None:
    args = parse_args()
    events = load_trace(args.trace_path)
    rows, total_duration_us = summarize(
        iter_duration_events(events, args.category, args.contains)
    )
    rows = sort_rows(rows, args.sort_by)
    print_rows(rows, total_duration_us, args.trace_path, args.category, args.top)
    output_path = args.output or Path(
        f"top{args.top}-kernels-{datetime.now().strftime('%m%d-%H%M')}.csv"
    )
    write_csv(rows, total_duration_us, output_path, args.top)
    print()
    print(f"csv: {output_path}")


if __name__ == "__main__":
    main()
