#!/usr/bin/env python3
"""Export concise measured results without copying tensors or profiler traces."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"measured_date": "2026-10-25", "scope": "single_device_operator_graph_only",
              "end_to_end_model_performance_validated": False,
              "model_task_accuracy_validated": False, "prefetch_default_enabled": False,
              "mtp": 3, "query_rows_per_request": 4, "measurements": [], "validation": []}
    experiments = [
        ("lim_pristine", "build/lim-copysfa-optimization/lim-pristine-glm32/result.json", "lim"),
        ("lim_retained", "build/lim-copysfa-optimization/lim-restored-glm32/result.json", "lim"),
        ("copysfa_same_query", "build/lim-copysfa-optimization/copysfa-head4-preparation.json", "heads"),
        ("chain_pristine", "build/lim-copysfa-optimization/chain-pristine-pr-glm64k/result.json", "chain"),
        ("chain_retained", "build/lim-copysfa-optimization/chain-head4-glm64k/result.json", "chain"),
        ("prefetch_comparison", "build/profiling/echo-prefetch-b16-mtp3-union400-20261025-r1/result.json", "chain"),
    ]
    for name, relative, kind in experiments:
        raw = (ROOT / relative).read_bytes()
        result = json.loads(raw)
        if not result["status"].startswith(("completed", "passed")):
            raise RuntimeError(f"Incomplete measurement: {relative}")
        entry = {"name": name, "source_artifact": relative, "source_sha256": hashlib.sha256(raw).hexdigest(),
                 "source_status": result["status"]}
        if kind == "lim":
            row = result["results"][0]
            entry.update(context=result["context"], hot_tokens=result["hot_tokens"], batch=result["batch"],
                         indexer_heads=result["indexer_heads"], union_misses_per_request=400,
                         event_mean_us=row["event_mean_us"], replays=row["replays"],
                         kernel=row["profiles"]["PipeUtilization"]["metrics"]["Duration(us)"],
                         correctness=row["correctness"], persistent_all_hit_replay=row["persistent_all_hit_replay"])
        elif kind == "heads":
            entry.update(batch=16, context=64000, hot_tokens=12288,
                         note="Four query rows share the same misses; kernel-only and preparation-inclusive timings differ",
                         cases=[row for row in result["cases"] if "native4" in row])
        else:
            entry.update(context=result["context_tokens"], hot_tokens=result["hot_tokens_per_request"],
                         indexer_heads=result["indexer_heads"], actual_execution_devices=result["actual_execution_devices"],
                         modes=[{key: row[key] for key in ("mode", "batch", "mtp", "replays", "event_mean_us", "stages",
                                                          "original_union_misses_per_request", "correctness")}
                                for row in result["metrics"]])
            if name.startswith("chain_"):
                entry["comparison_limit"] = "Matched shapes but native4 and padded8 use different random Attention query layout"
        report["measurements"].append(entry)
    for name, relative in [
        ("lim_occurrence_graph", "build/lim-copysfa-optimization/lim-occurrence-graph.json"),
        ("copysfa_four_heads_graph", "build/lim-copysfa-optimization/copysfa-head4-boundaries.json"),
    ]:
        raw = (ROOT / relative).read_bytes()
        result = json.loads(raw)
        if result["status"] != "passed":
            raise RuntimeError(f"Validation failed: {relative}")
        report["validation"].append({"name": name, "source_artifact": relative,
                                      "source_sha256": hashlib.sha256(raw).hexdigest(),
                                      "status": result["status"], "case_count": len(result["cases"]),
                                      "counts": [row.get("occurrences", row.get("misses_per_row"))
                                                 for row in result["cases"]]})
    report["interpretation_limits"] = [
        "LIM metadata and KV exact comparisons use rtol=0 and atol=0.",
        "Attention comparisons use rtol=0.02 and atol=0.002; no bitwise or model-quality claim.",
        "Profiler pipe times overlap and stage timings must not be added to independent event timings.",
        "All workloads use synthetic fixed scoring queries and controlled residency, not real model traces.",
        "No throughput doubling or full-model superiority over the PR has been established.",
        "Raw traces remain machine-local; hashes identify the source summaries but do not replace reruns.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(experiments)} measurements to {args.output}")


if __name__ == "__main__":
    main()
