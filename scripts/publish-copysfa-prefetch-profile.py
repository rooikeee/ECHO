#!/usr/bin/env python3
"""Publish isolated LIM/CopySFA traces and operator-only measurements."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    args = parser.parse_args()
    profile_root = args.profile_root.resolve()
    source_files = (
        subprocess.check_output(["git", "diff", "--name-only", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_files
        if name and (ROOT / name).is_file()
    }
    owner = "vllm-ascend/csrc/attention/fused_scatter_copy_sparse_flash_attention/op_kernel/prefetch_copy_owners.h"
    sources[owner] = hashlib.sha256((ROOT / owner).read_bytes()).hexdigest()
    binaries = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "vllm-ascend/vllm_ascend/_cann_ops_custom").rglob("*.o")
        if "fused_lightning_indexer_manage" in str(path)
        or "fused_scatter_copy_sparse_flash_attention" in str(path)
    }
    binding = (
        ROOT / "vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so"
    )
    binaries[str(binding.relative_to(ROOT))] = hashlib.sha256(
        binding.read_bytes()
    ).hexdigest()
    rows, validations = [], {}
    for directory, scenario in [
        ("final_miss400", "miss400"),
        ("final_all_hit", "all_hit"),
    ]:
        data = json.loads((profile_root / directory / "result.json").read_text())
        assert data["status"] == "completed_single_operator_profiles"
        assert data["profile_scope"] == "operators" and data["physical_device"] == "7"
        for metric in data["metrics"]:
            mode = metric["mode"]
            if mode == "context-budget-0":
                continue
            label = "baseline" if mode == "baseline" else "prefetch"
            validations[f"{scenario}_{label}"] = {
                "attention_reference": metric["independent_attention_reference"],
                "dynamic_graph_checks": metric.get("dynamic_graph_checks", []),
                "staging_source_check": metric.get("staging_source_check"),
                "actual_prefetched_tokens": sum(metric["prefetched_per_query"]),
            }
            for op, timing in metric["operators"].items():
                target = profile_root / (
                    f"{label}_{op}"
                    if scenario == "miss400"
                    else f"all_hit/{label}_{op}"
                )
                target.mkdir(parents=True, exist_ok=True)
                source = profile_root / directory / "batch-24" / mode / op
                capture = next(source.glob("*_ascend_pt"))
                output = capture / "ASCEND_PROFILER_OUTPUT"
                trace = output / "trace_view.json"
                assert trace.exists()
                shutil.copy2(trace, target / "trace.json")
                for filename in (
                    "kernel_details.csv",
                    "operator_details.csv",
                    "ascend_pytorch_profiler.db",
                ):
                    shutil.copy2(output / filename, target / filename)
                kernels = list(csv.DictReader((output / "kernel_details.csv").open()))
                names = {row["Name"] for row in kernels}
                forbidden = ("_mtp_", "Cumsum", "ScatterElements")
                assert not any(
                    token in name for name in names for token in forbidden
                ), names
                wanted = (
                    "FusedLightningIndexerManage"
                    if op == "lim"
                    else "FusedScatterCopySparseFlashAttention"
                )
                assert sum(row["Name"] == wanted for row in kernels) >= 10
                raw = next(capture.glob("PROF_*"))
                export_root = target / "msprof"
                if not export_root.exists():
                    export_root.mkdir()
                    shutil.copytree(raw, export_root / raw.name)
                    with (target / "msprof-export.log").open("w") as log:
                        subprocess.run(
                            [
                                "msprof",
                                "--export=on",
                                "--type=text",
                                "--output=" + str(export_root),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                shutil.copy2(
                    next(export_root.rglob("msprof_*.db")), target / "msprof.db"
                )
                shutil.copy2(
                    next(export_root.rglob("msprof_*.json")), target / "msprof.json"
                )
                record = {
                    "scenario": scenario,
                    "variant": label,
                    "operator": op,
                    "kernel_profile_mean_us": timing["profile"]["Duration(us)"],
                    "event_median_us": timing["event_median_us"],
                    "profile_calls": timing["profile"]["profile_kernel_calls"],
                    "trace_directory": str(target.relative_to(profile_root)),
                    "scope": timing["scope"],
                }
                rows.append(record)
                (target / "summary.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
    result = {
        "status": "completed",
        "scope": "single fused operator profiles; no end-to-end/model speedup claim",
        "batch": 24,
        "mtp": 3,
        "context_tokens": 143360,
        "hot_tokens": 8192,
        "dtype": "bfloat16",
        "physical_device": 7,
        "rows": rows,
        "validation": validations,
        "source_sha256": sources,
        "binary_sha256": binaries,
        "discarded_performance_run": "all_hit_140k: device6 concurrently occupied by another process",
        "removed_consumer_kernels": 20,
    }
    (profile_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (profile_root / "comparison.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
