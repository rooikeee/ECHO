#!/usr/bin/env python3
"""Frozen 910 LIM/ECHO evaluator used by the experimental ForgeKernel adapter."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NATIVE = (
    ROOT
    / "vllm-ascend/csrc/attention/fused_lightning_indexer_manage/op_kernel/echo_indexer_prefetch.h"
)
BUILD_STATE = ROOT / "build/forge-lim/native-source.sha256"


def run(command, log, timeout):
    with log.open("w") as output:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Command failed rc={result.returncode}: {log}")


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((NATIVE.parent).glob("*.h")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--profile-repeats", type=int, default=0)
    parser.add_argument("--attention-reference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["ECHO_NPU_DEVICE"] = str(args.device)
    lock = Path("/tmp") / f"echo-forge-npu-{args.device}.lock"
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        actual = source_hash()
        known = BUILD_STATE.read_text().strip() if BUILD_STATE.exists() else ""
        if known != actual:
            # Header changes need explicit cache invalidation. The build script
            # configures this isolated worktree; never rebuild the original ECHO.
            run(["bash", "scripts/forge-build-lim.sh"], args.output / "build.log", 1800)
            BUILD_STATE.write_text(actual + "\n")
        resultdir = args.output / "chain"
        command = [
            "bash",
            "scripts/vllm-ascend-run.sh",
            "python",
            "-u",
            "scripts/collect-echo-prefetch-profile-npu.py",
            "--batches",
            "16",
            "--mtp",
            "3",
            "--context",
            "40960",
            "--hot-tokens",
            "8192",
            "--misses-per-request",
            "400",
            "--modes",
            "baseline",
            "context-budget-0",
            "prefetch-64-history",
            "--repeats",
            str(args.repeats),
            "--profile-repeats",
            str(args.profile_repeats),
            "--output",
            str(resultdir),
        ]
        if args.attention_reference:
            command.append("--attention-reference")
        run(command, args.output / "chain.log", 1200)
        data = json.loads((resultdir / "result.json").read_text())
        if data["status"] != "completed_operator_chain_only":
            raise RuntimeError("Incomplete chain validation")
        rows = {x["mode"]: x for x in data["metrics"]}
        for mode in ("context-budget-0", "prefetch-64-history"):
            if len(rows[mode].get("dynamic_graph_checks", [])) != 7:
                raise RuntimeError(f"Missing dynamic Graph scenarios in {mode}")
        ref = rows["baseline"]["event_median_us"] / 1000
        candidate = rows["prefetch-64-history"]["event_median_us"] / 1000
        if not all(
            x == 400 for x in rows["baseline"]["original_union_misses_per_request"]
        ):
            raise RuntimeError("The frozen request-union miss count changed")
        if sum(rows["prefetch-64-history"]["prefetched_per_query"]) != 16 * 4 * 64:
            raise RuntimeError(
                "Candidate disabled or reduced the requested actual prefetch work"
            )
        result = {
            "correctness": {
                "pass": True,
                "dynamic_scenarios": 14,
                "oracle": "exact original LIM outputs and complete KV/KPE; attention rtol .02 atol .002",
            },
            "performance": {
                "device_median_ms": candidate,
                "ref_device_ms": ref,
                "vs_ref_device": ref / candidate,
                "device_window_basis": "npu_graph",
                "budget_zero_median_ms": rows["context-budget-0"]["event_median_us"]
                / 1000,
            },
            "native_source_sha256": actual,
            "installed_binary_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(
                    (ROOT / "vllm-ascend/vllm_ascend/_cann_ops_custom").rglob("*.o")
                )
                if "fused_lightning_indexer_manage" in str(path)
                or "fused_scatter_copy_sparse_flash_attention" in str(path)
            },
            "consumer_source_sha256": hashlib.sha256(
                (ROOT / "scripts/echo_indexer_prefetch.py").read_bytes()
            ).hexdigest(),
            "scope": "B16 Q4 L40960 C8192 request-union miss400; fixed synthetic queries, one device; whole Graph",
        }
        (args.output / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
        print("=== JSON ===")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("=== JSON ===")
        print(
            json.dumps(
                {"correctness": {"pass": False, "error": str(exc)}, "performance": {}}
            ),
            flush=True,
        )
        raise
