#!/usr/bin/env python3
"""Prepare or launch exactly the three document-aligned GLM configurations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("original", "pr", "optimized")


def make_command(variant, model, port, memory_fraction, transfer):
    batch = 5 if variant == "original" else 16
    capture_sizes = [4, 8, 16, 20] if batch == 5 else [4, 8, 16, 32, 64]
    command = ["bash", str(ROOT / "scripts/vllm-ascend-variant-run.sh"), variant,
               "vllm", "serve", str(model), "--host", "127.0.0.1", "--port", str(port),
               "--served-model-name", "glm52", "--tensor-parallel-size", "16",
               "--data-parallel-size", "1", "--enable-expert-parallel",
               "--quantization", "ascend", "--dtype", "bfloat16", "--block-size", "128",
               "--max-model-len", "65536", "--max-num-seqs", str(batch),
               "--max-num-batched-tokens", "4096", "--no-enable-prefix-caching",
               "--gpu-memory-utilization", str(memory_fraction), "--seed", "1024",
               "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3,
                                                    "enforce_eager": True}),
               "--compilation-config", json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY",
                                                    "cudagraph_capture_sizes": capture_sizes})]
    offload = {"enabled": variant != "original"}
    if variant != "original":
        offload.update(fused_op_type="fused_copy_sfa", topk_buffer_size=12288,
                       dram_size_per_dp_GB=128, keep_device_kv_cache=False, use_fused_overlap=False)
        if transfer is not None:
            command.extend(["--kv-transfer-config", json.dumps(transfer)])
    command.extend(["--additional-config", json.dumps({"sparse_kv_offload_config": offload})])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "build/models/GLM-5.2-w4a8")
    parser.add_argument("--port", type=int, default=8310)
    parser.add_argument("--memory-fraction", type=float, default=.92)
    parser.add_argument("--kv-transfer-config", type=Path,
                        help="real Prefill/Decode consumer connector configuration for offload variants")
    parser.add_argument("--run", choices=VARIANTS,
                        help="launch one configuration; omit to emit the three-group plan only")
    parser.add_argument("--device-check", type=Path,
                        help="completed fresh-process TP16 device preflight, required by --run")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.memory_fraction < 1:
        parser.error("memory fraction must be between zero and one")
    config_path = args.model / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "glm_moe_dsa" or config.get("num_hidden_layers") != 78:
        raise ValueError("Requires the full GLM-5.2 model; no reduced-layer substitute")
    transfer = None
    if args.kv_transfer_config:
        transfer = json.loads(args.kv_transfer_config.read_text())
        if transfer.get("kv_role") != "kv_consumer":
            raise ValueError("The offload comparison requires a real KV consumer configuration")
    env = {"ECHO_NPU_DEVICE": ",".join(map(str, range(16))),
           "VLLM_USE_V2_MODEL_RUNNER": "0", "HF_HUB_OFFLINE": "1",
           "HCCL_BUFFSIZE": "200", "HCCL_OP_EXPANSION_MODE": "AIV",
           "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"}
    groups = []
    for variant in VARIANTS:
        groups.append({"variant": variant, "batch": 5 if variant == "original" else 16,
                       "context_tokens": 64000, "hot_tokens": None if variant == "original" else 12288,
                       "tp": 16, "ep": 16, "mtp": 3,
                       "missing_prerequisites": (["real Prefill KV consumer configuration"]
                                                 if variant != "original" and transfer is None else []),
                       "command": make_command(variant, args.model.resolve(), args.port, args.memory_fraction, transfer)})
    result = {"status": "planned_not_executed", "groups": groups, "environment": env,
              "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "note": "Startup plan only. No full-model, throughput, acceptance or graph-replay claim.",
              "conditional_rate_sweep": {"rates": [.3, .5, .7, "inf"], "requests_per_rate": 100,
                                         "gate": "actual throughput >=2x original and faster than PR"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Prepared exactly three configurations: {args.output}")
    if args.run:
        group = next(row for row in groups if row["variant"] == args.run)
        if group["missing_prerequisites"]:
            raise RuntimeError(", ".join(group["missing_prerequisites"]))
        if not args.device_check:
            raise RuntimeError("Run check-glm-tp16-devices.py before launching TP16")
        check = json.loads(args.device_check.read_text())
        if time.time() - check.get("updated_at_unix", 0) > 600:
            raise RuntimeError("Device preflight is older than ten minutes; recheck before allocating TP16")
        devices = check.get("devices", [])
        if (not check.get("complete") or len(devices) != 16 or
                {row["device"] for row in devices} != set(range(16)) or
                any(row["returncode"] != 0 for row in devices)):
            raise RuntimeError("TP16 device preflight has failed or missing devices")
        if any(row["free_bytes"] < row["total_bytes"] * args.memory_fraction for row in devices):
            raise RuntimeError("Insufficient free device memory for the configured common memory budget")
        run_env = dict(os.environ, **env)
        result["status"] = "launching"
        result["selected_variant"] = args.run
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        completed = subprocess.run(group["command"], env=run_env, cwd=ROOT)
        result["status"] = "server_exited"
        result["returncode"] = completed.returncode
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
