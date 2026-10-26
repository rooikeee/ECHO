"""Launch the pinned full model for synthetic-KV performance experiments."""

import argparse
import importlib.util
import json
import os
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("variant", choices=["original", "optimized"])
p.add_argument("--port", type=int, default=8310)
p.add_argument("--memory", type=float, default=0.9)
p.add_argument("--scheduler-cap", type=int, default=30)
p.add_argument("--resident-capacity", type=int, default=16)
args = p.parse_args()
if args.memory != 0.9:
    p.error("This recorded experiment uses a fixed 90% weight/cache budget")
spec = importlib.util.spec_from_file_location("plan", root / "scripts/prepare-glm52-three-way.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
transfer = {
    "kv_connector": "EchoSharedDecodeBenchConnector",
    "kv_connector_module_path": "echo_decode_bench_connector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {"echo_shared_hot_cache": {"request_capacity": args.resident_capacity}},
}
cmd = mod.make_command(args.variant, root / "build/models/GLM-5.2-w4a8", args.port, args.memory, transfer)
if args.variant == "original":
    transfer["kv_connector"] = "EchoDecodeBenchConnector"
    transfer.pop("kv_connector_extra_config", None)
    cmd += ["--kv-transfer-config", json.dumps(transfer)]
cmd[cmd.index("--max-num-seqs") + 1] = str(args.scheduler_cap)
cmd[cmd.index("--max-num-batched-tokens") + 1] = "256"
cmd[cmd.index("--compilation-config") + 1] = json.dumps(
    {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [4, 8, 16, 20, 24] if args.variant == "original" else [4, 8, 16, 24, 32, 48, 64],
    }
)
cmd += ["--no-async-scheduling", "--worker-cls", "echo_memory_budget_worker.EchoMemoryBudgetWorker"]
env = dict(
    os.environ,
    ECHO_NPU_DEVICE=",".join(map(str, range(16))),
    VLLM_USE_V2_MODEL_RUNNER="0",
    HF_HUB_OFFLINE="1",
    HCCL_BUFFSIZE="200",
    HCCL_OP_EXPANSION_MODE="AIV",
    PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
)
env["PYTHONPATH"] = str(root / "scripts") + ":" + env.get("PYTHONPATH", "")
env["CPATH"] = str(root / "build/vllm-tools/usr/include/python3.11") + ":" + env.get("CPATH", "")
print(
    json.dumps(
        {"variant": args.variant, "command": cmd, "synthetic_kv": True, "devices": env["ECHO_NPU_DEVICE"]}, indent=2
    ),
    flush=True,
)
os.execvpe(cmd[0], cmd, env)
