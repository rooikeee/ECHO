#!/usr/bin/env python3
"""Original ECHO mixed-PD ShareGPT methodology, with half-sized recall pools.

Hardware, model checkpoint, NPU kernels and absent prefetch differ from CUDA.
The published mixed-PD script (not the conflicting prose) defines the profile.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_helper("sharegpt_driver", "bench-echo-sharegpt-npu.py")
parallel = load_helper("parallel_driver", "smoke-echo-parallel-npu.py")
REQUEST_RATES = ["inf", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"]
LOGICAL_TOKENS_REQUESTED = 500_000
LOGICAL_TOKENS = LOGICAL_TOKENS_REQUESTED // 128 * 128
HOT_TOKENS_REQUESTED = 100_000
HOT_TOKENS = HOT_TOKENS_REQUESTED // 128 * 128


def ensure_idle(info, devices):
    # This profile targets the documented two-chip-per-board machine layout.
    for board in {int(device) // 2 for device in devices}:
        if f"No running processes found in NPU {board}" not in info:
            raise RuntimeError(f"NPU board {board} is occupied; no processes were stopped")


def health_snapshot(directory, label, devices):
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                            timeout=120, check=True)
    (directory / f"{label}-npu-info.txt").write_text(result.stdout)
    health = []
    for physical_id in map(int, devices):
        check = subprocess.run(
            ["npu-smi", "info", "-t", "health", "-i", str(physical_id // 2),
             "-c", str(physical_id % 2)], capture_output=True, text=True,
            timeout=120, check=True,
        )
        fields = dict(line.split(":", 1) for line in check.stdout.splitlines() if ":" in line)
        health.append({"physical_id": physical_id,
                       **{key.strip(): value.strip() for key, value in fields.items()}})
    common.save_json(directory / f"{label}-health.json", health)
    return result.stdout, health


def server_command(args, port, echo):
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(args.model), "--trust-remote-code",
        "--device", "npu", "--attention-backend", "ascend",
        "--quantization", "awq", "--dtype", "bfloat16",
        "--tp-size", "8", "--dp-size", "8", "--enable-dp-attention",
        "--host", "127.0.0.1", "--port", str(port),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--disable-radix-cache", "--disable-overlap-schedule",
        "--chunked-prefill-size", "-1", "--cuda-graph-max-bs", "32", "--page-size", "128",
        "--disable-shared-experts-fusion", "--decode-log-interval", "1",
        "--watchdog-timeout", "1800", "--random-seed", "1",
    ]
    if echo:
        command += ["--enable-echo", "--max-total-tokens", str(LOGICAL_TOKENS),
                    "--echo-device-cache-size", str(HOT_TOKENS)]
    return command


def benchmark_command(args, port, name, rate):
    # No fixed output budget, context filter, or client concurrency cap.
    return [
        sys.executable, str(Path(__file__).with_name("bench-echo-sharegpt-npu.py")),
        "--benchmark-worker", "--backend", "sglang", "--host", "127.0.0.1",
        "--port", str(port), "--model", str(args.model),
        "--dataset-name", "sharegpt", "--dataset-path", str(args.dataset),
        "--num-prompts", "100", "--seed", "1", "--warmup-requests", "0",
        "--request-rate", rate, "--output-details", "--disable-tqdm",
        "--output-file", str(args.output_dir / f"{name}-{rate}-metrics.jsonl"),
        "--output-request-file", str(args.output_dir / f"{name}-{rate}-requests.jsonl"),
    ]


def validate_result(result, samples):
    if result["completed"] != 100 or len(samples) != 100:
        raise AssertionError("Expected 100 successful requests")
    if result["input_lens"] != [sample["input_tokens"] for sample in samples]:
        raise AssertionError("Input sampling differs from manifest")
    if result["output_lens"] != [sample["output_tokens"] for sample in samples]:
        raise AssertionError("Output budgets differ from dataset completion lengths")
    if any(result["errors"]) or len(result["generated_texts"]) != 100:
        raise AssertionError("Failed or missing requests")


def validate_capacity(log, echo):
    capacities = [int(value) for value in re.findall(r"max_total_num_tokens=(\d+)", log)]
    if len(capacities) < 8 or any(value <= 0 for value in capacities):
        raise AssertionError("Missing allocated token capacity on one or more ranks")
    if echo:
        allocations = re.findall(
            r"ECHO exact recall enabled: logical tokens=(\d+), cache tokens/layer=(\d+)", log,
        )
        if len(allocations) != 8 or any(
            int(logical) != LOGICAL_TOKENS or int(hot) != HOT_TOKENS
            for logical, hot in allocations
        ):
            raise AssertionError("ECHO pools were clamped or do not match half-pool profile")
        if any(value != LOGICAL_TOKENS for value in capacities):
            raise AssertionError("Scheduler does not expose the requested ECHO capacity")
    return capacities


def graph_summary(log, echo):
    evidence = parallel.validate_graph(log, 8, 8, echo)
    batches = {}
    for rank, batch, graph in re.findall(
        r"\[[^\n\]]*\bDP(\d+)\b[^\n\]]*\] Decode batch[^\n]*"
        r"#running-req: (\d+),[^\n]*npu graph: (True|False)", log,
    ):
        key = f"dp{rank}:batch{batch}:graph{graph}"
        batches[key] = batches.get(key, 0) + 1
    evidence["observed_local_decode_batch_log_counts"] = batches
    return evidence


@contextmanager
def running_server(args, echo):
    name = "echo" if echo else "baseline"
    info, health = health_snapshot(args.output_dir, f"{name}-before", args.devices)
    ensure_idle(info, args.devices)
    if any(item.get("Health Status") != "OK" for item in health):
        raise RuntimeError("Selected devices are not healthy; no server was started")
    port = parallel.choose_server_port()
    command = server_command(args, port, echo)
    common.save_json(args.output_dir / f"{name}-server-command.json", command)
    environment = dict(os.environ)
    for key, value in {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True", "HCCL_BUFFSIZE": "256",
        "HCCL_OP_EXPANSION_MODE": "AIV", "HCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo",
    }.items():
        environment.setdefault(key, value)
    common.save_json(args.output_dir / f"{name}-runtime-env.json", {
        key: value for key, value in environment.items()
        if key.startswith(("HCCL_", "GLOO_", "PYTORCH_NPU_", "ASCEND_RT_"))
    })
    path = args.output_dir / f"{name}-server.log"
    print(f"Starting {name} TP8/DP8: {path}", flush=True)
    with path.open("w") as log:
        server = subprocess.Popen(command, env=environment, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"{name} exited ({server.returncode}); see {path}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(2)
            else:
                raise TimeoutError(f"{name} startup timed out")
            validate_capacity(path.read_text(errors="replace"), echo)
            yield port, server
        finally:
            common.stop_process(server)


def graph_probe(args, port, echo):
    records = []
    for rank in range(8):
        for batch in (1, 2, 1):
            result = common.post_json(port, "/generate", {
                "text": ["Explain what a computer cache does."] * batch,
                "routed_dp_rank": rank,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            })
            if not isinstance(result, list) or len(result) != batch or any(
                row["meta_info"]["completion_tokens"] != 8
                or row["meta_info"].get("dp_rank") != rank for row in result
            ):
                raise AssertionError("Incorrect routed Graph probe response")
            records.append({"rank": rank, "batch": batch, "responses": result})
    name = "echo" if echo else "baseline"
    common.save_json(args.output_dir / f"{name}-graph-probe.json", records)
    path = args.output_dir / f"{name}-server.log"
    graph_summary(path.read_text(errors="replace"), echo)
    # Probes are untimed, then clear their KV before the first rate.
    request = urllib.request.Request(f"http://127.0.0.1:{port}/flush_cache", method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise AssertionError("Could not flush Graph probe requests")


def run_mode(args, echo, samples, report):
    name = "echo" if echo else "baseline"
    with running_server(args, echo) as (port, server):
        graph_probe(args, port, echo)
        for rate in REQUEST_RATES:
            command = benchmark_command(args, port, name, rate)
            common.save_json(args.output_dir / f"{name}-{rate}-command.json", command)
            with (args.output_dir / f"{name}-{rate}-benchmark.log").open("w") as log:
                client = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                try:
                    deadline = time.monotonic() + args.benchmark_timeout
                    while client.poll() is None:
                        if server.poll() is not None:
                            raise RuntimeError(f"{name} server exited during rate {rate}")
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"{name} rate {rate} timed out")
                        time.sleep(2)
                    if client.returncode:
                        raise RuntimeError(f"{name} rate {rate}: client exited {client.returncode}")
                finally:
                    common.stop_process(client)
            lines = (args.output_dir / f"{name}-{rate}-metrics.jsonl").read_text().splitlines()
            if len(lines) != 1:
                raise AssertionError("Expected one result per request rate")
            result = json.loads(lines[0])
            validate_result(result, samples)
            evidence = graph_summary((args.output_dir / f"{name}-server.log").read_text(errors="replace"), echo)
            common.save_json(args.output_dir / f"{name}-{rate}-graph.json", evidence)
            report[name][rate] = {key: result[key] for key in common.METRICS}
            common.save_json(args.output_dir / "result.json", report)
            print(f"{name} rate={rate}: {result['output_throughput']:.3f} token/s", flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "build/models/DeepSeek-V3.2-AWQ")
    parser.add_argument("--dataset", type=Path, default=root / "build/datasets/sharegpt" / common.DATASET_FILENAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--devices", default=",".join(common.DEVICES),
                        help="Eight physical chip IDs; two-chip-per-board layout required")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--benchmark-timeout", type=int, default=7200)
    args = parser.parse_args()
    args.devices = args.devices.split(",")
    if len(args.devices) != 8 or len(set(args.devices)) != 8 or not all(
        value.isdigit() for value in args.devices
    ):
        parser.error("--devices requires eight distinct nonnegative physical chip IDs")
    if not math.isfinite(args.mem_fraction_static) or not 0 < args.mem_fraction_static < 1:
        parser.error("--mem-fraction-static must be between 0 and 1")
    args.model = args.model.resolve()
    args.dataset = args.dataset.resolve()
    args.output_dir = (args.output_dir or root / "build/echo-sharegpt-paper" / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.seed, args.num_prompts, args.output_len, args.sample_context_len = 1, 100, None, None
    report = {
        "status": "preparing", "profile": "original_mixed_pd_script_half_pools_no_prefetch",
        "source_commit": "bc1b75c1000010d0ac6f032ebaac283255c050b1",
        "source_script": "sglang/e2e_test_mix_sharegpt.sh",
        "model": str(args.model), "dataset_sha256": common.DATASET_SHA256,
        "devices": args.devices, "tp_size": 8, "attention_dp_size": 8,
        "request_rates": REQUEST_RATES, "num_prompts_per_rate": 100,
        "client_max_concurrency": None, "server_max_running_requests": "framework_default",
        "output_lengths": "dataset_completion_token_lengths; ignore_eos=True",
        "warmup_requests": 0, "seed": 1, "pd_disaggregated": False,
        "chunked_prefill_size": -1, "graph_max_bs": 32,
        "logical_tokens_requested_per_rank": LOGICAL_TOKENS_REQUESTED,
        "logical_tokens_per_rank": LOGICAL_TOKENS,
        "hot_tokens_requested": HOT_TOKENS_REQUESTED, "hot_tokens_aligned": HOT_TOKENS,
        "mem_fraction_static": args.mem_fraction_static, "prefetch": False,
        "paper_configuration_reproduction": False,
        "differences": ["Ascend hardware/software/kernels", "V3.2-AWQ instead of V3.2-Exp-AWQ",
                        "no predictive prefetch", "half-sized ECHO pools; both pools aligned down to 128",
                        "BF16 MLA KV and Indexer K", "shared-experts fusion disabled on Ascend",
                        "original mixed-PD script uses prefill=-1, paper prose says 2048"],
        "baseline": {}, "echo": {},
    }
    if args.mem_fraction_static != 0.8:
        report["differences"].append("mem_fraction_static differs from original 0.8")
    path = args.output_dir / "result.json"
    common.save_json(path, report)
    started = time.monotonic()
    try:
        # Save the complete intended commands even when devices are occupied.
        for echo in (False, True):
            name = "echo" if echo else "baseline"
            common.save_json(args.output_dir / f"{name}-planned-server-command.json", server_command(args, 18000, echo))
            for rate in REQUEST_RATES:
                common.save_json(args.output_dir / f"{name}-{rate}-planned-command.json", benchmark_command(args, 18000, name, rate))
        if not args.prepare_only:
            if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") != args.devices:
                raise ValueError("ECHO_NPU_DEVICE must match --devices in the same order")
            info, health = health_snapshot(args.output_dir, "before", args.devices)
            report["selected_device_health"] = health
            ensure_idle(info, args.devices)
        samples = common.prepare_samples(args)
        report["input_tokens"] = sum(row["input_tokens"] for row in samples)
        report["output_tokens"] = sum(row["output_tokens"] for row in samples)
        if args.prepare_only:
            report["status"] = "prepared_not_run"
        else:
            for echo in (False, True):
                report["status"] = "echo_running" if echo else "baseline_running"
                common.save_json(path, report)
                run_mode(args, echo, samples, report)
            report["status"] = "completed_performance_only_not_accuracy_acceptance"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        common.save_json(path, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    main()
