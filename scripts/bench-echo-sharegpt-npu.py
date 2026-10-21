#!/usr/bin/env python3
"""Paired ShareGPT latency benchmark for native Ascend and exact ECHO recall."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import runpy
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


DATASET_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DATASET_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
DATASET_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
DEVICES = [str(i) for i in range(8, 16)]
METRICS = (
    "duration", "completed", "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_throughput",
    "mean_e2e_latency_ms", "median_e2e_latency_ms", "p99_e2e_latency_ms",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p99_itl_ms", "max_concurrent_requests",
)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_samples(args):
    from sglang.benchmark.datasets.sharegpt import sample_sharegpt_requests
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    checksum = file_hash(args.dataset)
    if checksum != DATASET_SHA256:
        raise ValueError(f"ShareGPT checksum mismatch: {checksum}")
    tokenizer = get_tokenizer(str(args.model), trust_remote_code=True)
    random.seed(args.seed)
    rows = sample_sharegpt_requests(
        str(args.dataset), args.num_prompts, tokenizer,
        fixed_output_len=args.output_len,
        context_len=args.sample_context_len,
    )
    if len(rows) != args.num_prompts:
        raise ValueError(f"Only {len(rows)} eligible ShareGPT samples")
    samples = [
        {
            "index": index,
            "prompt": row.prompt,
            "prompt_sha256": hashlib.sha256(row.prompt.encode()).hexdigest(),
            "input_tokens": row.prompt_len,
            "output_tokens": row.output_len,
        }
        for index, row in enumerate(rows)
    ]
    save_json(args.output_dir / "samples.json", samples)
    return samples


def health_snapshot(directory, label):
    completed = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, timeout=30,
        check=True,
    )
    (directory / f"{label}-npu-info.txt").write_text(completed.stdout)
    devices = []
    for physical_id in range(8, 16):
        result = subprocess.run(
            ["npu-smi", "info", "-t", "health", "-i", str(physical_id // 2),
             "-c", str(physical_id % 2)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        fields = dict(
            line.strip().split(":", 1)
            for line in result.stdout.splitlines() if ":" in line
        )
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        devices.append({"physical_id": physical_id, **fields})
    save_json(directory / f"{label}-health.json", devices)
    return completed.stdout, devices


def ensure_idle(info):
    for board in range(4, 8):
        if f"No running processes found in NPU {board}" not in info:
            raise RuntimeError(f"NPU board {board} is occupied; no processes were stopped")


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def server_command(args, port, echo):
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(args.model), "--trust-remote-code",
        "--device", "npu", "--attention-backend", "ascend",
        "--quantization", "awq", "--dtype", "bfloat16", "--tp-size", "8",
        "--host", "127.0.0.1", "--port", str(port),
        "--mem-fraction-static", "0.80", "--context-length", "4096",
        "--max-total-tokens", "8192", "--max-running-requests", "2",
        "--chunked-prefill-size", str(args.prefill_chunk_size),
        "--max-prefill-tokens", "2048",
        "--disable-radix-cache", "--disable-overlap-schedule",
        "--disable-shared-experts-fusion", "--cuda-graph-bs", "1", "2",
        "--decode-log-interval", "10", "--watchdog-timeout", "1800",
        "--random-seed", str(args.seed),
    ]
    if echo:
        command += ["--enable-echo", "--echo-device-cache-size", "4096"]
    return command


@contextmanager
def running_server(args, echo):
    name = "echo" if echo else "baseline"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = server_command(args, port, echo)
    save_json(args.output_dir / f"{name}-server-command.json", command)
    environment = dict(os.environ)
    environment.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("HCCL_BUFFSIZE", "1536")
    environment.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    environment.setdefault("HCCL_SOCKET_IFNAME", "lo")
    environment.setdefault("GLOO_SOCKET_IFNAME", "lo")
    log_path = args.output_dir / f"{name}-server.log"
    print(f"Starting {name} TP8 server: {log_path}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited ({process.returncode}); see {log_path}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=10) as response:
                        if response.status != 200:
                            raise OSError(f"Health status: {response.status}")
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(2)
            else:
                raise TimeoutError(f"{name} startup timed out; see {log_path}")
            yield port, process
        finally:
            stop_process(process)


def benchmark_command(args, port, name):
    return [
        sys.executable, str(Path(__file__).resolve()), "--benchmark-worker",
        "--backend", "sglang", "--host", "127.0.0.1", "--port", str(port),
        "--model", str(args.model), "--dataset-name", "sharegpt",
        "--dataset-path", str(args.dataset), "--num-prompts", str(args.num_prompts),
        "--sharegpt-output-len", str(args.output_len),
        "--sharegpt-context-len", str(args.sample_context_len),
        "--request-rate", "inf", "--max-concurrency", "2",
        "--seed", str(args.seed), "--warmup-requests", "2", "--flush-cache",
        "--output-details", "--disable-tqdm",
        "--output-file", str(args.output_dir / f"{name}-metrics.jsonl"),
        "--output-request-file", str(args.output_dir / f"{name}-requests.jsonl"),
    ]


def validate_result(result, samples, output_len):
    if result["completed"] != len(samples):
        raise AssertionError(f"Incomplete benchmark: {result['completed']}/{len(samples)}")
    if result["input_lens"] != [sample["input_tokens"] for sample in samples]:
        raise AssertionError("Benchmark sampling does not match the saved manifest")
    if result["output_lens"] != [output_len] * len(samples):
        raise AssertionError("Benchmark did not generate the fixed output budget")
    if any(result["errors"]):
        raise AssertionError("Benchmark contains failed requests")


def validate_graph(log, echo, batch_sizes=(1, 2)):
    if "Capture npu graph end" not in log:
        raise AssertionError("Missing NPU Graph capture evidence")
    for batch_size in batch_sizes:
        pattern = rf"Decode batch[^\n]*#running-req: {batch_size},[^\n]*npu graph: True"
        if not re.search(pattern, log):
            raise AssertionError(f"Missing Graph replay for batch {batch_size}")
    if re.search(r"Decode batch[^\n]*npu graph: False", log):
        raise AssertionError("Eager decode fallback observed")
    if echo and "ECHO exact recall enabled" not in log:
        raise AssertionError("ECHO cache pool was not activated")


def post_json(port, path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def graph_probe(args, port, name):
    payload = {
        "text": "Explain what a computer cache does.",
        "sampling_params": {"temperature": 0, "max_new_tokens": 32, "ignore_eos": True},
    }
    single = post_json(port, "/generate", payload)
    with ThreadPoolExecutor(max_workers=2) as executor:
        paired = list(executor.map(lambda _: post_json(port, "/generate", payload), range(2)))
    results = [single, *paired]
    save_json(args.output_dir / f"{name}-graph-probe.json", results)
    if any(item["meta_info"]["completion_tokens"] != 32 for item in results):
        raise AssertionError("Graph probe did not complete its generation budget")
    validate_graph((args.output_dir / f"{name}-server.log").read_text(), name == "echo")


def normalize_command(command, option_names, ignored_options=()):
    command = list(command)
    for name in ignored_options:
        if name in command:
            index = command.index(name)
            del command[index:index + 2]
    for name in option_names:
        command[command.index(name) + 1] = f"<{name}>"
    return command


def load_baseline(args, samples):
    source = args.reuse_baseline_dir.resolve()
    if json.loads((source / "samples.json").read_text()) != samples:
        raise AssertionError("Reused baseline has different ShareGPT inputs")
    old_server = json.loads((source / "baseline-server-command.json").read_text())
    if normalize_command(old_server, ["--port"]) != normalize_command(
        server_command(args, 0, False), ["--port"]
    ):
        raise AssertionError("Reused baseline has different server settings")
    old_benchmark = json.loads((source / "baseline-benchmark-command.json").read_text())
    old_args = copy(args)
    old_args.output_dir = source
    if normalize_command(old_benchmark, ["--port"], ["--output-request-file"]) != normalize_command(
        benchmark_command(old_args, 0, "baseline"), ["--port"], ["--output-request-file"]
    ):
        raise AssertionError("Reused baseline has different benchmark settings")
    lines = (source / "baseline-metrics.jsonl").read_text().splitlines()
    if len(lines) != 1:
        raise AssertionError("Expected exactly one saved baseline result")
    result = json.loads(lines[0])
    validate_result(result, samples, args.output_len)
    # The saved timed run must prove batch 2; the fresh probe proves both sizes.
    validate_graph((source / "baseline-server.log").read_text(), False, (2,))
    save_json(args.output_dir / "baseline-reused-metrics.json", result)
    return result


def run_mode(args, echo, samples, reused_baseline=None):
    name = "echo" if echo else "baseline"
    with running_server(args, echo) as (port, server):
        print(f"Checking {name} Graph replay for batch 1/2", flush=True)
        graph_probe(args, port, name)
        if reused_baseline is not None:
            print("Baseline Graph probe passed; reusing validated timed requests", flush=True)
            return reused_baseline
        command = benchmark_command(args, port, name)
        save_json(args.output_dir / f"{name}-benchmark-command.json", command)
        log_path = args.output_dir / f"{name}-benchmark.log"
        print(f"Running {name} ShareGPT benchmark: {log_path}", flush=True)
        with log_path.open("w") as log:
            client = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                deadline = time.monotonic() + args.benchmark_timeout
                while client.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError(f"{name} server exited during the benchmark")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{name} benchmark timed out")
                    time.sleep(2)
                if client.returncode:
                    raise RuntimeError(f"{name} benchmark exited ({client.returncode}); see {log_path}")
            finally:
                stop_process(client)
        lines = (args.output_dir / f"{name}-metrics.jsonl").read_text().splitlines()
        if len(lines) != 1:
            raise AssertionError("Expected exactly one benchmark result")
        result = json.loads(lines[0])
        validate_result(result, samples, args.output_len)
        validate_graph((args.output_dir / f"{name}-server.log").read_text(), echo)
        print(json.dumps({name: {key: result[key] for key in METRICS}}, indent=2), flush=True)
        return result


def compare_results(baseline, echo):
    agreements = sum(
        left == right
        for left, right in zip(baseline["generated_texts"], echo["generated_texts"])
    )
    return {
        "output_throughput_speedup": echo["output_throughput"] / baseline["output_throughput"],
        "mean_e2e_latency_speedup": baseline["mean_e2e_latency_ms"] / echo["mean_e2e_latency_ms"],
        "mean_ttft_speedup": baseline["mean_ttft_ms"] / echo["mean_ttft_ms"],
        "mean_tpot_speedup": baseline["mean_tpot_ms"] / echo["mean_tpot_ms"],
        "benchmark_duration_speedup": baseline["duration"] / echo["duration"],
        "identical_generated_texts": agreements,
        "generated_text_agreement_fraction": agreements / baseline["completed"],
        "quality_accuracy_measured": False,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "build/models/DeepSeek-V3.2-AWQ")
    parser.add_argument("--dataset", type=Path, default=root / "build/datasets/sharegpt" / DATASET_FILENAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--sample-context-len", type=int, default=2048)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--benchmark-timeout", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-baseline-dir", type=Path)
    args = parser.parse_args()
    if args.num_prompts < 2 or args.output_len < 4:
        parser.error("Need at least two prompts and four generated tokens")
    if not args.output_len < args.sample_context_len <= 2048:
        parser.error("Sample context must exceed output length and be <= 2048 for this first-stage run")
    if args.prefill_chunk_size < 128 or args.prefill_chunk_size % 128:
        parser.error("Prefill chunk size must be a positive multiple of 128")
    if args.output_dir is None:
        args.output_dir = root / "build/echo-sharegpt" / datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = args.output_dir.resolve()
    args.model = args.model.resolve()
    args.dataset = args.dataset.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "preparing", "model": str(args.model), "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION, "dataset_sha256": DATASET_SHA256,
        "dataset_path": str(args.dataset), "devices": DEVICES, "tp_size": 8,
        "dp_size": 1, "pd_disaggregated": False, "npu_graph": True,
        "graph_batch_sizes": [1, 2], "seed": args.seed,
        "num_prompts": args.num_prompts, "output_len": args.output_len,
        "sample_context_len": args.sample_context_len,
        "prefill_chunk_size": args.prefill_chunk_size, "max_concurrency": 2,
        "request_rate": "inf", "warmup_requests": 2,
        "server_context_length": 4096, "logical_token_capacity": 8192,
        "echo_device_cache_size": 4096, "ignore_eos": True,
        "apply_chat_template": False, "paper_configuration_reproduction": False,
        "repetitions_per_mode": 1, "startup_and_warmup_in_metrics": False,
        "benchmark_tokenizer": "sglang.srt.utils.hf_transformers_utils.get_tokenizer",
        "per_request_journal": True,
    }
    report_path = args.output_dir / "result.json"
    save_json(report_path, report)
    print(f"Result directory: {args.output_dir}", flush=True)
    start = time.monotonic()
    hardware_checked = False
    try:
        samples = prepare_samples(args)
        report["prompt_lengths"] = [sample["input_tokens"] for sample in samples]
        if args.prepare_only:
            report["status"] = "prepared"
            return
        reused_baseline = None
        if args.reuse_baseline_dir is not None:
            reused_baseline = load_baseline(args, samples)
            report["baseline_reused_from"] = str(args.reuse_baseline_dir.resolve())
            report["baseline_reuse_requires_fresh_graph_probe"] = True
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") != DEVICES:
            raise ValueError("Run with ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15")
        info, health = health_snapshot(args.output_dir, "before")
        ensure_idle(info)
        hardware_checked = True
        report["hardware_health_before"] = health
        report["hardware_health_ok"] = all(item.get("Health Status") == "OK" for item in health)
        if not report["hardware_health_ok"]:
            print("WARNING: device health alarm; no reset is performed, results are provisional", flush=True)
        report["status"] = "baseline_running"
        save_json(report_path, report)
        baseline = run_mode(args, False, samples, reused_baseline)
        report["baseline"] = {key: baseline[key] for key in METRICS}
        report["status"] = "echo_running"
        save_json(report_path, report)
        echo = run_mode(args, True, samples)
        report["echo"] = {key: echo[key] for key in METRICS}
        report["comparison"] = compare_results(baseline, echo)
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if hardware_checked:
            try:
                _, health = health_snapshot(args.output_dir, "after")
                report["hardware_health_after"] = health
                report["hardware_health_ok"] &= all(item.get("Health Status") == "OK" for item in health)
                if report["status"] == "completed" and not report["hardware_health_ok"]:
                    report["status"] = "completed_with_hardware_alarm"
            except Exception as error:
                report["hardware_check_error"] = str(error)
                report["hardware_health_ok"] = False
                if report["status"] == "completed":
                    report["status"] = "completed_with_health_check_error"
        report["elapsed_seconds_including_startup"] = round(time.monotonic() - start, 2)
        save_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark-worker":
        from functools import partial

        import sglang.benchmark.utils as benchmark_utils
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        # Use the server's v5 tokenizer fixes without changing upstream metrics.
        benchmark_utils.get_tokenizer = partial(get_tokenizer, trust_remote_code=True)
        sys.argv = ["sglang.bench_serving", *sys.argv[2:]]
        runpy.run_module("sglang.bench_serving", run_name="__main__")
    else:
        main()
