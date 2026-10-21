#!/usr/bin/env python3
"""Compare full DeepSeek NSA serving with/without exact ECHO recall on TP8."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def prompts(model, skip_long=False):
    from jinja2 import Template
    from tokenizers import Tokenizer

    template = Template((model / "chat_template.jinja").read_text())
    config = json.loads((model / "tokenizer_config.json").read_text())
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))

    def encode(content):
        text = template.render(
            messages=[{"role": "user", "content": content}],
            bos_token=config["bos_token"], add_generation_prompt=True, thinking=False,
        )
        return tokenizer.encode(text).ids

    first = encode("17 加 25 等于多少？请直接给出答案。")
    second = encode("请用一句话介绍中国的首都。")
    long = encode("这是用于检查长文本处理的背景记录。\n" * 280 + "请只回答：17 加 25 等于多少？")
    if not 2048 < len(long) < 4000:
        raise ValueError(f"Expected a prompt longer than NSA Top-K; got {len(long)}")
    batches = [[first], [first, second], [first]]
    if not skip_long:
        batches.append([long])
    return batches


def run_server(root, args, echo, batches):
    name = "echo" if echo else "baseline"
    directory = root / "build" / "echo-deepseek-smoke"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{name}-server.log"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(args.model), "--trust-remote-code",
        "--device", "npu", "--attention-backend", "ascend",
        "--quantization", "awq", "--dtype", "bfloat16", "--tp-size", "8",
        "--host", "127.0.0.1", "--port", str(port),
        "--mem-fraction-static", "0.80", "--context-length", "4096",
        "--max-total-tokens", "8192", "--max-running-requests", "2",
        "--chunked-prefill-size", "-1", "--max-prefill-tokens", "4096",
        "--disable-radix-cache", "--disable-overlap-schedule",
        "--disable-shared-experts-fusion", "--cuda-graph-bs", "1", "2",
        "--decode-log-interval", "1", "--watchdog-timeout", "1800",
        "--random-seed", "42",
    ]
    if echo:
        command += ["--enable-echo", "--echo-device-cache-size", "4096"]
    environment = dict(os.environ)
    environment.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("HCCL_BUFFSIZE", "1536")
    environment.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    environment.setdefault("HCCL_SOCKET_IFNAME", "lo")
    environment.setdefault("GLOO_SOCKET_IFNAME", "lo")
    print(f"Starting {name} TP8 server; log: {log_path}", flush=True)
    responses = []
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=environment)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited ({process.returncode}); see {log_path}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=10) as response:
                        if response.status != 200:
                            raise OSError(f"Unexpected health status: {response.status}")
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(2)
            else:
                raise TimeoutError(f"{name} startup timed out; see {log_path}")
            for batch in batches:
                result = request(base, "/generate", {
                    "input_ids": batch,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
                    "return_logprob": True,
                })
                items = result if isinstance(result, list) else [result]
                if len(items) != len(batch):
                    raise AssertionError(f"{name}: unexpected response count")
                for item in items:
                    if item["meta_info"]["completion_tokens"] != 8:
                        raise AssertionError(f"{name}: incomplete generation")
                responses.append(items)
                (directory / f"{name}-responses.json").write_text(
                    json.dumps(responses, ensure_ascii=False, indent=2) + "\n"
                )
                print(f"{name}: batch={len(batch)}, prompt lengths={[len(ids) for ids in batch]}", flush=True)
            server_log = log_path.read_text(errors="replace")
            if "Capture npu graph end" not in server_log:
                raise AssertionError(f"{name}: no Graph capture evidence")
            for batch_size in (1, 2):
                if not re.search(rf"Decode batch[^\n]*#running-req: {batch_size},[^\n]*npu graph: True", server_log):
                    raise AssertionError(f"{name}: no Graph replay for batch {batch_size}")
            if re.search(r"Decode batch[^\n]*npu graph: False", server_log):
                raise AssertionError(f"{name}: eager decode fallback")
            if echo and "ECHO exact recall enabled" not in server_log:
                raise AssertionError("ECHO pool was not enabled")
            (directory / f"{name}-responses.json").write_text(json.dumps(responses, ensure_ascii=False, indent=2) + "\n")
            return responses
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=root / "build/models/DeepSeek-V3.2-AWQ")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--skip-long-prompt", action="store_true",
        help="Isolate short-request integration; does not establish long-context support.",
    )
    parser.add_argument(
        "--baseline-responses", type=Path,
        help="Reuse saved baseline outputs from the same model, inputs and serving configuration.",
    )
    args = parser.parse_args()
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    if visible != [str(i) for i in range(8, 16)]:
        raise ValueError("Run with ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15")
    batches = prompts(args.model, skip_long=args.skip_long_prompt)
    start = time.monotonic()
    if args.baseline_responses is None:
        baseline = run_server(root, args, False, batches)
    else:
        baseline = json.loads(args.baseline_responses.read_text())
        if len(baseline) != len(batches):
            raise ValueError("Saved baseline batch count does not match test inputs")
        for batch, results in zip(batches, baseline):
            if len(batch) != len(results):
                raise ValueError("Saved baseline request count does not match test inputs")
            for ids, result in zip(batch, results):
                metadata = result["meta_info"]
                if metadata["prompt_tokens"] != len(ids) or metadata["completion_tokens"] != 8:
                    raise ValueError("Saved baseline token counts do not match test inputs")
        print(f"Reusing baseline outputs: {args.baseline_responses}", flush=True)
    echo = run_server(root, args, True, batches)
    differences = []
    max_logprob_error = 0.0
    for batch_index, (expected, actual) in enumerate(zip(baseline, echo)):
        for request_index, (left, right) in enumerate(zip(expected, actual)):
            a = left["meta_info"]["output_token_logprobs"]
            b = right["meta_info"]["output_token_logprobs"]
            if [value[1] for value in a] != [value[1] for value in b]:
                differences.append((batch_index, request_index))
            else:
                max_logprob_error = max(max_logprob_error, *(abs(x[0] - y[0]) for x, y in zip(a, b)))
    report = {
        "status": "passed" if not differences and max_logprob_error <= 0.05 else "failed",
        "model": str(args.model), "devices": visible, "tp_size": 8,
        "graph_batch_sizes": [1, 2], "logical_token_capacity": 8192,
        "device_cache_tokens_per_layer": 4096,
        "prompt_lengths": [[len(ids) for ids in batch] for batch in batches],
        "token_mismatches": differences, "max_logprob_error": max_logprob_error,
        "logprob_absolute_tolerance": 0.05,
        "long_prompt_tested": not args.skip_long_prompt,
        "hccl_buffsize_mb": os.environ.get("HCCL_BUFFSIZE", "1536"),
        "baseline_reused": args.baseline_responses is not None,
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
    path = root / "build/echo-deepseek-smoke/result.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise AssertionError("ECHO model-level output comparison failed")


if __name__ == "__main__":
    main()
