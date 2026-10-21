#!/usr/bin/env python3
"""Paired NSA serving checks for TP/DP attention, not model quality or speed."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def validate_topology(tp_size, dp_size, devices):
    if tp_size < 1 or dp_size < 1 or tp_size % dp_size:
        raise ValueError("DP attention requires positive dp_size dividing tp_size")
    if len(devices) != tp_size or len(set(devices)) != tp_size:
        raise ValueError("Visible devices must contain exactly tp_size distinct NPU IDs")
    if not all(device.isdecimal() for device in devices):
        raise ValueError("Visible NPU IDs must be numeric")


def ensure_devices_idle(info, devices):
    if "Process id" not in info:
        raise RuntimeError("Cannot verify NPU process table; no processes were stopped")
    selected = {int(device) for device in devices}
    for board, chip, pid in re.findall(
        r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|", info, re.MULTILINE
    ):
        physical_id = int(board) * 2 + int(chip)
        if physical_id in selected:
            raise RuntimeError(f"NPU {physical_id} occupied by PID {pid}; no processes were stopped")


def create_tiny_model(path, tp_size):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import DeepseekV3Config, PreTrainedTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    vocab.update({f"token{i}": i for i in range(4, 64)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>",
        bos_token="<bos>", eos_token="<eos>",
    )
    fast.save_pretrained(path)
    config = DeepseekV3Config(
        vocab_size=64, hidden_size=256, intermediate_size=512,
        moe_intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=16 * tp_size, num_key_value_heads=16 * tp_size,
        kv_lora_rank=512, q_lora_rank=128, qk_rope_head_dim=64,
        qk_nope_head_dim=128, v_head_dim=128,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        n_group=1, topk_group=1, first_k_dense_replace=1, moe_layer_freq=1,
        topk_method="noaux_tc", scoring_func="sigmoid",
        max_position_embeddings=4096, bos_token_id=2, eos_token_id=3,
        pad_token_id=0, torch_dtype="bfloat16",
        index_head_dim=128, index_n_heads=64, index_topk=2048,
    )
    config.architectures = ["DeepseekV3ForCausalLM"]
    config.save_pretrained(path)


def request_inputs(args):
    if args.tiny:
        return {
            "single": [[4 + rank % 30, 5, 6] for rank in range(args.dp_size)],
            "second": [7, 8, 9, 10, 11],
            "concurrent": [[4, 5] * (rank + 2) for rank in range(args.dp_size)],
            "long": [4, 5] * 1100,
        }
    from jinja2 import Template
    from tokenizers import Tokenizer

    template = Template((args.model / "chat_template.jinja").read_text())
    config = json.loads((args.model / "tokenizer_config.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))

    def encode(content):
        text = template.render(
            messages=[{"role": "user", "content": content}],
            bos_token=config["bos_token"], add_generation_prompt=True, thinking=False,
        )
        return tokenizer.encode(text).ids

    inputs = {
        "single": [encode(f"{17 + rank} 加 25 等于多少？请直接给出答案。")
                   for rank in range(args.dp_size)],
        "second": encode("请用一句话介绍中国的首都。"),
        "concurrent": [encode("这是背景记录。\n" * (rank + 2)
                              + "请用一句话解释矩阵乘法。")
                       for rank in range(args.dp_size)],
        "long": encode("这是用于检查长文本处理的背景记录。\n" * 280
                       + "请只回答：17 加 25 等于多少？"),
    }
    if not 2048 < len(inputs["long"]) < 4000:
        raise ValueError("Full-model long prompt must exceed NSA Top-K and fit context")
    return inputs


def server_command(args, port, echo):
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(args.model), "--device", "npu",
        "--attention-backend", "ascend", "--dtype", "bfloat16",
        "--tp-size", str(args.tp_size), "--dp-size", str(args.dp_size),
        "--host", "127.0.0.1", "--port", str(port),
        "--max-total-tokens", "8192", "--context-length", "4096",
        "--max-running-requests", str(2 * args.dp_size),
        "--chunked-prefill-size", str(512 * args.dp_size),
        "--max-prefill-tokens", "4096", "--mem-fraction-static", "0.1" if args.tiny else "0.96",
        "--disable-radix-cache", "--disable-overlap-schedule",
        "--disable-shared-experts-fusion", "--cuda-graph-bs", "1", "2",
        "--decode-log-interval", "1", "--watchdog-timeout", "180" if args.tiny else "1800",
        "--random-seed", "42",
    ]
    if args.dp_size > 1:
        command += ["--enable-dp-attention"]
    if args.tiny:
        command += ["--load-format", "dummy"]
    else:
        command += ["--trust-remote-code"]
        if args.quantization:
            command += ["--quantization", args.quantization]
    if echo:
        command += ["--enable-echo", "--echo-device-cache-size", "4096"]
    return command


def post_json(base, payload):
    request = urllib.request.Request(
        base + "/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def choose_server_port():
    # DP ZMQ uses HTTP port + 233..238. Keep the entire block below the
    # host ephemeral client-port range, so health probes cannot occupy it.
    for port in random.SystemRandom().sample(range(16000, 24000, 16), 500):
        sockets = []
        try:
            for offset in (0, 233, 234, 235, 236, 237, 238):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port + offset))
            return port
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("No free HTTP/DP port block; no server was started")


def validate_graph(log, tp_size, dp_size, echo):
    if log.count("Capture npu graph end") < tp_size:
        raise AssertionError("Missing Graph capture on one or more TP ranks")
    for batch in (1, 2):
        if not re.search(rf"Decode batch[^\n]*#running-req: {batch},[^\n]*npu graph: True", log):
            raise AssertionError(f"No actual Graph replay for local batch {batch}")
        if dp_size > 1:
            for rank in range(dp_size):
                if not re.search(
                    rf"\[[^\n\]]*\bDP{rank}\b[^\n\]]*\] Decode batch[^\n]*"
                    rf"#running-req: {batch},[^\n]*npu graph: True", log,
                ):
                    raise AssertionError(f"No Graph replay for DP rank {rank}, local batch {batch}")
    eager_steps = re.findall(r"[^\n]*Decode batch[^\n]*npu graph: False[^\n]*", log)
    if any(dp_size == 1 or "dp global prefill: True" not in line for line in eager_steps):
        raise AssertionError("Pure decode fell back to eager (or missing global prefill evidence)")
    if echo:
        attn_tp = tp_size // dp_size
        layouts = {
            tuple(map(int, match)) for match in re.findall(
                r"ECHO parallel layout: attention_tp_rank=(\d+)/(\d+), attention_dp_rank=(\d+)/(\d+)",
                log,
            )
        }
        expected = {(tp, attn_tp, dp, dp_size) for dp in range(dp_size) for tp in range(attn_tp)}
        if layouts != expected:
            raise AssertionError(f"Unexpected ECHO rank layouts: {layouts}; expected {expected}")
    return {"mixed_prefill_decode_eager_log_entries": len(eager_steps),
            "pure_decode_eager_log_entries": 0}


def run_server(args, echo):
    name = "echo" if echo else "baseline"
    port = choose_server_port()
    command = server_command(args, port, echo)
    (args.output_dir / f"{name}-command.json").write_text(json.dumps(command, indent=2) + "\n")
    log_path = args.output_dir / f"{name}-server.log"
    environment = dict(os.environ)
    for key, value in {
        "HCCL_OP_EXPANSION_MODE": "AIV", "HCCL_BUFFSIZE": "256" if args.tiny else "1536",
        "HCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }.items():
        environment.setdefault(key, value)
    print(f"Starting {name} TP{args.tp_size}/DP{args.dp_size}; {log_path}", flush=True)
    records = []
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited ({process.returncode}); see {log_path}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            else:
                raise TimeoutError(f"{name} startup timed out; see {log_path}")

            def generate(rank, batch, label):
                payload = {
                    "input_ids": batch,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
                    "return_logprob": True,
                }
                if args.dp_size > 1:
                    payload["routed_dp_rank"] = rank
                result = post_json(base, payload)
                items = result if isinstance(result, list) else [result]
                if len(items) != len(batch):
                    raise AssertionError("Incorrect response batch size")
                for item in items:
                    meta = item["meta_info"]
                    if meta["completion_tokens"] != 16:
                        raise AssertionError("Incomplete generation")
                    if args.dp_size > 1 and meta.get("dp_rank") != rank:
                        raise AssertionError(f"Request was not routed to DP rank {rank}: {meta}")
                print(f"{name}: {label}, dp_rank={rank}, local_batch={len(batch)}", flush=True)
                if not args.tiny:
                    print(f"  answers: {[item.get('text', '') for item in items]}", flush=True)
                return {"label": label, "dp_rank": rank, "inputs": batch, "responses": items}

            # Sequential routing forces all other DP ranks to participate idle.
            for rank in range(args.dp_size):
                first = args.inputs["single"][rank]
                for label, batch in (
                    ("single", [first]), ("pair", [first, args.inputs["second"]]),
                    ("reuse", [first]),
                ):
                    records.append(generate(rank, batch, label))
            # Different local batch sizes and lengths exercise DP padding.
            with ThreadPoolExecutor(max_workers=args.dp_size) as executor:
                futures = [executor.submit(
                    generate, rank,
                    [args.inputs["concurrent"][rank]] * (1 + rank % 2), "concurrent",
                ) for rank in range(args.dp_size)]
                records.extend(future.result() for future in futures)
            if args.long_prompt:
                records.append(generate(0, [args.inputs["long"]], "long"))
            (args.output_dir / f"{name}-responses.json").write_text(json.dumps(records, indent=2) + "\n")
            evidence = validate_graph(log_path.read_text(errors="replace"), args.tp_size, args.dp_size, echo)
            (args.output_dir / f"{name}-graph-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
            return records
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


def compare_results(baseline, echo):
    if len(baseline) != len(echo):
        raise AssertionError("Paired test case count differs")
    mismatches, max_error = [], 0.0
    for case, (left, right) in enumerate(zip(baseline, echo)):
        for key in ("label", "dp_rank", "inputs"):
            if left[key] != right[key]:
                raise AssertionError(f"Paired {key} differs")
        if len(left["responses"]) != len(right["responses"]):
            raise AssertionError("Paired response count differs")
        for item, (a, b) in enumerate(zip(left["responses"], right["responses"])):
            a = a["meta_info"]["output_token_logprobs"]
            b = b["meta_info"]["output_token_logprobs"]
            if len(a) != 16 or len(b) != 16:
                raise AssertionError("Missing token logprobs")
            if [value[1] for value in a] != [value[1] for value in b]:
                mismatches.append([case, item])
            else:
                errors = [abs(x[0] - y[0]) for x, y in zip(a, b)]
                if not all(math.isfinite(error) for error in errors):
                    raise AssertionError("Nonfinite token logprob comparison")
                max_error = max(max_error, *errors)
    return {"token_mismatches": mismatches, "max_logprob_error": max_error,
            "logprob_absolute_tolerance": 0.05,
            "passed": not mismatches and max_error <= 0.05}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--quantization")
    parser.add_argument("--long-prompt", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    validate_topology(args.tp_size, args.dp_size, devices)
    args.output_dir = args.output_dir or root / "build/echo-parallel" / datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "tp_size": args.tp_size, "dp_size": args.dp_size,
              "attention_tp_size": args.tp_size // args.dp_size, "devices": devices,
              "pd_disaggregated": False, "long_prompt_requested": args.long_prompt,
              "long_prompt_tested": False,
              "model_quality_measured": False, "performance_measured": False}
    start = time.monotonic()
    try:
        info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                              check=True, timeout=30).stdout
        (args.output_dir / "npu-smi-before.txt").write_text(info)
        ensure_devices_idle(info, devices)
        report["machine_has_hardware_alarm"] = "Alarm" in info
        report["selected_device_health"] = []
        for device in devices:
            physical = int(device)
            health = subprocess.run(
                ["npu-smi", "info", "-t", "health", "-i", str(physical // 2),
                 "-c", str(physical % 2)], capture_output=True, text=True,
                check=True, timeout=30,
            ).stdout
            match = re.search(r"Health Status\s*:\s*(\S+)", health)
            report["selected_device_health"].append(
                {"physical_id": physical, "health": match.group(1) if match else "unknown"}
            )
        report["selected_devices_healthy"] = all(
            item["health"] == "OK" for item in report["selected_device_health"]
        )
        args.tiny = args.model is None
        if args.tiny:
            args.model = args.output_dir / "tiny-nsa-model"
            create_tiny_model(args.model, args.tp_size)
        report["model"] = str(args.model)
        report["random_dummy_weights"] = args.tiny
        args.inputs = request_inputs(args)
        (args.output_dir / "inputs.json").write_text(json.dumps(args.inputs, indent=2) + "\n")
        report["prompt_lengths"] = {
            key: [len(ids) for ids in value] if key in ("single", "concurrent") else len(value)
            for key, value in args.inputs.items()
        }
        baseline = run_server(args, False)
        echo = run_server(args, True)
        report["long_prompt_tested"] = args.long_prompt
        report["comparison"] = compare_results(baseline, echo)
        report["graph_evidence"] = {
            name: json.loads((args.output_dir / f"{name}-graph-evidence.json").read_text())
            for name in ("baseline", "echo")
        }
        report["graph_capture_replay"] = "passed"
        report["status"] = "passed" if report["comparison"]["passed"] else "numerical_comparison_failed"
        if report["status"] != "passed":
            raise AssertionError("ECHO paired model comparison failed")
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - start, 2)
        (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
