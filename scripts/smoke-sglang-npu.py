#!/usr/bin/env python3
"""Check serving with a tiny random model; this does not test model quality."""
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


def create_model(path):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    vocab.update({f"token{i}": i for i in range(4, 64)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>", unk_token="<unk>",
        bos_token="<bos>", eos_token="<eos>",
    )
    fast.save_pretrained(path)
    config = Qwen2Config(
        vocab_size=64, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, bos_token_id=2, eos_token_id=3,
        pad_token_id=0, tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model_path = root / "build" / "npu-smoke-model"
    model_path.mkdir(parents=True, exist_ok=True)
    create_model(model_path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = root / "build" / "npu-smoke-server.log"
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(model_path), "--device", "npu",
        "--attention-backend", "ascend", "--host", "127.0.0.1",
        "--port", str(port), "--dtype", "bfloat16",
        "--cuda-graph-max-bs", "2", "--decode-log-interval", "1",
        "--mem-fraction-static", "0.1",
        "--max-total-tokens", "1024", "--context-length", "512",
        "--max-running-requests", "2",
    ]
    print(f"Starting smoke server; log: {log_path}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + args.timeout
            base = f"http://127.0.0.1:{port}"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with {process.returncode}; see {log_path}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"Server did not become ready; see {log_path}")
            batches = [
                ["token4 token5"],
                ["token6 token7 token8", "token9 token10 token11 token12"],
                ["token4 token5"],
            ]
            outputs = []
            for batch in batches:
                payload = {
                    "text": batch[0] if len(batch) == 1 else batch,
                    "sampling_params": {
                        "temperature": 0, "max_new_tokens": 8, "ignore_eos": True
                    },
                    "return_logprob": True,
                }
                request = urllib.request.Request(
                    base + "/generate", data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    result = json.load(response)
                results = result if isinstance(result, list) else [result]
                assert len(results) == len(batch), results
                for item in results:
                    assert item["meta_info"]["completion_tokens"] == 8, item
                outputs.append(results)
            first_ids = [item[1] for item in outputs[0][0]["meta_info"]["output_token_logprobs"]]
            repeated_ids = [item[1] for item in outputs[2][0]["meta_info"]["output_token_logprobs"]]
            assert len(first_ids) == 8 and first_ids == repeated_ids, (first_ids, repeated_ids)
            server_log = log_path.read_text()
            assert "Capture npu graph end" in server_log, "NPU graph capture did not finish"
            for batch_size in (1, 2):
                pattern = rf"Decode batch[^\n]*#running-req: {batch_size},[^\n]*npu graph: True"
                assert re.search(pattern, server_log), f"No graph replay for batch size {batch_size}"
            assert not re.search(r"Decode batch[^\n]*npu graph: False", server_log), "Decode fell back to eager execution"
            report = {
                "status": "passed", "model": "tiny random Qwen2",
                "device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                "completion_tokens": 8,
                "requests": sum(len(batch) for batch in batches),
                "graph_capture": "passed", "graph_replay": "passed",
                "request_batch_sizes": [len(batch) for batch in batches],
                "repeated_request_matches": True,
                "server_log": str(log_path),
            }
            result_path = root / "build" / "npu-smoke-result.json"
            result_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
