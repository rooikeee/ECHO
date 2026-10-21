#!/usr/bin/env python3
"""Exercise the running full AWQ model and require actual decode graph replay."""

import json
import os
import re
import time
import urllib.request
from pathlib import Path

from jinja2 import Template
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "build/models/DeepSeek-V3.2-AWQ"
OUTPUT = ROOT / "build/deepseek-v32-awq"
URL = f"http://127.0.0.1:{os.environ.get('ECHO_V32_PORT', '30032')}"


def request(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        URL + path, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    start = time.monotonic()
    responses = []
    report = {"status": "running", "url": URL, "model": str(MODEL)}
    log_path = OUTPUT / "server.log"
    log_offset = log_path.stat().st_size
    try:
        chat = request("/v1/chat/completions", {
            "model": "deepseek-v3.2-awq",
            "messages": [{"role": "user", "content": "17 加 25 等于多少？请直接给出答案。"}],
            "temperature": 0, "max_tokens": 64,
        })
        responses.append({"test": "chat", "response": chat})
        answer = chat["choices"][0]["message"]["content"]
        assert "42" in answer, f"Unexpected arithmetic answer: {answer}"
        print("Chat answer:", answer, flush=True)

        template = Template((MODEL / "chat_template.jinja").read_text())
        tokenizer_config = json.loads((MODEL / "tokenizer_config.json").read_text())

        def prompt(content):
            return template.render(
                messages=[{"role": "user", "content": content}],
                bos_token=tokenizer_config["bos_token"],
                add_generation_prompt=True, thinking=False,
            )

        first = prompt("请用一句话介绍中国的首都。")
        second = prompt("请用一句话解释什么是矩阵乘法。")
        batches = []
        for texts in ([first], [first, second], [first]):
            result = request("/generate", {
                "text": texts,
                "sampling_params": {"temperature": 0, "max_new_tokens": 24, "ignore_eos": True},
                "return_logprob": True,
            })
            items = result if isinstance(result, list) else [result]
            assert len(items) == len(texts)
            for item in items:
                assert item["meta_info"]["completion_tokens"] == 24
            batches.append(items)
            responses.append({"test": f"batch_{len(texts)}", "response": result})
            print(f"Batch {len(texts)} generated 24 tokens per request", flush=True)
        tokens = lambda item: [entry[1] for entry in item["meta_info"]["output_token_logprobs"]]
        repeated_matches = tokens(batches[0][0]) == tokens(batches[2][0])
        report["repeated_request_matches"] = repeated_matches
        for batch in batches:
            assert "北京" in batch[0]["text"], "Incorrect capital in repeated request"
        if not repeated_matches:
            print("Repeated free-form answer differs; factual answer remains Beijing", flush=True)

        tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        context = "这是用于检查长文本处理的背景记录。\n" * 280
        long_tokens = tokenizer.encode(prompt(context + "请只回答：17 加 25 等于多少？")).ids
        assert 2048 < len(long_tokens) < 4000, f"Unexpected long prompt size: {len(long_tokens)}"
        long_result = request("/generate", {
            "input_ids": long_tokens,
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        })
        responses.append({"test": "long_context", "response": long_result})
        assert "42" in long_result["text"], f"Unexpected long-context answer: {long_result['text']}"

        log = log_path.read_text(errors="replace")
        new_log = log_path.read_bytes()[log_offset:].decode(errors="replace")
        assert "Capture npu graph end" in log, "Graph capture did not finish"
        for batch_size in (1, 2):
            assert re.search(
                rf"Decode batch[^\n]*#running-req: {batch_size},[^\n]*npu graph: True", new_log
            ), f"No graph replay evidence for batch {batch_size}"
        assert not re.search(r"Decode batch[^\n]*npu graph: False", new_log), "Decode fell back to eager"
        report.update(
            status="passed" if repeated_matches else "passed_with_token_variation",
            chat_answer=answer, request_batch_sizes=[1, 2, 1],
            repeated_request_matches=repeated_matches, graph_capture="passed", graph_replay="passed",
            long_prompt_tokens=len(long_tokens), long_context_answer=long_result["text"],
            elapsed_seconds=round(time.monotonic() - start, 2),
        )
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (OUTPUT / "responses.json").write_text(json.dumps(responses, ensure_ascii=False, indent=2) + "\n")
        (OUTPUT / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
