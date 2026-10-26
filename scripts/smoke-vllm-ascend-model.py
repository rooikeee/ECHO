#!/usr/bin/env python3
"""Tiny local Qwen2 decode smoke, not DeepSeek/DSA or ECHO validation."""

import os
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    model = Path(__file__).resolve().parents[1] / "build/npu-smoke-model"
    llm = LLM(
        model=str(model),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        num_gpu_blocks_override=32,
        block_size=128,
        gpu_memory_utilization=0.1,
        enforce_eager=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    )
    params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True, detokenize=False)
    for batch in (1, 2, 1, 2):
        prompts = [{"prompt_token_ids": [2, 4 + i, 7, 8]} for i in range(batch)]
        results = llm.generate(prompts, params, use_tqdm=False)
        assert len(results) == batch
        assert all(len(result.outputs[0].token_ids) == 6 for result in results)
        print(f"PASS batch={batch}: {[r.outputs[0].token_ids for r in results]}", flush=True)


if __name__ == "__main__":
    main()
